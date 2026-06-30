r"""``EstimatorLaw`` --- the law of a statistic as a first-class interface (#144).

Every estimator in this framework constructs a random variable: :math:`\hat\theta
= T(X_1, \dots, X_N)` is a measurable map, and its *law* is the pushforward of the
sampling distribution of the data through :math:`T`. The package already holds
several partial representations of that law -- none of them first-class (the
adoption gate of ``docs/design.org`` §"Prospective sketch: =EstimatorLaw=" lists
five carriers). This module is the *narrow, point-implemented* v2.1 pass that
gate authorised: ONE query algebra (:meth:`~EstimatorLaw.cov` /
:meth:`~EstimatorLaw.quantile` / :meth:`~EstimatorLaw.prob` /
:meth:`~EstimatorLaw.sample` / :meth:`~EstimatorLaw.given`) over the statistic's
codomain :math:`S`, with the *implementation* fixing the epistemic grade of the
answer.

Two grade instances compose through the one interface (design.org: "=given=
returns another =EstimatorLaw= ... most conditionals do not exist in closed form
at the asymptotic grade -- the implementation REFUSES rather than approximates"):

* :class:`EmpiricalLaw` --- the empirical measure on :math:`S` carried by a
  stack of draws (the ``emu_gmm.studies`` :class:`~emu_gmm.types.FitRecord`
  records, a bootstrap's ``theta_boot`` / ``J_boot``, ...): draws
  :math:`\times` event-flags :math:`\{0,1\}^E` :math:`\times` validity counts
  :math:`\times` provenance. Its :meth:`~EmpiricalLaw.cov` /
  :meth:`~EmpiricalLaw.se` / :meth:`~EmpiricalLaw.quantile` /
  :meth:`~EmpiricalLaw.pvalue` queries DELEGATE to the
  :mod:`emu_gmm.inference.adaptive` ``Bootstrap*`` functionals and the
  :mod:`emu_gmm.studies.summaries` summarizers (no re-implementation;
  single-correct-implementation discipline). :meth:`~EmpiricalLaw.given`
  EXTENDS :func:`emu_gmm.studies.given`; :meth:`~EmpiricalLaw.couple` reuses
  :func:`emu_gmm.studies.crn_pair` verbatim (the joint constructor verifies
  key/provenance before zipping and refuses on mismatch).
* :class:`AsymptoticLaw` --- the Gaussian law :math:`\mathcal N(\hat\theta,
  \Sigma_\theta)` to first order. Its queries are the delta method
  (:func:`emu_gmm.inference.functional_se.functional_se`, already mirrored on
  :class:`~emu_gmm.types.EstimationResult`). :meth:`~AsymptoticLaw.given`
  RAISES: the conditional of a Gaussian given a data-dependent event has no
  closed form, and the framework refuses rather than approximates.

The codomain is generic: a gauge-invariant *functional* ``f`` of the parameter
components projects the law onto the quantity of interest. For a
``PSDFixedRank`` :math:`\Gamma = A A^\top` factor the shipped functionals
:func:`~emu_gmm.inference.functional_se.gamma_eigenvalues` /
:func:`~emu_gmm.inference.functional_se.gamma_vech` make eigenvalues / ``vech``
queryable, manifold/gauge-aware: at the asymptotic grade through the
gauge-invariant delta method (the gauge nullspace is already pinned out of
:math:`\Sigma_\theta`); at the empirical grade through the gauge-invariant
functional applied PER DRAW (``theta_flat[r]`` :math:`\to A \to \Gamma \to`
``eigvalsh``), never reducing the gauge-arbitrary raw ``theta_flat`` columns.
"""

from __future__ import annotations

import abc
from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import scipy.stats

from emu_gmm._internal.law_state import (
    MomentsBacking as _MomentsBacking,
)
from emu_gmm._internal.law_state import (
    euclidean_tag as _euclidean_tag,
)
from emu_gmm._internal.law_state import (
    locate_psd_leaf as _locate_psd_leaf,
)
from emu_gmm._internal.law_state import (
    manifold_to_tag as _manifold_to_tag,
)
from emu_gmm.inference.adaptive import (
    BootstrapMean,
    BootstrapPValue,
    BootstrapQuantile,
    BootstrapSE,
)
from emu_gmm.inference.functional_se import (
    _component_shapes,
    _flatten_components,
    _unflatten_to_components,
    gamma_eigenvalues,
    gamma_vech,
)
from emu_gmm.studies import summaries as _summaries
from emu_gmm.studies.conditioning import (
    CoupledRecords,
    EventShare,
    crn_pair,
    event_share,
    given,
)
from emu_gmm.studies.driver import MCRecords
from emu_gmm.types import EstimationResult, FitRecord

__all__ = [
    "EstimatorLaw",
    "EmpiricalLaw",
    "AsymptoticLaw",
    "couple",
    "eigenvalue_functional",
    "gamma_functional",
]

#: A codomain functional: maps the components tuple ``(A, phi, ...)`` (exactly
#: what :meth:`EstimationResult.components` returns) to a scalar or 1-D array.
#: ``None`` means the identity on the ambient flat parameter vector.
Functional = Callable[[tuple[Any, ...]], Any] | None


def eigenvalue_functional(
    rank: int, index: int = 0
) -> Callable[[tuple[Any, ...]], Any]:
    """The ``rank`` nonzero eigenvalues of ``Gamma = A @ A.T`` as a functional.

    A thin closure over
    :func:`emu_gmm.inference.functional_se.gamma_eigenvalues` so an eigenvalue
    query is a first-class codomain functional usable at *either* grade
    (delta-method SE at the asymptotic grade; per-draw application at the
    empirical grade). ``index`` selects the ``PSDFixedRank`` component (the
    K-Aggregators contract is ``index=0``).
    """
    return lambda comps: gamma_eigenvalues(comps, rank, index)


def gamma_functional(index: int = 0) -> Callable[[tuple[Any, ...]], Any]:
    """``vech(Gamma)``, ``Gamma = A @ A.T``, as a codomain functional.

    Thin closure over :func:`emu_gmm.inference.functional_se.gamma_vech`.
    """
    return lambda comps: gamma_vech(comps, index)


class EstimatorLaw(abc.ABC):
    r"""The frequentist law of an estimator under a DGP --- one query algebra.

    This is the *interface*, not a carrier of any one grade's state: it is the
    frequentist law of the ESTIMATOR (not a belief over :math:`\theta`). Two
    concrete grades --- :class:`EmpiricalLaw` and :class:`AsymptoticLaw` ---
    implement it; they compose through :meth:`given` and the coupling
    constructor. Every query takes an optional gauge-invariant *functional*
    ``f`` of the parameter components ``(A, phi, ...)`` that projects the law
    onto a derived codomain (``f=None`` == identity on the ambient flat
    parameter vector).
    """

    #: A short human-readable epistemic grade ("empirical" / "asymptotic").
    grade: str = "abstract"

    @property
    @abc.abstractmethod
    def param_names(self) -> tuple[str, ...]:
        """Labels of the ambient flat parameter axis (the ``f=None`` codomain)."""

    @abc.abstractmethod
    def mean(self, f: Functional = None) -> np.ndarray:
        """The mean of ``f`` under the law (the point estimate of ``f(theta)``)."""

    @abc.abstractmethod
    def cov(self, f: Functional = None) -> np.ndarray:
        """The ``(p, p)`` covariance of ``f`` under the law."""

    def se(self, f: Functional = None) -> np.ndarray:
        """Standard errors of ``f`` --- ``sqrt(diag(cov(f)))`` by default.

        Negative diagonal entries (finite-precision round-off, or a non-PD
        finite-sample covariance) propagate as ``nan`` rather than complex
        values, matching :attr:`EstimationResult.standard_errors`.
        """
        c = np.atleast_2d(np.asarray(self.cov(f)))
        d = np.diag(c)
        with np.errstate(invalid="ignore"):
            return np.sqrt(np.where(d >= 0.0, d, np.nan))

    @abc.abstractmethod
    def quantile(self, q: float, f: Functional = None) -> np.ndarray:
        """The per-coordinate ``q``-quantile of ``f`` under the law."""

    @abc.abstractmethod
    def prob(
        self, predicate: Callable[[np.ndarray], Any], f: Functional = None
    ) -> float:
        """``P(predicate(f(theta)))`` under the law."""

    @abc.abstractmethod
    def sample(self, key: jax.Array, n: int, f: Functional = None) -> np.ndarray:
        """``n`` draws of ``f(theta)`` from the law (shape ``(n, p)``)."""

    @abc.abstractmethod
    def given(
        self,
        event: Any,
        *,
        negate: bool = False,
        acknowledge_conditional: bool = False,
    ) -> EstimatorLaw:
        """The conditional law given ``event`` --- another :class:`EstimatorLaw`.

        Grade-aware: where the conditional has no closed form (the asymptotic
        grade) the implementation REFUSES rather than approximates.
        """

    def save(self, path: Any, *, factory_spec: Any = None) -> None:
        """Persist this law to ``path`` as a typed, versioned :class:`LawState` (#181).

        Writes an inert, cross-version artifact (a single ``.npz`` with a JSON
        manifest) via :func:`emu_gmm.persistence.save_law` --- NOT a pickle: no
        class references or model code are entombed, so it reloads under a
        future emu through :func:`emu_gmm.persistence.load_law`. Both the
        asymptotic and empirical grades are supported. Pass ``factory_spec`` (a
        :class:`emu_gmm.persistence.FactorySpec`) to record the estimator
        configuration that is part of T's identity (#142).
        """
        from emu_gmm.persistence import save_law

        save_law(self, path, factory_spec=factory_spec)


# ---------------------------------------------------------------------------
# Empirical grade: the law carried by a stack of draws.
# ---------------------------------------------------------------------------
class EmpiricalLaw(EstimatorLaw):
    r"""The empirical law of the estimator on a stack of draws.

    The carrier of charter item (a): stacked draws :math:`\times` event-flags
    :math:`\{0,1\}^E` :math:`\times` validity counts :math:`\times` provenance.
    Backed by either an :class:`~emu_gmm.studies.MCRecords` /
    :class:`~emu_gmm.types.FitRecord` (a repeated-sampling or bootstrap stack
    carrying the diagnostic event flags, so :meth:`given` and the summarizers
    are native) or a bare array of statistic draws (e.g. a wild bootstrap's
    ``J_boot`` --- the empirical law of :math:`Q`; carrier #4). Construct via
    :meth:`from_records` or :meth:`from_draws`.

    The query methods DELEGATE: :meth:`se` / :meth:`mean` / :meth:`quantile` /
    :meth:`pvalue` route through the :mod:`emu_gmm.inference.adaptive`
    ``Bootstrap*`` functionals; :meth:`bias_sd` / :meth:`coverage` /
    :meth:`size_power` / :meth:`tau_binding` / :meth:`j_calibration` route
    through the :mod:`emu_gmm.studies.summaries` summarizers; :meth:`given`
    extends :func:`emu_gmm.studies.given`; :meth:`couple` reuses
    :func:`emu_gmm.studies.crn_pair`. No statistic is re-implemented here.
    """

    grade = "empirical"

    def __init__(
        self,
        *,
        draws: np.ndarray,
        used: np.ndarray,
        names: tuple[str, ...],
        component_shapes: tuple[tuple[int, ...], ...] | None = None,
        records: MCRecords | FitRecord | None = None,
        coupling_id: Any = None,
        key: Any = None,
        events: dict[str, np.ndarray] | None = None,
        label: str = "empirical",
        conditioned: bool = False,
    ) -> None:
        # The constructors always pass a 2-D (n_draws, p) array; atleast_2d is a
        # belt-and-braces no-op for them (it never reshapes a genuine (n, p)).
        self._draws = np.atleast_2d(np.asarray(draws, dtype=float))
        self._used = np.asarray(used, dtype=bool).ravel()
        self._names = tuple(names)
        self._component_shapes = (
            None
            if component_shapes is None
            else tuple(tuple(s) for s in component_shapes)
        )
        self._records = records
        self._coupling_id = coupling_id
        self._key = key
        # Optional {flag_name: (n,) mask} carried by a raw-draws law so it can be
        # conditioned (given()) without a FitRecord (#181). Normalised to bool.
        self._events: dict[str, np.ndarray] | None = (
            None
            if events is None
            else {k: np.asarray(v).ravel() > 0.5 for k, v in events.items()}
        )
        self._label = label
        self._conditioned = bool(conditioned)

    # -- constructors -------------------------------------------------------
    @classmethod
    def from_records(
        cls,
        records: MCRecords | FitRecord,
        *,
        component_shapes: tuple[tuple[int, ...], ...] | None = None,
        label: str | None = None,
    ) -> EmpiricalLaw:
        """Build the law from a repeated-sampling / bootstrap record stack.

        ``records`` is an :class:`~emu_gmm.studies.MCRecords` (the
        :func:`emu_gmm.studies.replicate` output) or a bare stacked
        :class:`~emu_gmm.types.FitRecord`. The ``f=None`` codomain is the
        ambient flat parameter axis; pass ``component_shapes`` (e.g.
        ``((5, 2), (1,))`` for a ``Product(PSDFixedRank(5, 2), Euclidean(1))``
        estimate, or :func:`~emu_gmm.inference.functional_se._component_shapes`
        of a template's ``.components()``) to enable gauge-invariant
        component functionals at the empirical grade.
        """
        rec = records.records if isinstance(records, MCRecords) else records
        if not isinstance(rec, FitRecord):
            raise TypeError(
                "EmpiricalLaw.from_records expects an MCRecords or a stacked "
                f"FitRecord; got {type(records).__name__}."
            )
        coupling_id = records.coupling_id if isinstance(records, MCRecords) else None
        key = records.key if isinstance(records, MCRecords) else None
        return cls(
            draws=np.asarray(rec.theta_flat),
            used=np.asarray(rec.converged) > 0.5,
            names=tuple(rec.param_names),
            component_shapes=component_shapes,
            records=records,
            coupling_id=coupling_id,
            key=key,
            label=label or "empirical(records)",
            conditioned=False,
        )

    @classmethod
    def from_records_with_template(
        cls,
        records: MCRecords | FitRecord,
        template: EstimationResult,
        *,
        label: str | None = None,
    ) -> EmpiricalLaw:
        """:meth:`from_records` with ``component_shapes`` read off ``template``.

        Convenience for the manifold case: the per-leaf ambient shapes are
        taken from ``template.components()`` (a representative estimate sharing
        the records' parameter structure), so eigenvalue / ``gamma`` queries
        work without the caller spelling the shapes out.
        """
        shapes = tuple(_component_shapes(template.components()))
        return cls.from_records(records, component_shapes=shapes, label=label)

    @classmethod
    def from_draws(
        cls,
        values: Any,
        *,
        names: tuple[str, ...] | None = None,
        component_shapes: tuple[tuple[int, ...], ...] | None = None,
        events: dict[str, Any] | None = None,
        label: str | None = None,
    ) -> EmpiricalLaw:
        """Build the law from a bare array of statistic draws.

        ``values`` is ``(n,)`` (a scalar statistic, e.g. a wild bootstrap's
        ``J_boot`` --- the empirical law of :math:`Q`) or ``(n, p)``. Non-finite
        rows (e.g. a diverged refit) are excluded from the functional but kept
        in the denominator (the ``used`` mask), so a degenerate resampling world
        cannot masquerade as precision.

        Pass ``events={name: mask}`` (each ``mask`` an ``(n,)`` 0/1 or boolean
        array aligned with the draws) to let an array / CSV-backed bootstrap
        carry conditionable flags (#181): :meth:`given` on a named event then
        returns the conditioned sub-population. Without ``events`` (and without a
        backing :class:`FitRecord`) :meth:`given` and :meth:`couple` remain
        unavailable, as before.
        """
        arr = np.asarray(values, dtype=float)
        if arr.ndim == 1:
            arr = arr[:, None]
        if arr.ndim != 2:
            raise ValueError(
                f"EmpiricalLaw.from_draws: values must be 1-D or 2-D, got ndim "
                f"{arr.ndim}."
            )
        used = np.all(np.isfinite(arr), axis=1)
        if names is None:
            names = tuple(f"s{j}" for j in range(arr.shape[1]))
        if events is not None:
            n = arr.shape[0]
            for nm, mask in events.items():
                if np.asarray(mask).ravel().shape[0] != n:
                    raise ValueError(
                        f"EmpiricalLaw.from_draws: event {nm!r} mask has length "
                        f"{np.asarray(mask).ravel().shape[0]} != n_draws {n}."
                    )
        return cls(
            draws=arr,
            used=used,
            names=tuple(names),
            component_shapes=component_shapes,
            records=None,
            coupling_id=None,
            key=None,
            events=events,
            label=label or "empirical(draws)",
            conditioned=False,
        )

    # -- properties ---------------------------------------------------------
    @property
    def param_names(self) -> tuple[str, ...]:
        return self._names

    @property
    def n_draws(self) -> int:
        """Total number of draws (used + excluded)."""
        return int(self._draws.shape[0])

    @property
    def n_used(self) -> int:
        """Draws contributing to a functional (converged / finite)."""
        return int(self._used.sum())

    @property
    def conditioned(self) -> bool:
        """Whether this law is a :meth:`given` sub-population (not CRN-pairable)."""
        return self._conditioned

    @property
    def event_names(self) -> tuple[str, ...]:
        """Names of the event flags this law can be :meth:`given`-conditioned on.

        The :class:`FitRecord` flag fields for a records-backed law, or the keys
        of the ``events`` dict for a raw-draws law; empty if neither.
        """
        if self._events is not None:
            return tuple(self._events)
        if self._records is not None:
            from emu_gmm.studies.conditioning import FLAG_FIELDS

            return FLAG_FIELDS
        return ()

    # -- codomain projection (the per-draw gauge-invariant glue) ------------
    def _codomain(self, f: Functional) -> tuple[np.ndarray, tuple[str, ...]]:
        """The used draws projected through ``f`` --- ``(n_used, p)`` + labels.

        ``f=None`` returns the raw used draws. A component functional is applied
        PER DRAW via :func:`~emu_gmm.inference.functional_se._unflatten_to_components`
        (landmine 3: a ``PSDFixedRank`` leaf's raw ``theta_flat`` entries are
        gauge-arbitrary, so a gauge-invariant functional MUST be applied per
        draw -- ``theta_flat[r]`` -> ``A`` -> ``Gamma`` -> eigvalsh -- NEVER
        reduce the raw columns).
        """
        raw = self._draws[self._used]
        if f is None:
            return raw, self._names
        if self._component_shapes is None:
            raise ValueError(
                "EmpiricalLaw: a component functional needs component_shapes. "
                "Build the law with EmpiricalLaw.from_records(..., "
                "component_shapes=...) or .from_records_with_template(...) so "
                "each draw's ambient flat vector can be unflattened to the "
                "components (A, phi, ...) the functional consumes."
            )
        shapes = self._component_shapes

        def apply(tf: jax.Array) -> jax.Array:
            comps = _unflatten_to_components(tf, shapes)
            return jnp.atleast_1d(jnp.asarray(f(tuple(comps))))

        if raw.shape[0] == 0:
            # Probe the output width on a zero vector so the empty result is
            # still correctly shaped (n_used=0, p).
            width = int(apply(jnp.zeros(raw.shape[1])).shape[0])
            return np.zeros((0, width)), tuple(f"f{j}" for j in range(width))
        vals = np.asarray(jax.vmap(apply)(jnp.asarray(raw)))
        return vals, tuple(f"f{j}" for j in range(vals.shape[1]))

    @staticmethod
    def _finite_col(vals: np.ndarray, j: int) -> np.ndarray:
        col = vals[:, j]
        return col[np.isfinite(col)]

    # -- query algebra (delegates to the Bootstrap* functionals) ------------
    def mean(self, f: Functional = None) -> np.ndarray:
        """Per-coordinate mean, via :class:`~emu_gmm.inference.BootstrapMean`."""
        vals, _ = self._codomain(f)
        out = []
        for j in range(vals.shape[1]):
            col = self._finite_col(vals, j)
            out.append(BootstrapMean().evaluate(col)[0] if col.size else float("nan"))
        return np.asarray(out)

    def cov(self, f: Functional = None) -> np.ndarray:
        """Sample covariance (``ddof=1``) over the finite draws (``(p, p)``)."""
        vals, _ = self._codomain(f)
        finite_rows = vals[np.all(np.isfinite(vals), axis=1)]
        if finite_rows.shape[0] < 2:
            p = vals.shape[1]
            return np.full((p, p), np.nan)
        return np.atleast_2d(np.cov(finite_rows, rowvar=False, ddof=1))

    def se(self, f: Functional = None) -> np.ndarray:
        """Per-coordinate SE, via :class:`~emu_gmm.inference.BootstrapSE`."""
        vals, _ = self._codomain(f)
        out = []
        for j in range(vals.shape[1]):
            col = self._finite_col(vals, j)
            out.append(
                BootstrapSE().evaluate(col)[0] if col.size >= 2 else float("nan")
            )
        return np.asarray(out)

    def quantile(self, q: float, f: Functional = None) -> np.ndarray:
        """Per-coordinate ``q``-quantile, via
        :class:`~emu_gmm.inference.BootstrapQuantile`."""
        vals, _ = self._codomain(f)
        out = []
        for j in range(vals.shape[1]):
            col = self._finite_col(vals, j)
            out.append(
                BootstrapQuantile(q).evaluate(col)[0] if col.size >= 2 else float("nan")
            )
        return np.asarray(out)

    def pvalue(
        self,
        stat_observed: float,
        *,
        tail: str = "greater",
        f: Functional = None,
        coord: int = 0,
    ) -> float:
        """Bootstrap p-value of ``stat_observed`` in coordinate ``coord``.

        Routes through :class:`~emu_gmm.inference.BootstrapPValue` (the
        ``(1+count)/(B+1)`` correction). For a wild-bootstrap law of
        :math:`Q`, ``law.pvalue(J_observed)`` is the cluster-wild
        over-identification p-value ``P(J_boot >= J_observed)``.
        """
        vals, _ = self._codomain(f)
        col = self._finite_col(vals, coord)
        return float(BootstrapPValue(stat_observed, tail).evaluate(col)[0])

    def prob(
        self, predicate: Callable[[np.ndarray], Any], f: Functional = None
    ) -> float:
        """Empirical ``P(predicate(f(theta)))`` over the finite draws."""
        vals, _ = self._codomain(f)
        rows = vals[np.all(np.isfinite(vals), axis=1)]
        if rows.shape[0] == 0:
            return float("nan")
        hits = np.fromiter(
            (bool(predicate(rows[i])) for i in range(rows.shape[0])),
            dtype=bool,
            count=rows.shape[0],
        )
        return float(hits.mean())

    def sample(self, key: jax.Array, n: int, f: Functional = None) -> np.ndarray:
        """``n`` resamples (with replacement) of the finite codomain draws."""
        vals, _ = self._codomain(f)
        rows = vals[np.all(np.isfinite(vals), axis=1)]
        if rows.shape[0] == 0:
            return np.full((n, vals.shape[1]), np.nan)
        idx = np.asarray(jax.random.randint(key, (n,), 0, rows.shape[0]))
        return rows[idx]

    # -- conditioning (extends conditioning.given) --------------------------
    def given(
        self,
        event: Any,
        *,
        negate: bool = False,
        acknowledge_conditional: bool = False,
    ) -> EmpiricalLaw:
        """The conditional law on the sub-population where ``event`` holds.

        Extends :func:`emu_gmm.studies.given`: returns another
        :class:`EmpiricalLaw` masked to the selected replicates, preserving the
        :class:`~emu_gmm.studies.SelectionConditionalWarning` soft gate (a
        coverage / size summary over a ``binding_ridge`` /
        ``sigma_meat_indefinite`` subset is selection-conditional, not nominal).
        Masking destroys the rep-index alignment CRN relies on, so a
        conditioned law can never be :meth:`couple`-d (pair first, then
        condition a paired contrast).
        """
        if self._records is None:
            return self._given_from_events(
                event, negate=negate, acknowledge_conditional=acknowledge_conditional
            )
        masked = given(
            self._records,
            event,
            negate=negate,
            acknowledge_conditional=acknowledge_conditional,
        )
        return EmpiricalLaw(
            draws=np.asarray(masked.theta_flat),
            used=np.asarray(masked.converged) > 0.5,
            names=tuple(masked.param_names),
            component_shapes=self._component_shapes,
            records=masked,  # a bare masked FitRecord -> not CRN-pairable
            coupling_id=None,
            key=None,
            label=f"{self._label} | given",
            conditioned=True,
        )

    def _given_from_events(
        self, event: Any, *, negate: bool, acknowledge_conditional: bool
    ) -> EmpiricalLaw:
        """:meth:`given` for a raw-draws law carrying an ``events`` dict (#181).

        Masks the draws (and every event flag) by ``events[event]``, mirroring
        the records-backed path's selection-conditional soft gate so a
        ``binding_ridge`` / ``sigma_meat_indefinite`` subset still warns. The
        result is a conditioned raw-draws law (never CRN-pairable).
        """
        import warnings

        from emu_gmm.studies.conditioning import (
            SELECTION_CONDITIONAL_FLAGS,
            SelectionConditionalWarning,
        )

        if (
            self._events is None
            or not isinstance(event, str)
            or event not in self._events
        ):
            available = tuple(self._events) if self._events is not None else ()
            raise TypeError(
                "EmpiricalLaw.given() needs either a records-backed law (built "
                "via from_records, carrying FitRecord event flags) or a raw-draws "
                f"law with a matching events= entry. Got event {event!r}; this "
                f"law's events are {available}. (Build with "
                "EmpiricalLaw.from_draws(..., events={name: mask}).)"
            )
        mask = np.asarray(self._events[event])
        if negate:
            mask = ~mask
        if event in SELECTION_CONDITIONAL_FLAGS and not acknowledge_conditional:
            warnings.warn(
                f"EmpiricalLaw.given({event!r}): conditioning on an "
                "estimator-internal flag is selection-conditional, not nominal "
                "(the event is a function of the same draws as the statistic). "
                "Pass acknowledge_conditional=True to silence this.",
                SelectionConditionalWarning,
                stacklevel=3,
            )
        sub_events = {k: v[mask] for k, v in self._events.items()}
        return EmpiricalLaw(
            draws=self._draws[mask],
            used=self._used[mask],
            names=self._names,
            component_shapes=self._component_shapes,
            records=None,
            coupling_id=None,
            key=None,
            events=sub_events,
            label=f"{self._label} | given({event})",
            conditioned=True,
        )

    def event_share(self, event: Any, *, negate: bool = False) -> EventShare:
        """The loud size of a :meth:`given` selection (both denominators).

        Reuses :func:`emu_gmm.studies.event_share`.
        """
        if self._records is None:
            raise TypeError(
                "EmpiricalLaw.event_share() needs a records-backed law (built via "
                "from_records); this law carries no event flags."
            )
        return event_share(self._records, event, negate=negate)

    # -- couplings (reuses crn_pair verbatim) -------------------------------
    def couple(
        self, other: EmpiricalLaw, *, assert_coupled: bool = False
    ) -> CoupledRecords:
        """Verify a shared CRN probability space, then couple with ``other``.

        Reuses :func:`emu_gmm.studies.crn_pair` verbatim: the joint constructor
        checks ``n_reps`` / ``param_names`` / master ``key`` / ``coupling_id``
        and RAISES on a mismatch (master-key equality is necessary but not
        sufficient; the ``coupling_id`` witnesses the DGP's split scheme). Both
        laws must be whole-sweep records-backed --- a conditioned/masked or
        raw-draws law cannot be paired (masking destroys rep alignment).
        """
        if not isinstance(self._records, MCRecords) or not isinstance(
            other._records, MCRecords
        ):
            raise TypeError(
                "EmpiricalLaw.couple() requires two whole-sweep records-backed "
                "EmpiricalLaws (from_records over an MCRecords). A conditioned "
                "(given()) or raw-draws (from_draws) law cannot be CRN-paired: "
                "masking destroys the rep-index alignment couplings rely on. "
                "Pair first, then condition a paired contrast with "
                "flips(..., where=event_mask)."
            )
        return crn_pair(self._records, other._records, assert_coupled=assert_coupled)

    # -- gauge-aware codomain conveniences ----------------------------------
    def eigenvalue_se(self, rank: int, *, index: int = 0) -> np.ndarray:
        """Empirical SE of the ``rank`` nonzero eigenvalues of ``Gamma = A A^T``.

        Applies the gauge-invariant eigenvalue functional per draw (landmine 3)
        and routes the SE through :class:`~emu_gmm.inference.BootstrapSE`.
        """
        return self.se(eigenvalue_functional(rank, index))

    def eigenvalue_quantile(self, q: float, rank: int, *, index: int = 0) -> np.ndarray:
        """Empirical ``q``-quantile of the ``rank`` nonzero eigenvalues of Gamma."""
        return self.quantile(q, eigenvalue_functional(rank, index))

    def gamma_se(self, *, index: int = 0) -> np.ndarray:
        """Empirical SE of ``vech(Gamma)``, ``Gamma = A @ A.T`` (gauge-invariant)."""
        return self.se(gamma_functional(index))

    # -- summarizer delegations (charter item d; numeric-identical) ----------
    def bias_sd(self, theta0: Any) -> _summaries.BiasSD:
        """Bias / MC-SD / mean-SE summary, via :func:`emu_gmm.studies.bias_sd`."""
        return _summaries.bias_sd(self._require_records("bias_sd"), theta0)

    def coverage(self, theta0: Any, level: float = 0.95) -> _summaries.Coverage:
        """Wald CI coverage, via :func:`emu_gmm.studies.coverage`."""
        return _summaries.coverage(self._require_records("coverage"), theta0, level)

    def size_power(
        self, alpha: tuple[float, ...] = (0.01, 0.05, 0.10)
    ) -> _summaries.SizePower:
        """J-test rejection rates, via :func:`emu_gmm.studies.size_power`."""
        return _summaries.size_power(self._require_records("size_power"), alpha)

    def tau_binding(
        self, q: tuple[float, ...] = (0.05, 0.25, 0.5, 0.75, 0.95)
    ) -> _summaries.TauBinding:
        """Ridge-binding summary, via :func:`emu_gmm.studies.tau_binding`."""
        return _summaries.tau_binding(self._require_records("tau_binding"), q)

    def j_calibration(self) -> _summaries.JCalibration:
        """J p-value calibration, via :func:`emu_gmm.studies.j_calibration`."""
        return _summaries.j_calibration(self._require_records("j_calibration"))

    def _require_records(self, what: str) -> MCRecords | FitRecord:
        if self._records is None:
            raise TypeError(
                f"EmpiricalLaw.{what}() needs a records-backed law (built via "
                "from_records); this law was built from raw draws and carries no "
                "FitRecord (theta0 / J-triple / convergence) fields."
            )
        return self._records

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"EmpiricalLaw(grade={self.grade!r}, n_used={self.n_used}/"
            f"{self.n_draws}, names={self._names}, conditioned={self._conditioned})"
        )


# ---------------------------------------------------------------------------
# Asymptotic grade: the Gaussian law N(theta_hat, Sigma_theta).
# ---------------------------------------------------------------------------
class AsymptoticLaw(EstimatorLaw):
    r"""The first-order Gaussian law :math:`\mathcal N(\hat\theta, \Sigma_\theta)`.

    Wraps an :class:`~emu_gmm.types.EstimationResult`; its queries are the delta
    method, which already exists as
    :func:`~emu_gmm.types.EstimationResult.functional_se` (and the gauge-aware
    :meth:`~emu_gmm.types.EstimationResult.eigenvalue_se` /
    :meth:`~emu_gmm.types.EstimationResult.gamma_se`). A gauge-invariant
    functional ``f`` is integrated against the Gaussian *to first order*; the
    gauge nullspace is already pinned out of :math:`\Sigma_\theta`, so the
    eigenvalue / ``gamma`` codomain is gauge-invariant by construction.

    :meth:`given` and :meth:`prob` REFUSE: the conditional of a Gaussian given a
    data-dependent event, and the probability of a general predicate under the
    Gaussian, have no closed form, and the framework refuses rather than
    approximates (``docs/design.org``). Use :meth:`sample` to draw the law and
    build an :class:`EmpiricalLaw` if a Monte Carlo answer is wanted.
    """

    grade = "asymptotic"

    def __init__(
        self,
        result: EstimationResult | None = None,
        *,
        label: str | None = None,
        _backing: "_MomentsBacking | None" = None,
    ) -> None:
        # Two backings (one query algebra): a LIVE EstimationResult, or a
        # moments-only record reconstructed from persisted arrays
        # (``from_moments`` / a reloaded LawState, #181). The query methods
        # route to the SAME delta-method machinery either way --- the moments
        # path calls ``inference.functional_se`` directly, so no statistic is
        # re-implemented for the reloaded grade.
        if _backing is not None:
            self._result = None
            self._backing: _MomentsBacking | None = _backing
            self._names = _backing.names
            self._label = label or "asymptotic(from_moments)"
            return
        if not isinstance(result, EstimationResult):
            raise TypeError(
                "AsymptoticLaw wraps an EstimationResult (the emu_gmm.estimate "
                f"output); got {type(result).__name__}. To reconstruct without "
                "a live result use AsymptoticLaw.from_moments(...)."
            )
        self._result = result
        self._backing = None
        self._label = label or "asymptotic(N(theta_hat, Sigma_theta))"
        self._names = tuple(result.record().param_names)

    # -- moments-only constructor (#181) ------------------------------------
    @classmethod
    def from_moments(
        cls,
        theta_components: Any,
        sigma_theta: Any,
        *,
        leaf_specs: Any = None,
        names: tuple[str, ...] | None = None,
        label: str | None = None,
    ) -> AsymptoticLaw:
        r"""Reconstruct the asymptotic law from raw moments --- no live result.

        The reload target for a persisted asymptotic law (#181): a queryable
        :math:`\mathcal N(\hat\theta, \Sigma_\theta)` built from the component
        arrays and :math:`\Sigma_\theta` alone. ``se`` / ``functional_se`` /
        ``eigenvalue_se`` / ``gamma_se`` route through
        :mod:`emu_gmm.inference.functional_se` exactly as the result-backed law
        does. ``given`` / ``prob`` still refuse (no closed form at this grade).

        Parameters
        ----------
        theta_components
            The per-leaf ambient arrays ``(A, phi, ...)`` --- what
            :meth:`EstimationResult.components` returns.
        sigma_theta
            The ambient ``(D, D)`` covariance, gauge nullspace pinned to zero.
            ``D`` must equal the total flat size of ``theta_components``.
        leaf_specs
            Optional per-leaf manifold instances (``Euclidean`` /
            ``PSDFixedRank`` / ...), aligned with ``theta_components``. Used
            only to locate the unique ``PSDFixedRank`` factor (and its rank) so
            :meth:`eigenvalue_se` / :meth:`gamma_se` work; omit for an
            all-Euclidean law (those conveniences then raise, as on a
            no-PSD-leaf result).
        names
            Ambient tangent labels (length ``D``); positional by default.
        """
        comps = tuple(np.asarray(c, dtype=float) for c in theta_components)
        sigma = np.atleast_2d(np.asarray(sigma_theta, dtype=float))
        psd_index, psd_rank = _locate_psd_leaf(leaf_specs)
        leaf_tags = (
            tuple(_manifold_to_tag(m) for m in leaf_specs)
            if leaf_specs is not None
            else tuple(_euclidean_tag(c) for c in comps)
        )
        backing = _MomentsBacking.build(
            components=comps,
            sigma=sigma,
            names=names,
            leaf_tags=leaf_tags,
            psd_index=psd_index,
            psd_rank=psd_rank,
        )
        return cls(_backing=backing, label=label)

    @property
    def param_names(self) -> tuple[str, ...]:
        return self._names

    @property
    def result(self) -> EstimationResult:
        """The wrapped :class:`~emu_gmm.types.EstimationResult` (live-backed only).

        Raises if this law was reconstructed via :meth:`from_moments` (a
        reloaded law has no live result --- query it directly).
        """
        if self._result is None:
            raise AttributeError(
                "AsymptoticLaw.result: this law was reconstructed from moments "
                "(from_moments / a reloaded LawState) and has no live "
                "EstimationResult. Query the law directly (se, functional_se, "
                "eigenvalue_se, gamma_se)."
            )
        return self._result

    # -- backing-aware accessors (one algebra, two backings) ----------------
    def _components(self) -> tuple[Any, ...]:
        return (
            self._result.components()
            if self._result is not None
            else self._backing.components  # type: ignore[union-attr]
        )

    def _functional_cov(self, f: Functional) -> np.ndarray:
        # ``f`` is non-None here (cov() routes f=None to the plain covariance).
        assert f is not None
        if self._result is not None:
            _se, cov = self._result.functional_se(f)
            return np.atleast_2d(np.asarray(cov))
        from emu_gmm.inference.functional_se import functional_se as _fse

        assert self._backing is not None
        _se, cov = _fse(f, self._backing.components, jnp.asarray(self._backing.sigma))
        return np.atleast_2d(np.asarray(cov))

    def mean(self, f: Functional = None) -> np.ndarray:
        comps = self._components()
        if f is None:
            return np.asarray(_flatten_components(comps))
        return np.asarray(jnp.atleast_1d(jnp.asarray(f(tuple(comps)))))

    def cov(self, f: Functional = None) -> np.ndarray:
        if f is None:
            if self._result is not None:
                return np.asarray(self._result.Sigma_theta.array)
            return np.asarray(self._backing.sigma)  # type: ignore[union-attr]
        return self._functional_cov(f)

    def quantile(self, q: float, f: Functional = None) -> np.ndarray:
        """The Gaussian marginal ``q``-quantile, ``mean + z_q * se`` (closed form).

        Exact for the per-coordinate marginal of a Gaussian (not an
        approximation): each coordinate of :math:`f(\\hat\\theta)` is normal
        with the delta-method SE.
        """
        if not 0.0 < q < 1.0:
            raise ValueError(f"quantile(): q must be in (0, 1), got {q}")
        z = float(scipy.stats.norm.ppf(q))
        return self.mean(f) + z * self.se(f)

    def prob(
        self, predicate: Callable[[np.ndarray], Any], f: Functional = None
    ) -> float:
        raise NotImplementedError(
            "AsymptoticLaw.prob refuses: the probability of a general predicate "
            "indicator under the Gaussian law has no closed form, and the "
            "framework refuses rather than approximates (design.org). Draw the "
            "law with sample(key, n) and build an EmpiricalLaw if a Monte Carlo "
            "answer is acceptable."
        )

    def sample(self, key: jax.Array, n: int, f: Functional = None) -> np.ndarray:
        """``n`` draws from :math:`\\mathcal N(\\mathrm{mean}(f), \\mathrm{cov}(f))`.

        Sampling the known fitted Gaussian law is a closed-form operation (not
        an approximation of a conditional). The covariance can be rank-deficient
        (the gauge nullspace is pinned to zero), so the SVD method is used.
        """
        mean = jnp.asarray(self.mean(f))
        cov = jnp.asarray(np.atleast_2d(self.cov(f)))
        draws = jax.random.multivariate_normal(key, mean, cov, shape=(n,), method="svd")
        return np.asarray(draws)

    def given(
        self,
        event: Any,
        *,
        negate: bool = False,
        acknowledge_conditional: bool = False,
    ) -> EstimatorLaw:
        raise NotImplementedError(
            "AsymptoticLaw.given refuses: the conditional law of a Gaussian "
            "estimator given a data-dependent event (binding_ridge, "
            "sigma_meat_indefinite, ...) has no closed form at the asymptotic "
            "grade, and the framework refuses rather than approximates "
            "(design.org: 'given returns another EstimatorLaw ... the "
            "implementation refuses rather than approximates'). Condition at an "
            "empirical grade (EmpiricalLaw.given over a record stack)."
        )

    # -- gauge-aware codomain conveniences (reuse #117 leaf detection) ------
    def eigenvalue_se(self, rank: int | None = None) -> np.ndarray:
        """Delta-method SE of the nonzero eigenvalues of ``Gamma = A @ A.T``.

        Result-backed: reuses :meth:`EstimationResult.eigenvalue_se` (which
        locates the unique ``PSDFixedRank`` leaf via the manifold spec, #117).
        Moments-backed (reloaded): routes through
        :func:`emu_gmm.inference.functional_se.eigenvalue_se` with the persisted
        PSD-leaf index and rank. Gauge-invariant by construction; raises if the
        law carries no ``PSDFixedRank`` factor.
        """
        if self._result is not None:
            return np.asarray(self._result.eigenvalue_se(rank))
        from emu_gmm.inference.functional_se import eigenvalue_se as _ev

        b, idx, k = self._require_psd("eigenvalue_se")
        se, _cov = _ev(
            b.components,
            jnp.asarray(b.sigma),
            int(rank) if rank is not None else int(k),
            index=idx,
        )
        return np.asarray(se)

    def gamma_se(self) -> np.ndarray:
        """Delta-method SE of ``vech(Gamma)`` (gauge-invariant; #117 leaf)."""
        if self._result is not None:
            return np.asarray(self._result.gamma_se())
        from emu_gmm.inference.functional_se import gamma_se as _gse

        b, idx, _k = self._require_psd("gamma_se")
        se, _cov = _gse(b.components, jnp.asarray(b.sigma), index=idx)
        return np.asarray(se)

    def gamma_covariance(self) -> np.ndarray:
        """Delta-method covariance of ``vech(Gamma)``."""
        if self._result is not None:
            return np.asarray(self._result.gamma_covariance())
        from emu_gmm.inference.functional_se import gamma_se as _gse

        b, idx, _k = self._require_psd("gamma_covariance")
        _se, cov = _gse(b.components, jnp.asarray(b.sigma), index=idx)
        return np.asarray(cov)

    def _require_psd(self, what: str) -> tuple[_MomentsBacking, int, int]:
        """Return ``(backing, psd_index, psd_rank)`` for a moments-backed law, or raise."""
        b = self._backing
        if b is None or b.psd_index is None or b.psd_rank is None:
            raise TypeError(
                f"AsymptoticLaw.{what}: this (moments-backed) law carries no "
                "PSDFixedRank factor, so Gamma = A @ A.T is undefined. Pass "
                "leaf_specs with the PSDFixedRank leaf to from_moments, or use "
                "functional_se(f) with an explicit functional."
            )
        return b, int(b.psd_index), int(b.psd_rank)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"AsymptoticLaw(grade={self.grade!r}, names={self._names})"


def couple(
    a: EmpiricalLaw, b: EmpiricalLaw, *, assert_coupled: bool = False
) -> CoupledRecords:
    """Verify two empirical laws share a CRN probability space, then couple them.

    Free-function form of :meth:`EmpiricalLaw.couple` --- a thin alias over
    :func:`emu_gmm.studies.crn_pair`. The joint constructor checks
    key/provenance compatibility before zipping and RAISES on a mismatch (the
    CRN contract).
    """
    if not isinstance(a, EmpiricalLaw) or not isinstance(b, EmpiricalLaw):
        raise TypeError(
            "couple() pairs two EmpiricalLaw instances; the asymptotic grade has "
            "its own closed-form coupled object (stacked influence functions), "
            "not implemented here."
        )
    return a.couple(b, assert_coupled=assert_coupled)
