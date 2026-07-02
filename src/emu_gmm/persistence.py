r"""Typed, versioned persistence for an :class:`~emu_gmm.law.EstimatorLaw` (#181).

The estimate-once / query-many artifact for the K-Aggregators / CerealDemand
paper workflow: persist a fitted law to an **inert, cross-version** file and
reload it into a queryable law in the downstream paper repo --- *not* a pickle.

Why not pickle (the issue's case, which stands even though #147 fixed the
``ManifoldLeaf`` round-trip): pickle serialises by class reference, so a field
rename / module move silently breaks an artifact a referee reloads under a
future emu; and it can entomb model code (an exec/RCE surface). :class:`LawState`
is a typed, ``schema_version``-stamped record; the on-disk container is a single
``.npz`` (numpy + stdlib only --- no new runtime dependency) carrying the arrays
plus a JSON manifest, reconstructed through the normal constructors
(:meth:`AsymptoticLaw.from_moments`, the :func:`tag_to_manifold` codec) rather
than by resurrecting a live object.

**Both grades are supported.**

- *Asymptotic* --- SEs / ``eigenvalue_se`` / ``gamma_se`` / functional SEs off
  :math:`\mathcal N(\hat\theta, \Sigma_\theta)` (the homogeneous clustered / BLP
  standard errors). Reloads moments-backed (no live ``EstimationResult``).
- *Empirical* --- the consumer's primary need (the K-Aggregators heterogeneous
  law is bootstrap-only, the over-identification test is the cluster-wild Hansen
  :math:`J` = the empirical law of :math:`Q`, and the binding-ridge / indefinite
  -meat conditionals are ``given`` queries). Two backings round-trip:

  * *records-backed* (``EmpiricalLaw.from_records`` over an ``MCRecords`` /
    ``FitRecord`` stack): the full :class:`FitRecord` arrays + ``coupling_id`` /
    ``key`` provenance are persisted --- and the #183 per-draw custom
    statistics (``MCRecords.extra``) ride along --- so a reloaded law's
    ``given`` / ``couple`` / ``size_power`` (J) / ``se`` all work through the
    existing ``emu_gmm.studies`` reuse --- nothing re-implemented.
  * *draws-backed* (``from_draws``, e.g. a wild-bootstrap ``J_boot``): the draws
    + ``{0,1}^E`` event flags round-trip, so ``pvalue`` / ``se`` / ``quantile``
    and ``given`` over the persisted events work on reload.

  A *conditioned* (``given``-masked) law is refused --- persist the whole sweep
  and re-condition on reload.

**Companion, not successor** (the issue's recommendation): ``LawState`` is the
durable projection; :class:`~emu_gmm.types.EstimationResult` stays the live,
compute-coupled estimation output and is *not* on the reload path.
"""

from __future__ import annotations

import dataclasses
import json
from typing import Any

import numpy as np

from emu_gmm._internal.law_state import (
    SCHEMA_VERSION,
    MomentsBacking,
    euclidean_tag,
    locate_psd_leaf,
    manifold_to_tag,
    tag_to_manifold,
)
from emu_gmm.law import AsymptoticLaw, EmpiricalLaw, EstimatorLaw
from emu_gmm.studies.driver import MCRecords
from emu_gmm.types import FitRecord


@dataclasses.dataclass(frozen=True)
class FactorySpec:
    r"""Typed record of the estimator configuration that is part of T's identity (#142).

    The frequentist law is the law of the *estimator* ``T``, and ``T``'s identity
    includes the analyst-chosen configuration (weighting, regularisation,
    covariance, optimiser, ...; ``docs/design.org`` §EstimatorLaw, the #142
    anchor). This typed slot records that configuration alongside a persisted
    law so a reloaded artifact is self-describing --- a typed record, not a loose
    ``dict``.

    The strategy *objects* themselves are not serialisable (and are model code,
    which must not ride a data artifact), so each is recorded as its **type
    name** (a string). ``extra`` is a free, JSON-serialisable slot for the
    consumer's own ``Spec`` payload (e.g. CerealDemand's estimator spec).

    Attributes
    ----------
    weighting, regularization, covariance, optimizer, penalty, model
        Type-name strings of the resolved strategies (or ``None``).
    moment_names
        The moment labels, when known.
    extra
        Consumer-defined JSON-able payload (the embedded ``Spec``).
    """

    weighting: str | None = None
    regularization: str | None = None
    covariance: str | None = None
    optimizer: str | None = None
    penalty: str | None = None
    model: str | None = None
    moment_names: tuple[str, ...] | None = None
    extra: dict[str, Any] = dataclasses.field(default_factory=dict)

    @classmethod
    def from_estimator(
        cls, factory_spec: dict[str, Any], *, extra: dict[str, Any] | None = None
    ) -> FactorySpec:
        """Build from a ``build_estimator`` ``_emu_gmm_factory_spec`` dict.

        Records each resolved strategy as its type name (the objects are model
        code and never persisted). ``extra`` carries the consumer's own Spec.
        """

        def _name(obj: Any) -> str | None:
            if obj is None:
                return None
            return getattr(type(obj), "__name__", None) or repr(type(obj))

        mn = factory_spec.get("moment_names")
        return cls(
            weighting=_name(factory_spec.get("weighting")),
            regularization=_name(factory_spec.get("regularization")),
            covariance=_name(factory_spec.get("covariance")),
            optimizer=_name(factory_spec.get("optimizer")),
            penalty=_name(factory_spec.get("penalty")),
            model=_name(factory_spec.get("model")),
            moment_names=None if mn is None else tuple(str(m) for m in mn),
            extra=dict(extra or {}),
        )

    def to_json(self) -> dict[str, Any]:
        """A plain JSON-serialisable dict (rides the inert manifest)."""
        d = dataclasses.asdict(self)
        d["moment_names"] = (
            None if self.moment_names is None else list(self.moment_names)
        )
        return d

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> FactorySpec:
        """Inverse of :meth:`to_json`."""
        mn = d.get("moment_names")
        return cls(
            weighting=d.get("weighting"),
            regularization=d.get("regularization"),
            covariance=d.get("covariance"),
            optimizer=d.get("optimizer"),
            penalty=d.get("penalty"),
            model=d.get("model"),
            moment_names=None if mn is None else tuple(mn),
            extra=dict(d.get("extra") or {}),
        )


def _coerce_factory_spec(value: Any) -> FactorySpec | None:
    """Accept a :class:`FactorySpec`, a plain dict (-> ``extra``), or ``None``."""
    if value is None or isinstance(value, FactorySpec):
        return value
    if isinstance(value, dict):
        # Back-compat with the old dict[str, Any] usage: treat a bare dict as the
        # consumer's payload.
        return FactorySpec(extra=_jsonable(value, what="factory_spec"))
    raise TypeError(
        f"save_law: factory_spec must be a FactorySpec, a dict, or None; got "
        f"{type(value).__name__}."
    )


# The stackable (per-rep array) fields of a FitRecord, in declaration order.
_FITRECORD_ARRAY_FIELDS = (
    "theta_flat",
    "se",
    "J_stat",
    "J_pvalue",
    "J_pvalue_adjusted",
    "converged",
    "tau_realised",
    "binding_ridge",
    "sigma_meat_indefinite",
)


@dataclasses.dataclass(frozen=True)
class LawState:
    r"""Typed, versioned persistence record for an :class:`EstimatorLaw` (#181).

    The contract a consumer imports rather than reverse-engineers. Validated on
    load (``schema_version`` + array/shape consistency); the arrays
    (``theta_components`` / ``sigma_theta``) ride the ``.npz`` payload while the
    rest is the JSON manifest.

    Attributes
    ----------
    schema_version
        Layout version; :func:`load_law` refuses an unknown version with a
        migration hint rather than mis-parsing.
    grade
        The epistemic grade carried (``"asymptotic"`` in this slice).
    param_names
        Ambient tangent labels (length ``D``).
    leaf_tags
        Per-component typed manifold tags (e.g. ``{"type": "PSDFixedRank",
        "n": 5, "k": 2}``); reconstructed via :func:`tag_to_manifold`, never a
        live ``ManifoldLeaf`` (#147).
    component_shapes
        Per-component ambient shapes (``()`` for a scalar leaf).
    theta_components
        The per-leaf point arrays ``(A, phi, ...)``.
    sigma_theta
        The ``(D, D)`` covariance (gauge nullspace pinned to zero).
    psd_index, psd_rank
        Component index and rank of the unique ``PSDFixedRank`` factor, or
        ``None`` --- so a reloaded law answers ``eigenvalue_se`` / ``gamma_se``.
    diagnostics
        Scalar provenance (``J_stat``, ``J_dof``, ``gauge_nullspace_dim``, ...).
    factory_spec
        Optional typed :class:`FactorySpec` recording the estimator
        configuration that is part of T's identity (#142 anchor); ``None`` when
        none was supplied.
    """

    schema_version: int
    grade: str
    param_names: tuple[str, ...]
    leaf_tags: tuple[dict[str, Any], ...]
    component_shapes: tuple[tuple[int, ...], ...] | None
    diagnostics: dict[str, Any]
    factory_spec: FactorySpec | None = None
    # -- asymptotic grade --
    theta_components: tuple[np.ndarray, ...] | None = None
    sigma_theta: np.ndarray | None = None
    psd_index: int | None = None
    psd_rank: int | None = None
    # -- empirical grade --
    backing: str | None = None  # "records" | "draws"
    record_arrays: dict[str, np.ndarray] | None = None  # the FitRecord fields
    # The #183 per-draw custom statistics (``MCRecords.extra``), leading
    # ``n_reps`` axis on every column; ``None`` when none were collected.
    extra_arrays: dict[str, np.ndarray] | None = None
    j_dof: int | None = None
    n_reps: int | None = None
    draws: np.ndarray | None = None  # draws-backed
    used: np.ndarray | None = None
    events: dict[str, np.ndarray] | None = None
    coupling_id: Any = None
    key: np.ndarray | None = None
    conditioned: bool = False


def _asymptotic_state(law: AsymptoticLaw) -> LawState:
    """Build a :class:`LawState` from an asymptotic law (either backing)."""
    if law._result is None:  # moments-backed (e.g. a re-save of a reloaded law)
        b = law._backing
        assert b is not None
        return LawState(
            schema_version=SCHEMA_VERSION,
            grade="asymptotic",
            param_names=b.names,
            leaf_tags=b.leaf_tags,
            component_shapes=b.component_shapes,
            theta_components=b.components,
            sigma_theta=b.sigma,
            psd_index=b.psd_index,
            psd_rank=b.psd_rank,
            diagnostics=dict(b.diagnostics),
            factory_spec=b.factory_spec,
        )

    # Live-result-backed: extract the durable projection from the result.
    result = law._result
    comps = tuple(np.asarray(c) for c in result.components())
    sigma = np.asarray(result.Sigma_theta.array)
    spec = result.manifold_spec
    if spec is not None:
        leaf_manifolds = [ls.manifold for ls in spec.leaf_specs]
        leaf_tags = tuple(manifold_to_tag(m) for m in leaf_manifolds)
        psd_index, psd_rank = locate_psd_leaf(leaf_manifolds)
    else:
        # v1 / all-scalar tree: every component is a Euclidean leaf, no gauge.
        leaf_tags = tuple(euclidean_tag(c) for c in comps)
        psd_index, psd_rank = None, None
    component_shapes = tuple(tuple(int(s) for s in np.shape(c)) for c in comps)

    diag = result.diagnostics
    diagnostics: dict[str, Any] = {
        "J_stat": float(np.asarray(result.J_stat)),
        "J_dof": int(result.J_dof),
        "J_pvalue": float(np.asarray(result.J_pvalue)),
        "gauge_nullspace_dim": int(diag.gauge_nullspace_dim),
        "tau_realised": float(np.asarray(diag.tau_realised)),
        "kappa_V": float(np.asarray(diag.kappa_V)),
    }
    return LawState(
        schema_version=SCHEMA_VERSION,
        grade="asymptotic",
        param_names=tuple(law.param_names),
        leaf_tags=leaf_tags,
        component_shapes=component_shapes,
        theta_components=comps,
        sigma_theta=sigma,
        psd_index=psd_index,
        psd_rank=psd_rank,
        diagnostics=diagnostics,
        factory_spec=None,
    )


def _jsonable(value: Any, *, what: str) -> Any:
    """Return ``value`` if it round-trips through JSON, else raise a clear error."""
    try:
        json.loads(json.dumps(value))
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"save_law: {what} of type {type(value).__name__} is not "
            "JSON-serialisable, so it cannot ride the inert manifest. Use a "
            "JSON-able token (int / str / None) for it."
        ) from exc
    return value


def _empirical_state(law: EmpiricalLaw) -> LawState:
    """Build a :class:`LawState` from an empirical law (records- or draws-backed)."""
    if law._conditioned:
        raise NotImplementedError(
            "save_law: a conditioned (given()-masked) EmpiricalLaw is a "
            "sub-population, not a durable artifact -- persist the whole-sweep "
            "law and re-condition with given() on reload."
        )
    shapes = (
        None
        if law._component_shapes is None
        else tuple(tuple(int(s) for s in sh) for sh in law._component_shapes)
    )

    if law._records is not None:
        # Records-backed: persist the FitRecord stack (+ MCRecords provenance).
        rec = (
            law._records.records
            if isinstance(law._records, MCRecords)
            else law._records
        )
        record_arrays = {
            f: np.asarray(getattr(rec, f)) for f in _FITRECORD_ARRAY_FIELDS
        }
        key = None
        coupling_id = None
        extra_arrays = None
        n_reps = int(np.asarray(rec.theta_flat).shape[0])
        if isinstance(law._records, MCRecords):
            key = None if law._records.key is None else np.asarray(law._records.key)
            coupling_id = _jsonable(law._records.coupling_id, what="coupling_id")
            n_reps = int(law._records.n_reps)
            if law._records.extra is not None:
                # The #183 per-draw statistics channel: every column carries
                # the same leading n_reps axis as the FitRecord fields.
                extra_arrays = {
                    str(nm): np.asarray(v) for nm, v in law._records.extra.items()
                }
                for nm, a in extra_arrays.items():
                    if a.shape[:1] != (n_reps,):
                        raise ValueError(
                            f"save_law: extra statistic {nm!r} has shape "
                            f"{a.shape}; every MCRecords.extra column must "
                            f"carry the leading n_reps={n_reps} replication "
                            "axis, like the FitRecord fields."
                        )
        return LawState(
            schema_version=SCHEMA_VERSION,
            grade="empirical",
            param_names=tuple(rec.param_names),
            leaf_tags=(),
            component_shapes=shapes,
            diagnostics={},
            backing="records",
            record_arrays=record_arrays,
            extra_arrays=extra_arrays,
            j_dof=int(rec.J_dof),
            n_reps=n_reps,
            key=key,
            coupling_id=coupling_id,
        )

    # Draws-backed: persist the raw draws + {0,1}^E event flags.
    events = (
        None
        if law._events is None
        else {k: np.asarray(v, dtype=bool) for k, v in law._events.items()}
    )
    return LawState(
        schema_version=SCHEMA_VERSION,
        grade="empirical",
        param_names=tuple(law._names),
        leaf_tags=(),
        component_shapes=shapes,
        diagnostics={},
        backing="draws",
        draws=np.asarray(law._draws),
        used=np.asarray(law._used, dtype=bool),
        events=events,
    )


def _write_state(state: LawState, target: Any) -> None:
    """Write ``state`` to a single ``.npz`` (arrays) + embedded JSON manifest."""
    manifest: dict[str, Any] = {
        "schema_version": state.schema_version,
        "grade": state.grade,
        "param_names": list(state.param_names),
        "leaf_tags": list(state.leaf_tags),
        "component_shapes": (
            None
            if state.component_shapes is None
            else [list(s) for s in state.component_shapes]
        ),
        "diagnostics": state.diagnostics,
        "factory_spec": (
            None if state.factory_spec is None else state.factory_spec.to_json()
        ),
        "backing": state.backing,
    }
    arrays: dict[str, np.ndarray] = {}

    if state.grade == "asymptotic":
        assert state.theta_components is not None
        manifest.update(
            psd_index=state.psd_index,
            psd_rank=state.psd_rank,
            n_components=len(state.theta_components),
            has_sigma=state.sigma_theta is not None,
        )
        for i, c in enumerate(state.theta_components):
            arrays[f"comp_{i}"] = np.asarray(c)
        if state.sigma_theta is not None:
            arrays["sigma_theta"] = np.asarray(state.sigma_theta)
    elif state.backing == "records":
        assert state.record_arrays is not None
        extra_names = [] if state.extra_arrays is None else sorted(state.extra_arrays)
        manifest.update(
            j_dof=state.j_dof,
            n_reps=state.n_reps,
            coupling_id=state.coupling_id,
            has_key=state.key is not None,
            extra_names=extra_names,
        )
        for f, a in state.record_arrays.items():
            arrays[f"rec_{f}"] = np.asarray(a)
        # "rec_extra__<name>" cannot collide with the fixed "rec_<field>" keys
        # above (no _FITRECORD_ARRAY_FIELDS entry starts with "extra__"), nor
        # with "key" / the draws-backed keys (a different branch entirely).
        for nm in extra_names:
            arrays[f"rec_extra__{nm}"] = np.asarray(state.extra_arrays[nm])  # type: ignore[index]
        if state.key is not None:
            arrays["key"] = np.asarray(state.key)
    else:  # draws-backed
        assert state.draws is not None and state.used is not None
        ev_names = [] if state.events is None else sorted(state.events)
        manifest.update(event_names=ev_names, conditioned=state.conditioned)
        arrays["draws"] = np.asarray(state.draws)
        arrays["used"] = np.asarray(state.used)
        for nm in ev_names:
            arrays[f"event_{nm}"] = np.asarray(state.events[nm])  # type: ignore[index]

    # The manifest rides as a 0-d unicode array -> no pickle needed on load.
    arrays["__manifest__"] = np.asarray(json.dumps(manifest))
    # typeshed's savez stub doesn't model keyword array names (only *args +
    # allow_pickle), hence the suppression.
    if hasattr(target, "write"):
        # An already-open binary file-like / buffer (an fsspec target, BytesIO,
        # ...): write directly, no temp round-trip. np.savez does NOT append
        # ".npz" to a file object, so the bytes are literal.
        np.savez(target, **arrays)  # type: ignore[arg-type]
    else:
        # A filesystem path: open it ourselves so the path is LITERAL (np.savez
        # would append ".npz" to a bare path, mismatching load_law's open).
        with open(target, "wb") as fh:
            np.savez(fh, **arrays)  # type: ignore[arg-type]


def _read_state(target: Any) -> LawState:
    """Read + validate a :class:`LawState` written by :func:`_write_state`."""
    with np.load(target, allow_pickle=False) as data:
        if "__manifest__" not in data.files:
            raise ValueError(
                f"load_law: {target!r} has no __manifest__ entry; it was not "
                "written by emu_gmm.save_law (or is a bare array .npz)."
            )
        manifest = json.loads(str(data["__manifest__"]))
        ver = int(manifest["schema_version"])
        if ver != SCHEMA_VERSION:
            raise ValueError(
                f"load_law: artifact schema_version {ver} != supported "
                f"{SCHEMA_VERSION}. This file was written by a different "
                "emu_gmm; no migration is registered for it yet. (A future "
                "version adds an upgrader table keyed on schema_version.)"
            )
        shapes = (
            None
            if manifest.get("component_shapes") is None
            else tuple(tuple(int(s) for s in sh) for sh in manifest["component_shapes"])
        )
        common = dict(
            schema_version=ver,
            grade=manifest["grade"],
            param_names=tuple(manifest["param_names"]),
            leaf_tags=tuple(manifest["leaf_tags"]),
            component_shapes=shapes,
            diagnostics=manifest["diagnostics"],
            factory_spec=(
                None
                if manifest.get("factory_spec") is None
                else FactorySpec.from_json(manifest["factory_spec"])
            ),
            backing=manifest.get("backing"),
        )
        if manifest["grade"] == "asymptotic":
            n = int(manifest["n_components"])
            return LawState(
                **common,
                theta_components=tuple(np.asarray(data[f"comp_{i}"]) for i in range(n)),
                sigma_theta=(
                    np.asarray(data["sigma_theta"])
                    if manifest.get("has_sigma")
                    else None
                ),
                psd_index=manifest["psd_index"],
                psd_rank=manifest["psd_rank"],
            )
        if manifest.get("backing") == "records":
            # Absent on pre-#183 artifacts -> extra=None (fully additive).
            extra_names = manifest.get("extra_names") or []
            return LawState(
                **common,
                record_arrays={
                    f: np.asarray(data[f"rec_{f}"]) for f in _FITRECORD_ARRAY_FIELDS
                },
                extra_arrays=(
                    None
                    if not extra_names
                    else {
                        nm: np.asarray(data[f"rec_extra__{nm}"]) for nm in extra_names
                    }
                ),
                j_dof=int(manifest["j_dof"]),
                n_reps=int(manifest["n_reps"]),
                coupling_id=manifest.get("coupling_id"),
                key=np.asarray(data["key"]) if manifest.get("has_key") else None,
            )
        ev_names = manifest.get("event_names") or []
        return LawState(
            **common,
            draws=np.asarray(data["draws"]),
            used=np.asarray(data["used"]),
            events=(
                None
                if not ev_names
                else {nm: np.asarray(data[f"event_{nm}"]) for nm in ev_names}
            ),
            conditioned=bool(manifest.get("conditioned", False)),
        )


def _state_to_asymptotic(state: LawState) -> AsymptoticLaw:
    """Reconstruct a moments-backed :class:`AsymptoticLaw` from ``state``.

    Rebuilds the leaf manifolds through their normal constructors
    (:func:`tag_to_manifold`) and re-derives the PSD-leaf ``(index, rank)`` from
    them --- the #147 "reconstruct via the constructor, never the immutable live
    leaf" contract, and a cross-check on the persisted indices.
    """
    leaf_manifolds = tuple(tag_to_manifold(t) for t in state.leaf_tags)
    psd_index, psd_rank = locate_psd_leaf(leaf_manifolds)
    assert state.theta_components is not None  # asymptotic grade invariant
    backing = MomentsBacking(
        components=state.theta_components,
        sigma=np.asarray(state.sigma_theta),
        names=state.param_names,
        leaf_tags=state.leaf_tags,
        component_shapes=state.component_shapes or (),
        psd_index=psd_index,
        psd_rank=psd_rank,
        diagnostics=dict(state.diagnostics),
        factory_spec=state.factory_spec,
    )
    return AsymptoticLaw(_backing=backing, label="asymptotic(loaded)")


def _state_to_empirical(state: LawState) -> EmpiricalLaw:
    """Reconstruct a queryable :class:`EmpiricalLaw` from ``state``.

    Records-backed: rebuild the :class:`FitRecord` (and its :class:`MCRecords`
    provenance) and go through ``from_records``, so ``given`` / ``couple`` /
    ``size_power`` / ``se`` reuse the existing ``emu_gmm.studies`` machinery.
    Draws-backed: ``from_draws`` with the persisted event flags.
    """
    import jax.numpy as jnp

    if state.backing == "records":
        assert state.record_arrays is not None
        record = FitRecord(
            **{f: jnp.asarray(state.record_arrays[f]) for f in _FITRECORD_ARRAY_FIELDS},
            J_dof=int(state.j_dof),  # type: ignore[arg-type]
            param_names=tuple(state.param_names),
        )
        if state.key is not None:
            extra = (
                None
                if state.extra_arrays is None
                else {nm: jnp.asarray(a) for nm, a in state.extra_arrays.items()}
            )
            mc = MCRecords(
                records=record,
                key=jnp.asarray(state.key),
                n_reps=int(state.n_reps),  # type: ignore[arg-type]
                coupling_id=state.coupling_id,
                extra=extra,
            )
            return EmpiricalLaw.from_records(
                mc, component_shapes=state.component_shapes, label="empirical(loaded)"
            )
        return EmpiricalLaw.from_records(
            record, component_shapes=state.component_shapes, label="empirical(loaded)"
        )

    # Draws-backed.
    return EmpiricalLaw.from_draws(
        np.asarray(state.draws),
        names=tuple(state.param_names),
        component_shapes=state.component_shapes,
        events=state.events,
        label="empirical(loaded)",
    )


def save_law(
    law: EstimatorLaw, target: Any, *, factory_spec: FactorySpec | dict | None = None
) -> None:
    """Persist ``law`` as a typed, versioned :class:`LawState` (#181).

    Parameters
    ----------
    law
        The law to persist --- :class:`~emu_gmm.law.AsymptoticLaw` or
        :class:`~emu_gmm.law.EmpiricalLaw`. A *conditioned* (``given``-masked)
        empirical law is refused (persist the whole sweep, re-condition on
        reload).
    target
        A filesystem path **or** an already-open binary file-like / buffer
        (``io.BytesIO``, an ``fsspec`` handle, ...). Passing a file-like lets an
        object-store / URI-addressed law store (e.g. ``s3://``) write directly
        with no temp round-trip --- symmetric with :func:`load_law`, which goes
        through :func:`numpy.load` and already accepts a buffer.
    factory_spec
        Optional typed :class:`FactorySpec` recording the estimator
        configuration that is part of T's identity (#142): which weighting /
        regularisation / covariance / optimiser produced the law, plus an
        ``extra`` slot for the consumer's own ``Spec``. A plain ``dict`` is
        accepted and stored as the ``extra`` payload (back-compat). Round-trips
        on the inert manifest (no model code; type-name strings only) and comes
        back on ``LawState.factory_spec`` / via :func:`load_law_state`.
    """
    if isinstance(law, AsymptoticLaw):
        state = _asymptotic_state(law)
    elif isinstance(law, EmpiricalLaw):
        state = _empirical_state(law)
    else:
        raise NotImplementedError(
            f"save_law: don't know how to persist {type(law).__name__}; expected "
            "an AsymptoticLaw or EmpiricalLaw."
        )
    resolved = _coerce_factory_spec(factory_spec)
    if resolved is not None:
        state = dataclasses.replace(state, factory_spec=resolved)
    _write_state(state, target)


def load_law(target: Any) -> EstimatorLaw:
    """Load a law persisted by :func:`save_law`.

    ``target`` is a filesystem path or an open binary file-like / buffer (an
    ``fsspec`` handle, ``io.BytesIO``, ...) --- symmetric with :func:`save_law`.
    Returns a queryable law with no live :class:`~emu_gmm.types.EstimationResult`
    required: a moments-backed :class:`~emu_gmm.law.AsymptoticLaw` (``se`` /
    ``functional_se`` / ``eigenvalue_se`` / ``gamma_se`` / ``sample``) or an
    :class:`~emu_gmm.law.EmpiricalLaw` (records-backed -> ``given`` / ``couple`` /
    ``size_power`` / ``se``; draws-backed -> ``pvalue`` / ``se`` / ``given``).
    Validates the ``schema_version`` and refuses an unknown one.
    """
    state = _read_state(target)
    if state.grade == "asymptotic":
        return _state_to_asymptotic(state)
    if state.grade == "empirical":
        return _state_to_empirical(state)
    raise ValueError(
        f"load_law: unknown grade {state.grade!r} in the artifact manifest."
    )


def load_law_state(target: Any) -> LawState:
    """Read the typed :class:`LawState` from an artifact without rebuilding a law.

    The cheap inspection path: returns the validated record (``grade``,
    ``param_names``, ``diagnostics``, the typed :class:`FactorySpec`, ...) so a
    consumer can read a persisted law's provenance / configuration without
    reconstructing the queryable law. ``target`` is a path or a file-like, as for
    :func:`load_law`.
    """
    return _read_state(target)


__all__ = ["LawState", "FactorySpec", "save_law", "load_law", "load_law_state"]
