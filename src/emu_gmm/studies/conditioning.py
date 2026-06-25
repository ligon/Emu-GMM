"""Conditional and coupled empirical-law queries over MC records (#167).

First, minimum-viable increment of the ``EstimatorLaw`` design (#144): the
two genuinely *new* semantics that the #130 validation harness hand-rolled,
provided additively on top of the stacked records that
:func:`emu_gmm.studies.replicate` returns --- without (yet) the full
``EstimatorLaw`` interface.

Two queries:

* :func:`given` --- the **conditional** law: select the sub-population of
  replicates where a diagnostic event holds (``binding_ridge``, ...), as a
  masked :class:`~emu_gmm.types.FitRecord` that every layer-2 summarizer
  already consumes. :func:`event_share` reports the selection's size loudly
  (both the all-reps and the converged-reps denominators).
* :func:`crn_pair` --- the **coupled** law: verify two Monte Carlo arms share
  a probability space (Common Random Numbers) *before* zipping them, then
  expose paired contrasts (:meth:`CoupledRecords.flips`,
  :meth:`CoupledRecords.paired_diff`). The verification refuses, it never
  approximates: master-key equality is necessary but **not sufficient**
  (it cannot witness the DGP's internal split scheme), so a
  :attr:`~emu_gmm.studies.MCRecords.coupling_id` is required unless the
  caller explicitly asserts coupling.

Design notes (the red-team dispositions for #167):

* ``given`` returns a bare masked ``FitRecord`` so it composes with the
  existing summarizers with *zero* change to ``summaries.py``; the
  conditioning provenance rides on the separate :func:`event_share` so a
  conditional answer is never silently dressed as a marginal one.
* Conditioning on an *estimator-internal* event (``binding_ridge``,
  ``sigma_meat_indefinite``, ``converged``) is **selection-conditional, not
  nominal**. ``coverage(given(rec, "binding_ridge"), theta0)`` answers "among
  the reps where the ridge bound, how often did the CI cover" --- a
  within-selection *diagnostic*, NOT a coverage guarantee: the event is a
  function of the same data as ``theta_hat``/``se``. The blessed use is a
  within-subset *contrast* (nominal vs adjusted p-value calibration), not a
  coverage/size claim read off the subset.
* Masking destroys the rep-index alignment CRN relies on, so a conditioned
  record can never be paired. Pair first (:func:`crn_pair`), then condition a
  paired contrast via ``flips(..., where=event_mask)``.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import Any, NamedTuple

import jax
import numpy as np

from emu_gmm.studies.driver import MCRecords
from emu_gmm.types import FitRecord

#: The 0/1 diagnostic flag fields ``given``/``event_share`` accept by name.
#: Each is an exact-binary float on the records (``record()`` casts a bool to
#: float64). For any other condition, pass a predicate callable.
FLAG_FIELDS: tuple[str, ...] = (
    "converged",
    "binding_ridge",
    "sigma_meat_indefinite",
)


def _unwrap(records: MCRecords | FitRecord) -> FitRecord:
    """Return the stacked :class:`FitRecord` behind an arg (mirrors ``_stacked``)."""
    if isinstance(records, MCRecords):
        return records.records
    if isinstance(records, FitRecord):
        return records
    raise TypeError(
        "given()/event_share() expect an MCRecords or a stacked FitRecord; "
        f"got {type(records).__name__}."
    )


def _n_reps(rec: FitRecord) -> int:
    return int(np.asarray(rec.converged).shape[0])


def _event_mask(
    rec: FitRecord,
    event: str | Callable[[FitRecord], Any],
    negate: bool,
) -> np.ndarray:
    """Build the ``(n_reps,)`` boolean selection mask.

    String form: a 0/1 flag in :data:`FLAG_FIELDS`, thresholded at ``> 0.5``
    (the package convention --- ``summaries._used`` and
    ``MCRecords.converged_mask`` both use it; a NaN flag yields ``False`` =
    "event absent"). Callable form: a predicate returning a ``(n_reps,)``
    boolean array. ``negate`` is applied as ``~mask`` on the *single* computed
    mask, so the ``given(E)`` / ``given(~E)`` partition is exact even under a
    NaN flag.
    """
    if isinstance(event, str):
        if event not in FLAG_FIELDS:
            raise ValueError(
                f"given(): unknown flag {event!r}. The string form selects a "
                f"0/1 diagnostic flag, one of {FLAG_FIELDS}. For a general "
                "condition, pass a predicate FitRecord -> bool array."
            )
        mask = np.asarray(getattr(rec, event)) > 0.5
    elif callable(event):
        raw = np.asarray(event(rec))
        if raw.dtype != np.bool_:
            raise TypeError(
                "given(): a predicate must return a boolean array; got dtype "
                f"{raw.dtype}. (Threshold floats yourself so the selection is "
                "explicit.)"
            )
        if raw.shape != (_n_reps(rec),):
            raise ValueError(
                "given(): a predicate must return a (n_reps,) boolean array; "
                f"got shape {raw.shape} for n_reps={_n_reps(rec)}."
            )
        mask = raw
    else:
        raise TypeError(
            "given(): event must be a flag-name str or a predicate callable, "
            f"got {type(event).__name__}."
        )
    return ~mask if negate else mask


def given(
    records: MCRecords | FitRecord,
    event: str | Callable[[FitRecord], Any],
    *,
    negate: bool = False,
) -> FitRecord:
    """The conditional law: the sub-record where ``event`` holds.

    Returns a stacked :class:`~emu_gmm.types.FitRecord` masked to the selected
    replicates (static ``J_dof`` / ``param_names`` preserved). It composes with
    every layer-2 summarizer directly, e.g.
    ``coverage(given(rec, "binding_ridge"), theta0)`` --- which the
    summarizers further restrict to *converged* reps, so the result is the
    coverage among ``binding & converged`` reps.

    **This is selection-conditional, not nominal.** When ``event`` is an
    estimator-internal flag the selection is correlated with ``theta_hat`` and
    ``se``; read the result as a within-selection diagnostic, and pair it with
    :func:`event_share` so the conditioning is never lost. See the module
    docstring.

    Parameters
    ----------
    records
        An :class:`~emu_gmm.studies.MCRecords` or a bare stacked
        :class:`~emu_gmm.types.FitRecord`.
    event
        A flag name in :data:`FLAG_FIELDS`, or a predicate
        ``FitRecord -> (n_reps,) bool``.
    negate
        Select the complement (``~mask``).
    """
    rec = _unwrap(records)
    mask = _event_mask(rec, event, negate)
    return jax.tree_util.tree_map(lambda leaf: leaf[mask], rec)


class EventShare(NamedTuple):
    """How big a :func:`given` selection is --- with *both* denominators.

    Every summarizer *rate* (coverage, size, binding frequency) is over
    *converged* reps; ``fraction_all`` is over *all* reps. Reporting both
    stops a silent denominator mismatch when a consumer prints an event share
    next to a converged-rep rate.
    """

    event: str
    negate: bool
    n_selected: int
    n_total: int
    n_selected_converged: int
    n_total_converged: int
    fraction_all: float
    fraction_converged: float


def _event_label(event: str | Callable[[FitRecord], Any], negate: bool) -> str:
    base = (
        event if isinstance(event, str) else getattr(event, "__name__", "<predicate>")
    )
    return f"~{base}" if negate else base


def event_share(
    records: MCRecords | FitRecord,
    event: str | Callable[[FitRecord], Any],
    *,
    negate: bool = False,
) -> EventShare:
    """The loud size of a :func:`given` selection (see :class:`EventShare`)."""
    rec = _unwrap(records)
    mask = _event_mask(rec, event, negate)
    conv = np.asarray(rec.converged) > 0.5
    n_total = int(mask.size)
    n_selected = int(mask.sum())
    n_total_converged = int(conv.sum())
    n_selected_converged = int((mask & conv).sum())
    return EventShare(
        event=_event_label(event, negate),
        negate=negate,
        n_selected=n_selected,
        n_total=n_total,
        n_selected_converged=n_selected_converged,
        n_total_converged=n_total_converged,
        fraction_all=(n_selected / n_total) if n_total else float("nan"),
        fraction_converged=(
            (n_selected_converged / n_total_converged)
            if n_total_converged
            else float("nan")
        ),
    )


class Flips(NamedTuple):
    """Directional flip counts over a paired indicator.

    ``gain`` / ``lose`` are *direction-only* ("arm B has it, arm A doesn't" and
    vice versa); they carry no value judgement (gaining a coverage event is
    good, gaining a binding event is bad --- the caller knows which).
    ``n_both`` is the number of reps with a valid verdict in both arms.
    """

    gain: int
    lose: int
    n_both: int


@dataclasses.dataclass(frozen=True)
class CoupledRecords:
    """Two CRN-verified arms, aligned rep-for-rep (see :func:`crn_pair`)."""

    a: FitRecord
    b: FitRecord
    key: Any
    n_reps: int
    param_names: tuple[str, ...]
    coupling_id: Any

    def both_finite(self, arr_a: Any, arr_b: Any) -> np.ndarray:
        """Per-rep mask: finite in both arms (NaN = invalid verdict)."""
        return np.isfinite(np.asarray(arr_a)) & np.isfinite(np.asarray(arr_b))

    def flips(
        self,
        ind_a: Any,
        ind_b: Any,
        *,
        where: Any = None,
    ) -> Flips:
        """Directional flips of a 0/1 indicator from arm A to arm B.

        Over ``where`` (default: ``both_finite(ind_a, ind_b)``),
        ``gain = #{b==1 & a==0}``, ``lose = #{b==0 & a==1}``.
        """
        a = np.asarray(ind_a)
        b = np.asarray(ind_b)
        w = self.both_finite(a, b) if where is None else np.asarray(where)
        gain = int(np.sum((b == 1) & (a == 0) & w))
        lose = int(np.sum((b == 0) & (a == 1) & w))
        return Flips(gain=gain, lose=lose, n_both=int(np.sum(w)))

    def paired_diff(
        self,
        x_a: Any,
        x_b: Any,
        *,
        where: Any,
    ) -> np.ndarray:
        """The paired difference ``(x_b - x_a)`` over ``where``.

        ``where`` is **mandatory** by design: the contrast subpopulation is the
        caller's choice and must be explicit. Defaulting it to the arguments'
        own finiteness would silently disagree with :meth:`flips`'s
        denominator (e.g. ``J_stat`` is finite where the coverage indicator is
        not), so the printed ``n`` would not match the diff's ``n``.
        """
        w = np.asarray(where)
        return (np.asarray(x_b) - np.asarray(x_a))[w]

    def mean_paired_diff(self, x_a: Any, x_b: Any, *, where: Any) -> float:
        """Mean paired difference over ``where``; NaN (no warning) if empty."""
        d = self.paired_diff(x_a, x_b, where=where)
        finite = d[np.isfinite(d)]
        return float(finite.mean()) if finite.size else float("nan")


def crn_pair(
    a: MCRecords,
    b: MCRecords,
    *,
    assert_coupled: bool = False,
) -> CoupledRecords:
    """Verify two MC arms share a CRN probability space, then couple them.

    The arms are CRN-coupled iff replicate ``r`` of each drew the *same* data
    --- which needs the same master key **and** the same DGP/fold-in scheme.
    ``crn_pair`` refuses (raises) unless it can witness that:

    * ``n_reps``, ``param_names``, and the master ``key`` must match (necessary
      conditions; a mismatch is a hard refusal);
    * the :attr:`~emu_gmm.studies.MCRecords.coupling_id` must match. Master-key
      equality alone is necessary but **not sufficient** --- it does not see
      the DGP's internal ``split`` scheme, so two arms with the same seed but
      different DGPs (e.g. a 4-moment vs a 5-moment draw) would share a key yet
      draw different data. If either arm carries no ``coupling_id`` the
      coupling is unverifiable and ``crn_pair`` refuses unless you pass
      ``assert_coupled=True`` to assert same-DGP on your own authority.

    A conditioned/masked record can never be paired (masking destroys the
    rep-index alignment); pass whole :class:`~emu_gmm.studies.MCRecords` arms
    and condition a paired contrast afterwards via ``flips(..., where=...)``.
    """
    if not isinstance(a, MCRecords) or not isinstance(b, MCRecords):
        raise TypeError(
            "crn_pair() requires two whole MCRecords arms. A conditioned/masked "
            "record cannot be paired (masking destroys rep-index alignment); "
            "pair first, then condition a paired contrast with "
            "flips(..., where=event_mask)."
        )
    # n_reps before key: np.array_equal on shape-mismatched keys returns False,
    # which would mis-report a length difference as a key difference.
    if a.n_reps != b.n_reps:
        raise ValueError(
            f"crn_pair(): arms have different n_reps ({a.n_reps} vs {b.n_reps}); "
            "they are not coupled."
        )
    if tuple(a.param_names) != tuple(b.param_names):
        raise ValueError(
            "crn_pair(): arms have different param_names "
            f"({tuple(a.param_names)} vs {tuple(b.param_names)}); not the same "
            "estimand."
        )
    a_id, b_id = a.coupling_id, b.coupling_id
    if a_id is not None and b_id is not None:
        if a_id != b_id:
            raise ValueError(
                f"crn_pair(): coupling_id mismatch ({a_id!r} vs {b_id!r}); the "
                "arms were not drawn from the same DGP/CRN stream."
            )
    elif not assert_coupled:
        raise ValueError(
            "crn_pair(): cannot verify CRN coupling -- one or both arms carry no "
            "coupling_id, and master-key equality alone is necessary but NOT "
            "sufficient (it does not witness the DGP's split scheme). Rebuild "
            "the arms with replicate(..., coupling_id=...) / "
            "monte_carlo_study(..., coupling_id=...), or pass assert_coupled=True "
            "to assert same-DGP on your own authority."
        )
    if not np.array_equal(np.asarray(a.key), np.asarray(b.key)):
        raise ValueError(
            "crn_pair(): master PRNG keys differ; the arms cannot be CRN-aligned."
        )
    return CoupledRecords(
        a=a.records,
        b=b.records,
        key=a.key,
        n_reps=a.n_reps,
        param_names=tuple(a.param_names),
        coupling_id=a_id,
    )


__all__ = [
    "FLAG_FIELDS",
    "given",
    "event_share",
    "EventShare",
    "crn_pair",
    "CoupledRecords",
    "Flips",
]
