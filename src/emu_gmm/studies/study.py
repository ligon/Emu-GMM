"""Layer 3 of the Monte Carlo driver (#114): convenience composition.

:func:`monte_carlo_study` is sugar **by delegation** --- the ``Context``
precedent (design.org Section 2): it calls :func:`replicate` and the
layer-2 summarizers and packages their outputs, computing nothing
itself. Callers needing a diagnostic it doesn't surface should run the
summarizer (or a new one) over ``StudyResult.records`` directly.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import Any

import jax

from emu_gmm.studies.driver import MCRecords, replicate
from emu_gmm.studies.summaries import (
    BiasSD,
    Coverage,
    JCalibration,
    SizePower,
    TauBinding,
    bias_sd,
    coverage,
    j_calibration,
    size_power,
    tau_binding,
)
from emu_gmm.types import EstimationResult, Measure


@dataclasses.dataclass(frozen=True)
class StudyResult:
    """A Monte Carlo study's records plus the standard summaries.

    A plain host-side dataclass (not a pytree): the traced-world object
    is ``records`` (an :class:`~emu_gmm.studies.MCRecords` pytree); the
    summaries are eager numpy reductions. Everything here is computable
    from ``records`` --- this type is a convenience view, never a third
    primitive.
    """

    records: MCRecords
    bias_sd: BiasSD
    coverage: Coverage
    size_power: SizePower
    tau_binding: TauBinding
    j_calibration: JCalibration

    @property
    def n_reps(self) -> int:
        return self.records.n_reps

    @property
    def n_used(self) -> int:
        return self.records.n_converged

    @property
    def n_excluded(self) -> int:
        return self.records.n_excluded


def monte_carlo_study(
    run: Callable[[Any, Measure], EstimationResult],
    dgp: Callable[[jax.Array], Measure],
    *,
    n_reps: int,
    key: jax.Array,
    theta_init: Any,
    theta0: Any,
    level: float = 0.95,
    alpha: tuple[float, ...] = (0.01, 0.05, 0.10),
    anchor_per_rep: bool = False,
    coupling_id: Any = None,
) -> StudyResult:
    """Run a study and compute the standard summary battery.

    Pure composition of :func:`replicate` (layer 1) and the layer-2
    summarizers; see their docstrings for the CRN scheme, the
    exclude-but-count convention, and each summary's semantics.

    Parameters
    ----------
    run, dgp, n_reps, key, theta_init, anchor_per_rep
        Passed through to :func:`replicate`. Set
        ``anchor_per_rep=True`` (slow path; requires ``run`` to be a
        :func:`~emu_gmm.build_estimator` factory) whenever the
        regularisation anchor matters --- on the default factory path
        every replicate inherits the template measure's frozen
        ``tau_anchor``, so the :func:`~emu_gmm.studies.tau_binding`
        column degenerates to a constant 0/1 per study (#142). See
        :func:`replicate` for the full contract, cost, and the per-rep
        ``jax.clear_caches()`` memory note.
    theta0
        The DGP truth, as a length-D array or the user's parameter
        pytree; used by :func:`bias_sd` and :func:`coverage`. (Distinct
        from ``theta_init``, the optimiser's starting point.)
    level
        Wald CI level for :func:`coverage`.
    alpha
        Rejection levels for :func:`size_power`.
    """
    records = replicate(
        run,
        dgp,
        n_reps=n_reps,
        key=key,
        theta_init=theta_init,
        anchor_per_rep=anchor_per_rep,
        coupling_id=coupling_id,
    )
    return StudyResult(
        records=records,
        bias_sd=bias_sd(records, theta0),
        coverage=coverage(records, theta0, level=level),
        size_power=size_power(records, alpha=alpha),
        tau_binding=tau_binding(records),
        j_calibration=j_calibration(records),
    )


__all__ = ["StudyResult", "monte_carlo_study"]
