r"""Adaptive (precision-targeted) stopping for resampling-based inference (#91).

The resampling helpers in :mod:`emu_gmm.inference` (``cluster_bootstrap``,
``moment_wild_bootstrap``) take a **fixed** ``n_boot``. Choosing it up front is
a guess: too small under-powers the reported functional, too large wastes
expensive refits, and a fixed cap gives **no signal** when it was not enough.

:func:`adaptive_bootstrap` replaces the guess with a *precision-targeted*
stopping rule (Andrews & Buchinsky 2000, *Econometrica* 68(1):23-51): keep
drawing batches of replicates until the reported functional (a CI endpoint, a
bootstrap standard error, or a p-value) has converged to a target Monte Carlo
precision, then stop --- and surface a **loud** ``converged`` flag so that
hitting ``b_max`` *without* converging is itself reported rather than silently
read as "done".

Design
------
The driver is deliberately generic and **host-eager** (a data-dependent
stopping rule cannot be ``vmap``-ed). It wraps any *batched* draw callable:

    draw_batch(key, size) -> Float[Array, " size"]

returning ``size`` scalar replicate values of the statistic of interest
(e.g. ``cluster_bootstrap(..., n_boot=size, key=key).theta_boot.array[:, k]``
for a single coordinate, or ``.J_boot`` for the J-statistic). Non-converged
replicates may be returned as ``NaN``; they are excluded from the functional
but counted in the denominator (``n_invalid``), so a degenerate resampling
world cannot masquerade as precision.

A :class:`_Target` says how to map the pooled finite replicates to
``(value, mcse)`` --- the functional and its Monte Carlo standard error:

- :class:`BootstrapMean` --- mean; ``mcse = s / sqrt(B)``.
- :class:`BootstrapSE` --- standard deviation; ``mcse = s / sqrt(2(B-1))``
  (the normal-theory SE of an estimated SD; the Andrews-Buchinsky target).
- :class:`BootstrapQuantile` --- a CI endpoint; ``mcse`` via the
  **Maritz-Jarrett** (1978) order-statistic estimator, which avoids a fragile
  density estimate at the quantile.
- :class:`BootstrapPValue` --- a bootstrap p-value with the ``(1+count)/(B+1)``
  correction (Davison & Hinkley); ``mcse = sqrt(p~(1-p~)/B)``, the binomial SE
  (the ``+1`` correction also keeps the MCSE strictly positive at an empty
  tail, so ``p=0`` cannot spuriously read as zero-MCSE convergence).

Stopping: with ``z = Phi^{-1}((1+confidence)/2)``, the half-width of the
``confidence``-level Monte Carlo interval for the functional is ``z * mcse``;
the driver stops once ``n_valid >= b_min`` **and**
``z * mcse <= atol + rtol * |value|``. At ``b_max`` drawn replicates it stops
regardless, reporting ``converged=False``.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import Protocol

import jax
import numpy as np
from scipy import stats as _stats

__all__ = [
    "adaptive_bootstrap",
    "AdaptiveBootstrapResult",
    "BootstrapMean",
    "BootstrapSE",
    "BootstrapQuantile",
    "BootstrapPValue",
    "maritz_jarrett_quantile_se",
]


# ---------------------------------------------------------------------------
# Maritz-Jarrett quantile standard error.
# ---------------------------------------------------------------------------
def maritz_jarrett_quantile_se(values: np.ndarray, q: float) -> float:
    r"""Maritz-Jarrett (1978) standard error of the sample ``q``-quantile.

    The sample quantile is an L-estimator --- a weighted sum of order
    statistics --- whose weights are the cell probabilities of the
    ``m``-th order statistic, distributed as :math:`\mathrm{Beta}(m, n-m+1)`
    on the uniform scale, with ``m = clip(floor(qn + 0.5), 1, n)``. Writing
    :math:`W_i = I_{i/n}(m, n-m+1) - I_{(i-1)/n}(m, n-m+1)` for the Beta CDF
    increments, the estimator is
    :math:`\widehat{\mathrm{SE}} = \sqrt{C_2 - C_1^2}` with
    :math:`C_1 = \sum_i W_i x_{(i)}` and :math:`C_2 = \sum_i W_i x_{(i)}^2`.
    Unlike the asymptotic ``sqrt(q(1-q)/n)/f(x_q)`` form, this needs no
    density estimate at the quantile.

    Parameters
    ----------
    values : (B,) float ndarray
        Finite replicate values (NaNs must already be removed).
    q : float in (0, 1)
        Quantile level.

    Returns
    -------
    float
        The Maritz-Jarrett standard-error estimate (``0.0`` for ``n < 2``).
    """
    x = np.sort(np.asarray(values, dtype=np.float64))
    n = x.size
    if n < 2:
        return 0.0
    m = int(np.floor(q * n + 0.5))
    m = min(max(m, 1), n)
    # Beta(m, n-m+1) is the law of the m-th order statistic on the uniform
    # scale; W_i is its mass in the i-th cell [(i-1)/n, i/n].
    edges = np.arange(0, n + 1, dtype=np.float64) / n
    cdf = _stats.beta.cdf(edges, m, n - m + 1)
    w = np.diff(cdf)
    c1 = float(np.sum(w * x))
    c2 = float(np.sum(w * x * x))
    return float(np.sqrt(max(c2 - c1 * c1, 0.0)))


# ---------------------------------------------------------------------------
# Target functionals: pooled finite replicates -> (value, mcse).
# ---------------------------------------------------------------------------
class _Target(Protocol):
    """A bootstrap functional and its Monte Carlo standard error."""

    def evaluate(self, values: np.ndarray) -> tuple[float, float]:
        """Return ``(value, mcse)`` from the finite replicate values."""
        ...

    @property
    def label(self) -> str:
        """A short human-readable name for the functional."""
        ...


@dataclasses.dataclass(frozen=True)
class BootstrapMean:
    """Bootstrap mean; ``mcse = s / sqrt(B)`` (the usual SE of a mean)."""

    @property
    def label(self) -> str:
        return "mean"

    def evaluate(self, values: np.ndarray) -> tuple[float, float]:
        n = values.size
        value = float(np.mean(values))
        s = float(np.std(values, ddof=1)) if n >= 2 else float("inf")
        mcse = s / np.sqrt(n) if n >= 2 else float("inf")
        return value, mcse


@dataclasses.dataclass(frozen=True)
class BootstrapSE:
    """Bootstrap standard error (sample SD, ``ddof=1``).

    ``mcse = s / sqrt(2(B-1))`` --- the normal-theory asymptotic standard
    error of an estimated standard deviation (the Andrews-Buchinsky relative
    target for an SE functional). Approximate for heavy-tailed replicate
    distributions, but the standard choice.
    """

    @property
    def label(self) -> str:
        return "se"

    def evaluate(self, values: np.ndarray) -> tuple[float, float]:
        n = values.size
        if n < 2:
            return float("inf"), float("inf")
        s = float(np.std(values, ddof=1))
        mcse = s / np.sqrt(2.0 * (n - 1))
        return s, mcse


@dataclasses.dataclass(frozen=True)
class BootstrapQuantile:
    """Bootstrap quantile (a CI endpoint); ``mcse`` via Maritz-Jarrett.

    Parameters
    ----------
    q : float in (0, 1)
        Quantile level (e.g. ``0.025`` / ``0.975`` for a 95% percentile CI).
    """

    q: float

    def __post_init__(self) -> None:
        if not (0.0 < self.q < 1.0):
            raise ValueError(f"BootstrapQuantile: q must be in (0, 1), got {self.q}")

    @property
    def label(self) -> str:
        return f"quantile[{self.q:g}]"

    def evaluate(self, values: np.ndarray) -> tuple[float, float]:
        n = values.size
        if n < 2:
            return float("nan"), float("inf")
        # The conventional sample quantile (linear interpolation) is the
        # reported endpoint; Maritz-Jarrett supplies its standard error.
        value = float(np.quantile(values, self.q))
        mcse = maritz_jarrett_quantile_se(values, self.q)
        return value, mcse


@dataclasses.dataclass(frozen=True)
class BootstrapPValue:
    """Bootstrap p-value with the ``(1 + count)/(B + 1)`` correction.

    Parameters
    ----------
    stat_observed : float
        The observed statistic the replicates are compared against.
    tail : {"greater", "less", "two-sided"}, default "greater"
        ``"greater"`` is the upper-tail p-value ``P(boot >= observed)`` ---
        the right choice for a J-statistic over-identification test (large
        ``J`` = reject). ``"two-sided"`` reports ``min(1, 2 min(p_up, p_lo))``.

    Notes
    -----
    ``mcse = sqrt(p~(1-p~)/B)`` is the binomial SE of the proportion, using the
    ``+1``-corrected ``p~`` for both value and MCSE so that an empty tail
    (raw count 0) yields a strictly positive MCSE rather than a spurious
    zero-MCSE "converged".
    """

    stat_observed: float
    tail: str = "greater"

    def __post_init__(self) -> None:
        if self.tail not in ("greater", "less", "two-sided"):
            raise ValueError(
                f"BootstrapPValue: tail must be 'greater', 'less', or "
                f"'two-sided', got {self.tail!r}"
            )

    @property
    def label(self) -> str:
        return f"pvalue[{self.tail}]"

    def evaluate(self, values: np.ndarray) -> tuple[float, float]:
        n = values.size
        if n < 1:
            return float("nan"), float("inf")
        obs = float(self.stat_observed)
        cnt_ge = int(np.sum(values >= obs))
        cnt_le = int(np.sum(values <= obs))
        p_up = (1 + cnt_ge) / (n + 1)
        p_lo = (1 + cnt_le) / (n + 1)
        if self.tail == "greater":
            p = p_up
        elif self.tail == "less":
            p = p_lo
        else:  # two-sided
            p = min(1.0, 2.0 * min(p_up, p_lo))
        p_tilde = min(max(p, 1.0 / (n + 1)), 1.0 - 1.0 / (n + 1))
        mcse = float(np.sqrt(p_tilde * (1.0 - p_tilde) / n))
        return float(p), mcse


# ---------------------------------------------------------------------------
# Result + driver.
# ---------------------------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class AdaptiveBootstrapResult:
    """Output of :func:`adaptive_bootstrap`.

    A plain frozen dataclass (not a pytree): adaptive stopping is inherently
    host-sequential, so the result is a host-side summary rather than a value
    that flows through ``jit`` / ``vmap``.

    Attributes
    ----------
    value : float
        The converged functional (CI endpoint / SE / p-value).
    mcse : float
        Its Monte Carlo standard error at the final ``n_valid``.
    half_width : float
        ``z * mcse``, the half-width of the ``confidence``-level Monte Carlo
        interval --- the quantity compared against ``atol + rtol*|value|``.
    converged : bool
        ``True`` iff the precision target was met before ``b_max``. **Read
        this**: ``False`` means ``b_max`` was hit without converging.
    n_boot : int
        Total replicates drawn (valid + invalid).
    n_valid : int
        Finite replicates used for the functional.
    n_invalid : int
        Non-finite (e.g. non-converged-refit) replicates excluded.
    n_batches : int
        Number of batches drawn.
    confidence : float
        The ``1 - tau`` confidence level used for the half-width.
    target : str
        The functional's label.
    replicates : (n_boot,) float ndarray
        Every drawn replicate, in draw order (including NaNs), so the caller
        can recompute other functionals or continue the stream.
    key : jax.Array
        The PRNG key as consumed (returned to prevent accidental reuse).
    """

    value: float
    mcse: float
    half_width: float
    converged: bool
    n_boot: int
    n_valid: int
    n_invalid: int
    n_batches: int
    confidence: float
    target: str
    replicates: np.ndarray
    key: jax.Array


def adaptive_bootstrap(
    draw_batch: Callable[[jax.Array, int], np.ndarray],
    target: _Target,
    *,
    key: jax.Array,
    batch_size: int = 250,
    b_min: int = 250,
    b_max: int = 20_000,
    atol: float = 0.0,
    rtol: float = 0.0,
    confidence: float = 0.95,
) -> AdaptiveBootstrapResult:
    r"""Draw bootstrap batches until the functional hits a precision target.

    Parameters
    ----------
    draw_batch : callable ``(key, size) -> (size,) array``
        Returns ``size`` scalar replicate values of the statistic of interest.
        Receives a fresh split PRNG key each batch. May return ``NaN`` for
        non-converged replicates (excluded from the functional, counted in
        ``n_invalid``).
    target : :class:`_Target`
        The functional + MCSE estimator: :class:`BootstrapMean`,
        :class:`BootstrapSE`, :class:`BootstrapQuantile`, or
        :class:`BootstrapPValue`.
    key : jax.Array, keyword-only
        PRNG key; split once per batch. Returned (consumed) on the result.
    batch_size : int, default 250
        Replicates drawn per batch.
    b_min : int, default 250
        Minimum *valid* replicates before convergence may be declared.
    b_max : int, default 20000
        Hard cap on total replicates drawn. Reaching it without meeting the
        target yields ``converged=False`` (a loud, publication-relevant
        signal --- never silently truncated).
    atol, rtol : float, default 0.0
        Absolute / relative tolerance on the half-width: stop when
        ``z*mcse <= atol + rtol*|value|``. At least one must be positive.
    confidence : float in (0, 1), default 0.95
        Confidence level ``1 - tau`` for the Monte Carlo half-width
        ``z*mcse`` with ``z = Phi^{-1}((1+confidence)/2)``.

    Returns
    -------
    :class:`AdaptiveBootstrapResult`
    """
    if batch_size < 2:
        raise ValueError(
            f"adaptive_bootstrap: batch_size must be >= 2, got {batch_size}"
        )
    if b_min < 2:
        raise ValueError(f"adaptive_bootstrap: b_min must be >= 2, got {b_min}")
    if b_max < b_min:
        raise ValueError(
            f"adaptive_bootstrap: b_max ({b_max}) must be >= b_min ({b_min})"
        )
    if not (0.0 < confidence < 1.0):
        raise ValueError(
            f"adaptive_bootstrap: confidence must be in (0, 1), got {confidence}"
        )
    if atol < 0.0 or rtol < 0.0 or (atol == 0.0 and rtol == 0.0):
        raise ValueError(
            "adaptive_bootstrap: set at least one positive tolerance "
            f"(atol={atol}, rtol={rtol}); both zero can never converge"
        )

    z = float(_stats.norm.ppf(0.5 * (1.0 + confidence)))

    pooled: list[np.ndarray] = []
    value = float("nan")
    mcse = float("inf")
    half_width = float("inf")
    converged = False
    n_batches = 0
    n_boot = 0
    cur_key = key

    while n_boot < b_max:
        cur_key, sub = jax.random.split(cur_key)
        # Trim the final batch so we never overshoot b_max.
        size = min(batch_size, b_max - n_boot)
        batch = np.asarray(draw_batch(sub, size), dtype=np.float64).ravel()
        pooled.append(batch)
        n_batches += 1
        n_boot += int(batch.size)

        all_vals = np.concatenate(pooled)
        finite = all_vals[np.isfinite(all_vals)]
        n_valid = int(finite.size)
        if n_valid >= b_min:
            value, mcse = target.evaluate(finite)
            half_width = z * mcse
            threshold = atol + rtol * abs(value)
            if np.isfinite(half_width) and half_width <= threshold:
                converged = True
                break

    all_vals = np.concatenate(pooled) if pooled else np.empty(0)
    finite = all_vals[np.isfinite(all_vals)]
    n_valid = int(finite.size)
    # Recompute the functional on the final pool (covers the b_max-without-
    # convergence exit, where the loop may have broken before evaluating).
    if n_valid >= 2:
        value, mcse = target.evaluate(finite)
        half_width = z * mcse

    return AdaptiveBootstrapResult(
        value=float(value),
        mcse=float(mcse),
        half_width=float(half_width),
        converged=converged,
        n_boot=int(n_boot),
        n_valid=n_valid,
        n_invalid=int(n_boot - n_valid),
        n_batches=n_batches,
        confidence=float(confidence),
        target=target.label,
        replicates=all_vals,
        key=cur_key,
    )
