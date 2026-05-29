"""Cluster-wild Rademacher / Mammen bootstrap for the J-statistic.

The cluster-wild bootstrap is the standard cluster-robust inference
object for moment models in which the analytic variance is computed via
:class:`emu_gmm.covariance.clustered.ClusteredCovariance` (households
within villages, observations within survey clusters, time-by-market
cells in panel data). For a single bootstrap replicate ``b`` we draw
one sign :math:`\\eta_c \\in \\{-1, +1\\}` per cluster (Rademacher) or
:math:`\\{-(\\sqrt5 - 1)/2, (\\sqrt5 + 1)/2\\}` per cluster (Mammen),
broadcast the sign to every observation in the cluster, and recompute
the moment vector

.. math::
   m^{*,(b)}_j(\\hat\\theta)
   \\;=\\;
   \\frac{1}{N_j}\\,
   \\sum_{i=1}^N \\eta_{c(i)}\\, d_{ij}\\, w_i\\,
   \\psi_j(x_i, \\hat\\theta).

The bootstrap J-statistic is then

.. math::
   J^{*,(b)}
   \\;=\\;
   \\big\\| L^{-1}\\, m^{*,(b)}(\\hat\\theta) \\big\\|^2,

where :math:`L` is the lower-triangular Cholesky factor of the
analytic variance :math:`V_X(\\hat\\theta)` evaluated at the original
sample (the "refit-free" form). The bootstrap p-value is the empirical
right-tail probability:

.. math::
   p^\\star
   \\;=\\;
   \\frac{1}{B}\\sum_{b=1}^B \\mathbf{1}\\{J^{*,(b)} \\ge J_\\mathrm{obs}\\}.

v1 scope: refit-free. A "full" refit-per-replicate version that
re-estimates :math:`\\theta` on each bootstrap sample and reports a
``theta_boot`` distribution is deferred to v2; the current return
type already carries an optional ``theta_boot`` slot that v1 leaves
as ``None``.

Algorithm notes
---------------

The refit-free form matches the bootstrap target of the analytic
asymptotic distribution. The data-side resampling --- sign-flipping
cluster totals --- captures the cluster-robust variance under the
null that the moment restrictions hold at :math:`\\hat\\theta`. The
fixed-:math:`L` whitening avoids the recompile / refit overhead and
keeps the bootstrap loop vmappable across replicates.

The same V used for the analytic J-test must be passed into the
bootstrap to keep the calibration consistent --- typically obtained
from ``EstimationResult.V_X``. Passing the labelled
:class:`haliax.NamedArray` directly is supported; the helper
auto-unwraps to the underlying array.

JIT / vmap compatibility
------------------------

The returned :class:`WildBootstrapResult` is a
``@jdc.pytree_dataclass`` whose scalar fields (``p_value``,
``J_observed``) are 0-d JAX arrays and whose ``sign`` / ``n_boot``
fields are static (hashable). The helper itself routes through
:func:`jax.scipy.stats.chi2.sf` / :func:`jax.numpy.mean`-free arithmetic
on traced arrays, so the public ``moment_wild_bootstrap`` traces under
``jax.jit`` and composes under ``jax.vmap``. Match: the same
jit / vmap commitment the rest of the framework honours (see
``docs/reviews/v1x-api-design.org`` §1).
"""

from __future__ import annotations

from typing import Any, Literal

import haliax as ha
import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
from jaxtyping import Array, Float

from emu_gmm._internal import cholesky as cho
from emu_gmm.covariance.clustered import ClusteredCovariance
from emu_gmm.measures.empirical import EmpiricalMeasure
from emu_gmm.types import ParamsLike, StructuralModel

# Mammen two-point distribution: Pr(eta = a) = (sqrt5 + 1) / (2 sqrt5);
# Pr(eta = b) = (sqrt5 - 1) / (2 sqrt5); a = -(sqrt5 - 1)/2, b = (sqrt5 + 1)/2.
# These satisfy E[eta] = 0, E[eta^2] = 1, E[eta^3] = 1, which is the
# third-moment correction Mammen (1993) introduced over the symmetric
# Rademacher draw.
_SQRT5 = jnp.sqrt(jnp.asarray(5.0))
_MAMMEN_A = -(_SQRT5 - 1.0) / 2.0
_MAMMEN_B = (_SQRT5 + 1.0) / 2.0
_MAMMEN_PA = (_SQRT5 + 1.0) / (2.0 * _SQRT5)


@jdc.pytree_dataclass
class WildBootstrapResult:
    """Return type for :func:`moment_wild_bootstrap`.

    Pytree-dataclass so the record survives ``jax.jit`` / ``jax.vmap``
    boundaries. Scalar diagnostics are 0-d JAX arrays (traced); the
    sign label and replicate count are static fields (hashable, used
    only for re-tracing on shape / configuration change).

    Attributes
    ----------
    J_boot : (n_boot,) jax array
        The bootstrap J-statistics, one per replicate.
    p_value : 0-d jax array
        Empirical right-tail probability
        ``mean(J_boot >= J_observed)``. 0-d so it traces under ``jit`` /
        ``vmap``; cast with ``float(result.p_value)`` at the eager
        boundary if you want a Python scalar.
    J_observed : 0-d jax array
        The analytic J-statistic at ``theta_hat`` evaluated against the
        same ``V`` used for whitening; included so callers can
        reproduce the p-value calculation and so ``p_value`` is
        self-contained.
    sign : str (static)
        The sign-distribution used: ``"rademacher"`` or ``"mammen"``.
    n_boot : int (static)
        Number of bootstrap replicates.
    theta_boot : None
        Reserved for a future refit-per-replicate variant. Always
        ``None`` in v1; the field is kept on the dataclass so the v2
        addition is a non-breaking change.
    """

    J_boot: Float[Array, " B"]
    p_value: Float[Array, ""]
    J_observed: Float[Array, ""]
    sign: str = jdc.static_field()  # type: ignore[attr-defined]
    n_boot: int = jdc.static_field()  # type: ignore[attr-defined]
    theta_boot: Any = jdc.static_field(default=None)  # type: ignore[attr-defined]


def _to_plain(value: Any) -> Float[Array, "..."]:
    """Strip a :class:`haliax.NamedArray` wrapper if present.

    Mirrors the same helper in the covariance subpackage. The wild-
    bootstrap docstring guides callers to pass
    ``EstimationResult.V_X`` directly; ``V_X`` is a NamedArray, so
    without this unwrap ``jnp.asarray(V_X)`` would raise. Plain JAX
    arrays pass through unchanged.
    """
    if isinstance(value, ha.NamedArray):
        return jnp.asarray(value.array)
    return jnp.asarray(value)


def _draw_rademacher(key: jax.Array, n_clusters: int) -> Float[Array, " C"]:
    """Draw ``n_clusters`` independent Rademacher signs in ``{-1, +1}``."""
    u = jax.random.bernoulli(key, p=0.5, shape=(n_clusters,))
    return jnp.where(u, 1.0, -1.0)


def _draw_mammen(key: jax.Array, n_clusters: int) -> Float[Array, " C"]:
    """Draw ``n_clusters`` Mammen two-point signs.

    Distribution: ``Pr(eta = -(sqrt5 - 1)/2) = (sqrt5 + 1) / (2 sqrt5)``,
    ``Pr(eta = (sqrt5 + 1)/2) = (sqrt5 - 1) / (2 sqrt5)``. Satisfies
    ``E[eta] = 0``, ``E[eta^2] = 1``, ``E[eta^3] = 1`` --- the third-
    moment correction over Rademacher.
    """
    u = jax.random.bernoulli(key, p=_MAMMEN_PA, shape=(n_clusters,))
    return jnp.where(u, _MAMMEN_A, _MAMMEN_B)


def _per_obs_signs(
    eta_c: Float[Array, " C"],
    cluster_ids: Float[Array, " N"],
) -> Float[Array, " N"]:
    """Broadcast cluster-level signs to per-observation signs.

    ``eta_i = eta_{c(i)}``. The ``cluster_ids`` argument carries a
    float dtype to match :class:`ClusteredCovariance`; the index gather
    casts to int32.
    """
    return eta_c[cluster_ids.astype(jnp.int32)]


def _bootstrap_moment(
    contributions: Float[Array, "N M"],
    weight_mask: Float[Array, "N M"],
    eta_i: Float[Array, " N"],
) -> Float[Array, " M"]:
    """Compute one bootstrap moment vector.

    ``m^*_j = (sum_i eta_i * g_ij) / (sum_i d_ij * w_i)`` where
    ``g_ij = d_ij * w_i * psi_j(x_i, theta_hat)`` is the per-observation
    moment contribution from
    :meth:`EmpiricalMeasure.moment_contributions`. The denominator
    ``N_j = sum_i d_ij * w_i`` is /unchanged/ by the sign flip --- the
    bootstrap perturbs the numerator (the cluster totals) and keeps the
    per-coordinate normalisation fixed at the analytic value.

    Degenerate coordinates (``N_j = 0``) map to zero rather than NaN,
    matching :meth:`EmpiricalMeasure.expectation`.
    """
    numer = jnp.sum(eta_i[:, None] * contributions, axis=0)  # (M,)
    N_j = jnp.sum(weight_mask, axis=0)  # (M,)
    safe = jnp.where(N_j == 0.0, 1.0, N_j)
    out = numer / safe
    return jnp.where(N_j == 0.0, jnp.zeros_like(out), out)


def moment_wild_bootstrap(
    model: StructuralModel,
    theta_hat: ParamsLike,
    measure: EmpiricalMeasure,
    covariance: ClusteredCovariance,
    *,
    n_boot: int,
    key: jax.Array,
    sign: Literal["rademacher", "mammen"] = "rademacher",
    V: Float[Array, "M M"] | ha.NamedArray | None = None,
) -> WildBootstrapResult:
    """Cluster-wild bootstrap of the J-statistic (refit-free).

    Parameters
    ----------
    model : :data:`~emu_gmm.types.StructuralModel`
        Per-observation residual function ``psi(x, theta) -> (M,)``.
    theta_hat : :data:`~emu_gmm.types.ParamsLike`
        Estimated parameters. The bootstrap evaluates ``psi`` and the
        Cholesky factor of ``V`` at this point; in the v1 refit-free
        form, ``theta`` is /not/ re-estimated per replicate.
    measure : :class:`~emu_gmm.measures.empirical.EmpiricalMeasure`
        The sample-backed measure that ``estimate()`` consumed.
    covariance : :class:`~emu_gmm.covariance.clustered.ClusteredCovariance`
        The cluster covariance strategy that ``estimate()`` consumed.
        The bootstrap reads ``cluster_ids`` and ``n_clusters`` off this
        object; consistency with the original fit is the caller's
        responsibility.
    n_boot : int, keyword-only
        Number of bootstrap replicates. Each replicate produces one
        ``J^*`` value. Static (treated as a shape parameter under
        ``jit``); changing ``n_boot`` triggers a re-trace.
    key : :class:`jax.Array`, keyword-only
        PRNG key. Split internally to draw the per-replicate signs.
    sign : ``"rademacher"`` or ``"mammen"``, default ``"rademacher"``
        Sign distribution. Rademacher is the v1 default and matches the
        ManifoldGMM reference; Mammen gives a third-moment correction
        (``E[eta^3] = 1``) that is sometimes preferred for asymmetric
        residuals.
    V : (M, M) jax array or :class:`haliax.NamedArray`, optional, keyword-only
        The (regularised) variance matrix at ``theta_hat`` to whiten
        the bootstrap moments. When omitted the function recomputes it
        by calling ``covariance.covariance(model, theta_hat, measure)``;
        callers who already have ``EstimationResult.V_X`` should pass
        it directly (either the NamedArray or its ``.array``) to avoid
        the extra evaluation and to guarantee the Cholesky factor
        matches the one used by the analytic J-test. The helper
        auto-unwraps a :class:`haliax.NamedArray` to its underlying
        array.

    Returns
    -------
    :class:`WildBootstrapResult`
        Pytree-dataclass with traced scalar fields; survives ``jit`` /
        ``vmap`` and composes with the rest of the framework's pytree
        protocols.

    Notes
    -----
    The refit-free form fixes :math:`L = \\mathrm{chol}(V_X(\\hat\\theta))`
    across all replicates and bootstrap-perturbs only the moment
    vector. This matches the bootstrap target of the analytic
    asymptotic distribution and avoids re-running the GMM optimiser
    inside the bootstrap loop. A "full" refit-per-replicate variant is
    deferred to v2 (the ``theta_boot`` slot in
    :class:`WildBootstrapResult` is reserved for it).

    The bootstrap assumes the moment contributions are exchangeable
    /within/ each cluster and independent /across/ clusters. The
    cluster IDs are read off ``covariance``; the same array used for
    the analytic ``ClusteredCovariance`` is the right choice here.

    JIT / vmap behaviour: the scalar diagnostics (``p_value``,
    ``J_observed``) are 0-d JAX arrays rather than Python floats, so
    the helper traces under ``jax.jit`` and composes under
    ``jax.vmap``. Cast with ``float(...)`` at the eager boundary if a
    Python scalar is needed.
    """
    if sign not in ("rademacher", "mammen"):
        raise ValueError(
            f"moment_wild_bootstrap: sign must be 'rademacher' or 'mammen', "
            f"got {sign!r}"
        )
    if n_boot <= 0:
        raise ValueError(
            f"moment_wild_bootstrap: n_boot must be positive, got {n_boot}"
        )

    # Per-observation moment contributions g_ij = d_ij * w_i * psi_j(x_i, theta_hat).
    contributions = measure.moment_contributions(model, theta_hat)  # (N, M)
    weight_mask = measure.mask * measure.weights[:, None]  # (N, M)

    # Variance at theta_hat. Caller-supplied V wins to guarantee a match
    # with the regularised analytic V the EstimationResult exposes.
    # Auto-unwrap a haliax NamedArray (the natural ``result.V_X`` hand-off)
    # rather than letting jnp.asarray choke on the wrapper object.
    if V is None:
        V_arr = _to_plain(covariance.covariance(model, theta_hat, measure))
    else:
        V_arr = _to_plain(V)
    L = cho.cholesky(V_arr)  # (M, M) lower-triangular

    # Analytic J at theta_hat against the same V.
    m_hat = measure.expectation(model, theta_hat)
    y_hat = cho.forward_solve(L, _to_plain(m_hat))
    J_observed_arr = jnp.sum(y_hat * y_hat)

    cluster_ids = covariance.cluster_ids
    n_clusters = int(covariance.n_clusters)

    keys = jax.random.split(key, n_boot)

    if sign == "rademacher":
        draw_fn = _draw_rademacher
    else:
        draw_fn = _draw_mammen

    def one_replicate(k: jax.Array) -> Float[Array, ""]:
        eta_c = draw_fn(k, n_clusters)  # (n_clusters,)
        eta_i = _per_obs_signs(eta_c, cluster_ids)  # (N,)
        m_boot = _bootstrap_moment(contributions, weight_mask, eta_i)  # (M,)
        y_boot = cho.forward_solve(L, m_boot)
        return jnp.sum(y_boot * y_boot)

    J_boot = jax.vmap(one_replicate)(keys)  # (n_boot,)

    # Keep p_value and J_observed as 0-d traced arrays so the helper
    # traces cleanly under jit / vmap. Eager callers cast via float()
    # at the boundary.
    p_value_arr = jnp.mean((J_boot >= J_observed_arr).astype(J_boot.dtype))

    return WildBootstrapResult(
        J_boot=J_boot,
        p_value=p_value_arr,
        J_observed=J_observed_arr,
        sign=sign,
        n_boot=int(n_boot),
        theta_boot=None,
    )


__all__ = ["moment_wild_bootstrap", "WildBootstrapResult"]
