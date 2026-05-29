"""Zero-parameter J-test of over-identifying restrictions.

Given a ``model``, a ``measure``, a covariance strategy, and a
hypothesised parameter value ``theta_null``, evaluate

.. math::
   J \\;=\\; m_\\mu(\\theta_\\mathrm{null})^\\top
          V_\\mu(\\theta_\\mathrm{null})^{-1}
          m_\\mu(\\theta_\\mathrm{null})
          \\;\\sim\\;\\chi^2_M

under the null that the moment conditions hold at ``theta_null``. No
parameters are estimated; degrees of freedom are the full moment count
``M`` (contrast :class:`emu_gmm.EstimationResult`, where the estimator
has spent ``K`` dof on minimisation and reports ``J_dof = M - K``).

This is the helper called for by K-Aggregators's
``cross_moment_test_via_emu_gmm`` --- and more generally by any user
who already knows their parameter (or has none to estimate) and only
wants the over-identifying-restrictions test.

See ``docs/design.org`` Section 5 ("J-statistic") for the algorithmic
context and :class:`emu_gmm.estimator.estimate` for the closely related
post-estimation J-stat path that this helper deliberately bypasses.
"""

from __future__ import annotations

import haliax as ha
import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
from jaxtyping import Array, Float

from emu_gmm._internal import axes as axes_mod
from emu_gmm._internal import cholesky as cho
from emu_gmm._internal import labels as labels_mod
from emu_gmm.regularization import DiagonalTikhonov
from emu_gmm.types import (
    CovarianceStrategy,
    Measure,
    ParamsLike,
    RegularizationStrategy,
    StructuralModel,
)


@jdc.pytree_dataclass
class JTestResult:
    """Outcome of a zero-parameter J-test.

    A :func:`jax_dataclasses.pytree_dataclass` so the record flows
    through ``jit`` and ``vmap`` boundaries unchanged --- the array
    fields are traced leaves and ``J_dof`` is a static
    :func:`jax_dataclasses.static_field`. This matches
    :class:`emu_gmm.inference.k_statistic.KStatisticResult` and lets the
    helper compose with batched-null gestures (e.g. evaluating the J
    statistic over a panel of ``theta_null`` values via
    :func:`jax.vmap`).

    Attributes
    ----------
    J_stat : :class:`jax.Array`
        The realised :math:`J = m^\\top V^{-1} m`, computed via the
        whitened-residual form ``||L^{-1} m||^2`` (no explicit inverse).
        0-d JAX array; call ``float(result.J_stat)`` to materialise a
        Python scalar at the eager boundary.
    J_dof : int (static)
        Degrees of freedom, equal to the moment count :math:`M`. No
        parameters are estimated here, so the dof penalty that appears
        in :class:`emu_gmm.EstimationResult` (``J_dof = M - K``) is
        absent.
    J_pvalue : :class:`jax.Array`
        Tail probability ``jax.scipy.stats.chi2.sf(J_stat, J_dof)``.
        0-d JAX array; ``float(result.J_pvalue)`` at the eager boundary.
    V_X : :class:`haliax.NamedArray`
        Labelled :math:`(M, M)` regularised variance of the moment
        estimator at ``theta_null``. Axes are
        ``(moments, moments_dual)``. Returned for inspection /
        diagnostic plotting; pair with ``moment_names`` if you want
        named indices in a pandas frame.

    Notes
    -----
    Unlike :class:`emu_gmm.EstimationResult`, this record carries no
    ``Sigma_theta``, no ``theta_hat``, and no convergence diagnostics ---
    those concepts simply don't apply when no parameter is estimated.
    """

    J_stat: Float[Array, ""]
    J_pvalue: Float[Array, ""]
    V_X: ha.NamedArray
    J_dof: int = jdc.static_field()  # type: ignore[attr-defined]


def j_test(
    measure: Measure,
    covariance: CovarianceStrategy,
    model: StructuralModel,
    theta_null: ParamsLike,
    *,
    regularization: RegularizationStrategy | None = None,
) -> JTestResult:
    """Run a zero-parameter J-test at ``theta_null``.

    Evaluates :math:`m = \\mathbb{E}_\\mu[\\psi(\\cdot,\\theta_\\mathrm{null})]`
    and :math:`V = V_\\mu(\\theta_\\mathrm{null})`, applies the supplied
    regulariser to ``V`` (Cholesky factor ``L`` of the regularised
    matrix), and returns ``J = ||L^{-1} m||^2`` with the
    :math:`\\chi^2_M` p-value.

    The helper is jit/vmap compatible: ``J_stat`` and ``J_pvalue`` are
    returned as 0-d JAX arrays (call ``float(...)`` at the eager
    boundary), and ``J_dof`` is a static Python integer. The returned
    :class:`JTestResult` is a :func:`jax_dataclasses.pytree_dataclass`
    so it survives transformation boundaries unchanged.

    Parameters
    ----------
    measure : :class:`emu_gmm.Measure`
        Integration operator (synthetic / empirical / analytical).
    covariance : :class:`emu_gmm.CovarianceStrategy`
        Constructor for :math:`V_\\mu(\\theta)`. The user is responsible
        for pairing ``measure`` with a compatible covariance strategy
        (e.g. :class:`SyntheticMeasure` + :class:`SyntheticCovariance`).
    model : :data:`emu_gmm.StructuralModel`
        Per-observation residual ``psi(x, theta)``.
    theta_null
        The hypothesised parameter value at which the moment
        restrictions are tested. May be a ``@jdc.pytree_dataclass`` or
        any structure the chosen ``measure`` and ``model`` accept.
    regularization : :class:`emu_gmm.RegularizationStrategy`, optional
        PD-restoration strategy applied to :math:`V`. Defaults to
        :class:`emu_gmm.DiagonalTikhonov` with framework defaults; pass
        a configured instance to override.

    Returns
    -------
    :class:`JTestResult`

    Notes
    -----
    The whitened form ``J = ||L^{-1} m||^2`` is mathematically
    equivalent to ``m' V^{-1} m`` but avoids forming :math:`V^{-1}`
    explicitly --- the same Cholesky-based computation used inside
    :func:`emu_gmm.estimate`.

    The helper independently re-anchors ``tau`` at ``theta_null``; when
    paired with a fitted :class:`emu_gmm.EstimationResult` via
    ``j_test(..., theta_null=result.theta_hat)`` and the ridge is
    binding, the resulting ``J_stat`` will not in general equal
    ``result.J_stat`` (which used the first-stage-anchored ``tau``).
    """
    if regularization is None:
        regularization = DiagonalTikhonov()

    # Evaluate moments and their variance at theta_null.
    m = jnp.asarray(measure.expectation(model, theta_null))
    V = jnp.asarray(covariance.covariance(model, theta_null, measure))

    # Regularise V (anchor-once: there is no iterative loop here, so
    # there is no smoothness concern --- tau is set once at theta_null).
    V_star, _tau = regularization.apply(V)

    # Whiten and form the scalar statistic. ``J_stat`` stays as a 0-d
    # JAX array so the helper composes with ``jit`` / ``vmap``.
    L = cho.cholesky(V_star)
    y = cho.forward_solve(L, m)
    J_stat = jnp.sum(y * y)

    # ``M`` is a static shape, so ``J_dof`` is a static Python int even
    # under tracing.
    M = int(m.shape[0])
    J_dof = M

    # ``jax.scipy.stats.chi2.sf`` is traceable (cf. estimator.py:290);
    # ``scipy.stats.chi2.sf`` is not. ``J_dof`` is a static int here
    # rather than a tracer.
    J_pvalue = jax.scipy.stats.chi2.sf(J_stat, J_dof)

    # Labelled V_X. We give positional moment names here: the
    # zero-parameter helper has no theta_init to probe for a labelled
    # model return, and adding a moment_names kwarg now would not match
    # how callers (notably K-Aggregators) need to use the result.
    Moments = axes_mod.moments_axis(M)
    MomentsDual = axes_mod.moments_dual_axis(M)
    V_X = labels_mod.label_matrix(V_star, Moments, MomentsDual)

    return JTestResult(
        J_stat=J_stat,
        J_dof=J_dof,
        J_pvalue=J_pvalue,
        V_X=V_X,
    )


__all__ = ["JTestResult", "j_test"]
