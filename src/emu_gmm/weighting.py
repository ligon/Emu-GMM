"""Weighting / whitening strategies for the GMM objective.

The framework's objective is
:math:`Q_\\mu(\\theta) = \\| L_\\mu(\\theta)^{-1} m_\\mu(\\theta) \\|^2`,
where :math:`L L^\\top = V` is the Cholesky factor of the
moment-estimator variance and :math:`m` is the empirical moment.
A ``WeightingStrategy`` chooses how :math:`L` (and therefore the
weighting matrix :math:`\\Lambda = V^{-1}`) is constructed.

Four concrete strategies exist:

- :class:`Identity` --- :math:`\\Lambda \\equiv I`; ``whitening_residual``
  returns ``m`` unchanged.
- :class:`Fixed` --- :math:`L = L_0` precomputed from an anchor
  :math:`V_0`; the optimisation surface is quadratic in ``m`` alone.
- :class:`ContinuouslyUpdated` --- :math:`L(\\theta)` is recomputed at
  every call; JAX AD threads through the Cholesky and the triangular
  solve so the residual's gradient picks up the dependence of
  :math:`L` on :math:`\\theta` when :math:`V = V(\\theta)`.
- :class:`IteratedWeighting` --- legacy two-step / iterated GMM. The
  estimator runs an outer Python loop alternating Fixed-weight solves
  and variance refreshes until ``theta`` stabilises.

See ``docs/design.org`` Section 5 ("Architectural Core Highlights")
for the architectural commitment that the CU gradient must not drop
the :math:`\\nabla_\\theta V` term; that property is delivered here
by computing the Cholesky inside ``whitening_residual`` rather than
caching :math:`L`.

Outer-loop hook
---------------
The optional :attr:`requires_outer_loop` / :meth:`outer_loop_driver`
extension on :class:`~emu_gmm.types.WeightingStrategy` lets a strategy
opt out of the standard residual-path dispatch and provide its own
Python-level driver. :class:`IteratedWeighting` is the only built-in
strategy that uses this; third-party strategies that need their own
outer loop should follow the same pattern.
"""

from __future__ import annotations

import warnings
from typing import Any, cast

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
from jaxtyping import Array, Float

from emu_gmm._internal import params as params_mod
from emu_gmm._internal.cholesky import cholesky, forward_solve
from emu_gmm.types import (
    CovarianceStrategy,
    Measure,
    Optimizer,
    OptimizerInfo,
    ParamsLike,
    StructuralModel,
)


@jdc.pytree_dataclass
class Identity:
    """Identity weighting: ``y = m``.

    Equivalent to setting :math:`\\Lambda = I_M` in the GMM objective.
    The supplied ``V`` is ignored. Useful as a sanity-check weighting
    or when the user wants an unweighted sum of squares.
    """

    #: Standard residual-path strategy --- no outer driver needed.
    requires_outer_loop: bool = jdc.static_field(default=False)  # type: ignore[attr-defined]

    def whitening_residual(
        self,
        m: Float[Array, " M"],
        V: Float[Array, "M M"],
        theta: ParamsLike,
    ) -> Float[Array, " M"]:
        """Return ``m`` unchanged.

        Parameters
        ----------
        m : (M,) array
            Empirical moment vector.
        V : (M, M) array
            Ignored.
        theta : ParamsLike
            Ignored.
        """
        del V, theta  # accepted for protocol conformance; not used here
        return m


@jdc.pytree_dataclass(init=False)
class Fixed:
    """Pre-cholesky weighting at a frozen anchor :math:`V_0`.

    Stores the lower-triangular Cholesky factor :math:`L_0` of the
    anchor variance; the optimiser sees a quadratic-in-``m`` surface.
    The ``V`` argument to :meth:`whitening_residual` is accepted (so
    the protocol signature matches) but ignored.

    Construction is keyword-only and requires exactly one of ``L0`` or
    ``V0``:

    - ``Fixed(L0=L0)`` --- supply the Cholesky factor directly.
    - ``Fixed(V0=V0)`` --- supply the anchor variance; the framework
      computes :math:`L_0 = \\mathrm{chol}(V_0)` for you.

    Equivalent classmethod constructors :meth:`from_L0` and
    :meth:`from_V0` are also provided.

    Notes
    -----
    Positional construction is intentionally disallowed. In other GMM
    libraries (notably ManifoldGMM) the analogous one-arg constructor
    accepts the weighting matrix ``W`` directly; in :mod:`emu_gmm` the
    stored object is the Cholesky factor :math:`L_0` of the *variance*
    :math:`V_0 = W^{-1}`. Silently storing ``W`` as ``L0`` would yield
    a wrong-but-runnable estimator, so we require an explicit kwarg.

    Parameters
    ----------
    L0 : (M, M) lower-triangular array, keyword-only
        Cholesky factor of the anchor variance.
    V0 : (M, M) symmetric positive-definite array, keyword-only
        Anchor variance; the Cholesky factor is computed internally.
    """

    L0: Float[Array, "M M"]
    #: Standard residual-path strategy --- no outer driver needed.
    requires_outer_loop: bool = jdc.static_field(default=False)  # type: ignore[attr-defined]

    def __init__(
        self,
        *args: Any,
        L0: Float[Array, "M M"] | None = None,
        V0: Float[Array, "M M"] | None = None,
        requires_outer_loop: bool = False,
    ) -> None:
        if args:
            raise TypeError(
                "Fixed(...) does not accept positional arguments. "
                "Pass either Fixed(L0=L0) (Cholesky factor of the anchor "
                "variance) or Fixed(V0=V0) (anchor variance; Cholesky "
                "computed internally). Note: the stored object is the "
                "Cholesky factor of the variance V_0, not the weighting "
                "matrix W = V_0^{-1}. If you are porting code that wrote "
                "Fixed(W), use Fixed.from_V0(jnp.linalg.inv(W)) instead."
            )
        if L0 is None and V0 is None:
            raise TypeError(
                "Fixed(...) requires exactly one of L0= or V0=; neither "
                "was supplied. Use Fixed(V0=V0) if you have the anchor "
                "variance, or Fixed(L0=L0) if you already have its "
                "Cholesky factor."
            )
        if L0 is not None and V0 is not None:
            raise TypeError(
                "Fixed(...) requires exactly one of L0= or V0=; both "
                "were supplied. The two are redundant (L0 is the "
                "Cholesky factor of V0); pick one."
            )
        if V0 is not None:
            L0 = cholesky(V0)
        # object.__setattr__ because @jdc.pytree_dataclass is frozen.
        object.__setattr__(self, "L0", L0)
        # ``requires_outer_loop`` is a static (treedef) field; default
        # ``False`` --- ``Fixed`` rides the standard residual-path
        # dispatch in :func:`emu_gmm.estimate`. The kwarg here exists so
        # the jax_dataclasses pytree-unflatten path can round-trip the
        # static field after a ``jit`` boundary.
        object.__setattr__(self, "requires_outer_loop", requires_outer_loop)

    @classmethod
    def from_V0(cls, V0: Float[Array, "M M"]) -> Fixed:
        """Construct from an anchor variance ``V0`` (pre-cholesky).

        Equivalent to ``Fixed(V0=V0)``.
        """
        return cls(V0=V0)

    @classmethod
    def from_L0(cls, L0: Float[Array, "M M"]) -> Fixed:
        """Construct directly from a Cholesky factor ``L0``.

        Equivalent to ``Fixed(L0=L0)``. Useful if you already hold the
        lower-triangular factor and want to make the intent explicit at
        the call site.
        """
        return cls(L0=L0)

    def whitening_residual(
        self,
        m: Float[Array, " M"],
        V: Float[Array, "M M"],
        theta: ParamsLike,
    ) -> Float[Array, " M"]:
        """Return :math:`y = L_0^{-1} m`.

        The ``V`` and ``theta`` arguments are accepted for protocol
        conformance but ignored: the weighting is frozen at the anchor.
        """
        del V, theta
        return forward_solve(self.L0, m)


@jdc.pytree_dataclass
class ContinuouslyUpdated:
    """Continuously-updated (CU) weighting: :math:`L(\\theta)` per call.

    The Cholesky factor is recomputed at every evaluation, so JAX AD
    traces through the dependence of :math:`L` on :math:`\\theta` via
    :math:`V(\\theta)`. This is the default v1 weighting strategy.

    Also exported as the alias :data:`CUE` (continuously-updated
    estimator), the more common name in the econometrics literature
    following Hansen, Heaton & Yaron (1996, "Finite-Sample Properties
    of Some Alternative GMM Estimators", JBES 14(3), 262--280).

    Has no traced or static state aside from the protocol's
    ``requires_outer_loop`` flag.
    """

    #: Standard residual-path strategy --- no outer driver needed.
    requires_outer_loop: bool = jdc.static_field(default=False)  # type: ignore[attr-defined]

    def whitening_residual(
        self,
        m: Float[Array, " M"],
        V: Float[Array, "M M"],
        theta: ParamsLike,
    ) -> Float[Array, " M"]:
        """Return :math:`y = L(\\theta)^{-1} m` with :math:`L L^\\top = V`.

        Both ``cholesky`` and ``forward_solve`` are differentiable, so
        the gradient of any downstream scalar of ``y`` picks up the
        dependence of :math:`L` on :math:`\\theta` through :math:`V`.

        Parameters
        ----------
        m : (M,) array
            Empirical moment vector at ``theta``.
        V : (M, M) array
            Variance of the moment estimator at ``theta``.
        theta : ParamsLike
            Accepted for protocol conformance; not used directly --- the
            ``theta``-dependence enters via ``V``.
        """
        del theta
        L = cholesky(V)
        return forward_solve(L, m)


@jdc.pytree_dataclass
class IteratedWeighting:
    """Iterated (two-step / k-step) GMM weighting.

    The classic Hansen-style iterated GMM scheme. At each outer step
    :math:`k` the strategy:

    1. holds :math:`V` fixed at :math:`V(\\theta_k)` and solves
       :math:`\\theta_{k+1} = \\arg\\min_\\theta \\| L_k^{-1} m(\\theta) \\|^2`
       using the existing :class:`Fixed`-weight machinery, then
    2. refreshes :math:`V` to :math:`V(\\theta_{k+1})`.

    The loop stops when
    :math:`\\| \\theta_{k+1} - \\theta_k \\|_2 < \\texttt{weighting_tol}`
    or after ``weighting_iterations`` outer steps. The outer loop runs
    in pure Python in :func:`emu_gmm.estimate`; each inner Fixed-weight
    solve is JIT-compiled by the underlying optimiser.

    Iterated weighting and :class:`ContinuouslyUpdated` (CU) are
    asymptotically equivalent (Hansen-Heaton-Yaron 1996) but differ in
    finite samples. CU is the v1 default; ``IteratedWeighting`` exists
    for **legacy reproducibility** --- K-Aggregators' published
    headline pickles, for example, were produced with two-step / iterated
    weighting and the ManifoldGMM ``weighting_iterations`` /
    ``weighting_tol`` knobs.

    Convergence caveat
    ------------------
    Iterated GMM is **not** guaranteed to be a contraction on
    misspecified models; K-Aggregators' ``V2_PORT.org`` documents an
    explicit divergence case. When ``weighting_iterations`` is exhausted
    without reaching ``weighting_tol``, :func:`emu_gmm.estimate`
    surfaces a non-convergence flag through
    :class:`~emu_gmm.types.Diagnostics` and the result's ``converged``
    field rather than raising; the partially-converged
    :math:`\\theta_k` is returned. Users debugging non-convergence
    should compare to a CU run on the same problem.

    Parameters
    ----------
    weighting_iterations : int
        Maximum number of outer (V-refresh) iterations. Must be at
        least 1.
    weighting_tol : float
        Stop when :math:`\\| \\theta_{k+1} - \\theta_k \\|_2` falls below
        this. Must be strictly positive.

    Notes
    -----
    Both fields are :func:`jax_dataclasses.static_field` --- they are
    hyperparameters of the estimation procedure, not traced quantities,
    and changing either should retrigger JIT compilation of the inner
    solves.

    Calling :meth:`whitening_residual` directly (outside the
    estimator's outer loop) falls back to continuously-updated
    behaviour: :math:`L(\\theta)` is recomputed per call from the
    supplied :math:`V`. This makes the instance protocol-conformant and
    usable for ad-hoc residual evaluation, but the *iterated* algorithm
    only runs when the strategy is handed to :func:`emu_gmm.estimate`.
    """

    weighting_iterations: int = jdc.static_field(default=10)  # type: ignore[attr-defined]
    weighting_tol: float = jdc.static_field(default=1e-6)  # type: ignore[attr-defined]
    #: Iterated GMM needs the estimator to drive an outer Python loop;
    #: see :meth:`outer_loop_driver`.
    requires_outer_loop: bool = jdc.static_field(default=True)  # type: ignore[attr-defined]

    def __post_init__(self) -> None:
        if int(self.weighting_iterations) < 1:
            raise ValueError(
                "IteratedWeighting.weighting_iterations must be >= 1, got "
                f"{self.weighting_iterations}"
            )
        if float(self.weighting_tol) <= 0.0:
            raise ValueError(
                "IteratedWeighting.weighting_tol must be > 0, got "
                f"{self.weighting_tol}"
            )

    def whitening_residual(
        self,
        m: Float[Array, " M"],
        V: Float[Array, "M M"],
        theta: ParamsLike,
    ) -> Float[Array, " M"]:
        """Continuously-updated fallback :math:`y = L(\\theta)^{-1} m`.

        The *iterated* algorithm is driven by :func:`emu_gmm.estimate`,
        which runs the outer V-refresh loop in pure Python and
        dispatches each inner solve to a :class:`Fixed`-weight problem.
        This method exists so that ``IteratedWeighting`` satisfies the
        :class:`~emu_gmm.types.WeightingStrategy` protocol and remains
        usable for direct residual evaluation; in that direct-call
        path the behaviour is identical to
        :class:`ContinuouslyUpdated`.
        """
        del theta
        L = cholesky(V)
        return forward_solve(L, m)

    def outer_loop_driver(
        self,
        model: StructuralModel,
        measure: Measure,
        covariance: CovarianceStrategy,
        theta_init_flat: Float[Array, " K"],
        treedef: Any,
        *,
        make_residual_fn: Any,
        cu_residual_fn: Any,
        apply_ridge: Any,
        optimizer: Optimizer,
        fixed_kernel: Any = None,
        chol_kernel: Any = None,
    ) -> tuple[Float[Array, " K"], OptimizerInfo, str]:
        """Drive the outer iterated-GMM loop in pure Python.

        Called by :func:`emu_gmm.estimate` (or
        :func:`emu_gmm.build_estimator`) when ``requires_outer_loop`` is
        ``True``. Returns ``(theta_hat_flat, final_info,
        outer_status)``. The returned ``OptimizerInfo.final_objective``
        is the *CU-fallback* objective at the returned ``theta_hat``,
        not the inner Fixed-weight objective at the penultimate
        :math:`V_k`. The two coincide at the V-refresh fixed point but
        differ when the outer loop is capped early; the CU-fallback
        value is the one consistent with :meth:`whitening_residual`.

        Parameters
        ----------
        model, measure, covariance, theta_init_flat, treedef
            Standard estimator state; passed through from the caller.
        make_residual_fn
            Factory ``make_residual_fn(weighting) -> residual_fn`` so the
            driver can spawn a :class:`Fixed`-weight residual at each
            outer step. The factory must already close over the model,
            measure, anchored ridge, optional penalty, and any cached
            intermediates path so the driver itself remains free of
            estimator internals.
        cu_residual_fn
            Pre-built CU-fallback residual function; used only to
            report the user-facing ``final_objective`` at termination.
        apply_ridge
            Closure that applies the anchored ridge to a freshly-built
            ``V_k``. The driver doesn't introspect the regularisation
            strategy; this closure carries the anchored ``tau``.
        optimizer
            The user's :class:`~emu_gmm.types.Optimizer`. Each inner
            Fixed-weight solve is delegated to it.
        fixed_kernel, chol_kernel
            Optional #124 (PR B) traced-argument kernels supplied by
            :func:`emu_gmm.build_estimator` when the ported surface
            applies. ``fixed_kernel(theta_flat, (measure, L0))`` is the
            Fixed-weight whitened residual with the anchor's Cholesky
            factor ``L0`` riding as a *traced leaf*;
            ``chol_kernel(theta_flat, measure)`` is the jitted V-refresh
            ``chol(ridge(V(theta_k)))``. When both are supplied AND the
            optimiser exposes the ``args`` channel, every inner solve
            threads ``args=(measure, L0_k)`` through the single
            stable-identity ``fixed_kernel`` --- so all outer steps, and
            repeated fits with fresh same-structure measures, share ONE
            trace instead of rebuilding a fresh ``Fixed`` closure per
            outer step (which retraced the optimiser every step).
            ``None`` (the default, and what an estimator built before
            PR B passes) preserves the legacy ``make_residual_fn``
            pathway bit-for-bit; the kwargs are keyword-only with
            defaults so third-party callers of the OLD signature are
            unaffected.

        Termination
        -----------
        The loop terminates on either:

        - :math:`\\| \\theta_{k+1} - \\theta_k \\|_2 < \\texttt{weighting_tol}
          \\cdot \\max(\\| \\theta_k \\|_2, \\texttt{eps})` (status
          ``"converged"``) --- the tolerance is rescaled by the current
          parameter norm so that ``weighting_tol`` carries meaning across
          problems whose parameters differ by orders of magnitude, or
        - having performed ``weighting_iterations`` outer steps without
          meeting the rescaled tolerance (status ``"max_iterations"``).

        Inner-solve divergence handling
        -------------------------------
        Each inner :class:`Fixed`-weight solve returns its own
        :class:`~emu_gmm.types.OptimizerInfo`. If any inner ``info_k.status``
        is neither ``"converged"`` nor ``"traced"`` (i.e. the inner LM /
        least-squares run hit ``max_iterations`` or otherwise failed to
        certify convergence), the iterated driver emits a
        :class:`UserWarning` and returns outer status
        ``"inner_non_convergence"`` so the caller can flip
        ``EstimationResult.converged`` to ``False``.

        On ``"max_iterations"`` a :class:`UserWarning` is emitted so the
        caller is told the V-refresh fixed point was not reached; the
        partially-iterated ``theta_k`` is returned regardless.
        """
        theta_k_flat = jnp.asarray(theta_init_flat)
        last_info: OptimizerInfo | None = None
        total_inner_steps = 0
        outer_status = "max_iterations"
        # ``inner_non_convergence`` overrides ``max_iterations`` once seen,
        # because it indicates a deeper failure (the inner LM gave up).
        saw_inner_non_convergence = False
        inner_non_convergence_statuses: list[str] = []

        # eps for the rescaled-tolerance test; chosen at the float64 noise
        # floor so the absolute test still triggers when |theta_k| -> 0.
        rescale_eps = 1e-12

        # #124 (PR B): take the traced-argument path only when the
        # estimator supplied BOTH kernels and the optimiser actually
        # exposes the ``args`` channel. The capability probe here is
        # defence in depth -- :func:`emu_gmm.build_estimator` gates on
        # ``_supports_args`` before passing the kernels -- so a direct
        # caller handing kernels to a two-argument optimiser falls back
        # to the legacy closure pathway instead of crashing. Local
        # import: ``emu_gmm.optimizer`` does not import this module, but
        # keeping the dependency call-time avoids ordering surprises in
        # ``emu_gmm/__init__``.
        from emu_gmm.optimizer import _supports_args

        use_args_path = (
            fixed_kernel is not None
            and chol_kernel is not None
            and _supports_args(optimizer)
        )
        if use_args_path:
            # Normalise away the WEAK dtype of the factory-flattened
            # ``theta_init_flat`` (``~float64``): the optimiser returns
            # strong-typed arrays, so without this the step-1 trace
            # signature differs from step 2+ and the kernels retrace
            # exactly once mid-loop (jit caches key on avals INCLUDING
            # weak_type). ``convert_element_type`` to the same dtype is
            # the documented way to drop weak typing; values are
            # untouched.
            theta_k_flat = jax.lax.convert_element_type(
                theta_k_flat, theta_k_flat.dtype
            )

        for _k in range(int(self.weighting_iterations)):
            if use_args_path:
                # V-refresh through the factory's jitted kernel: the
                # anchor L0_k = chol(ridge(V(theta_k))) comes back as a
                # concrete array, then rides ``args=(measure, L0_k)``
                # into the stable-identity Fixed-weight kernel. All
                # outer steps (and repeated fits with fresh
                # same-structure measures) share the kernels' traces.
                L0_k = chol_kernel(theta_k_flat, measure)
                # ``args=`` is not part of the v1 Optimizer protocol;
                # ``use_args_path`` has already probed for it (mirrors
                # the estimator's cast on its traced branch).
                theta_next_flat, info_k = cast(Any, optimizer)(
                    fixed_kernel, theta_k_flat, args=(measure, L0_k)
                )
            else:
                theta_k = params_mod.unflatten_params(theta_k_flat, treedef)
                # Refresh V at the current theta_k and apply the *anchored*
                # ridge so the inner Fixed-weight surface uses the same tau
                # the rest of the framework does. Then freeze a Fixed-weight
                # closure at the resulting Cholesky anchor for the inner
                # solve.
                V_k = covariance.covariance(model, theta_k, measure)
                V_star_k = apply_ridge(V_k)
                fixed_k = Fixed.from_V0(V_star_k)
                inner_residual = make_residual_fn(fixed_k)

                theta_next_flat, info_k = optimizer(inner_residual, theta_k_flat)
            last_info = info_k
            # Inspect inner-solve status. ``"traced"`` is the placeholder
            # returned under jit (concrete status is not available); we
            # treat it as success because the iterated path is documented
            # as eager and any traced-status appearance there means the
            # user built a custom optimiser that doesn't surface a status.
            inner_status = str(getattr(info_k, "status", ""))
            if inner_status not in ("converged", "traced"):
                saw_inner_non_convergence = True
                inner_non_convergence_statuses.append(inner_status)
            try:
                total_inner_steps += int(info_k.steps)
            except (TypeError, ValueError):
                # ``steps`` may still be a traced scalar under jit; in
                # eager use (the contract for the iterated path) it is
                # concrete.
                pass

            delta = jnp.linalg.norm(theta_next_flat - theta_k_flat)
            # Rescale the tolerance by the current parameter norm so the
            # test is meaningful when parameters differ by orders of
            # magnitude (e.g. one component O(1), another O(1e6)). The
            # ``rescale_eps`` floor protects the limit |theta_k| -> 0,
            # where an absolute test on ``weighting_tol`` is still right.
            theta_scale = float(
                jnp.maximum(jnp.linalg.norm(theta_next_flat), rescale_eps)
            )
            theta_k_flat = theta_next_flat
            if float(delta) < float(self.weighting_tol) * theta_scale:
                outer_status = "converged"
                break

        if saw_inner_non_convergence:
            # Inner divergence dominates the outer status: a non-converged
            # inner solve invalidates the V-refresh step that follows it.
            outer_status = "inner_non_convergence"
            warnings.warn(
                "IteratedWeighting saw at least one inner Fixed-weight solve "
                "that did not certify convergence (inner statuses: "
                f"{inner_non_convergence_statuses!r}). The outer V-refresh "
                "is built on top of the inner LM step, so a non-converged "
                "inner solve invalidates the resulting V_{k+1}. The "
                "returned theta is the last accepted iterate but should "
                "not be trusted as a GMM estimate; rerun with a larger "
                "inner iteration budget or switch to ContinuouslyUpdated "
                "weighting.",
                UserWarning,
                stacklevel=3,
            )
        elif outer_status == "max_iterations":
            warnings.warn(
                "IteratedWeighting exhausted "
                f"{self.weighting_iterations} outer iterations without "
                f"reaching weighting_tol={self.weighting_tol:g} "
                "(rescaled by max(||theta||, eps)). The V-refresh fixed "
                "point was not reached; iterated GMM is not guaranteed "
                "to be a contraction on misspecified models. Consider "
                "switching to ContinuouslyUpdated weighting.",
                UserWarning,
                stacklevel=3,
            )

        assert last_info is not None  # weighting_iterations >= 1 enforced

        # The user-facing ``final_objective`` is the CU-fallback
        # objective at the final theta_hat, *not* the inner Fixed-weight
        # objective at the penultimate V_k. The two coincide when
        # iterated GMM converges (V == V_k at the fixed point) but
        # differ when it does not, and the CU-fallback value is the one
        # consistent with how the rest of the framework reports the
        # IteratedWeighting objective downstream. Match the
        # optimiser-side convention of reporting 0.5 * ||y||^2 so this
        # field is comparable across weighting strategies and backends.
        y_final = cu_residual_fn(theta_k_flat)
        final_objective_cu = 0.5 * jnp.sum(y_final * y_final)

        final_info = OptimizerInfo(
            steps=total_inner_steps,
            status=outer_status,
            final_objective=final_objective_cu,
            backend=last_info.backend,
        )
        return theta_k_flat, final_info, outer_status


#: Econometrics-literature alias for :class:`ContinuouslyUpdated`. See
#: Hansen, Heaton & Yaron (1996), "Finite-Sample Properties of Some
#: Alternative GMM Estimators", JBES 14(3), 262--280, where the
#: continuously-updated estimator (CUE) is introduced as an alternative
#: to the two-step and iterated GMM weighting schemes.
CUE = ContinuouslyUpdated


__all__ = ["Identity", "Fixed", "ContinuouslyUpdated", "CUE", "IteratedWeighting"]
