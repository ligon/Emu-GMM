"""Optimiser adapters for the framework's :class:`~emu_gmm.types.Optimizer`
protocol.

Two factories are provided:

- :func:`optimistix_lm` --- wraps :class:`optimistix.LevenbergMarquardt`
  and :func:`optimistix.least_squares`. The default for v1: JAX-native,
  JIT-pure, ``vmap``-able, and traceable through ``jax.grad`` for
  meta-level differentiation.
- :func:`scipy_lm` --- wraps :func:`scipy.optimize.least_squares` with
  ``method='lm'``. A non-JIT-pure fallback that converts JAX arrays to
  NumPy at the Python/SciPy boundary. Useful for diagnosing optimistix
  convergence problems, but **not** ``jit`` / ``vmap`` compatible because
  SciPy's solver loop lives in interpreted Python.

Both factories return callable instances satisfying the runtime-checkable
:class:`~emu_gmm.types.Optimizer` protocol. Each call returns
``(theta_opt, OptimizerInfo)`` where ``OptimizerInfo`` carries the step
count, a normalised status string, the final objective
:math:`\\tfrac{1}{2}\\lVert r(\\theta_{\\mathrm{opt}})\\rVert^2`, and the
backend identifier.

See ``docs/api-sketch.org`` Section 3 (/Optimizer/) and
``docs/implementation-plan.org`` Section 6 (/Phase 4/) for the
architectural context.
"""

from __future__ import annotations

import dataclasses
import weakref
from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optimistix as optx
import scipy.optimize as so
from jaxtyping import Array, Float

from emu_gmm.types import OptimizerInfo

# Module-level cache mapping a user ``residual_fn`` (by ``id``) to its
# optimistix ``fn(y, args)`` wrapper. Caching the wrapper is what gives
# :func:`build_estimator` its second-call no-retrace property:
# optimistix's internal pjit cache keys on the wrapper's identity, so
# rebuilding ``fn`` on each call would miss the cache every time even
# when ``residual_fn`` itself is unchanged.
#
# We hold the wrapper *strongly* (so it survives between optimiser
# invocations) and register a :class:`weakref.finalize` on the
# user-supplied ``residual_fn`` to evict the cache entry when the
# residual closure is garbage-collected. This avoids unbounded growth
# in long-running processes that build many factories.
_OPTIMISTIX_FN_CACHE: dict[int, Any] = {}


def _optimistix_wrap(
    residual_fn: Callable[[Float[Array, " K"]], Float[Array, " M"]],
) -> Callable[[Float[Array, " K"], Any], Float[Array, " M"]]:
    """Return a memoised optimistix ``fn(y, args)`` wrapper for ``residual_fn``.

    Optimistix expects two-argument ``fn(y, args)``. We supply a thin
    wrapper that ignores ``args`` and forwards to the user's
    one-argument residual. Building the wrapper afresh on every solver
    invocation defeats optimistix's pjit cache (the cache keys on
    closure identity); memoising on ``id(residual_fn)`` keeps the
    wrapper identity stable so the second call hits the cache.
    """
    key = id(residual_fn)
    cached = _OPTIMISTIX_FN_CACHE.get(key)
    if cached is not None:
        return cached

    def fn(y: Float[Array, " K"], args: Any) -> Float[Array, " M"]:
        return residual_fn(y)

    _OPTIMISTIX_FN_CACHE[key] = fn
    # Evict when ``residual_fn`` is GC'd. ``id()`` is captured by value
    # so the finalize closure does not keep ``residual_fn`` alive.
    weakref.finalize(residual_fn, _OPTIMISTIX_FN_CACHE.pop, key, None)
    return fn


# ---------------------------------------------------------------------------
# Optimistix adapter
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class _OptimistixLM:
    """Callable adapter around :class:`optimistix.LevenbergMarquardt`.

    Instances satisfy the :class:`~emu_gmm.types.Optimizer` protocol.
    Construction is cheap; the underlying solver is rebuilt on each call
    so the dataclass remains hashable and ``jit``-friendly.
    """

    rtol: float
    atol: float
    max_steps: int

    def __call__(
        self,
        residual_fn: Callable[[Float[Array, " K"]], Float[Array, " M"]],
        theta_init: Float[Array, " K"],
    ) -> tuple[Float[Array, " K"], OptimizerInfo]:
        """Solve ``min_theta || residual_fn(theta) ||^2`` via LM.

        Parameters
        ----------
        residual_fn
            Maps a flat 1-D parameter array of length ``K`` to a flat
            residual vector of length ``M``. Must be JAX-traceable.
        theta_init
            Initial parameter guess, a 1-D JAX array of length ``K``.

        Returns
        -------
        theta_opt : (K,) jax array
            The minimiser estimate.
        info : OptimizerInfo
            ``steps`` from optimistix's ``Solution.stats["num_steps"]``,
            ``status`` translated from ``Solution.result``, and
            ``final_objective`` recomputed from ``residual_fn(theta_opt)``.
        """
        solver: optx.LevenbergMarquardt = optx.LevenbergMarquardt(
            rtol=self.rtol, atol=self.atol
        )

        # ``fn`` is a memoised wrapper around ``residual_fn``: see
        # :func:`_optimistix_wrap`. Optimistix's internal pjit cache
        # keys on the wrapper's identity; memoising keeps the cache
        # warm across calls with the same residual.
        fn = _optimistix_wrap(residual_fn)

        sol: optx.Solution = optx.least_squares(
            fn,
            solver,
            theta_init,
            max_steps=self.max_steps,
            throw=False,
        )

        theta_opt = sol.value
        # Under jit, ``sol.stats["num_steps"]`` and ``final_objective``
        # remain traced JAX scalars; the dataclass field annotations
        # (``int``, ``float``) are nominal --- they describe the eager
        # contract, not the traced one.
        steps = sol.stats["num_steps"]
        status = _optimistix_status(sol.result)

        r = residual_fn(theta_opt)
        final_objective = 0.5 * jnp.sum(r * r)

        info = OptimizerInfo(
            steps=steps,
            final_objective=final_objective,
            status=status,
            backend="optimistix",
        )
        return theta_opt, info


def optimistix_lm(
    rtol: float = 1e-8,
    atol: float = 1e-8,
    max_steps: int = 200,
) -> _OptimistixLM:
    """Build an optimistix Levenberg--Marquardt optimiser.

    Parameters
    ----------
    rtol, atol
        Relative and absolute tolerances passed to
        :class:`optimistix.LevenbergMarquardt`.
    max_steps
        Maximum number of LM iterations. If reached without convergence
        the returned :class:`~emu_gmm.types.OptimizerInfo` has
        ``status="max_iterations"``.

    Returns
    -------
    optimiser
        A callable satisfying the :class:`~emu_gmm.types.Optimizer`
        protocol. ``jit`` / ``vmap`` compatible.
    """
    return _OptimistixLM(rtol=rtol, atol=atol, max_steps=max_steps)


def _optimistix_status(result: Any) -> str:
    """Translate an :class:`optimistix.RESULTS` value to a status string.

    Returns ``"converged"`` for a successful solve,
    ``"max_iterations"`` if the maximum step count was hit, and
    ``"diverged"`` for anything else (singular Hessian, breakdown,
    non-finite iterates, ...).

    Under ``jax.jit`` tracing the result code is a JAX tracer rather
    than a concrete value, so the eager Python branches cannot fire.
    In that case the status is reported as ``"traced"``; users who need
    the concrete status should inspect ``info.status`` outside of
    ``jit``, or rely on ``info.final_objective`` (which is JIT-pure).
    """
    if hasattr(result, "is_traced") and result.is_traced():
        return "traced"
    if result == optx.RESULTS.successful:
        return "converged"
    if result == optx.RESULTS.nonlinear_max_steps_reached:
        return "max_iterations"
    return "diverged"


# ---------------------------------------------------------------------------
# SciPy adapter
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class _ScipyLM:
    """Callable adapter around :func:`scipy.optimize.least_squares`.

    Performs the JAX <-> NumPy boundary conversion inside ``__call__``.
    Not ``jit`` / ``vmap`` compatible: the optimiser loop runs in
    interpreted Python via SciPy and cannot be traced by JAX. Suitable
    for diagnosing optimistix convergence trouble at the eager-call
    site, not as a default in compiled pipelines.
    """

    # Keyword arguments forwarded to scipy.optimize.least_squares. Stored
    # as a tuple of (key, value) pairs so the dataclass stays hashable.
    options: tuple[tuple[str, Any], ...]

    def __call__(
        self,
        residual_fn: Callable[[Float[Array, " K"]], Float[Array, " M"]],
        theta_init: Float[Array, " K"],
    ) -> tuple[Float[Array, " K"], OptimizerInfo]:
        """Solve ``min_theta || residual_fn(theta) ||^2`` via SciPy LM.

        Converts ``theta_init`` to NumPy on entry; wraps ``residual_fn``
        in a JAX <-> NumPy adapter; converts the returned solution back
        to JAX arrays.
        """

        def np_residual(x: np.ndarray) -> np.ndarray:
            return np.asarray(residual_fn(jnp.asarray(x)))

        x0 = np.asarray(theta_init)
        kwargs = dict(self.options)
        result = so.least_squares(np_residual, x0, method="lm", **kwargs)

        theta_opt = jnp.asarray(result.x)

        status = _scipy_status(int(result.status))

        # scipy returns cost = 0.5 * sum(residual**2); use that directly
        # when present, otherwise recompute.
        if hasattr(result, "cost") and result.cost is not None:
            final_objective = float(result.cost)
        else:
            r = np.asarray(result.fun)
            final_objective = float(0.5 * np.sum(r * r))

        # scipy reports nfev (function evaluations). LM uses nfev as a
        # proxy for the iteration count; njev is unset for method='lm'.
        steps = int(result.nfev)

        info = OptimizerInfo(
            steps=steps,
            final_objective=final_objective,
            status=status,
            backend="scipy",
        )
        return theta_opt, info


def scipy_lm(**kw: Any) -> _ScipyLM:
    """Build a SciPy Levenberg--Marquardt optimiser.

    Parameters
    ----------
    **kw
        Keyword arguments forwarded to
        :func:`scipy.optimize.least_squares`. The ``method='lm'`` choice
        is always supplied by the adapter and must not be overridden.

    Returns
    -------
    optimiser
        A callable satisfying the :class:`~emu_gmm.types.Optimizer`
        protocol. **Not** ``jit`` / ``vmap`` compatible because the
        SciPy solver loop runs in interpreted Python.
    """
    if "method" in kw:
        raise ValueError(
            "scipy_lm always uses method='lm'; do not pass 'method' through **kw"
        )
    return _ScipyLM(options=tuple(kw.items()))


def _scipy_status(code: int) -> str:
    """Translate a SciPy ``least_squares`` status code to a string.

    SciPy convention (see :func:`scipy.optimize.least_squares` docs):
    negative codes indicate improper input or divergence, ``0`` means
    the maximum number of function evaluations was exceeded, positive
    codes indicate one of several convergence criteria was met.
    """
    if code == 0:
        return "max_iterations"
    if code > 0:
        return "converged"
    return "diverged"


# ---------------------------------------------------------------------------
# Linear (affine-in-theta) fast-path adapter
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class _LinearSolver:
    """Certificate-based fast path for affine-in-``theta`` residuals.

    When ``residual_fn`` is affine in ``theta`` --- i.e. the structural
    moment is linear in the parameters and the whitening does *not* make
    it non-affine (so ``weighting=Identity()`` or ``Fixed(...)``, **not**
    ``ContinuouslyUpdated``) --- the least-squares minimiser is reached in
    a single Gauss--Newton step. This adapter takes that step and then
    *certifies* it by checking the first-order optimality condition at the
    candidate, rather than trying to *detect* affinity structurally
    (comparing Jacobians at two points is unsound: equal Jacobians at
    sampled points do not imply the function is affine --- e.g.
    ``cos(0) == cos(2*pi)``).

    The certificate is exact for an affine residual (the post-step
    gradient is zero up to round-off) and fails for a genuinely nonlinear
    one (a single Gauss--Newton step leaves the gradient at ``O(||g0||)``).
    On a failed certificate the call delegates *entirely* to the fallback
    optimiser and returns its ``(theta_opt, info)`` unchanged, so the
    reported ``backend`` reflects the fallback that actually solved the
    problem.

    The accept/reject decision is a Python ``bool(...)`` on a concrete
    value, which is correct in the eager path :func:`emu_gmm.estimate`
    drives the optimiser through. Under :func:`jax.jit` ``theta_init`` is
    a tracer and the ``bool(...)`` raises; that case is caught and the
    call delegates wholly to the fallback (always JIT-safe). The
    speculative linear solve is *not* attempted under trace.

    Instances satisfy the :class:`~emu_gmm.types.Optimizer` protocol.
    """

    fallback: Any  # an Optimizer; constructed in :func:`linear_solver`.
    tol: float

    def __call__(
        self,
        residual_fn: Callable[[Float[Array, " K"]], Float[Array, " M"]],
        theta_init: Float[Array, " K"],
    ) -> tuple[Float[Array, " K"], OptimizerInfo]:
        """Solve ``min_theta || residual_fn(theta) ||^2`` via the fast path.

        Take one linear least-squares step from ``theta_init`` and accept
        it iff the first-order optimality certificate holds; otherwise
        delegate to :attr:`fallback`.

        Parameters
        ----------
        residual_fn
            Maps a flat ``(K,)`` parameter array to a flat ``(M,)``
            residual. JAX-traceable.
        theta_init
            Initial parameter guess, a 1-D JAX array of length ``K``.

        Returns
        -------
        theta_opt : (K,) jax array
        info : OptimizerInfo
            ``backend == "linear"`` with ``steps == 1`` on an accepted
            certificate; otherwise the fallback's own ``info`` verbatim.
        """
        # Under jit, ``theta_init`` is a tracer: the speculative solve and
        # its concrete accept/reject are unsafe. Delegate wholly. We probe
        # for the traced case via the certificate's bool(...) below, but
        # also guard the whole linear attempt in try/except so any
        # tracer-leak path falls back safely.
        try:
            j0 = jax.jacfwd(residual_fn)(theta_init)  # (M, K)
            r0 = residual_fn(theta_init)  # (M,)

            # One Gauss--Newton / linear least-squares step. Exact for an
            # affine residual in both the just-identified (M == K) and
            # over-identified (M > K) cases.
            delta = -jnp.linalg.lstsq(j0, r0, rcond=None)[0]
            theta_hat = theta_init + delta

            # Certify the first-order optimality condition at the
            # candidate: g1 = J1' r1 should vanish for an affine residual.
            j1 = jax.jacfwd(residual_fn)(theta_hat)
            r1 = residual_fn(theta_hat)
            g1 = j1.T @ r1
            g0 = j0.T @ r0

            g1_norm = jnp.linalg.norm(g1)
            g0_norm = jnp.linalg.norm(g0)
            threshold = self.tol * jnp.maximum(g0_norm, 1.0)

            # Concrete bool: fine eagerly, raises under trace (caught
            # below to delegate to the fallback).
            accept = bool(g1_norm <= threshold)
        except (
            jax.errors.TracerBoolConversionError,
            jax.errors.ConcretizationTypeError,
        ):
            return self.fallback(residual_fn, theta_init)

        if accept:
            final_objective = 0.5 * jnp.sum(r1 * r1)
            info = OptimizerInfo(
                steps=1,
                final_objective=final_objective,
                status="converged",
                backend="linear",
            )
            return theta_hat, info

        # Certificate failed: genuinely nonlinear residual. Delegate
        # entirely to the fallback and surface its result unchanged.
        return self.fallback(residual_fn, theta_init)


def linear_solver(
    *,
    fallback: Any = None,
    tol: float = 1e-7,
) -> _LinearSolver:
    """Build a certificate-based linear fast-path optimiser.

    For an affine-in-``theta`` residual the least-squares minimiser is
    reached in one Gauss--Newton step; this optimiser takes that step and
    *certifies* it via the first-order optimality condition. On a failed
    certificate (a nonlinear residual --- including the continuously
    updated whitened moment, which is non-affine even for a linear model)
    it delegates to ``fallback``.

    Parameters
    ----------
    fallback : :class:`~emu_gmm.types.Optimizer`, optional
        Optimiser invoked when the certificate fails or when called under
        :func:`jax.jit` (where the speculative concrete solve is unsafe).
        Defaults to :func:`optimistix_lm` with default tolerances.
    tol : float, optional
        Certificate tolerance. The step is accepted iff
        ``||J1' r1|| <= tol * max(||J0' r0||, 1.0)``. Default ``1e-7``.

    Returns
    -------
    optimiser
        A callable satisfying the :class:`~emu_gmm.types.Optimizer`
        protocol. Safe under :func:`jax.jit` (delegates to ``fallback``).
    """
    if fallback is None:
        fallback = optimistix_lm()
    return _LinearSolver(fallback=fallback, tol=tol)


__all__ = ["optimistix_lm", "scipy_lm", "linear_solver"]
