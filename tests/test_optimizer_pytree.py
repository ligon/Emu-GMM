"""``OptimizerInfo`` is a JAX PyTree (wf2 JAX-AD MEDIUM fix).

``optimistix_lm`` advertises a "JIT-pure, vmap-able" optimiser, but the
old ``@dataclasses.dataclass(frozen=True)`` ``OptimizerInfo`` was not
registered with JAX's pytree machinery. Wrapping a function whose return
value included an ``OptimizerInfo`` in ``jax.jit`` / ``jax.vmap`` raised
``TypeError`` (output is not a valid JAX type).

The fix promotes ``OptimizerInfo`` to a ``@jdc.pytree_dataclass`` with
``status`` and ``backend`` (string-typed; can't be JAX leaves) marked as
``jdc.static_field()``. ``steps`` and ``final_objective`` are traced
fields that can be 0-d JAX arrays under jit, Python ints/floats eagerly.

This regression suite is intentionally lightweight --- the structural
fix is the focus, not full optimiser semantics (covered elsewhere in
``test_optimizer.py``).

See ``docs/reviews/v1x-jax-ad-review.org`` (wf2 JAX-AD MEDIUM finding).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
from emu_gmm.optimizer import optimistix_lm
from emu_gmm.types import OptimizerInfo

# ---------------------------------------------------------------------------
# Pytree registration: the structural fix.
# ---------------------------------------------------------------------------


class TestOptimizerInfoIsPyTree:
    """``OptimizerInfo`` is registered with JAX's pytree machinery."""

    def test_tree_flatten_succeeds(self):
        """``jax.tree_util.tree_flatten`` returns leaves + a treedef.

        Pre-fix this raised ``TypeError`` because the plain dataclass
        was treated as an opaque leaf, but optimistix's traced fields
        (``steps``, ``final_objective``) needed to be flattened into the
        outer jit output buffer.
        """
        info = OptimizerInfo(
            steps=jnp.asarray(7),
            final_objective=jnp.asarray(0.5),
            status="converged",
            backend="optimistix",
        )
        leaves, treedef = jax.tree_util.tree_flatten(info)
        # The two traced fields are leaves; the two static fields ride
        # on the treedef.
        assert len(leaves) == 2
        # Round-trip preserves the structure.
        restored = jax.tree_util.tree_unflatten(treedef, leaves)
        assert isinstance(restored, OptimizerInfo)
        assert restored.status == "converged"
        assert restored.backend == "optimistix"

    def test_tree_map_over_traced_fields(self):
        """``jax.tree_util.tree_map`` should hit only the traced fields."""
        info = OptimizerInfo(
            steps=jnp.asarray(3),
            final_objective=jnp.asarray(1.0),
            status="converged",
            backend="optimistix",
        )
        doubled = jax.tree_util.tree_map(lambda x: x * 2, info)
        assert isinstance(doubled, OptimizerInfo)
        # Strings are not JAX leaves and should pass through unchanged.
        assert doubled.status == "converged"
        assert doubled.backend == "optimistix"
        # Traced fields are doubled.
        assert int(doubled.steps) == 6
        assert float(doubled.final_objective) == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# jit / vmap over a (theta, info) return shape.
# ---------------------------------------------------------------------------


def _quadratic_residual(theta: jnp.ndarray) -> jnp.ndarray:
    """Tiny least-squares problem with a unique minimum at the origin.

    ``residual_fn(theta) = theta`` so ``min || residual ||^2`` is at
    ``theta = 0``. Two parameters keeps shapes non-degenerate.
    """
    return theta


class TestJitVmapOverOptimizerReturn:
    """A function returning ``(theta_opt, info)`` traces under jit/vmap."""

    def test_jit_over_optimizer_call_succeeds(self):
        """``jax.jit`` over the optimiser call returns a traced tuple.

        Pre-fix this raised ``TypeError`` at trace time because
        ``OptimizerInfo`` was not a valid jit output.
        """
        opt = optimistix_lm(rtol=1e-6, atol=1e-6, max_steps=50)

        def run(theta_init):
            return opt(_quadratic_residual, theta_init)

        theta_init = jnp.array([1.5, -0.7])
        theta_opt, info = jax.jit(run)(theta_init)
        # Converged to the unique minimum at the origin.
        assert jnp.allclose(theta_opt, jnp.zeros(2), atol=1e-6)
        # ``info`` is an ``OptimizerInfo``; traced fields are JAX arrays.
        assert isinstance(info, OptimizerInfo)
        assert isinstance(info.steps, jax.Array)
        assert isinstance(info.final_objective, jax.Array)
        # ``status`` and ``backend`` are static strings.
        assert info.backend == "optimistix"

    def test_vmap_over_theta_init_succeeds(self):
        """``jax.vmap`` over a batched ``theta_init`` returns a batched
        ``(theta_opt, info)``. Traced fields stack along the batch axis;
        static fields are shared.
        """
        opt = optimistix_lm(rtol=1e-6, atol=1e-6, max_steps=50)

        def run(theta_init):
            return opt(_quadratic_residual, theta_init)

        batch = jnp.array([[1.5, -0.7], [0.3, 0.4], [-2.0, 1.0]])
        theta_opts, infos = jax.vmap(run)(batch)
        # All three converged to the origin.
        assert theta_opts.shape == (3, 2)
        assert jnp.allclose(theta_opts, jnp.zeros((3, 2)), atol=1e-6)
        # ``infos`` is an ``OptimizerInfo`` with batched traced fields.
        assert isinstance(infos, OptimizerInfo)
        assert infos.steps.shape == (3,)
        assert infos.final_objective.shape == (3,)
        # Static fields are not batched: the leading dimension is gone.
        assert infos.backend == "optimistix"

    def test_jit_of_vmap_composes(self):
        """``jit(vmap(...))`` over the optimiser is the composition that
        the docstring claims and that ``estimate()`` relies on
        internally.
        """
        opt = optimistix_lm(rtol=1e-6, atol=1e-6, max_steps=50)

        def run(theta_init):
            return opt(_quadratic_residual, theta_init)

        batch = jnp.array([[1.0, 2.0], [-0.5, 0.5]])
        theta_opts, infos = jax.jit(jax.vmap(run))(batch)
        assert theta_opts.shape == (2, 2)
        assert jnp.allclose(theta_opts, jnp.zeros((2, 2)), atol=1e-6)
        assert isinstance(infos, OptimizerInfo)
        assert infos.final_objective.shape == (2,)
