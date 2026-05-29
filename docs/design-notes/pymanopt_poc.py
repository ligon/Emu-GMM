"""POC: pymanopt JAX backend on a tiny GMM-style problem.

Goal: estimate Y in R^{m x k} parametrising PSD X = Y Y^T,
so as to match a target moment matrix M_target. This is a
representative low-rank PSD recovery problem analogous to a
GMM moment-matching objective.

Outputs (printed to stdout):
- pymanopt JAX backend availability / float64 status
- whether the cost computes in JAX without errors
- the returned theta_hat (a numpy ndarray)
- type of returned point (numpy or JAX)
- whether jax.jit composes (we try it as a smoke test on the cost)
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

# Force float64
from jax import config as _jax_config
_jax_config.update("jax_enable_x64", True)

import pymanopt
import pymanopt.function
from pymanopt.manifolds import PSDFixedRank
from pymanopt.optimizers import TrustRegions

print("jax x64 enabled:", jax.config.read("jax_enable_x64"))

m, k = 3, 2
rng = np.random.default_rng(0)
Y_true = rng.normal(size=(m, k))
M_target = Y_true @ Y_true.T  # numpy float64

manifold = PSDFixedRank(n=m, k=k)
print("Manifold point dtype:", manifold.random_point().dtype)

# Define cost using pymanopt.function.jax decorator. The decorator
# requires the function reference the manifold.
@pymanopt.function.jax(manifold)
def cost(Y):
    # Y arrives as numpy ndarray; JAX backend treats it as a jnp array
    # only inside the autodiff trace. Inside this function we use jnp
    # so jax.grad can differentiate.
    diff = Y @ Y.T - jnp.asarray(M_target)
    return 0.5 * jnp.sum(diff * diff)

problem = pymanopt.Problem(manifold, cost)

# Smoke test: can we evaluate cost on a JAX array?
Y0 = jnp.asarray(rng.normal(size=(m, k)))
print("cost(Y0) eager:", float(cost(Y0)))

# Smoke test: can we jit the *cost* (the underlying user fn)?
try:
    cost_jit = jax.jit(lambda Y: cost(Y))
    print("cost jit value:", float(cost_jit(Y0)))
    jit_ok = True
except Exception as exc:
    print("jit failed:", exc)
    jit_ok = False

# Smoke test: vmap over a batch of starting points (cost only, not solver)
try:
    Y_batch = jnp.asarray(rng.normal(size=(4, m, k)))
    vals = jax.vmap(lambda Y: cost(Y))(Y_batch)
    print("vmap cost values:", np.asarray(vals))
    vmap_ok = True
except Exception as exc:
    print("vmap failed:", exc)
    vmap_ok = False

# Run the actual TrustRegions optimizer
optimizer = TrustRegions(verbosity=0, max_iterations=100)
result = optimizer.run(problem)
Y_hat = result.point

print("type(Y_hat):", type(Y_hat).__name__)
print("dtype(Y_hat):", getattr(Y_hat, "dtype", "n/a"))
print("Y_hat:\n", Y_hat)
print("residual ||Y_hat Y_hat^T - M_target||_F:",
      float(np.linalg.norm(Y_hat @ Y_hat.T - M_target)))
print("iterations:", result.iterations)
print("cost at solution:", result.cost)
print("jit_ok:", jit_ok, "vmap_ok:", vmap_ok)
