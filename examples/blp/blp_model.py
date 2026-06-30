"""JAX-native BLP demand model expressed against emu-gmm's interface.

The BLP (Berry-Levinsohn-Pakes 1995) random-coefficients logit demand model
estimated by GMM. The moment condition is

    E[ Z_jt * xi_jt(theta) ] = 0,

where xi_jt is the structural demand error and Z are (FE-residualised)
instruments. This module maps that onto emu-gmm's per-observation
``psi(x, theta) -> R^M`` interface with two ideas:

* **observation = market.** ``x`` is just the market index t; ``psi`` reads
  market t's data from closure-captured arrays and returns the market's
  contribution to the M=L moment vector. Averaging over markets (emu's
  measure) reproduces the BLP sample moment.

* **the share-inversion contraction is an implicitly-differentiated fixed
  point.** Given the nonlinear parameters theta2 = (sigma, pi), the mean
  utilities delta_t(theta2) solve s(delta_t, theta2) = S_obs_t per market.
  We solve it with a Berry contraction (``lax.while_loop``) and attach the
  correct forward-mode derivative via the implicit function theorem
  (``@jax.custom_jvp``): ddelta = -(ds/ddelta)^{-1} (ds/dtheta2). Forward
  mode is what emu's ``jacfwd`` (G = measure.jacobian) and the optimistix LM
  solver both consume.

The linear parameter beta (the price coefficient, after brand-FE absorption)
is estimated **jointly** with theta2 rather than concentrated out --- the
clean fit to emu's per-observation residual (it never needs a cross-market
op inside the moment).

Brand fixed effects are absorbed by pre-residualising the instruments
(``ZD_tilde`` from build_reference.py): since the within-brand projection A
is symmetric idempotent, Z'A(delta - X1 b) = (AZ)'(delta - X1 b), so demeaned
instruments + raw delta/prices reproduce the absorbed moment and the FE drop
out (AZ annihilates any brand-constant term).
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
from jaxtyping import Array, Float

CONTRACTION_TOL = 1e-13
CONTRACTION_MAXITER = 1000


# ── Parameters ───────────────────────────────────────────────────────────
# Estimated jointly: beta (linear, here the price coefficient), sigma (the
# diagonal RC standard deviations), pi (demographic interactions, K2 x D).
# A stage with no random coefficients passes empty sigma/pi; a stage with no
# demographics passes an empty (K2, 0) pi.


from emu_gmm.manifolds import Euclidean  # noqa: E402
from emu_gmm.manifolds.manifold_leaf import ManifoldLeaf  # noqa: E402

# Parameters are estimated jointly and live on flat Euclidean space (no
# constraints; sigma's sign is not identified so we don't impose positivity).
# emu's manifold-aware flatten requires non-scalar leaves wrapped in a
# ManifoldLeaf carrying their manifold, so each field is a ManifoldLeaf over
# the appropriate Euclidean block. ``leaf(arr)`` builds an unconstrained one.


def euclidean_leaf(arr) -> ManifoldLeaf:
    arr = jnp.asarray(arr, dtype=jnp.float64)
    return ManifoldLeaf(arr, Euclidean(*arr.shape))


@jdc.pytree_dataclass
class LogitParams:
    """Plain logit: just the linear coefficient(s) (price)."""

    beta: ManifoldLeaf


@jdc.pytree_dataclass
class RCParams:
    """Random-coefficients logit, no demographics: beta + diagonal sigma."""

    beta: ManifoldLeaf
    sigma: ManifoldLeaf


@jdc.pytree_dataclass
class BLPParams:
    """Full BLP: linear beta, diagonal sigma, demographic interactions pi."""

    beta: ManifoldLeaf
    sigma: ManifoldLeaf
    pi: ManifoldLeaf


# ── Share system (one market) ─────────────────────────────────────────────


def individual_mu(
    X2: Float[Array, "J K2"],
    nodes: Float[Array, "I K2"],
    demographics: Float[Array, "I D"],
    sigma: Float[Array, " K2"],
    pi: Float[Array, "K2 D"],
) -> Float[Array, "I J"]:
    """Agent-by-product taste deviation mu_ij = sum_r X2[j,r] * coef[i,r].

    coef[i,r] = sigma[r] * nodes[i,r] + sum_d pi[r,d] demographics[i,d].
    """
    coef = nodes * sigma[None, :]  # (I, K2)
    if pi.shape[1] > 0:
        coef = coef + demographics @ pi.T  # (I, K2)
    return coef @ X2.T  # (I, J)


def predicted_shares(
    delta: Float[Array, " J"],
    mu: Float[Array, "I J"],
    agent_weights: Float[Array, " I"],
) -> Float[Array, " J"]:
    """Mixed-logit predicted shares with an outside good (utility 0).

    s_j = sum_i w_i * exp(delta_j + mu_ij) / (1 + sum_k exp(delta_k + mu_ik)).
    Numerically stable via the log-sum-exp over [0, u_i].
    """
    u = delta[None, :] + mu  # (I, J)
    # denom_i = 1 + sum_j exp(u_ij); include the outside option as a 0 column.
    z = jnp.concatenate([jnp.zeros((u.shape[0], 1)), u], axis=1)  # (I, J+1)
    log_denom = jax.scipy.special.logsumexp(z, axis=1)  # (I,)
    probs = jnp.exp(u - log_denom[:, None])  # (I, J)
    return agent_weights @ probs  # (J,)


# ── Contraction with implicit-function-theorem derivative ──────────────────


@jax.custom_jvp
def solve_delta(
    mu: Float[Array, "I J"],
    log_shares: Float[Array, " J"],
    agent_weights: Float[Array, " I"],
) -> Float[Array, " J"]:
    """Berry (1994) contraction for the mean utilities delta(theta2).

    Solves s(delta, mu) = exp(log_shares) by iterating
    delta <- delta + log_shares - log s(delta). Differentiated only through
    ``mu`` (the channel theta2 enters); ``log_shares`` / ``agent_weights``
    are data. The custom JVP below supplies the IFT derivative so the
    non-differentiable ``while_loop`` never needs to be traced through.
    """
    delta0 = log_shares  # logit-style start (outside utility 0)

    def cond(state):
        _delta, i, err = state
        return (err > CONTRACTION_TOL) & (i < CONTRACTION_MAXITER)

    def body(state):
        delta, i, _err = state
        pred = predicted_shares(delta, mu, agent_weights)
        delta_new = delta + log_shares - jnp.log(pred)
        err = jnp.max(jnp.abs(delta_new - delta))
        return (delta_new, i + 1, err)

    delta, _, _ = jax.lax.while_loop(cond, body, (delta0, 0, jnp.inf))
    return delta


@solve_delta.defjvp
def _solve_delta_jvp(primals, tangents):
    mu, log_shares, agent_weights = primals
    dmu, _, _ = tangents  # log_shares / agent_weights are data (zero tangent)

    delta = solve_delta(mu, log_shares, agent_weights)

    # Implicit function theorem on F(delta, mu) = s(delta, mu) - S_obs = 0:
    #   ds/ddelta * ddelta + ds/dmu * dmu = 0  =>  ddelta = -(ds/ddelta)^-1 ds/dmu dmu
    def shares_of_delta(d):
        return predicted_shares(d, mu, agent_weights)

    J_delta = jax.jacfwd(shares_of_delta)(delta)  # (J, J)
    # Directional derivative ds/dmu . dmu at fixed delta.
    _, s_mu_dmu = jax.jvp(
        lambda m: predicted_shares(delta, m, agent_weights), (mu,), (dmu,)
    )
    ddelta = -jnp.linalg.solve(J_delta, s_mu_dmu)
    return delta, ddelta


def logit_delta(log_shares: Float[Array, " J"], outside_log_share: float) -> Float[Array, " J"]:
    """Closed-form logit mean utility: delta_j = log s_j - log s_0."""
    return log_shares - outside_log_share


# ── emu-gmm interface: the per-market psi factory ──────────────────────────


@dataclass(frozen=True)
class BLPData:
    """Market-grouped arrays for one Nevo spec (dense, balanced panel)."""

    shares: jnp.ndarray  # (T, J)
    prices: jnp.ndarray  # (T, J)  (X1; K1=1)
    X1: jnp.ndarray  # (T, J, K1)
    X2: jnp.ndarray  # (T, J, K2)
    Z: jnp.ndarray  # (T, J, M)  (brand-demeaned instruments)
    nodes: jnp.ndarray  # (T, I, K2)
    demographics: jnp.ndarray  # (T, I, D)
    agent_weights: jnp.ndarray  # (T, I)

    @property
    def T(self):
        return self.shares.shape[0]

    @property
    def J(self):
        return self.shares.shape[1]

    @property
    def M(self):
        return self.Z.shape[2]

    @property
    def K1(self):
        return self.X1.shape[2]

    @property
    def K2(self):
        return self.X2.shape[2]

    @property
    def D(self):
        return self.demographics.shape[2]


def solve_all_deltas(sigma, pi, data: "BLPData") -> Float[Array, "T J"]:
    """Mean utilities delta_t(theta2) for every market (vmapped contraction)."""
    log_sh = jnp.log(data.shares)  # (T, J)

    def one(t):
        mu = individual_mu(
            data.X2[t], data.nodes[t], data.demographics[t], sigma, pi
        )
        return solve_delta(mu, log_sh[t], data.agent_weights[t])

    return jax.vmap(one)(jnp.arange(data.T))


def make_psi(data: BLPData, *, model_type: str, pi_free_positions=None):
    """Build the emu ``psi(x, theta)`` for a Nevo spec.

    The observation is a **product-market** (N = T*J rows, matching pyblp's
    moment structure). ``x`` is the flat product-market row index n; the row's
    data is read from closure arrays flattened in (T, J) row-major order, so
    row n corresponds to market n//J, product n%J. ``psi`` returns the M-vector
    Z_n * xi_n; emu's mean over the N rows is exactly pyblp's sample moment
    g = (1/N) sum_jt Z_jt xi_jt.

    Because the per-market contraction (``solve_all_deltas``) does not depend on
    the per-row index, ``vmap`` over rows traces it **once** --- the all-markets
    delta is computed a single time per psi / jacobian evaluation and then
    gathered to the N rows.

    ``model_type`` is 'logit' (closed-form delta, no contraction) or 'rc'.

    ``pi_free_positions`` (rows, cols) lists the free demographic-interaction
    entries; the rest of the (K2, D) pi matrix is fixed at zero (pyblp's
    convention: zero initial values are not estimated). ``theta.pi`` then holds
    only the free entries, scattered into the zero template here.
    """
    sh = jnp.asarray(data.shares)
    X1N = jnp.asarray(data.X1).reshape(data.T * data.J, data.K1)  # (N, K1)
    ZN = jnp.asarray(data.Z).reshape(data.T * data.J, data.M)  # (N, M)
    outside_log = jnp.log(1.0 - sh.sum(axis=1))  # (T,)
    log_sh = jnp.log(sh)  # (T, J)

    def all_deltas_flat(theta):
        if model_type == "logit":
            delta = logit_delta(log_sh, outside_log[:, None])  # (T, J)
        else:
            sigma = theta.sigma.array
            pi_leaf = getattr(theta, "pi", None)
            if pi_leaf is None:
                pi = jnp.zeros((sigma.shape[0], 0))
            elif pi_free_positions is not None:
                rows, cols = pi_free_positions
                pi = jnp.zeros((data.K2, data.D)).at[rows, cols].set(pi_leaf.array)
            else:
                pi = pi_leaf.array
            delta = solve_all_deltas(sigma, pi, data)  # (T, J)
        return delta.reshape(-1)  # (N,)

    def psi(x, theta):
        n = x[0].astype(jnp.int32)
        delta_flat = all_deltas_flat(theta)  # (N,) -- vmap traces this once
        xi = delta_flat[n] - X1N[n] @ theta.beta.array  # scalar
        return ZN[n] * xi  # (M,)

    return psi


def product_index_x(T: int, J: int) -> jnp.ndarray:
    """The emu observation array: one row per product-market, holding its index."""
    return jnp.arange(T * J, dtype=jnp.float64)[:, None]
