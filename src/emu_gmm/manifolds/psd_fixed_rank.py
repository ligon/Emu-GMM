r"""JAX-native :class:`PSDFixedRank` (plan §2.1, Phase 2).

Quotient manifold of :math:`n \times n` PSD matrices of rank :math:`k`,
parameterised as :math:`X = Y Y^\top` with :math:`Y \in
\mathbb{R}^{n \times k}` and :math:`Y \sim YQ` for :math:`Q \in O(k)`.

This file mirrors :file:`pymanopt/manifolds/psd.py` (the
:class:`pymanopt.manifolds.PSDFixedRank` reference) under three
constraints:

1. Pure JAX (no ``scipy.linalg.solve_continuous_lyapunov``). The
   Lyapunov solve in :meth:`projection` is formulated in Kronecker form
   :math:`(I_k \otimes A + A^\top \otimes I_k) \mathrm{vec}(\Omega) =
   \mathrm{vec}(B)` and solved by :func:`jax.numpy.linalg.solve` on a
   :math:`k^2 \times k^2` system. Cost :math:`O(k^6)`; negligible at
   :math:`k \le 3`.
2. Ambient storage for tangent vectors: an ambient ``(n, k)`` array, the
   same shape as ``Y``. The horizontal projection is applied inside
   :meth:`projection`; the result still lives in the ``(n, k)`` shape.
   Plan §2.1 explains why ambient storage is the right v2.0 choice
   (jit/vmap-friendly, pymanopt parity is array-to-array).
3. ``gauge_dim = k * (k - 1) // 2``: the dimension of the gauge nullspace
   (the orthogonal group :math:`O(k)`'s tangent space at identity is
   the skew-symmetric :math:`k \times k` matrices). The information
   dimension :math:`\dim_{\mathrm{info}} = n k - k(k-1)/2` is what Phase
   5 / Phase 6 surfaces; here we just publish ``gauge_dim`` as a class
   attribute and let downstream code consume it.

Caveat (per plan §2-probe Probe C): at :math:`k \to n` the gauge
fraction grows. Above :math:`k/n \approx 0.7` the spectral-gap
diagnostic becomes unreliable and the user should consider
horizontal-basis storage (deferred to v2.1).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp


class PSDFixedRank:
    r"""Quotient manifold of :math:`n \times n` PSD matrices of rank :math:`k`.

    Parameters
    ----------
    n
        Side of the ambient symmetric matrix :math:`X = Y Y^\top`.
    k
        Rank of :math:`X`. Must satisfy ``1 <= k <= n``.

    Notes
    -----
    The protocol attributes follow plan §2.1:

    * ``ambient_shape == (n, k)``,
    * ``dimension == n * k`` (the *ambient* :math:`nk`, not the quotient),
    * ``gauge_dim == k * (k - 1) // 2``.

    The *information dimension* :math:`n k - k(k-1)/2` is what an
    asymptotic info matrix has rank equal to; the rank assertion is
    surfaced by Phase 5/6 code that consumes ``manifold.dimension -
    manifold.gauge_dim``.
    """

    def __init__(self, n: int, k: int) -> None:
        if not (1 <= int(k) <= int(n)):
            raise ValueError(f"PSDFixedRank(n={n}, k={k}): expected 1 <= k <= n")
        self._n: int = int(n)
        self._k: int = int(k)
        # Annotated at the ManifoldParam protocol's width: protocol
        # *attribute* members are invariant under mypy, so the narrower
        # ``tuple[int, int]`` made PSDFixedRank fail isinstance-level
        # protocol conformance (#122). The value is still always (n, k).
        self.ambient_shape: tuple[int, ...] = (self._n, self._k)
        self.dimension: int = self._n * self._k
        self.gauge_dim: int = self._k * (self._k - 1) // 2

    # ------------------------------------------------------------------
    # Hash / equality / repr.
    # ------------------------------------------------------------------
    def __hash__(self) -> int:
        return hash(("PSDFixedRank", self._n, self._k))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PSDFixedRank):
            return NotImplemented
        return self._n == other._n and self._k == other._k

    def __repr__(self) -> str:
        return f"PSDFixedRank(n={self._n}, k={self._k})"

    # ------------------------------------------------------------------
    # ManifoldParam operators.
    # ------------------------------------------------------------------
    def projection(self, point: Any, ambient_vector: Any) -> Any:
        r"""Horizontal projection of an ambient vector at ``point``.

        Mirrors :file:`pymanopt/manifolds/psd.py:30-34`. With ``Y =
        point`` and ``V = ambient_vector``, we compute

        .. math::

            Y^\top Y \Omega + \Omega (Y^\top Y) = Y^\top V - V^\top Y

        for a skew-symmetric :math:`\Omega`, then return :math:`V - Y
        \Omega`. The Lyapunov solve uses the Kronecker form solved by
        :func:`jax.numpy.linalg.solve`.
        """
        Y = jnp.asarray(point)
        V = jnp.asarray(ambient_vector)
        YtY = Y.T @ Y
        AS = Y.T @ V - V.T @ Y  # skew-symmetric k x k matrix
        Omega = _solve_continuous_lyapunov_kron(YtY, AS)
        return V - Y @ Omega

    def retraction(self, point: Any, tangent_vector: Any) -> Any:
        r"""First-order retraction :math:`R_Y(V) = Y + V`.

        Pymanopt's :meth:`PSDFixedRank.retraction` is also just
        ``point + tangent_vector`` (see :file:`pymanopt/manifolds/psd.py`
        lines 46-49). The manifold is closed under addition as long as
        the rank of :math:`Y + V` does not drop; step-size control is
        the optimiser's job.
        """
        return jnp.asarray(point) + jnp.asarray(tangent_vector)

    def retraction_differential(self, point: Any) -> Any:  # noqa: ARG002
        r"""Retraction differential ``1`` (additive retraction ``Y + V``)."""
        del point
        return jnp.asarray(1.0)

    def riemannian_gradient(
        self, point: Any, euclidean_gradient: Any
    ) -> Any:  # noqa: ARG002
        r"""Identity (per :file:`pymanopt/manifolds/psd.py:38-40`)."""
        return euclidean_gradient

    def euclidean_to_riemannian_gradient(
        self, point: Any, euclidean_gradient: Any
    ) -> Any:  # noqa: ARG002
        r"""Phase-4 canonical name; identity under the embedded metric.

        (The horizontal projection is applied on the tangent vector
        during :meth:`projection` / retraction, not on the gradient.)
        """
        del point
        return euclidean_gradient

    def inner_product(self, point: Any, u: Any, v: Any) -> Any:  # noqa: ARG002
        r"""Embedded Frobenius inner product :math:`\langle U, V\rangle_F`."""
        del point
        return jnp.sum(jnp.asarray(u) * jnp.asarray(v))

    def norm(self, point: Any, tangent_vector: Any) -> Any:
        r"""Frobenius norm ``sqrt(inner_product(V, V))``."""
        return jnp.sqrt(self.inner_product(point, tangent_vector, tangent_vector))

    def distance(self, point_a: Any, point_b: Any) -> Any:
        r"""Geodesic distance via the manifold logarithm.

        Mirrors :file:`pymanopt/manifolds/psd.py:24-25` + ``log`` at
        lines 51-53. Computes :math:`\log_A B = B U V^\top - A` where
        :math:`U \Sigma V^\top = B^\top A`, then takes the Frobenius
        norm.
        """
        A = jnp.asarray(point_a)
        B = jnp.asarray(point_b)
        u, _, vh = jnp.linalg.svd(B.T @ A, full_matrices=False)
        log_AB = B @ u @ vh - A
        return jnp.linalg.norm(log_AB)

    def random_point(self, key: Any) -> Any:
        r"""Draw a random ``(n, k)`` standard-normal matrix."""
        return jax.random.normal(key, self.ambient_shape, dtype=jnp.float64)

    def tangent_basis_names(self, field_name: str) -> list[str]:
        r"""Return :math:`n k` ambient-coordinate labels.

        For ``PSDFixedRank(n, k)`` with field name ``L`` and matrix
        index ``(i, j)`` (row, column), the label is
        ``"L_t_<i><j>"``. We use the run-together index format
        (``"L_t_00"``, ``"L_t_01"``) when both indices stay in 0--9; for
        larger shapes we fall back to underscore-separated indices
        (``"L_t_10_2"``) to avoid ambiguity.
        """
        labels: list[str] = []
        single_digit = self._n <= 10 and self._k <= 10
        for i in range(self._n):
            for j in range(self._k):
                if single_digit:
                    labels.append(f"{field_name}_t_{i}{j}")
                else:
                    labels.append(f"{field_name}_t_{i}_{j}")
        return labels

    def invariants(self) -> dict[str, Callable[[Any], Any]]:
        r"""Canonical gauge-invariant functionals of an ambient ``(n, k)`` factor ``A``.

        :math:`\Gamma = A A^\top` is *the* :math:`O(k)`-quotient invariant (the
        rank-``k`` PSD matrix; unchanged under :math:`A \to A Q`, :math:`Q \in
        O(k)`), so its spectrum and its ``vech`` are the natural queryable
        summaries of a leaf on this manifold. Each functional maps the leaf's
        ambient ``(n, k)`` array to a 1-D array and is gauge-invariant, hence
        meaningful both per draw (empirical grade) and under the delta method
        (asymptotic grade) --- the raw ``(n, k)`` entries themselves are
        gauge-arbitrary and are deliberately NOT offered.

        * ``"eigenvalues"`` --- the ``k`` nonzero eigenvalues of :math:`\Gamma`,
          ascending (the ``n - k`` structural zeros, a degenerate repeated
          eigenvalue, are excluded).
        * ``"gamma"`` --- ``vech(Gamma)``, the ``n(n+1)/2`` unique
          lower-triangular entries in row-major order.
        """
        n, k = self._n, self._k
        ii, jj = jnp.tril_indices(n)

        def eigenvalues(A: Any) -> Any:
            arr = jnp.asarray(A)
            return jnp.linalg.eigvalsh(arr @ arr.T)[n - k :]

        def gamma(A: Any) -> Any:
            arr = jnp.asarray(A)
            return (arr @ arr.T)[ii, jj]

        return {"eigenvalues": eigenvalues, "gamma": gamma}


# ---------------------------------------------------------------------------
# Kronecker-form continuous Lyapunov solve.
# ---------------------------------------------------------------------------


def _solve_continuous_lyapunov_kron(A: jnp.ndarray, B: jnp.ndarray) -> jnp.ndarray:
    r"""Solve :math:`A X + X A^\top = B` for a square :math:`A` and right-hand side :math:`B`.

    Used by :meth:`PSDFixedRank.projection` with :math:`A = Y^\top Y`
    (symmetric PD) and :math:`B = Y^\top V - V^\top Y` (skew-symmetric).
    The Kronecker form :math:`(I_k \otimes A + A \otimes I_k)\,
    \mathrm{vec}(X) = \mathrm{vec}(B)` is solved by
    :func:`jax.numpy.linalg.solve` on a :math:`k^2 \times k^2` system.

    With :math:`A` symmetric (the only case we use), the operator
    :math:`A \otimes I + I \otimes A` coincides with :math:`A^\top
    \otimes I + I \otimes A`, so the formula reduces to the symmetric-
    Lyapunov form scipy implements. We match scipy's
    ``solve_continuous_lyapunov`` exactly when both inputs are real and
    :math:`A = A^\top`.
    """
    A = jnp.asarray(A)
    B = jnp.asarray(B)
    k = A.shape[-1]
    eye_k = jnp.eye(k, dtype=A.dtype)
    # Kronecker form: vec(AX + XA^T) = (I_k kron A + A kron I_k) vec(X).
    M = jnp.kron(eye_k, A) + jnp.kron(A, eye_k)
    x = jnp.linalg.solve(M, B.reshape(-1))
    return x.reshape(k, k)


__all__ = ["PSDFixedRank"]
