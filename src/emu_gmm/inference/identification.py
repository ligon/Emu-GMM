r"""Per-parameter-block identification-strength diagnostic (#177).

A constructive answer to "*which* parameter block is weakly identified?" —
computed off the same pipeline as the estimate and under the result's own
:class:`~emu_gmm.types.CovarianceStrategy`, so a consumer no longer has to
hand-roll a criterion-landscape scan plus an FWL partial-first-stage
regression to find the weak block (the K-Aggregators motivation: "level and
``Gamma`` strongly identified; the heterogeneity contrast strong-in-first-stage
but parameterization-flat").

What it measures
----------------
The GMM information (curvature) matrix is

.. math::
    \mathcal I \;=\; G'\,\Lambda\,G, \qquad \Lambda = (V^\star)^{-1},

the same matrix :func:`emu_gmm.diagnostics.compute_cond_info` reports the
condition number of and the same one the :math:`\Sigma_\theta` bread inverts
(CLAUDE.md commitment 5 — built directly, never as a numerical Hessian, via
:func:`emu_gmm.diagnostics.information_matrix`). Under the *efficient* metric
:math:`\Lambda = (V^\star)^{-1}` this is a property of the moment system and
design — **not** of the weighting the analyst happened to estimate under — so
it is the right object for "how strongly is this block identified", and it
matches what ``cond_info`` already uses.

For a block of coordinates :math:`b` (with complement :math:`c`) the per-block
strength is the **Schur complement** — the curvature of the criterion in the
:math:`b` directions *after concentrating out* the other coordinates:

.. math::
    \mathcal I_{b\cdot c}
    \;=\; \mathcal I_{bb} - \mathcal I_{bc}\,\mathcal I_{cc}^{-1}\,\mathcal I_{cb}.

This is the FWL / partial-first-stage analogue: the eigenvalues of
:math:`\mathcal I_{b\cdot c}` are the (sample-size-scaled, since
:math:`V^\star \sim 1/N`) **concentration parameters** of block :math:`b`, and
its inverse is exactly the block-:math:`b` sub-block of the (efficient)
:math:`\Sigma_\theta` (the block-inverse identity), so a weak block has small
:math:`\min\mathrm{eig}(\mathcal I_{b\cdot c})` and a large marginal variance.
The smallest identified eigenvalue is the *binding* concentration parameter
for the block — the direction that drives the weak-identification-robust /
Wald divergence (#41's K-statistic): where it is small, the Wald CI from
:math:`\Sigma_\theta` understates uncertainty while the K-statistic stays
valid.

Gauge-bearing blocks (manifolds)
--------------------------------
For a :class:`~emu_gmm.manifolds.PSDFixedRank` factor ``A`` (``Gamma = A A'``)
the moment function is exactly gauge-invariant, so the ``gauge_dim`` orbit
directions are an exact nullspace of :math:`\mathcal I` *within that leaf's
coordinate block*. The diagnostic drops them **by count** — the same
:func:`~emu_gmm._internal.pinv_eigvalrule.pinv_eigvalrule` rule the
:math:`\Sigma_\theta` bread and ``cond_info['exclude_gauge']`` use — on both
sides: when a gauge-bearing leaf sits in the *complement*, its
:math:`\mathcal I_{cc}` is inverted with that leaf's ``gauge_dim`` smallest
eigenvalues pinned out; the *target* block reports the
``len(indices) - gauge_dim`` identified eigenvalues only. The gauge directions
of a block survive the Schur complement as exact zeros (``I`` annihilates them
on both factors), so dropping by count is exact, never a magnitude threshold.
A gauge-bearing leaf must therefore be wholly inside or wholly outside each
block — splitting one across blocks has no well-defined per-block gauge count
and is refused.

Eager-only: call outside any ``jax.jit`` boundary (it returns a host-side
record keyed by block name; the underlying linear algebra is plain JAX).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from typing import Any

import jax.numpy as jnp
import numpy as np
import pandas as pd
from jaxtyping import Array, Float

from emu_gmm._internal.params import manifold_spec_from_params
from emu_gmm._internal.pinv_eigvalrule import pinv_eigvalrule
from emu_gmm.diagnostics import information_matrix
from emu_gmm.regularization import DiagonalTikhonov
from emu_gmm.types import (
    EstimationResult,
    RegularizationStrategy,
    StructuralModel,
)


def _to_plain(value: Any) -> Float[Array, "..."]:
    """Strip a :class:`haliax.NamedArray` wrapper if present, else ``asarray``.

    Mirrors :func:`emu_gmm.inference.k_statistic._to_plain`: tests
    ``hasattr(value, "array")`` rather than coercing blindly, so a plain JAX
    array passes straight through.
    """
    return jnp.asarray(getattr(value, "array", value))


@dataclasses.dataclass(frozen=True)
class BlockStrength:
    """Identification strength of one parameter block (#177).

    Attributes
    ----------
    name
        The block label (a dataclass field name for the default leaf
        blocking, or the caller's key for an explicit ``blocks`` mapping).
    indices
        The ambient tangent indices this block spans, into the same axis
        ``Sigma_theta`` / :attr:`~emu_gmm.types.EstimationResult.standard_errors`
        / ``coef_table`` are sized by (``total_dimension``).
    dim
        The *identified* dimension ``len(indices) - gauge_dim`` — the number
        of reported eigenvalues.
    gauge_dim
        Exact gauge-nullspace directions contained in this block
        (``k(k-1)/2`` per fully-contained :class:`PSDFixedRank` leaf; ``0``
        for a Euclidean / scalar block). Dropped by count, never reported as
        weak identification.
    eigenvalues
        The ``dim`` identified eigenvalues of the Schur-complemented
        information matrix :math:`\\mathcal I_{b\\cdot c}` (the per-direction
        concentration parameters), ascending. ``eigenvalues[0]`` is the
        binding (weakest) direction.
    min_eigenvalue
        ``eigenvalues[0]`` as a 0-d array — the block's concentration
        parameter; the headline scalar for ranking blocks by strength.
    partial_information
        The full ``(len(indices), len(indices))`` Schur complement
        :math:`\\mathcal I_{b\\cdot c}` (gauge directions included as exact
        zeros), for callers who want the matrix rather than its spectrum.
    """

    name: str
    indices: tuple[int, ...]
    dim: int
    gauge_dim: int
    eigenvalues: Float[Array, " d"]
    min_eigenvalue: Float[Array, ""]
    partial_information: Float[Array, "b b"]


@dataclasses.dataclass(frozen=True)
class IdentificationStrength:
    """Per-block identification-strength report (#177).

    Returned by :func:`identification_strength`. Maps each block name to its
    :class:`BlockStrength`; indexable by name (``result_strength["c_slopes"]``)
    and iterable over ``(name, BlockStrength)`` items via :meth:`items`.

    Attributes
    ----------
    blocks
        Ordered mapping ``{name: BlockStrength}`` in the order the blocks were
        supplied (leaf-walk order for the default blocking).
    metric
        Human-readable note on the metric :math:`\\Lambda` used to form the
        information matrix (``"(V*)^{-1}"`` — the efficient metric).
    """

    blocks: dict[str, BlockStrength]
    metric: str = "(V*)^{-1}"

    def __getitem__(self, name: str) -> BlockStrength:
        return self.blocks[name]

    def __iter__(self) -> Any:
        return iter(self.blocks)

    def items(self) -> Any:
        return self.blocks.items()

    @property
    def weakest(self) -> str:
        """Name of the block with the smallest concentration parameter.

        ``argmin`` over each block's :attr:`BlockStrength.min_eigenvalue`. A
        ``NaN`` min-eigenvalue (a numerically singular block) sorts as the
        weakest — the honest answer that the block is effectively
        unidentified. Raises if the report is empty.
        """
        if not self.blocks:
            raise ValueError("IdentificationStrength.weakest: no blocks")

        def _key(item: tuple[str, BlockStrength]) -> float:
            v = float(jnp.asarray(item[1].min_eigenvalue))
            return -np.inf if np.isnan(v) else v

        return min(self.blocks.items(), key=_key)[0]

    def to_pandas(self) -> pd.DataFrame:
        """Block summary as a DataFrame indexed by block name.

        Columns: ``dim``, ``gauge_dim``, ``min_eigenvalue`` (the concentration
        parameter), ``max_eigenvalue``. Rows in the report's block order.
        """
        rows: dict[str, list[float]] = {
            "dim": [],
            "gauge_dim": [],
            "min_eigenvalue": [],
            "max_eigenvalue": [],
        }
        index: list[str] = []
        for name, blk in self.blocks.items():
            index.append(name)
            ev = np.asarray(blk.eigenvalues)
            rows["dim"].append(blk.dim)
            rows["gauge_dim"].append(blk.gauge_dim)
            rows["min_eigenvalue"].append(float(ev[0]) if ev.size else np.nan)
            rows["max_eigenvalue"].append(float(ev[-1]) if ev.size else np.nan)
        return pd.DataFrame(rows, index=index)


def _leaf_index_ranges(manifold_spec: Any) -> list[tuple[Any, range, int]]:
    """``(leaf_spec, ambient index range, gauge_dim)`` per leaf, in walk order."""
    out = []
    for ls in manifold_spec.leaf_specs:
        size = int(np.prod(ls.ambient_shape)) if ls.ambient_shape != () else 1
        out.append((ls, range(ls.offset, ls.offset + size), int(ls.manifold.gauge_dim)))
    return out


def _subset_gauge_dim(
    index_set: frozenset[int],
    leaf_ranges: list[tuple[Any, range, int]],
    *,
    where: str,
) -> int:
    """Gauge dim contributed by gauge-bearing leaves fully inside ``index_set``.

    Raises if a gauge-bearing leaf is *split* by ``index_set`` (partially in,
    partially out): the gauge nullspace of a :class:`PSDFixedRank` leaf is
    spanned by vectors supported on that leaf's coordinates, so a split block
    has no well-defined per-block gauge count and the drop-by-count rule
    cannot apply.
    """
    total = 0
    for ls, rng, gauge in leaf_ranges:
        if gauge == 0:
            continue
        leaf_idx = set(rng)
        inside = leaf_idx & index_set
        if not inside:
            continue
        if inside != leaf_idx:
            field = getattr(ls, "field_name", None) or "<unnamed>"
            raise ValueError(
                f"identification_strength: gauge-bearing leaf {field!r} "
                f"(ambient indices {rng.start}..{rng.stop - 1}, gauge_dim "
                f"{gauge}) is split across the {where} — a gauge-bearing "
                "leaf (e.g. a PSDFixedRank factor) must sit wholly inside or "
                "wholly outside each block, so its gauge directions can be "
                "dropped by count. Group the whole leaf into one block."
            )
        total += gauge
    return total


def _default_blocks(manifold_spec: Any) -> dict[str, tuple[int, ...]]:
    """One block per parameter leaf, labelled by field name (fallback positional)."""
    blocks: dict[str, tuple[int, ...]] = {}
    for i, (ls, rng, _gauge) in enumerate(_leaf_index_ranges(manifold_spec)):
        name = getattr(ls, "field_name", None) or f"leaf_{i}"
        # Disambiguate a duplicate / missing field name positionally.
        if name in blocks:
            name = f"{name}_{i}"
        blocks[name] = tuple(rng)
    return blocks


def _block_strength(
    name: str,
    indices: tuple[int, ...],
    info: Float[Array, "D D"],
    leaf_ranges: list[tuple[Any, range, int]],
    total_dim: int,
) -> BlockStrength:
    """Schur-complement partial information + identified eigenvalues for one block."""
    idx = np.asarray(sorted(set(indices)), dtype=int)
    block_set = frozenset(int(i) for i in idx)
    comp = np.asarray([i for i in range(total_dim) if i not in block_set], dtype=int)

    gauge_b = _subset_gauge_dim(block_set, leaf_ranges, where=f"block {name!r}")

    I_bb = info[jnp.asarray(idx)[:, None], jnp.asarray(idx)[None, :]]
    if comp.size == 0:
        partial = I_bb
    else:
        comp_set = frozenset(int(i) for i in comp)
        gauge_c = _subset_gauge_dim(
            comp_set, leaf_ranges, where=f"complement of block {name!r}"
        )
        I_cc = info[jnp.asarray(comp)[:, None], jnp.asarray(comp)[None, :]]
        I_bc = info[jnp.asarray(idx)[:, None], jnp.asarray(comp)[None, :]]
        # Gauge-aware inverse of the complement curvature: pin out the
        # complement's gauge directions BY COUNT (pinv_eigvalrule), so a
        # gauge-bearing leaf in the complement does not register as a
        # singular (inf) complement.
        I_cc_pinv = pinv_eigvalrule(I_cc, drop_smallest=gauge_c)
        partial = I_bb - I_bc @ I_cc_pinv @ I_bc.T

    partial = 0.5 * (partial + partial.T)
    db = int(idx.shape[0])
    if db - gauge_b < 1:
        raise ValueError(
            f"identification_strength: block {name!r} has dimension {db} but "
            f"{gauge_b} gauge directions, leaving no identified directions. A "
            "block must contain at least one non-gauge coordinate."
        )
    # eigvalsh: ascending. Drop the gauge_b smallest (exact zeros) BY COUNT;
    # the trailing block is the identified spectrum (concentration parameters).
    eigs = jnp.linalg.eigvalsh(partial)
    eigs_id = eigs[gauge_b:]
    return BlockStrength(
        name=name,
        indices=tuple(int(i) for i in idx),
        dim=db - gauge_b,
        gauge_dim=gauge_b,
        eigenvalues=eigs_id,
        min_eigenvalue=eigs_id[0],
        partial_information=partial,
    )


def identification_strength(
    result: EstimationResult,
    model: StructuralModel,
    *,
    blocks: Mapping[str, Sequence[int]] | None = None,
    regularization: RegularizationStrategy | None = None,
    V_star: Float[Array, "M M"] | None = None,
) -> IdentificationStrength:
    r"""Per-block identification strength (concentration / partial first stage).

    For each parameter block, returns the eigenvalues of the
    Schur-complemented GMM information matrix
    :math:`\mathcal I_{b\cdot c} = \mathcal I_{bb} - \mathcal I_{bc}\,
    \mathcal I_{cc}^{-1}\,\mathcal I_{cb}` with
    :math:`\mathcal I = G'(V^\star)^{-1}G` evaluated at ``result.theta_hat``
    under the result's own covariance strategy — the per-direction
    concentration parameters of block :math:`b` after concentrating out the
    other coordinates (the FWL / partial-first-stage analogue). See the module
    docstring for the full derivation and the gauge handling.

    Parameters
    ----------
    result
        A fitted :class:`~emu_gmm.types.EstimationResult`. Supplies
        ``theta_hat``, the ``measure`` / ``covariance`` provenance, the
        ``regularization`` (when present), and the ``manifold_spec`` (for the
        gauge bookkeeping and the default per-leaf blocking).
    model
        The per-observation residual ``psi(x, theta) -> R^M``. Passed
        separately because it is a callable the result does not store (same
        convention as :func:`emu_gmm.inference.k_statistic`).
    blocks
        Optional mapping ``{name: indices}`` partitioning the ambient tangent
        axis (the ``Sigma_theta`` / ``coef_table`` axis) into named blocks;
        ``indices`` are integer positions into that axis. When ``None`` (the
        default) one block per parameter leaf is used, labelled by the
        dataclass field name — exactly the ``θ = (level, c-slopes, Γ)``
        decomposition the K-Aggregators consumer wants. A gauge-bearing leaf
        (e.g. a :class:`PSDFixedRank` factor) must lie wholly within one block.
    regularization
        Regulariser used to form :math:`V^\star` from
        ``covariance.covariance(model, theta_hat, measure)``. Defaults to the
        result's own strategy, or :class:`~emu_gmm.regularization.DiagonalTikhonov`
        when the result carries none. Ignored when ``V_star`` is supplied.
    V_star
        Optional pre-computed regularised variance to reuse the exact ridge
        frozen during :func:`emu_gmm.estimate` (bypasses the
        ``covariance`` / ``regularization`` recompute).

    Returns
    -------
    :class:`IdentificationStrength`
        The per-block report; ``.weakest`` names the block with the smallest
        concentration parameter, ``.to_pandas()`` summarises.

    Notes
    -----
    The metric is the *efficient* :math:`\Lambda = (V^\star)^{-1}`,
    independent of the weighting actually used to estimate, so the diagnostic
    measures identification of the moment system itself (consistent with
    ``cond_info``). Eager-only.
    """
    theta_hat = result.theta_hat
    measure = result.measure
    covariance = result.covariance

    manifold_spec = result.manifold_spec
    if manifold_spec is None:
        manifold_spec = manifold_spec_from_params(theta_hat)
    leaf_ranges = _leaf_index_ranges(manifold_spec)
    total_dim = int(manifold_spec.total_dimension)

    # Ambient moment-Jacobian at theta_hat (the (M, total_dimension) layout
    # Sigma_theta is sized by). For a gauge-invariant model the ambient G
    # annihilates the vertical directions, so the information matrix carries
    # the exact gauge zeros — same precondition compute_cond_info relies on.
    G = _to_plain(measure.jacobian(model, theta_hat))

    if V_star is None:
        if regularization is None:
            regularization = result.regularization or DiagonalTikhonov()
        V = _to_plain(covariance.covariance(model, theta_hat, measure))
        V_star_arr, _tau = regularization.apply(V)
    else:
        V_star_arr = jnp.asarray(V_star)

    info = information_matrix(G, V_star_arr)
    D = int(info.shape[-1])
    if D != total_dim:
        raise ValueError(
            f"identification_strength: information matrix dimension {D} does "
            f"not match the parameter ambient dimension {total_dim} from the "
            "manifold spec — a Jacobian / spec routing mismatch."
        )

    if blocks is None:
        block_map = _default_blocks(manifold_spec)
    else:
        block_map = {name: tuple(int(i) for i in idx) for name, idx in blocks.items()}
        _validate_block_indices(block_map, total_dim)

    report = {
        name: _block_strength(name, indices, info, leaf_ranges, total_dim)
        for name, indices in block_map.items()
    }
    return IdentificationStrength(blocks=report)


def _validate_block_indices(
    block_map: Mapping[str, Sequence[int]], total_dim: int
) -> None:
    """Each index in range; no empty block; no index in two blocks."""
    seen: dict[int, str] = {}
    for name, indices in block_map.items():
        if len(indices) == 0:
            raise ValueError(f"identification_strength: block {name!r} is empty.")
        for i in indices:
            if not 0 <= i < total_dim:
                raise ValueError(
                    f"identification_strength: block {name!r} index {i} is "
                    f"outside [0, {total_dim})."
                )
            if i in seen and seen[i] != name:
                raise ValueError(
                    f"identification_strength: ambient index {i} appears in "
                    f"both block {seen[i]!r} and block {name!r}; blocks must "
                    "be disjoint."
                )
            seen[i] = name


__all__ = [
    "identification_strength",
    "IdentificationStrength",
    "BlockStrength",
]
