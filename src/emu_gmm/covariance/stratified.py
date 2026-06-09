r"""Design-based (Neyman) between-PSU stratified variance and the
law-of-total-variance design/sampling assembly.

This module ships two strategies:

``StratifiedCovariance`` (emu-gmm #79)
    The design-exact between-primary-sampling-unit (PSU) Neyman variance
    :math:`V_{TT}` for a stratified experiment. Centering happens *within*
    each cell :math:`c = (\text{stratum} \times \text{arm})`, so the
    stratum-additive form falls out automatically (cross-cell products are
    never formed). Each covariance entry :math:`(j, k)` uses the
    *pair-specific* effective PSU count :math:`H_{c,jk}` --- the PSUs that
    support **both** coordinates --- mirroring the available-pairs
    :math:`N_j N_k` overlap rule already used by
    :class:`~emu_gmm.covariance.iid.IIDCovariance` and
    :class:`~emu_gmm.covariance.clustered.ClusteredCovariance` (see
    ``docs/design.org`` "Empirical Covariance Strategies" and CLAUDE.md
    commitment 9). This is the single detail most easily dropped in a
    hand-rolled port.

``DesignAwareCovariance`` (emu-gmm #80)
    The law-of-total-variance assembly
    :math:`V_X = V_{TT} + V_{SS} + V_{TS}` for a moment vector mixing
    randomized (design-driven, :math:`z_T`) and sampled (:math:`z_S`)
    instruments. A **composition** of strategies: :math:`V_{TT}` delegated
    to a shared :class:`StratifiedCovariance` (the design-exact block),
    :math:`V_{SS}` to a shared :class:`ClusteredCovariance` (uncentered
    cluster-total), and the off-diagonal :math:`V_{TS}` computed inline ---
    *estimated, never zeroed* (``ai_neyman_covariance.org`` eq:offdiag).
    When every coordinate is design-driven it reduces bit-for-bit to
    :math:`V_{TT}` via the shared engine. PSD is not guaranteed (the
    design-exact :math:`V_{TT}` is glued to the sampling blocks);
    ``DiagonalTikhonov`` handles repair.

Scaling convention
------------------
Like every empirical strategy, the output is on the variance-of-the-mean
scale: :math:`V_X = N_X \big(\sum_c \Omega_c\big) N_X` with
:math:`N_X = \operatorname{diag}(1/N_j)` and
:math:`N_j = \sum_i d_{ij} w_i`. There is **no** explicit :math:`N` or
:math:`\sqrt N` anywhere; all sample-size and per-moment-overlap
bookkeeping lives in the :math:`1/(N_j N_k)` normalisation. Adding such a
factor "to match a textbook" is the canonical bug commitment 9 guards
against.
"""

from __future__ import annotations

from typing import Any

import haliax as ha
import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
from jaxtyping import Array, Float

from emu_gmm._internal.nan_safety import safe_x_for_psi
from emu_gmm.covariance.clustered import ClusteredCovariance
from emu_gmm.types import ParamsLike, StructuralModel


def _to_plain(value: Any) -> jnp.ndarray:
    """Strip a haliax :class:`NamedArray` wrapper, returning the array."""
    if isinstance(value, ha.NamedArray):
        return value.array
    return jnp.asarray(value)


def _safe_outer_divide(
    numer: Float[Array, "M M"],
    denom_vec: Float[Array, " M"],
) -> Float[Array, "M M"]:
    """Return ``numer / (denom_vec outer denom_vec)`` with zero on degeneracy.

    Mirrors the helper in :mod:`emu_gmm.covariance.iid` /
    :mod:`emu_gmm.covariance.clustered`: a coordinate with
    :math:`N_j = 0` contributes a zero row/column rather than ``inf`` /
    ``nan``.
    """
    denom = jnp.outer(denom_vec, denom_vec)
    safe = jnp.where(denom == 0.0, 1.0, denom)
    out = numer / safe
    return jnp.where(denom == 0.0, jnp.zeros_like(out), out)


@jdc.pytree_dataclass
class StratifiedCovariance:
    r"""Design-based between-PSU Neyman stratified variance (emu-gmm #79).

    .. math::
        V_X &= N_X \Big( \sum_c \Omega_c \Big) N_X, \qquad
            N_X = \operatorname{diag}(1 / N_j), \\
        [\Omega_c]_{jk} &= \frac{H_{c,jk}}{H_{c,jk} - 1}
            \sum_{g \in c} u_{g,jk}\,
            (t_{g,j} - \bar t^{(jk)}_{c,j})(t_{g,k} - \bar t^{(jk)}_{c,k}),

    where :math:`t_{g,j} = \sum_{i \in g} d_{ij} w_i \psi_j(x_i, \theta)`
    is the PSU total of coordinate :math:`j`,
    :math:`s_{g,j} = \mathbf 1\{\sum_{i \in g} d_{ij} > 0\}` is the support
    indicator, :math:`u_{g,jk} = s_{g,j} s_{g,k}` is the pairwise overlap
    weight, :math:`H_{c,jk} = \sum_{g \in c} u_{g,jk}` is the **pair-specific**
    effective PSU count, and the centering mean
    :math:`\bar t^{(jk)}_{c,j}` averages :math:`t_{g,j}` over exactly the
    :math:`(j,k)`-overlap PSUs. The centered sum is evaluated by the
    sum-of-products identity
    :math:`\sum_g u (t_j - A/H)(t_k - B/H) = P - A B / H` with
    :math:`A = \sum_g u t_j`, :math:`B = \sum_g u t_k`,
    :math:`P = \sum_g u t_j t_k`.

    A cell with :math:`H_{c,jk} < 2` cannot furnish a between-PSU variance
    and contributes exactly zero to entry :math:`(j,k)`. This is the
    design-side analogue of the available-pairs / pairwise-overlap
    :math:`N_j N_k` rule used by :class:`IIDCovariance` and
    :class:`ClusteredCovariance`.

    Parameters
    ----------
    psu_ids : (N,) jax array of floats
        Per-observation PSU (primary sampling unit / group) index in
        ``[0, n_psu)`` --- the i.i.d. unit under the design. Float dtype
        for traced compatibility; rounded and cast to int32 inside
        :func:`jax.ops.segment_sum`.
    cell_ids : (N,) jax array of floats
        Per-observation cell index in ``[0, n_cells)``, where a cell is a
        ``(stratum x arm)`` combination --- the centering unit. **Must**
        be the stratum-by-arm cell, not the stratum: centering on the bare
        stratum would pool arms and inject the treatment effect into the
        variance.
    stratum_ids : (N,) jax array of floats
        Per-observation stratum index in ``[0, n_strata)``. Used only by
        the finite-population correction (``fpc=True``); pass zeros when
        ``fpc`` is off.
    n_psu, n_cells, n_strata : int (static)
        Segment counts. ``n_psu`` must equal ``max(psu_ids) + 1`` for
        0-based contiguous ids (``segment_sum`` silently drops ids
        ``>= num_segments``).
    fpc : bool (static), default ``False``
        Apply the :math:`(1 - H_{sD}/H_s)` finite-population correction
        (off by default; conservative). The numerator is the
        coordinate-independent design assignment count :math:`H_{sD}` --- the
        SAME per-cell scalar for every :math:`(j,k)`, not the per-pair overlap
        --- because the FPC is a property of the assignment, not of
        observability (which already enters via the dof and centering). See
        :meth:`_fpc_factor` and ``docs/design.org``. The FPC belongs to the
        design estimand (no masking); the masked GMM path runs ``fpc=False``.
    """

    psu_ids: Float[Array, " N"]
    cell_ids: Float[Array, " N"]
    stratum_ids: Float[Array, " N"]
    n_psu: int = jdc.static_field()  # type: ignore[attr-defined]
    n_cells: int = jdc.static_field()  # type: ignore[attr-defined]
    n_strata: int = jdc.static_field()  # type: ignore[attr-defined]
    fpc: bool = jdc.static_field(default=False)  # type: ignore[attr-defined]

    def covariance(
        self,
        psi: StructuralModel,
        theta: ParamsLike,
        measure: Any,
        cached_intermediates: (
            tuple[
                Float[Array, " M"],
                Float[Array, "N M"],
                Float[Array, "N M"],
                Float[Array, " M"],
            ]
            | None
        ) = None,
    ) -> Float[Array, "M M"]:
        r"""Construct :math:`V_X(\theta)` via the per-pair Neyman sandwich.

        Parameters
        ----------
        psi : :data:`StructuralModel`
            Per-observation residual function.
        theta : :data:`ParamsLike`
            User parameter dataclass.
        measure
            An :class:`~emu_gmm.measures.empirical.EmpiricalMeasure`
            exposing ``x``, ``mask``, and ``weights``.
        cached_intermediates : optional 4-tuple
            ``(m, psi_safe, weight_mask, N_j)`` from
            :meth:`EmpiricalMeasure.expectation_and_contributions`. When
            supplied, ``psi_safe`` and ``N_j`` are reused (the cached
            ``psi_safe`` is produced by the *same* ``safe_x_for_psi`` pass
            as the self-computing branch below, so the two paths are
            bit-identical by construction --- see
            ``tests/covariance/test_stratified.py`` parity test).

        Returns
        -------
        V : (M, M) jax array
            Symmetric. PSD with complete data; may be indefinite under
            unequal per-coordinate support, exactly as the pairwise-overlap
            :class:`IIDCovariance` can be --- the regularization layer
            (:class:`DiagonalTikhonov`) handles the finite-sample
            non-PD risk. This routine performs **no** internal PD repair.
        """
        mask = measure.mask  # (N, M)
        weights = measure.weights  # (N,)

        if cached_intermediates is not None:
            # Tuple order is the house cached-intermediates contract
            # (m, psi_safe, weight_mask, N_j) from
            # EmpiricalMeasure.expectation_and_contributions. The estimator
            # dispatches by sniffing the ``cached_intermediates`` kwarg name
            # via inspect.signature (it is not part of the CovarianceStrategy
            # Protocol), so neither the kwarg name nor this unpacking order
            # may be reordered without updating that producer in lockstep.
            _m, psi_safe, _weight_mask, N_j = cached_intermediates
        else:
            # Pre-sanitise so a singular psi (log / 1/x / sqrt) cannot
            # introduce NaN/Inf at masked-out cells and poison reverse-mode
            # AD. The cached producer (expectation_and_contributions) runs
            # this same safe_x_for_psi pass, so cached == self-compute.
            x_safe = safe_x_for_psi(measure.x)

            def psi_at(x: Any) -> jnp.ndarray:
                return _to_plain(psi(x, theta))

            psi_batch = jax.vmap(psi_at)(x_safe)  # (N, M)
            mask_bool = mask > 0.0
            psi_safe = jnp.where(mask_bool, psi_batch, 0.0)  # (N, M)
            N_j = jnp.sum(mask * weights[:, None], axis=0)  # (M,)

        # Per-observation weighted contribution and support. Both are built
        # from measure.mask/weights in BOTH branches: this is a
        # cluster-total family (like ClusteredCovariance), so w_i enters
        # ONCE per unit (do NOT square weights as IIDCovariance does).
        contrib = mask * weights[:, None] * psi_safe  # (N, M)  d_ij w_i psi_j
        support_unit = (mask > 0.0).astype(psi_safe.dtype)  # (N, M)  d_ij

        # --- aggregate to PSU totals and per-(PSU, coord) support --------
        psu_seg = jnp.round(self.psu_ids).astype(jnp.int32)
        psu_tot = jax.ops.segment_sum(
            contrib, psu_seg, num_segments=self.n_psu
        )  # (n_psu, M)
        psu_sup = jax.ops.segment_sum(
            support_unit, psu_seg, num_segments=self.n_psu
        )  # (n_psu, M)
        s = (psu_sup > 0.0).astype(psu_tot.dtype)  # s_{g,j}, (n_psu, M)

        # PSU -> cell membership (constant within a PSU). MUST be the
        # stratum-by-arm cell, never the bare stratum. An unpopulated PSU
        # slot (an id in [0, n_psu) with no observation) takes segment_max's
        # empty-segment fill (-inf -> INT32_MIN on the int32 cast); clip into
        # the valid cell range so it can never become a garbage segment id or
        # an out-of-bounds gather. Such a PSU carries zero support (s = 0),
        # so it contributes exactly zero wherever it lands -- the clip is
        # behaviour-preserving for dense ids and removes the reliance on
        # segment_sum silently dropping the garbage index.
        cell_of_psu = jnp.clip(
            jax.ops.segment_max(
                jnp.round(self.cell_ids), psu_seg, num_segments=self.n_psu
            ).astype(jnp.int32),
            0,
            self.n_cells - 1,
        )  # (n_psu,)

        # --- per-cell, per-pair Neyman cross-product (O(M^2) vectorised) -
        # Per-PSU building blocks, with st_{g,j} = s_{g,j} t_{g,j}:
        st = s * psu_tot  # (n_psu, M)
        ss = s[:, :, None] * s[:, None, :]  # u_{g,jk} = s_gj s_gk
        P_blk = st[:, :, None] * st[:, None, :]  # u t_gj t_gk
        A_blk = st[:, :, None] * s[:, None, :]  # u t_gj
        B_blk = s[:, :, None] * st[:, None, :]  # u t_gk

        H = jax.ops.segment_sum(ss, cell_of_psu, num_segments=self.n_cells)
        A = jax.ops.segment_sum(A_blk, cell_of_psu, num_segments=self.n_cells)
        B = jax.ops.segment_sum(B_blk, cell_of_psu, num_segments=self.n_cells)
        P = jax.ops.segment_sum(P_blk, cell_of_psu, num_segments=self.n_cells)
        # all (n_cells, M, M)

        # Within-overlap centered cross-product P - A B / H (NaN-safe at H=0).
        H_safe = jnp.where(H == 0.0, 1.0, H)
        centered = P - A * B / H_safe

        # Bessel dof H/(H-1); pairs with H < 2 contribute exactly 0.
        Hm1_safe = jnp.where(H < 2.0, 1.0, H - 1.0)
        dof = H / Hm1_safe
        valid = (H >= 2.0).astype(H.dtype)
        cell_term = valid * dof * centered  # (n_cells, M, M)

        if self.fpc:
            # Coordinate-independent (per-cell) FPC scalar, broadcast over (M, M).
            cell_term = cell_term * self._fpc_factor(psu_seg, cell_of_psu)

        numer = jnp.sum(cell_term, axis=0)  # (M, M)
        V = _safe_outer_divide(numer, N_j)
        return 0.5 * (V + V.T)  # symmetrise against round-off

    def _fpc_factor(
        self,
        psu_seg: Float[Array, " N"],
        cell_of_psu: Float[Array, " n_psu"],
    ) -> Float[Array, "n_cells 1 1"]:
        r"""Finite-population correction :math:`1 - H_{sD}/H_s` (convention (ii)).

        The correction is a property of the without-replacement *assignment* of
        groups to cells, **not** of per-coordinate observability, so it is the
        SAME per-cell scalar applied to every entry :math:`(j,k)`:

        .. math::
            1 - \frac{H_{sD,c}}{H_s},

        with :math:`H_{sD,c}` the populated-PSU count assigned to cell :math:`c`
        and :math:`H_s` the stratum-total populated-PSU count (across all arms).
        The per-pair overlap :math:`H_{c,jk}` is deliberately **not** used in the
        numerator (the rejected convention (i)): per-coordinate observability
        already enters through the per-pair dof :math:`H_{c,jk}/(H_{c,jk}-1)` and
        the centering, so a per-pair numerator would double-count it and
        perversely *shrink* the correction for sparser pairs. The denominator is
        likewise the stratum-total (a per-pair denominator would collapse to
        :math:`H_{c,jk}` for a single-arm coordinate and zero its variance).
        Resolved deliberately; see ``docs/design.org`` and the appendix
        (Seasonality PR #23). ``fpc=False`` remains the default and the masked
        GMM path; the FPC belongs to the design estimand where there is no
        masking.

        Returns a ``(n_cells, 1, 1)`` array that broadcasts identically over the
        ``(M, M)`` block of each cell.
        """
        # Populated-PSU indicator: an empty / over-declared slot must not be
        # counted (and its segment_max stratum/cell fill must not corrupt a real
        # one). We weight every count by ``populated`` and clip the segment_max
        # id fills into range (harmless: empty slots are weighted out).
        n_obs_psu = jax.ops.segment_sum(
            jnp.ones_like(psu_seg, dtype=jnp.float64),
            psu_seg,
            num_segments=self.n_psu,
        )  # (n_psu,) observations per PSU
        populated = (n_obs_psu > 0.0).astype(jnp.float64)  # (n_psu,)

        # H_{sD,c}: populated PSUs assigned to each cell (the assignment count).
        H_sD_cell = jax.ops.segment_sum(
            populated, cell_of_psu, num_segments=self.n_cells
        )  # (n_cells,)

        # H_s: populated PSUs per stratum (all arms), mapped back to each cell.
        strat_of_psu = jnp.clip(
            jax.ops.segment_max(
                jnp.round(self.stratum_ids), psu_seg, num_segments=self.n_psu
            ).astype(jnp.int32),
            0,
            self.n_strata - 1,
        )  # (n_psu,)
        H_s_by_stratum = jax.ops.segment_sum(
            populated, strat_of_psu, num_segments=self.n_strata
        )  # (n_strata,)
        strat_of_cell = jnp.clip(
            jax.ops.segment_max(
                strat_of_psu, cell_of_psu, num_segments=self.n_cells
            ).astype(jnp.int32),
            0,
            self.n_strata - 1,
        )  # (n_cells,)
        H_s_cell = H_s_by_stratum[strat_of_cell]  # (n_cells,)
        H_s_safe = jnp.where(H_s_cell == 0.0, 1.0, H_s_cell)
        fpc_c = 1.0 - H_sD_cell / H_s_safe  # (n_cells,) coordinate-independent
        return fpc_c[:, None, None]


@jdc.pytree_dataclass
class DesignAwareCovariance:
    r"""Mixed design/sampling covariance via law of total variance (#80).

    For a moment vector mixing randomized instruments :math:`z_T`
    (design-driven) and sampled instruments :math:`z_S`, conditioning on
    the sample and applying the law of total variance gives

    .. math::
        V_X = \underbrace{\mathbb E_S[\operatorname{Var}_W(\bar m_X \mid S)]}_{V^{\mathrm{des}}}
            + \underbrace{\operatorname{Var}_S[\mathbb E_W(\bar m_X \mid S)]}_{V^{\mathrm{smp}}}.

    The estimator is a **composition** of covariance strategies: :math:`V_{TT}`
    (design coords) is delegated to a shared :class:`StratifiedCovariance` ---
    the design-exact known-:math:`\pi` Neyman block --- :math:`V_{SS}` (sampled
    coords) to a shared :class:`ClusteredCovariance` (uncentered cluster-total
    form), and only the off-diagonal coupling :math:`V_{TS}` is computed inline
    as a cluster-robust cross pass, **estimated, not zeroed**
    (``ai_neyman_covariance.org`` eq:offdiag). See :meth:`covariance` for the
    assembly, the shared-:math:`N_j` scale, and the all-design bit-for-bit
    reduction.

    .. note::
        The design-exact :math:`V_{TT}` is **glued** to the sampling
        cross/:math:`V_{SS}` blocks (not the literal :math:`V^{\mathrm{des}} +
        V^{\mathrm{smp}}` sum), so the result is **not** guaranteed PSD --- it
        can be indefinite under missingness / few PSUs exactly as
        :class:`StratifiedCovariance` can. No internal PD repair is done; the
        regularization layer (:class:`DiagonalTikhonov`) handles it. This is the
        *coupled* composition a plain additive sum cannot express (the cross
        corner is non-zero).

    Parameters
    ----------
    design : StratifiedCovariance
        The :math:`V_{TT}` engine. Held as a **shared reference**, never
        copied, so the all-design reduction is identical to calling it
        directly. Carries the (optional) design FPC, which enters
        :math:`V_{TT}` only --- never :math:`V_{SS}` / :math:`V_{TS}`.
    sampling : ClusteredCovariance
        The :math:`V_{SS}` engine (uncentered cluster-total form). Its
        ``cluster_ids`` / ``n_clusters`` are **also** the clustering unit for
        the inline :math:`V_{TS}` cross pass. Because :math:`z_T` and
        :math:`z_S` live in different arms, build ``sampling`` with
        **stratum-level** clustering to capture the within-stratum cross-arm
        term of eq:offdiag. Its ``dof_correction`` flag is likewise
        **inherited by the cross pass** (#119, resolved convention): with
        ``dof_correction=True`` every cross pair :math:`(j, k)` is inflated
        by the per-pair :math:`G_{jk}/(G_{jk}-1)` counted over the sampling
        clusters supporting both coordinates, exactly as inside
        :math:`V_{SS}`. The design FPC, by contrast, stays inside
        :math:`V_{TT}` only.
    design_moment_mask : (M,) jax array of floats
        ``1.0`` for design-driven (:math:`z_T`) coordinates, ``0.0`` for
        sampled (:math:`z_S`) coordinates. Traced leaf.
    all_design : bool (static)
        ``True`` iff every coordinate is design-driven. Computed at
        construction by :meth:`from_design_mask`; gates the Python-level
        dispatch between the bit-for-bit reduction and the mixed composition.
    couple : bool (static), default ``True``
        Whether to include the off-diagonal :math:`V_{TS}` cross block (#109).
        ``True`` (default) is the full law-of-total-variance assembly
        :math:`V_{TT} + V_{SS} + V_{TS}`. ``couple=False`` zeroes the cross
        corners, giving the **block-diagonal** :math:`V_{TT} \oplus V_{SS}` on
        the same coords / shared :math:`N_j` scale --- the :math:`V_{TS}=0`
        counterfactual for the cross-block *materiality* diagnostic (compare
        the J-statistic with and without the coupling). The estimated cross
        block itself is also exposed directly via :meth:`cross_block`. When
        ``all_design`` there are no sampled coords, so :math:`V_{TS}\equiv 0`
        and ``couple`` has no effect.
    """

    design: StratifiedCovariance
    sampling: ClusteredCovariance
    design_moment_mask: Float[Array, " M"]
    all_design: bool = jdc.static_field()  # type: ignore[attr-defined]
    couple: bool = jdc.static_field(default=True)  # type: ignore[attr-defined]

    @classmethod
    def from_design_mask(
        cls,
        design: StratifiedCovariance,
        sampling: ClusteredCovariance,
        design_moment_mask: Any,
        couple: bool = True,
    ) -> "DesignAwareCovariance":
        """Build, computing the static ``all_design`` flag eagerly.

        ``all_design`` must be a concrete Python bool (it drives a
        Python-level branch in :meth:`covariance`), so it is evaluated here
        at construction time rather than under trace.

        Pass ``couple=False`` for the block-diagonal :math:`V_{TT}\\oplus
        V_{SS}` assembly (the :math:`V_{TS}=0` counterfactual, #109); the
        default ``couple=True`` is the full coupled :math:`V_{TT}+V_{SS}+
        V_{TS}`.
        """
        m = jnp.asarray(design_moment_mask)
        all_design = bool(jnp.all(m > 0.0))
        return cls(
            design=design,
            sampling=sampling,
            design_moment_mask=m,
            all_design=all_design,
            couple=couple,
        )

    def covariance(
        self,
        psi: StructuralModel,
        theta: ParamsLike,
        measure: Any,
        cached_intermediates: (
            tuple[
                Float[Array, " M"],
                Float[Array, "N M"],
                Float[Array, "N M"],
                Float[Array, " M"],
            ]
            | None
        ) = None,
    ) -> Float[Array, "M M"]:
        r"""Assemble the mixed design/sampling :math:`V_X`.

        Composition of the law-of-total-variance blocks
        :math:`V_X = V_{TT} + V_{SS} + V_{TS}` (+ :math:`V_{ST}`), placed into
        the full :math:`(M, M)` matrix by outer-product masks of
        :attr:`design_moment_mask` on a single shared :math:`N_j` Var(mean)
        scale:

        - :math:`V_{TT}` (design-driven coords) is **delegated** to the shared
          :attr:`design` engine --- the design-exact known-:math:`\pi` Neyman
          block (the lower-variance payoff). The child owns its residual
          convention.
        - :math:`V_{SS}` (sampled coords) is **delegated** to the shared
          :attr:`sampling` engine (the uncentered cluster-total form).
        - :math:`V_{TS}` (the cross coupling) is computed here as a
          cluster-robust cross pass at :attr:`sampling`'s cluster unit
          (caller-controlled) and is **estimated, never zeroed**
          (``ai_neyman_covariance.org`` eq:offdiag).

        When every coordinate is design-driven (``all_design``) this reduces
        bit-for-bit to :meth:`StratifiedCovariance.covariance` via the shared
        engine. With ``couple=False`` the :math:`V_{TS}` term is dropped,
        yielding the block-diagonal :math:`V_{TT}\oplus V_{SS}` counterfactual
        (#109).

        PSD is *not* guaranteed: the design-exact :math:`V_{TT}` glued to the
        sampling cross/:math:`V_{SS}` blocks can be indefinite under
        missingness / few PSUs, exactly as :class:`StratifiedCovariance` can.
        No internal PD repair is performed; the regularization layer
        (:class:`DiagonalTikhonov`) handles it.
        """
        if self.all_design:
            return self.design.covariance(psi, theta, measure, cached_intermediates)

        # --- delegated diagonal blocks (compose; cached tuple threads through)
        V_TT = self.design.covariance(psi, theta, measure, cached_intermediates)
        V_SS = self.sampling.covariance(psi, theta, measure, cached_intermediates)

        # --- assemble via design_moment_mask outer-product masks
        # (exhaustive and mutually exclusive: P_TT + P_SS + P_TS == 1) -----
        t = (self.design_moment_mask > 0.0).astype(V_TT.dtype)  # (M,) design
        s = 1.0 - t  # (M,) sampled
        P_TT = jnp.outer(t, t)
        P_SS = jnp.outer(s, s)

        V = V_TT * P_TT + V_SS * P_SS
        if self.couple:
            # + V_TS (+ V_ST): the estimated off-diagonal cross block. Omitted
            # for couple=False -> block-diagonal V_TT (+) V_SS (#109). The
            # term is added last so couple=True is bit-identical to the
            # pre-#109 assembly V_TT*P_TT + V_SS*P_SS + V_cross*P_TS.
            V = V + self._cross_corners(psi, theta, measure, cached_intermediates)
        return 0.5 * (V + V.T)  # symmetrise against round-off

    def _cross_corners(
        self,
        psi: StructuralModel,
        theta: ParamsLike,
        measure: Any,
        cached_intermediates: (
            tuple[
                Float[Array, " M"],
                Float[Array, "N M"],
                Float[Array, "N M"],
                Float[Array, " M"],
            ]
            | None
        ) = None,
    ) -> Float[Array, "M M"]:
        r"""Estimated cross block :math:`V_{TS}` placed in the (T,S)/(S,T) corners.

        A cluster-total outer product over **all** coords at the
        :attr:`sampling` engine's cluster unit, masked to the off-diagonal
        cross corners by ``P_TS = outer(t,s) + outer(s,t)`` (the TT and SS
        blocks are zeroed). One shared :math:`N_j` divides every entry, so a
        cross-pair :math:`(j,k)` is scaled by exactly :math:`1/(N_j N_k)` ---
        the same scale as :math:`V_{TT}` / :math:`V_{SS}`. The intermediates
        are unpacked from the cache or self-computed with the same
        ``safe_x_for_psi`` sanitisation the child engines use, so cached and
        self-compute agree by construction. Used by :meth:`covariance` (when
        ``couple``) and exposed via :meth:`cross_block`.
        """
        mask = measure.mask  # (N, M)
        weights = measure.weights  # (N,)
        if cached_intermediates is not None:
            _m, psi_safe, _weight_mask, N_j = cached_intermediates
        else:
            x_safe = safe_x_for_psi(measure.x)

            def psi_at(x: Any) -> jnp.ndarray:
                return _to_plain(psi(x, theta))

            psi_batch = jax.vmap(psi_at)(x_safe)  # (N, M)
            psi_safe = jnp.where(mask > 0.0, psi_batch, 0.0)  # (N, M)
            N_j = jnp.sum(mask * weights[:, None], axis=0)  # (M,)

        contrib = mask * weights[:, None] * psi_safe  # (N, M)  d_ij w_i psi_j
        seg = jnp.round(self.sampling.cluster_ids).astype(jnp.int32)
        cluster_totals = jax.ops.segment_sum(
            contrib, seg, num_segments=self.sampling.n_clusters
        )  # (n_clusters, M)
        numer_cross = jnp.einsum("cj,ck->jk", cluster_totals, cluster_totals)
        if self.sampling.dof_correction:
            # Resolved convention (#119): the cross block INHERITS the
            # sampling engine's per-pair finite-cluster correction
            # G_jk/(G_jk-1), with G_jk counted over the sampling clusters
            # holding an observed unit for BOTH coordinates of the cross
            # pair -- symmetric with the V_SS treatment and consistent
            # with the per-pair philosophy of CLAUDE.md commitments 9/10.
            # Same helper, same mask, same cluster unit as the V_SS
            # delegate, so the corrected assembly stays internally
            # consistent; dof_correction=False (the default) leaves this
            # pass bit-for-bit unchanged.
            numer_cross = numer_cross * self.sampling._finite_cluster_correction(mask)
        V_cross = _safe_outer_divide(numer_cross, N_j)  # (M, M)

        t = (self.design_moment_mask > 0.0).astype(V_cross.dtype)  # (M,) design
        s = 1.0 - t  # (M,) sampled
        P_TS = jnp.outer(t, s) + jnp.outer(s, t)
        return V_cross * P_TS

    def cross_block(
        self,
        psi: StructuralModel,
        theta: ParamsLike,
        measure: Any,
        cached_intermediates: (
            tuple[
                Float[Array, " M"],
                Float[Array, "N M"],
                Float[Array, "N M"],
                Float[Array, " M"],
            ]
            | None
        ) = None,
    ) -> Float[Array, "M M"]:
        r"""Return the estimated off-diagonal cross block :math:`V_{TS}` alone (#109).

        The :math:`(M, M)` matrix carrying the estimated design/sampling cross
        coupling in its (T,S) and (S,T) corners and **zeros** on the TT and SS
        blocks, on the same shared :math:`N_j` Var(mean) scale as
        :meth:`covariance`. By construction

        .. math::
            \text{covariance}(\text{couple=True})
            = \text{covariance}(\text{couple=False}) + \text{cross\_block},

        so :math:`\lVert\text{cross\_block}\rVert / \lVert V_{TT}\oplus
        V_{SS}\rVert` quantifies the cross-block *materiality* without
        hand-rolling a block-diagonal wrapper downstream. When ``all_design``
        there are no sampled coords, so the cross block is exactly zero.

        Parameters and ``cached_intermediates`` semantics match
        :meth:`covariance`.
        """
        if self.all_design:
            M = int(self.design_moment_mask.shape[0])
            return jnp.zeros((M, M), dtype=self.design_moment_mask.dtype)
        cross = self._cross_corners(psi, theta, measure, cached_intermediates)
        return 0.5 * (cross + cross.T)  # symmetrise against round-off


__all__ = ["StratifiedCovariance", "DesignAwareCovariance"]
