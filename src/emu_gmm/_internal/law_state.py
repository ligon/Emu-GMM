r"""Low-level carriers + manifold codec for law persistence (#181).

Shared, dependency-light building blocks for the typed/versioned persistence of
an :class:`~emu_gmm.law.EstimatorLaw` (issue #181). Kept here, importing neither
``emu_gmm.law`` nor ``emu_gmm.persistence``, so both can depend on it without a
cycle:

- :class:`MomentsBacking` --- the moments-only backing an
  :class:`~emu_gmm.law.AsymptoticLaw` queries when it has no live
  :class:`~emu_gmm.types.EstimationResult` (``from_moments`` / a reloaded law).
- :func:`manifold_to_tag` / :func:`tag_to_manifold` --- the typed-tag codec that
  is the key to #147: a ``PSDFixedRank`` leaf persists as
  ``{"type": "PSDFixedRank", "n": 5, "k": 2}`` + its array, and reconstructs
  through the normal constructor on load --- never touching the immutable live
  ``ManifoldLeaf``.
- :func:`locate_psd_leaf` --- the per-leaf PSD-factor detection (the same
  "unique ``PSDFixedRank`` leaf" rule #117 uses), so the reloaded law can answer
  ``eigenvalue_se`` / ``gamma_se``.

The on-disk container (a single ``.npz`` + JSON manifest) and the public
:class:`~emu_gmm.persistence.LawState` schema live in
:mod:`emu_gmm.persistence`.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np

#: Bumped when the persisted layout changes; ``load_law`` validates it and
#: routes through an upgrader table (trivial at v1) rather than guessing.
SCHEMA_VERSION = 1


def euclidean_tag(arr: Any) -> dict[str, Any]:
    """A ``Euclidean`` manifold tag matching the ambient shape of ``arr``.

    Used for an all-Euclidean / v1 result whose ``manifold_spec`` is ``None``:
    each component is a plain (scalar or vector) Euclidean leaf, so its tag is
    just its shape. A 0-d scalar leaf -> ``{"type": "Euclidean", "shape": []}``.
    """
    return {"type": "Euclidean", "shape": [int(s) for s in np.shape(arr)]}


def manifold_to_tag(manifold: Any) -> dict[str, Any]:
    """Serialise a manifold instance to a typed, JSON-able tag.

    Supports the persistable manifold menu (``Euclidean`` / ``PSDFixedRank`` /
    ``Positive`` / ``Interval``). The tag carries the constructor arguments, so
    :func:`tag_to_manifold` rebuilds an equal instance --- no live object or
    class reference is stored on disk.
    """
    from emu_gmm.manifolds.euclidean import Euclidean
    from emu_gmm.manifolds.interval import Interval
    from emu_gmm.manifolds.positive import Positive
    from emu_gmm.manifolds.psd_fixed_rank import PSDFixedRank

    if isinstance(manifold, PSDFixedRank):
        n, k = (int(s) for s in manifold.ambient_shape)
        return {"type": "PSDFixedRank", "n": n, "k": k}
    if isinstance(manifold, Euclidean):
        return {"type": "Euclidean", "shape": [int(s) for s in manifold.ambient_shape]}
    if isinstance(manifold, Positive):
        return {"type": "Positive"}
    if isinstance(manifold, Interval):
        return {"type": "Interval", "lo": float(manifold.lo), "hi": float(manifold.hi)}
    raise TypeError(
        f"manifold_to_tag: unsupported manifold {type(manifold).__name__}; law "
        "persistence supports Euclidean / PSDFixedRank / Positive / Interval. "
        "Add a tag case (and a tag_to_manifold branch) for a new manifold type."
    )


def tag_to_manifold(tag: dict[str, Any]) -> Any:
    """Inverse of :func:`manifold_to_tag`: rebuild the manifold via its constructor."""
    from emu_gmm.manifolds.euclidean import Euclidean
    from emu_gmm.manifolds.interval import Interval
    from emu_gmm.manifolds.positive import Positive
    from emu_gmm.manifolds.psd_fixed_rank import PSDFixedRank

    t = tag.get("type")
    if t == "PSDFixedRank":
        return PSDFixedRank(int(tag["n"]), int(tag["k"]))
    if t == "Euclidean":
        return Euclidean(*[int(s) for s in tag["shape"]])
    if t == "Positive":
        return Positive()
    if t == "Interval":
        return Interval(float(tag["lo"]), float(tag["hi"]))
    raise ValueError(
        f"tag_to_manifold: unknown manifold tag type {t!r}. The artifact may be "
        "from a newer emu_gmm with an unsupported manifold; upgrade emu_gmm."
    )


def locate_psd_leaf(leaf_specs: Any) -> tuple[int | None, int | None]:
    """``(component_index, rank)`` of the unique ``PSDFixedRank`` leaf, or ``(None, None)``.

    ``leaf_specs`` is a per-component tuple of manifold instances (aligned with
    the components tuple). Mirrors the #117 rule used by
    :meth:`EstimationResult._gamma_leaf`: exactly one ``PSDFixedRank`` factor is
    the canonical ``Gamma`` source; zero -> no Gamma (eigenvalue/gamma queries
    unavailable); more than one -> a typed error (no canonical Gamma).
    """
    if leaf_specs is None:
        return None, None
    from emu_gmm.manifolds.psd_fixed_rank import PSDFixedRank

    idxs = [i for i, m in enumerate(leaf_specs) if isinstance(m, PSDFixedRank)]
    if not idxs:
        return None, None
    if len(idxs) > 1:
        raise TypeError(
            f"locate_psd_leaf: {len(idxs)} PSDFixedRank leaves (component "
            f"indices {idxs}); there is no canonical Gamma. Use functional_se "
            "with a functional that selects the intended factor."
        )
    i = idxs[0]
    return i, int(leaf_specs[i].ambient_shape[1])  # (n, k) -> k


@dataclasses.dataclass(frozen=True)
class MomentsBacking:
    """Moments-only state an :class:`AsymptoticLaw` queries without a live result.

    Carries exactly what the delta-method queries need --- the component arrays,
    :math:`\\Sigma_\\theta`, the ambient labels, the typed leaf tags (for
    re-save fidelity), and the persisted PSD-leaf ``(index, rank)`` so the
    gauge-invariant ``eigenvalue_se`` / ``gamma_se`` conveniences work. Pure
    arrays + plain Python: no JAX pytree, no ``ManifoldLeaf``.
    """

    components: tuple[np.ndarray, ...]
    sigma: np.ndarray
    names: tuple[str, ...]
    leaf_tags: tuple[dict[str, Any], ...]
    component_shapes: tuple[tuple[int, ...], ...]
    psd_index: int | None
    psd_rank: int | None
    diagnostics: dict[str, Any] = dataclasses.field(default_factory=dict)
    # The typed FactorySpec (emu_gmm.persistence) rides opaquely here so this
    # _internal carrier stays decoupled from the public persistence type.
    factory_spec: Any = None

    @classmethod
    def build(
        cls,
        *,
        components: Any,
        sigma: Any,
        names: tuple[str, ...] | None,
        leaf_tags: Any,
        psd_index: int | None,
        psd_rank: int | None,
        diagnostics: dict[str, Any] | None = None,
        factory_spec: dict[str, Any] | None = None,
    ) -> MomentsBacking:
        """Validate + normalise the moments state (shapes, names length, D)."""
        comps = tuple(np.asarray(c, dtype=float) for c in components)
        sig = np.atleast_2d(np.asarray(sigma, dtype=float))
        shapes = tuple(tuple(int(s) for s in np.shape(c)) for c in comps)
        flat_dim = int(sum(int(np.prod(s)) if s != () else 1 for s in shapes))
        if sig.shape != (flat_dim, flat_dim):
            raise ValueError(
                f"MomentsBacking: sigma has shape {sig.shape} but the flattened "
                f"components imply ambient dimension D={flat_dim}; the two must "
                "match."
            )
        if names is None:
            nm = tuple(f"theta_{i}" for i in range(flat_dim))
        else:
            nm = tuple(names)
            if len(nm) != flat_dim:
                raise ValueError(
                    f"MomentsBacking: names length {len(nm)} != ambient "
                    f"dimension D={flat_dim}."
                )
        return cls(
            components=comps,
            sigma=sig,
            names=nm,
            leaf_tags=tuple(leaf_tags),
            component_shapes=shapes,
            psd_index=psd_index,
            psd_rank=psd_rank,
            diagnostics=dict(diagnostics or {}),
            factory_spec=factory_spec,
        )


__all__ = [
    "SCHEMA_VERSION",
    "MomentsBacking",
    "manifold_to_tag",
    "tag_to_manifold",
    "euclidean_tag",
    "locate_psd_leaf",
]
