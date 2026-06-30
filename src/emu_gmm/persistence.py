r"""Typed, versioned persistence for an :class:`~emu_gmm.law.EstimatorLaw` (#181).

The estimate-once / query-many artifact for the K-Aggregators / CerealDemand
paper workflow: persist a fitted law to an **inert, cross-version** file and
reload it into a queryable law in the downstream paper repo --- *not* a pickle.

Why not pickle (the issue's case, which stands even though #147 fixed the
``ManifoldLeaf`` round-trip): pickle serialises by class reference, so a field
rename / module move silently breaks an artifact a referee reloads under a
future emu; and it can entomb model code (an exec/RCE surface). :class:`LawState`
is a typed, ``schema_version``-stamped record; the on-disk container is a single
``.npz`` (numpy + stdlib only --- no new runtime dependency) carrying the arrays
plus a JSON manifest, reconstructed through the normal constructors
(:meth:`AsymptoticLaw.from_moments`, the :func:`tag_to_manifold` codec) rather
than by resurrecting a live object.

**Scope (first slice).** The *asymptotic* grade --- the paper's primary need
(SEs, ``eigenvalue_se`` / ``gamma_se``, functional SEs read off
:math:`\mathcal N(\hat\theta, \Sigma_\theta)`). The *empirical* grade
(``EmpiricalLaw`` draws + ``{0,1}^E`` event flags + coupling, for ``given`` /
bootstrap queries) is the next slice; :func:`save_law` refuses it loudly rather
than writing a half-supported artifact.

**Companion, not successor** (the issue's recommendation): ``LawState`` is the
durable projection; :class:`~emu_gmm.types.EstimationResult` stays the live,
compute-coupled estimation output and is *not* on the reload path --- a reloaded
law is moments-backed and never reconstructs the live result.
"""

from __future__ import annotations

import dataclasses
import json
from typing import Any

import numpy as np

from emu_gmm._internal.law_state import (
    SCHEMA_VERSION,
    MomentsBacking,
    euclidean_tag,
    locate_psd_leaf,
    manifold_to_tag,
    tag_to_manifold,
)
from emu_gmm.law import AsymptoticLaw, EstimatorLaw


@dataclasses.dataclass(frozen=True)
class LawState:
    r"""Typed, versioned persistence record for an :class:`EstimatorLaw` (#181).

    The contract a consumer imports rather than reverse-engineers. Validated on
    load (``schema_version`` + array/shape consistency); the arrays
    (``theta_components`` / ``sigma_theta``) ride the ``.npz`` payload while the
    rest is the JSON manifest.

    Attributes
    ----------
    schema_version
        Layout version; :func:`load_law` refuses an unknown version with a
        migration hint rather than mis-parsing.
    grade
        The epistemic grade carried (``"asymptotic"`` in this slice).
    param_names
        Ambient tangent labels (length ``D``).
    leaf_tags
        Per-component typed manifold tags (e.g. ``{"type": "PSDFixedRank",
        "n": 5, "k": 2}``); reconstructed via :func:`tag_to_manifold`, never a
        live ``ManifoldLeaf`` (#147).
    component_shapes
        Per-component ambient shapes (``()`` for a scalar leaf).
    theta_components
        The per-leaf point arrays ``(A, phi, ...)``.
    sigma_theta
        The ``(D, D)`` covariance (gauge nullspace pinned to zero).
    psd_index, psd_rank
        Component index and rank of the unique ``PSDFixedRank`` factor, or
        ``None`` --- so a reloaded law answers ``eigenvalue_se`` / ``gamma_se``.
    diagnostics
        Scalar provenance (``J_stat``, ``J_dof``, ``gauge_nullspace_dim``, ...).
    factory_spec
        Optional estimator-configuration record (#142 anchor); ``None`` when
        the source law did not carry one.
    """

    schema_version: int
    grade: str
    param_names: tuple[str, ...]
    leaf_tags: tuple[dict[str, Any], ...]
    component_shapes: tuple[tuple[int, ...], ...]
    theta_components: tuple[np.ndarray, ...]
    sigma_theta: np.ndarray | None
    psd_index: int | None
    psd_rank: int | None
    diagnostics: dict[str, Any]
    factory_spec: dict[str, Any] | None = None


def _asymptotic_state(law: AsymptoticLaw) -> LawState:
    """Build a :class:`LawState` from an asymptotic law (either backing)."""
    if law._result is None:  # moments-backed (e.g. a re-save of a reloaded law)
        b = law._backing
        assert b is not None
        return LawState(
            schema_version=SCHEMA_VERSION,
            grade="asymptotic",
            param_names=b.names,
            leaf_tags=b.leaf_tags,
            component_shapes=b.component_shapes,
            theta_components=b.components,
            sigma_theta=b.sigma,
            psd_index=b.psd_index,
            psd_rank=b.psd_rank,
            diagnostics=dict(b.diagnostics),
            factory_spec=b.factory_spec,
        )

    # Live-result-backed: extract the durable projection from the result.
    result = law._result
    comps = tuple(np.asarray(c) for c in result.components())
    sigma = np.asarray(result.Sigma_theta.array)
    spec = result.manifold_spec
    if spec is not None:
        leaf_manifolds = [ls.manifold for ls in spec.leaf_specs]
        leaf_tags = tuple(manifold_to_tag(m) for m in leaf_manifolds)
        psd_index, psd_rank = locate_psd_leaf(leaf_manifolds)
    else:
        # v1 / all-scalar tree: every component is a Euclidean leaf, no gauge.
        leaf_tags = tuple(euclidean_tag(c) for c in comps)
        psd_index, psd_rank = None, None
    component_shapes = tuple(tuple(int(s) for s in np.shape(c)) for c in comps)

    diag = result.diagnostics
    diagnostics: dict[str, Any] = {
        "J_stat": float(np.asarray(result.J_stat)),
        "J_dof": int(result.J_dof),
        "J_pvalue": float(np.asarray(result.J_pvalue)),
        "gauge_nullspace_dim": int(diag.gauge_nullspace_dim),
        "tau_realised": float(np.asarray(diag.tau_realised)),
        "kappa_V": float(np.asarray(diag.kappa_V)),
    }
    return LawState(
        schema_version=SCHEMA_VERSION,
        grade="asymptotic",
        param_names=tuple(law.param_names),
        leaf_tags=leaf_tags,
        component_shapes=component_shapes,
        theta_components=comps,
        sigma_theta=sigma,
        psd_index=psd_index,
        psd_rank=psd_rank,
        diagnostics=diagnostics,
        factory_spec=None,
    )


def _write_state(state: LawState, path: Any) -> None:
    """Write ``state`` to a single ``.npz`` (arrays) + embedded JSON manifest."""
    manifest = {
        "schema_version": state.schema_version,
        "grade": state.grade,
        "param_names": list(state.param_names),
        "leaf_tags": list(state.leaf_tags),
        "component_shapes": [list(s) for s in state.component_shapes],
        "psd_index": state.psd_index,
        "psd_rank": state.psd_rank,
        "diagnostics": state.diagnostics,
        "factory_spec": state.factory_spec,
        "n_components": len(state.theta_components),
        "has_sigma": state.sigma_theta is not None,
    }
    arrays: dict[str, np.ndarray] = {
        f"comp_{i}": np.asarray(c) for i, c in enumerate(state.theta_components)
    }
    if state.sigma_theta is not None:
        arrays["sigma_theta"] = np.asarray(state.sigma_theta)
    # The manifest rides as a 0-d unicode array -> no pickle needed on load.
    arrays["__manifest__"] = np.asarray(json.dumps(manifest))
    # Write through an explicit file handle so the path is LITERAL: np.savez
    # appends ".npz" to a bare path, which would then mismatch load_law's
    # literal open. typeshed's savez stub doesn't model keyword array names
    # (only *args + allow_pickle), hence the suppression.
    with open(path, "wb") as fh:
        np.savez(fh, **arrays)  # type: ignore[arg-type]


def _read_state(path: Any) -> LawState:
    """Read + validate a :class:`LawState` from a ``.npz`` written by :func:`_write_state`."""
    with np.load(path, allow_pickle=False) as data:
        if "__manifest__" not in data.files:
            raise ValueError(
                f"load_law: {path!r} has no __manifest__ entry; it was not "
                "written by emu_gmm.save_law (or is a bare array .npz)."
            )
        manifest = json.loads(str(data["__manifest__"]))
        ver = int(manifest["schema_version"])
        if ver != SCHEMA_VERSION:
            raise ValueError(
                f"load_law: artifact schema_version {ver} != supported "
                f"{SCHEMA_VERSION}. This file was written by a different "
                "emu_gmm; no migration is registered for it yet. (A future "
                "version adds an upgrader table keyed on schema_version.)"
            )
        n = int(manifest["n_components"])
        comps = tuple(np.asarray(data[f"comp_{i}"]) for i in range(n))
        sigma = np.asarray(data["sigma_theta"]) if manifest.get("has_sigma") else None
    return LawState(
        schema_version=ver,
        grade=manifest["grade"],
        param_names=tuple(manifest["param_names"]),
        leaf_tags=tuple(manifest["leaf_tags"]),
        component_shapes=tuple(
            tuple(int(s) for s in sh) for sh in manifest["component_shapes"]
        ),
        theta_components=comps,
        sigma_theta=sigma,
        psd_index=manifest["psd_index"],
        psd_rank=manifest["psd_rank"],
        diagnostics=manifest["diagnostics"],
        factory_spec=manifest["factory_spec"],
    )


def _state_to_law(state: LawState) -> AsymptoticLaw:
    """Reconstruct a queryable (moments-backed) :class:`AsymptoticLaw` from ``state``.

    Rebuilds the leaf manifolds through their normal constructors
    (:func:`tag_to_manifold`) and re-derives the PSD-leaf ``(index, rank)`` from
    them --- the #147 "reconstruct via the constructor, never the immutable live
    leaf" contract, and a cross-check on the persisted indices.
    """
    leaf_manifolds = tuple(tag_to_manifold(t) for t in state.leaf_tags)
    psd_index, psd_rank = locate_psd_leaf(leaf_manifolds)
    backing = MomentsBacking(
        components=state.theta_components,
        sigma=np.asarray(state.sigma_theta),
        names=state.param_names,
        leaf_tags=state.leaf_tags,
        component_shapes=state.component_shapes,
        psd_index=psd_index,
        psd_rank=psd_rank,
        diagnostics=dict(state.diagnostics),
        factory_spec=state.factory_spec,
    )
    return AsymptoticLaw(_backing=backing, label="asymptotic(loaded)")


def save_law(law: EstimatorLaw, path: Any) -> None:
    """Persist ``law`` to ``path`` as a typed, versioned :class:`LawState` (#181).

    Asymptotic grade only in this slice; an :class:`~emu_gmm.law.EmpiricalLaw`
    is refused loudly (its draws + event-flag persistence is the next slice)
    rather than silently dropping the events / coupling provenance.
    """
    if isinstance(law, AsymptoticLaw):
        _write_state(_asymptotic_state(law), path)
        return
    raise NotImplementedError(
        f"save_law: persistence of {type(law).__name__} is not implemented yet "
        "(only the asymptotic grade). The empirical grade (stacked draws x "
        "event flags x coupling) is the next #181 slice; persisting it without "
        "the event/coupling provenance would lose exactly what makes given() / "
        "couple() sound."
    )


def load_law(path: Any) -> EstimatorLaw:
    """Load a law persisted by :func:`save_law`.

    Returns a moments-backed :class:`~emu_gmm.law.AsymptoticLaw` --- queryable
    (``se`` / ``functional_se`` / ``eigenvalue_se`` / ``gamma_se`` / ``sample``)
    with no live :class:`~emu_gmm.types.EstimationResult` required. Validates the
    ``schema_version`` and refuses an unknown one.
    """
    state = _read_state(path)
    if state.grade == "asymptotic":
        return _state_to_law(state)
    raise NotImplementedError(
        f"load_law: grade {state.grade!r} is not supported yet (only "
        "'asymptotic'). The empirical grade is the next #181 slice."
    )


__all__ = ["LawState", "save_law", "load_law"]
