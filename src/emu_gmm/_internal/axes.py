"""Canonical Haliax axis definitions for emu-gmm.

The axis taxonomy used throughout the framework:

==================  ====  ==============================================
Name                Size  Role
==================  ====  ==============================================
``parameters``      K     First index of parameter-shaped tensors
``parameters_dual`` K     Second index of K x K matrices (symmetric)
``moments``         M     First index of moment-shaped tensors
``moments_dual``    M     Second index of M x M matrices (symmetric)
``observations``    N     Sample index for empirical / synthetic draws
==================  ====  ==============================================

The ``*_dual`` axes carry the same size as their primary counterparts but
distinct names, so that :func:`haliax.dot` contractions never alias the
two indices of a symmetric matrix. See ``docs/design.org`` Section 6 for
the rationale.

Axis sizes are problem-dependent (``K``, ``M``, ``N`` come from the
user's model and data), so the canonical definitions are exposed as
factory functions rather than module-level constants.
"""

from __future__ import annotations

import haliax as ha

#: Canonical axis name strings; exposed for introspection / testing.
PARAMS_NAME = "parameters"
PARAMS_DUAL_NAME = "parameters_dual"
MOMENTS_NAME = "moments"
MOMENTS_DUAL_NAME = "moments_dual"
OBS_NAME = "observations"


def params_axis(size: int) -> ha.Axis:
    """Return the canonical ``parameters`` axis of the given size."""
    return ha.Axis(name=PARAMS_NAME, size=size)


def params_dual_axis(size: int) -> ha.Axis:
    """Return the canonical ``parameters_dual`` axis of the given size."""
    return ha.Axis(name=PARAMS_DUAL_NAME, size=size)


def moments_axis(size: int) -> ha.Axis:
    """Return the canonical ``moments`` axis of the given size."""
    return ha.Axis(name=MOMENTS_NAME, size=size)


def moments_dual_axis(size: int) -> ha.Axis:
    """Return the canonical ``moments_dual`` axis of the given size."""
    return ha.Axis(name=MOMENTS_DUAL_NAME, size=size)


def obs_axis(size: int) -> ha.Axis:
    """Return the canonical ``observations`` axis of the given size."""
    return ha.Axis(name=OBS_NAME, size=size)


def dual(axis: ha.Axis) -> ha.Axis:
    """Return the dual axis: same size, name with ``_dual`` suffix.

    Used to disambiguate the two indices of a symmetric matrix when
    contracting via :func:`haliax.dot`.

    Raises
    ------
    ValueError
        If ``axis`` already has a name ending in ``_dual``.
    """
    if axis.name.endswith("_dual"):
        raise ValueError(
            f"axis {axis.name!r} is already a dual axis; "
            "applying dual() to a dual axis is not defined"
        )
    return ha.Axis(name=f"{axis.name}_dual", size=axis.size)


__all__ = [
    "PARAMS_NAME",
    "PARAMS_DUAL_NAME",
    "MOMENTS_NAME",
    "MOMENTS_DUAL_NAME",
    "OBS_NAME",
    "params_axis",
    "params_dual_axis",
    "moments_axis",
    "moments_dual_axis",
    "obs_axis",
    "dual",
]
