"""Static model dispatch and adapters (ledger Section 26).

Auxiliary state is prepared inside each differentiated batch evaluation.
It is never cached across parameters or measures.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, cast

import haliax as ha
import jax.numpy as jnp

from emu_gmm.types import ParamsLike, ReducingModel, RowModel, StructuralModel


def is_reducing_model(model: StructuralModel) -> bool:
    """Recognize the complete, explicit two-stage contract statically."""
    aux = callable(getattr(model, "aux", None))
    psi = callable(getattr(model, "psi", None))
    if not (aux and psi) and not callable(model):
        raise TypeError(
            "A model must be row-callable or supply both callable aux and psi methods"
        )
    return aux and psi


def reject_reducing_model(model: StructuralModel, measure_name: str) -> None:
    """Keep unsupported population/sampler derivative contracts explicit."""
    if is_reducing_model(model):
        raise TypeError(
            f"Reducing models are supported only on EmpiricalMeasure; "
            f"{measure_name} needs an explicit population/sampler adapter"
        )


def prepare_row_model(
    model: StructuralModel, measure: Any, theta: ParamsLike
) -> Callable[[Any], Any]:
    """Bind one aux computation and current parameters for a shared row pass."""
    if is_reducing_model(model):
        # Local import avoids a measures/types import cycle.
        from emu_gmm.measures.empirical import EmpiricalMeasure

        if not isinstance(measure, EmpiricalMeasure):
            reject_reducing_model(model, type(measure).__name__)
        reducing = cast(ReducingModel, model)
        aux = reducing.aux(measure, theta)
        return lambda row: reducing.psi(row, theta, aux)
    row_model = cast(RowModel, model)
    return lambda row: row_model(row, theta)


@dataclass(frozen=True)
class _ParameterMappedModel:
    model: ReducingModel
    transform: Callable[[ParamsLike], ParamsLike]

    def aux(self, measure: Any, theta: ParamsLike) -> Any:
        return self.model.aux(measure, self.transform(theta))

    def psi(self, row: Any, theta: ParamsLike, aux: Any) -> Any:
        return self.model.psi(row, self.transform(theta), aux)


def map_model_parameters(
    model: StructuralModel, transform: Callable[[ParamsLike], ParamsLike]
) -> StructuralModel:
    """Adapt profile parameters without freezing the sample reduction."""
    if is_reducing_model(model):
        return _ParameterMappedModel(cast(ReducingModel, model), transform)
    row_model = cast(RowModel, model)
    return lambda row, theta: row_model(row, transform(theta))


def _subtract_shift(value: Any, shift: Any) -> Any:
    if isinstance(value, ha.NamedArray):
        return ha.NamedArray(value.array - shift, value.axes)
    return jnp.asarray(value) - shift


@dataclass(frozen=True)
class _ShiftedModel:
    model: ReducingModel
    shift: Any

    def aux(self, measure: Any, theta: ParamsLike) -> Any:
        return self.model.aux(measure, theta)

    def psi(self, row: Any, theta: ParamsLike, aux: Any) -> Any:
        return _subtract_shift(self.model.psi(row, theta, aux), self.shift)


def shift_model(model: StructuralModel, shift: Any) -> StructuralModel:
    """Preserve aux recomputation and labels in null-recentered refits."""
    if is_reducing_model(model):
        return _ShiftedModel(cast(ReducingModel, model), shift)
    row_model = cast(RowModel, model)
    return lambda row, theta: _subtract_shift(row_model(row, theta), shift)
