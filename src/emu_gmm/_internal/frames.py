"""Eager dataframe adapters; optional backends never enter compiled code."""

from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class _Adapter:
    module: str
    frame: str
    vector: str
    indexed: bool = True

    def accepts(self, value: Any) -> bool:
        module = sys.modules.get(self.module)
        return module is not None and isinstance(
            value, getattr(module, self.frame) | getattr(module, self.vector)
        )


# Subclasses precede their pandas base classes. Optional modules are loaded
# only for an explicit output request; incoming objects already loaded theirs.
_ADAPTERS = {
    "datamat": _Adapter("datamat", "DataMat", "DataVec"),
    "polars": _Adapter("polars", "DataFrame", "Series", indexed=False),
    "pandas": _Adapter("pandas", "DataFrame", "Series"),
}


def adapter_for(value: Any) -> _Adapter | None:
    return next((a for a in _ADAPTERS.values() if a.accepts(value)), None)


def as_pandas(value: Any) -> pd.DataFrame | pd.Series | None:
    """Canonical boundary view without requiring PyArrow or dataframe math."""
    adapter = adapter_for(value)
    if adapter is None:
        return None
    if adapter.module == "polars":
        module = sys.modules["polars"]

        def vector(series: Any) -> pd.Series:
            if series.dtype == module.Boolean:
                return pd.Series(series.to_list(), dtype="boolean", name=series.name)
            return pd.Series(series.to_numpy(), name=series.name)

        if isinstance(value, module.Series):
            return vector(value)
        return pd.DataFrame({name: vector(value[name]) for name in value.columns})
    if isinstance(value, pd.Series):
        return pd.Series(value, copy=False)
    return pd.DataFrame(value, copy=False)


def _index_keys(index: pd.Index) -> pd.Index:
    # A flat index and DataMat's one-level MultiIndex denote the same keys.
    values = list(index) if isinstance(index, pd.MultiIndex) else [(v,) for v in index]
    return pd.Index(values, tupleize_cols=False)


def numeric_values(frame: pd.DataFrame | pd.Series) -> np.ndarray:
    """Unwrap numeric/nullable columns without parsing strings as numbers."""
    dtypes = frame.dtypes if isinstance(frame, pd.DataFrame) else [frame.dtype]
    if not all(pd.api.types.is_numeric_dtype(dtype) for dtype in dtypes):
        raise ValueError("Dataframe inputs must contain numeric columns")
    values = frame.to_numpy()
    if values.dtype.kind == "O":
        values = frame.to_numpy(dtype=float, na_value=np.nan)
    return values


def align_rows(value: Any, frame: Any, *, name: str) -> Any:
    """Align indexed auxiliary inputs by exact observation keys, or refuse.

    Unindexed carriers (Polars and plain arrays) are explicitly positional.
    Mask columns are moment coordinates: never align them to data variables.
    """
    data_adapter, value_adapter = adapter_for(frame), adapter_for(value)
    if (
        data_adapter is None
        or value_adapter is None
        or not data_adapter.indexed
        or not value_adapter.indexed
    ):
        return value
    if frame.index.equals(value.index):
        return value
    target, source = _index_keys(frame.index), _index_keys(value.index)
    if target.equals(source):
        return value
    if not target.is_unique or not source.is_unique:
        raise ValueError(f"{name}: cannot realign duplicate observation labels")
    positions = source.get_indexer(target)
    if len(source) != len(target) or np.any(positions < 0):
        raise ValueError(f"{name}: observation labels do not match the data index")
    canonical = as_pandas(value)
    assert canonical is not None
    return canonical.iloc[positions]


def materialise(value: pd.DataFrame | pd.Series, backend: str) -> Any:
    """Materialise an existing labelled readout without recomputing statistics.

    Polars has no row index: index levels become explicit columns. Its column
    names must be distinct strings; unrepresentable labels fail explicitly.
    DataMat retains its native MultiIndex convention, including one-level axes.
    """
    if backend not in _ADAPTERS:
        raise ValueError(
            f"Unknown dataframe backend {backend!r}; choose pandas, polars or datamat"
        )
    if backend == "pandas":
        return value
    try:
        module = importlib.import_module(backend)
    except ImportError as exc:
        raise ImportError(f"Install {backend!r} to use backend={backend!r}") from exc
    if backend == "datamat":
        kwargs = {"idxnames": list(value.index.names)}
        if isinstance(value, pd.Series):
            return module.DataVec(value, **kwargs)
        return module.DataMat(value, colnames=list(value.columns.names), **kwargs)

    if isinstance(value, pd.Series):
        table = value.to_frame(name=value.name if value.name is not None else "value")
    else:
        table = value
    if isinstance(table.columns, pd.MultiIndex) or any(
        not isinstance(c, str) for c in table.columns
    ):
        raise ValueError(
            "Polars output requires string data-column labels; use pandas or datamat"
        )
    names = [
        (
            name
            if name is not None
            else ("index" if table.index.nlevels == 1 else f"index_{i}")
        )
        for i, name in enumerate(table.index.names)
    ]
    if any(not isinstance(n, str) for n in names) or len(
        set(names + list(table.columns))
    ) != len(names) + len(table.columns):
        raise ValueError(
            "Polars index-column names must be distinct strings and not collide with data columns"
        )

    def series(name: str, values: Any) -> Any:
        values = list(values)
        types = {type(v) for v in values if v is not None}
        # A summary Series mixes bool/int/float. Preserve its scalar types
        # rather than silently turning a convergence flag into a float.
        dtype = (
            module.Object if len(types) > 1 and any(t is bool for t in types) else None
        )
        return module.Series(name, values, dtype=dtype)

    columns = [series(n, table.index.get_level_values(i)) for i, n in enumerate(names)]
    columns.extend(series(n, table[n]) for n in table.columns)
    return module.DataFrame(columns)
