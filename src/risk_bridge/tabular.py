from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import polars as pl


FrameLike = pl.DataFrame | pd.DataFrame


def ensure_polars_frame(df: FrameLike, *, clone: bool = True) -> pl.DataFrame:
  """Normalize pandas/polars inputs to a polars DataFrame."""

  if isinstance(df, pl.DataFrame):
    return df.clone() if clone else df
  if isinstance(df, pd.DataFrame):
    return pl.from_pandas(df, include_index=False)
  raise TypeError(f"Unsupported DataFrame type: {type(df)!r}")


def column_to_numpy(
  df: pl.DataFrame, column_name: str, *, dtype: np.dtype[Any] | None = None
) -> np.ndarray:
  arr = df.get_column(column_name).to_numpy()
  return np.asarray(arr, dtype=dtype) if dtype is not None else np.asarray(arr)


def select_to_numpy(
  df: pl.DataFrame,
  columns: list[str] | tuple[str, ...],
  *,
  dtype: np.dtype[Any] | None = None,
) -> np.ndarray:
  arr = df.select(list(columns)).to_numpy()
  return np.asarray(arr, dtype=dtype) if dtype is not None else np.asarray(arr)


def rows_to_frame(
  rows: list[dict[str, Any]],
  *,
  columns: list[str] | tuple[str, ...] | None = None,
) -> pl.DataFrame:
  if rows:
    frame = pl.DataFrame(rows)
  elif columns is not None:
    frame = pl.DataFrame({col: [] for col in columns})
  else:
    frame = pl.DataFrame()
  if columns is not None:
    return frame.select(list(columns))
  return frame


def gather_rows(df: pl.DataFrame, positions: np.ndarray) -> pl.DataFrame:
  pos = np.asarray(positions, dtype=np.int64)
  return df.select(pl.all().gather(pos))


def append_csv(df: pl.DataFrame, path: Path) -> None:
  exists = path.exists()
  with path.open("a", encoding="utf-8", newline="") as fh:
    df.write_csv(fh, include_header=not exists)


def write_tabular_export(df: pl.DataFrame, csv_path: Path, write_parquet: bool) -> bool:
  df.write_csv(csv_path)
  if not write_parquet:
    return False
  try:
    df.write_parquet(csv_path.with_suffix(".parquet"))
  except (ImportError, ModuleNotFoundError, ValueError, OSError):
    return False
  return True


def save_pickle_bundle(bundle: dict[str, object], out_path: Path) -> None:
  with out_path.open("wb") as fh:
    pickle.dump(bundle, fh)
