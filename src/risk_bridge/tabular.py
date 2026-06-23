from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any
import warnings

import numpy as np
import pandas as pd
import polars as pl
import rdata


FrameLike = pl.DataFrame | pd.DataFrame


def load_tabular_input(
  path: str | Path,
  *,
  object_name: str | None = None,
  dataset_name: str = "dataset",
) -> pl.DataFrame:
  """Load a supported tabular file into a Polars DataFrame.

  RData files can contain multiple named objects. A name is optional only when
  exactly one DataFrame object is present.
  """

  input_path = Path(path)
  suffix = input_path.suffix.lower()
  if suffix == ".csv":
    return pl.read_csv(input_path)
  if suffix == ".parquet":
    return pl.read_parquet(input_path)
  if suffix not in {".rda", ".rdata"}:
    raise ValueError(
      f"Unsupported {dataset_name} input format {input_path.suffix!r}. "
      "Expected .csv, .parquet, .rda, or .RData."
    )

  with warnings.catch_warnings():
    warnings.filterwarnings(
      "ignore", message='Missing constructor for R class "tbl(_df)?".*'
    )
    objects = rdata.read_rda(input_path)
  if object_name is not None:
    if object_name not in objects:
      available = ", ".join(sorted(objects)) or "<none>"
      raise ValueError(
        f"{dataset_name} RData object {object_name!r} is missing. "
        f"Available objects: {available}."
      )
    selected = objects[object_name]
    if not isinstance(selected, pd.DataFrame):
      raise ValueError(
        f"{dataset_name} RData object {object_name!r} is not a DataFrame "
        f"(got {type(selected).__name__})."
      )
    return ensure_polars_frame(selected, clone=False)

  frames = {name: value for name, value in objects.items() if isinstance(value, pd.DataFrame)}
  if len(frames) == 1:
    return ensure_polars_frame(next(iter(frames.values())), clone=False)
  if not frames:
    available = ", ".join(sorted(objects)) or "<none>"
    raise ValueError(
      f"{dataset_name} RData file contains no DataFrame objects. "
      f"Available objects: {available}."
    )
  names = ", ".join(sorted(frames))
  raise ValueError(
    f"{dataset_name} RData file contains multiple DataFrame objects: {names}. "
    "Select one explicitly with an object name."
  )


def ensure_polars_frame(df: FrameLike, *, clone: bool = True) -> pl.DataFrame:
  """Normalize pandas/polars inputs to a polars DataFrame."""

  if isinstance(df, pl.DataFrame):
    return df.clone() if clone else df
  if isinstance(df, pd.DataFrame):
    # Building from Python lists also handles pandas extension dtypes emitted by
    # RData readers without requiring the optional PyArrow dependency.
    return pl.DataFrame(df.to_dict(orient="list"))
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
