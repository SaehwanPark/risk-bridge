from __future__ import annotations

import numpy as np
import polars as pl

from risk_bridge.simulate import categorize_z
from risk_bridge.tabular import FrameLike, ensure_polars_frame

_POS_LABELS = {"1", "true", "t", "yes", "y"}
_NEG_LABELS = {"0", "false", "f", "no", "n"}
_INT_DTYPES = {
  pl.Int8,
  pl.Int16,
  pl.Int32,
  pl.Int64,
  pl.UInt8,
  pl.UInt16,
  pl.UInt32,
  pl.UInt64,
}
_FLOAT_DTYPES = {
  pl.Float32,
  pl.Float64,
}


def _coerce_binary_target(
  series: pl.Series, dataset_name: str, column_name: str
) -> np.ndarray:
  vals = series.to_numpy()

  if vals.dtype == np.bool_:
    return vals.astype(np.int64)

  if np.issubdtype(vals.dtype, np.number):
    out = np.asarray(vals, dtype=np.float64)
    if np.any(~np.isfinite(out)):
      raise ValueError(f"{dataset_name}.{column_name} contains non-finite values.")
    if not np.allclose(out, np.round(out)):
      raise ValueError(f"{dataset_name}.{column_name} must be binary (0/1).")
    out_int = out.astype(np.int64)
    if not set(np.unique(out_int)).issubset({0, 1}):
      raise ValueError(f"{dataset_name}.{column_name} must be binary (0/1).")
    return out_int

  lowered = [str(v).strip().lower() for v in vals]
  out_int = np.empty(len(lowered), dtype=np.int64)
  for i, val in enumerate(lowered):
    if val in _POS_LABELS:
      out_int[i] = 1
    elif val in _NEG_LABELS:
      out_int[i] = 0
    else:
      raise ValueError(
        f"{dataset_name}.{column_name} contains unsupported label value: {vals[i]!r}"
      )
  return out_int


def _coerce_int_feature(
  series: pl.Series, dataset_name: str, column_name: str
) -> np.ndarray:
  vals = np.asarray(series.to_numpy(), dtype=np.float64)
  if np.any(~np.isfinite(vals)):
    raise ValueError(f"{dataset_name}.{column_name} contains non-finite values.")
  if not np.allclose(vals, np.round(vals)):
    raise ValueError(
      f"{dataset_name}.{column_name} must be integer-coded for current discrete X modeling."
    )
  return vals.astype(np.int64)


def _coerce_z_origin(series: pl.Series, dataset_name: str, column_name: str) -> np.ndarray:
  z = np.asarray(series.to_numpy(), dtype=np.float64)
  if np.any(~np.isfinite(z)):
    raise ValueError(f"{dataset_name}.{column_name} contains non-finite values.")
  if np.any(z <= 0.0) or np.any(z > 1.0):
    raise ValueError(f"{dataset_name}.{column_name} must be in (0, 1].")
  return z


def _coerce_z_cat(series: pl.Series, dataset_name: str, column_name: str) -> np.ndarray:
  zc = np.asarray(series.to_numpy(), dtype=np.float64)
  if np.any(~np.isfinite(zc)):
    raise ValueError(f"{dataset_name}.{column_name} contains non-finite values.")
  if not np.allclose(zc, np.round(zc)):
    raise ValueError(f"{dataset_name}.{column_name} must be integer-coded.")
  zc_int = zc.astype(np.int64)
  if np.any(zc_int < 0):
    raise ValueError(f"{dataset_name}.{column_name} must be non-negative.")
  return zc_int


def _z_origin_from_z_cat(z_cat: np.ndarray, z_bins: np.ndarray) -> np.ndarray:
  edges = np.concatenate([[0.0], z_bins, [1.0]]).astype(np.float64)
  if np.any(z_cat < 0) or np.any(z_cat >= len(edges) - 1):
    raise ValueError("zCat values must be in [0, len(z_bins)] when deriving zOrigin.")
  left = edges[z_cat]
  right = edges[z_cat + 1]
  z_mid = (left + right) / 2.0
  return np.clip(z_mid, 1e-12, 1.0)


def _binary_expr_or_series(
  frame: pl.DataFrame,
  *,
  source_col: str,
  dataset_name: str,
) -> pl.Expr | pl.Series:
  series = frame.get_column(source_col)
  if series.null_count() > 0:
    raise ValueError(f"{dataset_name}.{source_col} contains non-finite values.")

  if series.dtype == pl.Boolean:
    return pl.col(source_col).cast(pl.Int64).alias("caseY")

  if series.dtype in _INT_DTYPES:
    min_val = series.min()
    max_val = series.max()
    if min_val is None or max_val is None or int(min_val) < 0 or int(max_val) > 1:
      raise ValueError(f"{dataset_name}.{source_col} must be binary (0/1).")
    return pl.col(source_col).cast(pl.Int64).alias("caseY")

  out_int = _coerce_binary_target(series, dataset_name, source_col)
  return pl.Series("caseY", out_int)


def _int_feature_expr_or_series(
  frame: pl.DataFrame,
  *,
  source_col: str,
  dataset_name: str,
) -> pl.Expr | pl.Series:
  series = frame.get_column(source_col)
  if series.null_count() > 0:
    raise ValueError(f"{dataset_name}.{source_col} contains non-finite values.")
  if series.dtype in _INT_DTYPES:
    return pl.col(source_col).cast(pl.Int64).alias(source_col)
  return pl.Series(
    source_col, _coerce_int_feature(series, dataset_name, source_col)
  )


def _z_origin_expr_or_series(
  frame: pl.DataFrame,
  *,
  source_col: str,
  dataset_name: str,
) -> tuple[pl.Expr | pl.Series, np.ndarray]:
  series = frame.get_column(source_col)
  if series.null_count() > 0:
    raise ValueError(f"{dataset_name}.{source_col} contains non-finite values.")

  if series.dtype in _INT_DTYPES | _FLOAT_DTYPES:
    z = np.asarray(series.to_numpy(), dtype=np.float64)
    if np.any(~np.isfinite(z)):
      raise ValueError(f"{dataset_name}.{source_col} contains non-finite values.")
    if np.any(z <= 0.0) or np.any(z > 1.0):
      raise ValueError(f"{dataset_name}.{source_col} must be in (0, 1].")
    if series.dtype == pl.Float64:
      return pl.col(source_col).alias("zOrigin"), z
    return pl.col(source_col).cast(pl.Float64).alias("zOrigin"), z

  z = _coerce_z_origin(series, dataset_name, source_col)
  return pl.Series("zOrigin", z), z


def _z_cat_expr_or_series(
  frame: pl.DataFrame,
  *,
  source_col: str,
  dataset_name: str,
) -> tuple[pl.Expr | pl.Series, np.ndarray]:
  series = frame.get_column(source_col)
  if series.null_count() > 0:
    raise ValueError(f"{dataset_name}.{source_col} contains non-finite values.")

  if series.dtype in _INT_DTYPES:
    zc = np.asarray(series.to_numpy(), dtype=np.int64)
    if np.any(zc < 0):
      raise ValueError(f"{dataset_name}.{source_col} must be non-negative.")
    return pl.col(source_col).cast(pl.Int64).alias("zCat"), zc

  zc = _coerce_z_cat(series, dataset_name, source_col)
  return pl.Series("zCat", zc), zc


def preprocess_user_dataset(
  df: FrameLike,
  *,
  x_cols: list[str],
  z_bins: tuple[float, ...] | np.ndarray,
  dataset_name: str,
  y_col: str = "caseY",
  z_origin_col: str = "zOrigin",
  z_cat_col: str = "zCat",
  allow_z_origin_from_z_cat: bool = False,
) -> pl.DataFrame:
  """Normalize and auto-complete user dataset into canonical pipeline columns.

  Output columns are always: `caseY`, `x_cols...`, `zOrigin`, `zCat`.

  Auto-completion behavior:
  - If `zCat` is missing and `zOrigin` exists, `zCat` is derived from `zOrigin`.
  - If `zOrigin` is missing and `allow_z_origin_from_z_cat=True`, `zOrigin` is
    synthesized from `zCat` interval midpoints.
  """

  frame = ensure_polars_frame(df, clone=False)

  if len(frame) == 0:
    raise ValueError(f"{dataset_name} must contain at least one row.")
  if not x_cols:
    raise ValueError("x_cols must be non-empty.")

  z_bins_arr = np.asarray(z_bins, dtype=np.float64)
  if len(z_bins_arr) == 0 or np.any(np.diff(z_bins_arr) <= 0):
    raise ValueError("z_bins must be strictly increasing.")
  if np.any(z_bins_arr <= 0.0) or np.any(z_bins_arr >= 1.0):
    raise ValueError("z_bins must be in (0, 1).")

  required = [y_col, *x_cols]
  missing_base = [c for c in required if c not in frame.columns]
  if missing_base:
    raise ValueError(f"{dataset_name} is missing required columns: {missing_base}")

  has_z_origin = z_origin_col in frame.columns
  has_z_cat = z_cat_col in frame.columns
  if not has_z_origin and not has_z_cat:
    raise ValueError(
      f"{dataset_name} must contain at least one of {z_origin_col!r} or {z_cat_col!r}."
    )

  exprs: list[pl.Expr] = []
  series_out: list[pl.Series] = []
  case_y_out = _binary_expr_or_series(
    frame, source_col=y_col, dataset_name=dataset_name
  )
  if isinstance(case_y_out, pl.Expr):
    exprs.append(case_y_out)
  else:
    series_out.append(case_y_out)
  for col in x_cols:
    col_out = _int_feature_expr_or_series(
      frame, source_col=col, dataset_name=dataset_name
    )
    if isinstance(col_out, pl.Expr):
      exprs.append(col_out)
    else:
      series_out.append(col_out)

  z_origin: np.ndarray | None = None
  z_cat: np.ndarray | None = None

  if has_z_origin:
    z_origin_out, z_origin = _z_origin_expr_or_series(
      frame, source_col=z_origin_col, dataset_name=dataset_name
    )
    if isinstance(z_origin_out, pl.Expr):
      exprs.append(z_origin_out)
    else:
      series_out.append(z_origin_out)

  if has_z_cat:
    z_cat_out, z_cat = _z_cat_expr_or_series(
      frame, source_col=z_cat_col, dataset_name=dataset_name
    )
    if isinstance(z_cat_out, pl.Expr):
      exprs.append(z_cat_out)
    else:
      series_out.append(z_cat_out)

  if z_cat is None and z_origin is not None:
    z_cat = categorize_z(z_origin, z_bins_arr)
    series_out.append(pl.Series("zCat", z_cat))

  if z_origin is None and z_cat is not None:
    if not allow_z_origin_from_z_cat:
      raise ValueError(
        f"{dataset_name}.{z_origin_col} is missing. "
        "Either provide zOrigin or enable allow_z_origin_from_z_cat."
      )
    z_origin = _z_origin_from_z_cat(z_cat, z_bins_arr)
    series_out.append(pl.Series("zOrigin", z_origin))

  assert z_origin is not None
  assert z_cat is not None

  if np.any(z_cat > len(z_bins_arr)):
    raise ValueError(
      f"{dataset_name}.{z_cat_col} exceeds supported range. "
      "Expected zCat in [0, len(z_bins)]."
    )

  out = frame.select(exprs) if exprs else pl.DataFrame()
  if series_out:
    out = out.with_columns(series_out)
  return out.select(["caseY", *x_cols, "zOrigin", "zCat"])
