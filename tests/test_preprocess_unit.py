import numpy as np
import pandas as pd
import polars as pl
import pytest

from risk_bridge.preprocess import preprocess_user_dataset


def test_preprocess_derives_zcat_from_zorigin() -> None:
  df = pl.DataFrame(
    {
      "outcome": [1, 0, 1],
      "X1": [0, 1, 1],
      "X2": [0, 1, 0],
      "z": [0.12, 0.42, 0.88],
    }
  )
  out = preprocess_user_dataset(
    df,
    x_cols=["X1", "X2"],
    z_bins=(0.2, 0.5, 0.9),
    dataset_name="df",
    y_col="outcome",
    z_origin_col="z",
    z_cat_col="zCat",
  )
  assert isinstance(out, pl.DataFrame)
  assert set(["caseY", "X1", "X2", "zOrigin", "zCat"]) == set(out.columns)
  assert np.array_equal(
    out.get_column("zCat").to_numpy(), np.array([0, 1, 2], dtype=np.int64)
  )


def test_preprocess_derives_zorigin_from_zcat_when_enabled() -> None:
  df = pl.DataFrame(
    {
      "caseY": [1, 0, 1],
      "X1": [0, 1, 1],
      "X2": [0, 1, 0],
      "zcat_in": [0, 2, 3],
    }
  )
  out = preprocess_user_dataset(
    df,
    x_cols=["X1", "X2"],
    z_bins=(0.2, 0.5, 0.9),
    dataset_name="df",
    z_origin_col="zOrigin",
    z_cat_col="zcat_in",
    allow_z_origin_from_z_cat=True,
  )
  assert "zOrigin" in out.columns
  z_origin = out.get_column("zOrigin").to_numpy()
  assert np.all((z_origin > 0.0) & (z_origin <= 1.0))


def test_preprocess_accepts_pandas_inputs_and_returns_polars() -> None:
  df = pd.DataFrame(
    {
      "caseY": [1, 0, 1],
      "X1": [0, 1, 1],
      "X2": [0, 1, 0],
      "zOrigin": [0.12, 0.42, 0.88],
    }
  )
  out = preprocess_user_dataset(
    df,
    x_cols=["X1", "X2"],
    z_bins=(0.2, 0.5, 0.9),
    dataset_name="df",
  )
  assert isinstance(out, pl.DataFrame)
  assert list(out.columns) == ["caseY", "X1", "X2", "zOrigin", "zCat"]


def test_preprocess_raises_when_neither_z_column_exists() -> None:
  df = pd.DataFrame({"caseY": [1], "X1": [0], "X2": [0]})
  with pytest.raises(ValueError):
    preprocess_user_dataset(
      df,
      x_cols=["X1", "X2"],
      z_bins=(0.2, 0.5, 0.9),
      dataset_name="df",
    )
