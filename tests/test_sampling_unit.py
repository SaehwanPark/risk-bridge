import numpy as np
import pandas as pd
import polars as pl
import pytest

from risk_bridge.sampling import propensity_scores_target_vs_source, psm_sample_source


def _target_source() -> tuple[pl.DataFrame, pl.DataFrame]:
  target = pl.DataFrame(
    {
      "caseY": [0, 1, 0, 1, 1, 0],
      "X1": [0, 1, 0, 1, 1, 0],
      "X2": [0, 1, 1, 0, 1, 0],
      "X3": [0, 0, 1, 1, 1, 0],
      "X4": [0, 1, 0, 1, 0, 1],
    }
  )
  source = pl.DataFrame(
    {
      "caseY": [0, 0, 1, 0, 1, 0],
      "X1": [1, 0, 1, 0, 1, 0],
      "X2": [1, 0, 0, 1, 0, 1],
      "X3": [1, 0, 1, 0, 1, 0],
      "X4": [1, 0, 1, 0, 0, 1],
    }
  )
  return target, source


def test_propensity_scores_bounds() -> None:
  target, source = _target_source()
  pooled = propensity_scores_target_vs_source(
    target, source, covariates=["caseY", "X1", "X2", "X3", "X4"]
  )
  assert isinstance(pooled, pl.DataFrame)
  assert "propensity_score" in pooled.columns
  scores = pooled.get_column("propensity_score").to_numpy()
  assert np.all((scores > 0.0) & (scores < 1.0))


def test_psm_sample_source_size() -> None:
  target, source = _target_source()
  pooled = propensity_scores_target_vs_source(
    target, source, covariates=["caseY", "X1", "X2", "X3", "X4"]
  )
  sample = psm_sample_source(np.random.default_rng(5), pooled, sample_size=4)
  assert isinstance(sample, pl.DataFrame)
  assert len(sample) == 4
  assert np.all(sample.get_column("Treat").to_numpy() == 0)


def test_sampling_accepts_pandas_inputs_and_returns_polars() -> None:
  target, source = _target_source()
  target_pd = pd.DataFrame(target.to_dict(as_series=False))
  source_pd = pd.DataFrame(source.to_dict(as_series=False))
  pooled = propensity_scores_target_vs_source(
    target_pd, source_pd, covariates=["caseY", "X1", "X2", "X3", "X4"]
  )
  sample = psm_sample_source(np.random.default_rng(5), pooled, sample_size=4)
  assert isinstance(pooled, pl.DataFrame)
  assert isinstance(sample, pl.DataFrame)


def test_psm_sample_source_too_large_raises() -> None:
  target, source = _target_source()
  pooled = propensity_scores_target_vs_source(
    target, source, covariates=["caseY", "X1", "X2", "X3", "X4"]
  )
  with pytest.raises(ValueError):
    psm_sample_source(np.random.default_rng(5), pooled, sample_size=100)


def test_psm_sample_source_matches_reference_greedy_sequence() -> None:
  target, source = _target_source()
  pooled = propensity_scores_target_vs_source(
    target, source, covariates=["caseY", "X1", "X2", "X3", "X4"]
  )
  pooled_pd = pd.DataFrame(pooled.to_dict(as_series=False))

  def reference_impl(rng: np.random.Generator, pooled_df: pd.DataFrame) -> pd.DataFrame:
    tgt = pooled_df.loc[pooled_df["Treat"] == 1].copy()
    src = pooled_df.loc[pooled_df["Treat"] == 0].copy()
    chosen_idx = rng.choice(tgt.index.to_numpy(), size=4, replace=False)
    chosen = tgt.loc[chosen_idx].copy()
    available = src.copy()
    rows: list[pd.Series] = []
    for _, row in chosen.iterrows():
      dist = np.abs(
        available["propensity_score"].to_numpy() - float(row["propensity_score"])
      )
      best_pos = int(np.argmin(dist))
      best_idx = available.index[best_pos]
      rows.append(available.loc[best_idx])
      available = available.drop(index=best_idx)
    return pd.DataFrame(rows).reset_index(drop=True)

  old_out = reference_impl(np.random.default_rng(9), pooled_pd)
  new_out = psm_sample_source(np.random.default_rng(9), pooled, sample_size=4)
  assert old_out["propensity_score"].to_numpy().tolist() == (
    new_out.get_column("propensity_score").to_numpy().tolist()
  )
