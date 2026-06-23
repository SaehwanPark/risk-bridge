from __future__ import annotations

import warnings

import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression

from risk_bridge.tabular import FrameLike, ensure_polars_frame, gather_rows, select_to_numpy


def propensity_scores_target_vs_source(
  target_df: FrameLike,
  source_df: FrameLike,
  covariates: list[str],
) -> pl.DataFrame:
  """Fit Treat~covariates and attach propensity scores to pooled data."""

  t = ensure_polars_frame(target_df, clone=False)
  s = ensure_polars_frame(source_df, clone=False)
  t = t.with_columns(pl.lit(1).alias("Treat"))
  s = s.with_columns(pl.lit(0).alias("Treat"))

  pooled = pl.concat([t, s], how="vertical_relaxed")

  model = LogisticRegression(max_iter=1000, solver="lbfgs")
  x = select_to_numpy(pooled, covariates, dtype=np.float64)
  y = pooled.get_column("Treat").to_numpy()
  model.fit(x, y)
  propensity_score = model.predict_proba(x)[:, 1].astype(np.float64)

  return pooled.with_columns(pl.Series("propensity_score", propensity_score))


def psm_sample_source(
  rng: np.random.Generator,
  pooled_df: FrameLike,
  sample_size: int,
) -> pl.DataFrame:
  """Sample target rows and perform 1:1 nearest-neighbor matching from source."""

  if sample_size <= 0:
    raise ValueError("sample_size must be > 0")

  pooled = ensure_polars_frame(pooled_df, clone=False)
  treat = np.asarray(pooled.get_column("Treat").to_numpy(), dtype=np.int64)
  target_pos = np.flatnonzero(treat == 1)
  source_pos = np.flatnonzero(treat == 0)

  if sample_size > len(target_pos) or sample_size > len(source_pos):
    raise ValueError("sample_size exceeds available target/source records")

  chosen_target_pos = np.asarray(
    rng.choice(len(target_pos), size=sample_size, replace=False), dtype=np.int64
  )
  propensity_scores = np.asarray(
    pooled.get_column("propensity_score").to_numpy(), dtype=np.float64
  )
  source_scores = propensity_scores[source_pos]
  target_scores = propensity_scores[target_pos[chosen_target_pos]]
  work_scores = source_scores.copy()
  dist = np.empty_like(source_scores)
  matched_pos = np.empty(sample_size, dtype=np.int64)
  unavailable_score = 10.0

  for i, score in enumerate(target_scores):
    np.abs(work_scores - score, out=dist)
    best_pos = int(np.argmin(dist))
    matched_pos[i] = best_pos
    work_scores[best_pos] = unavailable_score

  return gather_rows(pooled, source_pos[matched_pos])


def propensity_scores_x_only(
  target_x: FrameLike,
  source_df: FrameLike,
  covariates: list[str],
) -> tuple[np.ndarray, np.ndarray]:
  """Fit an unpenalized Treat~X model for an X-only pseudo target."""

  target = ensure_polars_frame(target_x, clone=False)
  source = ensure_polars_frame(source_df, clone=False)
  x_target = select_to_numpy(target, covariates, dtype=np.float64)
  x_source = select_to_numpy(source, covariates, dtype=np.float64)
  x = np.vstack([x_target, x_source])
  y = np.concatenate(
    [np.ones(len(target), dtype=np.int64), np.zeros(len(source), dtype=np.int64)]
  )
  model = LogisticRegression(max_iter=1000, solver="lbfgs", penalty=None)
  with warnings.catch_warnings():
    warnings.filterwarnings(
      "ignore", message="'penalty' was deprecated.*", category=FutureWarning
    )
    model.fit(x, y)
  scores = model.predict_proba(x)[:, 1].astype(np.float64)
  return scores[: len(target)], scores[len(target) :]


def psm_sample_source_from_scores(
  rng: np.random.Generator,
  source_df: FrameLike,
  target_scores: np.ndarray,
  source_scores: np.ndarray,
  sample_size: int,
) -> pl.DataFrame:
  """Sample target scores and greedily match source row instances once."""

  source = ensure_polars_frame(source_df, clone=False)
  target_arr = np.asarray(target_scores, dtype=np.float64)
  source_arr = np.asarray(source_scores, dtype=np.float64)
  if sample_size <= 0:
    raise ValueError("sample_size must be > 0")
  if sample_size > len(target_arr) or sample_size > len(source_arr):
    raise ValueError("sample_size exceeds available target/source records")
  if len(source_arr) != len(source):
    raise ValueError("source_scores length must match source_df")

  chosen = np.asarray(
    rng.choice(len(target_arr), size=sample_size, replace=False), dtype=np.int64
  )
  work_scores = source_arr.copy()
  distances = np.empty_like(work_scores)
  matched = np.empty(sample_size, dtype=np.int64)
  for i, score in enumerate(target_arr[chosen]):
    np.abs(work_scores - score, out=distances)
    best = int(np.argmin(distances))
    matched[i] = best
    work_scores[best] = np.inf
  return gather_rows(source, matched)
