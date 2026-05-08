from __future__ import annotations

from itertools import product

import numpy as np
import numpy.typing as npt
import polars as pl
from sklearn.linear_model import LogisticRegression

from risk_bridge.config import FeatureSpec
from risk_bridge.tabular import select_to_numpy
from risk_bridge.types import CalibrationArtifacts, Population

ArrayF = npt.NDArray[np.float64]
ArrayI = npt.NDArray[np.int64]


def fit_base_risk_model(
  ref_pop: Population, x_cols: list[str]
) -> tuple[ArrayF, callable]:
  """Fit phi(X)=P(Y=1|X) using logistic regression on reference population."""

  X = select_to_numpy(ref_pop.X, x_cols, dtype=np.float64)
  y = ref_pop.y
  uniq = np.unique(y)
  if len(uniq) < 2:
    p = float(uniq[0])
    coef = np.zeros(len(x_cols) + 1, dtype=np.float64)

    def predict_fn(df: pl.DataFrame) -> ArrayF:
      return np.full(len(df), p, dtype=np.float64)

    return coef, predict_fn

  model = LogisticRegression(max_iter=1000, solver="lbfgs")
  model.fit(X, y)

  coef = np.concatenate(
    [model.intercept_.astype(np.float64), model.coef_.ravel().astype(np.float64)]
  )

  def predict_fn(df: pl.DataFrame) -> ArrayF:
    return model.predict_proba(select_to_numpy(df, x_cols, dtype=np.float64))[
      :, 1
    ].astype(np.float64)

  return coef, predict_fn


def compute_risk_limits(
  phi_hat: ArrayF, quantiles: tuple[float, float, float] = (0.25, 0.5, 0.75)
) -> ArrayF:
  """Compute risk interval boundaries from quantiles of phi_hat."""

  return np.quantile(
    np.asarray(phi_hat, dtype=np.float64), np.asarray(quantiles, dtype=np.float64)
  ).astype(np.float64)


def _feature_support(spec: FeatureSpec) -> list[int]:
  if spec.kind == "categorical_cut":
    breaks = tuple(float(v) for v in spec.params["breaks"])
    if len(breaks) < 2:
      raise ValueError("categorical_cut breaks must have length >= 2")
    return list(range(len(breaks) - 1))
  if spec.kind == "capped_poisson":
    cap = int(spec.params.get("cap", 2))
    return list(range(cap + 1))
  if spec.kind == "custom":
    return [int(v) for v in spec.params["values"]]
  raise ValueError(f"Unsupported feature kind: {spec.kind}")


def enumerate_x_combinations(feature_specs: tuple[FeatureSpec, ...]) -> pl.DataFrame:
  """Enumerate full Cartesian support of discrete X levels.

  Example
  -------
  >>> specs = (FeatureSpec("X1", "categorical_cut", {"breaks": (0, 0.5, 1)}),)
  >>> enumerate_x_combinations(specs).shape
  (2, 1)
  """

  supports = [_feature_support(spec) for spec in feature_specs]
  rows = list(product(*supports))
  return pl.DataFrame(
    rows,
    schema=[spec.name for spec in feature_specs],
    orient="row",
  ).with_columns(pl.all().cast(pl.Int64))


def assign_risk_interval(phi_hat: ArrayF, risk_limits: ArrayF) -> ArrayI:
  """Assign each prediction to 1..(len(risk_limits)+1) risk interval index."""

  idx = np.digitize(
    np.asarray(phi_hat, dtype=np.float64),
    np.asarray(risk_limits, dtype=np.float64),
    right=True,
  )
  return (idx + 1).astype(np.int64)


def estimate_external_prevalence(
  y: ArrayI, interval_idx: ArrayI, n_bins: int
) -> ArrayF:
  """Estimate stratum-specific prevalence; return 0.0 for empty bins."""

  yy = np.asarray(y, dtype=np.float64)
  idx = np.asarray(interval_idx, dtype=np.int64)
  pe = np.zeros(n_bins, dtype=np.float64)

  for b in range(1, n_bins + 1):
    mask = idx == b
    if np.any(mask):
      pe[b - 1] = float(np.mean(yy[mask]))
    else:
      pe[b - 1] = 0.0

  return pe


def build_calibration_artifacts(
  ref_pop: Population, feature_specs: tuple[FeatureSpec, ...]
) -> CalibrationArtifacts:
  """Build risk limits, X-combination interval index, and external prevalence.

  Example
  -------
  >>> import polars as pl
  >>> pop = Population(pl.DataFrame({"X1": [0, 1], "X2": [0, 1], "X3": [0, 1], "X4": [0, 1]}),
  ...                  np.array([0.2, 0.3]), np.array([1, 2]), np.array([0, 1]))
  >>> art = build_calibration_artifacts(pop, (
  ...     FeatureSpec("X1", "categorical_cut", {"breaks": (0, 0.5, 1)}),
  ...     FeatureSpec("X2", "categorical_cut", {"breaks": (0, 0.5, 1)}),
  ...     FeatureSpec("X3", "categorical_cut", {"breaks": (0, 0.5, 1)}),
  ...     FeatureSpec("X4", "categorical_cut", {"breaks": (0, 0.5, 1)}),
  ... ))
  >>> art.risk_limits.shape
  (3,)
  """

  x_cols = [spec.name for spec in feature_specs]
  _, predict_phi = fit_base_risk_model(ref_pop, x_cols=x_cols)

  phi_ref = predict_phi(ref_pop.X)
  risk_limits = compute_risk_limits(phi_ref)

  x_combos = enumerate_x_combinations(feature_specs)
  phi_combos = predict_phi(x_combos)
  x_interval_index = assign_risk_interval(phi_combos, risk_limits)

  ref_interval_index = assign_risk_interval(phi_ref, risk_limits)
  p_external = estimate_external_prevalence(
    ref_pop.y, ref_interval_index, n_bins=len(risk_limits) + 1
  )

  return CalibrationArtifacts(
    risk_limits=np.asarray(risk_limits, dtype=np.float64),
    x_interval_index=np.asarray(x_interval_index, dtype=np.int64),
    p_external=np.asarray(p_external, dtype=np.float64),
  )
