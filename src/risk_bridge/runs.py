from __future__ import annotations

import os
import sys

if __name__ == "__main__" and __package__ is None:
  script_dir = os.path.dirname(os.path.abspath(__file__))
  project_root = os.path.dirname(script_dir)
  if script_dir in sys.path:
    sys.path.remove(script_dir)
  if project_root not in sys.path:
    sys.path.insert(0, project_root)

import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
import json
import multiprocessing as mp
from pathlib import Path
from typing import Callable

from comp_builders import Err, Ok, Result, result
import numpy as np
import polars as pl
from scipy.optimize import minimize
from scipy.stats import poisson
from sklearn.linear_model import LogisticRegression

from risk_bridge.calibration import build_calibration_artifacts, enumerate_x_combinations
from risk_bridge.config import (
  DEFAULT_Z_BINS,
  ExternalCalibrationBootstrapConfig,
  FeatureSpec,
  OptimizationConfig,
  RunConfig,
  SCENARIO1_INIT_THETA,
  SimulationConfig,
  Scenario1PipelineOptions,
  UserDataRunConfig,
  UserDataSchema,
  ZModelSpec,
)
from risk_bridge.constraints import (
  calibration_inequalities,
  calibration_inequalities_jacobian,
  interval_expected_risk,
)
from risk_bridge.likelihood import (
  joint_negative_log_likelihood,
  joint_negative_log_likelihood_grad,
  logistic_risk,
)
from risk_bridge.metrics import (
  calibration_metrics,
  find_threshold_for_fpr,
  roc_auc_binary,
  threshold_metrics,
)
from risk_bridge.optimize import solve_cmle_with_ladder
from risk_bridge.output_schema import OUTPUT_SCHEMA_VERSION
from risk_bridge.preprocess import preprocess_user_dataset
from risk_bridge.sampling import propensity_scores_target_vs_source, psm_sample_source
from risk_bridge.simulate import generate_population
from risk_bridge.tabular import (
  FrameLike,
  append_csv,
  column_to_numpy,
  ensure_polars_frame,
  gather_rows,
  load_tabular_input,
  rows_to_frame,
  select_to_numpy,
  write_tabular_export,
)
from risk_bridge.types import CalibrationArtifacts, FitResult, Population

MetricRow = dict[str, float | int]
ObjectRow = dict[str, object]

CALIBRATION_METRIC_COLUMNS = [
  "iter",
  "estimator",
  "path",
  "calibration_in_the_large",
  "calibration_slope",
  "observed_expected_ratio",
  "brier_score",
]
CALIBRATION_RESIDUAL_COLUMNS = [
  "iter",
  "estimator",
  "path",
  "risk_interval",
  "residual",
  "expected_risk",
  "p_external",
]
FIT_DIAGNOSTIC_COLUMNS = [
  "iter",
  "path",
  "mle_success",
  "cmle_success",
  "mle_status",
  "cmle_status",
  "mle_objective",
  "cmle_objective",
  "cmle_max_violation",
]
ROC_METRIC_COLUMNS = [
  "iter",
  "roc_CML_PSM",
  "roc_CML_RS",
  "roc_ML_PSM",
  "roc_ML_RS",
  "roc_base",
  "roc_ref",
]


def _parse_float_tuple(text: str) -> tuple[float, ...]:
  vals = tuple(float(v.strip()) for v in text.split(",") if v.strip())
  if not vals:
    raise ValueError("Expected at least one numeric value.")
  return vals


def _scenario1_target_feature_specs() -> tuple[FeatureSpec, ...]:
  return (
    FeatureSpec(
      name="X1",
      kind="categorical_cut",
      params={"breaks": (0.0, 0.2, 0.56, 0.9, 1.0)},
    ),
    FeatureSpec(
      name="X2",
      kind="categorical_cut",
      params={"breaks": (0.0, 0.28, 0.83, 1.0)},
    ),
    FeatureSpec(name="X3", kind="capped_poisson", params={"lambda": 0.3, "cap": 2}),
    FeatureSpec(name="X4", kind="capped_poisson", params={"lambda": 0.2, "cap": 2}),
  )


def _scenario23_source_feature_specs() -> tuple[FeatureSpec, ...]:
  return (
    FeatureSpec(
      name="X1",
      kind="categorical_cut",
      params={"breaks": (0.0, 0.05, 0.21, 0.77, 1.0)},
    ),
    FeatureSpec(
      name="X2",
      kind="categorical_cut",
      params={"breaks": (0.0, 0.23, 0.82, 1.0)},
    ),
    FeatureSpec(name="X3", kind="capped_poisson", params={"lambda": 0.5, "cap": 2}),
    FeatureSpec(name="X4", kind="capped_poisson", params={"lambda": 0.2, "cap": 2}),
  )


def _default_z_spec() -> ZModelSpec:
  return ZModelSpec(
    family="trunc_lognormal",
    gamma_init=(-2.0, 0.1, -0.1, 0.1, 0.1, 0.6),
    bins=DEFAULT_Z_BINS,
  )


def _build_simulated_run_config(
  *,
  seed: int,
  nsim: int,
  n_target: int,
  n_source: int,
  n_reference: int,
  sample_size: int,
  target_prevalence: float,
  target_fpr: float,
  alpha: float,
  beta_z: float,
  output_root: str,
  maxiter: int,
  scenario_name: str,
  scenario_run_label: str,
  target_feature_specs: tuple[FeatureSpec, ...],
  source_feature_specs: tuple[FeatureSpec, ...] | None,
  source_miscalibration_a: float,
  source_miscalibration_b: float,
) -> RunConfig:
  sim = SimulationConfig(
    nsim=nsim,
    n_target=n_target,
    n_source=n_source,
    n_reference=n_reference,
    sample_size=sample_size,
    target_prevalence=target_prevalence,
    target_fpr=target_fpr,
    alpha=alpha,
    beta=(-0.5, 0.4, 0.3, 0.65, beta_z),
    feature_specs=target_feature_specs,
    z_spec=_default_z_spec(),
    source_feature_specs=source_feature_specs,
    source_miscalibration_a=source_miscalibration_a,
    source_miscalibration_b=source_miscalibration_b,
    scenario_name=scenario_name,
    scenario_run_label=scenario_run_label,
  )
  opt = OptimizationConfig(
    mle_method="BFGS", cmle_method="trust-constr", tol=1e-6, maxiter=maxiter
  )
  return RunConfig(seed=seed, output_root=output_root, sim=sim, opt=opt)


def _serialize_feature_specs(feature_specs: tuple[FeatureSpec, ...]) -> str:
  return json.dumps(
    [
      {
        "name": spec.name,
        "kind": spec.kind,
        "params": spec.params,
      }
      for spec in feature_specs
    ],
    sort_keys=True,
  )


def build_scenario1_run_config(
  *,
  seed: int = 631,
  nsim: int = 1000,
  n_target: int = 50_000,
  n_source: int = 10_000,
  n_reference: int = 50_000,
  sample_size: int = 1000,
  target_prevalence: float = 0.1,
  target_fpr: float = 0.1,
  alpha: float = -3.8,
  beta_z: float = 0.9,
  output_root: str = "data",
  maxiter: int = 200,
) -> RunConfig:
  """Build a RunConfig matching `PSM_Scenario1_run.R` Scenario-1 defaults."""

  return _build_simulated_run_config(
    seed=seed,
    nsim=nsim,
    n_target=n_target,
    n_source=n_source,
    n_reference=n_reference,
    sample_size=sample_size,
    target_prevalence=target_prevalence,
    target_fpr=target_fpr,
    alpha=alpha,
    beta_z=beta_z,
    output_root=output_root,
    maxiter=maxiter,
    scenario_name="Scenario1",
    scenario_run_label="scenario1",
    target_feature_specs=_scenario1_target_feature_specs(),
    source_feature_specs=None,
    source_miscalibration_a=0.0,
    source_miscalibration_b=1.0,
  )


def build_scenario2_run_config(
  *,
  seed: int = 475,
  nsim: int = 1000,
  n_target: int = 50_000,
  n_source: int = 10_000,
  n_reference: int = 50_000,
  sample_size: int = 1000,
  target_prevalence: float = 0.1,
  target_fpr: float = 0.1,
  alpha: float = -2.38,
  beta_z: float = 0.1,
  output_root: str = "data",
  maxiter: int = 200,
) -> RunConfig:
  """Build a RunConfig matching `PSM_Scenario2_run.R` defaults."""

  return _build_simulated_run_config(
    seed=seed,
    nsim=nsim,
    n_target=n_target,
    n_source=n_source,
    n_reference=n_reference,
    sample_size=sample_size,
    target_prevalence=target_prevalence,
    target_fpr=target_fpr,
    alpha=alpha,
    beta_z=beta_z,
    output_root=output_root,
    maxiter=maxiter,
    scenario_name="Scenario2",
    scenario_run_label="scenario2",
    target_feature_specs=_scenario1_target_feature_specs(),
    source_feature_specs=_scenario23_source_feature_specs(),
    source_miscalibration_a=0.5,
    source_miscalibration_b=1.0,
  )


def build_scenario3_run_config(
  *,
  seed: int = 73,
  nsim: int = 1000,
  n_target: int = 50_000,
  n_source: int = 10_000,
  n_reference: int = 50_000,
  sample_size: int = 1000,
  target_prevalence: float = 0.1,
  target_fpr: float = 0.1,
  alpha: float = -3.8,
  beta_z: float = 0.9,
  output_root: str = "data",
  maxiter: int = 200,
) -> RunConfig:
  """Build a RunConfig matching `PSM_Scenario3_run.R` defaults."""

  return _build_simulated_run_config(
    seed=seed,
    nsim=nsim,
    n_target=n_target,
    n_source=n_source,
    n_reference=n_reference,
    sample_size=sample_size,
    target_prevalence=target_prevalence,
    target_fpr=target_fpr,
    alpha=alpha,
    beta_z=beta_z,
    output_root=output_root,
    maxiter=maxiter,
    scenario_name="Scenario3",
    scenario_run_label="scenario3",
    target_feature_specs=_scenario1_target_feature_specs(),
    source_feature_specs=_scenario23_source_feature_specs(),
    source_miscalibration_a=-0.5,
    source_miscalibration_b=1.2,
  )


def _validated_init_theta(
  init_theta: tuple[float, ...] | np.ndarray, *, x_cols: list[str]
) -> np.ndarray:
  expected_theta_len = 1 + (len(x_cols) + 1) + (len(x_cols) + 2)
  theta0 = np.asarray(init_theta, dtype=np.float64)
  if len(theta0) != expected_theta_len:
    raise ValueError(
      f"init_theta length must be {expected_theta_len}; got {len(theta0)}"
    )
  return theta0


def _population_to_df(pop: Population) -> pl.DataFrame:
  return pop.X.with_columns(
    pl.Series("caseY", pop.y.astype(np.int64)),
    pl.Series("zOrigin", pop.z_cont.astype(np.float64)),
    pl.Series("zCat", pop.z_cat.astype(np.int64)),
  ).select(["caseY", *pop.X.columns, "zOrigin", "zCat"])


def _population_from_df(df: FrameLike, x_cols: list[str]) -> Population:
  frame = ensure_polars_frame(df, clone=False)
  return Population(
    X=frame.select(x_cols).with_columns(pl.all().cast(pl.Int64)),
    z_cont=column_to_numpy(frame, "zOrigin", dtype=np.float64),
    z_cat=column_to_numpy(frame, "zCat", dtype=np.int64),
    y=column_to_numpy(frame, "caseY", dtype=np.int64),
  )


def _feature_specs_from_data(
  reference_df: pl.DataFrame,
  x_cols: list[str],
  max_levels_per_feature: int = 20,
  max_total_combinations: int = 50_000,
) -> tuple[FeatureSpec, ...]:
  specs: list[FeatureSpec] = []
  total_combinations = 1

  for col in x_cols:
    values = tuple(
      int(v)
      for v in reference_df.get_column(col).cast(pl.Int64).unique().sort().to_list()
    )
    if not values:
      raise ValueError(f"No observed values for {col} in reference data.")
    if len(values) > max_levels_per_feature:
      raise ValueError(
        f"{col} has {len(values)} levels; cap is {max_levels_per_feature}. "
        "Please recode/bucket levels for tractable discrete support."
      )
    total_combinations *= len(values)
    specs.append(FeatureSpec(name=col, kind="custom", params={"values": values}))

  if total_combinations > max_total_combinations:
    raise ValueError(
      "Cartesian X support is too large for current calibration enumeration "
      f"({total_combinations} > {max_total_combinations}). "
      "Please reduce feature cardinality."
    )

  return tuple(specs)


def _x_probability_external_from_reference(
  reference_df: pl.DataFrame, x_combos: pl.DataFrame
) -> np.ndarray:
  x_cols = list(x_combos.columns)
  total = max(float(len(reference_df)), 1.0)
  probs = np.asarray(
    x_combos.join(
      reference_df.group_by(x_cols).agg(pl.len().alias("__count")),
      on=x_cols,
      how="left",
    )
    .get_column("__count")
    .fill_null(0)
    .cast(pl.Float64)
    .to_numpy(),
    dtype=np.float64,
  ) / total

  s = probs.sum()
  if s <= 0:
    return np.full(len(probs), 1.0 / len(probs), dtype=np.float64)
  return probs / s


def _fit_probability_model(
  x: np.ndarray, y: np.ndarray
) -> Callable[[np.ndarray], np.ndarray]:
  y_int = np.asarray(y, dtype=np.int64)
  uniq = np.unique(y_int)
  if len(uniq) < 2:
    p = float(uniq[0])

    def predict_fn(x_new: np.ndarray) -> np.ndarray:
      return np.full(len(x_new), p, dtype=np.float64)

    return predict_fn

  model = LogisticRegression(max_iter=1000, solver="lbfgs")
  model.fit(x, y_int)

  def predict_fn(x_new: np.ndarray) -> np.ndarray:
    return model.predict_proba(x_new)[:, 1].astype(np.float64)

  return predict_fn


def _fit_unconstrained(
  theta0: np.ndarray,
  x: np.ndarray,
  y: np.ndarray,
  z: np.ndarray,
  z_cat: np.ndarray,
  maxiter: int,
) -> FitResult:
  def objective(t: np.ndarray) -> float:
    return joint_negative_log_likelihood(t, x, y, z, z_cat)

  def jacobian(t: np.ndarray) -> np.ndarray:
    return joint_negative_log_likelihood_grad(t, x, y, z, z_cat)

  res = minimize(
    objective, theta0, jac=jacobian, method="BFGS", options={"maxiter": maxiter}
  )
  return FitResult(
    theta=np.asarray(res.x, dtype=np.float64),
    success=bool(res.success),
    status=str(res.message),
    n_iter=int(getattr(res, "nit", 0) or 0),
    objective=float(res.fun),
    diagnostics={"solver": "BFGS", "status_code": int(getattr(res, "status", -1))},
  )


def _feature_probability_map(spec: FeatureSpec) -> dict[int, float]:
  if spec.kind == "categorical_cut":
    breaks = np.asarray(spec.params["breaks"], dtype=np.float64)
    probs = np.diff(breaks)
    probs = probs / probs.sum()
    return {i: float(p) for i, p in enumerate(probs)}

  if spec.kind == "capped_poisson":
    lam = float(spec.params["lambda"])
    cap = int(spec.params.get("cap", 2))
    probs = [float(poisson.pmf(k, lam)) for k in range(cap)]
    probs.append(float(1.0 - poisson.cdf(cap - 1, lam)))
    return {i: p for i, p in enumerate(probs)}

  if spec.kind == "custom":
    values = [int(v) for v in spec.params["values"]]
    probs = np.asarray(spec.params.get("probs", [1.0] * len(values)), dtype=np.float64)
    probs = probs / probs.sum()
    return {values[i]: float(probs[i]) for i in range(len(values))}

  raise ValueError(f"Unsupported feature kind: {spec.kind}")


def _x_probability_from_specs(
  feature_specs: tuple[FeatureSpec, ...], x_combos: pl.DataFrame
) -> np.ndarray:
  probs = np.ones(len(x_combos), dtype=np.float64)

  for spec in feature_specs:
    pmap = _feature_probability_map(spec)
    vals = column_to_numpy(x_combos, spec.name, dtype=np.int64)
    this_prob = np.array([pmap.get(int(v), 0.0) for v in vals], dtype=np.float64)
    probs *= this_prob

  s = probs.sum()
  if s <= 0:
    return np.full(len(probs), 1.0 / max(len(probs), 1), dtype=np.float64)
  return probs / s


def _theta_columns(x_cols: list[str]) -> list[str]:
  beta_cols = [f"beta_{name}" for name in x_cols]
  gamma_cols = [f"gamma_{name}" for name in x_cols]
  return ["alpha", *beta_cols, "beta_Zcat", "gamma_0", *gamma_cols, "gamma_sigma"]


def _flush_intermediate_buffers(
  *,
  buffers: dict[str, list[pl.DataFrame]],
  intermediate_dir: Path,
) -> None:
  path_map = {
    "xind": intermediate_dir / "xind_by_iter.csv",
    "pe": intermediate_dir / "pe_by_iter.csv",
    "sample_psm": intermediate_dir / "sample_psm_by_iter.csv",
    "sample_rs": intermediate_dir / "sample_rs_by_iter.csv",
  }
  for key, frames in buffers.items():
    if not frames:
      continue
    append_csv(pl.concat(frames, how="vertical_relaxed"), path_map[key])
    frames.clear()


def _fit_path(
  sample_df: pl.DataFrame,
  theta0: np.ndarray,
  x_cols: list[str],
  x_combs: np.ndarray,
  x_prob_external: np.ndarray,
  x_interval_index: np.ndarray,
  p_external: np.ndarray,
  tempcateg: np.ndarray,
  maxiter: int,
  feasibility_tol: float,
  calibration_tolerance: float,
) -> tuple[FitResult, FitResult]:
  x = select_to_numpy(sample_df, x_cols, dtype=np.float64)
  y = column_to_numpy(sample_df, "caseY", dtype=np.int64)
  z = column_to_numpy(sample_df, "zOrigin", dtype=np.float64)
  z_cat = column_to_numpy(sample_df, "zCat", dtype=np.int64)

  mle = _fit_unconstrained(theta0=theta0, x=x, y=y, z=z, z_cat=z_cat, maxiter=maxiter)

  def objective_fn(theta: np.ndarray) -> float:
    return joint_negative_log_likelihood(theta, x, y, z, z_cat)

  def objective_jac_fn(theta: np.ndarray) -> np.ndarray:
    return joint_negative_log_likelihood_grad(theta, x, y, z, z_cat)

  def constraints_fn(theta: np.ndarray) -> np.ndarray:
    return calibration_inequalities(
      theta=theta,
      x_combs=x_combs,
      x_prob_external=x_prob_external,
      calibration_index=x_interval_index,
      y_external=p_external,
      tolerance=calibration_tolerance,
      tempcateg=tempcateg,
    )

  def constraints_jac_fn(theta: np.ndarray) -> np.ndarray:
    return calibration_inequalities_jacobian(
      theta=theta,
      x_combs=x_combs,
      x_prob_external=x_prob_external,
      calibration_index=x_interval_index,
      y_external=p_external,
      tempcateg=tempcateg,
    )

  cmle_start = mle.theta if np.all(np.isfinite(mle.theta)) else theta0
  cmle, _ = solve_cmle_with_ladder(
    theta0=cmle_start,
    objective_fn=objective_fn,
    objective_jac_fn=objective_jac_fn,
    constraints_fn=constraints_fn,
    constraints_jac_fn=constraints_jac_fn,
    maxiter=maxiter,
    feasibility_tol=feasibility_tol,
  )
  return mle, cmle


def _fit_two_paths(
  *,
  sample_psm_export: pl.DataFrame,
  sample_rs_export: pl.DataFrame,
  theta0: np.ndarray,
  x_cols: list[str],
  x_combs_np: np.ndarray,
  x_prob_external: np.ndarray,
  x_interval_index: np.ndarray,
  p_external: np.ndarray,
  tempcateg: np.ndarray,
  maxiter: int,
  feasibility_tol: float,
  calibration_tolerance: float,
  path_jobs: int,
) -> tuple[FitResult, FitResult, FitResult, FitResult]:
  psm_kwargs = {
    "sample_df": sample_psm_export,
    "theta0": theta0,
    "x_cols": x_cols,
    "x_combs": x_combs_np,
    "x_prob_external": x_prob_external,
    "x_interval_index": x_interval_index,
    "p_external": p_external,
    "tempcateg": tempcateg,
    "maxiter": maxiter,
    "feasibility_tol": feasibility_tol,
    "calibration_tolerance": calibration_tolerance,
  }
  rs_kwargs = {
    "sample_df": sample_rs_export,
    "theta0": theta0,
    "x_cols": x_cols,
    "x_combs": x_combs_np,
    "x_prob_external": x_prob_external,
    "x_interval_index": x_interval_index,
    "p_external": p_external,
    "tempcateg": tempcateg,
    "maxiter": maxiter,
    "feasibility_tol": feasibility_tol,
    "calibration_tolerance": calibration_tolerance,
  }

  if path_jobs > 1:
    with ThreadPoolExecutor(max_workers=2) as pool:
      f_psm = pool.submit(_fit_path, **psm_kwargs)
      f_rs = pool.submit(_fit_path, **rs_kwargs)
      mle_psm, cmle_psm = f_psm.result()
      mle_rs, cmle_rs = f_rs.result()
    return mle_psm, cmle_psm, mle_rs, cmle_rs

  mle_psm, cmle_psm = _fit_path(**psm_kwargs)
  mle_rs, cmle_rs = _fit_path(**rs_kwargs)
  return mle_psm, cmle_psm, mle_rs, cmle_rs


def _theta_row_dict(
  *,
  iteration: int,
  theta_cols: list[str],
  theta: np.ndarray,
) -> MetricRow:
  row: MetricRow = {"iter": iteration}
  for c, v in zip(theta_cols, theta, strict=True):
    row[c] = float(v)
  return row


def _accuracy_row(iteration: int, metrics: dict[str, float]) -> MetricRow:
  return {
    "iter": iteration,
    "TPR": metrics["tpr"],
    "PPV": metrics["ppv"],
    "TNR": metrics["tnr"],
  }


def _evaluate_theta_on_roc(
  *,
  theta: np.ndarray,
  x_cols: list[str],
  roc_x: np.ndarray,
  roc_zc: np.ndarray,
  roc_y: np.ndarray,
  threshold: float,
) -> tuple[float, dict[str, float], np.ndarray]:
  px = len(x_cols)
  alpha = float(theta[0])
  beta = np.asarray(theta[1 : 1 + px + 1], dtype=np.float64)
  score = logistic_risk(alpha, beta, roc_x, roc_zc)
  return (
    roc_auc_binary(roc_y, score),
    threshold_metrics(roc_y, score, threshold=threshold),
    np.asarray(score, dtype=np.float64),
  )


@dataclass(frozen=True)
class PathFits:
  mle_psm: FitResult
  cmle_psm: FitResult
  mle_rs: FitResult
  cmle_rs: FitResult


@dataclass(frozen=True)
class EvaluationInputs:
  iteration: int
  x_cols: list[str]
  roc_x: np.ndarray
  roc_zc: np.ndarray
  roc_y: np.ndarray
  threshold: float
  base_score: np.ndarray
  ref_score: np.ndarray


@dataclass(frozen=True)
class ResidualExportInputs:
  x_combs: np.ndarray
  x_prob_external: np.ndarray
  x_interval_index: np.ndarray
  p_external: np.ndarray
  tempcateg: np.ndarray


@dataclass(frozen=True)
class IterationExportRows:
  est_cml_psm_row: MetricRow
  est_cml_rs_row: MetricRow
  est_ml_psm_row: MetricRow
  est_ml_rs_row: MetricRow
  acc_cml_psm_row: MetricRow
  acc_cml_rs_row: MetricRow
  acc_ml_psm_row: MetricRow
  acc_ml_rs_row: MetricRow
  roc_row: MetricRow
  fit_diag_rows: tuple[ObjectRow, ObjectRow]
  calibration_metric_rows: tuple[ObjectRow, ...]
  calibration_residual_rows: tuple[ObjectRow, ...]


def _path_fit_diagnostics(
  *, iteration: int, path: str, mle: FitResult, cmle: FitResult
) -> ObjectRow:
  return {
    "iter": iteration,
    "path": path,
    "mle_success": mle.success,
    "cmle_success": cmle.success,
    "mle_status": mle.status,
    "cmle_status": cmle.status,
    "mle_objective": mle.objective,
    "cmle_objective": cmle.objective,
    "cmle_max_violation": cmle.diagnostics.get("max_violation"),
  }


def _calibration_metric_row(
  *,
  iteration: int,
  estimator: str,
  path: str,
  y_true: np.ndarray,
  predictions: np.ndarray,
) -> ObjectRow:
  metrics = calibration_metrics(y_true, predictions)
  return {
    "iter": iteration,
    "estimator": estimator,
    "path": path,
    "calibration_in_the_large": metrics["calibration_in_the_large"],
    "calibration_slope": metrics["calibration_slope"],
    "observed_expected_ratio": metrics["observed_expected_ratio"],
    "brier_score": metrics["brier_score"],
  }


def _calibration_residual_rows_for_theta(
  *,
  iteration: int,
  estimator: str,
  path: str,
  theta: np.ndarray,
  residual_inputs: ResidualExportInputs,
) -> list[ObjectRow]:
  p_external = np.asarray(residual_inputs.p_external, dtype=np.float64)
  expected = interval_expected_risk(
    theta=theta,
    x_combs=residual_inputs.x_combs,
    x_prob_external=residual_inputs.x_prob_external,
    calibration_index=residual_inputs.x_interval_index,
    n_bins=len(p_external),
    tempcateg=residual_inputs.tempcateg,
  )
  residuals = expected - p_external
  rows: list[ObjectRow] = []
  for bin_idx, (residual, expected_risk, p_ext) in enumerate(
    zip(residuals, expected, p_external, strict=True), start=1
  ):
    rows.append(
      {
        "iter": iteration,
        "estimator": estimator,
        "path": path,
        "risk_interval": bin_idx,
        "residual": float(residual),
        "expected_risk": float(expected_risk),
        "p_external": float(p_ext),
      }
    )
  return rows


def _iteration_export_rows(
  *,
  fits: PathFits,
  theta_cols: list[str],
  eval_inputs: EvaluationInputs,
  residual_inputs: ResidualExportInputs,
) -> IterationExportRows:
  iteration = eval_inputs.iteration
  roc_cml_psm, acc_cml_psm, pred_cml_psm = _evaluate_theta_on_roc(
    theta=fits.cmle_psm.theta,
    x_cols=eval_inputs.x_cols,
    roc_x=eval_inputs.roc_x,
    roc_zc=eval_inputs.roc_zc,
    roc_y=eval_inputs.roc_y,
    threshold=eval_inputs.threshold,
  )
  roc_cml_rs, acc_cml_rs, pred_cml_rs = _evaluate_theta_on_roc(
    theta=fits.cmle_rs.theta,
    x_cols=eval_inputs.x_cols,
    roc_x=eval_inputs.roc_x,
    roc_zc=eval_inputs.roc_zc,
    roc_y=eval_inputs.roc_y,
    threshold=eval_inputs.threshold,
  )
  roc_ml_psm, acc_ml_psm, pred_ml_psm = _evaluate_theta_on_roc(
    theta=fits.mle_psm.theta,
    x_cols=eval_inputs.x_cols,
    roc_x=eval_inputs.roc_x,
    roc_zc=eval_inputs.roc_zc,
    roc_y=eval_inputs.roc_y,
    threshold=eval_inputs.threshold,
  )
  roc_ml_rs, acc_ml_rs, pred_ml_rs = _evaluate_theta_on_roc(
    theta=fits.mle_rs.theta,
    x_cols=eval_inputs.x_cols,
    roc_x=eval_inputs.roc_x,
    roc_zc=eval_inputs.roc_zc,
    roc_y=eval_inputs.roc_y,
    threshold=eval_inputs.threshold,
  )

  path_specs: tuple[tuple[str, str, FitResult, np.ndarray], ...] = (
    ("cMLE", "PSM", fits.cmle_psm, pred_cml_psm),
    ("cMLE", "RS", fits.cmle_rs, pred_cml_rs),
    ("ML", "PSM", fits.mle_psm, pred_ml_psm),
    ("ML", "RS", fits.mle_rs, pred_ml_rs),
  )
  calibration_metric_rows = tuple(
    _calibration_metric_row(
      iteration=iteration,
      estimator=estimator,
      path=path,
      y_true=eval_inputs.roc_y,
      predictions=predictions,
    )
    for estimator, path, _fit, predictions in path_specs
  )
  residual_row_lists = [
    _calibration_residual_rows_for_theta(
      iteration=iteration,
      estimator=estimator,
      path=path,
      theta=fit.theta,
      residual_inputs=residual_inputs,
    )
    for estimator, path, fit, _predictions in path_specs
  ]
  calibration_residual_rows = tuple(
    row for rows in residual_row_lists for row in rows
  )

  return IterationExportRows(
    est_cml_psm_row=_theta_row_dict(
      iteration=iteration, theta_cols=theta_cols, theta=fits.cmle_psm.theta
    ),
    est_cml_rs_row=_theta_row_dict(
      iteration=iteration, theta_cols=theta_cols, theta=fits.cmle_rs.theta
    ),
    est_ml_psm_row=_theta_row_dict(
      iteration=iteration, theta_cols=theta_cols, theta=fits.mle_psm.theta
    ),
    est_ml_rs_row=_theta_row_dict(
      iteration=iteration, theta_cols=theta_cols, theta=fits.mle_rs.theta
    ),
    acc_cml_psm_row=_accuracy_row(iteration, acc_cml_psm),
    acc_cml_rs_row=_accuracy_row(iteration, acc_cml_rs),
    acc_ml_psm_row=_accuracy_row(iteration, acc_ml_psm),
    acc_ml_rs_row=_accuracy_row(iteration, acc_ml_rs),
    roc_row={
      "iter": iteration,
      "roc_CML_PSM": roc_cml_psm,
      "roc_CML_RS": roc_cml_rs,
      "roc_ML_PSM": roc_ml_psm,
      "roc_ML_RS": roc_ml_rs,
      "roc_base": roc_auc_binary(eval_inputs.roc_y, eval_inputs.base_score),
      "roc_ref": roc_auc_binary(eval_inputs.roc_y, eval_inputs.ref_score),
    },
    fit_diag_rows=(
      _path_fit_diagnostics(
        iteration=iteration, path="PSM", mle=fits.mle_psm, cmle=fits.cmle_psm
      ),
      _path_fit_diagnostics(
        iteration=iteration, path="RS", mle=fits.mle_rs, cmle=fits.cmle_rs
      ),
    ),
    calibration_metric_rows=calibration_metric_rows,
    calibration_residual_rows=calibration_residual_rows,
  )


@dataclass(frozen=True)
class UserDataPreparedContext:
  target_clean: pl.DataFrame
  source_clean: pl.DataFrame
  reference_clean: pl.DataFrame
  x_cols: list[str]
  export_cols: list[str]
  theta_cols: list[str]
  z_bins_arr: np.ndarray
  theta0: np.ndarray
  x_combos_np: np.ndarray
  x_prob_external: np.ndarray
  artifacts: CalibrationArtifacts
  base_predict: Callable[[np.ndarray], np.ndarray]
  ref_predict: Callable[[np.ndarray], np.ndarray]
  target_y: np.ndarray
  target_ref_score: np.ndarray
  fixed_threshold: float


def _sample_export_with_iter(
  sample_df: pl.DataFrame, *, iteration: int, export_cols: list[str]
) -> pl.DataFrame:
  return sample_df.select(pl.lit(iteration).alias("iter"), *export_cols)


def _prepare_user_data_context(config: UserDataRunConfig) -> UserDataPreparedContext:
  target_df = config.target_df
  source_df = config.source_df
  reference_df = config.reference_df
  x_cols = list(config.schema.x_cols)
  y_col = config.schema.y_col
  z_origin_col = config.schema.z_origin_col
  z_cat_col = config.schema.z_cat_col
  allow_z_origin_from_z_cat = config.schema.allow_z_origin_from_z_cat
  z_bins_arr = np.asarray(config.schema.z_bins, dtype=np.float64)

  target_clean = preprocess_user_dataset(
    target_df,
    x_cols=x_cols,
    z_bins=z_bins_arr,
    dataset_name="target_df",
    y_col=y_col,
    z_origin_col=z_origin_col,
    z_cat_col=z_cat_col,
    allow_z_origin_from_z_cat=allow_z_origin_from_z_cat,
  )
  source_clean = preprocess_user_dataset(
    source_df,
    x_cols=x_cols,
    z_bins=z_bins_arr,
    dataset_name="source_df",
    y_col=y_col,
    z_origin_col=z_origin_col,
    z_cat_col=z_cat_col,
    allow_z_origin_from_z_cat=allow_z_origin_from_z_cat,
  )
  reference_clean = preprocess_user_dataset(
    reference_df,
    x_cols=x_cols,
    z_bins=z_bins_arr,
    dataset_name="reference_df",
    y_col=y_col,
    z_origin_col=z_origin_col,
    z_cat_col=z_cat_col,
    allow_z_origin_from_z_cat=allow_z_origin_from_z_cat,
  )

  if config.sample_size > len(target_clean):
    raise ValueError("sample_size must be <= number of rows in target_df")
  if config.sample_size > len(source_clean):
    raise ValueError("sample_size must be <= number of rows in source_df")

  max_z_cat = int(
    max(
      target_clean.get_column("zCat").max(),
      source_clean.get_column("zCat").max(),
      reference_clean.get_column("zCat").max(),
    )
  )
  if max_z_cat > len(z_bins_arr):
    raise ValueError(
      "zCat level exceeds supported range for provided z_bins. "
      "Expected zCat in [0, len(z_bins)]."
    )

  feature_specs = _feature_specs_from_data(reference_clean, x_cols)
  x_combos = enumerate_x_combinations(feature_specs)
  x_combos_np = x_combos.to_numpy().astype(np.float64)
  x_prob_external = _x_probability_external_from_reference(
    reference_clean.select(x_cols), x_combos
  )
  reference_pop = _population_from_df(reference_clean, x_cols)
  artifacts = build_calibration_artifacts(reference_pop, feature_specs)

  theta0 = _validated_init_theta(config.init_theta, x_cols=x_cols)
  export_cols = ["caseY", *x_cols, "zOrigin", "zCat"]
  theta_cols = _theta_columns(x_cols)

  ref_predict = _fit_probability_model(
    select_to_numpy(target_clean, [*x_cols, "zCat"], dtype=np.float64),
    column_to_numpy(target_clean, "caseY", dtype=np.int64),
  )
  base_predict = _fit_probability_model(
    select_to_numpy(reference_clean, x_cols, dtype=np.float64),
    column_to_numpy(reference_clean, "caseY", dtype=np.int64),
  )
  target_y = column_to_numpy(target_clean, "caseY", dtype=np.int64)
  target_ref_score = ref_predict(
    select_to_numpy(target_clean, [*x_cols, "zCat"], dtype=np.float64)
  )
  fixed_threshold = find_threshold_for_fpr(
    target_y, target_ref_score, target_fpr=config.target_fpr
  )

  return UserDataPreparedContext(
    target_clean=target_clean,
    source_clean=source_clean,
    reference_clean=reference_clean,
    x_cols=x_cols,
    export_cols=export_cols,
    theta_cols=theta_cols,
    z_bins_arr=z_bins_arr,
    theta0=theta0,
    x_combos_np=x_combos_np,
    x_prob_external=x_prob_external,
    artifacts=artifacts,
    base_predict=base_predict,
    ref_predict=ref_predict,
    target_y=target_y,
    target_ref_score=target_ref_score,
    fixed_threshold=fixed_threshold,
  )


def _prepare_user_data_iteration_inputs(
  context: UserDataPreparedContext,
  *,
  rng: np.random.Generator,
  sample_size: int,
  iteration: int,
) -> dict[str, object]:
  prevalence_row = {
    "iter": iteration,
    "prevalence_target": float(np.mean(context.target_y)),
  }

  roc_idx = np.asarray(
    rng.choice(len(context.target_clean), size=sample_size, replace=False), dtype=np.int64
  )
  roc_df = gather_rows(context.target_clean, roc_idx)
  roc_x = select_to_numpy(roc_df, context.x_cols, dtype=np.float64)
  roc_zc = column_to_numpy(roc_df, "zCat", dtype=np.int64)
  roc_y = column_to_numpy(roc_df, "caseY", dtype=np.int64)

  xind_df = rows_to_frame(
    [],
    columns=["iter", "x_combination", "risk_interval"],
  )
  if len(context.artifacts.x_interval_index):
    xind_df = pl.DataFrame(
      {
        "iter": np.full(len(context.artifacts.x_interval_index), iteration, dtype=np.int64),
        "x_combination": np.arange(
          1, len(context.artifacts.x_interval_index) + 1, dtype=np.int64
        ),
        "risk_interval": np.asarray(
          context.artifacts.x_interval_index, dtype=np.int64
        ),
      }
    )
  pe_df = pl.DataFrame(
    {
      "iter": np.full(len(context.artifacts.p_external), iteration, dtype=np.int64),
      "risk_interval": np.arange(1, len(context.artifacts.p_external) + 1, dtype=np.int64),
      "p_external": np.asarray(context.artifacts.p_external, dtype=np.float64),
    }
  )

  pooled = propensity_scores_target_vs_source(
    context.target_clean, context.source_clean, covariates=["caseY", *context.x_cols]
  )
  sample_psm_export = psm_sample_source(rng, pooled, sample_size=sample_size).select(
    context.export_cols
  )

  rs_idx = np.asarray(
    rng.choice(len(context.source_clean), size=sample_size, replace=False), dtype=np.int64
  )
  sample_rs_export = gather_rows(context.source_clean, rs_idx).select(context.export_cols)

  threshold_row = {
    "iter": iteration,
    "threshold_ref_fpr": float(context.fixed_threshold),
  }
  base_score = context.base_predict(select_to_numpy(roc_df, context.x_cols, dtype=np.float64))
  ref_score = context.ref_predict(
    select_to_numpy(roc_df, [*context.x_cols, "zCat"], dtype=np.float64)
  )
  acc_base = threshold_metrics(roc_y, base_score, threshold=context.fixed_threshold)
  acc_ref = threshold_metrics(
    context.target_y, context.target_ref_score, threshold=context.fixed_threshold
  )
  acc_base_row = _accuracy_row(iteration, acc_base)
  acc_ref_row = _accuracy_row(iteration, acc_ref)

  return {
    "prevalence_row": prevalence_row,
    "roc_df": roc_df,
    "roc_x": roc_x,
    "roc_zc": roc_zc,
    "roc_y": roc_y,
    "xind_df": xind_df,
    "pe_df": pe_df,
    "sample_psm_export": sample_psm_export,
    "sample_rs_export": sample_rs_export,
    "sample_psm_df": _sample_export_with_iter(
      sample_psm_export, iteration=iteration, export_cols=context.export_cols
    ),
    "sample_rs_df": _sample_export_with_iter(
      sample_rs_export, iteration=iteration, export_cols=context.export_cols
    ),
    "threshold_row": threshold_row,
    "acc_base_row": acc_base_row,
    "acc_ref_row": acc_ref_row,
    "base_score": base_score,
    "ref_score": ref_score,
  }


def run_user_data_prefit_path(
  config: UserDataRunConfig, *, iteration_seed_offset: int = 0
) -> dict[str, object]:
  """Run the user-data preprocessing/sampling path without solver fitting."""

  context = _prepare_user_data_context(config)
  rng = np.random.default_rng(config.seed + iteration_seed_offset)
  return _prepare_user_data_iteration_inputs(
    context, rng=rng, sample_size=config.sample_size, iteration=1
  )


def _run_parallel_sim_iteration(task: dict[str, object]) -> dict[str, object]:
  iteration = int(task["iteration"])
  rng = np.random.default_rng(int(task["seed"]))
  target_feature_specs = task["target_feature_specs"]
  source_feature_specs = task["source_feature_specs"]
  n_target = int(task["n_target"])
  n_source = int(task["n_source"])
  n_reference = int(task["n_reference"])
  sample_size = int(task["sample_size"])
  x_cols = list(task["x_cols"])
  export_cols = list(task["export_cols"])
  theta_cols = list(task["theta_cols"])
  x_combs_np = np.asarray(task["x_combs_np"], dtype=np.float64)
  x_prob_external = np.asarray(task["x_prob_external"], dtype=np.float64)
  z_bins = np.asarray(task["z_bins"], dtype=np.float64)
  gamma = np.asarray(task["gamma"], dtype=np.float64)
  theta0 = np.asarray(task["theta0"], dtype=np.float64)
  beta_target = np.asarray(task["beta_target"], dtype=np.float64)
  beta_source = np.asarray(task["beta_source"], dtype=np.float64)
  alpha_target = float(task["alpha_target"])
  alpha_source = float(task["alpha_source"])
  maxiter = int(task["maxiter"])
  feasibility_tol = float(task["feasibility_tol"])
  calibration_tolerance = float(task["calibration_tolerance"])
  target_fpr = float(task["target_fpr"])
  path_jobs = int(task.get("path_jobs", 1))

  target = generate_population(
    rng=rng,
    n=n_target,
    feature_specs=target_feature_specs,
    gamma=gamma,
    z_bins=z_bins,
    alpha=alpha_target,
    beta=beta_target,
  )
  source = generate_population(
    rng=rng,
    n=n_source,
    feature_specs=source_feature_specs,
    gamma=gamma,
    z_bins=z_bins,
    alpha=alpha_source,
    beta=beta_source,
  )
  reference = generate_population(
    rng=rng,
    n=n_reference,
    feature_specs=target_feature_specs,
    gamma=gamma,
    z_bins=z_bins,
    alpha=alpha_target,
    beta=beta_target,
  )

  target_df = _population_to_df(target)
  source_df = _population_to_df(source)
  reference_df = _population_to_df(reference)

  prevalence_row = {
    "iter": iteration,
    "prevalence_target": float(np.mean(column_to_numpy(target_df, "caseY", dtype=np.int64))),
  }

  roc_idx = np.asarray(rng.choice(len(target_df), size=sample_size, replace=False), dtype=np.int64)
  roc_df = gather_rows(target_df, roc_idx)
  roc_x = select_to_numpy(roc_df, x_cols, dtype=np.float64)
  roc_zc = column_to_numpy(roc_df, "zCat", dtype=np.int64)
  roc_y = column_to_numpy(roc_df, "caseY", dtype=np.int64)

  artifacts = build_calibration_artifacts(reference, target_feature_specs)
  xind_df = rows_to_frame(
    [],
    columns=["iter", "x_combination", "risk_interval"],
  )
  if len(artifacts.x_interval_index):
    xind_df = pl.DataFrame(
      {
        "iter": np.full(len(artifacts.x_interval_index), iteration, dtype=np.int64),
        "x_combination": np.arange(1, len(artifacts.x_interval_index) + 1, dtype=np.int64),
        "risk_interval": np.asarray(artifacts.x_interval_index, dtype=np.int64),
      }
    )
  pe_df = pl.DataFrame(
    {
      "iter": np.full(len(artifacts.p_external), iteration, dtype=np.int64),
      "risk_interval": np.arange(1, len(artifacts.p_external) + 1, dtype=np.int64),
      "p_external": np.asarray(artifacts.p_external, dtype=np.float64),
    }
  )

  pooled = propensity_scores_target_vs_source(
    target_df, source_df, covariates=["caseY", *x_cols]
  )
  sample_psm = psm_sample_source(rng, pooled, sample_size=sample_size)
  sample_psm_export = sample_psm.select(export_cols)

  rs_idx = np.asarray(rng.choice(len(source_df), size=sample_size, replace=False), dtype=np.int64)
  sample_rs_export = gather_rows(source_df, rs_idx).select(export_cols)

  sample_psm_df = _sample_export_with_iter(
    sample_psm_export, iteration=iteration, export_cols=export_cols
  )
  sample_rs_df = _sample_export_with_iter(
    sample_rs_export, iteration=iteration, export_cols=export_cols
  )

  ref_predict = _fit_probability_model(
    select_to_numpy(target_df, [*x_cols, "zCat"], dtype=np.float64),
    column_to_numpy(target_df, "caseY", dtype=np.int64),
  )
  base_predict = _fit_probability_model(
    select_to_numpy(reference_df, x_cols, dtype=np.float64),
    column_to_numpy(reference_df, "caseY", dtype=np.int64),
  )

  target_y = column_to_numpy(target_df, "caseY", dtype=np.int64)
  target_ref_score = ref_predict(
    select_to_numpy(target_df, [*x_cols, "zCat"], dtype=np.float64)
  )
  threshold = find_threshold_for_fpr(target_y, target_ref_score, target_fpr=target_fpr)
  threshold_row = {"iter": iteration, "threshold_ref_fpr": float(threshold)}

  base_score = base_predict(select_to_numpy(roc_df, x_cols, dtype=np.float64))
  ref_score = ref_predict(select_to_numpy(roc_df, [*x_cols, "zCat"], dtype=np.float64))
  acc_base = threshold_metrics(roc_y, base_score, threshold=threshold)
  acc_ref = threshold_metrics(target_y, target_ref_score, threshold=threshold)
  acc_base_row = _accuracy_row(iteration, acc_base)
  acc_ref_row = _accuracy_row(iteration, acc_ref)

  mle_psm, cmle_psm, mle_rs, cmle_rs = _fit_two_paths(
    sample_psm_export=sample_psm_export,
    sample_rs_export=sample_rs_export,
    theta0=theta0,
    x_cols=x_cols,
    x_combs_np=x_combs_np,
    x_prob_external=x_prob_external,
    x_interval_index=artifacts.x_interval_index,
    p_external=artifacts.p_external,
    tempcateg=z_bins,
    maxiter=maxiter,
    feasibility_tol=feasibility_tol,
    calibration_tolerance=calibration_tolerance,
    path_jobs=path_jobs,
  )
  rows = _iteration_export_rows(
    fits=PathFits(
      mle_psm=mle_psm, cmle_psm=cmle_psm, mle_rs=mle_rs, cmle_rs=cmle_rs
    ),
    theta_cols=theta_cols,
    eval_inputs=EvaluationInputs(
      iteration=iteration,
      x_cols=x_cols,
      roc_x=roc_x,
      roc_zc=roc_zc,
      roc_y=roc_y,
      threshold=threshold,
      base_score=base_score,
      ref_score=ref_score,
    ),
    residual_inputs=ResidualExportInputs(
      x_combs=x_combs_np,
      x_prob_external=x_prob_external,
      x_interval_index=artifacts.x_interval_index,
      p_external=artifacts.p_external,
      tempcateg=z_bins,
    ),
  )

  return {
    "iter": iteration,
    "xind_rows": xind_df.to_dicts(),
    "pe_rows": pe_df.to_dicts(),
    "sample_psm_rows": sample_psm_df.to_dicts(),
    "sample_rs_rows": sample_rs_df.to_dicts(),
    "est_cml_psm_row": rows.est_cml_psm_row,
    "est_cml_rs_row": rows.est_cml_rs_row,
    "est_ml_psm_row": rows.est_ml_psm_row,
    "est_ml_rs_row": rows.est_ml_rs_row,
    "acc_cml_psm_row": rows.acc_cml_psm_row,
    "acc_cml_rs_row": rows.acc_cml_rs_row,
    "acc_ml_psm_row": rows.acc_ml_psm_row,
    "acc_ml_rs_row": rows.acc_ml_rs_row,
    "acc_base_row": acc_base_row,
    "acc_ref_row": acc_ref_row,
    "roc_row": rows.roc_row,
    "threshold_row": threshold_row,
    "prevalence_row": prevalence_row,
    "fit_diag_rows": list(rows.fit_diag_rows),
    "calibration_metric_rows": list(rows.calibration_metric_rows),
    "calibration_residual_rows": list(rows.calibration_residual_rows),
  }


@dataclass(frozen=True)
class ScenarioRuntime:
  run_label: str
  miscalibration_a: float
  miscalibration_b: float
  source_feature_specs: tuple[FeatureSpec, ...]
  theta0: np.ndarray
  n_jobs: int
  path_jobs: int
  intermediate_flush_every: int


def _positive_int_result(value: int, name: str) -> Result[int, str]:
  if value <= 0:
    return Err(f"{name} must be > 0")
  return Ok(value)


def _theta0_result(init_theta: tuple[float, ...], x_cols: list[str]) -> Result[np.ndarray, str]:
  expected_theta_len = 1 + (len(x_cols) + 1) + (len(x_cols) + 2)
  theta0 = np.asarray(init_theta, dtype=np.float64)
  if len(theta0) != expected_theta_len:
    return Err(f"init_theta length must be {expected_theta_len}; got {len(theta0)}")
  return Ok(theta0)


@result.block
def _scenario_runtime(
  *,
  cfg: RunConfig,
  x_cols: list[str],
  miscalibration_a: float | None,
  miscalibration_b: float | None,
  init_theta: tuple[float, ...],
  n_jobs: int,
  path_jobs: int,
  intermediate_flush_every: int,
  run_label: str | None,
) -> Result[ScenarioRuntime, str]:
  resolved_n_jobs = yield _positive_int_result(n_jobs, "n_jobs")
  resolved_path_jobs = yield _positive_int_result(path_jobs, "path_jobs")
  resolved_flush_every = yield _positive_int_result(
    intermediate_flush_every, "intermediate_flush_every"
  )
  theta0 = yield _theta0_result(init_theta, x_cols)

  return ScenarioRuntime(
    run_label=run_label or cfg.sim.scenario_run_label,
    miscalibration_a=(
      cfg.sim.source_miscalibration_a if miscalibration_a is None else miscalibration_a
    ),
    miscalibration_b=(
      cfg.sim.source_miscalibration_b if miscalibration_b is None else miscalibration_b
    ),
    source_feature_specs=cfg.sim.source_feature_specs or cfg.sim.feature_specs,
    theta0=theta0,
    n_jobs=resolved_n_jobs,
    path_jobs=resolved_path_jobs,
    intermediate_flush_every=resolved_flush_every,
  )


def run_scenario1_pipeline(
  cfg: RunConfig,
  *,
  miscalibration_a: float | None = None,
  miscalibration_b: float | None = None,
  calibration_tolerance: float = 0.1,
  init_theta: tuple[float, ...] = SCENARIO1_INIT_THETA,
  write_parquet: bool = False,
  n_jobs: int = 1,
  path_jobs: int = 1,
  intermediate_flush_every: int = 25,
  print_every: int = 100,
  run_label: str | None = None,
) -> Path:
  """Run the simulated orchestration loop and write package outputs."""

  output_root = Path(cfg.output_root)
  scenario_name = cfg.sim.scenario_name
  runtime_result = _scenario_runtime(
    cfg=cfg,
    x_cols=[spec.name for spec in cfg.sim.feature_specs],
    miscalibration_a=miscalibration_a,
    miscalibration_b=miscalibration_b,
    init_theta=init_theta,
    n_jobs=n_jobs,
    path_jobs=path_jobs,
    intermediate_flush_every=intermediate_flush_every,
    run_label=run_label,
  )
  if isinstance(runtime_result, Err):
    raise ValueError(runtime_result.error)
  runtime = runtime_result.value
  run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
  run_dir = output_root / f"python_{runtime.run_label}_{run_id}"
  runtime_dir = output_root / "python_runtime"
  intermediate_dir = run_dir / "intermediate"
  final_dir = run_dir / "final"
  runtime_dir.mkdir(parents=True, exist_ok=True)
  intermediate_dir.mkdir(parents=True, exist_ok=True)
  final_dir.mkdir(parents=True, exist_ok=True)

  x_cols = [spec.name for spec in cfg.sim.feature_specs]
  export_cols = ["caseY", *x_cols, "zOrigin", "zCat"]
  theta_cols = _theta_columns(x_cols)
  z_bins = np.asarray(cfg.sim.z_spec.bins, dtype=np.float64)
  gamma = np.asarray(cfg.sim.z_spec.gamma_init, dtype=np.float64)

  beta_target = np.asarray(cfg.sim.beta, dtype=np.float64)
  alpha_target = float(cfg.sim.alpha)
  alpha_source = float(runtime.miscalibration_a + runtime.miscalibration_b * alpha_target)
  beta_source = runtime.miscalibration_b * beta_target
  theta0 = runtime.theta0

  x_combos = enumerate_x_combinations(cfg.sim.feature_specs)
  x_combos_np = x_combos.to_numpy().astype(np.float64)
  x_prob_external = _x_probability_from_specs(cfg.sim.feature_specs, x_combos)

  est_cml_psm_rows: list[dict[str, float | int]] = []
  est_cml_rs_rows: list[dict[str, float | int]] = []
  est_ml_psm_rows: list[dict[str, float | int]] = []
  est_ml_rs_rows: list[dict[str, float | int]] = []
  acc_cml_psm_rows: list[dict[str, float | int]] = []
  acc_cml_rs_rows: list[dict[str, float | int]] = []
  acc_ml_psm_rows: list[dict[str, float | int]] = []
  acc_ml_rs_rows: list[dict[str, float | int]] = []
  acc_base_rows: list[dict[str, float | int]] = []
  acc_ref_rows: list[dict[str, float | int]] = []
  roc_rows: list[dict[str, float | int]] = []
  threshold_rows: list[dict[str, float | int]] = []
  prevalence_rows: list[dict[str, float | int]] = []
  fit_diag_rows: list[dict[str, object]] = []
  calibration_metric_rows: list[dict[str, object]] = []
  calibration_residual_rows: list[dict[str, object]] = []

  if runtime.n_jobs >= 1:
    seed_seq = np.random.SeedSequence(cfg.seed)
    iter_seeds = [
      int(ss.generate_state(1, dtype=np.uint32)[0])
      for ss in seed_seq.spawn(cfg.sim.nsim)
    ]
    effective_path_jobs = runtime.path_jobs if runtime.n_jobs == 1 else 1
    tasks: list[dict[str, object]] = []
    for tt in range(1, cfg.sim.nsim + 1):
      tasks.append(
        {
          "iteration": tt,
          "seed": iter_seeds[tt - 1],
          "target_feature_specs": cfg.sim.feature_specs,
          "source_feature_specs": runtime.source_feature_specs,
          "n_target": cfg.sim.n_target,
          "n_source": cfg.sim.n_source,
          "n_reference": cfg.sim.n_reference,
          "sample_size": cfg.sim.sample_size,
          "x_cols": x_cols,
          "export_cols": export_cols,
          "theta_cols": theta_cols,
          "x_combs_np": x_combos_np,
          "x_prob_external": x_prob_external,
          "z_bins": z_bins,
          "gamma": gamma,
          "theta0": theta0,
          "beta_target": beta_target,
          "beta_source": beta_source,
          "alpha_target": alpha_target,
          "alpha_source": alpha_source,
          "maxiter": cfg.opt.maxiter,
          "feasibility_tol": cfg.opt.tol,
          "calibration_tolerance": calibration_tolerance,
          "target_fpr": cfg.sim.target_fpr,
          "path_jobs": effective_path_jobs,
        }
      )

    intermediate_buffers: dict[str, list[pl.DataFrame]] = {
      "xind": [],
      "pe": [],
      "sample_psm": [],
      "sample_rs": [],
    }
    consumed = 0

    def consume_output(out: dict[str, object]) -> None:
      nonlocal consumed
      intermediate_buffers["xind"].append(
        rows_to_frame(
          out["xind_rows"],
          columns=["iter", "x_combination", "risk_interval"],
        )
      )
      intermediate_buffers["pe"].append(
        rows_to_frame(
          out["pe_rows"],
          columns=["iter", "risk_interval", "p_external"],
        )
      )
      intermediate_buffers["sample_psm"].append(
        rows_to_frame(
          out["sample_psm_rows"],
          columns=["iter", *export_cols],
        )
      )
      intermediate_buffers["sample_rs"].append(
        rows_to_frame(
          out["sample_rs_rows"],
          columns=["iter", *export_cols],
        )
      )

      est_cml_psm_rows.append(out["est_cml_psm_row"])
      est_cml_rs_rows.append(out["est_cml_rs_row"])
      est_ml_psm_rows.append(out["est_ml_psm_row"])
      est_ml_rs_rows.append(out["est_ml_rs_row"])
      acc_cml_psm_rows.append(out["acc_cml_psm_row"])
      acc_cml_rs_rows.append(out["acc_cml_rs_row"])
      acc_ml_psm_rows.append(out["acc_ml_psm_row"])
      acc_ml_rs_rows.append(out["acc_ml_rs_row"])
      acc_base_rows.append(out["acc_base_row"])
      acc_ref_rows.append(out["acc_ref_row"])
      roc_rows.append(out["roc_row"])
      threshold_rows.append(out["threshold_row"])
      prevalence_rows.append(out["prevalence_row"])
      fit_diag_rows.extend(out["fit_diag_rows"])
      calibration_metric_rows.extend(out["calibration_metric_rows"])
      calibration_residual_rows.extend(out["calibration_residual_rows"])
      consumed += 1

      if consumed % runtime.intermediate_flush_every == 0:
        _flush_intermediate_buffers(
          buffers=intermediate_buffers, intermediate_dir=intermediate_dir
        )

      tt = int(out["iter"])
      if print_every > 0 and tt % print_every == 0:
        print(tt)
        print(datetime.now())
        print(prevalence_rows[-1]["prevalence_target"])

    if runtime.n_jobs == 1:
      for out in map(_run_parallel_sim_iteration, tasks):
        consume_output(out)
    else:
      with ProcessPoolExecutor(
        max_workers=runtime.n_jobs, mp_context=mp.get_context("spawn")
      ) as pool:
        for out in pool.map(_run_parallel_sim_iteration, tasks):
          consume_output(out)
    _flush_intermediate_buffers(buffers=intermediate_buffers, intermediate_dir=intermediate_dir)

    est_cml_psm_df = rows_to_frame(est_cml_psm_rows, columns=["iter", *theta_cols])
    est_cml_rs_df = rows_to_frame(est_cml_rs_rows, columns=["iter", *theta_cols])
    est_ml_psm_df = rows_to_frame(est_ml_psm_rows, columns=["iter", *theta_cols])
    est_ml_rs_df = rows_to_frame(est_ml_rs_rows, columns=["iter", *theta_cols])

    acc_cols = ["iter", "TPR", "PPV", "TNR"]
    acc_cml_psm_df = rows_to_frame(acc_cml_psm_rows, columns=acc_cols)
    acc_cml_rs_df = rows_to_frame(acc_cml_rs_rows, columns=acc_cols)
    acc_ml_psm_df = rows_to_frame(acc_ml_psm_rows, columns=acc_cols)
    acc_ml_rs_df = rows_to_frame(acc_ml_rs_rows, columns=acc_cols)
    acc_base_df = rows_to_frame(acc_base_rows, columns=acc_cols)
    acc_ref_df = rows_to_frame(acc_ref_rows, columns=acc_cols)

    roc_df = rows_to_frame(roc_rows, columns=ROC_METRIC_COLUMNS)
    threshold_df = rows_to_frame(
      threshold_rows, columns=["iter", "threshold_ref_fpr"]
    )
    prevalence_df = rows_to_frame(
      prevalence_rows, columns=["iter", "prevalence_target"]
    )
    fit_diag_df = rows_to_frame(fit_diag_rows, columns=FIT_DIAGNOSTIC_COLUMNS)
    calibration_metrics_df = rows_to_frame(
      calibration_metric_rows, columns=CALIBRATION_METRIC_COLUMNS
    )
    calibration_residuals_df = rows_to_frame(
      calibration_residual_rows, columns=CALIBRATION_RESIDUAL_COLUMNS
    )

    run_metadata_df = rows_to_frame(
      [
        {
          "run_id": run_id,
          "scenario": scenario_name,
          "nsim": cfg.sim.nsim,
          "Ntarget": cfg.sim.n_target,
          "Nsource": cfg.sim.n_source,
          "Nrs": cfg.sim.n_reference,
          "samplesize": cfg.sim.sample_size,
          "y_prev": cfg.sim.target_prevalence,
          "betaz": float(cfg.sim.beta[-1]),
          "fpr_target": cfg.sim.target_fpr,
          "alpha": cfg.sim.alpha,
          "parameterOR": ";".join(str(v) for v in cfg.sim.beta),
          "gamma": ";".join(str(v) for v in cfg.sim.z_spec.gamma_init),
          "seed": cfg.seed,
          "miscalibration_a": runtime.miscalibration_a,
          "miscalibration_b": runtime.miscalibration_b,
          "source_miscalibration_a": runtime.miscalibration_a,
          "source_miscalibration_b": runtime.miscalibration_b,
          "target_feature_specs": _serialize_feature_specs(cfg.sim.feature_specs),
          "source_feature_specs": _serialize_feature_specs(runtime.source_feature_specs),
          "calibration_tolerance": calibration_tolerance,
          "n_jobs": runtime.n_jobs,
          "path_jobs": runtime.path_jobs,
          "intermediate_flush_every": runtime.intermediate_flush_every,
          "schema_version": OUTPUT_SCHEMA_VERSION,
        }
      ],
      columns=[
        "run_id",
        "scenario",
        "nsim",
        "Ntarget",
        "Nsource",
        "Nrs",
        "samplesize",
        "y_prev",
        "betaz",
        "fpr_target",
        "alpha",
        "parameterOR",
        "gamma",
        "seed",
        "miscalibration_a",
        "miscalibration_b",
        "source_miscalibration_a",
        "source_miscalibration_b",
        "target_feature_specs",
        "source_feature_specs",
        "calibration_tolerance",
        "n_jobs",
        "path_jobs",
        "intermediate_flush_every",
        "schema_version",
      ],
    )

    parquet_written = False
    parquet_written |= write_tabular_export(
      est_cml_psm_df, final_dir / "est_cml_psm.csv", write_parquet
    )
    parquet_written |= write_tabular_export(
      est_cml_rs_df, final_dir / "est_cml_rs.csv", write_parquet
    )
    parquet_written |= write_tabular_export(
      est_ml_psm_df, final_dir / "est_ml_psm.csv", write_parquet
    )
    parquet_written |= write_tabular_export(
      est_ml_rs_df, final_dir / "est_ml_rs.csv", write_parquet
    )
    parquet_written |= write_tabular_export(
      acc_cml_psm_df, final_dir / "accuracy_cml_psm.csv", write_parquet
    )
    parquet_written |= write_tabular_export(
      acc_cml_rs_df, final_dir / "accuracy_cml_rs.csv", write_parquet
    )
    parquet_written |= write_tabular_export(
      acc_ml_psm_df, final_dir / "accuracy_ml_psm.csv", write_parquet
    )
    parquet_written |= write_tabular_export(
      acc_ml_rs_df, final_dir / "accuracy_ml_rs.csv", write_parquet
    )
    parquet_written |= write_tabular_export(
      acc_base_df, final_dir / "accuracy_base.csv", write_parquet
    )
    parquet_written |= write_tabular_export(
      acc_ref_df, final_dir / "accuracy_ref.csv", write_parquet
    )
    parquet_written |= write_tabular_export(
      roc_df, final_dir / "roc_metrics.csv", write_parquet
    )
    parquet_written |= write_tabular_export(
      threshold_df, final_dir / "threshold_ref_fpr.csv", write_parquet
    )
    parquet_written |= write_tabular_export(
      prevalence_df, final_dir / "target_prevalence_by_iter.csv", write_parquet
    )
    write_tabular_export(
      fit_diag_df, final_dir / "fit_diagnostics.csv", write_parquet
    )
    parquet_written |= write_tabular_export(
      calibration_metrics_df, final_dir / "calibration_metrics.csv", write_parquet
    )
    parquet_written |= write_tabular_export(
      calibration_residuals_df,
      final_dir / "calibration_residuals.csv",
      write_parquet,
    )
    run_metadata_df.write_csv(final_dir / "run_metadata.csv")
    if write_parquet and not parquet_written:
      print(
        "Package 'arrow' not installed; parquet exports were skipped (CSV exports are available)."
      )

    print(f"Intermediate and final outputs written under: {run_dir}")
    return run_dir

  raise AssertionError("unreachable: n_jobs must be > 0")

def run_simulated_pipeline(
  cfg: RunConfig, options: Scenario1PipelineOptions | None = None
) -> Path:
  """Run the simulated pipeline using typed runtime options.

  Parameters
  ----------
  cfg:
    Simulated-run configuration, including the data-generating process.
  options:
    Optional runtime/export settings layered on top of `cfg`.

  Returns
  -------
  Path
    Run directory containing the `intermediate/` and `final/` outputs.
  """

  resolved = options or Scenario1PipelineOptions()
  return run_scenario1_pipeline(
    cfg,
    miscalibration_a=resolved.miscalibration_a,
    miscalibration_b=resolved.miscalibration_b,
    calibration_tolerance=resolved.calibration_tolerance,
    init_theta=resolved.init_theta,
    write_parquet=resolved.write_parquet,
    n_jobs=resolved.n_jobs,
    path_jobs=resolved.path_jobs,
    intermediate_flush_every=resolved.intermediate_flush_every,
    print_every=resolved.print_every,
    run_label=resolved.run_label,
  )


def run_user_data_pipeline(config: UserDataRunConfig) -> Path:
  """Run the Scenario-1 statistical pipeline on user-provided in-memory DataFrames.

  Parameters
  ----------
  config:
    Typed user-data pipeline configuration. Input frames are preprocessed into
    canonical columns `caseY`, `x_cols...`, `zOrigin`, `zCat` before fitting.

  Returns
  -------
  Path
    Run directory containing the `intermediate/` and `final/` outputs.
  """

  if config.n_jobs > 1:
    raise ValueError(
      "user-data mode currently supports only n_jobs=1. "
      "Use simulated mode for multi-process iteration parallelism."
    )

  context = _prepare_user_data_context(config)

  output_root_path = Path(config.output_root)
  run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
  run_dir = output_root_path / f"python_userdata_{config.run_label}_{run_id}"
  intermediate_dir = run_dir / "intermediate"
  final_dir = run_dir / "final"
  intermediate_dir.mkdir(parents=True, exist_ok=True)
  final_dir.mkdir(parents=True, exist_ok=True)

  rng = np.random.default_rng(config.seed)

  est_cml_psm_rows: list[dict[str, float | int]] = []
  est_cml_rs_rows: list[dict[str, float | int]] = []
  est_ml_psm_rows: list[dict[str, float | int]] = []
  est_ml_rs_rows: list[dict[str, float | int]] = []
  acc_cml_psm_rows: list[dict[str, float | int]] = []
  acc_cml_rs_rows: list[dict[str, float | int]] = []
  acc_ml_psm_rows: list[dict[str, float | int]] = []
  acc_ml_rs_rows: list[dict[str, float | int]] = []
  acc_base_rows: list[dict[str, float | int]] = []
  acc_ref_rows: list[dict[str, float | int]] = []
  roc_rows: list[dict[str, float | int]] = []
  threshold_rows: list[dict[str, float | int]] = []
  prevalence_rows: list[dict[str, float | int]] = []
  fit_diag_rows: list[dict[str, object]] = []
  calibration_metric_rows: list[dict[str, object]] = []
  calibration_residual_rows: list[dict[str, object]] = []
  intermediate_buffers: dict[str, list[pl.DataFrame]] = {
    "xind": [],
    "pe": [],
    "sample_psm": [],
    "sample_rs": [],
  }

  for tt in range(1, config.nsim + 1):
    iter_data = _prepare_user_data_iteration_inputs(
      context,
      rng=rng,
      sample_size=config.sample_size,
      iteration=tt,
    )
    prevalence_rows.append(iter_data["prevalence_row"])
    intermediate_buffers["xind"].append(iter_data["xind_df"])
    intermediate_buffers["pe"].append(iter_data["pe_df"])
    intermediate_buffers["sample_psm"].append(iter_data["sample_psm_df"])
    intermediate_buffers["sample_rs"].append(iter_data["sample_rs_df"])
    threshold_rows.append(iter_data["threshold_row"])
    acc_base_rows.append(iter_data["acc_base_row"])
    acc_ref_rows.append(iter_data["acc_ref_row"])

    sample_psm_export = iter_data["sample_psm_export"]
    sample_rs_export = iter_data["sample_rs_export"]
    roc_x = np.asarray(iter_data["roc_x"], dtype=np.float64)
    roc_zc = np.asarray(iter_data["roc_zc"], dtype=np.int64)
    roc_y = np.asarray(iter_data["roc_y"], dtype=np.int64)
    base_score = np.asarray(iter_data["base_score"], dtype=np.float64)
    ref_score = np.asarray(iter_data["ref_score"], dtype=np.float64)

    mle_psm, cmle_psm, mle_rs, cmle_rs = _fit_two_paths(
      sample_psm_export=sample_psm_export,
      sample_rs_export=sample_rs_export,
      theta0=context.theta0,
      x_cols=context.x_cols,
      x_combs_np=context.x_combos_np,
      x_prob_external=context.x_prob_external,
      x_interval_index=context.artifacts.x_interval_index,
      p_external=context.artifacts.p_external,
      tempcateg=context.z_bins_arr,
      maxiter=config.maxiter,
      feasibility_tol=config.feasibility_tol,
      calibration_tolerance=config.calibration_tolerance,
      path_jobs=config.path_jobs,
    )

    rows = _iteration_export_rows(
      fits=PathFits(
        mle_psm=mle_psm, cmle_psm=cmle_psm, mle_rs=mle_rs, cmle_rs=cmle_rs
      ),
      theta_cols=context.theta_cols,
      eval_inputs=EvaluationInputs(
        iteration=tt,
        x_cols=context.x_cols,
        roc_x=roc_x,
        roc_zc=roc_zc,
        roc_y=roc_y,
        threshold=context.fixed_threshold,
        base_score=base_score,
        ref_score=ref_score,
      ),
      residual_inputs=ResidualExportInputs(
        x_combs=context.x_combos_np,
        x_prob_external=context.x_prob_external,
        x_interval_index=context.artifacts.x_interval_index,
        p_external=context.artifacts.p_external,
        tempcateg=context.z_bins_arr,
      ),
    )
    est_cml_psm_rows.append(rows.est_cml_psm_row)
    est_cml_rs_rows.append(rows.est_cml_rs_row)
    est_ml_psm_rows.append(rows.est_ml_psm_row)
    est_ml_rs_rows.append(rows.est_ml_rs_row)
    acc_cml_psm_rows.append(rows.acc_cml_psm_row)
    acc_cml_rs_rows.append(rows.acc_cml_rs_row)
    acc_ml_psm_rows.append(rows.acc_ml_psm_row)
    acc_ml_rs_rows.append(rows.acc_ml_rs_row)
    roc_rows.append(rows.roc_row)
    fit_diag_rows.extend(rows.fit_diag_rows)
    calibration_metric_rows.extend(rows.calibration_metric_rows)
    calibration_residual_rows.extend(rows.calibration_residual_rows)

    if config.print_every > 0 and tt % config.print_every == 0:
      print(
        f"[user-data] iter={tt}/{config.nsim} "
        f"target_prev={prevalence_rows[-1]['prevalence_target']:.5f}"
      )
    if tt % config.intermediate_flush_every == 0:
      _flush_intermediate_buffers(
        buffers=intermediate_buffers, intermediate_dir=intermediate_dir
      )

  _flush_intermediate_buffers(buffers=intermediate_buffers, intermediate_dir=intermediate_dir)

  est_cml_psm_df = rows_to_frame(est_cml_psm_rows, columns=["iter", *context.theta_cols])
  est_cml_rs_df = rows_to_frame(est_cml_rs_rows, columns=["iter", *context.theta_cols])
  est_ml_psm_df = rows_to_frame(est_ml_psm_rows, columns=["iter", *context.theta_cols])
  est_ml_rs_df = rows_to_frame(est_ml_rs_rows, columns=["iter", *context.theta_cols])

  acc_cols = ["iter", "TPR", "PPV", "TNR"]
  acc_cml_psm_df = rows_to_frame(acc_cml_psm_rows, columns=acc_cols)
  acc_cml_rs_df = rows_to_frame(acc_cml_rs_rows, columns=acc_cols)
  acc_ml_psm_df = rows_to_frame(acc_ml_psm_rows, columns=acc_cols)
  acc_ml_rs_df = rows_to_frame(acc_ml_rs_rows, columns=acc_cols)
  acc_base_df = rows_to_frame(acc_base_rows, columns=acc_cols)
  acc_ref_df = rows_to_frame(acc_ref_rows, columns=acc_cols)

  roc_df = rows_to_frame(roc_rows, columns=ROC_METRIC_COLUMNS)
  threshold_df = rows_to_frame(threshold_rows, columns=["iter", "threshold_ref_fpr"])
  prevalence_df = rows_to_frame(prevalence_rows, columns=["iter", "prevalence_target"])
  fit_diag_df = rows_to_frame(fit_diag_rows, columns=FIT_DIAGNOSTIC_COLUMNS)
  calibration_metrics_df = rows_to_frame(
    calibration_metric_rows, columns=CALIBRATION_METRIC_COLUMNS
  )
  calibration_residuals_df = rows_to_frame(
    calibration_residual_rows, columns=CALIBRATION_RESIDUAL_COLUMNS
  )

  run_metadata_df = rows_to_frame(
    [
      {
        "run_id": run_id,
        "mode": "user_data",
        "seed": config.seed,
        "nsim": config.nsim,
        "Ntarget": len(context.target_clean),
        "Nsource": len(context.source_clean),
        "Nrs": len(context.reference_clean),
        "samplesize": config.sample_size,
        "fpr_target": config.target_fpr,
        "x_cols": ";".join(context.x_cols),
        "z_bins": ";".join(str(v) for v in context.z_bins_arr),
        "calibration_tolerance": config.calibration_tolerance,
        "n_jobs": config.n_jobs,
        "path_jobs": config.path_jobs,
        "intermediate_flush_every": config.intermediate_flush_every,
        "schema_version": OUTPUT_SCHEMA_VERSION,
      }
    ],
    columns=[
      "run_id",
      "mode",
      "seed",
      "nsim",
      "Ntarget",
      "Nsource",
      "Nrs",
      "samplesize",
      "fpr_target",
      "x_cols",
      "z_bins",
      "calibration_tolerance",
      "n_jobs",
      "path_jobs",
      "intermediate_flush_every",
      "schema_version",
    ],
  )

  parquet_written = False
  parquet_written |= write_tabular_export(
    est_cml_psm_df, final_dir / "est_cml_psm.csv", config.write_parquet
  )
  parquet_written |= write_tabular_export(
    est_cml_rs_df, final_dir / "est_cml_rs.csv", config.write_parquet
  )
  parquet_written |= write_tabular_export(
    est_ml_psm_df, final_dir / "est_ml_psm.csv", config.write_parquet
  )
  parquet_written |= write_tabular_export(
    est_ml_rs_df, final_dir / "est_ml_rs.csv", config.write_parquet
  )
  parquet_written |= write_tabular_export(
    acc_cml_psm_df, final_dir / "accuracy_cml_psm.csv", config.write_parquet
  )
  parquet_written |= write_tabular_export(
    acc_cml_rs_df, final_dir / "accuracy_cml_rs.csv", config.write_parquet
  )
  parquet_written |= write_tabular_export(
    acc_ml_psm_df, final_dir / "accuracy_ml_psm.csv", config.write_parquet
  )
  parquet_written |= write_tabular_export(
    acc_ml_rs_df, final_dir / "accuracy_ml_rs.csv", config.write_parquet
  )
  parquet_written |= write_tabular_export(
    acc_base_df, final_dir / "accuracy_base.csv", config.write_parquet
  )
  parquet_written |= write_tabular_export(
    acc_ref_df, final_dir / "accuracy_ref.csv", config.write_parquet
  )
  parquet_written |= write_tabular_export(
    roc_df, final_dir / "roc_metrics.csv", config.write_parquet
  )
  parquet_written |= write_tabular_export(
    threshold_df, final_dir / "threshold_ref_fpr.csv", config.write_parquet
  )
  parquet_written |= write_tabular_export(
    prevalence_df, final_dir / "target_prevalence_by_iter.csv", config.write_parquet
  )
  write_tabular_export(
    fit_diag_df, final_dir / "fit_diagnostics.csv", config.write_parquet
  )
  parquet_written |= write_tabular_export(
    calibration_metrics_df, final_dir / "calibration_metrics.csv", config.write_parquet
  )
  parquet_written |= write_tabular_export(
    calibration_residuals_df,
    final_dir / "calibration_residuals.csv",
    config.write_parquet,
  )
  run_metadata_df.write_csv(final_dir / "run_metadata.csv")

  if config.write_parquet and not parquet_written:
    print(
      "[user-data] parquet export requested but no engine was available. "
      "CSV exports were written."
    )

  return run_dir


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
    description="Run Risk Bridge simulated or user-data constrained MLE workflows."
  )
  parser.add_argument(
    "--mode",
    type=str,
    choices=("simulated", "user-data", "external-calibration"),
    default="simulated",
    help=(
      "simulated: generate populations; user-data: use target/source/reference data; "
      "external-calibration: bootstrap one source cohort against fixed summaries."
    ),
  )
  parser.add_argument(
    "--scenario",
    type=int,
    choices=(1, 2, 3),
    default=1,
    help="Simulated DGP preset. Ignored in user-data mode.",
  )
  parser.add_argument("--seed", type=int, default=None)
  parser.add_argument("--nsim", type=int, default=None)
  parser.add_argument("--n-target", type=int, default=None)
  parser.add_argument("--n-source", type=int, default=None)
  parser.add_argument("--n-reference", type=int, default=None)
  parser.add_argument("--sample-size", type=int, default=None)
  parser.add_argument("--target-prevalence", type=float, default=None)
  parser.add_argument("--target-fpr", type=float, default=None)
  parser.add_argument("--alpha", type=float, default=None)
  parser.add_argument("--beta-z", type=float, default=None)
  parser.add_argument("--miscalibration-a", type=float, default=None)
  parser.add_argument("--miscalibration-b", type=float, default=None)
  parser.add_argument("--calibration-tolerance", type=float, default=0.1)
  parser.add_argument("--feasibility-tol", type=float, default=1e-6)
  parser.add_argument("--maxiter", type=int, default=200)
  parser.add_argument("--n-jobs", type=int, default=1)
  parser.add_argument("--path-jobs", type=int, default=1)
  parser.add_argument("--intermediate-flush-every", type=int, default=25)
  parser.add_argument("--output-root", type=str, default="data")
  parser.add_argument("--run-label", type=str, default=None)
  parser.add_argument("--print-every", type=int, default=100)
  parser.add_argument("--write-parquet", action="store_true")
  parser.add_argument("--external-calibration-json", type=str, default="")
  parser.add_argument("--z-origin-scale", type=float, default=1.0)
  parser.add_argument("--checkpoint-every", type=int, default=25)
  parser.add_argument("--resume-run-dir", type=str, default="")
  parser.add_argument("--write-sample-artifacts", action="store_true")
  parser.add_argument("--target-csv", type=str, default="")
  parser.add_argument("--source-csv", type=str, default="")
  parser.add_argument("--reference-csv", type=str, default="")
  parser.add_argument("--target-data", type=str, default="")
  parser.add_argument("--source-data", type=str, default="")
  parser.add_argument("--reference-data", type=str, default="")
  parser.add_argument("--target-object", type=str, default="")
  parser.add_argument("--source-object", type=str, default="")
  parser.add_argument("--reference-object", type=str, default="")
  parser.add_argument(
    "--y-col",
    type=str,
    default="caseY",
    help="Outcome column name in user-provided CSVs.",
  )
  parser.add_argument(
    "--z-origin-col",
    type=str,
    default="zOrigin",
    help="Continuous Z column name in user-provided CSVs.",
  )
  parser.add_argument(
    "--z-cat-col",
    type=str,
    default="zCat",
    help="Categorical Z column name in user-provided CSVs.",
  )
  parser.add_argument(
    "--allow-z-origin-from-zcat",
    action="store_true",
    help="Allow deriving zOrigin from zCat interval midpoints if zOrigin is absent.",
  )
  parser.add_argument(
    "--x-cols",
    type=str,
    default="",
    help="Comma-separated feature columns (required in user-data mode).",
  )
  parser.add_argument(
    "--z-bins",
    type=str,
    default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9",
    help="Comma-separated z bin boundaries used by constraints.",
  )
  return parser.parse_args()


def _resolve_cli_data_path(
  *, generic_path: str, legacy_csv_path: str, dataset_name: str
) -> str:
  if generic_path and legacy_csv_path:
    raise ValueError(
      f"Specify only one of --{dataset_name}-data or --{dataset_name}-csv."
    )
  return generic_path or legacy_csv_path


def main() -> None:
  args = _parse_args()

  if args.mode == "simulated":
    builder_map = {
      1: build_scenario1_run_config,
      2: build_scenario2_run_config,
      3: build_scenario3_run_config,
    }
    config_kwargs: dict[str, int | float | str] = {
      "output_root": args.output_root,
      "maxiter": args.maxiter,
    }
    override_map = {
      "seed": args.seed,
      "nsim": args.nsim,
      "n_target": args.n_target,
      "n_source": args.n_source,
      "n_reference": args.n_reference,
      "sample_size": args.sample_size,
      "target_prevalence": args.target_prevalence,
      "target_fpr": args.target_fpr,
      "alpha": args.alpha,
      "beta_z": args.beta_z,
    }
    for key, value in override_map.items():
      if value is not None:
        config_kwargs[key] = value

    cfg = builder_map[args.scenario](**config_kwargs)
    run_simulated_pipeline(
      cfg,
      Scenario1PipelineOptions(
        miscalibration_a=args.miscalibration_a,
        miscalibration_b=args.miscalibration_b,
        calibration_tolerance=args.calibration_tolerance,
        write_parquet=args.write_parquet,
        n_jobs=args.n_jobs,
        path_jobs=args.path_jobs,
        intermediate_flush_every=args.intermediate_flush_every,
        print_every=args.print_every,
        run_label=args.run_label,
      ),
    )
    return

  if args.mode == "external-calibration":
    from risk_bridge.external import (
      load_external_calibration_spec,
      run_external_calibration_bootstrap,
    )

    source_path = _resolve_cli_data_path(
      generic_path=args.source_data,
      legacy_csv_path=args.source_csv,
      dataset_name="source",
    )
    if not source_path or not args.external_calibration_json:
      raise ValueError(
        "external-calibration mode requires --source-data and "
        "--external-calibration-json."
      )
    if not args.x_cols.strip():
      raise ValueError("external-calibration mode requires --x-cols.")
    source_df = load_tabular_input(
      source_path,
      object_name=args.source_object or None,
      dataset_name="source",
    )
    x_cols = tuple(c.strip() for c in args.x_cols.split(",") if c.strip())
    run_external_calibration_bootstrap(
      ExternalCalibrationBootstrapConfig(
        source_df=source_df,
        schema=UserDataSchema(
          x_cols=x_cols,
          z_bins=_parse_float_tuple(args.z_bins),
          y_col=args.y_col,
          z_origin_col=args.z_origin_col,
          z_cat_col=args.z_cat_col,
          allow_z_origin_from_z_cat=args.allow_z_origin_from_zcat,
        ),
        calibration=load_external_calibration_spec(args.external_calibration_json),
        n_target=len(source_df) if args.n_target is None else args.n_target,
        seed=631 if args.seed is None else args.seed,
        nsim=1000 if args.nsim is None else args.nsim,
        sample_size=2000 if args.sample_size is None else args.sample_size,
        z_origin_scale=args.z_origin_scale,
        maxiter=args.maxiter,
        n_jobs=args.n_jobs,
        checkpoint_every=args.checkpoint_every,
        feasibility_tol=args.feasibility_tol,
        calibration_tolerance=args.calibration_tolerance,
        output_root=args.output_root,
        print_every=args.print_every,
        run_label="external_calibration" if args.run_label is None else args.run_label,
        write_sample_artifacts=args.write_sample_artifacts,
        resume_run_dir=args.resume_run_dir or None,
      )
    )
    return

  target_path = _resolve_cli_data_path(
    generic_path=args.target_data,
    legacy_csv_path=args.target_csv,
    dataset_name="target",
  )
  source_path = _resolve_cli_data_path(
    generic_path=args.source_data,
    legacy_csv_path=args.source_csv,
    dataset_name="source",
  )
  reference_path = _resolve_cli_data_path(
    generic_path=args.reference_data,
    legacy_csv_path=args.reference_csv,
    dataset_name="reference",
  )
  if not target_path or not source_path or not reference_path:
    raise ValueError(
      "user-data mode requires target, source, and reference data paths."
    )
  if not args.x_cols.strip():
    raise ValueError("user-data mode requires --x-cols.")

  x_cols = [c.strip() for c in args.x_cols.split(",") if c.strip()]
  z_bins = _parse_float_tuple(args.z_bins)

  target_df = load_tabular_input(
    target_path,
    object_name=args.target_object or None,
    dataset_name="target",
  )
  source_df = load_tabular_input(
    source_path,
    object_name=args.source_object or None,
    dataset_name="source",
  )
  reference_df = load_tabular_input(
    reference_path,
    object_name=args.reference_object or None,
    dataset_name="reference",
  )

  run_user_data_pipeline(
    UserDataRunConfig(
      target_df=target_df,
      source_df=source_df,
      reference_df=reference_df,
      schema=UserDataSchema(
        x_cols=tuple(x_cols),
        z_bins=z_bins,
        y_col=args.y_col,
        z_origin_col=args.z_origin_col,
        z_cat_col=args.z_cat_col,
        allow_z_origin_from_z_cat=args.allow_z_origin_from_zcat,
      ),
      seed=631 if args.seed is None else args.seed,
      nsim=1 if args.nsim is None else args.nsim,
      sample_size=1000 if args.sample_size is None else args.sample_size,
      target_fpr=0.1 if args.target_fpr is None else args.target_fpr,
      maxiter=args.maxiter,
      n_jobs=args.n_jobs,
      path_jobs=args.path_jobs,
      intermediate_flush_every=args.intermediate_flush_every,
      feasibility_tol=args.feasibility_tol,
      calibration_tolerance=args.calibration_tolerance,
      output_root=args.output_root,
      write_parquet=args.write_parquet,
      print_every=args.print_every,
      run_label="user_data" if args.run_label is None else args.run_label,
    )
  )


if __name__ == "__main__":
  main()
