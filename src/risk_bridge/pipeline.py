from __future__ import annotations

from dataclasses import asdict

import numpy as np
import polars as pl
from scipy.optimize import minimize

from risk_bridge.calibration import build_calibration_artifacts, enumerate_x_combinations
from risk_bridge.config import RunConfig
from risk_bridge.constraints import (
  calibration_inequalities,
  calibration_inequalities_jacobian,
)
from risk_bridge.likelihood import (
  joint_negative_log_likelihood,
  joint_negative_log_likelihood_grad,
  logistic_risk,
)
from risk_bridge.metrics import find_threshold_for_fpr, roc_auc_binary, threshold_metrics
from risk_bridge.optimize import solve_cmle_with_ladder
from risk_bridge.simulate import generate_population
from risk_bridge.tabular import gather_rows, rows_to_frame
from risk_bridge.types import (
  EvaluationSummary,
  FitResult,
  IterationMetrics,
  IterationResult,
  Population,
)


def _theta_from_sim_config(cfg: RunConfig) -> np.ndarray:
  return np.array(
    [cfg.sim.alpha, *cfg.sim.beta, *cfg.sim.z_spec.gamma_init], dtype=np.float64
  )


def _x_probability_external(
  reference_df: pl.DataFrame, x_combos: pl.DataFrame
) -> np.ndarray:
  x_cols = list(x_combos.columns)
  combo_keys = [tuple(int(v) for v in row) for row in x_combos.iter_rows()]

  counts_df = reference_df.group_by(x_cols).agg(pl.len().alias("__count"))
  counts = {
    tuple(int(row[col]) for col in x_cols): int(row["__count"])
    for row in counts_df.iter_rows(named=True)
  }
  total = max(float(len(reference_df)), 1.0)

  probs: np.ndarray = np.zeros(len(combo_keys), dtype=np.float64)
  for i, key in enumerate(combo_keys):
    probs[i] = float(counts.get(key, 0.0)) / total

  s = probs.sum()
  if s <= 0:
    return np.full(len(probs), 1.0 / len(probs), dtype=np.float64)
  return probs / s


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


def _evaluate(
  theta: np.ndarray, x: np.ndarray, z_cat: np.ndarray, y: np.ndarray, target_fpr: float
) -> EvaluationSummary:
  px = x.shape[1]
  alpha = float(theta[0])
  beta = np.asarray(theta[1 : 1 + px + 1], dtype=np.float64)

  score = logistic_risk(alpha, beta, x, z_cat)
  thr = find_threshold_for_fpr(y, score, target_fpr=target_fpr)
  return EvaluationSummary.from_metrics(
    auc=roc_auc_binary(y, score),
    threshold=thr,
    metrics=threshold_metrics(y, score, threshold=thr),
  )


def _generate_populations(
  *,
  cfg: RunConfig,
  rng: np.random.Generator,
  gamma: np.ndarray,
  z_bins: np.ndarray,
  beta: np.ndarray,
) -> tuple[Population, Population, Population]:
  return (
    generate_population(
      rng=rng,
      n=cfg.sim.n_target,
      feature_specs=cfg.sim.feature_specs,
      gamma=gamma,
      z_bins=z_bins,
      alpha=cfg.sim.alpha,
      beta=beta,
    ),
    generate_population(
      rng=rng,
      n=cfg.sim.n_source,
      feature_specs=cfg.sim.feature_specs,
      gamma=gamma,
      z_bins=z_bins,
      alpha=cfg.sim.alpha,
      beta=beta,
    ),
    generate_population(
      rng=rng,
      n=cfg.sim.n_reference,
      feature_specs=cfg.sim.feature_specs,
      gamma=gamma,
      z_bins=z_bins,
      alpha=cfg.sim.alpha,
      beta=beta,
    ),
  )


def run_single_iteration_result(
  cfg: RunConfig, iteration_seed_offset: int = 0
) -> IterationResult:
  """Run one small c-MLE iteration and return a typed result bundle.

  Example
  -------
  >>> from risk_bridge.config import FeatureSpec, OptimizationConfig, SimulationConfig, ZModelSpec
  >>> run_cfg = RunConfig(
  ...     seed=1,
  ...     output_root="data/out",
  ...     sim=SimulationConfig(
  ...         nsim=1, n_target=20, n_source=20, n_reference=20, sample_size=10,
  ...         target_prevalence=0.1, target_fpr=0.1, alpha=-2.0,
  ...         beta=(0.2, 0.1, 0.0, 0.0, 0.2),
  ...         feature_specs=(
  ...             FeatureSpec("X1", "categorical_cut", {"breaks": (0.0, 0.5, 1.0)}),
  ...             FeatureSpec("X2", "categorical_cut", {"breaks": (0.0, 0.5, 1.0)}),
  ...             FeatureSpec("X3", "capped_poisson", {"lambda": 0.5, "cap": 2}),
  ...             FeatureSpec("X4", "capped_poisson", {"lambda": 0.5, "cap": 2}),
  ...         ),
  ...         z_spec=ZModelSpec("trunc_lognormal", (-1.0, 0.1, 0.1, 0.0, 0.0, 0.5), tuple(np.arange(0.1, 1.0, 0.1))),
  ...     ),
  ...     opt=OptimizationConfig("BFGS", "trust-constr", 1e-6, 50),
  ... )
  >>> out = run_single_iteration_result(run_cfg)
  >>> out.metrics.cmle.auc >= 0.0
  True
  """

  rng = np.random.default_rng(cfg.seed + iteration_seed_offset)
  z_bins = np.asarray(cfg.sim.z_spec.bins, dtype=np.float64)
  gamma = np.asarray(cfg.sim.z_spec.gamma_init, dtype=np.float64)
  beta = np.asarray(cfg.sim.beta, dtype=np.float64)

  target, source, reference = _generate_populations(
    cfg=cfg, rng=rng, gamma=gamma, z_bins=z_bins, beta=beta
  )

  artifacts = build_calibration_artifacts(reference, cfg.sim.feature_specs)
  x_combos = enumerate_x_combinations(cfg.sim.feature_specs)
  x_prob_external = _x_probability_external(reference.X, x_combos)

  idx = np.asarray(
    rng.choice(cfg.sim.n_source, size=cfg.sim.sample_size, replace=False),
    dtype=np.int64,
  )
  sample_x = gather_rows(source.X, idx).to_numpy().astype(np.float64)
  sample_y = np.asarray(source.y[idx], dtype=np.int64)
  sample_z = np.asarray(source.z_cont[idx], dtype=np.float64)
  sample_zc = np.asarray(source.z_cat[idx], dtype=np.int64)

  theta_true = _theta_from_sim_config(cfg)
  theta0 = theta_true + rng.normal(0.0, 0.1, size=len(theta_true))

  mle = _fit_unconstrained(
    theta0=theta0,
    x=sample_x,
    y=sample_y,
    z=sample_z,
    z_cat=sample_zc,
    maxiter=cfg.opt.maxiter,
  )

  def objective_fn(t: np.ndarray) -> float:
    return joint_negative_log_likelihood(t, sample_x, sample_y, sample_z, sample_zc)

  def objective_jac_fn(t: np.ndarray) -> np.ndarray:
    return joint_negative_log_likelihood_grad(
      t, sample_x, sample_y, sample_z, sample_zc
    )

  def constraints_fn(t: np.ndarray) -> np.ndarray:
    return calibration_inequalities(
      theta=t,
      x_combs=x_combos.to_numpy().astype(np.float64),
      x_prob_external=x_prob_external,
      calibration_index=artifacts.x_interval_index,
      y_external=artifacts.p_external,
      tolerance=0.1,
      tempcateg=z_bins,
    )

  def constraints_jac_fn(t: np.ndarray) -> np.ndarray:
    return calibration_inequalities_jacobian(
      theta=t,
      x_combs=x_combos.to_numpy().astype(np.float64),
      x_prob_external=x_prob_external,
      calibration_index=artifacts.x_interval_index,
      y_external=artifacts.p_external,
      tempcateg=z_bins,
    )

  cmle, history = solve_cmle_with_ladder(
    theta0=mle.theta,
    objective_fn=objective_fn,
    objective_jac_fn=objective_jac_fn,
    constraints_fn=constraints_fn,
    constraints_jac_fn=constraints_jac_fn,
    maxiter=cfg.opt.maxiter,
    feasibility_tol=cfg.opt.tol,
  )

  target_x = target.X.to_numpy().astype(np.float64)
  return IterationResult(
    mle=mle,
    cmle=cmle,
    metrics=IterationMetrics(
      mle=_evaluate(
        mle.theta, target_x, target.z_cat, target.y, target_fpr=cfg.sim.target_fpr
      ),
      cmle=_evaluate(
        cmle.theta, target_x, target.z_cat, target.y, target_fpr=cfg.sim.target_fpr
      ),
    ),
    solver_history=tuple(history),
  )


def run_single_iteration(
  cfg: RunConfig, iteration_seed_offset: int = 0
) -> dict[str, object]:
  """Return dictionary-based iteration output for compact exploratory runs."""

  result = run_single_iteration_result(cfg, iteration_seed_offset=iteration_seed_offset)
  return {
    "mle": result.mle,
    "cmle": result.cmle,
    "metrics": {
      "mle": result.metrics.mle.as_dict(),
      "cmle": result.metrics.cmle.as_dict(),
    },
    "solver_history": [asdict(item) for item in result.solver_history],
  }


def run_pipeline(cfg: RunConfig) -> pl.DataFrame:
  rows: list[dict[str, float | int | bool]] = []
  for i in range(cfg.sim.nsim):
    out = run_single_iteration_result(cfg, iteration_seed_offset=i)
    row = {
      "iter": i,
      "mle_success": bool(out.mle.success),
      "cmle_success": bool(out.cmle.success),
      "auc_mle": float(out.metrics.mle.auc),
      "auc_cmle": float(out.metrics.cmle.auc),
    }
    rows.append(row)
  return rows_to_frame(
    rows,
    columns=["iter", "mle_success", "cmle_success", "auc_mle", "auc_cmle"],
  )
