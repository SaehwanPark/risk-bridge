"""Synthetic fixed-summary external-calibration recovery and calibration gates."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
import json
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from risk_bridge.calibration import (
  build_calibration_artifacts,
  enumerate_x_combinations,
)
from risk_bridge.config import (
  DEFAULT_Z_BINS,
  ExternalCalibrationBootstrapConfig,
  ExternalCalibrationSpec,
  FeatureSpec,
  SCENARIO1_INIT_THETA,
  UserDataSchema,
)
from risk_bridge.constraints import calibration_residuals
from risk_bridge.external import run_external_calibration_bootstrap
from risk_bridge.reproducibility import capture_run_environment
from risk_bridge.simulate import generate_population
from risk_bridge.types import Population

CASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = CASE_DIR.parents[1]
DEFAULT_SEED = 475
ALPHA_TARGET = -2.38
BETA_TARGET = np.array([-0.5, 0.4, 0.3, 0.65, 0.1], dtype=np.float64)
GAMMA = np.array([-2.0, 0.1, -0.1, 0.1, 0.1, 0.6], dtype=np.float64)
MISCALIBRATION_A = 0.5
MISCALIBRATION_B = 1.0
Z_BINS = np.asarray(DEFAULT_Z_BINS, dtype=np.float64)
X_COLS = ("X1", "X2", "X3", "X4")
OUTCOME_LEN = 1 + len(X_COLS) + 1  # alpha, beta_X*, beta_Zcat
OUTCOME_RMSE_THRESHOLD = 1.25
CALIBRATION_ABS_TOL = 0.05
DEGRADED_LOGIT_SHIFT = 6.0


@dataclass(frozen=True)
class ProfileSpec:
  name: str
  n_target: int
  n_source: int
  n_reference: int
  sample_size: int
  nsim: int
  maxiter: int


PROFILES: dict[str, ProfileSpec] = {
  "smoke": ProfileSpec(
    name="smoke",
    n_target=1200,
    n_source=900,
    n_reference=1200,
    sample_size=180,
    nsim=1,
    maxiter=100,
  ),
  "full": ProfileSpec(
    name="full",
    n_target=5000,
    n_source=2000,
    n_reference=5000,
    sample_size=500,
    nsim=50,
    maxiter=200,
  ),
}


@dataclass(frozen=True)
class CheckResult:
  name: str
  passed: bool
  detail: str
  metrics: dict[str, float]


def _mc_summary(values: list[float]) -> dict[str, float | int | None]:
  """Summarize successful replicate values with a sample-based MCSE."""

  if not values:
    return {"n_successful": 0, "mean": None, "sd": None, "mcse": None}
  array = np.asarray(values, dtype=np.float64)
  mean = float(np.mean(array))
  if len(array) < 2:
    return {"n_successful": 1, "mean": mean, "sd": None, "mcse": None}
  sd = float(np.std(array, ddof=1))
  return {
    "n_successful": len(array),
    "mean": mean,
    "sd": sd,
    "mcse": float(sd / np.sqrt(len(array))),
  }


def _condition_external_prevalence(
  p_external: np.ndarray, condition: str
) -> np.ndarray:
  """Apply the predeclared matched or fixed-summary degradation condition."""

  if condition == "matched":
    return p_external.copy()
  if condition == "degraded":
    clipped = np.clip(p_external, 1e-6, 1.0 - 1e-6)
    logits = np.log(clipped / (1.0 - clipped)) + DEGRADED_LOGIT_SHIFT
    return (1.0 / (1.0 + np.exp(-logits))).astype(np.float64)
  raise ValueError("condition must be 'matched' or 'degraded'")


def _target_feature_specs() -> tuple[FeatureSpec, ...]:
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


def _source_feature_specs() -> tuple[FeatureSpec, ...]:
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


def _support_values(spec: FeatureSpec) -> list[int]:
  if spec.kind == "categorical_cut":
    breaks = tuple(float(v) for v in spec.params["breaks"])
    return list(range(len(breaks) - 1))
  if spec.kind == "capped_poisson":
    return list(range(int(spec.params.get("cap", 2)) + 1))
  if spec.kind == "custom":
    return [int(v) for v in spec.params["values"]]
  raise ValueError(f"Unsupported feature kind: {spec.kind}")


def _population_to_source_frame(pop: Population) -> pl.DataFrame:
  return pop.X.with_columns(
    pl.Series("caseY", pop.y),
    pl.Series("zOrigin", pop.z_cont),
    pl.Series("zCat", pop.z_cat),
  ).select(["caseY", *pop.X.columns, "zOrigin", "zCat"])


def _empirical_custom_specs(
  target: Population, generation_specs: tuple[FeatureSpec, ...]
) -> tuple[FeatureSpec, ...]:
  """Freeze target-X marginals on the generation support as custom FeatureSpecs."""

  specs: list[FeatureSpec] = []
  for spec in generation_specs:
    values = _support_values(spec)
    observed = target.X.get_column(spec.name).to_numpy().astype(np.int64)
    counts = np.array([(observed == v).sum() for v in values], dtype=np.float64)
    # Tiny floor so every generation level keeps positive mass after renormalization.
    counts = counts + 1e-6
    probs = counts / counts.sum()
    specs.append(
      FeatureSpec(
        name=spec.name,
        kind="custom",
        params={
          "values": tuple(values),
          "probs": tuple(float(p) for p in probs),
        },
      )
    )
  return tuple(specs)


def _truth_outcome() -> np.ndarray:
  return np.concatenate(
    [[ALPHA_TARGET], BETA_TARGET.astype(np.float64)]
  ).astype(np.float64)


def _x_probability_from_custom_specs(
  feature_specs: tuple[FeatureSpec, ...], x_combs: pl.DataFrame
) -> np.ndarray:
  """Cartesian product probabilities for frozen custom FeatureSpecs."""

  probs = np.ones(len(x_combs), dtype=np.float64)
  for spec in feature_specs:
    values = tuple(int(v) for v in spec.params["values"])
    levels = tuple(float(p) for p in spec.params["probs"])
    pmap = dict(zip(values, levels, strict=True))
    observed = x_combs.get_column(spec.name).to_numpy().astype(np.int64)
    probs *= np.asarray([pmap.get(int(v), 0.0) for v in observed], dtype=np.float64)
  total = float(probs.sum())
  if total <= 0.0:
    return np.full(len(probs), 1.0 / max(len(probs), 1), dtype=np.float64)
  return probs / total


def _json_safe(value: object) -> object:
  if isinstance(value, float) and not np.isfinite(value):
    return None
  if isinstance(value, dict):
    return {str(key): _json_safe(item) for key, item in value.items()}
  if isinstance(value, list):
    return [_json_safe(item) for item in value]
  return value


def _theta_from_estimate_row(row: dict[str, Any], x_cols: tuple[str, ...]) -> np.ndarray:
  names = [
    "alpha",
    *[f"beta_{name}" for name in x_cols],
    "beta_Zcat",
    "gamma_0",
    *[f"gamma_{name}" for name in x_cols],
    "gamma_sigma",
  ]
  return np.asarray([float(row[name]) for name in names], dtype=np.float64)


def _mean_abs_residual(
  theta: np.ndarray,
  *,
  x_combs: np.ndarray,
  x_prob: np.ndarray,
  interval_index: np.ndarray,
  p_external: np.ndarray,
  z_bins: np.ndarray,
) -> float:
  residuals = calibration_residuals(
    theta=theta,
    x_combs=x_combs,
    x_prob_external=x_prob,
    calibration_index=interval_index,
    y_external=p_external,
    tempcateg=z_bins,
  )
  return float(np.mean(np.abs(residuals)))


def resolve_profile(name: str) -> ProfileSpec:
  if name not in PROFILES:
    known = ", ".join(sorted(PROFILES))
    raise ValueError(f"unknown profile {name!r}; expected one of: {known}")
  return PROFILES[name]


def run_suite(
  *,
  profile: str = "smoke",
  seed: int = DEFAULT_SEED,
  output_root: Path | None = None,
  run_label: str = "external_calibration_validation",
  condition: str = "matched",
) -> dict[str, Any]:
  """Generate truth, freeze fixed summaries, fit external paths, and score gates."""

  sizes = resolve_profile(profile)
  if condition not in {"matched", "degraded"}:
    raise ValueError("condition must be 'matched' or 'degraded'")
  rng = np.random.default_rng(seed)
  target_specs = _target_feature_specs()
  source_specs = _source_feature_specs()
  alpha_source = MISCALIBRATION_A + MISCALIBRATION_B * ALPHA_TARGET
  beta_source = MISCALIBRATION_B * BETA_TARGET

  target = generate_population(
    rng=rng,
    n=sizes.n_target,
    feature_specs=target_specs,
    gamma=GAMMA,
    z_bins=Z_BINS,
    alpha=ALPHA_TARGET,
    beta=BETA_TARGET,
  )
  source = generate_population(
    rng=rng,
    n=sizes.n_source,
    feature_specs=source_specs,
    gamma=GAMMA,
    z_bins=Z_BINS,
    alpha=alpha_source,
    beta=beta_source,
  )
  reference = generate_population(
    rng=rng,
    n=sizes.n_reference,
    feature_specs=target_specs,
    gamma=GAMMA,
    z_bins=Z_BINS,
    alpha=ALPHA_TARGET,
    beta=BETA_TARGET,
  )

  custom_specs = _empirical_custom_specs(target, target_specs)
  artifacts = build_calibration_artifacts(reference, target_specs)
  base_p_external = np.asarray(artifacts.p_external, dtype=np.float64)
  conditioned_p_external = _condition_external_prevalence(
    base_p_external, condition
  )
  calibration = ExternalCalibrationSpec(
    target_feature_specs=custom_specs,
    x_interval_index=tuple(int(v) for v in artifacts.x_interval_index.tolist()),
    p_external=tuple(float(v) for v in conditioned_p_external.tolist()),
  )
  fixed_summary = {
    "feature_distributions": [
      {
        "name": spec.name,
        "values": list(spec.params["values"]),
        "probabilities": list(spec.params["probs"]),
      }
      for spec in custom_specs
    ],
    "x_interval_index": list(calibration.x_interval_index),
    "p_external": list(calibration.p_external),
    "condition": condition,
    "degraded_logit_shift": DEGRADED_LOGIT_SHIFT if condition == "degraded" else 0.0,
  }

  work_root = output_root if output_root is not None else Path("data")
  case_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
  case_root = (
    work_root / "external_calibration_validation" / f"{run_label}_{case_stamp}"
  )
  runs_root = case_root / "runs"
  source_df = _population_to_source_frame(source)

  run_dir = run_external_calibration_bootstrap(
    ExternalCalibrationBootstrapConfig(
      source_df=source_df,
      schema=UserDataSchema(
        x_cols=X_COLS,
        z_bins=tuple(float(v) for v in Z_BINS),
        y_col="caseY",
        z_origin_col="zOrigin",
        z_cat_col="zCat",
      ),
      calibration=calibration,
      n_target=sizes.n_target,
      seed=seed,
      nsim=sizes.nsim,
      sample_size=sizes.sample_size,
      maxiter=sizes.maxiter,
      n_jobs=1,
      checkpoint_every=max(1, sizes.nsim),
      init_theta=SCENARIO1_INIT_THETA,
      output_root=str(runs_root),
      print_every=0,
      run_label=run_label,
      write_sample_artifacts=False,
    )
  )

  est_frames = []
  for path_name, fit_name, filename in (
    ("PSM", "CMLE", "est_cml_psm.csv"),
    ("RS", "CMLE", "est_cml_rs.csv"),
    ("PSM", "MLE", "est_ml_psm.csv"),
    ("RS", "MLE", "est_ml_rs.csv"),
  ):
    frame = pl.read_csv(run_dir / "final" / filename).with_columns(
      pl.lit(path_name).alias("path"),
      pl.lit(fit_name).alias("fit_type"),
    )
    est_frames.append(frame)
  estimates = pl.concat(est_frames, how="vertical_relaxed")
  diagnostics = pl.read_csv(run_dir / "final" / "fit_diagnostics.csv")
  if diagnostics["success"].dtype != pl.Boolean:
    diagnostics = diagnostics.with_columns(
      pl.col("success")
      .cast(pl.Utf8)
      .str.to_lowercase()
      .is_in(["true", "1"])
      .alias("success")
    )

  x_combs = enumerate_x_combinations(custom_specs)
  x_combs_np = x_combs.to_numpy().astype(np.float64)
  x_prob = _x_probability_from_custom_specs(custom_specs, x_combs)
  interval_index = np.asarray(calibration.x_interval_index, dtype=np.int64)
  p_external = np.asarray(calibration.p_external, dtype=np.float64)
  truth = _truth_outcome()

  recovery_rows: list[dict[str, object]] = []
  calibration_rows: list[dict[str, object]] = []
  diag_rows: list[dict[str, object]] = []
  path_metrics: dict[str, dict[str, float | bool]] = {}

  for path_name in ("PSM", "RS"):
    for fit_type in ("CMLE", "MLE"):
      diag = diagnostics.filter(
        (pl.col("path") == path_name) & (pl.col("fit_type") == fit_type)
      )
      if diag.height == 0:
        success_rate = 0.0
        mean_violation = float("inf")
      else:
        success_rate = float(diag.get_column("success").cast(pl.Float64).mean() or 0.0)
        mean_violation = float(
          diag.get_column("max_violation").mean()
          if diag.get_column("max_violation").mean() is not None
          else float("inf")
        )
      diag_rows.append(
        {
          "path": path_name,
          "fit_type": fit_type,
          "success_rate": success_rate,
          "mean_max_violation": mean_violation,
          "n_iters": diag.height,
          "success_count": int(diag.get_column("success").sum())
          if diag.height
          else 0,
          "success_rate_mcse": (
            float(np.sqrt(success_rate * (1.0 - success_rate) / diag.height))
            if diag.height
            else None
          ),
        }
      )
      est = estimates.filter(
        (pl.col("path") == path_name) & (pl.col("fit_type") == fit_type)
      )
      # Average coefficients across successful bootstrap iterations for both fits.
      ok_iters = set(diag.filter(pl.col("success")).get_column("iter").to_list())
      if ok_iters:
        est = est.filter(pl.col("iter").is_in(list(ok_iters)))
      elif success_rate <= 0.0:
        est = est.head(0)
      if est.is_empty():
        outcome_rmse = float("inf")
        mean_abs_resid = float("inf")
        outcome_rmse_samples: list[float] = []
        residual_samples: list[float] = []
      else:
        thetas = [
          _theta_from_estimate_row(row, X_COLS) for row in est.iter_rows(named=True)
        ]
        outcome_rmse_samples = [
          float(np.sqrt(np.mean(np.square(theta[:OUTCOME_LEN] - truth))))
          for theta in thetas
        ]
        residual_samples = [
          _mean_abs_residual(
            theta,
            x_combs=x_combs_np,
            x_prob=x_prob,
            interval_index=interval_index,
            p_external=p_external,
            z_bins=Z_BINS,
          )
          for theta in thetas
        ]
        mean_theta = np.mean(np.stack(thetas, axis=0), axis=0)
        outcome_rmse = float(
          np.sqrt(np.mean(np.square(mean_theta[:OUTCOME_LEN] - truth)))
        )
        mean_abs_resid = _mean_abs_residual(
          mean_theta,
          x_combs=x_combs_np,
          x_prob=x_prob,
          interval_index=interval_index,
          p_external=p_external,
          z_bins=Z_BINS,
        )
      recovery_rows.append(
        {
          "path": path_name,
          "fit_type": fit_type,
          "outcome_rmse": outcome_rmse,
          "success_rate": success_rate,
          "outcome_rmse_mean": _mc_summary(outcome_rmse_samples)["mean"],
          "outcome_rmse_sd": _mc_summary(outcome_rmse_samples)["sd"],
          "outcome_rmse_mcse": _mc_summary(outcome_rmse_samples)["mcse"],
          "n_successful": _mc_summary(outcome_rmse_samples)["n_successful"],
        }
      )
      calibration_rows.append(
        {
          "path": path_name,
          "fit_type": fit_type,
          "mean_abs_residual": mean_abs_resid,
          "success_rate": success_rate,
          "mean_abs_residual_mean": _mc_summary(residual_samples)["mean"],
          "mean_abs_residual_sd": _mc_summary(residual_samples)["sd"],
          "mean_abs_residual_mcse": _mc_summary(residual_samples)["mcse"],
          "n_successful": _mc_summary(residual_samples)["n_successful"],
        }
      )
      path_metrics[f"{path_name}:{fit_type}"] = {
        "outcome_rmse": outcome_rmse,
        "mean_abs_residual": mean_abs_resid,
        "success_rate": success_rate,
      }

  successful_cmle_paths = [
    path
    for path in ("PSM", "RS")
    if float(path_metrics[f"{path}:CMLE"]["success_rate"]) > 0.0
  ]
  feasibility_pass = len(successful_cmle_paths) > 0

  calibration_pass = False
  recovery_pass = False
  best_path = successful_cmle_paths[0] if successful_cmle_paths else None
  if best_path is not None:
    # Prefer the successful path with the smaller cMLE residual.
    best_path = min(
      successful_cmle_paths,
      key=lambda p: float(path_metrics[f"{p}:CMLE"]["mean_abs_residual"]),
    )
    cmle_resid = float(path_metrics[f"{best_path}:CMLE"]["mean_abs_residual"])
    # Absolute adequacy gate; MLE residuals remain in metrics for comparison.
    calibration_pass = bool(np.isfinite(cmle_resid) and cmle_resid <= CALIBRATION_ABS_TOL)

    cmle_rmse = float(path_metrics[f"{best_path}:CMLE"]["outcome_rmse"])
    recovery_pass = bool(
      np.isfinite(cmle_rmse) and cmle_rmse < OUTCOME_RMSE_THRESHOLD
    )

  checks = [
    CheckResult(
      name="feasibility",
      passed=feasibility_pass,
      detail="At least one cMLE path (PSM/RS) reports a successful constrained solve",
      metrics={
        "n_successful_cmle_paths": float(len(successful_cmle_paths)),
        "psm_cmle_success_rate": float(path_metrics["PSM:CMLE"]["success_rate"]),
        "rs_cmle_success_rate": float(path_metrics["RS:CMLE"]["success_rate"]),
      },
    ),
    CheckResult(
      name="calibration",
      passed=calibration_pass,
      detail=(
        "Mean |calibration residual| for a successful cMLE path is "
        f"<= {CALIBRATION_ABS_TOL}"
      ),
      metrics={
        "best_path_psm": 1.0 if best_path == "PSM" else 0.0,
        "calibration_abs_tol": CALIBRATION_ABS_TOL,
        "cmle_mean_abs_residual": float(
          path_metrics[f"{best_path}:CMLE"]["mean_abs_residual"]
        )
        if best_path
        else float("inf"),
        "mle_mean_abs_residual": float(
          path_metrics[f"{best_path}:MLE"]["mean_abs_residual"]
        )
        if best_path
        else float("inf"),
      },
    ),
    CheckResult(
      name="recovery",
      passed=recovery_pass,
      detail=(
        "Outcome-coefficient RMSE for a successful cMLE path is below "
        f"{OUTCOME_RMSE_THRESHOLD}"
      ),
      metrics={
        "outcome_rmse_threshold": OUTCOME_RMSE_THRESHOLD,
        "cmle_outcome_rmse": float(path_metrics[f"{best_path}:CMLE"]["outcome_rmse"])
        if best_path
        else float("inf"),
        "mle_outcome_rmse": float(path_metrics[f"{best_path}:MLE"]["outcome_rmse"])
        if best_path
        else float("inf"),
      },
    ),
  ]
  passed = all(check.passed for check in checks)
  environment = capture_run_environment(
    cwd=REPO_ROOT,
    extra={
      "case": "external_calibration_validation",
      "profile": sizes.name,
      "seed": seed,
      "run_label": run_label,
      "n_target": sizes.n_target,
      "n_source": sizes.n_source,
      "n_reference": sizes.n_reference,
      "sample_size": sizes.sample_size,
      "nsim": sizes.nsim,
      "maxiter": sizes.maxiter,
      "condition": condition,
      "degraded_logit_shift": DEGRADED_LOGIT_SHIFT
      if condition == "degraded"
      else 0.0,
    },
  )
  return {
    "passed": passed,
    "checks": [asdict(check) for check in checks],
    "recovery_rows": recovery_rows,
    "calibration_rows": calibration_rows,
    "diagnostics_rows": diag_rows,
    "fixed_summary": fixed_summary,
    "environment": environment,
    "case_root": str(case_root),
    "run_dir": str(run_dir),
    "truth_outcome": truth.tolist(),
    "best_path": best_path,
    "profile": sizes.name,
    "seed": seed,
    "condition": condition,
  }


def write_suite_artifacts(suite: dict[str, Any], output_dir: Path | None = None) -> Path:
  """Persist summary, environment, fixed summary, and gate tables."""

  out = Path(suite["case_root"]) if output_dir is None else output_dir
  out.mkdir(parents=True, exist_ok=True)
  (out / "summary.json").write_text(
    json.dumps(
      _json_safe(
        {
          "passed": suite["passed"],
          "profile": suite["profile"],
          "seed": suite["seed"],
          "best_path": suite["best_path"],
          "condition": suite["condition"],
          "truth_outcome": suite["truth_outcome"],
          "run_dir": suite["run_dir"],
          "checks": suite["checks"],
          "demotion_rule": (
            "If passed is false, do not promote external-calibration claims in "
            "manuscript or SPEC completion notes."
          ),
        }
      ),
      indent=2,
      sort_keys=True,
      allow_nan=False,
    )
    + "\n",
    encoding="utf-8",
  )
  (out / "environment.json").write_text(
    json.dumps(suite["environment"], indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
  )
  (out / "fixed_summary.json").write_text(
    json.dumps(suite["fixed_summary"], indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
  )
  pl.DataFrame(suite["recovery_rows"]).write_csv(out / "recovery.csv")
  pl.DataFrame(suite["calibration_rows"]).write_csv(out / "calibration_gates.csv")
  pl.DataFrame(suite["diagnostics_rows"]).write_csv(
    out / "fit_diagnostics_summary.csv"
  )
  return out
