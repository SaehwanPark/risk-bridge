from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import polars as pl

from risk_bridge.calibration import enumerate_x_combinations
from risk_bridge.config import (
  ExternalCalibrationBootstrapConfig,
  ExternalCalibrationSpec,
  FeatureSpec,
)
from risk_bridge.preprocess import preprocess_user_dataset
from risk_bridge.runs import _fit_path, _theta_columns, _x_probability_from_specs
from risk_bridge.sampling import propensity_scores_x_only, psm_sample_source_from_scores
from risk_bridge.simulate import categorize_z, sample_x
from risk_bridge.tabular import ensure_polars_frame, gather_rows, rows_to_frame


@dataclass(frozen=True)
class ExternalPreparedContext:
  config: ExternalCalibrationBootstrapConfig
  source: pl.DataFrame
  x_cols: tuple[str, ...]
  export_cols: tuple[str, ...]
  x_combinations: pl.DataFrame
  x_prob_external: np.ndarray
  theta0: np.ndarray


@dataclass(frozen=True)
class ExternalIterationInputs:
  iteration: int
  bootstrap_indices: np.ndarray
  pseudo_target: pl.DataFrame
  sample_psm: pl.DataFrame
  sample_rs: pl.DataFrame


_WORKER_CONTEXT: ExternalPreparedContext | None = None


def load_external_calibration_spec(path: str | Path) -> ExternalCalibrationSpec:
  """Load fixed external calibration inputs from a JSON document."""

  payload = json.loads(Path(path).read_text(encoding="utf-8"))
  distributions = payload.get("feature_distributions")
  if not isinstance(distributions, list):
    raise ValueError("external calibration JSON requires feature_distributions")
  specs = tuple(
    FeatureSpec(
      name=str(item["name"]),
      kind="custom",
      params={
        "values": tuple(int(value) for value in item["values"]),
        "probs": tuple(float(value) for value in item["probabilities"]),
      },
    )
    for item in distributions
  )
  return ExternalCalibrationSpec(
    target_feature_specs=specs,
    x_interval_index=tuple(int(value) for value in payload["x_interval_index"]),
    p_external=tuple(float(value) for value in payload["p_external"]),
  )


def _prepare_external_context(
  config: ExternalCalibrationBootstrapConfig,
) -> ExternalPreparedContext:
  schema = config.schema
  raw = ensure_polars_frame(config.source_df, clone=False)
  if schema.z_origin_col in raw.columns and config.z_origin_scale != 1.0:
    raw = raw.with_columns(
      (pl.col(schema.z_origin_col).cast(pl.Float64) * config.z_origin_scale)
      .round(12)
      .alias(schema.z_origin_col)
    )
  source = preprocess_user_dataset(
    raw,
    x_cols=list(schema.x_cols),
    z_bins=schema.z_bins,
    dataset_name="source_df",
    y_col=schema.y_col,
    z_origin_col=schema.z_origin_col,
    z_cat_col=schema.z_cat_col,
    allow_z_origin_from_z_cat=schema.allow_z_origin_from_z_cat,
  )
  if config.sample_size > len(source):
    raise ValueError("sample_size must not exceed the source cohort size")
  if schema.z_origin_col in raw.columns and schema.z_cat_col in raw.columns:
    derived = categorize_z(
      source.get_column("zOrigin").to_numpy(), np.asarray(schema.z_bins, dtype=np.float64)
    )
    observed = source.get_column("zCat").to_numpy()
    if not np.array_equal(derived, observed):
      mismatch = int(np.sum(derived != observed))
      raise ValueError(
        f"source_df.{schema.z_cat_col} does not match scaled zOrigin and z_bins "
        f"for {mismatch} row(s)."
      )

  specs = config.calibration.target_feature_specs
  spec_names = tuple(spec.name for spec in specs)
  if spec_names != schema.x_cols:
    raise ValueError("Target feature specification names/order must match schema.x_cols")
  x_combinations = enumerate_x_combinations(specs)
  x_prob_external = _x_probability_from_specs(specs, x_combinations)
  interval_index = np.asarray(config.calibration.x_interval_index, dtype=np.int64)
  interval_mass = np.bincount(
    interval_index - 1,
    weights=x_prob_external,
    minlength=len(config.calibration.p_external),
  )
  if np.any(interval_mass <= 0.0):
    raise ValueError("Every external calibration interval must have positive X mass")
  expected_theta = 2 * len(schema.x_cols) + 4
  theta0 = np.asarray(config.init_theta, dtype=np.float64)
  if len(theta0) != expected_theta:
    raise ValueError(
      f"init_theta must have length {expected_theta} for {len(schema.x_cols)} X columns"
    )
  return ExternalPreparedContext(
    config=config,
    source=source,
    x_cols=schema.x_cols,
    export_cols=("caseY", *schema.x_cols, "zOrigin", "zCat"),
    x_combinations=x_combinations,
    x_prob_external=x_prob_external,
    theta0=theta0,
  )


def _prepare_external_iteration(
  context: ExternalPreparedContext, *, seed: int, iteration: int
) -> ExternalIterationInputs:
  rng = np.random.default_rng(seed)
  source_size = len(context.source)
  bootstrap_indices = np.asarray(
    rng.choice(source_size, size=source_size, replace=True), dtype=np.int64
  )
  bootstrap = gather_rows(context.source, bootstrap_indices)
  pseudo_target = sample_x(
    rng,
    context.config.n_target,
    context.config.calibration.target_feature_specs,
  )
  target_scores, source_scores = propensity_scores_x_only(
    pseudo_target, bootstrap, list(context.x_cols)
  )
  sample_psm = psm_sample_source_from_scores(
    rng,
    bootstrap,
    target_scores,
    source_scores,
    context.config.sample_size,
  ).select(context.export_cols)
  rs_indices = np.asarray(
    rng.choice(source_size, size=context.config.sample_size, replace=False),
    dtype=np.int64,
  )
  sample_rs = gather_rows(bootstrap, rs_indices).select(context.export_cols)
  return ExternalIterationInputs(
    iteration=iteration,
    bootstrap_indices=bootstrap_indices,
    pseudo_target=pseudo_target,
    sample_psm=sample_psm,
    sample_rs=sample_rs,
  )


def _fit_external_iteration(
  context: ExternalPreparedContext, prepared: ExternalIterationInputs
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
  config = context.config
  x_combinations = context.x_combinations.to_numpy().astype(np.float64)
  interval_index = np.asarray(config.calibration.x_interval_index, dtype=np.int64)
  p_external = np.asarray(config.calibration.p_external, dtype=np.float64)
  estimates: list[dict[str, object]] = []
  diagnostics: list[dict[str, object]] = []
  theta_cols = _theta_columns(list(context.x_cols))
  for path, sample in (("PSM", prepared.sample_psm), ("RS", prepared.sample_rs)):
    mle, cmle = _fit_path(
      sample,
      context.theta0,
      list(context.x_cols),
      x_combinations,
      context.x_prob_external,
      interval_index,
      p_external,
      np.asarray(config.schema.z_bins, dtype=np.float64),
      config.maxiter,
      config.feasibility_tol,
      config.calibration_tolerance,
    )
    for fit_type, fit in (("MLE", mle), ("CMLE", cmle)):
      estimates.append(
        {
          "iter": prepared.iteration,
          "path": path,
          "fit_type": fit_type,
          **{name: float(value) for name, value in zip(theta_cols, fit.theta, strict=True)},
        }
      )
      diagnostics.append(
        {
          "iter": prepared.iteration,
          "path": path,
          "fit_type": fit_type,
          "success": bool(fit.success),
          "status": fit.status,
          "objective": float(fit.objective),
          "max_violation": float(fit.diagnostics.get("max_violation", 0.0)),
          "solver": str(fit.diagnostics.get("solver", "")),
        }
      )
  return estimates, diagnostics


def _initialize_external_worker(context: ExternalPreparedContext) -> None:
  global _WORKER_CONTEXT
  _WORKER_CONTEXT = context


def _run_external_task(
  task: tuple[int, int], context: ExternalPreparedContext | None = None
) -> tuple[
  int,
  list[dict[str, object]],
  list[dict[str, object]],
  pl.DataFrame | None,
  pl.DataFrame | None,
]:
  active = context if context is not None else _WORKER_CONTEXT
  if active is None:
    raise RuntimeError("External bootstrap worker context was not initialized")
  iteration, seed = task
  try:
    prepared = _prepare_external_iteration(active, seed=seed, iteration=iteration)
    estimates, diagnostics = _fit_external_iteration(active, prepared)
    return (
      iteration,
      estimates,
      diagnostics,
      prepared.sample_psm if active.config.write_sample_artifacts else None,
      prepared.sample_rs if active.config.write_sample_artifacts else None,
    )
  except Exception as exc:
    diagnostics = [
      {
        "iter": iteration,
        "path": path,
        "fit_type": fit_type,
        "success": False,
        "status": f"error: {type(exc).__name__}: {exc}",
        "objective": float("nan"),
        "max_violation": float("nan"),
        "solver": "",
      }
      for path in ("PSM", "RS")
      for fit_type in ("MLE", "CMLE")
    ]
    return iteration, [], diagnostics, None, None


def _write_external_calibration(context: ExternalPreparedContext, path: Path) -> None:
  interval_index = context.config.calibration.x_interval_index
  frame = context.x_combinations.with_columns(
    pl.Series("x_probability", context.x_prob_external),
    pl.Series("risk_interval", interval_index),
  )
  frame.write_csv(path)
  pl.DataFrame(
    {
      "risk_interval": np.arange(1, len(context.config.calibration.p_external) + 1),
      "p_external": context.config.calibration.p_external,
    }
  ).write_csv(path.with_name("p_external.csv"))


def _configuration_fingerprint(context: ExternalPreparedContext) -> str:
  config = context.config
  payload = {
    "schema": {
      "x_cols": config.schema.x_cols,
      "z_bins": config.schema.z_bins,
      "y_col": config.schema.y_col,
      "z_origin_col": config.schema.z_origin_col,
      "z_cat_col": config.schema.z_cat_col,
    },
    "calibration": {
      "feature_specs": [
        {"name": spec.name, "kind": spec.kind, "params": spec.params}
        for spec in config.calibration.target_feature_specs
      ],
      "x_interval_index": config.calibration.x_interval_index,
      "p_external": config.calibration.p_external,
    },
    "run": {
      "seed": config.seed,
      "nsim": config.nsim,
      "sample_size": config.sample_size,
      "n_target": config.n_target,
      "z_origin_scale": config.z_origin_scale,
      "maxiter": config.maxiter,
      "feasibility_tol": config.feasibility_tol,
      "calibration_tolerance": config.calibration_tolerance,
      "init_theta": config.init_theta,
      "write_sample_artifacts": config.write_sample_artifacts,
    },
  }
  digest = hashlib.sha256(
    json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
  )
  digest.update(context.source.hash_rows(seed=0).to_numpy().tobytes())
  return digest.hexdigest()


def _write_checkpoint(
  estimates: list[dict[str, object]],
  diagnostics: list[dict[str, object]],
  intermediate_dir: Path,
) -> None:
  checkpoint_path = intermediate_dir / "checkpoint.json"
  temporary_path = checkpoint_path.with_suffix(".json.tmp")
  payload = {"estimates": estimates, "diagnostics": diagnostics}
  with temporary_path.open("w", encoding="utf-8") as handle:
    json.dump(payload, handle, separators=(",", ":"))
    handle.flush()
    os.fsync(handle.fileno())
  temporary_path.replace(checkpoint_path)


def _completed_iterations(diagnostics: list[dict[str, object]]) -> set[int]:
  expected = {
    ("PSM", "MLE"),
    ("PSM", "CMLE"),
    ("RS", "MLE"),
    ("RS", "CMLE"),
  }
  keys: dict[int, set[tuple[str, str]]] = {}
  retryable_error: set[int] = set()
  for row in diagnostics:
    iteration = int(row["iter"])
    keys.setdefault(iteration, set()).add((str(row["path"]), str(row["fit_type"])))
    if str(row.get("status", "")).startswith("error:"):
      retryable_error.add(iteration)
  return {
    iteration
    for iteration, observed in keys.items()
    if observed == expected and iteration not in retryable_error
  }


def _replace_iteration_rows(
  rows: list[dict[str, object]],
  iteration: int,
  replacements: list[dict[str, object]],
) -> None:
  rows[:] = [row for row in rows if int(row["iter"]) != iteration]
  rows.extend(replacements)


def _write_iteration_samples(
  intermediate_dir: Path,
  iteration: int,
  sample_psm: pl.DataFrame,
  sample_rs: pl.DataFrame,
) -> None:
  sample_dir = intermediate_dir / "samples"
  sample_dir.mkdir(parents=True, exist_ok=True)
  for name, frame in (("psm", sample_psm), ("rs", sample_rs)):
    destination = sample_dir / f"iter_{iteration:06d}_{name}.csv"
    temporary = destination.with_suffix(".csv.tmp")
    frame.write_csv(temporary)
    temporary.replace(destination)


def _write_bootstrap_summary(
  estimates_df: pl.DataFrame, diagnostics_df: pl.DataFrame, path: Path
) -> None:
  parameter_cols = [
    col for col in estimates_df.columns if col not in {"iter", "path", "fit_type"}
  ]
  aggregates: list[pl.Expr] = []
  for col in parameter_cols:
    aggregates.extend(
      [
        pl.col(col).mean().alias(f"{col}_mean"),
        pl.col(col).std().alias(f"{col}_sd"),
        pl.col(col).quantile(0.025).alias(f"{col}_q025"),
        pl.col(col).quantile(0.975).alias(f"{col}_q975"),
      ]
    )
  fit_counts = diagnostics_df.group_by("path", "fit_type").agg(
    pl.len().alias("total_count"),
    pl.col("success").sum().alias("success_count"),
  )
  successful_keys = diagnostics_df.filter(pl.col("success")).select(
    "iter", "path", "fit_type"
  )
  successful_estimates = estimates_df.join(
    successful_keys, on=["iter", "path", "fit_type"], how="inner"
  )
  if successful_estimates.is_empty():
    fit_counts.sort("path", "fit_type").write_csv(path)
    return
  summary = successful_estimates.group_by("path", "fit_type").agg(aggregates)
  fit_counts.join(summary, on=["path", "fit_type"], how="left").sort(
    "path", "fit_type"
  ).write_csv(path)


def run_external_calibration_bootstrap(
  config: ExternalCalibrationBootstrapConfig,
) -> Path:
  """Run source bootstrap estimation against fixed external calibration inputs."""

  context = _prepare_external_context(config)
  run_dir = (
    Path(config.resume_run_dir)
    if config.resume_run_dir is not None
    else Path(config.output_root)
    / f"python_external_{config.run_label}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
  )
  intermediate_dir = run_dir / "intermediate"
  final_dir = run_dir / "final"
  intermediate_dir.mkdir(parents=True, exist_ok=True)
  final_dir.mkdir(parents=True, exist_ok=True)
  fingerprint = _configuration_fingerprint(context)
  fingerprint_path = intermediate_dir / "configuration_fingerprint.txt"
  if fingerprint_path.exists():
    existing = fingerprint_path.read_text(encoding="utf-8").strip()
    if existing != fingerprint:
      raise ValueError("resume_run_dir configuration fingerprint does not match")
  else:
    fingerprint_path.write_text(fingerprint + "\n", encoding="utf-8")
    _write_external_calibration(context, intermediate_dir / "external_calibration.csv")

  checkpoint_path = intermediate_dir / "checkpoint.json"
  if checkpoint_path.exists():
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    estimates = list(checkpoint["estimates"])
    diagnostics = list(checkpoint["diagnostics"])
  else:
    estimates = []
    diagnostics = []
  completed = _completed_iterations(diagnostics)
  seeds = np.random.SeedSequence(config.seed).spawn(config.nsim)
  tasks = [
    (iteration, int(seed_sequence.generate_state(1, dtype=np.uint64)[0]))
    for iteration, seed_sequence in enumerate(seeds, start=1)
    if iteration not in completed
  ]
  processed = 0
  try:
    if config.n_jobs == 1:
      results = (_run_external_task(task, context) for task in tasks)
      for result in results:
        iteration, iter_estimates, iter_diagnostics, sample_psm, sample_rs = result
        _replace_iteration_rows(estimates, iteration, iter_estimates)
        _replace_iteration_rows(diagnostics, iteration, iter_diagnostics)
        if sample_psm is not None and sample_rs is not None:
          _write_iteration_samples(intermediate_dir, iteration, sample_psm, sample_rs)
        processed += 1
        if processed % config.checkpoint_every == 0:
          _write_checkpoint(estimates, diagnostics, intermediate_dir)
        if config.print_every > 0 and iteration % config.print_every == 0:
          print(f"[external-calibration] iter={iteration}/{config.nsim}")
    else:
      with ProcessPoolExecutor(
        max_workers=config.n_jobs,
        initializer=_initialize_external_worker,
        initargs=(context,),
      ) as executor:
        futures = [executor.submit(_run_external_task, task) for task in tasks]
        for future in as_completed(futures):
          iteration, iter_estimates, iter_diagnostics, sample_psm, sample_rs = (
            future.result()
          )
          _replace_iteration_rows(estimates, iteration, iter_estimates)
          _replace_iteration_rows(diagnostics, iteration, iter_diagnostics)
          if sample_psm is not None and sample_rs is not None:
            _write_iteration_samples(intermediate_dir, iteration, sample_psm, sample_rs)
          processed += 1
          if processed % config.checkpoint_every == 0:
            _write_checkpoint(estimates, diagnostics, intermediate_dir)
          if config.print_every > 0 and iteration % config.print_every == 0:
            print(f"[external-calibration] iter={iteration}/{config.nsim}")
  finally:
    _write_checkpoint(estimates, diagnostics, intermediate_dir)

  estimates_df = rows_to_frame(
    estimates,
    columns=["iter", "path", "fit_type", *_theta_columns(list(context.x_cols))],
  ).sort("iter", "path", "fit_type")
  diagnostics_df = rows_to_frame(
    diagnostics,
    columns=[
      "iter",
      "path",
      "fit_type",
      "success",
      "status",
      "objective",
      "max_violation",
      "solver",
    ],
  ).sort("iter", "path", "fit_type")
  for path in ("PSM", "RS"):
    for fit_type in ("CMLE", "MLE"):
      estimates_df.filter(
        (pl.col("path") == path) & (pl.col("fit_type") == fit_type)
      ).drop("path", "fit_type").write_csv(
        final_dir / f"est_{'cml' if fit_type == 'CMLE' else 'ml'}_{path.lower()}.csv"
      )
  diagnostics_df.write_csv(final_dir / "fit_diagnostics.csv")
  _write_bootstrap_summary(
    estimates_df, diagnostics_df, final_dir / "bootstrap_summary.csv"
  )
  pl.DataFrame(
    {
      "mode": ["external_calibration"],
      "seed": [config.seed],
      "nsim": [config.nsim],
      "Nsource": [len(context.source)],
      "Ntarget": [config.n_target],
      "samplesize": [config.sample_size],
      "z_origin_scale": [config.z_origin_scale],
      "x_cols": [";".join(context.x_cols)],
      "n_jobs": [config.n_jobs],
      "checkpoint_every": [config.checkpoint_every],
      "configuration_fingerprint": [fingerprint],
    }
  ).write_csv(final_dir / "run_metadata.csv")
  return run_dir
