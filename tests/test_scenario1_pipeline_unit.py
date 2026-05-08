from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

from risk_bridge.config import FeatureSpec, UserDataRunConfig, UserDataSchema
from risk_bridge.cli import (
  build_scenario1_run_config,
  build_scenario2_run_config,
  build_scenario3_run_config,
  run_scenario1_pipeline,
  run_user_data_pipeline,
)
from risk_bridge.runs import (
  EvaluationInputs,
  PathFits,
  _accuracy_row,
  _iteration_export_rows,
  _scenario_runtime,
  _theta_row_dict,
)
from risk_bridge.simulate import generate_population
from risk_bridge.types import FitResult


def _assert_simulated_run_contract(
  *,
  run_dir: Path,
  scenario_name: str,
  run_prefix: str,
) -> pl.DataFrame:
  final_dir = run_dir / "final"

  assert run_dir.name.startswith(run_prefix)
  assert (final_dir / "est_cml_psm.csv").exists()
  assert (final_dir / "roc_metrics.csv").exists()
  assert (final_dir / "run_metadata.csv").exists()
  assert (final_dir / "fit_diagnostics.csv").exists()

  metadata = pl.read_csv(final_dir / "run_metadata.csv")
  assert metadata.get_column("scenario").item(0) == scenario_name
  assert "source_miscalibration_a" in metadata.columns
  assert "source_miscalibration_b" in metadata.columns
  assert "target_feature_specs" in metadata.columns
  assert "source_feature_specs" in metadata.columns
  return metadata


def test_run_scenario1_pipeline_smoke(tmp_path: Path) -> None:
  cfg = build_scenario1_run_config(
    seed=7,
    nsim=1,
    n_target=120,
    n_source=80,
    n_reference=120,
    sample_size=20,
    alpha=-3.8,
    beta_z=0.9,
    output_root=str(tmp_path),
    maxiter=30,
  )

  run_dir = run_scenario1_pipeline(cfg, print_every=0)
  final_dir = run_dir / "final"
  intermediate_dir = run_dir / "intermediate"

  metadata = _assert_simulated_run_contract(
    run_dir=run_dir,
    scenario_name="Scenario1",
    run_prefix="python_scenario1_",
  )

  est_cml_psm = pl.read_csv(final_dir / "est_cml_psm.csv")
  assert len(est_cml_psm) == 1
  assert "alpha" in est_cml_psm.columns
  assert "beta_Zcat" in est_cml_psm.columns

  assert "y_prev" in metadata.columns
  assert "betaz" in metadata.columns
  assert "alpha" in metadata.columns
  assert "n_jobs" in metadata.columns
  assert "path_jobs" in metadata.columns
  assert "intermediate_flush_every" in metadata.columns
  assert int(metadata.get_column("n_jobs").item(0)) == 1
  assert int(metadata.get_column("path_jobs").item(0)) == 1

  sample_psm = pl.read_csv(intermediate_dir / "sample_psm_by_iter.csv")
  assert len(sample_psm) == cfg.sim.sample_size
  assert set(["iter", "caseY", "X1", "X2", "X3", "X4", "zOrigin", "zCat"]).issubset(
    sample_psm.columns
  )

  fit_diagnostics = pl.read_csv(final_dir / "fit_diagnostics.csv")
  assert set(fit_diagnostics.get_column("path").to_list()) == {"PSM", "RS"}


def test_run_scenario2_pipeline_smoke(tmp_path: Path) -> None:
  cfg = build_scenario2_run_config(
    nsim=1,
    n_target=120,
    n_source=80,
    n_reference=120,
    sample_size=20,
    output_root=str(tmp_path),
    maxiter=12,
  )

  run_dir = run_scenario1_pipeline(cfg, print_every=0)
  metadata = _assert_simulated_run_contract(
    run_dir=run_dir,
    scenario_name="Scenario2",
    run_prefix="python_scenario2_",
  )

  assert float(metadata.get_column("source_miscalibration_a").item(0)) == 0.5
  assert float(metadata.get_column("source_miscalibration_b").item(0)) == 1.0
  assert (
    metadata.get_column("target_feature_specs").item(0)
    != metadata.get_column("source_feature_specs").item(0)
  )


def test_run_scenario3_pipeline_smoke(tmp_path: Path) -> None:
  cfg = build_scenario3_run_config(
    nsim=1,
    n_target=120,
    n_source=80,
    n_reference=120,
    sample_size=20,
    output_root=str(tmp_path),
    maxiter=12,
  )

  run_dir = run_scenario1_pipeline(cfg, print_every=0)
  metadata = _assert_simulated_run_contract(
    run_dir=run_dir,
    scenario_name="Scenario3",
    run_prefix="python_scenario3_",
  )

  assert float(metadata.get_column("source_miscalibration_a").item(0)) == -0.5
  assert float(metadata.get_column("source_miscalibration_b").item(0)) == 1.2


def test_run_user_data_pipeline_smoke(tmp_path: Path) -> None:
  rng = np.random.default_rng(17)
  feature_specs = (
    FeatureSpec("X1", "categorical_cut", {"breaks": (0.0, 0.4, 1.0)}),
    FeatureSpec("X2", "categorical_cut", {"breaks": (0.0, 0.3, 1.0)}),
    FeatureSpec("X3", "capped_poisson", {"lambda": 0.4, "cap": 1}),
    FeatureSpec("X4", "capped_poisson", {"lambda": 0.3, "cap": 1}),
  )
  z_bins = (0.3, 0.6, 0.9)
  gamma = np.array([-1.2, 0.2, -0.1, 0.1, 0.0, 0.5], dtype=np.float64)
  beta = np.array([0.4, 0.2, -0.1, 0.1, 0.35], dtype=np.float64)

  def make_df(n: int, alpha: float) -> pd.DataFrame:
    pop = generate_population(
      rng=rng,
      n=n,
      feature_specs=feature_specs,
      gamma=gamma,
      z_bins=np.array(z_bins, dtype=np.float64),
      alpha=alpha,
      beta=beta,
    )
    df = pop.X.with_columns(
      pl.Series("caseY", pop.y),
      pl.Series("zOrigin", pop.z_cont),
      pl.Series("zCat", pop.z_cat),
    ).select(["caseY", *pop.X.columns, "zOrigin", "zCat"])
    return pd.DataFrame(df.to_dict(as_series=False))

  target_df = make_df(80, alpha=-2.1)
  source_df = make_df(70, alpha=-2.0)
  reference_df = make_df(90, alpha=-2.1)
  target_df = target_df.rename(columns={"caseY": "label"}).drop(columns=["zCat"])
  source_df = source_df.rename(columns={"caseY": "label"}).drop(columns=["zCat"])
  reference_df = reference_df.rename(columns={"caseY": "label"}).drop(columns=["zCat"])

  run_dir = run_user_data_pipeline(
    UserDataRunConfig(
      target_df=target_df,
      source_df=source_df,
      reference_df=reference_df,
      schema=UserDataSchema(
        x_cols=("X1", "X2", "X3", "X4"),
        z_bins=z_bins,
        y_col="label",
        z_origin_col="zOrigin",
        z_cat_col="zCat",
      ),
      seed=5,
      nsim=1,
      sample_size=24,
      target_fpr=0.1,
      maxiter=20,
      path_jobs=2,
      output_root=str(tmp_path),
      print_every=0,
    )
  )

  final_dir = run_dir / "final"
  assert (final_dir / "est_cml_psm.csv").exists()
  assert (final_dir / "run_metadata.csv").exists()
  meta = pl.read_csv(final_dir / "run_metadata.csv")
  assert meta.get_column("mode").item(0) == "user_data"
  assert int(meta.get_column("path_jobs").item(0)) == 2


def test_run_scenario1_pipeline_parallel_smoke(tmp_path: Path) -> None:
  cfg = build_scenario1_run_config(
    seed=11,
    nsim=2,
    n_target=60,
    n_source=40,
    n_reference=60,
    sample_size=15,
    alpha=-3.8,
    beta_z=0.9,
    output_root=str(tmp_path),
    maxiter=8,
  )

  run_dir = run_scenario1_pipeline(cfg, n_jobs=2, print_every=0)
  final_dir = run_dir / "final"
  est = pl.read_csv(final_dir / "est_cml_psm.csv")
  meta = pl.read_csv(final_dir / "run_metadata.csv")

  assert len(est) == 2
  assert int(meta.get_column("n_jobs").item(0)) == 2


def test_run_scenario1_pipeline_path_parallel_smoke(tmp_path: Path) -> None:
  cfg = build_scenario1_run_config(
    seed=19,
    nsim=1,
    n_target=60,
    n_source=40,
    n_reference=60,
    sample_size=15,
    alpha=-3.8,
    beta_z=0.9,
    output_root=str(tmp_path),
    maxiter=8,
  )

  run_dir = run_scenario1_pipeline(
    cfg, n_jobs=1, path_jobs=2, intermediate_flush_every=1, print_every=0
  )
  meta = pl.read_csv(run_dir / "final" / "run_metadata.csv")
  assert int(meta.get_column("n_jobs").item(0)) == 1
  assert int(meta.get_column("path_jobs").item(0)) == 2
  assert int(meta.get_column("intermediate_flush_every").item(0)) == 1


def test_functional_theta_and_accuracy_rows_are_stable() -> None:
  theta_row = _theta_row_dict(
    iteration=3,
    theta_cols=["alpha", "beta_X1"],
    theta=np.array([-1.25, 0.5], dtype=np.float64),
  )
  assert theta_row == {"iter": 3, "alpha": -1.25, "beta_X1": 0.5}

  assert _accuracy_row(3, {"tpr": 0.1, "ppv": 0.2, "tnr": 0.3}) == {
    "iter": 3,
    "TPR": 0.1,
    "PPV": 0.2,
    "TNR": 0.3,
  }


def test_functional_iteration_export_rows_preserve_contract() -> None:
  def fit(theta: tuple[float, float]) -> FitResult:
    return FitResult(
      theta=np.array(theta, dtype=np.float64),
      success=True,
      status="ok",
      n_iter=2,
      objective=1.5,
      diagnostics={"max_violation": 0.0},
    )

  rows = _iteration_export_rows(
    fits=PathFits(
      mle_psm=fit((-0.2, 0.1)),
      cmle_psm=fit((-0.3, 0.2)),
      mle_rs=fit((-0.4, 0.3)),
      cmle_rs=fit((-0.5, 0.4)),
    ),
    theta_cols=["alpha", "beta_Zcat"],
    eval_inputs=EvaluationInputs(
      iteration=7,
      x_cols=[],
      roc_x=np.empty((4, 0), dtype=np.float64),
      roc_zc=np.array([0, 1, 0, 1], dtype=np.int64),
      roc_y=np.array([0, 1, 0, 1], dtype=np.int64),
      threshold=0.5,
      base_score=np.array([0.1, 0.8, 0.2, 0.9], dtype=np.float64),
      ref_score=np.array([0.2, 0.7, 0.3, 0.8], dtype=np.float64),
    ),
  )

  assert rows.est_cml_psm_row == {"iter": 7, "alpha": -0.3, "beta_Zcat": 0.2}
  assert set(rows.roc_row) == {
    "iter",
    "roc_CML_PSM",
    "roc_CML_RS",
    "roc_ML_PSM",
    "roc_ML_RS",
    "roc_base",
    "roc_ref",
  }
  assert rows.fit_diag_rows[0]["path"] == "PSM"
  assert rows.fit_diag_rows[1]["path"] == "RS"
  assert rows.fit_diag_rows[0]["cmle_max_violation"] == 0.0


def test_scenario_runtime_returns_explicit_result_for_invalid_options() -> None:
  cfg = build_scenario1_run_config()

  result = _scenario_runtime(
    cfg=cfg,
    x_cols=[spec.name for spec in cfg.sim.feature_specs],
    miscalibration_a=None,
    miscalibration_b=None,
    init_theta=tuple([0.0]),
    n_jobs=1,
    path_jobs=1,
    intermediate_flush_every=1,
    run_label=None,
  )

  assert getattr(result, "error") == "init_theta length must be 12; got 1"
