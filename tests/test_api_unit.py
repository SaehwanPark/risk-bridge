from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

from risk_bridge.api import (
  build_scenario2_run_config,
  run_scenario1,
  run_simulation,
  run_summary,
  run_user_data,
)
from risk_bridge.config import (
  FeatureSpec,
  Scenario1PipelineOptions,
  UserDataRunConfig,
  UserDataSchema,
)
from risk_bridge.cli import build_scenario1_run_config
from risk_bridge.simulate import generate_population


def test_run_scenario1_api_smoke(tmp_path: Path) -> None:
  cfg = build_scenario1_run_config(
    seed=13,
    nsim=1,
    n_target=80,
    n_source=60,
    n_reference=80,
    sample_size=12,
    output_root=str(tmp_path),
    maxiter=8,
  )

  run_dir = run_scenario1(
    cfg,
    Scenario1PipelineOptions(
      n_jobs=1,
      path_jobs=1,
      intermediate_flush_every=1,
      print_every=0,
      run_label="api_sim",
    ),
  )

  assert (run_dir / "final" / "run_metadata.csv").exists()


def test_run_user_data_api_smoke(tmp_path: Path) -> None:
  rng = np.random.default_rng(17)
  feature_specs = (
    FeatureSpec("X1", "categorical_cut", {"breaks": (0.0, 0.4, 1.0)}),
    FeatureSpec("X2", "categorical_cut", {"breaks": (0.0, 0.3, 1.0)}),
    FeatureSpec("X3", "capped_poisson", {"lambda": 0.4, "cap": 1}),
    FeatureSpec("X4", "capped_poisson", {"lambda": 0.3, "cap": 1}),
  )
  gamma = np.array([-1.2, 0.2, -0.1, 0.1, 0.0, 0.5], dtype=np.float64)
  beta = np.array([0.4, 0.2, -0.1, 0.1, 0.35], dtype=np.float64)

  def make_df(n: int, alpha: float) -> pd.DataFrame:
    pop = generate_population(
      rng=rng,
      n=n,
      feature_specs=feature_specs,
      gamma=gamma,
      z_bins=np.array((0.3, 0.6, 0.9), dtype=np.float64),
      alpha=alpha,
      beta=beta,
    )
    df = pop.X.with_columns(
      pl.Series("label", pop.y),
      pl.Series("zOrigin", pop.z_cont),
    ).select(["label", *pop.X.columns, "zOrigin"])
    return pd.DataFrame(df.to_dict(as_series=False))

  run_dir = run_user_data(
    UserDataRunConfig(
      target_df=make_df(80, alpha=-2.1),
      source_df=make_df(70, alpha=-2.0),
      reference_df=make_df(90, alpha=-2.1),
      schema=UserDataSchema(
        x_cols=("X1", "X2", "X3", "X4"),
        z_bins=(0.3, 0.6, 0.9),
        y_col="label",
        z_origin_col="zOrigin",
      ),
      seed=5,
      nsim=1,
      sample_size=24,
      target_fpr=0.1,
      maxiter=8,
      path_jobs=2,
      output_root=str(tmp_path),
      print_every=0,
      run_label="api_user",
    )
  )

  meta = pd.read_csv(run_dir / "final" / "run_metadata.csv")
  assert meta.loc[0, "mode"] == "user_data"


def test_run_simulation_api_scenario2_smoke(tmp_path: Path) -> None:
  cfg = build_scenario2_run_config(
    seed=29,
    nsim=1,
    n_target=80,
    n_source=60,
    n_reference=80,
    sample_size=12,
    output_root=str(tmp_path),
    maxiter=8,
  )

  run_dir = run_simulation(
    cfg,
    Scenario1PipelineOptions(
      n_jobs=1,
      path_jobs=1,
      intermediate_flush_every=1,
      print_every=0,
    ),
  )

  metadata = pl.read_csv(run_dir / "final" / "run_metadata.csv")
  assert metadata.get_column("scenario").item(0) == "Scenario2"
  assert run_dir.name.startswith("python_scenario2_")
  assert (run_dir / "final" / "fit_diagnostics.csv").exists()


def test_run_summary_returns_polars_frame() -> None:
  cfg = build_scenario1_run_config(
    seed=23,
    nsim=2,
    n_target=40,
    n_source=30,
    n_reference=40,
    sample_size=10,
    output_root="data/out",
    maxiter=5,
  )
  out = run_summary(cfg)
  assert isinstance(out, pl.DataFrame)
  assert out.shape == (2, 5)
