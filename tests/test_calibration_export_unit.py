from pathlib import Path

import numpy as np
import polars as pl

from risk_bridge.cli import build_scenario2_run_config, run_scenario1_pipeline
from risk_bridge.output_schema import OUTPUT_SCHEMA_VERSION


def test_scenario2_exports_four_path_calibration_summaries(tmp_path: Path) -> None:
  cfg = build_scenario2_run_config(
    seed=41,
    nsim=1,
    n_target=80,
    n_source=60,
    n_reference=80,
    sample_size=12,
    output_root=str(tmp_path),
    maxiter=8,
  )
  run_dir = run_scenario1_pipeline(cfg, print_every=0)
  final_dir = run_dir / "final"

  metadata = pl.read_csv(final_dir / "run_metadata.csv")
  assert metadata.get_column("schema_version").item(0) == OUTPUT_SCHEMA_VERSION

  cal = pl.read_csv(final_dir / "calibration_metrics.csv")
  assert cal.columns == [
    "iter",
    "estimator",
    "path",
    "calibration_in_the_large",
    "calibration_slope",
    "observed_expected_ratio",
    "brier_score",
  ]
  assert set(zip(cal["estimator"].to_list(), cal["path"].to_list(), strict=True)) == {
    ("cMLE", "PSM"),
    ("cMLE", "RS"),
    ("ML", "PSM"),
    ("ML", "RS"),
  }
  for col in (
    "calibration_in_the_large",
    "calibration_slope",
    "observed_expected_ratio",
    "brier_score",
  ):
    values = cal.get_column(col).to_numpy()
    assert values.dtype.kind == "f"
  assert np.all(np.isfinite(cal.get_column("brier_score").to_numpy()))
  assert np.all(
    (cal.get_column("brier_score").to_numpy() >= 0.0)
    & (cal.get_column("brier_score").to_numpy() <= 1.0)
  )

  residuals = pl.read_csv(final_dir / "calibration_residuals.csv")
  assert residuals.columns == [
    "iter",
    "estimator",
    "path",
    "risk_interval",
    "residual",
    "expected_risk",
    "p_external",
  ]
  pe = pl.read_csv(run_dir / "intermediate" / "pe_by_iter.csv")
  n_bins = pe.height
  assert residuals.height == 4 * n_bins
  assert set(residuals["risk_interval"].to_list()) == set(range(1, n_bins + 1))
  pe_map = dict(
    zip(pe["risk_interval"].to_list(), pe["p_external"].to_list(), strict=True)
  )
  for row in residuals.iter_rows(named=True):
    assert row["p_external"] == pe_map[row["risk_interval"]]
    assert abs(row["residual"] - (row["expected_risk"] - row["p_external"])) < 1e-12
  assert set(
    zip(residuals["estimator"].to_list(), residuals["path"].to_list(), strict=True)
  ) == {("cMLE", "PSM"), ("cMLE", "RS"), ("ML", "PSM"), ("ML", "RS")}
  assert np.all(np.isfinite(residuals.get_column("residual").to_numpy()))
