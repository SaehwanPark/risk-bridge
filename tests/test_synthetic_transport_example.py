from __future__ import annotations

from pathlib import Path
import sys

import polars as pl

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

from cases.synthetic_transport_example.suite import (  # noqa: E402
  generate_cohort_frames,
  load_case_config,
  resolve_profile,
  run_synthetic_transport_example,
)


def test_resolve_profile_smoke_and_midsize() -> None:
  config = load_case_config()
  smoke = resolve_profile(config, "smoke")
  midsize = resolve_profile(config, "midsize")
  assert smoke.nsim == 1
  assert smoke.sample_size < midsize.sample_size
  assert midsize.nsim == 5


def test_generate_cohort_frames_shapes() -> None:
  config = load_case_config()
  sizes = resolve_profile(config, "smoke")
  target_df, source_df, reference_df = generate_cohort_frames(config, sizes, seed=11)
  expected_cols = {"caseY", "X1", "X2", "X3", "X4", "zOrigin"}
  assert set(target_df.columns) == expected_cols
  assert len(target_df) == sizes.n_target
  assert len(source_df) == sizes.n_source
  assert len(reference_df) == sizes.n_reference
  assert set(target_df["caseY"].unique()) <= {0, 1}


def test_run_synthetic_transport_example_smoke(tmp_path: Path) -> None:
  manifest = run_synthetic_transport_example(
    output_root=tmp_path,
    profile="smoke",
    run_label="synthetic_smoke",
    seed=19,
  )
  case_root = Path(manifest["case_root"])
  run_dir = Path(manifest["run_dir"])
  assert (case_root / "case_manifest.json").is_file()
  assert (case_root / "environment.json").is_file()
  assert (case_root / "cohorts" / "target.csv").is_file()
  final_dir = run_dir / "final"
  assert (final_dir / "calibration_metrics.csv").is_file()
  assert (final_dir / "calibration_residuals.csv").is_file()
  meta = pl.read_csv(final_dir / "run_metadata.csv")
  assert meta.get_column("schema_version").item(0) == "1.1.0"
  assert meta.get_column("mode").item(0) == "user_data"
  cal = pl.read_csv(final_dir / "calibration_metrics.csv")
  assert set(zip(cal["estimator"].to_list(), cal["path"].to_list(), strict=True)) == {
    ("cMLE", "PSM"),
    ("cMLE", "RS"),
    ("ML", "PSM"),
    ("ML", "RS"),
  }
