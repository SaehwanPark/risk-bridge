"""Generate Scenario-2-shaped synthetic cohorts and run the user-data pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any

import numpy as np
import pandas as pd
import polars as pl

from risk_bridge import __version__ as package_version
from risk_bridge import run_user_data
from risk_bridge.config import FeatureSpec, UserDataRunConfig, UserDataSchema
from risk_bridge.simulate import generate_population


CASE_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = CASE_DIR / "case_config.json"


@dataclass(frozen=True)
class ProfileSizes:
  n_target: int
  n_source: int
  n_reference: int
  sample_size: int
  nsim: int
  maxiter: int
  path_jobs: int


def _target_feature_specs() -> tuple[FeatureSpec, ...]:
  """Match Scenario 1/2 target feature support (categorical cuts + capped Poisson)."""

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
  """Match Scenario 2/3 source feature support (shifted cuts / rates)."""

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


def load_case_config(path: Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
  return json.loads(path.read_text(encoding="utf-8"))


def resolve_profile(config: dict[str, Any], profile: str) -> ProfileSizes:
  profiles = config["profiles"]
  if profile not in profiles:
    known = ", ".join(sorted(profiles))
    raise ValueError(f"unknown profile {profile!r}; expected one of: {known}")
  raw = profiles[profile]
  return ProfileSizes(
    n_target=int(raw["n_target"]),
    n_source=int(raw["n_source"]),
    n_reference=int(raw["n_reference"]),
    sample_size=int(raw["sample_size"]),
    nsim=int(raw["nsim"]),
    maxiter=int(raw["maxiter"]),
    path_jobs=int(raw["path_jobs"]),
  )


def capture_environment() -> dict[str, Any]:
  git_sha = None
  try:
    git_sha = (
      subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=CASE_DIR.parents[1],
        stderr=subprocess.DEVNULL,
        text=True,
      ).strip()
      or None
    )
  except (subprocess.CalledProcessError, FileNotFoundError, OSError):
    git_sha = None
  return {
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "python_version": sys.version,
    "platform": platform.platform(),
    "machine": platform.machine(),
    "package_version": package_version,
    "git_sha": git_sha,
  }


def _population_frame(
  *,
  rng: np.random.Generator,
  n: int,
  feature_specs: tuple[FeatureSpec, ...],
  gamma: np.ndarray,
  z_bins: np.ndarray,
  alpha: float,
  beta: np.ndarray,
) -> pd.DataFrame:
  pop = generate_population(
    rng=rng,
    n=n,
    feature_specs=feature_specs,
    gamma=gamma,
    z_bins=z_bins,
    alpha=alpha,
    beta=beta,
  )
  frame = pop.X.with_columns(
    pl.Series("caseY", pop.y),
    pl.Series("zOrigin", pop.z_cont),
  ).select(["caseY", *pop.X.columns, "zOrigin"])
  return pd.DataFrame(frame.to_dict(as_series=False))


def generate_cohort_frames(
  config: dict[str, Any],
  sizes: ProfileSizes,
  *,
  seed: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
  """Build privacy-safe target/source/reference frames with Scenario-2 shift."""

  rng = np.random.default_rng(config["seed"] if seed is None else seed)
  gamma = np.asarray(config["gamma"], dtype=np.float64)
  z_bins = np.asarray(config["z_bins"], dtype=np.float64)
  beta_target = np.asarray(config["beta"], dtype=np.float64)
  alpha_target = float(config["alpha_target"])
  a = float(config["miscalibration_a"])
  b = float(config["miscalibration_b"])
  alpha_source = a + b * alpha_target
  beta_source = b * beta_target

  target_specs = _target_feature_specs()
  source_specs = _source_feature_specs()

  target_df = _population_frame(
    rng=rng,
    n=sizes.n_target,
    feature_specs=target_specs,
    gamma=gamma,
    z_bins=z_bins,
    alpha=alpha_target,
    beta=beta_target,
  )
  source_df = _population_frame(
    rng=rng,
    n=sizes.n_source,
    feature_specs=source_specs,
    gamma=gamma,
    z_bins=z_bins,
    alpha=alpha_source,
    beta=beta_source,
  )
  reference_df = _population_frame(
    rng=rng,
    n=sizes.n_reference,
    feature_specs=target_specs,
    gamma=gamma,
    z_bins=z_bins,
    alpha=alpha_target,
    beta=beta_target,
  )
  return target_df, source_df, reference_df


def write_cohort_csvs(
  output_dir: Path,
  target_df: pd.DataFrame,
  source_df: pd.DataFrame,
  reference_df: pd.DataFrame,
) -> dict[str, Path]:
  output_dir.mkdir(parents=True, exist_ok=True)
  paths = {
    "target": output_dir / "target.csv",
    "source": output_dir / "source.csv",
    "reference": output_dir / "reference.csv",
  }
  target_df.to_csv(paths["target"], index=False)
  source_df.to_csv(paths["source"], index=False)
  reference_df.to_csv(paths["reference"], index=False)
  return paths


def run_synthetic_transport_example(
  *,
  output_root: Path,
  profile: str = "smoke",
  config_path: Path = DEFAULT_CONFIG_PATH,
  run_label: str = "synthetic_transport",
  seed: int | None = None,
) -> dict[str, Any]:
  """Generate synthetic cohorts, fit user-data paths, and return a manifest."""

  config = load_case_config(config_path)
  sizes = resolve_profile(config, profile)
  resolved_seed = int(config["seed"] if seed is None else seed)
  target_df, source_df, reference_df = generate_cohort_frames(
    config, sizes, seed=resolved_seed
  )

  run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
  case_root = output_root / "synthetic_transport_example" / f"{run_label}_{run_id}"
  cohort_dir = case_root / "cohorts"
  csv_paths = write_cohort_csvs(cohort_dir, target_df, source_df, reference_df)

  z_bins = tuple(float(v) for v in config["z_bins"])
  run_dir = run_user_data(
    UserDataRunConfig(
      target_df=target_df,
      source_df=source_df,
      reference_df=reference_df,
      schema=UserDataSchema(
        x_cols=("X1", "X2", "X3", "X4"),
        z_bins=z_bins,
        y_col="caseY",
        z_origin_col="zOrigin",
        z_cat_col="zCat",
      ),
      seed=resolved_seed,
      nsim=sizes.nsim,
      sample_size=sizes.sample_size,
      target_fpr=float(config["target_fpr"]),
      maxiter=sizes.maxiter,
      path_jobs=sizes.path_jobs,
      output_root=str(case_root / "runs"),
      print_every=0,
      run_label=run_label,
    )
  )

  environment = capture_environment()
  manifest = {
    "profile": profile,
    "seed": resolved_seed,
    "package_version": package_version,
    "cohort_paths": {key: str(path) for key, path in csv_paths.items()},
    "run_dir": str(run_dir),
    "case_root": str(case_root),
    "sizes": {
      "n_target": sizes.n_target,
      "n_source": sizes.n_source,
      "n_reference": sizes.n_reference,
      "sample_size": sizes.sample_size,
      "nsim": sizes.nsim,
      "maxiter": sizes.maxiter,
      "path_jobs": sizes.path_jobs,
    },
    "miscalibration_a": float(config["miscalibration_a"]),
    "miscalibration_b": float(config["miscalibration_b"]),
    "environment": environment,
  }
  (case_root / "case_manifest.json").write_text(
    json.dumps(manifest, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
  )
  (case_root / "environment.json").write_text(
    json.dumps(environment, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
  )
  return manifest
