from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
import polars as pl

DEFAULT_Z_BINS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
SCENARIO1_INIT_THETA = (
  -3.0,
  -0.2,
  0.2,
  0.2,
  0.2,
  0.2,
  -2.0,
  0.2,
  -0.2,
  0.2,
  0.2,
  0.5,
)


@dataclass(frozen=True)
class FeatureSpec:
  """Discrete feature generation or support specification.

  Parameters
  ----------
  name:
    Canonical column name for the feature.
  kind:
    One of `categorical_cut`, `capped_poisson`, or `custom`.
  params:
    Kind-specific parameter mapping used during simulation or support
    enumeration.
  """

  name: str
  kind: str
  params: dict[str, Any]

  def __post_init__(self) -> None:
    if not self.name:
      raise ValueError("Feature name must be non-empty.")
    if self.kind not in {"categorical_cut", "capped_poisson", "custom"}:
      raise ValueError(f"Unsupported feature kind: {self.kind}")


def _feature_support_signature(spec: FeatureSpec) -> tuple[int, ...]:
  if spec.kind == "categorical_cut":
    breaks = tuple(float(v) for v in spec.params["breaks"])
    return tuple(range(len(breaks) - 1))
  if spec.kind == "capped_poisson":
    cap = int(spec.params.get("cap", 2))
    return tuple(range(cap + 1))
  if spec.kind == "custom":
    return tuple(int(v) for v in spec.params["values"])
  raise ValueError(f"Unsupported feature kind: {spec.kind}")


@dataclass(frozen=True)
class ZModelSpec:
  """Specification for the continuous/categorical Z model.

  Parameters
  ----------
  family:
    Distribution family identifier. Only `trunc_lognormal` is supported.
  gamma_init:
    Initial gamma vector used by the truncated lognormal model.
  bins:
    Strictly increasing Z cut points in `(0, 1)`.
  """

  family: str
  gamma_init: tuple[float, ...]
  bins: tuple[float, ...]

  def __post_init__(self) -> None:
    if self.family != "trunc_lognormal":
      raise ValueError("Only 'trunc_lognormal' is currently supported.")
    if len(self.bins) == 0:
      raise ValueError("bins must be non-empty")
    prev = float("-inf")
    for b in self.bins:
      if not (0.0 < b < 1.0):
        raise ValueError("Each bin boundary must be in (0, 1).")
      if b <= prev:
        raise ValueError("bins must be strictly increasing")
      prev = b


@dataclass(frozen=True)
class SimulationConfig:
  """Synthetic data-generation parameters for one simulated run."""

  nsim: int
  n_target: int
  n_source: int
  n_reference: int
  sample_size: int
  target_prevalence: float
  target_fpr: float
  alpha: float
  beta: tuple[float, ...]
  feature_specs: tuple[FeatureSpec, ...]
  z_spec: ZModelSpec
  source_feature_specs: tuple[FeatureSpec, ...] | None = None
  source_miscalibration_a: float = 0.0
  source_miscalibration_b: float = 1.0
  scenario_name: str = "Scenario1"
  scenario_run_label: str = "scenario1"

  def __post_init__(self) -> None:
    for key, value in {
      "nsim": self.nsim,
      "n_target": self.n_target,
      "n_source": self.n_source,
      "n_reference": self.n_reference,
      "sample_size": self.sample_size,
    }.items():
      if value <= 0:
        raise ValueError(f"{key} must be > 0")
    if self.sample_size > self.n_source:
      raise ValueError("sample_size must not exceed n_source")
    if not (0.0 < self.target_prevalence < 1.0):
      raise ValueError("target_prevalence must be in (0, 1)")
    if not (0.0 < self.target_fpr < 1.0):
      raise ValueError("target_fpr must be in (0, 1)")
    if len(self.feature_specs) == 0:
      raise ValueError("feature_specs must be non-empty")
    if len(self.beta) != len(self.feature_specs) + 1:
      raise ValueError(
        "beta must contain one coefficient per X feature plus one for z_cat"
      )
    if not self.scenario_name:
      raise ValueError("scenario_name must be non-empty")
    if not self.scenario_run_label:
      raise ValueError("scenario_run_label must be non-empty")
    if self.source_feature_specs is not None:
      if len(self.source_feature_specs) != len(self.feature_specs):
        raise ValueError(
          "source_feature_specs must match feature_specs in length and order"
        )
      for target_spec, source_spec in zip(
        self.feature_specs, self.source_feature_specs, strict=True
      ):
        if target_spec.name != source_spec.name:
          raise ValueError(
            "source_feature_specs must match feature_specs by feature name and order"
          )
        if target_spec.kind != source_spec.kind:
          raise ValueError(
            "source_feature_specs must use the same feature kinds as feature_specs"
          )
        if _feature_support_signature(target_spec) != _feature_support_signature(
          source_spec
        ):
          raise ValueError(
            "source_feature_specs must preserve the same discrete support as feature_specs"
          )


@dataclass(frozen=True)
class OptimizationConfig:
  """Optimization settings shared by the pipeline solvers."""

  mle_method: str
  cmle_method: str
  tol: float
  maxiter: int

  def __post_init__(self) -> None:
    if self.tol <= 0:
      raise ValueError("tol must be > 0")
    if self.maxiter <= 0:
      raise ValueError("maxiter must be > 0")


@dataclass(frozen=True)
class RunConfig:
  """Top-level configuration for simulated cMLE pipeline runs."""

  seed: int
  output_root: str
  sim: SimulationConfig
  opt: OptimizationConfig

  def __post_init__(self) -> None:
    if self.seed < 0:
      raise ValueError("seed must be >= 0")
    if not self.output_root:
      raise ValueError("output_root must be non-empty")


@dataclass(frozen=True)
class Scenario1PipelineOptions:
  """Optional runtime settings for the Scenario-1 simulated pipeline.

  Parameters
  ----------
  miscalibration_a:
    Optional additive source-population miscalibration override.
  miscalibration_b:
    Optional multiplicative source-population miscalibration override.
  calibration_tolerance:
    Allowed calibration inequality slack passed to the cMLE constraints.
  init_theta:
    Initial parameter vector for ML/cMLE fitting.
  write_parquet:
    Whether to emit parquet mirrors of CSV exports when an engine is available.
  n_jobs:
    Process-level iteration parallelism for simulated mode.
  path_jobs:
    Thread-level PSM/RS path parallelism inside each iteration.
  intermediate_flush_every:
    Number of iterations between buffered intermediate CSV flushes.
  print_every:
    Progress-print interval. Use `0` to disable progress printing.
  run_label:
    Optional label override used in the output directory name.
  """

  miscalibration_a: float | None = None
  miscalibration_b: float | None = None
  calibration_tolerance: float = 0.1
  init_theta: tuple[float, ...] = SCENARIO1_INIT_THETA
  write_parquet: bool = False
  n_jobs: int = 1
  path_jobs: int = 1
  intermediate_flush_every: int = 25
  print_every: int = 100
  run_label: str | None = None

  def __post_init__(self) -> None:
    if self.calibration_tolerance <= 0.0:
      raise ValueError("calibration_tolerance must be > 0")
    if self.n_jobs <= 0:
      raise ValueError("n_jobs must be > 0")
    if self.path_jobs <= 0:
      raise ValueError("path_jobs must be > 0")
    if self.intermediate_flush_every <= 0:
      raise ValueError("intermediate_flush_every must be > 0")
    if self.print_every < 0:
      raise ValueError("print_every must be >= 0")
    if self.run_label is not None and not self.run_label:
      raise ValueError("run_label must be non-empty")


@dataclass(frozen=True)
class UserDataSchema:
  """Column mapping and categorical-Z settings for user-provided data.

  Parameters
  ----------
  x_cols:
    Feature columns to treat as discrete X inputs.
  z_bins:
    Strictly increasing Z-category cut points in `(0, 1)`.
  y_col:
    Outcome column name in the user-provided DataFrames.
  z_origin_col:
    Continuous-Z column name in the user-provided DataFrames.
  z_cat_col:
    Categorical-Z column name in the user-provided DataFrames.
  allow_z_origin_from_z_cat:
    Whether missing `zOrigin` may be derived from `zCat` interval midpoints.
  """

  x_cols: tuple[str, ...]
  z_bins: tuple[float, ...] = DEFAULT_Z_BINS
  y_col: str = "caseY"
  z_origin_col: str = "zOrigin"
  z_cat_col: str = "zCat"
  allow_z_origin_from_z_cat: bool = False

  def __post_init__(self) -> None:
    if not self.x_cols:
      raise ValueError("x_cols must be non-empty")
    if any(not col for col in self.x_cols):
      raise ValueError("x_cols must not contain empty names")
    if not self.y_col:
      raise ValueError("y_col must be non-empty")
    if not self.z_origin_col:
      raise ValueError("z_origin_col must be non-empty")
    if not self.z_cat_col:
      raise ValueError("z_cat_col must be non-empty")
    if not self.z_bins:
      raise ValueError("z_bins must be non-empty")
    prev = float("-inf")
    for boundary in self.z_bins:
      if not (0.0 < boundary < 1.0):
        raise ValueError("z_bins must contain values in (0, 1)")
      if boundary <= prev:
        raise ValueError("z_bins must be strictly increasing")
      prev = boundary


@dataclass(frozen=True)
class UserDataRunConfig:
  """Library-facing configuration for end-to-end user-data pipeline runs.

  Parameters
  ----------
  target_df, source_df, reference_df:
    Input datasets supplied as pandas or polars DataFrames.
  schema:
    Column mapping and Z-category settings for the input datasets.
  seed:
    Random seed for sampling and repeated iterations.
  nsim:
    Number of repeated sampling/fitting iterations to run.
  sample_size:
    Per-iteration source/target sample size for fitting/evaluation.
  target_fpr:
    False-positive-rate target used for threshold selection.
  maxiter:
    Maximum iterations for ML/cMLE optimizers.
  n_jobs:
    Process-level iteration parallelism. User-data mode currently requires `1`.
  path_jobs:
    Thread-level parallelism across the PSM and RS paths.
  intermediate_flush_every:
    Number of iterations between buffered intermediate CSV flushes.
  feasibility_tol:
    Maximum accepted post-solve calibration violation.
  calibration_tolerance:
    Allowed calibration inequality slack passed to the cMLE constraints.
  init_theta:
    Initial parameter vector for ML/cMLE fitting.
  output_root:
    Root directory for run outputs.
  write_parquet:
    Whether to emit parquet mirrors of CSV exports when an engine is available.
  print_every:
    Progress-print interval. Use `0` to disable progress printing.
  run_label:
    Label used in the output directory name.
  """

  target_df: pl.DataFrame | pd.DataFrame
  source_df: pl.DataFrame | pd.DataFrame
  reference_df: pl.DataFrame | pd.DataFrame
  schema: UserDataSchema
  seed: int = 631
  nsim: int = 1
  sample_size: int = 1000
  target_fpr: float = 0.1
  maxiter: int = 200
  n_jobs: int = 1
  path_jobs: int = 1
  intermediate_flush_every: int = 25
  feasibility_tol: float = 1e-6
  calibration_tolerance: float = 0.1
  init_theta: tuple[float, ...] = SCENARIO1_INIT_THETA
  output_root: str = "data"
  write_parquet: bool = False
  print_every: int = 100
  run_label: str = "user_data"

  def __post_init__(self) -> None:
    if self.seed < 0:
      raise ValueError("seed must be >= 0")
    if self.nsim <= 0:
      raise ValueError("nsim must be > 0")
    if self.sample_size <= 0:
      raise ValueError("sample_size must be > 0")
    if not (0.0 < self.target_fpr < 1.0):
      raise ValueError("target_fpr must be in (0, 1)")
    if self.maxiter <= 0:
      raise ValueError("maxiter must be > 0")
    if self.n_jobs <= 0:
      raise ValueError("n_jobs must be > 0")
    if self.path_jobs <= 0:
      raise ValueError("path_jobs must be > 0")
    if self.intermediate_flush_every <= 0:
      raise ValueError("intermediate_flush_every must be > 0")
    if self.feasibility_tol <= 0.0:
      raise ValueError("feasibility_tol must be > 0")
    if self.calibration_tolerance <= 0.0:
      raise ValueError("calibration_tolerance must be > 0")
    if not self.output_root:
      raise ValueError("output_root must be non-empty")
    if self.print_every < 0:
      raise ValueError("print_every must be >= 0")
    if not self.run_label:
      raise ValueError("run_label must be non-empty")
