"""Public library-facing API for Risk Bridge."""

from __future__ import annotations

from pathlib import Path

import polars as pl

from risk_bridge.config import RunConfig, Scenario1PipelineOptions, UserDataRunConfig
from risk_bridge.pipeline import run_pipeline, run_single_iteration_result
from risk_bridge.cli import (
  build_scenario1_run_config,
  build_scenario2_run_config,
  build_scenario3_run_config,
  run_scenario1_pipeline,
  run_simulated_pipeline,
  run_user_data_pipeline,
)
from risk_bridge.types import IterationResult

__all__ = [
  "IterationResult",
  "build_scenario1_run_config",
  "build_scenario2_run_config",
  "build_scenario3_run_config",
  "run_pipeline",
  "run_scenario1",
  "run_scenario1_pipeline",
  "run_simulation",
  "run_simulated_pipeline",
  "run_single_iteration_result",
  "run_summary",
  "run_user_data",
  "run_user_data_pipeline",
]


def run_scenario1(
  cfg: RunConfig, options: Scenario1PipelineOptions | None = None
) -> Path:
  """Run the simulated pipeline via the public API.

  The supplied `RunConfig` may describe Scenario 1, 2, or 3.
  """

  return run_simulated_pipeline(cfg, options)


def run_simulation(
  cfg: RunConfig, options: Scenario1PipelineOptions | None = None
) -> Path:
  """Run a simulated Scenario 1-3 pipeline."""

  return run_scenario1(cfg, options)


def run_user_data(config: UserDataRunConfig) -> Path:
  """Run the user-data pipeline via the public API."""

  return run_user_data_pipeline(config)


def run_summary(cfg: RunConfig) -> pl.DataFrame:
  """Run the compact development pipeline and return one summary row per iteration."""

  return run_pipeline(cfg)
