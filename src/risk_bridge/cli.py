"""Command-line entrypoint for simulated and user-data Risk Bridge runs."""

from __future__ import annotations

import os
import sys

if __name__ == "__main__" and __package__ is None:
  script_dir = os.path.dirname(os.path.abspath(__file__))
  project_root = os.path.dirname(script_dir)
  if script_dir in sys.path:
    sys.path.remove(script_dir)
  if project_root not in sys.path:
    sys.path.insert(0, project_root)

from risk_bridge.runs import (
  build_scenario1_run_config,
  build_scenario2_run_config,
  build_scenario3_run_config,
  main,
  run_scenario1_pipeline,
  run_simulated_pipeline,
  run_user_data_pipeline,
)

__all__ = [
  "build_scenario1_run_config",
  "build_scenario2_run_config",
  "build_scenario3_run_config",
  "main",
  "run_scenario1_pipeline",
  "run_simulated_pipeline",
  "run_user_data_pipeline",
]


if __name__ == "__main__":
  main()
