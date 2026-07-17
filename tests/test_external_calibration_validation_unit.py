from __future__ import annotations

from pathlib import Path
import json
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

from cases.external_calibration_validation.suite import (  # noqa: E402
  DEGRADED_LOGIT_SHIFT,
  PROFILES,
  _mc_summary,
  run_suite,
  write_suite_artifacts,
)


def test_smoke_external_calibration_validation_gates(tmp_path: Path) -> None:
  suite = run_suite(
    profile="smoke",
    seed=475,
    output_root=tmp_path,
    run_label="unit_smoke",
  )
  out = write_suite_artifacts(suite)

  assert suite["passed"], suite["checks"]
  assert (out / "summary.json").is_file()
  assert (out / "environment.json").is_file()
  assert (out / "fixed_summary.json").is_file()
  assert (out / "recovery.csv").is_file()
  assert (out / "calibration_gates.csv").is_file()
  assert (out / "fit_diagnostics_summary.csv").is_file()

  summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
  assert summary["passed"] is True
  assert {check["name"] for check in summary["checks"]} == {
    "feasibility",
    "calibration",
    "recovery",
  }
  assert all(check["passed"] for check in summary["checks"])

  env = json.loads((out / "environment.json").read_text(encoding="utf-8"))
  assert env["case"] == "external_calibration_validation"
  assert env["profile"] == "smoke"
  assert env["package_version"]
  assert "git_sha" in env
  assert env["reproducibility_contract"]["version"]
  assert env["thread_environment"]

  fixed = json.loads((out / "fixed_summary.json").read_text(encoding="utf-8"))
  assert len(fixed["feature_distributions"]) == 4
  assert len(fixed["p_external"]) >= 2
  assert len(fixed["x_interval_index"]) > len(fixed["p_external"])
  assert set(fixed["x_interval_index"]) <= set(range(1, len(fixed["p_external"]) + 1))


def test_profiles_are_documented() -> None:
  assert set(PROFILES) == {"smoke", "full"}
  assert PROFILES["smoke"].nsim == 1
  assert PROFILES["full"].nsim == 50


def test_mc_summary_uses_successful_iteration_sample_error() -> None:
  summary = _mc_summary([1.0, 2.0, 3.0])

  assert summary["n_successful"] == 3
  assert summary["mean"] == 2.0
  assert summary["sd"] == 1.0
  assert summary["mcse"] == 1.0 / np.sqrt(3.0)


def test_single_iteration_mc_summary_has_no_mcse() -> None:
  summary = _mc_summary([1.0])

  assert summary["n_successful"] == 1
  assert summary["mean"] == 1.0
  assert summary["sd"] is None
  assert summary["mcse"] is None


def test_degraded_condition_is_prespecified() -> None:
  assert DEGRADED_LOGIT_SHIFT == 6.0


def test_degraded_smoke_condition_fails_a_gate(tmp_path: Path) -> None:
  suite = run_suite(
    profile="smoke",
    seed=475,
    output_root=tmp_path,
    run_label="unit_degraded",
    condition="degraded",
  )

  assert not suite["passed"]
  assert any(not check["passed"] for check in suite["checks"])
