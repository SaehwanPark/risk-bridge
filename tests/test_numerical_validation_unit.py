from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

from cases.numerical_validation.suite import (  # noqa: E402
  run_derivative_checks,
  run_invariance_checks,
  run_optimizer_comparison,
  run_recovery_check,
  run_suite,
)


def test_derivative_checks_pass_on_reduced_seeds() -> None:
  rows, result = run_derivative_checks(seeds=(11, 22))
  assert rows
  assert result.passed
  assert result.metrics["max_abs_error"] < 1e-3


def test_optimizer_comparison_runs() -> None:
  rows, result = run_optimizer_comparison(seed=101)
  assert len(rows) == 1
  assert result.passed


def test_recovery_check_passes() -> None:
  rows, result = run_recovery_check(seed=202)
  assert len(rows) == 1
  assert result.passed
  assert rows[0]["outcome_rmse"] < 0.35


def test_invariance_checks_pass() -> None:
  rows, result = run_invariance_checks(seed=303)
  assert len(rows) == 2
  assert result.passed


def test_full_suite_summary_shape() -> None:
  suite = run_suite(derivative_seeds=(11,))
  assert "passed" in suite
  assert len(suite["checks"]) == 4
  assert suite["environment"]["package_version"]
  assert suite["passed"]
