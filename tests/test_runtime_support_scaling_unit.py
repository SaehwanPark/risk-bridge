from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

from cases.runtime_support_scaling import run_suite as run_suite_cli  # noqa: E402
from cases.runtime_support_scaling.suite import (  # noqa: E402
  PROFILES,
  assert_finite_timings,
  resolve_profile,
  run_runtime_protocol,
  run_suite,
  write_suite_artifacts,
)


def test_resolve_profile_rejects_unknown() -> None:
  try:
    resolve_profile("not-a-profile")
  except ValueError as exc:
    assert "unknown profile" in str(exc)
  else:
    raise AssertionError("expected ValueError for unknown profile")


def test_smoke_suite_shape_and_environment() -> None:
  suite = run_suite(profile="smoke")
  assert suite["passed"]
  assert suite["profile"] == "smoke"
  assert suite["protocol_version"]

  env = suite["environment"]
  for key in (
    "platform",
    "machine",
    "cpu_count",
    "package_version",
    "n_jobs",
    "path_jobs",
    "thread_env_after_runner",
    "pin_blas_threads_requested",
    "blas_pinning_note",
    "warmups",
    "timed_reps",
    "protocol_version",
  ):
    assert key in env
  assert env["process_start_method"] is None

  runtime_rows = suite["runtime_rows"]
  assert runtime_rows
  phases = {row["phase"] for row in runtime_rows}
  assert "summary_pipeline_compute" in phases
  assert "export_pipeline_with_io" in phases
  assert any(row["include_io"] is False for row in runtime_rows)
  assert any(row["include_io"] is True for row in runtime_rows)
  assert_finite_timings(runtime_rows, key="elapsed_s")
  assert_finite_timings(runtime_rows, key="process_cpu_s")

  scaling_rows = suite["scaling_rows"]
  statuses = {str(row["status"]) for row in scaling_rows}
  assert "ok" in statuses
  assert "rejected_by_cap" in statuses
  caps = {
    str(row["cap_triggered"])
    for row in scaling_rows
    if row["status"] == "rejected_by_cap"
  }
  assert "max_total_combinations" in caps
  assert "max_levels_per_feature" in caps
  assert any(row.get("peak_rss_mb") is not None for row in scaling_rows if row["status"] == "ok")


def test_warmup_discard_renumbers_timed_reps() -> None:
  profile = replace(
    PROFILES["smoke"],
    warmups=1,
    timed_reps=1,
    support_points=((2, 2),),
    include_cap_rejection=False,
    include_level_cap_rejection=False,
  )
  rows = run_runtime_protocol(profile)
  reps = {int(row["rep"]) for row in rows}
  assert reps == {1}
  phases = {str(row["phase"]) for row in rows}
  assert "enumerate_support" in phases
  assert "export_pipeline_with_io" in phases


def test_write_suite_artifacts(tmp_path: Path) -> None:
  suite = run_suite(profile="smoke")
  out = write_suite_artifacts(suite, tmp_path / "out")
  assert (out / "summary.json").is_file()
  assert (out / "environment.json").is_file()
  assert (out / "runtime_protocol.csv").is_file()
  assert (out / "support_scaling.csv").is_file()


def test_cli_main_writes_artifacts(tmp_path: Path) -> None:
  code = run_suite_cli.main(
    [
      "--output-root",
      str(tmp_path),
      "--profile",
      "smoke",
      "--run-label",
      "unit_cli",
    ]
  )
  assert code == 0
  written = list((tmp_path / "runtime_support_scaling").glob("unit_cli_*"))
  assert len(written) == 1
  assert (written[0] / "runtime_protocol.csv").is_file()
