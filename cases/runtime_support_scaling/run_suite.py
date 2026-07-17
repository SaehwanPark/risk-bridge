"""Regenerate runtime and support-scaling protocol artifacts under data/."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import sys

# Request single-threaded native pools before importing numpy-backed modules.
for _thread_env in (
  "OMP_NUM_THREADS",
  "OPENBLAS_NUM_THREADS",
  "MKL_NUM_THREADS",
  "NUMEXPR_NUM_THREADS",
  "POLARS_MAX_THREADS",
):
  os.environ.setdefault(_thread_env, "1")

CASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = CASE_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

from cases.runtime_support_scaling.suite import (  # noqa: E402
  PROFILES,
  run_suite,
  write_suite_artifacts,
)


def main(argv: list[str] | None = None) -> int:
  parser = argparse.ArgumentParser(
    description=(
      "Run the controlled runtime / Cartesian-support scaling protocol "
      "and write regenerable artifacts."
    )
  )
  parser.add_argument(
    "--output-root",
    type=Path,
    default=Path("data"),
    help="Root directory for artifacts (default: data).",
  )
  parser.add_argument(
    "--profile",
    choices=sorted(PROFILES),
    default="smoke",
    help="Protocol profile: smoke (fast) or protocol (manuscript regen).",
  )
  parser.add_argument(
    "--run-label",
    default="runtime_support_scaling",
    help="Label embedded in the output directory name.",
  )
  args = parser.parse_args(argv)

  suite = run_suite(profile=args.profile)
  run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
  out_dir = (
    args.output_root / "runtime_support_scaling" / f"{args.run_label}_{run_id}"
  )
  write_suite_artifacts(suite, out_dir)
  print(f"Wrote runtime/support-scaling artifacts under: {out_dir}")
  print(f"Profile: {suite['profile']}")
  print(f"Overall passed: {suite['passed']}")
  for check in suite["checks"]:
    status = "PASS" if check["passed"] else "FAIL"
    print(f"  [{status}] {check['name']}: {check['detail']}")
  return 0 if suite["passed"] else 1


if __name__ == "__main__":
  raise SystemExit(main())
