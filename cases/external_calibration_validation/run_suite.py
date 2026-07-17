"""Regenerate external-calibration validation artifacts under data/."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

CASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = CASE_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

from cases.external_calibration_validation.suite import (  # noqa: E402
  PROFILES,
  run_suite,
  write_suite_artifacts,
)


def main(argv: list[str] | None = None) -> int:
  parser = argparse.ArgumentParser(
    description=(
      "Run synthetic fixed-summary external-calibration recovery/calibration "
      "validation and write regenerable artifacts."
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
    help="smoke (CI-sized) or full (manuscript-citable) profile.",
  )
  parser.add_argument(
    "--run-label",
    default="external_calibration_validation",
    help="Label embedded in the output directory name.",
  )
  parser.add_argument(
    "--seed",
    type=int,
    default=475,
    help="RNG seed for the synthetic DGP and bootstrap (default: 475).",
  )
  parser.add_argument(
    "--condition",
    choices=("matched", "degraded"),
    default="matched",
    help="Matched validation or prespecified fixed-summary degradation control.",
  )
  args = parser.parse_args(argv)

  suite = run_suite(
    profile=args.profile,
    seed=args.seed,
    output_root=args.output_root,
    run_label=args.run_label,
    condition=args.condition,
  )
  out_dir = write_suite_artifacts(suite)
  print(f"Wrote external-calibration validation artifacts under: {out_dir}")
  print(f"Overall passed: {suite['passed']}")
  for check in suite["checks"]:
    status = "PASS" if check["passed"] else "FAIL"
    print(f"  [{status}] {check['name']}: {check['detail']}")
  if not suite["passed"]:
    print(
      "Demotion: do not promote external-calibration claims until gates pass."
    )
  return 0 if suite["passed"] else 1


if __name__ == "__main__":
  raise SystemExit(main())
