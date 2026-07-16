"""Regenerate independent numerical validation artifacts under data/."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys

CASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = CASE_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

from cases.numerical_validation.suite import run_suite, write_suite_artifacts


def main(argv: list[str] | None = None) -> int:
  parser = argparse.ArgumentParser(
    description="Run independent numerical validation and write regenerable artifacts."
  )
  parser.add_argument(
    "--output-root",
    type=Path,
    default=Path("data"),
    help="Root directory for artifacts (default: data).",
  )
  parser.add_argument(
    "--run-label",
    default="numerical_validation",
    help="Label embedded in the output directory name.",
  )
  args = parser.parse_args(argv)

  suite = run_suite()
  run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
  out_dir = args.output_root / "numerical_validation" / f"{args.run_label}_{run_id}"
  write_suite_artifacts(suite, out_dir)
  print(f"Wrote numerical validation artifacts under: {out_dir}")
  print(f"Overall passed: {suite['passed']}")
  for check in suite["checks"]:
    status = "PASS" if check["passed"] else "FAIL"
    print(f"  [{status}] {check['name']}: {check['detail']}")
  return 0 if suite["passed"] else 1


if __name__ == "__main__":
  raise SystemExit(main())
