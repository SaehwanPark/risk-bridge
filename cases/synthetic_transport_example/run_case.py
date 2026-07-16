"""Regenerate the synthetic transport example under data/."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

CASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = CASE_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

from cases.synthetic_transport_example.suite import (  # noqa: E402
  run_synthetic_transport_example,
)


def main(argv: list[str] | None = None) -> int:
  parser = argparse.ArgumentParser(
    description=(
      "Generate privacy-safe Scenario-2-shaped cohorts and run the user-data "
      "pipeline."
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
    choices=("smoke", "midsize"),
    default="smoke",
    help="Size profile from case_config.json (default: smoke).",
  )
  parser.add_argument(
    "--run-label",
    default="synthetic_transport",
    help="Label embedded in the output directory name.",
  )
  parser.add_argument(
    "--seed",
    type=int,
    default=None,
    help="Optional seed override (default: case_config.json seed).",
  )
  args = parser.parse_args(argv)

  manifest = run_synthetic_transport_example(
    output_root=args.output_root,
    profile=args.profile,
    run_label=args.run_label,
    seed=args.seed,
  )
  print(f"Wrote synthetic transport example under: {manifest['case_root']}")
  print(f"Pipeline run directory: {manifest['run_dir']}")
  print(f"Profile: {manifest['profile']}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
