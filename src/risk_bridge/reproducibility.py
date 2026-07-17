"""Capture machine-readable run environment metadata for regenerable artifacts."""

from __future__ import annotations

from datetime import datetime, timezone
from importlib import metadata
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any


REPRODUCIBILITY_CONTRACT_VERSION = "1.0.0"
THREAD_ENV_VARS = (
  "OMP_NUM_THREADS",
  "OPENBLAS_NUM_THREADS",
  "MKL_NUM_THREADS",
  "NUMEXPR_NUM_THREADS",
  "POLARS_MAX_THREADS",
)


def _reproducibility_contract() -> dict[str, Any]:
  """Return the acceptance contract recorded beside regenerable artifacts."""

  return {
    "version": REPRODUCIBILITY_CONTRACT_VERSION,
    "floating_point": {
      "same_environment": {"atol": 1e-8, "rtol": 1e-6},
      "cross_platform": {"atol": 1e-6, "rtol": 1e-4},
    },
    "structural_outputs": "exact",
    "gate_status": "exact",
    "stochastic_outputs": (
      "Compare seeded summaries with MCSE; do not require bitwise equality "
      "across platforms."
    ),
  }


def _package_version() -> str:
  """Resolve installed package version without importing ``risk_bridge`` root."""

  try:
    return metadata.version("risk-bridge")
  except metadata.PackageNotFoundError:
    init_text = Path(__file__).with_name("__init__.py").read_text(encoding="utf-8")
    for line in init_text.splitlines():
      if line.startswith("__version__"):
        return line.split("=", 1)[1].strip().strip("'\"")
    return "unknown"


def capture_run_environment(
  *,
  cwd: str | Path | None = None,
  extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
  """Return package/platform/git metadata for a run artifact directory.

  Parameters
  ----------
  cwd:
    Directory used for best-effort ``git rev-parse HEAD``. Defaults to the
    process working directory.
  extra:
    Optional caller-specific keys layered on top of the base environment.
  """

  git_cwd = Path.cwd() if cwd is None else Path(cwd)
  git_sha: str | None = None
  try:
    git_sha = (
      subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=git_cwd,
        stderr=subprocess.DEVNULL,
        text=True,
      ).strip()
      or None
    )
  except (subprocess.CalledProcessError, FileNotFoundError, OSError):
    git_sha = None

  env: dict[str, Any] = {
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "python_version": sys.version,
    "platform": platform.platform(),
    "machine": platform.machine(),
    "cpu_count": os.cpu_count(),
    "package_version": _package_version(),
    "git_sha": git_sha,
    "thread_environment": {
      name: os.environ.get(name) for name in THREAD_ENV_VARS
    },
    "reproducibility_contract": _reproducibility_contract(),
  }
  if extra:
    env.update(extra)
  return env


def write_environment_json(
  path: Path,
  *,
  cwd: str | Path | None = None,
  extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
  """Capture environment metadata and write it as indented JSON."""

  env = capture_run_environment(cwd=cwd, extra=extra)
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(json.dumps(env, indent=2, sort_keys=True) + "\n", encoding="utf-8")
  return env
