"""Controlled runtime and Cartesian-support scaling protocol for Risk Bridge."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import inspect
import json
import math
import os
from pathlib import Path
import resource
import sys
import tempfile
import time
import tracemalloc
from typing import Any, Iterator

import numpy as np
import polars as pl

from risk_bridge.api import (
  build_scenario2_run_config,
  run_scenario1,
  run_summary,
)
from risk_bridge.calibration import build_calibration_artifacts, enumerate_x_combinations
from risk_bridge.config import FeatureSpec, Scenario1PipelineOptions
from risk_bridge.reproducibility import capture_run_environment
from risk_bridge.runs import _feature_specs_from_data
from risk_bridge.simulate import generate_population
from risk_bridge.types import Population

CASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = CASE_DIR.parents[1]
PROTOCOL_VERSION = "1.0.0"
THREAD_ENV_VARS = (
  "OMP_NUM_THREADS",
  "OPENBLAS_NUM_THREADS",
  "MKL_NUM_THREADS",
  "NUMEXPR_NUM_THREADS",
  "POLARS_MAX_THREADS",
)
_FEATURE_SPEC_DEFAULTS = inspect.signature(_feature_specs_from_data).parameters
MAX_LEVELS_PER_FEATURE = int(_FEATURE_SPEC_DEFAULTS["max_levels_per_feature"].default)
MAX_TOTAL_COMBINATIONS = int(_FEATURE_SPEC_DEFAULTS["max_total_combinations"].default)
BLAS_PINNING_NOTE = (
  "Native BLAS/OpenMP/polars thread pools typically honor these variables only "
  "if they are set before numpy/polars import. Prefer exporting them in the "
  "shell before `uv run`, or launch via run_suite.py which setdefaults them "
  "before importing this module."
)


@dataclass(frozen=True)
class ProfileSpec:
  name: str
  n_target: int
  n_source: int
  n_reference: int
  sample_size: int
  nsim: int
  maxiter: int
  n_jobs: int
  path_jobs: int
  warmups: int
  timed_reps: int
  pin_blas_threads: bool
  support_points: tuple[tuple[int, ...], ...]
  include_cap_rejection: bool
  include_level_cap_rejection: bool


PROFILES: dict[str, ProfileSpec] = {
  "smoke": ProfileSpec(
    name="smoke",
    n_target=120,
    n_source=100,
    n_reference=120,
    sample_size=40,
    nsim=1,
    maxiter=25,
    n_jobs=1,
    path_jobs=1,
    warmups=0,
    timed_reps=1,
    pin_blas_threads=True,
    support_points=((2, 2), (4, 3, 3, 3)),
    include_cap_rejection=True,
    include_level_cap_rejection=True,
  ),
  "protocol": ProfileSpec(
    name="protocol",
    n_target=500,
    n_source=300,
    n_reference=500,
    sample_size=100,
    nsim=2,
    maxiter=50,
    n_jobs=1,
    path_jobs=1,
    warmups=1,
    timed_reps=3,
    pin_blas_threads=True,
    support_points=(
      (2, 2),
      (3, 3),
      (3, 3, 3),
      (4, 3, 3, 3),
      (10, 10, 10),
      (10, 10, 10, 10),
    ),
    include_cap_rejection=True,
    include_level_cap_rejection=True,
  ),
}


def resolve_profile(name: str) -> ProfileSpec:
  if name not in PROFILES:
    known = ", ".join(sorted(PROFILES))
    raise ValueError(f"unknown profile {name!r}; expected one of: {known}")
  return PROFILES[name]


def _custom_feature_specs(level_counts: tuple[int, ...]) -> tuple[FeatureSpec, ...]:
  specs: list[FeatureSpec] = []
  for i, n_levels in enumerate(level_counts, start=1):
    if n_levels < 1:
      raise ValueError("each feature must have at least one level")
    specs.append(
      FeatureSpec(
        name=f"X{i}",
        kind="custom",
        params={"values": tuple(range(n_levels))},
      )
    )
  return tuple(specs)


def _n_combinations(level_counts: tuple[int, ...]) -> int:
  total = 1
  for n in level_counts:
    total *= int(n)
  return total


def _synthetic_reference(
  feature_specs: tuple[FeatureSpec, ...], *, n: int, seed: int
) -> Population:
  rng = np.random.default_rng(seed)
  cols: dict[str, list[int]] = {}
  for spec in feature_specs:
    values = [int(v) for v in spec.params["values"]]
    cols[spec.name] = [int(rng.choice(values)) for _ in range(n)]
  y = rng.integers(0, 2, size=n).astype(np.int64)
  if np.all(y == y[0]):
    y[0] = 1 - int(y[0])
  return Population(
    X=pl.DataFrame(cols),
    z_cont=rng.uniform(0.05, 0.95, size=n).astype(np.float64),
    z_cat=rng.integers(0, 5, size=n).astype(np.int64),
    y=y,
  )


def _reference_df_for_cap(
  *,
  n_features: int,
  levels_per_feature: int,
  n_rows: int,
  seed: int,
) -> pl.DataFrame:
  """Build a reference frame that observes every level at least once."""

  rng = np.random.default_rng(seed)
  if n_rows < levels_per_feature:
    raise ValueError("n_rows must be >= levels_per_feature to observe every level")
  data: dict[str, list[int]] = {}
  for i in range(1, n_features + 1):
    col = list(range(levels_per_feature))
    while len(col) < n_rows:
      col.append(int(rng.integers(0, levels_per_feature)))
    data[f"X{i}"] = col[:n_rows]
  return pl.DataFrame(data)


def _peak_rss_mb() -> float:
  usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
  # Linux reports kilobytes; macOS reports bytes.
  if sys.platform == "darwin":
    return float(usage) / (1024.0 * 1024.0)
  return float(usage) / 1024.0


@contextmanager
def _pinned_thread_env(*, enabled: bool) -> Iterator[dict[str, Any]]:
  """Request BLAS/OpenMP/polars thread pins when unset; restore afterward."""

  previous = {name: os.environ.get(name) for name in THREAD_ENV_VARS}
  pinned: dict[str, str] = {}
  if enabled:
    for name in THREAD_ENV_VARS:
      if previous[name] is None:
        os.environ[name] = "1"
        pinned[name] = "1"
  try:
    yield {
      "pin_blas_threads_requested": enabled,
      "thread_env_before": previous,
      "thread_env_pinned_by_runner": pinned,
      "thread_env_after_runner": {
        name: os.environ.get(name) for name in THREAD_ENV_VARS
      },
      "blas_pinning_note": BLAS_PINNING_NOTE,
    }
  finally:
    for name, value in previous.items():
      if value is None:
        os.environ.pop(name, None)
      else:
        os.environ[name] = value


def capture_environment(
  *,
  profile: ProfileSpec,
  thread_meta: dict[str, Any],
) -> dict[str, Any]:
  if profile.n_jobs > 1:
    import multiprocessing as mp

    process_start_method: str | None = (
      mp.get_start_method(allow_none=True) or "spawn"
    )
  else:
    process_start_method = None
  return capture_run_environment(
    cwd=REPO_ROOT,
    extra={
      "protocol_version": PROTOCOL_VERSION,
      "profile": profile.name,
      "n_jobs": profile.n_jobs,
      "path_jobs": profile.path_jobs,
      "warmups": profile.warmups,
      "timed_reps": profile.timed_reps,
      "max_levels_per_feature": MAX_LEVELS_PER_FEATURE,
      "max_total_combinations": MAX_TOTAL_COMBINATIONS,
      "process_start_method": process_start_method,
      **thread_meta,
    },
  )


def _timed_call(fn: Any) -> tuple[float, float, Any]:
  start_wall = time.perf_counter()
  start_cpu = time.process_time()
  result = fn()
  elapsed_s = time.perf_counter() - start_wall
  process_cpu_s = time.process_time() - start_cpu
  return elapsed_s, process_cpu_s, result


def _runtime_config(profile: ProfileSpec, *, output_root: str) -> Any:
  return build_scenario2_run_config(
    seed=475,
    nsim=profile.nsim,
    n_target=profile.n_target,
    n_source=profile.n_source,
    n_reference=profile.n_reference,
    sample_size=profile.sample_size,
    output_root=output_root,
    maxiter=profile.maxiter,
  )


def _runtime_row(
  *,
  phase: str,
  include_io: bool,
  rep: int,
  elapsed_s: float,
  process_cpu_s: float,
  profile: ProfileSpec,
  status: str,
) -> dict[str, object]:
  return {
    "phase": phase,
    "include_io": include_io,
    "rep": rep,
    "elapsed_s": elapsed_s,
    "process_cpu_s": process_cpu_s,
    "n_jobs": profile.n_jobs,
    "path_jobs": profile.path_jobs,
    "nsim": profile.nsim,
    "n_target": profile.n_target,
    "n_source": profile.n_source,
    "n_reference": profile.n_reference,
    "sample_size": profile.sample_size,
    "maxiter": profile.maxiter,
    "seed": 475,
    "status": status,
  }


def _measure_runtime_phases(
  profile: ProfileSpec, *, rep: int, io_root: Path
) -> list[dict[str, object]]:
  rows: list[dict[str, object]] = []
  cfg_compute = _runtime_config(profile, output_root=str(io_root / "compute_unused"))
  feature_specs = cfg_compute.sim.feature_specs

  elapsed, cpu, _ = _timed_call(lambda: enumerate_x_combinations(feature_specs))
  rows.append(
    _runtime_row(
      phase="enumerate_support",
      include_io=False,
      rep=rep,
      elapsed_s=elapsed,
      process_cpu_s=cpu,
      profile=profile,
      status="ok",
    )
  )

  rng = np.random.default_rng(profile.n_reference + rep)
  z_bins = np.asarray(cfg_compute.sim.z_spec.bins, dtype=np.float64)
  gamma = np.asarray(cfg_compute.sim.z_spec.gamma_init, dtype=np.float64)
  beta = np.asarray(cfg_compute.sim.beta, dtype=np.float64)
  reference = generate_population(
    rng,
    profile.n_reference,
    feature_specs,
    gamma=gamma,
    z_bins=z_bins,
    alpha=float(cfg_compute.sim.alpha),
    beta=beta,
  )

  elapsed, cpu, _ = _timed_call(
    lambda: build_calibration_artifacts(reference, feature_specs)
  )
  rows.append(
    _runtime_row(
      phase="build_calibration",
      include_io=False,
      rep=rep,
      elapsed_s=elapsed,
      process_cpu_s=cpu,
      profile=profile,
      status="ok",
    )
  )

  # Compact single-path in-memory summary pipeline (not the four-path export).
  elapsed, cpu, summary = _timed_call(lambda: run_summary(cfg_compute))
  status = "ok" if len(summary) == profile.nsim else "error"
  rows.append(
    _runtime_row(
      phase="summary_pipeline_compute",
      include_io=False,
      rep=rep,
      elapsed_s=elapsed,
      process_cpu_s=cpu,
      profile=profile,
      status=status,
    )
  )

  # Full Scenario-2-shaped two-path export pipeline including CSV writes.
  cfg_io = _runtime_config(profile, output_root=str(io_root / f"rep_{rep}"))
  options = Scenario1PipelineOptions(
    n_jobs=profile.n_jobs,
    path_jobs=profile.path_jobs,
    print_every=0,
    run_label=f"runtime_io_rep{rep}",
  )

  def _run_io() -> Path:
    return run_scenario1(cfg_io, options)

  elapsed, cpu, out_path = _timed_call(_run_io)
  io_status = "ok" if out_path.exists() else "error"
  rows.append(
    _runtime_row(
      phase="export_pipeline_with_io",
      include_io=True,
      rep=rep,
      elapsed_s=elapsed,
      process_cpu_s=cpu,
      profile=profile,
      status=io_status,
    )
  )
  return rows


def run_runtime_protocol(profile: ProfileSpec) -> list[dict[str, object]]:
  rows: list[dict[str, object]] = []
  with tempfile.TemporaryDirectory(prefix="rb_runtime_") as tmp:
    io_root = Path(tmp)
    total_passes = profile.warmups + profile.timed_reps
    for pass_idx in range(total_passes):
      measured = _measure_runtime_phases(profile, rep=pass_idx, io_root=io_root)
      if pass_idx < profile.warmups:
        continue
      timed_rep = pass_idx - profile.warmups + 1
      for row in measured:
        row = dict(row)
        row["rep"] = timed_rep
        rows.append(row)
  return rows


def _scaling_row(
  *,
  level_counts: tuple[int, ...],
  n_combinations: int,
  elapsed_enumerate_s: float | None,
  elapsed_calibration_s: float | None,
  peak_traced_mb: float | None,
  peak_rss_mb: float | None,
  status: str,
  cap_triggered: str | None,
) -> dict[str, object]:
  return {
    "n_features": len(level_counts),
    "levels_per_feature": ",".join(str(v) for v in level_counts),
    "n_combinations": n_combinations,
    "elapsed_enumerate_s": elapsed_enumerate_s,
    "elapsed_calibration_s": elapsed_calibration_s,
    "peak_traced_mb": peak_traced_mb,
    "peak_rss_mb": peak_rss_mb,
    "status": status,
    "cap_triggered": cap_triggered,
  }


def _run_ok_scaling_point(
  level_counts: tuple[int, ...], *, seed: int
) -> dict[str, object]:
  n_combos = _n_combinations(level_counts)
  specs = _custom_feature_specs(level_counts)
  if max(level_counts) > MAX_LEVELS_PER_FEATURE:
    return _scaling_row(
      level_counts=level_counts,
      n_combinations=n_combos,
      elapsed_enumerate_s=None,
      elapsed_calibration_s=None,
      peak_traced_mb=None,
      peak_rss_mb=None,
      status="rejected_by_cap",
      cap_triggered="max_levels_per_feature",
    )
  if n_combos > MAX_TOTAL_COMBINATIONS:
    return _scaling_row(
      level_counts=level_counts,
      n_combinations=n_combos,
      elapsed_enumerate_s=None,
      elapsed_calibration_s=None,
      peak_traced_mb=None,
      peak_rss_mb=None,
      status="rejected_by_cap",
      cap_triggered="max_total_combinations",
    )

  rss_before = _peak_rss_mb()
  tracemalloc.start()
  try:
    elapsed_enum, _, combos = _timed_call(lambda: enumerate_x_combinations(specs))
    ref = _synthetic_reference(specs, n=min(400, max(40, n_combos)), seed=seed)

    def _calib() -> None:
      build_calibration_artifacts(ref, specs)

    elapsed_calib, _, _ = _timed_call(_calib)
    _, peak = tracemalloc.get_traced_memory()
    peak_mb = peak / (1024.0 * 1024.0)
    peak_rss = max(_peak_rss_mb(), rss_before)
  except Exception as exc:  # noqa: BLE001 - record protocol failure status
    return _scaling_row(
      level_counts=level_counts,
      n_combinations=n_combos,
      elapsed_enumerate_s=None,
      elapsed_calibration_s=None,
      peak_traced_mb=None,
      peak_rss_mb=None,
      status="error",
      cap_triggered=str(exc.__class__.__name__),
    )
  finally:
    tracemalloc.stop()

  if len(combos) != n_combos:
    return _scaling_row(
      level_counts=level_counts,
      n_combinations=n_combos,
      elapsed_enumerate_s=elapsed_enum,
      elapsed_calibration_s=elapsed_calib,
      peak_traced_mb=peak_mb,
      peak_rss_mb=peak_rss,
      status="error",
      cap_triggered="combo_count_mismatch",
    )
  return _scaling_row(
    level_counts=level_counts,
    n_combinations=n_combos,
    elapsed_enumerate_s=elapsed_enum,
    elapsed_calibration_s=elapsed_calib,
    peak_traced_mb=peak_mb,
    peak_rss_mb=peak_rss,
    status="ok",
    cap_triggered=None,
  )


def _run_cap_rejection_rows(profile: ProfileSpec) -> list[dict[str, object]]:
  """Exercise package cardinality guards with known-over-limit inputs."""

  rows: list[dict[str, object]] = []
  if profile.include_cap_rejection:
    # 10^5 > package Cartesian cap; status set from known intent, not message text.
    level_counts = (10, 10, 10, 10, 10)
    n_combos = _n_combinations(level_counts)
    ref = _reference_df_for_cap(
      n_features=len(level_counts),
      levels_per_feature=10,
      n_rows=200,
      seed=901,
    )
    x_cols = [f"X{i}" for i in range(1, len(level_counts) + 1)]
    status = "error"
    cap_triggered: str | None = "expected_rejection_missing"
    try:
      _feature_specs_from_data(ref, x_cols)
    except ValueError:
      status = "rejected_by_cap"
      cap_triggered = "max_total_combinations"
    rows.append(
      _scaling_row(
        level_counts=level_counts,
        n_combinations=n_combos,
        elapsed_enumerate_s=None,
        elapsed_calibration_s=None,
        peak_traced_mb=None,
        peak_rss_mb=None,
        status=status,
        cap_triggered=cap_triggered,
      )
    )

  if profile.include_level_cap_rejection:
    level_counts = (21,)
    n_combos = 21
    ref = _reference_df_for_cap(
      n_features=1, levels_per_feature=21, n_rows=80, seed=902
    )
    status = "error"
    cap_triggered = "expected_rejection_missing"
    try:
      _feature_specs_from_data(ref, ["X1"])
    except ValueError:
      status = "rejected_by_cap"
      cap_triggered = "max_levels_per_feature"
    rows.append(
      _scaling_row(
        level_counts=level_counts,
        n_combinations=n_combos,
        elapsed_enumerate_s=None,
        elapsed_calibration_s=None,
        peak_traced_mb=None,
        peak_rss_mb=None,
        status=status,
        cap_triggered=cap_triggered,
      )
    )
  return rows


def run_support_scaling(profile: ProfileSpec) -> list[dict[str, object]]:
  rows: list[dict[str, object]] = []
  for idx, level_counts in enumerate(profile.support_points):
    rows.append(_run_ok_scaling_point(level_counts, seed=700 + idx))
  rows.extend(_run_cap_rejection_rows(profile))
  return rows


def run_suite(*, profile: str = "smoke") -> dict[str, Any]:
  spec = resolve_profile(profile)
  with _pinned_thread_env(enabled=spec.pin_blas_threads) as thread_meta:
    runtime_rows = run_runtime_protocol(spec)
    scaling_rows = run_support_scaling(spec)
    environment = capture_environment(profile=spec, thread_meta=thread_meta)

  runtime_ok = all(str(row["status"]) == "ok" for row in runtime_rows)
  scaling_ok = all(
    str(row["status"]) in {"ok", "rejected_by_cap"} for row in scaling_rows
  )
  has_ok_scaling = any(str(row["status"]) == "ok" for row in scaling_rows)
  has_rejected = any(str(row["status"]) == "rejected_by_cap" for row in scaling_rows)
  return {
    "passed": runtime_ok and scaling_ok and has_ok_scaling and has_rejected,
    "profile": spec.name,
    "protocol_version": PROTOCOL_VERSION,
    "runtime_rows": runtime_rows,
    "scaling_rows": scaling_rows,
    "environment": environment,
    "checks": [
      {
        "name": "runtime_protocol",
        "passed": runtime_ok,
        "detail": (
          "Timed Scenario-2-sized summary compute and full export phases "
          "(distinct workloads; do not difference as I/O-only)"
        ),
      },
      {
        "name": "support_scaling",
        "passed": scaling_ok and has_ok_scaling and has_rejected,
        "detail": "Cartesian support timing plus package cardinality-cap rejections",
      },
    ],
  }


def write_suite_artifacts(suite: dict[str, Any], output_dir: Path) -> Path:
  output_dir.mkdir(parents=True, exist_ok=True)
  (output_dir / "summary.json").write_text(
    json.dumps(
      {
        "passed": suite["passed"],
        "profile": suite["profile"],
        "protocol_version": suite["protocol_version"],
        "checks": suite["checks"],
      },
      indent=2,
    )
    + "\n",
    encoding="utf-8",
  )
  (output_dir / "environment.json").write_text(
    json.dumps(suite["environment"], indent=2) + "\n", encoding="utf-8"
  )
  pl.DataFrame(suite["runtime_rows"]).write_csv(output_dir / "runtime_protocol.csv")
  pl.DataFrame(suite["scaling_rows"]).write_csv(output_dir / "support_scaling.csv")
  return output_dir


def assert_finite_timings(rows: list[dict[str, object]], *, key: str) -> None:
  for row in rows:
    value = row.get(key)
    if value is None:
      continue
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
      raise AssertionError(f"non-finite timing for {key}: {value!r}")
