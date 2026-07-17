# Runtime and Support-Scaling Protocol

This case regenerates controlled runtime and Cartesian-support scaling artifacts
for manuscript-facing performance interpretation (reviewer items M11/M14).

Artifacts are written under gitignored `data/` and must not be committed.

These tables support **controlled interpretation only**. They do not re-assert
historical MATLAB/R versus Python speedup ratios. Report hardware, threading,
I/O inclusion, warm-up, and repetition counts alongside any cited timings.

## Regenerate

Smoke profile (fast local / CI-oriented sanity):

```bash
python -m cases.runtime_support_scaling.run_suite \
  --output-root data \
  --profile smoke
```

Protocol profile (manuscript-oriented refresh):

```bash
python -m cases.runtime_support_scaling.run_suite \
  --output-root data \
  --profile protocol \
  --run-label manuscript_runtime_scaling
```

For the most reliable native-thread control, export thread caps in the shell
before launch (the CLI also `setdefault`s them before importing numpy-backed
modules):

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 POLARS_MAX_THREADS=1 \
python -m cases.runtime_support_scaling.run_suite \
  --output-root data \
  --profile protocol
```

## Outputs

Each run creates:

```text
data/runtime_support_scaling/<run_label>_<timestamp>/
  summary.json
  environment.json
  runtime_protocol.csv
  support_scaling.csv
```

| Artifact | Contents |
| --- | --- |
| `summary.json` | Overall pass/fail plus per-check status |
| `environment.json` | Platform, CPU count, package/git versions, `n_jobs`/`path_jobs`, requested thread-env pins, warm-up/rep counts, protocol version |
| `runtime_protocol.csv` | Timed phases with elapsed and process CPU time, I/O inclusion, repetitions, Scenario-2-sized populations |
| `support_scaling.csv` | Cartesian cardinality grid with enumerate/calibration timings, memory columns, and package cap rejections |

### Runtime phases

| `phase` | `include_io` | Meaning |
| --- | --- | --- |
| `enumerate_support` | `false` | In-memory Cartesian support enumeration |
| `build_calibration` | `false` | In-memory calibration artifact construction only (reference population generated outside the timer) |
| `summary_pipeline_compute` | `false` | Compact single-path in-memory `run_summary` pipeline (no CSV export; not the four-path exporter) |
| `export_pipeline_with_io` | `true` | Full Scenario-2-shaped two-path simulated export (`run_scenario1`) including CSV writes |

`summary_pipeline_compute` and `export_pipeline_with_io` are **distinct workloads**.
Do not interpret their timing difference as I/O overhead alone; the export path
runs PSM/RS sampling and four fits per iteration, not only file writes.

Smoke uses 0 warm-ups and 1 timed repetition. Protocol uses 1 discarded warm-up
and 3 timed repetitions. `environment.json` records requested thread-env pins
and notes that native pools typically honor them only when set before
numpy/polars import.

### Support-scaling grid

Protocol cardinalities include Scenario-2-like `108 = 4×3×3×3` plus smaller and
larger products up to `10,000`. Dedicated `rejected_by_cap` rows exercise the
package guards in `_feature_specs_from_data` (`20` levels per feature;
`50,000` total combinations).

Memory columns:

- `peak_traced_mb`: CPython allocator peak via `tracemalloc` (excludes most
  NumPy/Arrow native buffers).
- `peak_rss_mb`: process max RSS from `resource.getrusage` (platform units
  normalized to MiB; more appropriate for rough footprint interpretation).

## Fast pytest coverage

```bash
uv run pytest tests/test_runtime_support_scaling_unit.py
```

The unit test runs the smoke profile in-process and does not require writing
artifacts under `data/`.
