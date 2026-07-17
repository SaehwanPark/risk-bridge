# Reproduction Runbook

This runbook regenerates the public manuscript-facing artifacts that ship with Risk
Bridge: Scenario 2 calibration exports, the independent numerical validation suite, the
external-calibration validation study, and the synthetic transport second example. All
commands write under gitignored `data/` and must not be committed.

The public-safe case runners are included in the wheel and sdist. After installing
`risk-bridge` from PyPI, run them as Python modules; a source checkout may use the same
module commands from its `uv` environment.

Breast-cancer or other patient-level replication cases are **not** part of this public
bundle.

Recorded package schema for these exports: `schema_version=1.1.0` in
`final/run_metadata.csv`.

## Prerequisites

```bash
uv sync --locked
uv run pytest
uv run basedpyright
```

## 1. Scenario 2 four-path calibration (local reduced sizes)

Suitable for local artifact prep. This is **not** a full publication Monte Carlo.

```bash
uv run risk-bridge \
  --mode simulated \
  --scenario 2 \
  --nsim 5 \
  --n-target 5000 \
  --n-source 2000 \
  --n-reference 5000 \
  --sample-size 500 \
  --output-root data \
  --run-label scenario2_calib
```

Inspect:

```text
data/python_scenario2_calib_*/final/
  calibration_metrics.csv
  calibration_residuals.csv
  roc_metrics.csv
  run_metadata.csv
  environment.json
  est_*.csv
  fit_diagnostics.csv
```

`environment.json` records package version, git SHA, Python/platform metadata,
thread settings, run identity fields, and a reproducibility contract. The
contract requires exact structural outputs and gate status, with numerical
tolerances of `rtol=1e-6, atol=1e-8` within the same environment and
`rtol=1e-4, atol=1e-6` across platforms. Seeded stochastic summaries are
compared using their reported MCSEs.

### Full Scenario 2 preset

Omit the size overrides to use the package defaults (`nsim=1000`,
`n_target=50000`, `n_source=10000`, `n_reference=50000`, `sample_size=1000`):

```bash
uv run risk-bridge \
  --mode simulated \
  --scenario 2 \
  --output-root data \
  --run-label scenario2_full
```

Do not treat the reduced-size command above as a substitute for this full preset when
preparing publication Monte Carlo tables.

## 2. Independent numerical validation

```bash
python -m cases.numerical_validation.run_suite \
  --output-root data \
  --run-label manuscript_validation
```

Artifacts land under `data/numerical_validation/<run_label>_<timestamp>/` with
`summary.json`, `environment.json`, and the per-check CSV files. See
[cases/numerical_validation/README.md](cases/numerical_validation/README.md).

## 3. External calibration validation

Smoke profile (fast gate check):

```bash
python -m cases.external_calibration_validation.run_suite \
  --output-root data \
  --profile smoke
```

Full profile (manuscript-oriented refresh):

```bash
python -m cases.external_calibration_validation.run_suite \
  --output-root data \
  --profile full \
  --condition matched \
  --run-label manuscript_external_calibration
```

The manuscript-oriented profile uses `nsim=50`. The prespecified synthetic
degradation negative control is expected to fail at least one gate:

```bash
python -m cases.external_calibration_validation.run_suite \
  --output-root data \
  --profile smoke \
  --run-label external_calibration_degraded \
  --condition degraded
```

It applies a fixed `+6.0` logit shift to every external prevalence while
leaving the known outcome truth unchanged; it is not a manuscript scientific
claim.

Artifacts land under
`data/external_calibration_validation/<run_label>_<timestamp>/` with
`summary.json`, `environment.json`, `fixed_summary.json`, and gate CSVs. If
`summary.json` reports `"passed": false`, demote external-calibration claims;
do not invent success. Use `--profile full` for manuscript citation; smoke is a
CI/local gate only. See
[cases/external_calibration_validation/README.md](cases/external_calibration_validation/README.md).

No empirical baseline is included because an inequivalent comparator would be
misleading. The evidence remains limited to the documented Scenario 2-shaped
synthetic setting and does not make full Scenario 1/3 claims.

## 4. Synthetic transport second example

Smoke profile (fast):

```bash
python -m cases.synthetic_transport_example.run_case \
  --output-root data \
  --profile smoke
```

Mid-size profile (manuscript-oriented refresh, still not full Monte Carlo):

```bash
python -m cases.synthetic_transport_example.run_case \
  --output-root data \
  --profile midsize \
  --run-label synthetic_transport_midsize
```

Inspect `case_manifest.json`, `environment.json`, cohort CSVs, and
`runs/python_*/final/calibration_metrics.csv`. See
[cases/synthetic_transport_example/README.md](cases/synthetic_transport_example/README.md).

## 5. Runtime and Cartesian-support scaling protocol

Smoke profile (fast):

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

Artifacts land under `data/runtime_support_scaling/<run_label>_<timestamp>/` with
`summary.json`, `environment.json`, `runtime_protocol.csv`, and
`support_scaling.csv`. Cite the regeneration commands and reported environment
metadata; do not invent timings. Historical MATLAB/R versus Python speedups are
not re-asserted by these tables. Treat `summary_pipeline_compute` and
`export_pipeline_with_io` as distinct workloads (not an I/O-only delta). See
[cases/runtime_support_scaling/README.md](cases/runtime_support_scaling/README.md).

## Citation and archival DOI

Package citation metadata lives in [CITATION.cff](CITATION.cff), including:

- Historical v1.0.2 archive DOI: `10.5281/zenodo.21418590`
- Concept DOI: `10.5281/zenodo.21401396`

After cutting a new public tag (for example `v1.0.4`), refresh the Zenodo version
asset so the version DOI resolves to that tag's README/`CITATION.cff`.

## Release operator notes (development checkout)

Public export and package publication are driven from the development repository's
`deployment/` scripts (not shipped in this public tree):

1. `deployment/publish_public_repo.sh` — dry-run by default; `--apply` writes the local
   public checkout; `--push` updates the public remote.
2. Build artifacts in the public checkout with `uv build`.
3. `deployment/publish_pypi.sh --repository testpypi` — dry-run validation; add
   `--execute` with a publish token (then repeat for the primary index).

The `v1.0.2` and `v1.0.3` public repositories, GitHub releases, and package-index
uploads are complete. The `1.0.4` packaging release preserves the immutable
`v1.0.2` archive DOI and adds no new scientific evidence; its PyPI upload and
Zenodo verification are operator steps after merge.
