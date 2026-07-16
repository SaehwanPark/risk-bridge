# Reproduction Runbook

This runbook regenerates the public manuscript-facing artifacts that ship with Risk
Bridge: Scenario 2 calibration exports, the independent numerical validation suite, and
the synthetic transport second example. All commands write under gitignored `data/` and
must not be committed.

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
  est_*.csv
  fit_diagnostics.csv
```

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
uv run python cases/numerical_validation/run_suite.py \
  --output-root data \
  --run-label manuscript_validation
```

Artifacts land under `data/numerical_validation/<run_label>_<timestamp>/` with
`summary.json`, `environment.json`, and the per-check CSV files. See
[cases/numerical_validation/README.md](cases/numerical_validation/README.md).

## 3. Synthetic transport second example

Smoke profile (fast):

```bash
uv run python cases/synthetic_transport_example/run_case.py \
  --output-root data \
  --profile smoke
```

Mid-size profile (manuscript-oriented refresh, still not full Monte Carlo):

```bash
uv run python cases/synthetic_transport_example/run_case.py \
  --output-root data \
  --profile midsize \
  --run-label synthetic_transport_midsize
```

Inspect `case_manifest.json`, `environment.json`, cohort CSVs, and
`runs/python_*/final/calibration_metrics.csv`. See
[cases/synthetic_transport_example/README.md](cases/synthetic_transport_example/README.md).

## Citation and archival DOI

Package citation metadata lives in [CITATION.cff](CITATION.cff). An archival DOI is
minted **after** the public GitHub archive is deposited (for example via Zenodo). Until
that deposit completes, cite the versioned GitHub release / tag and leave the DOI field
empty in `CITATION.cff`.

## Release operator notes (development checkout)

Public export and package publication are driven from the development repository's
`deployment/` scripts (not shipped in this public tree):

1. `deployment/publish_public_repo.sh` — dry-run by default; `--apply` writes the local
   public checkout; `--push` updates the public remote only after explicit approval.
2. Build artifacts in the public checkout with `uv build`.
3. `deployment/publish_pypi.sh --repository testpypi` — dry-run validation; add
   `--execute` only with a publish token and explicit approval (then repeat for the
   primary index).

Do not claim public-release readiness until visibility, push, package publication, and
DOI minting are actually complete.
