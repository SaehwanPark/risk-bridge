# Synthetic Transport Example

Privacy-safe second example that regenerates Scenario-2-shaped target, source, and
reference cohorts (shifted covariate support plus mild source miscalibration), then
runs the public user-data pipeline. No patient-level data is used or required.

Artifacts are written under gitignored `data/` and must not be committed.

## Profiles

| Profile | Intent |
| --- | --- |
| `smoke` | Fast local/CI check (`nsim=1`, small populations) |
| `midsize` | Manuscript-oriented refresh (`nsim=5`, reduced populations) |

Neither profile is a full publication Monte Carlo. For the full simulated Scenario 2
preset (`nsim=1000`), omit size overrides on the `risk-bridge` CLI (see the public
`REPRODUCTION.md` runbook and the root README).

## Regenerate

From the repository root:

```bash
uv run python cases/synthetic_transport_example/run_case.py --output-root data
```

Mid-size publication-oriented recipe:

```bash
uv run python cases/synthetic_transport_example/run_case.py \
  --output-root data \
  --profile midsize \
  --run-label synthetic_transport_midsize
```

## Outputs

Each run creates:

```text
data/synthetic_transport_example/<run_label>_<timestamp>/
  case_manifest.json
  environment.json
  cohorts/{target,source,reference}.csv
  runs/python_<run_label>_*/final/
    calibration_metrics.csv
    calibration_residuals.csv
    run_metadata.csv
    est_*.csv
    ...
```

Inspect `final/run_metadata.csv` for `schema_version=1.1.0` and
`final/calibration_metrics.csv` for the four-path CITL/slope/O/E/Brier summary.

## Fast pytest coverage

```bash
uv run pytest tests/test_synthetic_transport_example.py
```

The unit test runs the smoke profile in a temporary directory and does not require
writing artifacts under `data/`.
