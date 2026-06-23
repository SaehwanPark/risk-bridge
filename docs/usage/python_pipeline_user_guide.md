# Risk Bridge User Guide

Risk Bridge runs constrained maximum likelihood estimation for source-to-target risk
bridging workflows.
It supports synthetic scenario runs for development and prepared user datasets for
applied analyses.

## Install

Use `uv` when possible:

```bash
uv add "risk-bridge @ git+https://github.com/<owner>/<repo>.git"
```

`pip` also works:

```bash
pip install "risk-bridge @ git+https://github.com/<owner>/<repo>.git"
```

For local development:

```bash
uv sync
uv run pytest
uv run basedpyright
```

## CLI

After installation, run:

```bash
uv run risk-bridge --help
```

Small simulated run:

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
  --run-label dev_smoke
```

User-data run:

```bash
uv run risk-bridge \
  --mode user-data \
  --target-csv /path/to/target.csv \
  --source-csv /path/to/source.csv \
  --reference-csv /path/to/reference.csv \
  --y-col label \
  --z-origin-col z_cont \
  --z-cat-col z_cat \
  --x-cols X1,X2,X3,X4 \
  --z-bins 0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9 \
  --sample-size 500 \
  --nsim 1 \
  --target-fpr 0.1 \
  --output-root data \
  --run-label hospital_a
```

## Library API

```python
from risk_bridge import UserDataRunConfig, UserDataSchema, run_user_data

run_dir = run_user_data(
    UserDataRunConfig(
        target_df=target_df,
        source_df=source_df,
        reference_df=reference_df,
        schema=UserDataSchema(
            x_cols=("X1", "X2", "X3", "X4"),
            y_col="label",
            z_origin_col="z_cont",
        ),
        sample_size=500,
        output_root="data",
        run_label="hospital_a",
    )
)
```

## User-Data Inputs

Non-simulation runs require three tabular files in CSV, Parquet, RData, or RDA
format:

- `target.csv`
- `source.csv`
- `reference.csv`

All three files use the same column mapping configured through `--y-col`,
`--x-cols`, `--z-origin-col`, `--z-cat-col`, and `--z-bins`.

### Shared Requirements

Each user-data CSV must include:

- one binary outcome column, mapped with `--y-col`
- all feature columns listed with `--x-cols`
- at least one of the Z columns: `--z-origin-col`, `--z-cat-col`, or both

Column names are matched literally, so the same schema must work across all
three files.

Use `--target-data`, `--source-data`, and `--reference-data` for format-neutral
paths. RData files may hold several named objects; when more than one data frame
is present, select one with the corresponding `--target-object`,
`--source-object`, or `--reference-object` option. Legacy `--*-csv` arguments
remain accepted.

If `zCat` is missing, Risk Bridge derives it from `zOrigin` and `--z-bins`.
If `zOrigin` is missing, pass `--allow-z-origin-from-zcat` to derive interval
midpoints from `zCat`.

`--z-bins` must be strictly increasing and lie in `(0, 1)` when `zCat` needs
to be derived from `zOrigin`.

### `target.csv`

`target.csv` is the evaluation cohort used to compute threshold metrics and
target prevalence.

It must satisfy the shared requirements above and contain at least
`sample_size` rows.

### `source.csv`

`source.csv` is the cohort from which the PSM and random source samples are
drawn.

It must satisfy the shared requirements above and contain at least
`sample_size` rows.

### `reference.csv`

`reference.csv` defines the discrete feature support used during calibration.

It must satisfy the shared requirements above. The reference file should cover
the feature combinations you expect the calibration step to enumerate.
Very high-cardinality feature spaces should be bucketed before running the
package.

### Optional and Conditional Columns

- If `zCat` is omitted, `zOrigin` must be present so Risk Bridge can derive the
  categorical bins.
- If `zOrigin` is omitted, `--allow-z-origin-from-zcat` must be set so Risk
  Bridge can derive interval midpoints from `zCat`.
- If both Z columns are present, Risk Bridge preserves both after
  preprocessing.

## Outputs

Each run writes an output directory with:

- `intermediate/`: sampled data and calibration artifacts by iteration
- `final/`: estimates, accuracy metrics, ROC metrics, threshold metadata,
  target prevalence, fit diagnostics, and run metadata

The most useful first files to inspect are:

- `final/run_metadata.csv`
- `final/fit_diagnostics.csv`
- `final/est_cml_psm.csv`
- `final/roc_metrics.csv`

## External-Calibration Bootstrap Runs

Use `--mode external-calibration` when only a labeled source cohort is observed
and target information is supplied as discrete X distributions, an interval
assignment for every Cartesian X combination, and external prevalence by
interval. Supply those fixed values in JSON with `feature_distributions`,
`x_interval_index`, and `p_external`; see
`examples/breast_cancer_external_calibration.json`.

Each iteration bootstraps the complete source cohort with replacement, generates
an X-only pseudo target, matches source row instances without replacement, draws
a random source sample, and fits both paths. `zOrigin` may be scaled at ingestion
with `--z-origin-scale`; when both Z columns are present, the scaled values must
reproduce `zCat` exactly under `--z-bins`.

Runs checkpoint estimates and diagnostics under `intermediate/`. Resume with
`--resume-run-dir`; a configuration fingerprint prevents incompatible resumes.
Patient-level sampled rows are omitted by default and are written only with
`--write-sample-artifacts`.

External-calibration final outputs include:

- `est_cml_psm.csv`, `est_cml_rs.csv`, `est_ml_psm.csv`, `est_ml_rs.csv`
- `fit_diagnostics.csv`
- `bootstrap_summary.csv`
- `run_metadata.csv`

For the README simulated quick start, see
[Quickstart walkthrough](quickstart_walkthrough.md) for process steps and a
column-by-column reference for every generated CSV.
