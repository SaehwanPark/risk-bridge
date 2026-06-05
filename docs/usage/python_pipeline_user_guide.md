# Risk Bridge User Guide

Risk Bridge runs constrained maximum likelihood estimation for source-to-target risk
bridging workflows.
It supports synthetic scenario runs for reproducible experiments and prepared user datasets for
applied analyses.

## Install

Use `uv` when possible:

```bash
uv add "risk-bridge @ git+https://github.com/SaehwanPark/risk-bridge.git"
```

`pip` also works:

```bash
pip install "risk-bridge @ git+https://github.com/SaehwanPark/risk-bridge.git"
```

For a local checkout:

```bash
uv sync
uv run pytest
uv run basedpyright
```

The orchestration layer uses [`comp-builders`](https://pypi.org/project/comp-builders/) for explicit `Result` composition in recoverable validation paths.

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
  --run-label quickstart
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

## Input Assumptions

Each user-data CSV must include:

- a binary outcome column, mapped with `--y-col`
- integer-coded discrete feature columns, listed with `--x-cols`
- `zOrigin`, `zCat`, or both

When `zCat` is missing, Risk Bridge derives it from `zOrigin` and `--z-bins`.
When `zOrigin` is missing, pass `--allow-z-origin-from-zcat` to derive interval
midpoints from `zCat`.

Reference data defines the discrete feature support used by calibration.
Very high-cardinality features should be bucketed before running the package.

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
