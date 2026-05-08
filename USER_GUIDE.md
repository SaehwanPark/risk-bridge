# User Guide

Risk Bridge supports two primary workflows: simulated experiments and prepared user-data runs.

## Concepts

A run uses three cohorts:

- `target`: the population where performance is evaluated.
- `source`: the population sampled for model fitting.
- `reference`: the population used to estimate external calibration information.

The workflow estimates ordinary ML and constrained ML models, evaluates both against target data, and writes reproducible diagnostics.

## Simulated workflow

Use simulated mode when you want to reproduce built-in Scenario 1-3 experiments.

```bash
uv run risk-bridge \
  --mode simulated \
  --scenario 3 \
  --nsim 10 \
  --n-target 10000 \
  --n-source 5000 \
  --n-reference 10000 \
  --sample-size 1000 \
  --target-fpr 0.1 \
  --output-root data \
  --run-label scenario3
```

Important options:

- `--scenario`: built-in data-generating process, one of `1`, `2`, or `3`.
- `--nsim`: number of repeated iterations.
- `--sample-size`: source and target sample size used in each iteration.
- `--target-fpr`: target false-positive rate used for threshold selection.
- `--n-jobs`: process-level parallelism across iterations.
- `--path-jobs`: thread-level parallelism across sampling paths.

## User-data workflow

Use user-data mode when your target, source, and reference cohorts are already available as CSV files.

```bash
uv run risk-bridge \
  --mode user-data \
  --target-csv target.csv \
  --source-csv source.csv \
  --reference-csv reference.csv \
  --x-cols X1,X2,X3,X4 \
  --y-col caseY \
  --z-origin-col zOrigin \
  --z-cat-col zCat \
  --sample-size 500 \
  --nsim 1 \
  --output-root data \
  --run-label cohort_run
```

### Required columns

All input datasets must include:

- Binary outcome column, default `caseY`.
- Discrete feature columns passed through `--x-cols`.
- Continuous Z column, default `zOrigin`, or a categorical Z column that can be converted when `--allow-z-origin-from-zcat` is set.
- Categorical Z column, default `zCat`, or enough continuous Z information to derive it from `--z-bins`.

### Data expectations

- `caseY` must be binary.
- `zOrigin` values should lie in `(0, 1]`.
- `zCat` values should be integer categories in `[0, len(z_bins)]`.
- Feature columns should be discrete or already encoded as stable categories.
- `sample_size` must not exceed the number of rows in target or source cohorts.

## Output files

Each run writes:

```text
<output_root>/<timestamp>_<run_label>/
  intermediate/
  final/
```

Common final files:

- `run_metadata.csv`: configuration and run-level context.
- `fit_diagnostics.csv`: optimizer status, objective values, and violations.
- `est_cml_psm.csv`: cMLE parameter estimates from the PSM path.
- `est_ml_psm.csv`: ML parameter estimates from the PSM path.
- `roc_metrics.csv`: AUC-style metrics.
- `accuracy_metrics.csv`: thresholded TPR, PPV, and TNR metrics.
- `target_prevalence.csv`: target outcome prevalence by iteration.

## Recommended workflow

1. Start with a small `--nsim` and `--sample-size` to verify schema and runtime.
2. Inspect `fit_diagnostics.csv` for convergence and calibration violations.
3. Increase `--nsim` for stable summaries.
4. Keep the entire timestamped run directory with any downstream report so results remain reproducible.
