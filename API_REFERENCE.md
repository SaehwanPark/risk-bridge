# API Reference

Risk Bridge exposes a small public API from the `risk_bridge` package.

## Configuration dataclasses

### `FeatureSpec`

Describes a discrete X feature for simulation or support enumeration.

```python
FeatureSpec(name: str, kind: str, params: dict[str, Any])
```

Supported `kind` values are `categorical_cut`, `capped_poisson`, and `custom`.

### `ZModelSpec`

Describes the Z model and categorical cut points.

```python
ZModelSpec(family: str, gamma_init: tuple[float, ...], bins: tuple[float, ...])
```

The current supported family is `trunc_lognormal`.

### `SimulationConfig`

Contains synthetic data-generation settings for simulated runs.

### `OptimizationConfig`

Contains optimizer methods, tolerance, and maximum iterations.

### `RunConfig`

Top-level simulated-run configuration.

### `Scenario1PipelineOptions`

Optional runtime settings for simulated pipeline execution, including calibration tolerance, initial theta, parallelism, parquet writing, and run labeling.

### `UserDataSchema`

Maps user-provided DataFrame columns to Risk Bridge concepts.

```python
UserDataSchema(
  x_cols=("X1", "X2"),
  y_col="caseY",
  z_origin_col="zOrigin",
  z_cat_col="zCat",
)
```

### `UserDataRunConfig`

Top-level configuration for end-to-end user-data runs.

## Run builders

### `build_scenario1_run_config(...) -> RunConfig`
### `build_scenario2_run_config(...) -> RunConfig`
### `build_scenario3_run_config(...) -> RunConfig`

Return typed configurations for the built-in simulated scenarios. Keyword arguments let you override seed, population sizes, sample size, target prevalence, target FPR, model coefficients, output root, and optimizer iteration limit.

## Execution functions

### `run_simulation(cfg, options=None) -> pathlib.Path`

Runs a simulated Scenario 1-3 pipeline and returns the output directory.

### `run_scenario1(cfg, options=None) -> pathlib.Path`

Compatibility alias for simulated pipeline execution.

### `run_user_data(config) -> pathlib.Path`

Runs the user-data pipeline and returns the output directory.

### `run_summary(cfg) -> polars.DataFrame`

Runs the compact development pipeline and returns one summary row per iteration.

### `run_single_iteration_result(...)`

Executes a lower-level single-iteration pipeline helper. Most users should prefer `run_simulation` or `run_user_data`.

## Result dataclasses

### `FitResult`

Contains fitted parameters, optimizer success flag, status, iteration count, objective value, and diagnostics.

### `EvaluationSummary`

Contains AUC, threshold, TPR, PPV, and TNR.

### `IterationMetrics`

Groups ML and cMLE evaluation summaries.

### `IterationResult`

Groups ML/cMLE fits, metrics, and solver history for one iteration.
