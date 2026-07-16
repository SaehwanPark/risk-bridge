# Pipeline Architecture

Risk Bridge is organized as a small functional core wrapped by run orchestration.

## Core Modules

- `risk_bridge.simulate`: synthetic target/source/reference population generation
- `risk_bridge.preprocess`: user-data schema mapping and validation
- `risk_bridge.sampling`: propensity score matching and random sampling
- `risk_bridge.calibration`: discrete support enumeration and calibration artifacts
- `risk_bridge.likelihood`: likelihood and analytic gradient calculations
- `risk_bridge.constraints`: calibration constraints and Jacobians
- `risk_bridge.optimize`: constrained solver ladder
- `risk_bridge.metrics`: ROC, threshold, and standalone calibration metrics for
  labeled predictions

## Orchestration

- `risk_bridge.runs` owns simulated and user-data run loops.
- `risk_bridge.cli` exposes the `risk-bridge` command.
- `risk_bridge.api` exposes typed library entrypoints.

The run loop is intentionally shaped as:

```text
impure setup -> typed orchestration state -> pure row builders -> impure writes
```

The setup edge creates directories, initializes random generators, and prepares
process/thread work.
The middle of the pipeline passes explicit frozen state bundles for fitted PSM/RS
paths and evaluation inputs.
Shared pure helpers then build parameter, accuracy, ROC, and fit-diagnostic rows
for both simulated and user-data modes.
The final edge writes CSV/parquet outputs and progress messages.

Recoverable orchestration validation uses `comp-builders` `Result` values where
fail-fast composition is clearer than nested conditionals.
The package is installed from
<https://pypi.org/project/comp-builders/>.
Numerical likelihood, calibration, constraint, and solver functions stay as
ordinary typed Python functions so the statistical core remains easy to test and
port.

Both simulated and user-data modes follow the same estimation flow:

1. Prepare target, source, and reference data.
2. Build calibration artifacts from reference data.
3. Create PSM and random-sampling analysis samples.
4. Fit unconstrained ML.
5. Fit constrained cMLE with the solver ladder.
6. Evaluate threshold metrics, ROC/AUC (secondary), and primary calibration
   metrics (CITL, slope, O/E, Brier) plus post-fit moment residuals.
7. Write intermediate and final CSV outputs with `schema_version` in run
   metadata.

The CSV output contract version is `1.1.0` (`risk_bridge.output_schema`).
Optimizer-bound moment residuals remain computed in `risk_bridge.constraints`
and are also exported post-fit for manuscript-facing diagnostics.

## Public Configuration

Prefer typed config objects over long keyword-only entrypoints:

- `RunConfig`, `SimulationConfig`, and `Scenario1PipelineOptions` for simulated runs
- `UserDataSchema` and `UserDataRunConfig` for user-data runs
- `OptimizationConfig`, `FeatureSpec`, and `ZModelSpec` for lower-level configuration

## Data Contract

User data is canonicalized to:

- `caseY`: binary outcome
- user-selected integer-coded feature columns
- `zOrigin`: continuous source risk variable
- `zCat`: categorical risk bin

The model currently assumes discrete feature support can be enumerated.
If reference-data support is too large, users should recode or bucket features before fitting.
