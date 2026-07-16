# Risk Bridge Architecture

Risk Bridge is a Python package for source-to-target risk bridging with constrained
maximum likelihood estimation.
The codebase keeps a small statistical core behind typed configuration, CLI, and
library wrappers.

## Entry Points

- `risk_bridge.cli` and the `risk-bridge` console script run simulated and
  user-data workflows from command-line options.
- `risk_bridge.api` exposes library-style wrappers around typed config objects.
- `risk_bridge.pipeline` provides compact development iterations and summary
  outputs.

## Module Responsibilities

- `risk_bridge.config`: frozen typed configuration objects and validation.
- `risk_bridge.simulate`: synthetic target, source, and reference populations.
- `risk_bridge.preprocess`: user-data schema mapping and canonical columns.
- `risk_bridge.sampling`: propensity-score matching and random sampling.
- `risk_bridge.calibration`: discrete support and calibration artifacts.
- `risk_bridge.likelihood`: likelihood and analytic objective gradients.
- `risk_bridge.constraints`: calibration inequalities and analytic Jacobians.
- `risk_bridge.optimize`: unconstrained ML and constrained cMLE solver ladder.
- `risk_bridge.metrics`: threshold, accuracy, ROC/AUC, and standalone
  calibration metrics for labeled predictions.
- `risk_bridge.output_schema`: versioned CSV output-contract constant.
- `risk_bridge.runs`: simulated and user-data orchestration loops.
- `risk_bridge.tabular`: CSV/parquet-oriented output helpers.
- `cases/`: privacy-safe replication harnesses (`numerical_validation` and
  `synthetic_transport_example`) that configure package entrypoints without
  owning estimator logic.

## Data Flow

Both simulated and user-data modes follow the same estimation shape:

```text
typed config
  -> simulated data or canonicalized user data
  -> PSM and random-sampling analysis samples
  -> calibration artifacts from reference data
  -> unconstrained ML and constrained cMLE
  -> threshold, ROC, calibration-metric, residual, and fit diagnostic rows
  -> intermediate and final tabular outputs (schema_version in run metadata)
```

Run orchestration keeps side effects at the edges:

```text
impure setup -> typed orchestration state -> pure row builders -> impure writes
```

Directory creation, random generator setup, parallel execution, and file writes
live at the orchestration edge.
Likelihood, constraints, calibration, metrics, and row construction remain typed
functions that are easier to test in isolation.
Recoverable orchestration validation uses `comp-builders` `Result` composition
where fail-fast control flow is clearer than nested conditionals.

## Contributor Guardrails

- Prefer typed config objects in `risk_bridge.config` and wrappers in
  `risk_bridge.api` over long public keyword argument lists.
- Keep likelihood, calibration, constraint, and solver changes covered by focused
  numerical tests.
- Preserve the canonical user-data columns: `caseY`, selected integer-coded
  feature columns, `zOrigin`, and `zCat`.
- Keep architecture notes, user documentation, and `CHANGELOG.md` aligned with
  meaningful feature or behavior changes.

Detailed references:

- [Pipeline architecture](docs/architecture/python_pipeline_analysis.md)
- [Solver strategy](docs/architecture/python_solver_strategy.md)
- [User guide](docs/usage/python_pipeline_user_guide.md)
