# Independent Numerical Validation

This case regenerates independent numerical validation artifacts for the Risk
Bridge estimation core: randomized derivative checks, a reference-optimizer
comparison, outcome-coefficient recovery under a matching joint DGP, and
invariance checks.

Artifacts are written under gitignored `data/` and must not be committed.

## Regenerate

From the repository root:

```bash
uv run python cases/numerical_validation/run_suite.py --output-root data
```

Optional label:

```bash
uv run python cases/numerical_validation/run_suite.py \
  --output-root data \
  --run-label manuscript_validation
```

## Outputs

Each run creates:

```text
data/numerical_validation/<run_label>_<timestamp>/
  summary.json
  environment.json
  derivative_checks.csv
  optimizer_comparison.csv
  recovery.csv
  invariance.csv
```

| Artifact | Contents |
| --- | --- |
| `summary.json` | Overall pass/fail plus per-check metrics |
| `environment.json` | Python/platform/package version and optional git SHA |
| `derivative_checks.csv` | Multi-seed NLL gradient and constraint Jacobian FD errors |
| `optimizer_comparison.csv` | Analytic vs finite-difference Jacobian constrained solves |
| `recovery.csv` | Outcome-model coefficient recovery errors under a matching joint DGP |
| `invariance.csv` | Seeded redraw reproducibility and row-permutation NLL checks |

## Fast pytest coverage

```bash
uv run pytest tests/test_numerical_validation_unit.py
```

The unit test runs a reduced seed set in-process and does not require writing
artifacts under `data/`.
