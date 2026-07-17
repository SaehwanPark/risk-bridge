# External Calibration Validation

Privacy-safe synthetic study that freezes target-X distributions and interval
prevalence summaries, runs the public external-calibration bootstrap path on
source data only, and applies executable recovery/calibration pass/fail gates.

Artifacts are written under gitignored `data/` and must not be committed.

If `summary.json` reports `"passed": false`, **demote** external-calibration
claims in the manuscript and SPEC completion notes. Do not invent success.

## Regenerate

From the repository root:

```bash
python -m cases.external_calibration_validation.run_suite \
  --output-root data \
  --profile smoke
```

Manuscript-citable profile:

```bash
python -m cases.external_calibration_validation.run_suite \
  --output-root data \
  --profile full \
  --condition matched \
  --run-label manuscript_external_calibration
```

The manuscript-oriented profile uses `nsim=50`. The prespecified synthetic
degradation negative control should fail at least one gate:

```bash
python -m cases.external_calibration_validation.run_suite \
  --output-root data \
  --profile smoke \
  --run-label external_calibration_degraded \
  --condition degraded
```

The degraded condition applies a fixed `+6.0` logit shift to every external
prevalence while leaving the known outcome truth unchanged. It is a synthetic
negative control, not a scientific comparison.

## Profiles

| Profile | Role | Approx. sizes |
| --- | --- | --- |
| `smoke` | CI / local gate check | n≈1200/900/1200, sample 180, nsim=1 |
| `full` | Manuscript-citable regeneration | n=5000/2000/5000, sample 500, nsim=50 |

## Outputs

Each run creates:

```text
data/external_calibration_validation/<run_label>_<timestamp>/
  summary.json
  environment.json
  fixed_summary.json
  recovery.csv
  calibration_gates.csv
  fit_diagnostics_summary.csv
  runs/python_external_.../
```

| Artifact | Contents |
| --- | --- |
| `summary.json` | Overall pass/fail, per-check metrics, demotion rule |
| `environment.json` | Python/platform/package/git metadata, thread settings, and reproducibility tolerances |
| `fixed_summary.json` | Frozen target-X marginals, interval index, `p_external` |
| `recovery.csv` | Outcome-coefficient RMSE plus replicate SD/MCSE for MLE/cMLE × PSM/RS |
| `calibration_gates.csv` | Mean absolute calibration residuals plus replicate SD/MCSE by path/fit |
| `fit_diagnostics_summary.csv` | Success rates, Bernoulli MCSEs, and mean max violation |

## Pass/fail gates

1. **feasibility** — at least one cMLE path (PSM or RS) succeeds.
2. **calibration** — for a successful cMLE path, mean `|residual|` vs frozen
   `p_external` is ≤ 0.05.
3. **recovery** — outcome-coefficient RMSE (vs known Scenario-2-shaped DGP) for
   that cMLE path is below 1.25.

Matching MLE residuals/RMSE are recorded for comparison but do not alone grant
a pass. Overall failure exits with code 1.

Manuscript citation should use a passing `--profile full` regeneration; the
smoke profile is a CI/local gate check only.

No empirical baseline is included: only a comparator for the same source-only
fixed-summary task, inputs, and evaluation target would be defensible. The
evidence remains limited to this Scenario 2-shaped synthetic setting and does
not make full Scenario 1/3 claims.

## Fast pytest coverage

```bash
uv run pytest tests/test_external_calibration_validation_unit.py
```

The unit test runs the smoke profile under a temporary output root.
