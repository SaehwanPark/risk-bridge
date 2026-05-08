# Quickstart

This guide gets Risk Bridge running in about five minutes.

## 1. Install dependencies

```bash
git clone https://github.com/<your-org>/risk-bridge.git
cd risk-bridge
uv sync --locked
```

## 2. Verify the environment

```bash
uv run pytest
```

## 3. Run a small simulation

```bash
uv run risk-bridge \
  --mode simulated \
  --scenario 1 \
  --nsim 2 \
  --n-target 1000 \
  --n-source 500 \
  --n-reference 1000 \
  --sample-size 100 \
  --print-every 1 \
  --output-root data \
  --run-label smoke
```

## 4. Inspect outputs

The command prints the run directory. The most useful files are:

```text
data/<timestamp>_smoke/final/run_metadata.csv
data/<timestamp>_smoke/final/fit_diagnostics.csv
data/<timestamp>_smoke/final/roc_metrics.csv
data/<timestamp>_smoke/final/accuracy_metrics.csv
```

## 5. Use the package from Python

```python
from risk_bridge import build_scenario1_run_config, run_simulation

cfg = build_scenario1_run_config(
  nsim=2,
  n_target=1000,
  n_source=500,
  n_reference=1000,
  sample_size=100,
  output_root="data",
)
run_dir = run_simulation(cfg)
print(run_dir)
```
