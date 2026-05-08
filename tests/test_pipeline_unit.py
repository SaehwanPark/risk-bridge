from risk_bridge.config import (
  FeatureSpec,
  OptimizationConfig,
  RunConfig,
  SimulationConfig,
  ZModelSpec,
)
from risk_bridge.pipeline import run_single_iteration, run_single_iteration_result
from risk_bridge.types import IterationResult


def test_run_single_iteration_smoke() -> None:
  cfg = RunConfig(
    seed=42,
    output_root="data/out",
    sim=SimulationConfig(
      nsim=1,
      n_target=24,
      n_source=24,
      n_reference=32,
      sample_size=12,
      target_prevalence=0.1,
      target_fpr=0.1,
      alpha=-2.2,
      beta=(0.35, 0.2, -0.1, 0.1, 0.25),
      feature_specs=(
        FeatureSpec(
          name="X1",
          kind="categorical_cut",
          params={"breaks": (0.0, 0.2, 0.56, 0.9, 1.0)},
        ),
        FeatureSpec(
          name="X2", kind="categorical_cut", params={"breaks": (0.0, 0.5, 0.8, 1.0)}
        ),
        FeatureSpec(name="X3", kind="capped_poisson", params={"lambda": 0.5, "cap": 1}),
        FeatureSpec(name="X4", kind="capped_poisson", params={"lambda": 0.4, "cap": 1}),
      ),
      z_spec=ZModelSpec(
        family="trunc_lognormal",
        gamma_init=(-1.0, 0.2, 0.1, 0.0, -0.1, 0.5),
        bins=(0.3, 0.6, 0.9),
      ),
    ),
    opt=OptimizationConfig(
      mle_method="BFGS", cmle_method="trust-constr", tol=1e-6, maxiter=20
    ),
  )

  out = run_single_iteration(cfg)
  assert "mle" in out
  assert "cmle" in out
  assert "metrics" in out

  typed = run_single_iteration_result(cfg)
  assert isinstance(typed, IterationResult)
  assert typed.metrics.cmle.auc >= 0.0
