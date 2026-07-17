from risk_bridge.reproducibility import (
  REPRODUCIBILITY_CONTRACT_VERSION,
  capture_run_environment,
)


def test_environment_includes_reproducibility_contract() -> None:
  env = capture_run_environment()

  contract = env["reproducibility_contract"]
  assert contract["version"] == REPRODUCIBILITY_CONTRACT_VERSION
  assert contract["floating_point"]["same_environment"] == {
    "atol": 1e-8,
    "rtol": 1e-6,
  }
  assert contract["floating_point"]["cross_platform"] == {
    "atol": 1e-6,
    "rtol": 1e-4,
  }
  assert set(env["thread_environment"]) == {
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "POLARS_MAX_THREADS",
  }
