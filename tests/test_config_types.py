import numpy as np
import pandas as pd
import polars as pl
import pytest

from risk_bridge.config import (
  DEFAULT_Z_BINS,
  FeatureSpec,
  OptimizationConfig,
  RunConfig,
  Scenario1PipelineOptions,
  SimulationConfig,
  UserDataRunConfig,
  UserDataSchema,
  ZModelSpec,
)
from risk_bridge.cli import (
  build_scenario2_run_config,
  build_scenario3_run_config,
)
from risk_bridge.types import Population


def test_valid_config_construction() -> None:
  cfg = RunConfig(
    seed=123,
    output_root="data/out",
    sim=SimulationConfig(
      nsim=2,
      n_target=20,
      n_source=20,
      n_reference=40,
      sample_size=10,
      target_prevalence=0.1,
      target_fpr=0.1,
      alpha=-2.0,
      beta=(0.2, 0.1),
      feature_specs=(
        FeatureSpec(
          name="X1", kind="categorical_cut", params={"breaks": (0.0, 0.5, 1.0)}
        ),
      ),
      z_spec=ZModelSpec(
        family="trunc_lognormal", gamma_init=(-1.0, 0.2, 0.1), bins=(0.1, 0.9)
      ),
    ),
    opt=OptimizationConfig(
      mle_method="BFGS", cmle_method="trust-constr", tol=1e-6, maxiter=100
    ),
  )
  assert cfg.sim.sample_size == 10


def test_invalid_prevalence_raises() -> None:
  with pytest.raises(ValueError):
    SimulationConfig(
      nsim=1,
      n_target=10,
      n_source=10,
      n_reference=10,
      sample_size=5,
      target_prevalence=1.2,
      target_fpr=0.1,
      alpha=-2.0,
      beta=(0.2, 0.1, -0.1, 0.05, 0.15),
      feature_specs=(
        FeatureSpec(name="X1", kind="categorical_cut", params={"breaks": (0.0, 1.0)}),
      ),
      z_spec=ZModelSpec(
        family="trunc_lognormal", gamma_init=(-1.0, 0.2, 0.1), bins=(0.1, 0.9)
      ),
    )


def test_invalid_bins_raises() -> None:
  with pytest.raises(ValueError):
    ZModelSpec(family="trunc_lognormal", gamma_init=(-1.0, 0.2, 0.1), bins=(0.5, 0.2))


def test_invalid_source_feature_support_raises() -> None:
  with pytest.raises(ValueError, match="preserve the same discrete support"):
    SimulationConfig(
      nsim=1,
      n_target=10,
      n_source=10,
      n_reference=10,
      sample_size=5,
      target_prevalence=0.1,
      target_fpr=0.1,
      alpha=-2.0,
      beta=(0.2, 0.1),
      feature_specs=(
        FeatureSpec(
          name="X1", kind="categorical_cut", params={"breaks": (0.0, 0.5, 1.0)}
        ),
      ),
      source_feature_specs=(
        FeatureSpec(
          name="X1",
          kind="categorical_cut",
          params={"breaks": (0.0, 0.2, 0.4, 1.0)},
        ),
      ),
      z_spec=ZModelSpec(
        family="trunc_lognormal", gamma_init=(-1.0, 0.2, 0.1), bins=(0.1, 0.9)
      ),
    )


def test_population_shape_validation() -> None:
  x = pl.DataFrame({"X1": [0, 1], "X2": [1, 0]})
  with pytest.raises(ValueError):
    Population(
      X=x,
      z_cont=np.array([0.2]),
      z_cat=np.array([1, 2]),
      y=np.array([0, 1]),
    )


def test_valid_library_api_configs() -> None:
  schema = UserDataSchema(x_cols=("X1", "X2"), z_bins=DEFAULT_Z_BINS[:3])
  cfg = UserDataRunConfig(
    target_df=pd.DataFrame({"caseY": [0, 1], "X1": [0, 1], "X2": [1, 0], "zOrigin": [0.2, 0.4]}),
    source_df=pd.DataFrame({"caseY": [1, 0], "X1": [1, 0], "X2": [0, 1], "zOrigin": [0.3, 0.5]}),
    reference_df=pd.DataFrame({"caseY": [0, 0], "X1": [0, 1], "X2": [1, 1], "zOrigin": [0.25, 0.55]}),
    schema=schema,
    sample_size=1,
  )
  options = Scenario1PipelineOptions(run_label="typed_api")

  assert cfg.schema.x_cols == ("X1", "X2")
  assert options.run_label == "typed_api"


def test_scenario2_and_scenario3_preset_metadata_defaults() -> None:
  scenario2 = build_scenario2_run_config(output_root="data/out")
  scenario3 = build_scenario3_run_config(output_root="data/out")

  assert scenario2.seed == 475
  assert scenario2.sim.scenario_name == "Scenario2"
  assert scenario2.sim.scenario_run_label == "scenario2"
  assert scenario2.sim.source_miscalibration_a == 0.5
  assert scenario2.sim.source_miscalibration_b == 1.0
  assert scenario2.sim.source_feature_specs is not None
  assert scenario2.sim.beta[-1] == 0.1

  assert scenario3.seed == 73
  assert scenario3.sim.scenario_name == "Scenario3"
  assert scenario3.sim.scenario_run_label == "scenario3"
  assert scenario3.sim.source_miscalibration_a == -0.5
  assert scenario3.sim.source_miscalibration_b == 1.2
  assert scenario3.sim.source_feature_specs is not None
  assert scenario3.sim.beta[-1] == 0.9


def test_invalid_user_data_schema_raises() -> None:
  with pytest.raises(ValueError):
    UserDataSchema(x_cols=(), z_bins=(0.2, 0.4))
