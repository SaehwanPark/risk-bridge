import numpy as np
import polars as pl

from risk_bridge.calibration import (
  assign_risk_interval,
  build_calibration_artifacts,
  compute_risk_limits,
  enumerate_x_combinations,
  estimate_external_prevalence,
)
from risk_bridge.config import FeatureSpec
from risk_bridge.types import Population


def test_compute_risk_limits_quantiles() -> None:
  phi = np.array([0.1, 0.2, 0.3, 0.4])
  limits = compute_risk_limits(phi)
  assert np.allclose(limits, np.array([0.175, 0.25, 0.325]))


def test_assign_risk_interval() -> None:
  phi = np.array([0.1, 0.2, 0.5, 0.8])
  idx = assign_risk_interval(phi, np.array([0.2, 0.5, 0.7]))
  assert np.array_equal(idx, np.array([1, 1, 2, 4]))


def test_estimate_external_prevalence_handles_empty_bins() -> None:
  y = np.array([1, 0, 1])
  interval_idx = np.array([1, 1, 3])
  pe = estimate_external_prevalence(y=y, interval_idx=interval_idx, n_bins=4)
  assert np.allclose(pe, np.array([0.5, 0.0, 1.0, 0.0]))


def test_enumerate_x_combinations() -> None:
  specs = (
    FeatureSpec(name="X1", kind="categorical_cut", params={"breaks": (0.0, 0.5, 1.0)}),
    FeatureSpec(name="X2", kind="capped_poisson", params={"lambda": 0.5, "cap": 2}),
  )
  combos = enumerate_x_combinations(specs)
  assert combos.shape == (6, 2)


def test_build_calibration_artifacts_shapes() -> None:
  x = pl.DataFrame(
    {
      "X1": [0, 0, 1, 1, 0, 1],
      "X2": [0, 1, 0, 1, 1, 0],
      "X3": [0, 1, 2, 0, 2, 1],
      "X4": [1, 0, 1, 2, 0, 2],
    }
  )
  pop = Population(
    X=x,
    z_cont=np.array([0.2, 0.3, 0.4, 0.5, 0.6, 0.7]),
    z_cat=np.array([1, 2, 3, 4, 5, 6]),
    y=np.array([0, 1, 0, 1, 0, 1]),
  )
  specs = (
    FeatureSpec(name="X1", kind="categorical_cut", params={"breaks": (0.0, 0.5, 1.0)}),
    FeatureSpec(name="X2", kind="categorical_cut", params={"breaks": (0.0, 0.5, 1.0)}),
    FeatureSpec(name="X3", kind="capped_poisson", params={"lambda": 0.6, "cap": 2}),
    FeatureSpec(name="X4", kind="capped_poisson", params={"lambda": 0.6, "cap": 2}),
  )
  art = build_calibration_artifacts(pop, specs)
  assert art.risk_limits.shape == (3,)
  assert art.x_interval_index.shape == (36,)
  assert art.p_external.shape == (4,)
