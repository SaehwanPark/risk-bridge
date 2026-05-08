import numpy as np
import polars as pl
import pytest

from risk_bridge.config import FeatureSpec
from risk_bridge.simulate import (
  categorize_z,
  expit,
  generate_population,
  sample_binary_y,
  sample_trunc_lognormal_z,
  sample_x,
)


def test_expit_stability_at_extremes() -> None:
  vals = expit(np.array([-1000.0, 0.0, 1000.0]))
  assert np.isclose(vals[0], 0.0, atol=1e-12)
  assert np.isclose(vals[1], 0.5, atol=1e-12)
  assert np.isclose(vals[2], 1.0, atol=1e-12)


def test_sample_x_with_mixed_specs() -> None:
  rng = np.random.default_rng(7)
  specs = (
    FeatureSpec(name="X1", kind="categorical_cut", params={"breaks": (0.0, 0.2, 1.0)}),
    FeatureSpec(name="X2", kind="capped_poisson", params={"lambda": 0.8, "cap": 2}),
  )
  x = sample_x(rng=rng, n=100, feature_specs=specs)
  assert isinstance(x, pl.DataFrame)
  assert list(x.columns) == ["X1", "X2"]
  assert set(x.get_column("X1").unique().to_list()).issubset({0, 1})
  assert set(x.get_column("X2").unique().to_list()).issubset({0, 1, 2})


def test_sample_trunc_lognormal_bounds() -> None:
  rng = np.random.default_rng(1)
  x = np.zeros((50, 4), dtype=float)
  z = sample_trunc_lognormal_z(
    rng=rng, X=x, gamma=np.array([-1.0, 0.2, 0.1, -0.1, 0.05, 0.5])
  )
  assert z.shape == (50,)
  assert np.all(z > 0.0)
  assert np.all(z <= 1.0)


def test_categorize_z_boundaries() -> None:
  z = np.array([0.01, 0.10, 0.11, 0.95])
  z_cat = categorize_z(z, bins=np.arange(0.1, 1.0, 0.1))
  assert np.array_equal(z_cat, np.array([0, 0, 1, 9]))


def test_sample_binary_y_extreme_linear_predictor() -> None:
  rng = np.random.default_rng(3)
  x = np.zeros((100, 4), dtype=float)
  z_cat = np.zeros(100, dtype=int)

  y_hi = sample_binary_y(
    rng=rng, X=x, z_cat=z_cat, alpha=8.0, beta=np.array([0.0, 0.0, 0.0, 0.0, 0.0])
  )
  assert y_hi.mean() > 0.95

  y_lo = sample_binary_y(
    rng=rng, X=x, z_cat=z_cat, alpha=-8.0, beta=np.array([0.0, 0.0, 0.0, 0.0, 0.0])
  )
  assert y_lo.mean() < 0.05


def test_generate_population_shapes() -> None:
  rng = np.random.default_rng(11)
  specs = (
    FeatureSpec(name="X1", kind="categorical_cut", params={"breaks": (0.0, 0.5, 1.0)}),
    FeatureSpec(name="X2", kind="categorical_cut", params={"breaks": (0.0, 0.4, 1.0)}),
    FeatureSpec(name="X3", kind="capped_poisson", params={"lambda": 0.5, "cap": 2}),
    FeatureSpec(name="X4", kind="capped_poisson", params={"lambda": 0.4, "cap": 2}),
  )
  pop = generate_population(
    rng=rng,
    n=40,
    feature_specs=specs,
    gamma=np.array([-1.0, 0.2, 0.1, 0.1, -0.1, 0.5]),
    z_bins=np.arange(0.1, 1.0, 0.1),
    alpha=-2.0,
    beta=np.array([0.4, 0.2, -0.1, 0.1, 0.25]),
  )
  assert pop.X.shape == (40, 4)
  assert pop.z_cont.shape == (40,)
  assert pop.z_cat.shape == (40,)
  assert pop.y.shape == (40,)


def test_sample_x_invalid_kind_raises() -> None:
  with pytest.raises(ValueError):
    FeatureSpec(name="X1", kind="unknown", params={})
