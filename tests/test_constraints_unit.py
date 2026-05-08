import numpy as np

from risk_bridge.constraints import (
  calibration_inequalities,
  calibration_inequalities_jacobian,
  calibration_residuals,
  interval_expected_risk,
)


def _theta() -> np.ndarray:
  return np.array([-2.0, 0.2, 0.1, -0.05, 0.0, 0.1, -1.0, 0.2, 0.1, 0.0, -0.1, 0.5])


def test_interval_expected_risk_range() -> None:
  x_combs = np.array([[0, 0, 0, 0], [1, 0, 0, 0]], dtype=float)
  x_prob = np.array([0.6, 0.4])
  calib_idx = np.array([1, 2])
  out = interval_expected_risk(
    theta=_theta(),
    x_combs=x_combs,
    x_prob_external=x_prob,
    calibration_index=calib_idx,
    n_bins=2,
    tempcateg=np.arange(0.1, 1.0, 0.1),
  )
  assert out.shape == (2,)
  assert np.all((out >= 0.0) & (out <= 1.0))


def test_calibration_residuals_zero_for_matching_external() -> None:
  x_combs = np.array([[0, 0, 0, 0], [1, 0, 0, 0]], dtype=float)
  x_prob = np.array([0.6, 0.4])
  calib_idx = np.array([1, 2])

  expected = interval_expected_risk(
    theta=_theta(),
    x_combs=x_combs,
    x_prob_external=x_prob,
    calibration_index=calib_idx,
    n_bins=2,
    tempcateg=np.arange(0.1, 1.0, 0.1),
  )
  residuals = calibration_residuals(
    theta=_theta(),
    x_combs=x_combs,
    x_prob_external=x_prob,
    calibration_index=calib_idx,
    y_external=expected,
    tempcateg=np.arange(0.1, 1.0, 0.1),
  )
  assert np.allclose(residuals, np.zeros_like(residuals), atol=1e-10)


def test_calibration_inequalities_two_sided_shape() -> None:
  x_combs = np.array([[0, 0, 0, 0], [1, 0, 0, 0]], dtype=float)
  x_prob = np.array([0.6, 0.4])
  calib_idx = np.array([1, 2])
  y_external = np.array([0.1, 0.2])

  cineq = calibration_inequalities(
    theta=_theta(),
    x_combs=x_combs,
    x_prob_external=x_prob,
    calibration_index=calib_idx,
    y_external=y_external,
    tolerance=0.1,
    tempcateg=np.arange(0.1, 1.0, 0.1),
  )
  assert cineq.shape == (4,)


def test_calibration_inequalities_jacobian_matches_finite_difference() -> None:
  x_combs = np.array([[0, 0, 0, 0], [1, 0, 0, 0]], dtype=float)
  x_prob = np.array([0.6, 0.4])
  calib_idx = np.array([1, 2])
  y_external = np.array([0.1, 0.2])
  theta = _theta()
  tempcateg = np.arange(0.1, 1.0, 0.1)

  jac = calibration_inequalities_jacobian(
    theta=theta,
    x_combs=x_combs,
    x_prob_external=x_prob,
    calibration_index=calib_idx,
    y_external=y_external,
    tempcateg=tempcateg,
  )
  assert jac.shape == (4, len(theta))

  fd = np.zeros_like(jac)
  eps = 1e-6
  for j in range(len(theta)):
    plus = theta.copy()
    minus = theta.copy()
    plus[j] += eps
    minus[j] -= eps
    c_plus = calibration_inequalities(
      theta=plus,
      x_combs=x_combs,
      x_prob_external=x_prob,
      calibration_index=calib_idx,
      y_external=y_external,
      tolerance=0.1,
      tempcateg=tempcateg,
    )
    c_minus = calibration_inequalities(
      theta=minus,
      x_combs=x_combs,
      x_prob_external=x_prob,
      calibration_index=calib_idx,
      y_external=y_external,
      tolerance=0.1,
      tempcateg=tempcateg,
    )
    fd[:, j] = (c_plus - c_minus) / (2.0 * eps)

  assert np.allclose(jac, fd, atol=5e-4, rtol=5e-3)
