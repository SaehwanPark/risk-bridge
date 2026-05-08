import numpy as np
import pytest

from risk_bridge.likelihood import (
  joint_negative_log_likelihood,
  joint_negative_log_likelihood_grad,
  unpack_theta,
)


def _tiny_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
  x = np.array(
    [
      [0.0, 0.0, 0.0, 0.0],
      [1.0, 0.0, 1.0, 0.0],
      [1.0, 1.0, 0.0, 1.0],
    ]
  )
  z_cont = np.array([0.2, 0.4, 0.7])
  z_cat = np.array([1, 3, 7])
  y = np.array([0, 1, 1])
  return x, y, z_cont, z_cat


def test_unpack_theta_validates_length() -> None:
  with pytest.raises(ValueError):
    unpack_theta(np.zeros(3), px=4)


def test_joint_negative_log_likelihood_is_finite() -> None:
  x, y, z_cont, z_cat = _tiny_data()
  theta = np.array([-2.0, 0.4, 0.1, -0.1, 0.2, 0.1, -1.0, 0.2, 0.1, 0.05, -0.1, 0.5])
  val = joint_negative_log_likelihood(theta, x, y, z_cont, z_cat)
  assert np.isfinite(val)


def test_nll_prefers_more_consistent_alpha() -> None:
  x, y, z_cont, z_cat = _tiny_data()
  theta_good = np.array(
    [-1.0, 0.4, 0.1, -0.1, 0.2, 0.2, -1.0, 0.2, 0.1, 0.05, -0.1, 0.5]
  )
  theta_bad = theta_good.copy()
  theta_bad[0] = -7.0

  assert joint_negative_log_likelihood(
    theta_good, x, y, z_cont, z_cat
  ) < joint_negative_log_likelihood(
    theta_bad,
    x,
    y,
    z_cont,
    z_cat,
  )


def test_joint_negative_log_likelihood_grad_matches_finite_difference() -> None:
  x, y, z_cont, z_cat = _tiny_data()
  theta = np.array([-2.0, 0.4, 0.1, -0.1, 0.2, 0.1, -1.0, 0.2, 0.1, 0.05, -0.1, 0.5])

  grad = joint_negative_log_likelihood_grad(theta, x, y, z_cont, z_cat)
  fd = np.zeros_like(theta)
  eps = 1e-6
  for i in range(len(theta)):
    plus = theta.copy()
    minus = theta.copy()
    plus[i] += eps
    minus[i] -= eps
    fd[i] = (
      joint_negative_log_likelihood(plus, x, y, z_cont, z_cat)
      - joint_negative_log_likelihood(minus, x, y, z_cont, z_cat)
    ) / (2.0 * eps)

  assert np.allclose(grad, fd, atol=5e-5, rtol=5e-4)
