from __future__ import annotations

import numpy as np
import numpy.typing as npt
from scipy.special import ndtr

from risk_bridge.simulate import expit

ArrayF = npt.NDArray[np.float64]
ArrayI = npt.NDArray[np.int64]
_INV_SQRT_2PI = 1.0 / np.sqrt(2.0 * np.pi)


def unpack_theta(theta: ArrayF, px: int) -> tuple[float, ArrayF, ArrayF]:
  """Unpack theta=[alpha, beta_x..., beta_z, gamma0, gamma_x..., sigma]."""

  expected = 1 + (px + 1) + (px + 2)
  if len(theta) != expected:
    raise ValueError(f"theta length must be {expected} for px={px}, got {len(theta)}")

  alpha = float(theta[0])
  beta = np.asarray(theta[1 : 1 + px + 1], dtype=np.float64)
  gamma = np.asarray(theta[1 + px + 1 :], dtype=np.float64)
  return alpha, beta, gamma


def logistic_risk(alpha: float, beta: ArrayF, X: ArrayF, z_cat: ArrayI) -> ArrayF:
  px = X.shape[1]
  lp = alpha + X @ beta[:px] + beta[-1] * z_cat
  return np.clip(expit(lp), 1e-12, 1.0 - 1e-12)


def _std_norm_pdf(x: ArrayF) -> ArrayF:
  return _INV_SQRT_2PI * np.exp(-0.5 * np.square(x))


def truncated_lognormal_density(gamma: ArrayF, X: ArrayF, z_cont: ArrayF) -> ArrayF:
  px = X.shape[1]
  tau = gamma[0] + X @ gamma[1 : 1 + px]
  sigma = max(float(gamma[-1]), 1e-8)
  trun0 = np.clip(ndtr(-tau / sigma), 1e-12, None)

  logz = np.log(np.clip(z_cont, 1e-300, None))
  a = (logz - tau) / sigma
  log_pdf = -0.5 * np.square(a) - np.log(np.sqrt(2.0 * np.pi)) - np.log(sigma)
  ftau = np.clip(np.exp(log_pdf) / trun0, 1e-300, None)
  return ftau


def joint_negative_log_likelihood(
  theta: ArrayF,
  xinput: ArrayF,
  yinput: ArrayI,
  zinput: ArrayF,
  zcinput: ArrayI,
) -> float:
  """Joint negative log-likelihood for Y|X,Zcat and truncated-lognormal Z|X.

  Examples
  --------
  >>> x = np.zeros((2, 4))
  >>> y = np.array([0, 1])
  >>> z = np.array([0.2, 0.4])
  >>> zc = np.array([1, 3])
  >>> th = np.array([-2.0, 0.3, 0.1, 0.0, 0.0, 0.2, -1.0, 0.2, 0.1, 0.0, 0.0, 0.5])
  >>> float(joint_negative_log_likelihood(th, x, y, z, zc)) > 0
  True
  """

  X = np.asarray(xinput, dtype=np.float64)
  y = np.asarray(yinput, dtype=np.float64)
  z = np.asarray(zinput, dtype=np.float64)
  zc = np.asarray(zcinput, dtype=np.float64)

  alpha, beta, gamma = unpack_theta(np.asarray(theta, dtype=np.float64), px=X.shape[1])
  p_all = logistic_risk(alpha, beta, X, zc)
  f_tau = truncated_lognormal_density(gamma, X, z)

  loglike = np.sum(y * np.log(p_all) + (1.0 - y) * np.log(1.0 - p_all) + np.log(f_tau))
  return float(-loglike)


def joint_negative_log_likelihood_grad(
  theta: ArrayF,
  xinput: ArrayF,
  yinput: ArrayI,
  zinput: ArrayF,
  zcinput: ArrayI,
) -> ArrayF:
  """Gradient for `joint_negative_log_likelihood` with matching theta layout."""

  X = np.asarray(xinput, dtype=np.float64)
  y = np.asarray(yinput, dtype=np.float64)
  z = np.asarray(zinput, dtype=np.float64)
  zc = np.asarray(zcinput, dtype=np.float64)

  px = X.shape[1]
  theta_arr = np.asarray(theta, dtype=np.float64)
  alpha, beta, gamma = unpack_theta(theta_arr, px=px)

  p_all = logistic_risk(alpha, beta, X, zc)
  err = p_all - y

  grad = np.zeros_like(theta_arr)
  grad[0] = np.sum(err)
  grad[1 : 1 + px] = X.T @ err
  grad[1 + px] = np.dot(zc, err)

  tau = gamma[0] + X @ gamma[1 : 1 + px]
  sigma_raw = float(gamma[-1])
  sigma = max(sigma_raw, 1e-8)
  dsigma_dgamma = 1.0 if sigma_raw > 1e-8 else 0.0

  logz = np.log(np.clip(z, 1e-300, None))
  u0 = -tau / sigma
  trun0 = np.clip(ndtr(u0), 1e-12, None)
  lambda0 = _std_norm_pdf(u0) / trun0

  delta = logz - tau
  sigma_sq = sigma * sigma

  dlogf_dtau = delta / sigma_sq + lambda0 / sigma
  grad[1 + px + 1] = -np.sum(dlogf_dtau)
  grad[1 + px + 2 : 1 + px + 2 + px] = -(X.T @ dlogf_dtau)

  dlogf_dsigma = (
    -1.0 / sigma
    + np.square(delta) / (sigma_sq * sigma)
    - lambda0 * tau / sigma_sq
  )
  grad[-1] = -np.sum(dlogf_dsigma) * dsigma_dgamma
  return grad.astype(np.float64, copy=False)
