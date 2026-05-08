from __future__ import annotations

import numpy as np
import numpy.typing as npt
from scipy.special import ndtr

from risk_bridge.likelihood import unpack_theta
from risk_bridge.simulate import expit

ArrayF = npt.NDArray[np.float64]
ArrayI = npt.NDArray[np.int64]
_INV_SQRT_2PI = 1.0 / np.sqrt(2.0 * np.pi)


def _std_norm_pdf(x: ArrayF) -> ArrayF:
  return _INV_SQRT_2PI * np.exp(-0.5 * np.square(x))


def _category_prob_terms(
  tau: ArrayF, sigma: float, tempcateg: ArrayF
) -> tuple[ArrayF, ArrayF, ArrayF]:
  """Return P(Zcat=k|X), dP/dtau, and dP/dsigma for k=0..len(tempcateg)."""

  n = len(tau)
  k = len(tempcateg)
  if k == 0:
    probs = np.ones((n, 1), dtype=np.float64)
    zeros = np.zeros_like(probs)
    return probs, zeros, zeros

  log_bins = np.log(np.asarray(tempcateg, dtype=np.float64))

  q = (log_bins[None, :] - tau[:, None]) / sigma
  cdf_bins = ndtr(q)
  pdf_bins = _std_norm_pdf(q)

  u0 = -tau / sigma
  trun0 = np.clip(ndtr(u0), 1e-12, None)
  pdf_u0 = _std_norm_pdf(u0)

  dtrun_dtau = -pdf_u0 / sigma
  dtrun_dsigma = pdf_u0 * tau / (sigma * sigma)

  dcdf_dtau = -pdf_bins / sigma
  dcdf_dsigma = -pdf_bins * (log_bins[None, :] - tau[:, None]) / (sigma * sigma)

  num = np.empty((n, k + 1), dtype=np.float64)
  dnum_dtau = np.empty_like(num)
  dnum_dsigma = np.empty_like(num)

  num[:, 0] = cdf_bins[:, 0]
  dnum_dtau[:, 0] = dcdf_dtau[:, 0]
  dnum_dsigma[:, 0] = dcdf_dsigma[:, 0]

  if k > 1:
    num[:, 1:k] = cdf_bins[:, 1:] - cdf_bins[:, :-1]
    dnum_dtau[:, 1:k] = dcdf_dtau[:, 1:] - dcdf_dtau[:, :-1]
    dnum_dsigma[:, 1:k] = dcdf_dsigma[:, 1:] - dcdf_dsigma[:, :-1]

  num[:, k] = trun0 - cdf_bins[:, -1]
  dnum_dtau[:, k] = dtrun_dtau - dcdf_dtau[:, -1]
  dnum_dsigma[:, k] = dtrun_dsigma - dcdf_dsigma[:, -1]

  tr_col = trun0[:, None]
  probs = num / tr_col
  dp_dtau = (dnum_dtau * tr_col - num * dtrun_dtau[:, None]) / (tr_col * tr_col)
  dp_dsigma = (
    dnum_dsigma * tr_col - num * dtrun_dsigma[:, None]
  ) / (tr_col * tr_col)

  probs = np.clip(probs, 0.0, 1.0)
  row_sum = probs.sum(axis=1, keepdims=True)
  row_sum = np.where(row_sum <= 0.0, 1.0, row_sum)
  probs = probs / row_sum
  return probs, dp_dtau, dp_dsigma


def _expected_risk_and_grad_per_x(
  theta: ArrayF, x_combs: ArrayF, tempcateg: ArrayF
) -> tuple[ArrayF, ArrayF]:
  """Return E[risk|X] and Jacobian dE/dtheta for each X combination."""

  x_arr = np.asarray(x_combs, dtype=np.float64)
  px = x_arr.shape[1]
  alpha, beta, gamma = unpack_theta(np.asarray(theta, dtype=np.float64), px=px)

  sigma_raw = float(gamma[-1])
  sigma = max(sigma_raw, 1e-8)
  dsigma_dgamma = 1.0 if sigma_raw > 1e-8 else 0.0

  tau = gamma[0] + x_arr @ gamma[1 : 1 + px]
  probs_z, dp_dtau, dp_dsigma = _category_prob_terms(tau, sigma, tempcateg)

  z_levels = np.arange(len(tempcateg) + 1, dtype=np.float64)
  a = alpha + x_arr @ beta[:px]
  risk = expit(a[:, None] + beta[-1] * z_levels[None, :])
  drisk_da = risk * (1.0 - risk)

  expected = np.sum(risk * probs_z, axis=1)
  expected = np.clip(expected, 0.0, 1.0)

  dE_da = np.sum(drisk_da * probs_z, axis=1)
  dE_dbetaz = np.sum(drisk_da * probs_z * z_levels[None, :], axis=1)
  dE_dtau = np.sum(risk * dp_dtau, axis=1)
  dE_dsigma = np.sum(risk * dp_dsigma, axis=1)

  n_params = 1 + (px + 1) + (px + 2)
  grad = np.zeros((len(x_arr), n_params), dtype=np.float64)

  grad[:, 0] = dE_da
  grad[:, 1 : 1 + px] = dE_da[:, None] * x_arr
  grad[:, 1 + px] = dE_dbetaz

  gamma_start = 1 + (px + 1)
  grad[:, gamma_start] = dE_dtau
  grad[:, gamma_start + 1 : gamma_start + 1 + px] = dE_dtau[:, None] * x_arr
  grad[:, -1] = dE_dsigma * dsigma_dgamma
  return expected, grad


def interval_expected_risk(
  theta: ArrayF,
  x_combs: ArrayF,
  x_prob_external: ArrayF,
  calibration_index: ArrayI,
  n_bins: int,
  tempcateg: ArrayF,
) -> ArrayF:
  """Expected predicted risk in each calibration bin under external X distribution."""

  x = np.asarray(x_combs, dtype=np.float64)
  w = np.asarray(x_prob_external, dtype=np.float64)
  idx = np.asarray(calibration_index, dtype=np.int64)

  if len(x) != len(w) or len(x) != len(idx):
    raise ValueError(
      "x_combs, x_prob_external, and calibration_index lengths must match"
    )

  expected_x, _ = _expected_risk_and_grad_per_x(
    np.asarray(theta, dtype=np.float64), x, np.asarray(tempcateg, dtype=np.float64)
  )
  idx0 = idx - 1
  denom = np.bincount(idx0, weights=w, minlength=n_bins).astype(np.float64)
  numer = np.bincount(idx0, weights=w * expected_x, minlength=n_bins).astype(
    np.float64
  )

  out = np.zeros(n_bins, dtype=np.float64)
  mask = denom > 0.0
  out[mask] = numer[mask] / denom[mask]
  return out


def _interval_expected_risk_jacobian(
  theta: ArrayF,
  x_combs: ArrayF,
  x_prob_external: ArrayF,
  calibration_index: ArrayI,
  n_bins: int,
  tempcateg: ArrayF,
) -> ArrayF:
  x = np.asarray(x_combs, dtype=np.float64)
  w = np.asarray(x_prob_external, dtype=np.float64)
  idx = np.asarray(calibration_index, dtype=np.int64)

  if len(x) != len(w) or len(x) != len(idx):
    raise ValueError(
      "x_combs, x_prob_external, and calibration_index lengths must match"
    )

  _, grad_per_x = _expected_risk_and_grad_per_x(
    np.asarray(theta, dtype=np.float64), x, np.asarray(tempcateg, dtype=np.float64)
  )

  idx0 = idx - 1
  denom = np.bincount(idx0, weights=w, minlength=n_bins).astype(np.float64)
  n_params = grad_per_x.shape[1]
  jac = np.zeros((n_bins, n_params), dtype=np.float64)
  valid = denom > 0.0

  for j in range(n_params):
    numer_j = np.bincount(
      idx0, weights=w * grad_per_x[:, j], minlength=n_bins
    ).astype(np.float64)
    jac[valid, j] = numer_j[valid] / denom[valid]
  return jac


def calibration_residuals(
  theta: ArrayF,
  x_combs: ArrayF,
  x_prob_external: ArrayF,
  calibration_index: ArrayI,
  y_external: ArrayF,
  tempcateg: ArrayF,
) -> ArrayF:
  """Residuals: expected interval risk minus external prevalence target."""

  y_ext = np.asarray(y_external, dtype=np.float64)
  expected = interval_expected_risk(
    theta=theta,
    x_combs=x_combs,
    x_prob_external=x_prob_external,
    calibration_index=calibration_index,
    n_bins=len(y_ext),
    tempcateg=tempcateg,
  )
  return expected - y_ext


def calibration_inequalities(
  theta: ArrayF,
  x_combs: ArrayF,
  x_prob_external: ArrayF,
  calibration_index: ArrayI,
  y_external: ArrayF,
  tolerance: float,
  tempcateg: ArrayF,
) -> ArrayF:
  """Two-sided inequalities implementing |residual_r| <= tolerance * y_external_r.

  Returns c(theta) <= 0 for optimizer APIs.
  """

  residuals = calibration_residuals(
    theta=theta,
    x_combs=x_combs,
    x_prob_external=x_prob_external,
    calibration_index=calibration_index,
    y_external=y_external,
    tempcateg=tempcateg,
  )
  slack = tolerance * np.asarray(y_external, dtype=np.float64)
  return np.concatenate([residuals - slack, -residuals - slack]).astype(np.float64)


def calibration_inequalities_jacobian(
  theta: ArrayF,
  x_combs: ArrayF,
  x_prob_external: ArrayF,
  calibration_index: ArrayI,
  y_external: ArrayF,
  tempcateg: ArrayF,
) -> ArrayF:
  """Jacobian of `calibration_inequalities` with shape (2*n_bins, n_theta)."""

  n_bins = len(np.asarray(y_external, dtype=np.float64))
  jac_res = _interval_expected_risk_jacobian(
    theta=theta,
    x_combs=x_combs,
    x_prob_external=x_prob_external,
    calibration_index=calibration_index,
    n_bins=n_bins,
    tempcateg=tempcateg,
  )
  return np.vstack([jac_res, -jac_res]).astype(np.float64)
