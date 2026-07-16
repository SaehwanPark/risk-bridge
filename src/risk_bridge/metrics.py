from __future__ import annotations

import numpy as np
import numpy.typing as npt
from scipy.optimize import brentq, minimize
from scipy.special import expit, logit
from sklearn.metrics import roc_auc_score

ArrayF = npt.NDArray[np.float64]
ArrayI = npt.NDArray[np.int64]

_PROBABILITY_EPS = 1e-12
_NEAR_CONSTANT_LOGIT_SPAN = 1e-12


def _validate_calibration_inputs(
  y_true: ArrayI, predictions: ArrayF
) -> tuple[ArrayF, ArrayF]:
  try:
    y = np.asarray(y_true, dtype=np.float64)
    p = np.asarray(predictions, dtype=np.float64)
  except (TypeError, ValueError) as exc:
    raise ValueError("y_true and predictions must be numeric arrays") from exc

  if y.ndim != 1 or p.ndim != 1:
    raise ValueError("y_true and predictions must be one-dimensional")
  if len(y) == 0 or len(y) != len(p):
    raise ValueError("y_true and predictions must be non-empty and equal in length")
  if not np.all(np.isfinite(y)) or not np.all(np.isin(y, (0.0, 1.0))):
    raise ValueError("y_true must contain only finite binary outcomes")
  if not np.all(np.isfinite(p)) or np.any((p < 0.0) | (p > 1.0)):
    raise ValueError("predictions must be finite probabilities in [0, 1]")
  return y.astype(np.float64, copy=False), p.astype(np.float64, copy=False)


def _calibration_intercept(logit_predictions: ArrayF, y: ArrayF) -> float:
  y_mean = float(np.mean(y))
  if y_mean <= 0.0 or y_mean >= 1.0:
    return float("nan")

  target = float(np.sum(y))

  def score(intercept: float) -> float:
    return float(np.sum(expit(logit_predictions + intercept)) - target)

  lower = -1.0
  upper = 1.0
  while score(lower) > 0.0:
    lower *= 2.0
  while score(upper) < 0.0:
    upper *= 2.0

  try:
    return float(brentq(score, lower, upper, xtol=1e-12, rtol=1e-12))
  except (ValueError, RuntimeError):
    return float("nan")


def _calibration_slope(logit_predictions: ArrayF, y: ArrayF, intercept: float) -> float:
  if np.ptp(logit_predictions) <= _NEAR_CONSTANT_LOGIT_SPAN:
    return float("nan")
  if not np.isfinite(intercept):
    return float("nan")

  def objective(params: ArrayF) -> float:
    eta = params[0] + params[1] * logit_predictions
    return float(np.sum(np.logaddexp(0.0, eta) - y * eta))

  def gradient(params: ArrayF) -> ArrayF:
    eta = params[0] + params[1] * logit_predictions
    error = expit(eta) - y
    return np.array(
      [np.sum(error), np.dot(error, logit_predictions)], dtype=np.float64
    )

  result = minimize(
    objective,
    np.array([intercept, 1.0], dtype=np.float64),
    jac=gradient,
    method="BFGS",
    options={"gtol": 1e-8, "maxiter": 1000},
  )
  if not result.success or not np.all(np.isfinite(result.x)):
    return float("nan")
  return float(result.x[1])


def calibration_metrics(y_true: ArrayI, predictions: ArrayF) -> dict[str, float]:
  """Compute standalone calibration metrics for labeled predictions.

  Calibration-in-the-large is the intercept in the offset logistic model
  ``expit(logit(p) + a)``. Calibration slope is the slope from unregularized
  logistic recalibration ``expit(a + b * logit(p))`` with an intercept. Both
  use predictions clipped to ``[1e-12, 1 - 1e-12]`` only for logit calculations.
  The observed/expected ratio uses the unclipped probability sums, and the
  Brier score is the mean squared probability error.

  Invalid inputs raise ``ValueError``. A calibration intercept or slope that
  is not finitely estimable, including one-class outcomes or effectively
  constant predictions for the slope, is returned as ``NaN``. An observed /
  expected ratio with a zero expected-risk denominator is also ``NaN``.
  """

  y, p = _validate_calibration_inputs(y_true, predictions)
  clipped = np.clip(p, _PROBABILITY_EPS, 1.0 - _PROBABILITY_EPS)
  logit_predictions = np.asarray(logit(clipped), dtype=np.float64)
  intercept = _calibration_intercept(logit_predictions, y)
  expected = float(np.sum(p))

  return {
    "calibration_in_the_large": intercept,
    "calibration_slope": _calibration_slope(logit_predictions, y, intercept),
    "observed_expected_ratio": (
      float(np.sum(y) / expected) if expected > 0.0 else float("nan")
    ),
    "brier_score": float(np.mean(np.square(p - y))),
  }
def roc_auc_binary(y_true: ArrayI, scores: ArrayF) -> float:
  """Return ROC AUC for binary labels; NaN when only one class is present."""

  y = np.asarray(y_true)
  s = np.asarray(scores, dtype=np.float64)
  if len(np.unique(y)) < 2:
    return float("nan")
  return float(roc_auc_score(y, s))


def threshold_metrics(
  y_true: ArrayI, scores: ArrayF, threshold: float
) -> dict[str, float]:
  """Compute TPR, PPV, and TNR at a given threshold."""

  y = np.asarray(y_true, dtype=np.int64)
  s = np.asarray(scores, dtype=np.float64)

  pred = (s >= threshold).astype(np.int64)
  pos = y == 1
  neg = y == 0

  tp = np.sum((pred == 1) & pos)
  tn = np.sum((pred == 0) & neg)

  n_pos = int(np.sum(pos))
  n_neg = int(np.sum(neg))
  pred_pos = int(np.sum(pred == 1))

  tpr = float(tp / n_pos) if n_pos > 0 else float("nan")
  ppv = float(tp / pred_pos) if pred_pos > 0 else float("nan")
  tnr = float(tn / n_neg) if n_neg > 0 else float("nan")

  return {"tpr": tpr, "ppv": ppv, "tnr": tnr}


def find_threshold_for_fpr(
  y_true: ArrayI, scores: ArrayF, target_fpr: float, grid_step: float = 0.001
) -> float:
  """Find threshold with false-positive rate closest to target_fpr.

  Example
  -------
  >>> y = np.array([0, 0, 1, 1])
  >>> s = np.array([0.1, 0.4, 0.6, 0.9])
  >>> 0.0 <= find_threshold_for_fpr(y, s, 0.5) <= 1.0
  True
  """

  if not (0.0 <= target_fpr <= 1.0):
    raise ValueError("target_fpr must be in [0, 1]")
  if grid_step <= 0:
    raise ValueError("grid_step must be > 0")

  y = np.asarray(y_true, dtype=np.int64)
  s = np.asarray(scores, dtype=np.float64)

  thresholds = np.arange(0.0, 1.0 + 1e-12, grid_step)
  neg = y == 0

  if not np.any(neg):
    return 1.0

  fprs = np.empty_like(thresholds)
  for i, t in enumerate(thresholds):
    pred = s >= t
    fp = np.sum(pred & neg)
    tn = np.sum((~pred) & neg)
    fprs[i] = fp / max(fp + tn, 1)

  idx = int(np.argmin(np.abs(fprs - target_fpr)))
  return float(thresholds[idx])
