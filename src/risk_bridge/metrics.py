from __future__ import annotations

import numpy as np
import numpy.typing as npt
from sklearn.metrics import roc_auc_score

ArrayF = npt.NDArray[np.float64]
ArrayI = npt.NDArray[np.int64]


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
