import numpy as np

from risk_bridge.metrics import find_threshold_for_fpr, roc_auc_binary, threshold_metrics


def test_roc_auc_binary_perfect() -> None:
  y = np.array([0, 0, 1, 1])
  s = np.array([0.1, 0.2, 0.8, 0.9])
  assert roc_auc_binary(y, s) == 1.0


def test_roc_auc_binary_single_class_returns_nan() -> None:
  y = np.array([1, 1, 1])
  s = np.array([0.2, 0.4, 0.7])
  assert np.isnan(roc_auc_binary(y, s))


def test_threshold_metrics_handles_no_predicted_positive() -> None:
  y = np.array([0, 1, 1, 0])
  s = np.array([0.1, 0.2, 0.3, 0.4])
  m = threshold_metrics(y, s, threshold=0.99)
  assert np.isnan(m["ppv"])
  assert m["tpr"] == 0.0


def test_find_threshold_for_fpr() -> None:
  y = np.array([0, 0, 0, 1, 1, 1])
  s = np.array([0.1, 0.2, 0.4, 0.6, 0.8, 0.9])
  t = find_threshold_for_fpr(y, s, target_fpr=1 / 3, grid_step=0.01)
  assert 0.20 <= t <= 0.22
