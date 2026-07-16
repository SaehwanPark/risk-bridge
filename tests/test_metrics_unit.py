import numpy as np
import pytest

from risk_bridge.metrics import (
  calibration_metrics,
  find_threshold_for_fpr,
  roc_auc_binary,
  threshold_metrics,
)


def test_calibration_metrics_perfectly_calibrated_fixture() -> None:
  predictions = np.array([0.2] * 10 + [0.8] * 10)
  y_true = np.array([1, 1] + [0] * 8 + [1] * 8 + [0, 0])

  metrics = calibration_metrics(y_true, predictions)

  assert np.isclose(metrics["calibration_in_the_large"], 0.0, atol=1e-8)
  assert np.isclose(metrics["calibration_slope"], 1.0, atol=1e-8)
  assert np.isclose(metrics["observed_expected_ratio"], 1.0, atol=1e-12)
  assert np.isclose(metrics["brier_score"], 0.16, atol=1e-12)


def test_calibration_metrics_constant_predictions() -> None:
  predictions = np.full(10, 0.3)
  y_true = np.array([1, 1, 1] + [0] * 7)

  metrics = calibration_metrics(y_true, predictions)

  assert np.isclose(metrics["calibration_in_the_large"], 0.0, atol=1e-8)
  assert np.isnan(metrics["calibration_slope"])
  assert np.isclose(metrics["observed_expected_ratio"], 1.0, atol=1e-12)
  assert np.isclose(metrics["brier_score"], 0.21, atol=1e-12)


def test_calibration_metrics_nearly_constant_predictions_return_nan_slope() -> None:
  predictions = np.array([0.5, 0.5 + 1e-14, 0.5, 0.5 + 1e-14])
  y_true = np.array([0, 1, 1, 0])

  metrics = calibration_metrics(y_true, predictions)

  assert np.isfinite(metrics["calibration_in_the_large"])
  assert np.isnan(metrics["calibration_slope"])


def test_calibration_metrics_accepts_endpoint_probabilities() -> None:
  predictions = np.array([0.0, 1.0, 0.25, 0.75, 0.2, 0.8])
  y_true = np.array([0, 1, 0, 1, 0, 1])

  metrics = calibration_metrics(y_true, predictions)

  assert all(np.isfinite(value) for value in metrics.values())
  assert 0.0 <= metrics["brier_score"] <= 1.0


@pytest.mark.parametrize("outcome", [0, 1])
def test_calibration_metrics_all_one_outcome_class_is_non_estimable(
  outcome: int,
) -> None:
  predictions = np.array([0.2, 0.4, 0.6, 0.8])
  y_true = np.full(4, outcome)

  metrics = calibration_metrics(y_true, predictions)

  assert np.isnan(metrics["calibration_in_the_large"])
  assert np.isnan(metrics["calibration_slope"])
  assert np.isfinite(metrics["observed_expected_ratio"])
  assert np.isfinite(metrics["brier_score"])


@pytest.mark.parametrize(
  ("y_true", "predictions"),
  [
    (np.array([0, 2]), np.array([0.2, 0.8])),
    (np.array([0, 1]), np.array([np.nan, 0.8])),
    (np.array([0, 1]), np.array([-0.1, 0.8])),
    (np.array([0, 1]), np.array([0.2, 1.1])),
    (np.array([0]), np.array([0.2, 0.8])),
    (np.array([], dtype=np.int64), np.array([], dtype=np.float64)),
  ],
)
def test_calibration_metrics_rejects_invalid_inputs(
  y_true: np.ndarray, predictions: np.ndarray
) -> None:
  with pytest.raises(ValueError):
    calibration_metrics(y_true, predictions)


@pytest.mark.parametrize("y_true", [np.array([0, 0]), np.array([0, 1])])
def test_calibration_metrics_zero_expected_risk_returns_nan_oe(
  y_true: np.ndarray,
) -> None:
  metrics = calibration_metrics(y_true, np.zeros(2))

  assert np.isnan(metrics["observed_expected_ratio"])


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
