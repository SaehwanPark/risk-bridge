from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, TypeAlias

import numpy as np
import numpy.typing as npt
import polars as pl

ArrayF: TypeAlias = npt.NDArray[np.float64]
ArrayI: TypeAlias = npt.NDArray[np.int64]


@dataclass(frozen=True)
class Population:
  """Synthetic or canonicalized dataset used by the estimation pipeline.

  Attributes
  ----------
  X:
    Discrete design matrix with one column per X feature.
  z_cont:
    Continuous `zOrigin` values.
  z_cat:
    Integer-coded categorical `zCat` values.
  y:
    Binary outcomes.
  """

  X: pl.DataFrame
  z_cont: ArrayF
  z_cat: ArrayI
  y: ArrayI

  def __post_init__(self) -> None:
    n = len(self.X)
    if len(self.z_cont) != n or len(self.z_cat) != n or len(self.y) != n:
      raise ValueError("Population arrays must all have length len(X).")


@dataclass(frozen=True)
class CalibrationArtifacts:
  """Precomputed calibration arrays derived from the reference population."""

  risk_limits: ArrayF
  x_interval_index: ArrayI
  p_external: ArrayF


@dataclass(frozen=True)
class FitResult:
  """Outcome of an ML or cMLE optimization attempt."""

  theta: ArrayF
  success: bool
  status: str
  n_iter: int
  objective: float
  diagnostics: dict[str, float | int | str] = field(default_factory=dict)


@dataclass(frozen=True)
class EvaluationSummary:
  """Threshold and ROC metrics for one fitted parameter vector."""

  auc: float
  threshold: float
  tpr: float
  ppv: float
  tnr: float

  @classmethod
  def from_metrics(
    cls, *, auc: float, threshold: float, metrics: Mapping[str, float]
  ) -> EvaluationSummary:
    """Build an evaluation summary from `threshold_metrics` output."""

    return cls(
      auc=float(auc),
      threshold=float(threshold),
      tpr=float(metrics["tpr"]),
      ppv=float(metrics["ppv"]),
      tnr=float(metrics["tnr"]),
    )

  def as_dict(self) -> dict[str, float]:
    """Return a JSON/DataFrame-friendly representation of the metrics."""

    return {
      "auc": self.auc,
      "threshold": self.threshold,
      "tpr": self.tpr,
      "ppv": self.ppv,
      "tnr": self.tnr,
    }


@dataclass(frozen=True)
class IterationMetrics:
  """Evaluation summaries for ML and cMLE fits from one iteration."""

  mle: EvaluationSummary
  cmle: EvaluationSummary


@dataclass(frozen=True)
class IterationResult:
  """Typed result bundle for one development/simulation iteration."""

  mle: FitResult
  cmle: FitResult
  metrics: IterationMetrics
  solver_history: tuple[FitResult, ...]
