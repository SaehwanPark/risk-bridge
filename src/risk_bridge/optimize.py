from __future__ import annotations

from typing import Callable

import numpy as np
from scipy.optimize import NonlinearConstraint, minimize

from risk_bridge.types import FitResult

ObjectiveFn = Callable[[np.ndarray], float]
ObjectiveJacFn = Callable[[np.ndarray], np.ndarray]
ConstraintsFn = Callable[[np.ndarray], np.ndarray]
ConstraintsJacFn = Callable[[np.ndarray], np.ndarray]


def _max_violation(cineq: np.ndarray) -> float:
  return float(np.max(np.maximum(cineq, 0.0))) if len(cineq) else 0.0


def _run_once(
  method: str,
  theta0: np.ndarray,
  objective_fn: ObjectiveFn,
  objective_jac_fn: ObjectiveJacFn | None,
  constraints_fn: ConstraintsFn,
  constraints_jac_fn: ConstraintsJacFn | None,
  maxiter: int,
) -> FitResult:
  theta_start = np.asarray(theta0, dtype=np.float64)

  if method == "trust-constr":
    con = NonlinearConstraint(
      lambda x: constraints_fn(x),
      -np.inf,
      0.0,
      jac=(lambda x: constraints_jac_fn(x)) if constraints_jac_fn is not None else "2-point",
    )
    res = minimize(
      objective_fn,
      theta_start,
      method="trust-constr",
      jac=objective_jac_fn,
      constraints=[con],
      options={"maxiter": maxiter},
    )
  elif method == "SLSQP":
    con = {"type": "ineq", "fun": lambda x: -constraints_fn(x)}
    if constraints_jac_fn is not None:
      con["jac"] = lambda x: -constraints_jac_fn(x)
    res = minimize(
      objective_fn,
      theta_start,
      method="SLSQP",
      jac=objective_jac_fn,
      constraints=[con],
      options={"maxiter": maxiter},
    )
  else:
    raise ValueError(f"Unsupported method: {method}")

  cineq = np.asarray(constraints_fn(res.x), dtype=np.float64)
  violation = _max_violation(cineq)
  n_iter = int(getattr(res, "nit", getattr(res, "niter", 0)) or 0)

  return FitResult(
    theta=np.asarray(res.x, dtype=np.float64),
    success=bool(res.success),
    status=str(res.message),
    n_iter=n_iter,
    objective=float(res.fun),
    diagnostics={
      "solver": method,
      "max_violation": violation,
      "status_code": int(getattr(res, "status", -1)),
    },
  )


def solve_cmle_with_ladder(
  theta0: np.ndarray,
  objective_fn: ObjectiveFn,
  constraints_fn: ConstraintsFn,
  objective_jac_fn: ObjectiveJacFn | None = None,
  constraints_jac_fn: ConstraintsJacFn | None = None,
  methods: tuple[str, ...] = ("trust-constr", "SLSQP"),
  maxiter: int = 500,
  feasibility_tol: float = 1e-6,
) -> tuple[FitResult, list[FitResult]]:
  """Run constrained solver ladder and return best result plus attempt history.

  Feasible acceptance: `max(constraint, 0) <= feasibility_tol`.
  """

  history: list[FitResult] = []
  current = np.asarray(theta0, dtype=np.float64)

  for method in methods:
    attempt = _run_once(
      method=method,
      theta0=current,
      objective_fn=objective_fn,
      objective_jac_fn=objective_jac_fn,
      constraints_fn=constraints_fn,
      constraints_jac_fn=constraints_jac_fn,
      maxiter=maxiter,
    )
    history.append(attempt)
    current = attempt.theta

    max_v = float(attempt.diagnostics.get("max_violation", np.inf))
    if attempt.success and max_v <= feasibility_tol:
      return attempt, history

  # If no feasible+successful attempt exists, return least-violation attempt explicitly flagged.
  best = min(
    history,
    key=lambda h: (float(h.diagnostics.get("max_violation", np.inf)), h.objective),
  )
  max_v = float(best.diagnostics.get("max_violation", np.inf))
  return (
    FitResult(
      theta=best.theta,
      success=False,
      status=f"Infeasible after ladder (min_violation={max_v:.3e})",
      n_iter=best.n_iter,
      objective=best.objective,
      diagnostics=dict(best.diagnostics),
    ),
    history,
  )
