import numpy as np

from risk_bridge.optimize import solve_cmle_with_ladder


def test_solve_cmle_with_ladder_finds_feasible_solution() -> None:
  def objective(t: np.ndarray) -> float:
    return float((t[0] - 1.0) ** 2)

  def constraints(t: np.ndarray) -> np.ndarray:
    return np.array([t[0] - 1.5])

  best, history = solve_cmle_with_ladder(
    theta0=np.array([4.0]),
    objective_fn=objective,
    constraints_fn=constraints,
    maxiter=100,
  )
  assert history
  assert best.success
  assert best.theta[0] <= 1.5 + 1e-4


def test_solve_cmle_with_ladder_reports_infeasible_when_no_feasible_point() -> None:
  def objective(t: np.ndarray) -> float:
    return float(t[0] ** 2)

  def constraints(t: np.ndarray) -> np.ndarray:
    return np.array([1.0])

  best, _ = solve_cmle_with_ladder(
    theta0=np.array([0.0]),
    objective_fn=objective,
    constraints_fn=constraints,
    maxiter=50,
  )
  assert not best.success
  assert "infeasible" in best.status.lower()
