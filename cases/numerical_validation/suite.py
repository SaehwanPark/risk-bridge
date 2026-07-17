"""Independent numerical validation helpers for Risk Bridge estimation core."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from scipy.optimize import minimize

from risk_bridge.constraints import (
  calibration_inequalities,
  calibration_inequalities_jacobian,
)
from risk_bridge.likelihood import (
  joint_negative_log_likelihood,
  joint_negative_log_likelihood_grad,
)
from risk_bridge.optimize import solve_cmle_with_ladder
from risk_bridge.reproducibility import capture_run_environment


@dataclass(frozen=True)
class CheckResult:
  name: str
  passed: bool
  detail: str
  metrics: dict[str, float]


def _central_difference_grad(
  fn: Any, theta: np.ndarray, *, eps: float = 1e-6
) -> np.ndarray:
  fd = np.zeros_like(theta, dtype=np.float64)
  for i in range(len(theta)):
    plus = theta.copy()
    minus = theta.copy()
    plus[i] += eps
    minus[i] -= eps
    fd[i] = (fn(plus) - fn(minus)) / (2.0 * eps)
  return fd


def _central_difference_jacobian(
  fn: Any, theta: np.ndarray, *, eps: float = 1e-6
) -> np.ndarray:
  base = np.asarray(fn(theta), dtype=np.float64)
  jac = np.zeros((len(base), len(theta)), dtype=np.float64)
  for j in range(len(theta)):
    plus = theta.copy()
    minus = theta.copy()
    plus[j] += eps
    minus[j] -= eps
    jac[:, j] = (np.asarray(fn(plus), dtype=np.float64) - np.asarray(fn(minus), dtype=np.float64)) / (
      2.0 * eps
    )
  return jac


def _draw_likelihood_problem(
  rng: np.random.Generator, *, n: int = 8, px: int = 2
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
  x = rng.integers(0, 2, size=(n, px)).astype(np.float64)
  z_cont = rng.uniform(0.05, 0.95, size=n)
  z_cat = rng.integers(0, 5, size=n).astype(np.int64)
  y = rng.integers(0, 2, size=n).astype(np.int64)
  if np.all(y == y[0]):
    y[0] = 1 - int(y[0])
  theta_len = 1 + (px + 1) + (px + 2)
  theta = rng.normal(0.0, 0.25, size=theta_len).astype(np.float64)
  theta[0] = -1.5 + 0.2 * rng.normal()
  theta[-1] = 0.4 + 0.1 * abs(rng.normal())
  return theta, x, y, z_cont, z_cat


def run_derivative_checks(
  *,
  seeds: tuple[int, ...] = (11, 22, 33),
  atol: float = 5e-5,
  rtol: float = 5e-4,
) -> tuple[list[dict[str, object]], CheckResult]:
  rows: list[dict[str, object]] = []
  max_abs_err = 0.0
  max_rel_err = 0.0
  all_passed = True

  for seed in seeds:
    rng = np.random.default_rng(seed)
    theta, x, y, z_cont, z_cat = _draw_likelihood_problem(rng)
    analytic = joint_negative_log_likelihood_grad(theta, x, y, z_cont, z_cat)
    fd = _central_difference_grad(
      lambda t: joint_negative_log_likelihood(t, x, y, z_cont, z_cat), theta
    )
    abs_err = float(np.max(np.abs(analytic - fd)))
    rel_err = float(
      np.max(np.abs(analytic - fd) / np.maximum(np.abs(fd), 1e-8))
    )
    passed = bool(np.allclose(analytic, fd, atol=atol, rtol=rtol))
    all_passed = all_passed and passed
    max_abs_err = max(max_abs_err, abs_err)
    max_rel_err = max(max_rel_err, rel_err)
    rows.append(
      {
        "check": "likelihood_grad",
        "seed": seed,
        "max_abs_error": abs_err,
        "max_rel_error": rel_err,
        "passed": passed,
      }
    )

    x_combs = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float64)
    x_prob = np.full(len(x_combs), 1.0 / len(x_combs), dtype=np.float64)
    calib_idx = np.array([1, 1, 2, 2], dtype=np.int64)
    y_external = np.array([0.15, 0.25], dtype=np.float64)
    tempcateg = np.linspace(0.1, 0.9, 5)

    def cineq(t: np.ndarray) -> np.ndarray:
      return calibration_inequalities(
        theta=t,
        x_combs=x_combs,
        x_prob_external=x_prob,
        calibration_index=calib_idx,
        y_external=y_external,
        tolerance=0.1,
        tempcateg=tempcateg,
      )

    jac = calibration_inequalities_jacobian(
      theta=theta,
      x_combs=x_combs,
      x_prob_external=x_prob,
      calibration_index=calib_idx,
      y_external=y_external,
      tempcateg=tempcateg,
    )
    fd_jac = _central_difference_jacobian(cineq, theta)
    abs_err_j = float(np.max(np.abs(jac - fd_jac)))
    rel_err_j = float(
      np.max(np.abs(jac - fd_jac) / np.maximum(np.abs(fd_jac), 1e-8))
    )
    # Jacobian FD is noisier; keep the looser tolerance from unit tests.
    passed_j = bool(np.allclose(jac, fd_jac, atol=5e-4, rtol=5e-3))
    all_passed = all_passed and passed_j
    max_abs_err = max(max_abs_err, abs_err_j)
    max_rel_err = max(max_rel_err, rel_err_j)
    rows.append(
      {
        "check": "constraint_jacobian",
        "seed": seed,
        "max_abs_error": abs_err_j,
        "max_rel_error": rel_err_j,
        "passed": passed_j,
      }
    )

  result = CheckResult(
    name="derivative_checks",
    passed=all_passed,
    detail="Randomized central-difference checks for NLL grad and constraint Jacobian",
    metrics={"max_abs_error": max_abs_err, "max_rel_error": max_rel_err},
  )
  return rows, result


def run_optimizer_comparison(*, seed: int = 101) -> tuple[list[dict[str, object]], CheckResult]:
  """Compare analytic-Jacobian trust-constr solve vs finite-difference Jacobian."""

  rng = np.random.default_rng(seed)
  theta0, x, y, z_cont, z_cat = _draw_likelihood_problem(rng, n=12, px=2)
  x_combs = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float64)
  x_prob = np.full(4, 0.25, dtype=np.float64)
  calib_idx = np.array([1, 1, 2, 2], dtype=np.int64)
  y_external = np.array([0.2, 0.3], dtype=np.float64)
  tempcateg = np.linspace(0.1, 0.9, 5)

  def objective_fn(theta: np.ndarray) -> float:
    return joint_negative_log_likelihood(theta, x, y, z_cont, z_cat)

  def objective_jac_fn(theta: np.ndarray) -> np.ndarray:
    return joint_negative_log_likelihood_grad(theta, x, y, z_cont, z_cat)

  def constraints_fn(theta: np.ndarray) -> np.ndarray:
    return calibration_inequalities(
      theta=theta,
      x_combs=x_combs,
      x_prob_external=x_prob,
      calibration_index=calib_idx,
      y_external=y_external,
      tolerance=0.2,
      tempcateg=tempcateg,
    )

  def constraints_jac_fn(theta: np.ndarray) -> np.ndarray:
    return calibration_inequalities_jacobian(
      theta=theta,
      x_combs=x_combs,
      x_prob_external=x_prob,
      calibration_index=calib_idx,
      y_external=y_external,
      tempcateg=tempcateg,
    )

  analytic, _ = solve_cmle_with_ladder(
    theta0=theta0,
    objective_fn=objective_fn,
    objective_jac_fn=objective_jac_fn,
    constraints_fn=constraints_fn,
    constraints_jac_fn=constraints_jac_fn,
    maxiter=80,
    feasibility_tol=1e-4,
  )
  fd_jac, _ = solve_cmle_with_ladder(
    theta0=theta0,
    objective_fn=objective_fn,
    objective_jac_fn=objective_jac_fn,
    constraints_fn=constraints_fn,
    constraints_jac_fn=None,
    maxiter=80,
    feasibility_tol=1e-4,
  )

  theta_diff = float(np.max(np.abs(analytic.theta - fd_jac.theta)))
  obj_diff = abs(analytic.objective - fd_jac.objective)
  analytic_violation = float(analytic.diagnostics.get("max_violation", np.nan))
  fd_violation = float(fd_jac.diagnostics.get("max_violation", np.nan))
  both_success = bool(analytic.success) and bool(fd_jac.success)
  both_feasible = (
    np.isfinite(analytic_violation)
    and np.isfinite(fd_violation)
    and analytic_violation <= 1e-4
    and fd_violation <= 1e-4
  )
  passed = (
    both_success
    and both_feasible
    and obj_diff < 0.1
    and theta_diff < 0.35
  )
  row = {
    "seed": seed,
    "analytic_objective": analytic.objective,
    "fd_jacobian_objective": fd_jac.objective,
    "max_abs_theta_diff": theta_diff,
    "abs_objective_diff": obj_diff,
    "analytic_success": bool(analytic.success),
    "fd_success": bool(fd_jac.success),
    "analytic_max_violation": analytic_violation,
    "fd_max_violation": fd_violation,
    "passed": passed,
  }
  result = CheckResult(
    name="optimizer_comparison",
    passed=passed,
    detail=(
      "Analytic constraint Jacobian vs 2-point FD Jacobian on the same problem "
      "(requires success, feasibility, and objective+theta agreement)"
    ),
    metrics={"max_abs_theta_diff": theta_diff, "abs_objective_diff": obj_diff},
  )
  return [row], result


def run_recovery_check(*, seed: int = 202) -> tuple[list[dict[str, object]], CheckResult]:
  """Recover known outcome-model coefficients under the joint NLL.

  Continuous Z and outcomes are generated from the same truth parameterization
  used by the joint likelihood (outcome logit in X/zCat plus a simple truncated
  lognormal Z model for z_cont). Pass criteria apply only to the outcome block
  (alpha, beta), which is the scientifically meaningful recovery target here.
  """

  rng = np.random.default_rng(seed)
  px = 2
  truth = np.array(
    [-1.2, 0.4, -0.2, 0.3, -0.5, 0.1, -0.05, 0.6], dtype=np.float64
  )
  n = 400
  x = rng.integers(0, 2, size=(n, px)).astype(np.float64)
  alpha = float(truth[0])
  beta = truth[1 : 1 + px + 1]
  gamma = truth[1 + px + 1 :]
  # Generate continuous Z from the truncated-lognormal mean structure.
  tau = gamma[0] + x @ gamma[1 : 1 + px]
  sigma = max(float(gamma[-1]), 1e-3)
  z_cont = np.clip(
    np.exp(tau + sigma * rng.normal(size=n)), 1e-4, 1.0 - 1e-4
  ).astype(np.float64)
  z_bins = np.array([0.25, 0.5, 0.75], dtype=np.float64)
  z_cat = np.digitize(z_cont, z_bins).astype(np.int64)
  logit = alpha + x @ beta[:px] + beta[-1] * z_cat.astype(np.float64)
  p = 1.0 / (1.0 + np.exp(-logit))
  y = (rng.uniform(size=n) < p).astype(np.int64)

  def objective(theta: np.ndarray) -> float:
    return joint_negative_log_likelihood(theta, x, y, z_cont, z_cat)

  def jac(theta: np.ndarray) -> np.ndarray:
    return joint_negative_log_likelihood_grad(theta, x, y, z_cont, z_cat)

  start = truth + rng.normal(0.0, 0.1, size=len(truth))
  start[-1] = abs(float(start[-1])) + 0.2
  res = minimize(objective, start, method="L-BFGS-B", jac=jac, options={"maxiter": 400})
  est = np.asarray(res.x, dtype=np.float64)
  outcome_rmse = float(
    np.sqrt(np.mean(np.square(est[: 1 + px + 1] - truth[: 1 + px + 1])))
  )
  max_abs = float(np.max(np.abs(est[: 1 + px + 1] - truth[: 1 + px + 1])))
  passed = bool(res.success) and outcome_rmse < 0.35 and max_abs < 0.75
  row = {
    "seed": seed,
    "outcome_rmse": outcome_rmse,
    "outcome_max_abs_error": max_abs,
    "optimizer_success": bool(res.success),
    "passed": passed,
  }
  result = CheckResult(
    name="recovery",
    passed=passed,
    detail=(
      "Joint-NLL recovery of known outcome-model coefficients "
      "(alpha/beta) under a matching X/zCat/z_cont DGP"
    ),
    metrics={"outcome_rmse": outcome_rmse, "outcome_max_abs_error": max_abs},
  )
  return [row], result


def run_invariance_checks(*, seed: int = 303) -> tuple[list[dict[str, object]], CheckResult]:
  rng_a = np.random.default_rng(seed)
  theta_a, x_a, y_a, z_cont_a, z_cat_a = _draw_likelihood_problem(rng_a, n=10, px=2)
  rng_b = np.random.default_rng(seed)
  theta_b, x_b, y_b, z_cont_b, z_cat_b = _draw_likelihood_problem(rng_b, n=10, px=2)

  nll_a = joint_negative_log_likelihood(theta_a, x_a, y_a, z_cont_a, z_cat_a)
  grad_a = joint_negative_log_likelihood_grad(theta_a, x_a, y_a, z_cont_a, z_cat_a)
  nll_b = joint_negative_log_likelihood(theta_b, x_b, y_b, z_cont_b, z_cat_b)
  grad_b = joint_negative_log_likelihood_grad(theta_b, x_b, y_b, z_cont_b, z_cat_b)
  seed_replay_pass = (
    abs(nll_a - nll_b) < 1e-12
    and np.allclose(theta_a, theta_b)
    and np.allclose(grad_a, grad_b)
  )

  # Row permutation should leave the joint NLL unchanged.
  order = rng_a.permutation(len(y_a))
  nll_perm = joint_negative_log_likelihood(
    theta_a, x_a[order], y_a[order], z_cont_a[order], z_cat_a[order]
  )
  perm_pass = abs(nll_a - nll_perm) < 1e-10

  rows = [
    {
      "check": "seed_replay_draw",
      "abs_nll_diff": abs(nll_a - nll_b),
      "max_abs_grad_diff": float(np.max(np.abs(grad_a - grad_b))),
      "passed": seed_replay_pass,
    },
    {
      "check": "row_permutation",
      "abs_nll_diff": abs(nll_a - nll_perm),
      "max_abs_grad_diff": float("nan"),
      "passed": perm_pass,
    },
  ]
  result = CheckResult(
    name="invariance",
    passed=seed_replay_pass and perm_pass,
    detail="Seeded redraw reproducibility and row-permutation invariance of the NLL",
    metrics={
      "abs_nll_diff_seed_replay": abs(nll_a - nll_b),
      "abs_nll_diff_permutation": abs(nll_a - nll_perm),
    },
  )
  return rows, result


def capture_environment() -> dict[str, Any]:
  return capture_run_environment(cwd=Path(__file__).resolve().parents[2])


def run_suite(
  *,
  derivative_seeds: tuple[int, ...] = (11, 22, 33),
  optimizer_seed: int = 101,
  recovery_seed: int = 202,
  invariance_seed: int = 303,
) -> dict[str, Any]:
  deriv_rows, deriv = run_derivative_checks(seeds=derivative_seeds)
  opt_rows, opt = run_optimizer_comparison(seed=optimizer_seed)
  rec_rows, rec = run_recovery_check(seed=recovery_seed)
  inv_rows, inv = run_invariance_checks(seed=invariance_seed)
  checks = [deriv, opt, rec, inv]
  return {
    "passed": all(c.passed for c in checks),
    "checks": [asdict(c) for c in checks],
    "derivative_rows": deriv_rows,
    "optimizer_rows": opt_rows,
    "recovery_rows": rec_rows,
    "invariance_rows": inv_rows,
    "environment": capture_environment(),
  }


def write_suite_artifacts(suite: dict[str, Any], output_dir: Path) -> Path:
  output_dir.mkdir(parents=True, exist_ok=True)
  (output_dir / "summary.json").write_text(
    json.dumps(
      {
        "passed": suite["passed"],
        "checks": suite["checks"],
      },
      indent=2,
    )
    + "\n",
    encoding="utf-8",
  )
  (output_dir / "environment.json").write_text(
    json.dumps(suite["environment"], indent=2) + "\n", encoding="utf-8"
  )
  pl.DataFrame(suite["derivative_rows"]).write_csv(output_dir / "derivative_checks.csv")
  pl.DataFrame(suite["optimizer_rows"]).write_csv(output_dir / "optimizer_comparison.csv")
  pl.DataFrame(suite["recovery_rows"]).write_csv(output_dir / "recovery.csv")
  pl.DataFrame(suite["invariance_rows"]).write_csv(output_dir / "invariance.csv")
  return output_dir
