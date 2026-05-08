# Solver Strategy

Risk Bridge fits a joint likelihood for outcome and risk-bin structure, then
adds calibration inequalities for constrained MLE.

## Default Flow

1. Fit unconstrained ML with BFGS.
2. Use the ML estimate as the warm start for constrained cMLE.
3. Attempt `trust-constr` with analytic objective gradients and constraint Jacobians.
4. Fall back to SLSQP when needed.
5. Persist fit status, objective values, and maximum calibration violation.

The constrained fit is accepted only when solver diagnostics satisfy the configured
feasibility tolerance.

## Numerical Guardrails

- Objective code clamps probabilities away from exact zero.
- Constraint code uses analytic Jacobians and explicit calibration tolerance.
- Fit diagnostics record solver name, status text, objective value, and max violation.
- Tests include finite-difference checks for likelihood gradients and constraint Jacobians.

## Contributor Guidance

When changing likelihood, constraints, or calibration behavior:

- update analytic derivatives in the same change
- add or update focused numerical tests
- run `uv run mypy`
- run the likelihood, constraint, calibration, optimize, and pipeline tests
