from __future__ import annotations

from itertools import repeat

import numpy as np
import numpy.typing as npt
import polars as pl
from scipy.stats import norm

from risk_bridge.config import FeatureSpec
from risk_bridge.types import Population

ArrayF = npt.NDArray[np.float64]
ArrayI = npt.NDArray[np.int64]


def expit(x: ArrayF | float) -> ArrayF:
  """Numerically stable logistic transform.

  Examples
  --------
  >>> expit(np.array([0.0]))
  array([0.5])
  >>> expit(np.array([-1000.0, 1000.0]))
  array([0., 1.])
  """

  arr = np.asarray(x, dtype=np.float64)
  pos = arr >= 0
  out = np.empty_like(arr)
  out[pos] = 1.0 / (1.0 + np.exp(-arr[pos]))
  exp_x = np.exp(arr[~pos])
  out[~pos] = exp_x / (1.0 + exp_x)
  return out


def _sample_categorical_cut(
  rng: np.random.Generator, n: int, breaks: tuple[float, ...]
) -> ArrayI:
  probs = np.diff(np.asarray(breaks, dtype=np.float64))
  if np.any(probs <= 0):
    raise ValueError("categorical_cut breaks must be strictly increasing")
  probs = probs / probs.sum()
  return rng.choice(len(probs), size=n, p=probs).astype(np.int64)


def _sample_capped_poisson(
  rng: np.random.Generator, n: int, lam: float, cap: int
) -> ArrayI:
  draws = rng.poisson(lam=lam, size=n)
  return np.minimum(draws, cap).astype(np.int64)


def sample_x(
  rng: np.random.Generator,
  n: int,
  feature_specs: tuple[FeatureSpec, ...],
) -> pl.DataFrame:
  """Generate a discrete design matrix from feature specifications.

  Supported kinds:
  - `categorical_cut`: probabilities induced by interval widths of `breaks`
  - `capped_poisson`: draws from Poisson capped at `cap`

  Examples
  --------
  >>> rng = np.random.default_rng(0)
  >>> specs = (FeatureSpec("X1", "categorical_cut", {"breaks": (0.0, 0.5, 1.0)}),)
  >>> sample_x(rng, 3, specs).shape
  (3, 1)
  """

  out: dict[str, ArrayI] = {}
  for spec in feature_specs:
    if spec.kind == "categorical_cut":
      breaks = tuple(float(v) for v in spec.params["breaks"])
      out[spec.name] = _sample_categorical_cut(rng, n, breaks)
    elif spec.kind == "capped_poisson":
      lam = float(spec.params["lambda"])
      cap = int(spec.params.get("cap", 2))
      out[spec.name] = _sample_capped_poisson(rng, n, lam, cap)
    elif spec.kind == "custom":
      values = np.asarray(spec.params["values"], dtype=np.int64)
      probs = np.asarray(
        spec.params.get("probs", list(repeat(1.0, len(values)))), dtype=np.float64
      )
      probs = probs / probs.sum()
      out[spec.name] = rng.choice(values, size=n, p=probs)
    else:
      raise ValueError(f"Unsupported feature kind: {spec.kind}")

  return pl.DataFrame(out)


def sample_trunc_lognormal_z(
  rng: np.random.Generator,
  X: pl.DataFrame | npt.NDArray[np.float64],
  gamma: ArrayF,
) -> ArrayF:
  """Draw Z from lognormal(exp(N(mu, sigma))) truncated to (0, 1]."""

  x = np.asarray(X, dtype=np.float64)
  if x.ndim != 2:
    raise ValueError("X must be a 2D array or DataFrame")
  px = x.shape[1]
  if len(gamma) != px + 2:
    raise ValueError("gamma must have length px + 2")

  tau = gamma[0] + x @ gamma[1 : 1 + px]
  sigma = max(float(gamma[-1]), 1e-8)

  cdf0 = np.clip(norm.cdf(0.0, loc=tau, scale=sigma), 1e-12, 1.0)
  u = rng.uniform(low=1e-12, high=1.0 - 1e-12, size=len(tau))
  logz = norm.ppf(u * cdf0, loc=tau, scale=sigma)
  return np.exp(logz)


def categorize_z(z_cont: ArrayF, bins: ArrayF) -> ArrayI:
  """Map continuous Z in (0, 1] to ordinal categories 0..len(bins)."""

  z = np.asarray(z_cont, dtype=np.float64)
  b = np.asarray(bins, dtype=np.float64)
  if np.any(np.diff(b) <= 0):
    raise ValueError("bins must be strictly increasing")
  cats = np.digitize(z, b, right=True).astype(np.int64)
  return np.clip(cats, 0, len(b)).astype(np.int64)


def sample_binary_y(
  rng: np.random.Generator,
  X: pl.DataFrame | npt.NDArray[np.float64],
  z_cat: ArrayI,
  alpha: float,
  beta: ArrayF,
) -> ArrayI:
  """Generate binary outcomes from logistic(alpha + beta_x@X + beta_z*z_cat)."""

  x = np.asarray(X, dtype=np.float64)
  zc = np.asarray(z_cat, dtype=np.float64)
  px = x.shape[1]
  if len(beta) != px + 1:
    raise ValueError("beta must have length px + 1")
  lp = alpha + x @ beta[:px] + beta[-1] * zc
  p = expit(lp)
  y = rng.uniform(size=len(zc)) < p
  return y.astype(np.int64)


def generate_population(
  rng: np.random.Generator,
  n: int,
  feature_specs: tuple[FeatureSpec, ...],
  gamma: ArrayF,
  z_bins: ArrayF,
  alpha: float,
  beta: ArrayF,
) -> Population:
  """Generate a synthetic population with X, continuous/categorical Z, and Y."""

  X = sample_x(rng=rng, n=n, feature_specs=feature_specs)
  z_cont = sample_trunc_lognormal_z(
    rng=rng, X=X, gamma=np.asarray(gamma, dtype=np.float64)
  )
  z_cat = categorize_z(z_cont=z_cont, bins=np.asarray(z_bins, dtype=np.float64))
  y = sample_binary_y(
    rng=rng, X=X, z_cat=z_cat, alpha=alpha, beta=np.asarray(beta, dtype=np.float64)
  )
  return Population(
    X=X, z_cont=z_cont, z_cat=z_cat.astype(np.int64), y=y.astype(np.int64)
  )
