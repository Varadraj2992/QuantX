from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.optimize import minimize


@dataclass
class OptimizationResult:
    """Structured output from a portfolio optimization run."""

    method: str
    weights: pd.Series
    expected_return: float
    volatility: float
    sharpe_ratio: float
    covariance: pd.DataFrame
    correlation: pd.DataFrame


def compute_expected_returns(returns: pd.DataFrame) -> pd.Series:
    """Estimate annualized expected returns from a returns DataFrame."""
    if returns.empty:
        raise ValueError("Returns DataFrame cannot be empty.")
    annualized = returns.mean() * 252.0
    return annualized.sort_index()


def compute_covariance_matrix(returns: pd.DataFrame) -> pd.DataFrame:
    """Estimate the covariance matrix using daily returns."""
    if returns.empty:
        raise ValueError("Returns DataFrame cannot be empty.")
    return returns.cov() * 252.0


def compute_correlation_matrix(returns: pd.DataFrame) -> pd.DataFrame:
    """Estimate asset correlation matrix."""
    if returns.empty:
        raise ValueError("Returns DataFrame cannot be empty.")
    return returns.corr()


def _normalize_weights(weights: Sequence[float]) -> np.ndarray:
    values = np.asarray(weights, dtype=float)
    if np.any(values < 0):
        raise ValueError("Portfolio weights cannot be negative.")
    total = values.sum()
    if not np.isclose(total, 1.0):
        values = values / total
    return values


def _portfolio_metrics(
    weights: Sequence[float],
    returns: pd.DataFrame,
    risk_free_rate: float = 0.02,
) -> tuple[float, float, float]:
    w = np.asarray(weights, dtype=float)
    expected = float(returns.mean().dot(w) * 252.0)
    cov = compute_covariance_matrix(returns)
    volatility = float(np.sqrt(w @ cov.to_numpy() @ w))
    excess_return = expected - risk_free_rate
    sharpe = float(excess_return / volatility) if volatility > 0 else 0.0
    return expected, volatility, sharpe


def optimize_equal_weight(returns: pd.DataFrame) -> OptimizationResult:
    """Create a simple equal-weight portfolio."""
    names = list(returns.columns)
    weights = pd.Series(np.full(len(names), 1.0 / len(names), dtype=float), index=names)
    expected, volatility, sharpe = _portfolio_metrics(weights.values, returns)
    return OptimizationResult(
        method="equal_weight",
        weights=weights,
        expected_return=expected,
        volatility=volatility,
        sharpe_ratio=sharpe,
        covariance=compute_covariance_matrix(returns),
        correlation=compute_correlation_matrix(returns),
    )


def optimize_min_volatility(
    returns: pd.DataFrame,
    min_weight: float = 0.05,
    max_weight: float = 0.5,
) -> OptimizationResult:
    """Minimize portfolio variance subject to practical weight bounds."""
    names = list(returns.columns)
    n_assets = len(names)
    cov = compute_covariance_matrix(returns).to_numpy()

    def objective(w: np.ndarray) -> float:
        return float(w @ cov @ w)

    bounds = [(min_weight, max_weight) for _ in range(n_assets)]
    constraints = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},)
    initial = np.full(n_assets, 1.0 / n_assets, dtype=float)

    result = minimize(objective, initial, method="SLSQP", bounds=bounds, constraints=constraints)
    if not result.success:
        raise RuntimeError(f"Minimum-volatility optimization failed: {result.message}")

    weights = pd.Series(result.x, index=names)
    expected, volatility, sharpe = _portfolio_metrics(weights.values, returns)
    return OptimizationResult(
        method="min_volatility",
        weights=weights,
        expected_return=expected,
        volatility=volatility,
        sharpe_ratio=sharpe,
        covariance=compute_covariance_matrix(returns),
        correlation=compute_correlation_matrix(returns),
    )


def optimize_max_sharpe(
    returns: pd.DataFrame,
    min_weight: float = 0.05,
    max_weight: float = 0.5,
    risk_free_rate: float = 0.02,
) -> OptimizationResult:
    """Maximize the Sharpe ratio under weight constraints."""
    names = list(returns.columns)
    n_assets = len(names)
    mu = compute_expected_returns(returns).to_numpy()
    cov = compute_covariance_matrix(returns).to_numpy()

    def neg_sharpe(weights: np.ndarray) -> float:
        w = np.asarray(weights, dtype=float)
        portfolio_return = float(mu @ w)
        portfolio_vol = float(np.sqrt(w @ cov @ w))
        if portfolio_vol == 0:
            return 0.0
        return float(-(portfolio_return - risk_free_rate) / portfolio_vol)

    bounds = [(min_weight, max_weight) for _ in range(n_assets)]
    constraints = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},)
    initial = np.full(n_assets, 1.0 / n_assets, dtype=float)

    result = minimize(neg_sharpe, initial, method="SLSQP", bounds=bounds, constraints=constraints)
    if not result.success:
        raise RuntimeError(f"Max-Sharpe optimization failed: {result.message}")

    weights = pd.Series(result.x, index=names)
    expected, volatility, sharpe = _portfolio_metrics(weights.values, returns, risk_free_rate=risk_free_rate)
    return OptimizationResult(
        method="max_sharpe",
        weights=weights,
        expected_return=expected,
        volatility=volatility,
        sharpe_ratio=sharpe,
        covariance=compute_covariance_matrix(returns),
        correlation=compute_correlation_matrix(returns),
    )


def optimize_risk_parity(
    returns: pd.DataFrame,
    min_weight: float = 0.05,
    max_weight: float = 0.5,
) -> OptimizationResult:
    """Allocate weights inversely proportional to each asset's volatility."""
    names = list(returns.columns)
    covariance = compute_covariance_matrix(returns)
    volatilities = np.sqrt(np.diag(covariance.to_numpy()))
    inv_vol = 1.0 / np.clip(volatilities, 1e-8, None)
    weights = np.asarray(inv_vol, dtype=float)
    weights = weights / weights.sum()
    weights = np.clip(weights, min_weight, max_weight)
    total = weights.sum()
    weights = weights / total
    weights_series = pd.Series(weights, index=names)
    expected, volatility, sharpe = _portfolio_metrics(weights_series.values, returns)
    return OptimizationResult(
        method="risk_parity",
        weights=weights_series,
        expected_return=expected,
        volatility=volatility,
        sharpe_ratio=sharpe,
        covariance=covariance,
        correlation=compute_correlation_matrix(returns),
    )


def optimize_portfolio(
    returns: pd.DataFrame,
    method: str,
    min_weight: float = 0.05,
    max_weight: float = 0.5,
    risk_free_rate: float = 0.02,
) -> OptimizationResult:
    """Dispatch to a supported optimization method."""
    if method == "equal_weight":
        return optimize_equal_weight(returns)
    if method == "min_volatility":
        return optimize_min_volatility(returns, min_weight=min_weight, max_weight=max_weight)
    if method == "max_sharpe":
        return optimize_max_sharpe(returns, min_weight=min_weight, max_weight=max_weight, risk_free_rate=risk_free_rate)
    if method == "risk_parity":
        return optimize_risk_parity(returns, min_weight=min_weight, max_weight=max_weight)
    available = ["equal_weight", "min_volatility", "max_sharpe", "risk_parity"]
    raise ValueError(f"Unsupported method. Available: {available}")
