from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd


@dataclass
class RiskMetrics:
    """Container for a portfolio risk snapshot."""

    volatility: float
    beta: float
    var_95: float
    cvar_95: float
    max_drawdown: float
    concentration: float
    correlation_risk: float
    sector_exposure: pd.Series
    portfolio_exposure: pd.Series
    risk_contribution: pd.Series
    stress_results: dict[str, float]


def compute_volatility(returns: pd.Series | pd.DataFrame, annualization: int = 252) -> float:
    """Compute annualized volatility from daily returns."""
    if isinstance(returns, pd.DataFrame):
        if returns.empty:
            raise ValueError("Returns DataFrame cannot be empty.")
        series = returns.mean(axis=1)
    else:
        series = returns
    if series.empty:
        raise ValueError("Returns series cannot be empty.")
    std = series.std(ddof=1)
    return float(std * np.sqrt(annualization))


def compute_beta(asset_returns: pd.Series, benchmark_returns: pd.Series) -> float:
    """Estimate beta against a benchmark series with naive covariance-based formula."""
    if asset_returns.empty or benchmark_returns.empty:
        raise ValueError("Asset and benchmark returns cannot be empty.")
    if len(asset_returns) != len(benchmark_returns):
        raise ValueError("Asset and benchmark returns must be aligned by length.")
    cov = np.cov(asset_returns.to_numpy(), benchmark_returns.to_numpy(), ddof=1)
    variance = float(cov[1, 1])
    if variance == 0:
        return 0.0
    return float(cov[0, 1] / variance)


def compute_var(series: pd.Series, confidence: float = 0.95) -> float:
    """Compute Value at Risk for a return series at the specified confidence level."""
    if series.empty:
        raise ValueError("Series cannot be empty.")
    quantile = float(np.quantile(series.to_numpy(), 1.0 - confidence))
    return float(abs(quantile))


def compute_cvar(series: pd.Series, confidence: float = 0.95) -> float:
    """Compute Conditional Value at Risk (Expected Shortfall)."""
    if series.empty:
        raise ValueError("Series cannot be empty.")
    tail = series[series <= np.quantile(series.to_numpy(), 1.0 - confidence)]
    if tail.empty:
        return 0.0
    return float(abs(tail.mean()))


def compute_drawdown(equity_curve: pd.Series) -> float:
    """Compute max drawdown percentage for an equity series."""
    if equity_curve.empty:
        raise ValueError("Equity curve cannot be empty.")
    running_max = equity_curve.cummax()
    drawdowns = (equity_curve / running_max - 1.0) * 100.0
    return float(abs(drawdowns.min()))


def asset_contribution_to_risk(weights: pd.Series, covariance: pd.DataFrame) -> pd.Series:
    """Compute marginal risk contribution of each asset to total portfolio variance."""
    if weights.empty or covariance.empty:
        raise ValueError("Weights and covariance matrix cannot be empty.")
    w = weights.reindex(covariance.columns).to_numpy(dtype=float)
    sigma = np.sqrt(float(w @ covariance.to_numpy() @ w))
    if sigma == 0:
        return pd.Series(0.0, index=weights.index)
    contribution = (covariance.to_numpy() @ w) * w / sigma
    return pd.Series(contribution, index=weights.index)


def sector_exposure(weights: pd.Series, sectors: Mapping[str, str]) -> pd.Series:
    """Aggregate portfolio weights by economic sector."""
    if weights.empty:
        raise ValueError("Weights cannot be empty.")
    grouped = pd.Series(0.0, index=pd.Index(sorted(set(sectors.values()))))
    for ticker, weight in weights.items():
        sector_name = sectors.get(str(ticker), "Unassigned")
        grouped[sector_name] = grouped.get(sector_name, 0.0) + float(weight)
    return grouped


def concentration_risk(weights: pd.Series) -> float:
    """Measure concentration using the Herfindahl-Hirschman Index."""
    if weights.empty:
        raise ValueError("Weights cannot be empty.")
    normalized = weights / weights.sum()
    return float(np.sum(np.square(normalized.to_numpy(dtype=float))))


def portfolio_exposure(weights: pd.Series) -> pd.Series:
    """Return the asset allocation profile in descending portfolio weight order."""
    if weights.empty:
        raise ValueError("Weights cannot be empty.")
    return weights.sort_values(ascending=False)


def correlation_risk(correlation: pd.DataFrame) -> float:
    """Summarize the average absolute off-diagonal correlation in the portfolio."""
    if correlation.empty:
        raise ValueError("Correlation matrix cannot be empty.")
    matrix = correlation.to_numpy()
    mask = ~np.eye(matrix.shape[0], dtype=bool)
    average_corr = np.abs(matrix[mask]).mean()
    return float(average_corr)


def stress_test_portfolio(
    portfolio_returns: pd.Series,
    scenarios: Mapping[str, float] | None = None,
) -> dict[str, float]:
    """Apply stylized market stress scenarios to a portfolio return series."""
    if portfolio_returns.empty:
        raise ValueError("Portfolio returns cannot be empty.")
    if scenarios is None:
        scenarios = {
            "market_crash": -0.25,
            "high_volatility": -0.12,
            "sector_shock": -0.18,
            "rate_shock": -0.08,
        }

    base_return = float(portfolio_returns.mean())
    results: dict[str, float] = {}
    for name, shock in scenarios.items():
        results[name] = float(base_return + shock)
    return results


def portfolio_risk_snapshot(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
    weights: pd.Series,
    covariance: pd.DataFrame,
    correlation: pd.DataFrame,
    sectors: Mapping[str, str],
) -> RiskMetrics:
    """Create a risk snapshot for a portfolio and asset set."""
    risk_contrib = asset_contribution_to_risk(weights, covariance)
    metrics = RiskMetrics(
        volatility=compute_volatility(portfolio_returns),
        beta=compute_beta(portfolio_returns, benchmark_returns),
        var_95=compute_var(portfolio_returns),
        cvar_95=compute_cvar(portfolio_returns),
        max_drawdown=compute_drawdown(pd.Series(np.cumprod(1.0 + portfolio_returns), index=portfolio_returns.index)),
        concentration=concentration_risk(weights),
        correlation_risk=correlation_risk(correlation),
        sector_exposure=sector_exposure(weights, sectors),
        portfolio_exposure=portfolio_exposure(weights),
        risk_contribution=risk_contrib,
        stress_results=stress_test_portfolio(portfolio_returns),
    )
    return metrics
