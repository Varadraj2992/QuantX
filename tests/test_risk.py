import pandas as pd

from quantx.risk.engine import (
    asset_contribution_to_risk,
    compute_beta,
    compute_cvar,
    compute_drawdown,
    compute_var,
    compute_volatility,
    concentration_risk,
    correlation_risk,
    portfolio_risk_snapshot,
    sector_exposure,
    stress_test_portfolio,
)

PORTFOLIO_RETURNS = pd.Series([0.01, -0.02, 0.03, -0.01, 0.02, 0.015, -0.03, 0.04])
BENCHMARK_RETURNS = pd.Series([0.008, -0.015, 0.025, -0.012, 0.021, 0.013, -0.02, 0.03])
COVARIANCE = pd.DataFrame(
    [[0.0004, 0.00012], [0.00012, 0.0003]],
    index=["AAPL", "MSFT"],
    columns=["AAPL", "MSFT"],
)
CORRELATION = pd.DataFrame(
    [[1.0, 0.6], [0.6, 1.0]],
    index=["AAPL", "MSFT"],
    columns=["AAPL", "MSFT"],
)
WEIGHTS = pd.Series({"AAPL": 0.6, "MSFT": 0.4})
SECTORS = {"AAPL": "Technology", "MSFT": "Technology"}


def test_compute_volatility() -> None:
    result = compute_volatility(PORTFOLIO_RETURNS)
    assert result > 0


def test_compute_beta() -> None:
    result = compute_beta(PORTFOLIO_RETURNS, BENCHMARK_RETURNS)
    assert isinstance(result, float)


def test_var_and_cvar() -> None:
    var_value = compute_var(PORTFOLIO_RETURNS)
    cvar_value = compute_cvar(PORTFOLIO_RETURNS)
    assert var_value >= 0
    assert cvar_value >= 0


def test_drawdown_and_concentration() -> None:
    equity_curve = pd.Series([100, 110, 90, 95, 80, 85], dtype=float)
    assert compute_drawdown(equity_curve) >= 0
    assert concentration_risk(WEIGHTS) > 0


def test_sector_and_correlation_and_stress() -> None:
    exposure = sector_exposure(WEIGHTS, SECTORS)
    assert exposure["Technology"] > 0
    assert correlation_risk(CORRELATION) >= 0
    stress = stress_test_portfolio(PORTFOLIO_RETURNS)
    assert set(stress).issuperset({"market_crash", "high_volatility", "sector_shock", "rate_shock"})


def test_asset_contribution_and_snapshot() -> None:
    contribution = asset_contribution_to_risk(WEIGHTS, COVARIANCE)
    assert contribution.index.tolist() == ["AAPL", "MSFT"]
    snapshot = portfolio_risk_snapshot(PORTFOLIO_RETURNS, BENCHMARK_RETURNS, WEIGHTS, COVARIANCE, CORRELATION, SECTORS)
    assert snapshot.volatility >= 0
    assert snapshot.max_drawdown >= 0
    assert snapshot.sector_exposure["Technology"] > 0
