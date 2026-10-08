import numpy as np
import pandas as pd

from quantx.optimization.engine import (
    optimize_equal_weight,
    optimize_max_sharpe,
    optimize_min_volatility,
    optimize_portfolio,
    optimize_risk_parity,
)

RETURNS = pd.DataFrame(
    {
        "AAPL": [0.01, 0.02, -0.01, 0.03, 0.02, -0.02, 0.01, 0.04, 0.015, 0.005],
        "MSFT": [0.008, 0.012, 0.003, 0.011, 0.009, 0.006, -0.004, 0.015, 0.013, 0.01],
        "NVDA": [0.02, -0.015, 0.03, 0.02, -0.01, 0.025, 0.018, 0.03, -0.005, 0.02],
    }
)


def test_optimize_equal_weight() -> None:
    result = optimize_equal_weight(RETURNS)
    assert result.weights.shape[0] == RETURNS.shape[1]
    assert np.isclose(result.weights.sum(), 1.0)
    assert (result.weights >= 0).all()


def test_optimize_min_volatility() -> None:
    result = optimize_min_volatility(RETURNS)
    assert result.weights.shape[0] == RETURNS.shape[1]
    assert np.isclose(result.weights.sum(), 1.0)
    assert (result.weights >= 0).all()
    assert result.volatility >= 0.0


def test_optimize_max_sharpe() -> None:
    result = optimize_max_sharpe(RETURNS)
    assert result.weights.shape[0] == RETURNS.shape[1]
    assert np.isclose(result.weights.sum(), 1.0)
    assert (result.weights >= 0).all()
    assert result.sharpe_ratio >= -10_000


def test_optimize_risk_parity() -> None:
    result = optimize_risk_parity(RETURNS)
    assert result.weights.shape[0] == RETURNS.shape[1]
    assert np.isclose(result.weights.sum(), 1.0)
    assert (result.weights > 0).all()


def test_optimize_portfolio_dispatch() -> None:
    result = optimize_portfolio(RETURNS, method="equal_weight")
    assert result.method == "equal_weight"
    assert np.isclose(result.weights.sum(), 1.0)
