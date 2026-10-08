import pandas as pd

from quantx.backtesting.engine import Backtester
from quantx.data.ingestion import generate_sample_market_data
from quantx.strategies.base import MovingAverageCrossoverStrategy


def test_backtester_returns_equity_and_metrics() -> None:
    frame = generate_sample_market_data(
        tickers=["AAPL"],
        start_date="2024-01-01",
        end_date="2024-03-31",
    )
    strategy = MovingAverageCrossoverStrategy(fast_window=5, slow_window=10)
    backtester = Backtester(initial_capital=100000.0, transaction_cost=0.0005)
    result = backtester.run(frame, strategy)

    assert len(result.equity_curve) == len(frame)
    assert len(result.drawdown_curve) == len(frame)
    assert result.metrics["final_portfolio_value"] > 0
    assert "final_portfolio_value" in result.metrics
    assert result.metrics["max_drawdown"] >= 0
    assert result.metrics["num_trades"] >= 0
    assert result.monthly_returns.empty or not result.monthly_returns["monthly_return"].isna().all()


def test_backtester_handles_empty_input() -> None:
    backtester = Backtester()
    empty = pd.DataFrame(columns=["date", "close"])
    try:
        backtester.run(empty, MovingAverageCrossoverStrategy())
        raise AssertionError("Expected ValueError for empty frame")
    except ValueError:
        pass


def test_backtester_benchmark_comparison() -> None:
    frame = generate_sample_market_data(
        tickers=["AAPL", "SPY"],
        start_date="2024-01-01",
        end_date="2024-03-31",
    )
    strategy = MovingAverageCrossoverStrategy(fast_window=5, slow_window=10)
    comparison = Backtester(initial_capital=100000.0, transaction_cost=0.0005).compare_with_benchmark(
        frame,
        strategy,
        benchmark_name="SPY",
    )

    assert comparison["benchmark_name"] == "SPY"
    assert "strategy_metrics" in comparison
    assert "benchmark_metrics" in comparison
    assert "excess_return" in comparison["benchmark_metrics"]
    assert "strategy_total_return" in comparison["benchmark_metrics"]
