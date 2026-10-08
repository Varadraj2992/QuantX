
from quantx.data.ingestion import generate_sample_market_data
from quantx.strategies.base import (
    MeanReversionStrategy,
    MomentumStrategy,
    MovingAverageCrossoverStrategy,
)


def test_moving_average_strategy_generates_positions() -> None:
    frame = generate_sample_market_data(
        tickers=["AAPL"],
        start_date="2024-01-01",
        end_date="2024-02-15",
    )
    strategy = MovingAverageCrossoverStrategy(fast_window=5, slow_window=10)
    signal = strategy.generate_signals(frame)
    position = strategy.generate_positions(frame)
    assert len(signal) == len(frame)
    assert position.notna().all()
    assert set(signal.unique()).issubset({-1.0, 0.0, 1.0})


def test_strategy_returns_are_numeric() -> None:
    frame = generate_sample_market_data(
        tickers=["MSFT"],
        start_date="2024-01-01",
        end_date="2024-02-15",
    )
    strategy = MomentumStrategy(fast_window=5)
    result = strategy.calculate_returns(frame)
    assert result.notna().all()
    assert result.dtype.kind in {"f", "i"}


def test_mean_reversion_strategy_is_stable() -> None:
    frame = generate_sample_market_data(
        tickers=["NVDA"],
        start_date="2024-01-01",
        end_date="2024-02-15",
    )
    strategy = MeanReversionStrategy(fast_window=5)
    signal = strategy.generate_signals(frame)
    assert signal.notna().all()
