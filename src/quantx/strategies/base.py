from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class StrategyResult:
    signals: pd.Series
    positions: pd.Series
    strategy_name: str


class BaseStrategy:
    """Base interface for all quant strategies."""

    def __init__(self, fast_window: int = 20, slow_window: int = 50, transaction_cost: float = 0.0005):
        self.fast_window = fast_window
        self.slow_window = slow_window
        self.transaction_cost = transaction_cost

    def generate_signals(self, frame: pd.DataFrame) -> pd.Series:
        raise NotImplementedError("Subclasses must implement generate_signals().")

    def generate_positions(self, frame: pd.DataFrame) -> pd.Series:
        signals = self.generate_signals(frame)
        positions = signals.shift(1).fillna(0.0)
        return positions

    def calculate_returns(self, frame: pd.DataFrame) -> pd.Series:
        positions = self.generate_positions(frame)
        returns = frame["close"].pct_change().fillna(0.0)
        strategy_returns = returns * positions
        return strategy_returns - self.transaction_cost * (positions.diff().abs().fillna(0.0))


def signal_from_moving_average_crossover(
    frame: pd.DataFrame,
    fast_window: int = 20,
    slow_window: int = 50,
) -> pd.Series:
    """Create a simple moving-average crossover signal with a one-bar lag."""
    if frame.empty:
        return pd.Series(dtype=float)

    short_ma = frame["close"].rolling(window=fast_window, min_periods=fast_window).mean()
    long_ma = frame["close"].rolling(window=slow_window, min_periods=slow_window).mean()
    signal = (short_ma > long_ma).astype(float)
    signal = signal.where(signal != 0, -1.0)
    return signal.shift(1).fillna(0.0)


class MovingAverageCrossoverStrategy(BaseStrategy):
    def generate_signals(self, frame: pd.DataFrame) -> pd.Series:
        return signal_from_moving_average_crossover(frame, self.fast_window, self.slow_window)


class MeanReversionStrategy(BaseStrategy):
    def generate_signals(self, frame: pd.DataFrame) -> pd.Series:
        close = frame["close"]
        rolling_mean = close.rolling(window=self.fast_window, min_periods=self.fast_window).mean()
        signal = (close < rolling_mean * 0.995).astype(float)
        signal = signal - (close > rolling_mean * 1.005).astype(float)
        return signal.shift(1).fillna(0.0)


class MomentumStrategy(BaseStrategy):
    def generate_signals(self, frame: pd.DataFrame) -> pd.Series:
        returns = frame["close"].pct_change(self.fast_window).fillna(0.0)
        signal = (returns > 0).astype(float)
        signal = signal - (returns < 0).astype(float)
        return signal.shift(1).fillna(0.0)
