"""Quant strategy framework for QuantX."""

from .base import BaseStrategy, StrategyResult, signal_from_moving_average_crossover

__all__ = [
    "BaseStrategy",
    "StrategyResult",
    "signal_from_moving_average_crossover",
]
