"""Machine learning signal engine for QuantX."""

from .engine import (
    MLSignalResult,
    build_signal_dataset,
    train_classifier,
    train_model_suite,
)

__all__ = [
    "MLSignalResult",
    "build_signal_dataset",
    "train_classifier",
    "train_model_suite",
]
