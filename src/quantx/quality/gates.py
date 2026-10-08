from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np
import pandas as pd


@dataclass
class QualityReport:
    """Structured summary of a dataset or portfolio quality check."""

    passes: bool
    missing_values: int = 0
    duplicate_rows: int = 0
    invalid_prices: int = 0
    warnings: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)


_REQUIRED_MARKET_COLUMNS = {
    "ticker",
    "date",
    "open",
    "high",
    "low",
    "close",
    "adj_close",
    "volume",
}


def evaluate_market_data_quality(frame: pd.DataFrame) -> QualityReport:
    """Validate market data quality before research or reporting use."""
    if frame.empty:
        return QualityReport(passes=False, warnings=["Market data is empty."], details={"empty": True})

    missing_columns = sorted(_REQUIRED_MARKET_COLUMNS.difference(frame.columns))
    warnings: list[str] = []
    if missing_columns:
        warnings.append(f"Missing required columns: {missing_columns}")

    missing_values = int(frame.isna().sum().sum())
    duplicate_rows = int(frame.duplicated(subset=["ticker", "date"]).sum())

    numeric_columns = ["open", "high", "low", "close", "adj_close", "volume"]
    invalid_prices = 0
    for column in numeric_columns:
        if column in frame.columns:
            invalid_prices += int((frame[column].to_numpy() <= 0).sum())

    if invalid_prices:
        warnings.append(f"Non-positive values found in numeric market columns: {invalid_prices}")

    if missing_values:
        warnings.append(f"Missing values detected: {missing_values}")
    if duplicate_rows:
        warnings.append(f"Duplicate ticker/date rows detected: {duplicate_rows}")

    if not pd.to_datetime(frame["date"], errors="coerce").notna().all():
        warnings.append("Date column contains invalid timestamps.")

    passes = not warnings
    return QualityReport(
        passes=passes,
        missing_values=missing_values,
        duplicate_rows=duplicate_rows,
        invalid_prices=invalid_prices,
        warnings=warnings,
        details={
            "required_columns": sorted(frame.columns),
            "missing_columns": missing_columns,
            "row_count": int(len(frame)),
            "tickers": sorted(frame["ticker"].dropna().unique().tolist()) if "ticker" in frame.columns else [],
        },
    )


def evaluate_portfolio_weights(weights: Iterable[float] | pd.Series) -> QualityReport:
    """Check that portfolio weights are normalized and within practical bounds."""
    values = pd.Series(weights, dtype=float)
    if values.empty:
        return QualityReport(passes=False, warnings=["Portfolio weights are empty."], details={"weight_total": 0.0})

    warnings: list[str] = []
    if np.any(values < 0):
        warnings.append("Weights contain negative allocations.")
    if np.any(values > 1.0):
        warnings.append("Weights exceed 100% allocation.")

    total = float(values.sum())
    if not np.isclose(total, 1.0, atol=1e-6):
        warnings.append(f"Portfolio weights do not sum to 1.0; total={total:.6f}")

    passes = not warnings
    return QualityReport(
        passes=passes,
        missing_values=0,
        duplicate_rows=0,
        invalid_prices=0,
        warnings=warnings,
        details={"weight_total": total, "min_weight": float(values.min()), "max_weight": float(values.max())},
    )


def fail_if_quality_issues(report: QualityReport, message: str = "Quality gate failed.") -> None:
    """Raise a ValueError when a dataset or portfolio does not meet the minimum quality bar."""
    if not report.passes:
        details = "; ".join(report.warnings) if report.warnings else message
        raise ValueError(f"{message} {details}")
