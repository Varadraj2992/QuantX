from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

REQUIRED_COLUMNS = {
    "ticker",
    "date",
    "open",
    "high",
    "low",
    "close",
    "adj_close",
    "volume",
}


@dataclass
class DataValidationResult:
    """Container for validation output from a market-data DataFrame."""

    valid: bool
    issues: list[str] = field(default_factory=list)
    cleaned_data: pd.DataFrame | None = None


def validate_market_data_frame(frame: pd.DataFrame) -> DataValidationResult:
    """Validate and normalize a market-data DataFrame before ingestion.

    The function intentionally fails fast on clear structural problems so that bad
    data is not silently accepted into the research pipeline.
    """

    issues: list[str] = []
    cleaned = frame.copy()

    if cleaned.empty:
        return DataValidationResult(valid=False, issues=["DataFrame is empty."], cleaned_data=cleaned)

    cleaned.columns = [str(column).strip().lower() for column in cleaned.columns]

    missing_columns = sorted(REQUIRED_COLUMNS - set(cleaned.columns))
    if missing_columns:
        issues.append(f"Missing required columns: {', '.join(missing_columns)}")

    if not missing_columns:
        cleaned["date"] = pd.to_datetime(cleaned["date"], errors="coerce")
        if cleaned["date"].isna().any():
            issues.append("At least one record has an invalid date.")

        cleaned["ticker"] = cleaned["ticker"].astype(str).str.strip()
        cleaned["volume"] = pd.to_numeric(cleaned["volume"], errors="coerce")
        for column in ["open", "high", "low", "close", "adj_close"]:
            cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")

        if cleaned[["open", "high", "low", "close", "adj_close"]].isna().any().any():
            issues.append("One or more price columns contain missing values.")

        if cleaned["volume"].isna().any():
            issues.append("One or more volume values are missing or non-numeric.")

        duplicates = cleaned.duplicated(subset=["ticker", "date"], keep=False)
        if duplicates.any():
            issues.append(f"Duplicate ticker/date combinations found: {int(duplicates.sum())} rows.")

        if not cleaned["ticker"].str.len().gt(0).all():
            issues.append("One or more tickers are empty strings.")

        if (cleaned["high"] < cleaned["low"]).any():
            issues.append("At least one row has high < low.")

        for column in ["open", "high", "low", "close", "adj_close"]:
            if (cleaned[column] <= 0).any():
                issues.append(f"Non-positive values found in {column}.")

        if len(cleaned) > 1:
            daily_returns = cleaned.sort_values("date").groupby("ticker")["close"].pct_change()
            if daily_returns.abs().gt(5).any():
                issues.append("Extreme price moves detected; outlier screening flagged suspicious values.")

    valid = not issues
    if valid:
        cleaned = cleaned.sort_values(["ticker", "date"]).reset_index(drop=True)
    return DataValidationResult(valid=valid, issues=issues, cleaned_data=cleaned if valid else cleaned)


def ensure_required_columns(frame: pd.DataFrame) -> None:
    """Validate the presence of essential columns for market-data ingestion."""
    missing = sorted(REQUIRED_COLUMNS - set(str(c).strip().lower() for c in frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")
