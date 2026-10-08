from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .validation import validate_market_data_frame


def generate_sample_market_data(
    tickers: list[str] | None = None,
    start_date: str = "2023-01-01",
    end_date: str = "2024-12-31",
    seed: int = 42,
) -> pd.DataFrame:
    """Create synthetic market data for a local pipeline test and demonstration.

    This is intentionally not a production-grade market-data vendor implementation,
    but it is suitable for structured local testing and dashboard prototyping.
    """

    rng = np.random.default_rng(seed)
    tickers = tickers or ["AAPL", "MSFT", "NVDA", "AMZN", "SPY"]

    date_index = pd.date_range(start=start_date, end=end_date, freq="B")
    rows: list[dict[str, object]] = []

    base_prices = {ticker: 100 + idx * 10 for idx, ticker in enumerate(tickers)}
    volatility = {ticker: 0.012 + idx * 0.003 for idx, ticker in enumerate(tickers)}

    for ticker in tickers:
        drift = 0.0005 + rng.random() * 0.0015
        price = base_prices[ticker]
        for current_date in date_index:
            shock = rng.normal(0.0, volatility[ticker])
            price *= 1 + drift + shock
            open_price = price * (1 - rng.uniform(-0.005, 0.005))
            close_price = price
            high_price = max(open_price, close_price) * (1 + rng.uniform(0.0, 0.012))
            low_price = min(open_price, close_price) * (1 - rng.uniform(0.0, 0.012))
            row = {
                "ticker": ticker,
                "date": current_date.strftime("%Y-%m-%d"),
                "open": round(float(open_price), 4),
                "high": round(float(high_price), 4),
                "low": round(float(low_price), 4),
                "close": round(float(close_price), 4),
                "adj_close": round(float(close_price), 4),
                "volume": int(max(500_000, rng.integers(1_000_000, 3_000_000))),
            }
            rows.append(row)

    result = pd.DataFrame(rows)
    result["ticker"] = result["ticker"].astype(str)
    result["date"] = pd.to_datetime(result["date"])
    result = result.sort_values(["ticker", "date"]).reset_index(drop=True)
    validation = validate_market_data_frame(result)
    if not validation.valid:
        raise ValueError(f"Generated sample data failed validation: {validation.issues}")
    return validation.cleaned_data


def ingest_market_data_from_csv(path: str | Path) -> pd.DataFrame:
    """Load a CSV file and validate the data before returning a sanitized DataFrame."""
    csv_path = Path(path)
    frame = pd.read_csv(csv_path)
    validation = validate_market_data_frame(frame)
    if not validation.valid:
        raise ValueError(f"Data validation failed for {csv_path}: {validation.issues}")
    return validation.cleaned_data
