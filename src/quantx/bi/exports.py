from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from quantx.data.ingestion import generate_sample_market_data
from quantx.features.engineering import add_technical_indicators, compute_macd

_DEFAULT_SECTORS: dict[str, str] = {
    "AAPL": "Technology",
    "MSFT": "Technology",
    "NVDA": "Semiconductors",
    "AMZN": "Consumer Discretionary",
    "SPY": "Benchmark",
}


def build_dimension_tables(
    market_data: pd.DataFrame | None = None,
    sectors: Mapping[str, str] | None = None,
    tickers: list[str] | None = None,
    start_date: str = "2024-01-01",
    end_date: str = "2024-06-30",
) -> dict[str, pd.DataFrame]:
    """Create dimensional tables suitable for a Power BI semantic model."""
    if market_data is None:
        market_data = generate_sample_market_data(
            tickers=tickers or ["AAPL", "MSFT", "NVDA", "AMZN", "SPY"],
            start_date=start_date,
            end_date=end_date,
        )

    asset_frame = market_data[["ticker"]].drop_duplicates().copy().sort_values("ticker").reset_index(drop=True)
    asset_frame["ticker"] = asset_frame["ticker"].astype(str)
    asset_frame["sector"] = asset_frame["ticker"].map(sectors or _DEFAULT_SECTORS).fillna("Unknown")
    asset_frame["asset_name"] = asset_frame["ticker"]
    asset_frame["market_segment"] = asset_frame["sector"]
    asset_frame["is_benchmark"] = asset_frame["ticker"].eq("SPY")

    date_frame = pd.DataFrame({
        "date": pd.to_datetime(market_data["date"]).dt.normalize().drop_duplicates().sort_values(),
    })
    date_frame["year"] = date_frame["date"].dt.year
    date_frame["month"] = date_frame["date"].dt.month
    date_frame["month_name"] = date_frame["date"].dt.strftime("%b")
    date_frame["quarter"] = date_frame["date"].dt.quarter
    date_frame["week_of_year"] = date_frame["date"].dt.isocalendar().week.astype(int)

    return {
        "dim_asset": asset_frame,
        "dim_date": date_frame,
    }


def build_fact_tables(
    market_data: pd.DataFrame | None = None,
    weights: Mapping[str, float] | pd.Series | None = None,
    tickers: list[str] | None = None,
    start_date: str = "2024-01-01",
    end_date: str = "2024-06-30",
) -> dict[str, pd.DataFrame]:
    """Create fact tables representing daily market observations and portfolio weights."""
    if market_data is None:
        market_data = generate_sample_market_data(
            tickers=tickers or ["AAPL", "MSFT", "NVDA", "AMZN", "SPY"],
            start_date=start_date,
            end_date=end_date,
        )

    enriched = add_technical_indicators(market_data.copy())
    enriched["date"] = pd.to_datetime(enriched["date"]).dt.normalize()
    enriched["return_1d"] = (
        enriched.groupby("ticker")["close"]
        .pct_change()
        .replace([float("inf"), float("-inf")], np.nan)
    )
    enriched["return_1d"] = enriched["return_1d"].fillna(0.0)

    if "moving_average_5" not in enriched.columns:
        enriched["moving_average_5"] = enriched.groupby("ticker")["close"].transform(
            lambda x: x.rolling(window=5, min_periods=5).mean()
        )
    if "moving_average_10" not in enriched.columns:
        enriched["moving_average_10"] = enriched.groupby("ticker")["close"].transform(
            lambda x: x.rolling(window=10, min_periods=10).mean()
        )
    if "rolling_volatility_10" not in enriched.columns:
        enriched["rolling_volatility_10"] = enriched.groupby("ticker")["close"].transform(
            lambda x: x.pct_change().fillna(0.0).rolling(window=10, min_periods=10).std().mul(100)
        )
    if "rsi_14" not in enriched.columns:
        enriched["rsi_14"] = enriched.groupby("ticker")["close"].transform(
            lambda x: x.pct_change().fillna(0.0).rolling(window=14, min_periods=14).std()
        )
    if "macd" not in enriched.columns:
        macd_line, _, _ = compute_macd(enriched["close"])
        enriched["macd"] = macd_line

    enriched["signal_direction"] = np.where(
        enriched["moving_average_5"].fillna(0.0) >= enriched["moving_average_10"].fillna(0.0),
        "Bullish",
        "Bearish",
    )

    if weights is not None:
        if isinstance(weights, pd.Series):
            weight_map = weights.to_dict()
        else:
            weight_map = dict(weights)
        enriched["portfolio_weight"] = enriched["ticker"].map(weight_map).fillna(0.0)
    else:
        enriched["portfolio_weight"] = 0.0

    market_fact = enriched[[
        "date",
        "ticker",
        "open",
        "high",
        "low",
        "close",
        "adj_close",
        "volume",
        "return_1d",
        "moving_average_5",
        "moving_average_10",
        "rolling_volatility_10",
        "rsi_14",
        "macd",
        "signal_direction",
        "portfolio_weight",
    ]].copy()
    market_fact = market_fact.sort_values(["ticker", "date"]).reset_index(drop=True)

    portfolio_fact = (
        market_fact[["date", "ticker", "close", "return_1d", "portfolio_weight"]]
        .copy()
        .rename(columns={"close": "close_price", "return_1d": "daily_return"})
    )

    return {
        "fact_market_data": market_fact,
        "fact_portfolio_metrics": portfolio_fact,
    }


def build_power_bi_dataset(
    market_data: pd.DataFrame | None = None,
    weights: Mapping[str, float] | pd.Series | None = None,
    sectors: Mapping[str, str] | None = None,
    tickers: list[str] | None = None,
    start_date: str = "2024-01-01",
    end_date: str = "2024-06-30",
) -> pd.DataFrame:
    """Combine fact and dimension tables into a single analyst-friendly dataset."""
    fact_tables = build_fact_tables(
        market_data=market_data,
        weights=weights,
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
    )
    dim_tables = build_dimension_tables(
        market_data=fact_tables["fact_market_data"],
        sectors=sectors,
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
    )

    market_fact = fact_tables["fact_market_data"].copy()
    market_fact = market_fact.merge(dim_tables["dim_asset"], on="ticker", how="left")
    market_fact = market_fact.merge(dim_tables["dim_date"], on="date", how="left")
    return market_fact.sort_values(["ticker", "date"]).reset_index(drop=True)


def build_power_bi_tables(
    market_data: pd.DataFrame | None = None,
    weights: Mapping[str, float] | pd.Series | None = None,
    sectors: Mapping[str, str] | None = None,
    tickers: list[str] | None = None,
    start_date: str = "2024-01-01",
    end_date: str = "2024-06-30",
) -> dict[str, pd.DataFrame]:
    """Return the full Power BI model: dim + fact tables ready for export."""
    fact_tables = build_fact_tables(
        market_data=market_data,
        weights=weights,
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
    )
    dim_tables = build_dimension_tables(
        market_data=fact_tables["fact_market_data"],
        sectors=sectors,
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
    )
    dataset = build_power_bi_dataset(
        market_data=fact_tables["fact_market_data"],
        weights=weights,
        sectors=sectors,
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
    )

    return {
        "fact_market_data": fact_tables["fact_market_data"],
        "fact_portfolio_metrics": fact_tables["fact_portfolio_metrics"],
        "dim_asset": dim_tables["dim_asset"],
        "dim_date": dim_tables["dim_date"],
        "dataset": dataset,
    }


def export_power_bi_dataset(
    path: str | Path,
    market_data: pd.DataFrame | None = None,
    weights: Mapping[str, float] | pd.Series | None = None,
    sectors: Mapping[str, str] | None = None,
    tickers: list[str] | None = None,
    start_date: str = "2024-01-01",
    end_date: str = "2024-06-30",
) -> pd.DataFrame:
    """Write a flattened dataset to a CSV that can be imported into Power BI."""
    dataset = build_power_bi_dataset(
        market_data=market_data,
        weights=weights,
        sectors=sectors,
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
    )
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(output_path, index=False)
    return dataset


__all__ = [
    "build_dimension_tables",
    "build_fact_tables",
    "build_power_bi_dataset",
    "build_power_bi_tables",
    "export_power_bi_dataset",
]
