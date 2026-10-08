from __future__ import annotations

from pathlib import Path

import pandas as pd

from quantx.bi import build_power_bi_dataset, build_power_bi_tables, export_power_bi_dataset
from quantx.data.ingestion import generate_sample_market_data


def test_build_power_bi_dataset_creates_analytical_table() -> None:
    frame = generate_sample_market_data(tickers=["AAPL", "MSFT"], start_date="2024-01-01", end_date="2024-01-31")
    dataset = build_power_bi_dataset(frame, weights={"AAPL": 0.6, "MSFT": 0.4})

    assert isinstance(dataset, pd.DataFrame)
    assert {"date", "ticker", "sector", "is_benchmark"}.issubset(dataset.columns)
    assert dataset["ticker"].nunique() == 2
    assert dataset["sector"].notna().all()


def test_build_power_bi_tables_exposes_dim_and_fact_tables() -> None:
    frame = generate_sample_market_data(tickers=["AAPL"], start_date="2024-01-01", end_date="2024-01-10")
    tables = build_power_bi_tables(frame)

    assert set(tables).issuperset({"dim_asset", "dim_date", "fact_market_data", "fact_portfolio_metrics", "dataset"})
    assert not tables["dim_asset"].empty
    assert not tables["fact_market_data"].empty
    assert "return_1d" in tables["fact_market_data"].columns


def test_export_power_bi_dataset_writes_csv(tmp_path: Path) -> None:
    frame = generate_sample_market_data(tickers=["AAPL"], start_date="2024-01-01", end_date="2024-01-05")
    output = tmp_path / "exports" / "power_bi_dataset.csv"

    exported = export_power_bi_dataset(output, frame)

    assert output.exists()
    assert not exported.empty
    assert output.read_text(encoding="utf-8").startswith("date")
