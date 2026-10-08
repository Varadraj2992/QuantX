import pandas as pd

from quantx.data.ingestion import generate_sample_market_data
from quantx.features.engineering import add_technical_indicators, compute_rsi


def test_rsi_values_are_bounded() -> None:
    price = pd.Series([100, 101, 102, 101, 100, 99, 98, 101, 105], dtype=float)
    series = compute_rsi(price, window=5)
    assert series.notna().all()
    assert (series >= 0).all()
    assert (series <= 100).all()


def test_add_technical_indicators_adds_columns() -> None:
    frame = generate_sample_market_data(
        tickers=["AAPL", "MSFT"],
        start_date="2024-01-02",
        end_date="2024-01-10",
    )
    enriched = add_technical_indicators(frame)

    expected = {"ret_1d", "ret_5d", "vol_20d", "sma_20", "rsi_14", "macd", "macd_signal", "macd_hist"}
    assert expected.issubset(set(enriched.columns))
