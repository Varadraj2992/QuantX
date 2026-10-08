import pandas as pd

from quantx.data.ingestion import generate_sample_market_data
from quantx.data.validation import validate_market_data_frame


def test_generate_sample_market_data() -> None:
    data = generate_sample_market_data(
        tickers=["AAPL", "MSFT"],
        start_date="2024-01-01",
        end_date="2024-01-05",
    )

    assert list(data.columns) == [
        "ticker",
        "date",
        "open",
        "high",
        "low",
        "close",
        "adj_close",
        "volume",
    ]
    assert not data.empty
    assert data["ticker"].nunique() == 2


def test_validate_market_data_frame_rejects_bad_rows() -> None:
    valid = pd.DataFrame(
        {
            "ticker": ["AAPL", "AAPL"],
            "date": ["2024-01-01", "2024-01-02"],
            "open": [100.0, 101.0],
            "high": [105.0, 106.0],
            "low": [99.0, 100.0],
            "close": [103.0, 104.0],
            "adj_close": [103.0, 104.0],
            "volume": [1000, 1200],
        }
    )

    result = validate_market_data_frame(valid)
    assert result.valid is True

    invalid = valid.copy()
    invalid.loc[0, "close"] = -50
    invalid_result = validate_market_data_frame(invalid)
    assert invalid_result.valid is False
    assert any("Non-positive values" in issue for issue in invalid_result.issues)
