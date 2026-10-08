from __future__ import annotations

import pandas as pd
import pytest

from quantx.data.ingestion import generate_sample_market_data
from quantx.quality.gates import evaluate_market_data_quality, evaluate_portfolio_weights, fail_if_quality_issues


def test_evaluate_market_data_quality_accepts_clean_dataset() -> None:
    frame = generate_sample_market_data(tickers=["AAPL", "MSFT"], start_date="2024-01-01", end_date="2024-01-10")

    result = evaluate_market_data_quality(frame)

    assert result.passes is True
    assert result.missing_values == 0
    assert result.duplicate_rows == 0
    assert result.warnings == []


def test_evaluate_market_data_quality_flags_bad_dataset() -> None:
    frame = pd.DataFrame({
        "ticker": ["AAPL", "AAPL"],
        "date": ["2024-01-01", "2024-01-01"],
        "open": [100.0, 95.0],
        "high": [110.0, 105.0],
        "low": [90.0, 85.0],
        "close": [105.0, 100.0],
        "adj_close": [105.0, 100.0],
        "volume": [1_000_000, None],
    })

    result = evaluate_market_data_quality(frame)

    assert result.passes is False
    assert result.duplicate_rows >= 1
    assert result.missing_values >= 1


def test_evaluate_portfolio_weights_rejects_unbalanced_allocation() -> None:
    result = evaluate_portfolio_weights([0.7, 0.2])

    assert result.passes is False
    assert any("sum to 1.0" in warning for warning in result.warnings)

    with pytest.raises(ValueError):
        fail_if_quality_issues(result)
