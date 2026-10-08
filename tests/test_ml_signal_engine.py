
from quantx.data.ingestion import generate_sample_market_data
from quantx.features.engineering import add_technical_indicators
from quantx.ml.engine import build_signal_dataset, train_classifier, train_model_suite


def test_build_signal_dataset_creates_labels() -> None:
    frame = generate_sample_market_data(
        tickers=["AAPL", "MSFT"],
        start_date="2024-01-01",
        end_date="2024-05-31",
    )
    enriched = add_technical_indicators(frame)
    dataset = build_signal_dataset(enriched, target_window=3)
    assert {"target"}.issubset(set(dataset.columns))
    assert not dataset.empty
    assert dataset["target"].nunique() >= 2


def test_train_classifier_returns_metrics() -> None:
    frame = generate_sample_market_data(
        tickers=["AAPL", "MSFT"],
        start_date="2024-01-01",
        end_date="2024-06-30",
    )
    enriched = add_technical_indicators(frame)
    dataset = build_signal_dataset(enriched, target_window=3)
    result = train_classifier(dataset.drop(columns=["target"]), dataset["target"], model_name="logistic_regression")
    assert result.accuracy >= 0
    assert result.precision >= 0
    assert result.recall >= 0
    assert result.f1 >= 0


def test_train_model_suite_runs_multiple_models() -> None:
    frame = generate_sample_market_data(
        tickers=["AAPL"],
        start_date="2024-01-01",
        end_date="2024-06-30",
    )
    enriched = add_technical_indicators(frame)
    results = train_model_suite(enriched, target_window=2)
    assert set(results).issuperset({"logistic_regression", "random_forest"})
