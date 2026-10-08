from app.api.main import app
from fastapi.testclient import TestClient

client = TestClient(app)


def test_health_endpoint() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_assets_endpoint() -> None:
    response = client.get("/assets")
    assert response.status_code == 200
    payload = response.json()
    assert len(payload) >= 3
    assert payload[0]["ticker"]


def test_strategy_and_backtest_endpoints() -> None:
    strategy_payload = {
        "ticker": "AAPL",
        "strategy": "moving_average",
        "start_date": "2024-01-01",
        "end_date": "2024-06-30",
        "capital": 100000.0,
        "fast_window": 5,
        "slow_window": 10,
    }
    strategy_response = client.post("/strategy/run", json=strategy_payload)
    assert strategy_response.status_code == 200
    assert strategy_response.json()["strategy"] == "moving_average"

    backtest_response = client.post("/backtest", json=strategy_payload)
    assert backtest_response.status_code == 200
    assert "id" in backtest_response.json()


def test_portfolio_and_risk_endpoints() -> None:
    portfolio_response = client.get("/portfolio")
    assert portfolio_response.status_code == 200
    assert "weights" in portfolio_response.json()

    risk_response = client.get("/risk")
    assert risk_response.status_code == 200
    assert risk_response.json()["volatility"] >= 0


def test_ml_predict_endpoint() -> None:
    response = client.post("/ml/predict", json={"ticker": "AAPL", "start_date": "2024-01-01", "end_date": "2024-06-30"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["ticker"] == "AAPL"
    assert payload["signal"] in {"BUY", "HOLD", "SELL"}


def test_benchmark_endpoint() -> None:
    response = client.post(
        "/benchmark",
        json={
            "ticker": "AAPL",
            "strategy": "moving_average",
            "start_date": "2024-01-01",
            "end_date": "2024-06-30",
            "capital": 100000.0,
            "fast_window": 5,
            "slow_window": 10,
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["benchmark_name"] == "SPY"
    assert "benchmark_metrics" in payload
    assert "excess_return" in payload["benchmark_metrics"]
