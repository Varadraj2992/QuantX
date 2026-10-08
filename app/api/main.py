from __future__ import annotations

import uuid
from typing import Literal

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from config.settings import settings
from quantx.backtesting.engine import Backtester, BenchmarkConfig, resolve_benchmark_config, resolve_benchmark_returns
from quantx.data.ingestion import generate_sample_market_data
from quantx.features.engineering import add_technical_indicators
from quantx.ml.engine import build_signal_dataset, train_model_suite
from quantx.optimization.engine import optimize_portfolio
from quantx.risk.engine import portfolio_risk_snapshot
from quantx.strategies.base import (
    MeanReversionStrategy,
    MomentumStrategy,
    MovingAverageCrossoverStrategy,
)

app = FastAPI(
    title="QuantX API",
    description="Quantitative trading, portfolio optimization, and risk analytics service.",
    version="0.1.0",
)

DEFAULT_TICKERS = ["AAPL", "MSFT", "NVDA", "AMZN", "SPY"]
SECTOR_MAP = {
    "AAPL": "Technology",
    "MSFT": "Technology",
    "NVDA": "Technology",
    "AMZN": "Consumer Discretionary",
    "SPY": "Benchmark",
}
BACKTEST_RESULTS: dict[str, dict] = {}


class HealthResponse(BaseModel):
    status: str
    app: str
    environment: str


class AssetSummary(BaseModel):
    ticker: str
    sector: str
    industry: str | None = None
    benchmark: str | None = None


class BenchmarkRequest(BaseModel):
    benchmark: str = "SPY"
    benchmark_ticker: str | None = None
    description: str | None = None


class StrategyRequest(BaseModel):
    ticker: str = Field(default="AAPL")
    strategy: Literal["moving_average", "momentum", "mean_reversion"] = "moving_average"
    start_date: str = "2024-01-01"
    end_date: str = "2024-06-30"
    capital: float = 100000.0
    transaction_cost: float = 0.0005
    fast_window: int = 5
    slow_window: int = 10
    benchmark: str = "SPY"
    benchmark_ticker: str | None = None


class BacktestRequest(StrategyRequest):
    pass


class OptimizeRequest(BaseModel):
    tickers: list[str] = Field(default_factory=lambda: ["AAPL", "MSFT", "NVDA"])
    method: Literal["equal_weight", "min_volatility", "max_sharpe", "risk_parity"] = "max_sharpe"
    start_date: str = "2024-01-01"
    end_date: str = "2024-06-30"
    min_weight: float = 0.05
    max_weight: float = 0.5
    risk_free_rate: float = 0.02


class RiskRequest(BaseModel):
    tickers: list[str] = Field(default_factory=lambda: ["AAPL", "MSFT", "NVDA"])
    weights: dict[str, float] | None = None
    start_date: str = "2024-01-01"
    end_date: str = "2024-06-30"


class MLPredictRequest(BaseModel):
    ticker: str = "AAPL"
    start_date: str = "2024-01-01"
    end_date: str = "2024-06-30"
    target_window: int = 5


class StrategyResultResponse(BaseModel):
    strategy: str
    signal: str
    latest_signal_value: float
    metrics: dict


class BacktestResultResponse(BaseModel):
    id: str
    strategy: str
    metrics: dict
    equity_curve: list[float]
    drawdown_curve: list[float]


@app.post("/benchmark")
def benchmark(request: BenchmarkRequest) -> dict:
    if request.benchmark.lower() == "custom" and not request.benchmark_ticker:
        raise HTTPException(status_code=400, detail="benchmark_ticker is required when benchmark='custom'.")
    config = resolve_benchmark_config(request.benchmark, request.benchmark_ticker)
    benchmark_ticker = request.benchmark_ticker or config.ticker
    frame = generate_sample_market_data(
        tickers=[benchmark_ticker],
        start_date="2024-01-01",
        end_date="2024-06-30",
    )
    strategy = MovingAverageCrossoverStrategy(fast_window=5, slow_window=10, transaction_cost=0.0005)
    comparison = Backtester(initial_capital=100000.0, transaction_cost=0.0005).compare_with_benchmark(
        frame,
        strategy,
        benchmark_name=config.name,
        benchmark_ticker=benchmark_ticker,
    )
    comparison["benchmark_config"] = {
        "name": config.name,
        "ticker": config.ticker,
        "description": config.description,
        "category": config.category,
    }
    return comparison


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        app="QuantX",
        environment=settings.app_env,
    )


@app.get("/")
def root() -> dict[str, str]:
    return {
        "message": "QuantX API is running.",
        "docs": "/docs",
    }


@app.get("/assets", response_model=list[AssetSummary])
def list_assets() -> list[AssetSummary]:
    return [
        AssetSummary(ticker=ticker, sector=SECTOR_MAP.get(ticker, "Unknown"), industry=None, benchmark="SPY")
        for ticker in DEFAULT_TICKERS
    ]


@app.get("/market-data")
def get_market_data(
    ticker: str | None = Query(default=None),
    start_date: str = Query(default="2024-01-01"),
    end_date: str = Query(default="2024-06-30"),
) -> dict:
    tickers = [ticker] if ticker else DEFAULT_TICKERS
    frame = generate_sample_market_data(tickers=tickers, start_date=start_date, end_date=end_date)
    return {
        "count": len(frame),
        "rows": frame.to_dict(orient="records"),
    }


@app.post("/strategy/run", response_model=StrategyResultResponse)
def run_strategy(request: StrategyRequest) -> StrategyResultResponse:
    frame = generate_sample_market_data(
        tickers=[request.ticker],
        start_date=request.start_date,
        end_date=request.end_date,
    )
    strategy_map = {
        "moving_average": MovingAverageCrossoverStrategy(
            fast_window=request.fast_window,
            slow_window=request.slow_window,
            transaction_cost=request.transaction_cost,
        ),
        "momentum": MomentumStrategy(
            fast_window=request.fast_window,
            slow_window=request.slow_window,
            transaction_cost=request.transaction_cost,
        ),
        "mean_reversion": MeanReversionStrategy(
            fast_window=request.fast_window,
            slow_window=request.slow_window,
            transaction_cost=request.transaction_cost,
        ),
    }
    strategy = strategy_map.get(request.strategy)
    if strategy is None:
        raise HTTPException(status_code=400, detail=f"Unsupported strategy: {request.strategy}")

    signal = strategy.generate_signals(frame)
    last_signal = float(signal.iloc[-1]) if not signal.empty else 0.0
    signal_label = "BUY" if last_signal > 0 else "SELL" if last_signal < 0 else "HOLD"
    return StrategyResultResponse(
        strategy=request.strategy,
        signal=signal_label,
        latest_signal_value=last_signal,
        metrics={
            "capital": request.capital,
            "transaction_cost": request.transaction_cost,
            "signal_count": int(len(signal)),
        },
    )


@app.post("/backtest", response_model=BacktestResultResponse)
def create_backtest(request: BacktestRequest) -> BacktestResultResponse:
    frame = generate_sample_market_data(
        tickers=[request.ticker],
        start_date=request.start_date,
        end_date=request.end_date,
    )
    strategy_map = {
        "moving_average": MovingAverageCrossoverStrategy(
            fast_window=request.fast_window,
            slow_window=request.slow_window,
            transaction_cost=request.transaction_cost,
        ),
        "momentum": MomentumStrategy(
            fast_window=request.fast_window,
            slow_window=request.slow_window,
            transaction_cost=request.transaction_cost,
        ),
        "mean_reversion": MeanReversionStrategy(
            fast_window=request.fast_window,
            slow_window=request.slow_window,
            transaction_cost=request.transaction_cost,
        ),
    }
    strategy = strategy_map.get(request.strategy)
    if strategy is None:
        raise HTTPException(status_code=400, detail=f"Unsupported strategy: {request.strategy}")

    result = Backtester(initial_capital=request.capital, transaction_cost=request.transaction_cost).run(frame, strategy)
    backtest_id = str(uuid.uuid4())
    BACKTEST_RESULTS[backtest_id] = {
        "strategy": request.strategy,
        "metrics": result.metrics,
        "equity_curve": result.equity_curve.tolist(),
        "drawdown_curve": result.drawdown_curve.tolist(),
    }
    return BacktestResultResponse(
        id=backtest_id,
        strategy=request.strategy,
        metrics=result.metrics,
        equity_curve=result.equity_curve.tolist(),
        drawdown_curve=result.drawdown_curve.tolist(),
    )


@app.get("/backtest/{backtest_id}")
def get_backtest(backtest_id: str) -> dict:
    if backtest_id not in BACKTEST_RESULTS:
        raise HTTPException(status_code=404, detail="Backtest result not found.")
    return BACKTEST_RESULTS[backtest_id]


@app.get("/portfolio")
def get_portfolio() -> dict:
    tickers = ["AAPL", "MSFT", "NVDA"]
    frame = generate_sample_market_data(tickers=tickers, start_date="2024-01-01", end_date="2024-06-30")
    returns = frame.pivot(index="date", columns="ticker", values="close").pct_change().dropna()
    result = optimize_portfolio(returns, method="max_sharpe")
    return {
        "method": result.method,
        "weights": result.weights.to_dict(),
        "expected_return": result.expected_return,
        "volatility": result.volatility,
        "sharpe_ratio": result.sharpe_ratio,
    }


@app.get("/risk")
def get_risk() -> dict:
    tickers = ["AAPL", "MSFT", "NVDA"]
    frame = generate_sample_market_data(tickers=tickers, start_date="2024-01-01", end_date="2024-06-30")
    returns = frame.pivot(index="date", columns="ticker", values="close").pct_change().dropna()
    benchmark = generate_sample_market_data(tickers=["SPY"], start_date="2024-01-01", end_date="2024-06-30")
    benchmark_returns = benchmark.set_index("date")["close"].pct_change().dropna()
    weights = pd.Series({ticker: 1.0 / len(tickers) for ticker in tickers})
    covariance = returns.cov() * 252.0
    correlation = returns.corr()
    risk = portfolio_risk_snapshot(
        portfolio_returns=returns.mean(axis=1),
        benchmark_returns=benchmark_returns,
        weights=weights,
        covariance=covariance,
        correlation=correlation,
        sectors={ticker: SECTOR_MAP.get(ticker, "Unknown") for ticker in tickers},
    )
    return {
        "volatility": risk.volatility,
        "beta": risk.beta,
        "var_95": risk.var_95,
        "cvar_95": risk.cvar_95,
        "max_drawdown": risk.max_drawdown,
        "concentration": risk.concentration,
        "correlation_risk": risk.correlation_risk,
        "portfolio_exposure": risk.portfolio_exposure.to_dict(),
        "risk_contribution": risk.risk_contribution.to_dict(),
        "sector_exposure": risk.sector_exposure.to_dict(),
        "stress_results": risk.stress_results,
    }


@app.post("/optimize", response_model=dict)
def optimize(request: OptimizeRequest) -> dict:
    frame = generate_sample_market_data(tickers=request.tickers, start_date=request.start_date, end_date=request.end_date)
    returns = frame.pivot(index="date", columns="ticker", values="close").pct_change().dropna()
    result = optimize_portfolio(
        returns,
        method=request.method,
        min_weight=request.min_weight,
        max_weight=request.max_weight,
        risk_free_rate=request.risk_free_rate,
    )
    return {
        "method": result.method,
        "weights": result.weights.to_dict(),
        "expected_return": result.expected_return,
        "volatility": result.volatility,
        "sharpe_ratio": result.sharpe_ratio,
    }


@app.post("/ml/predict")
def ml_predict(request: MLPredictRequest) -> dict:
    frame = generate_sample_market_data(
        tickers=[request.ticker],
        start_date=request.start_date,
        end_date=request.end_date,
    )
    enriched = add_technical_indicators(frame)
    dataset = build_signal_dataset(enriched, target_window=request.target_window)
    if dataset.empty:
        raise HTTPException(status_code=400, detail="No valid labels were created from the provided window.")

    results = train_model_suite(enriched, target_window=request.target_window)
    winner_name = max(results, key=lambda name: results[name].f1)
    winner = results[winner_name]
    last_prediction = winner.predictions.iloc[-1] if not winner.predictions.empty else "HOLD"
    last_probability = max(0.5, winner.accuracy)
    return {
        "ticker": request.ticker,
        "signal": str(last_prediction),
        "probability": last_probability,
        "model": winner_name,
        "confidence": "High" if last_probability >= 0.7 else "Medium" if last_probability >= 0.55 else "Low",
    }
