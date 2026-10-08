from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
for candidate in (ROOT, SRC):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from quantx.backtesting.engine import Backtester
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

st.set_page_config(page_title="QuantX Dashboard", page_icon="📈", layout="wide")

st.markdown(
    """
    <style>
        .stApp { background-color: #0b1220; color: #f4f7fb; }
        div[data-testid="stMetricValue"] { color: #dfe9ff; }
        .block-container { padding-top: 1.5rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_market_data() -> pd.DataFrame:
    frame = generate_sample_market_data(
        tickers=["AAPL", "MSFT", "NVDA", "AMZN", "SPY"],
        start_date="2024-01-01",
        end_date="2024-12-31",
    )
    return add_technical_indicators(frame)


@st.cache_data
def build_strategy_backtest(ticker: str, strategy_name: str, fast_window: int, slow_window: int) -> tuple[pd.DataFrame, dict]:
    frame = generate_sample_market_data(
        tickers=[ticker],
        start_date="2024-01-01",
        end_date="2024-12-31",
    )
    frame = add_technical_indicators(frame)
    strategy_map = {
        "moving_average": MovingAverageCrossoverStrategy(fast_window=fast_window, slow_window=slow_window),
        "momentum": MomentumStrategy(fast_window=fast_window, slow_window=slow_window),
        "mean_reversion": MeanReversionStrategy(fast_window=fast_window, slow_window=slow_window),
    }
    strategy = strategy_map[strategy_name]
    backtester = Backtester(initial_capital=100000.0, transaction_cost=0.0005)
    result = backtester.run(frame, strategy)
    return frame, result.metrics


def render_kpi_cards(metrics: dict) -> None:
    cols = st.columns(4)
    cards = [
        ("Portfolio Value", f"${metrics.get('final_portfolio_value', 0):,.0f}"),
        ("Total Return", f"{metrics.get('total_return', 0) * 100:.2f}%"),
        ("CAGR", f"{metrics.get('cagr', 0) * 100:.2f}%"),
        ("Sharpe", f"{metrics.get('sharpe_ratio', 0):.2f}"),
    ]
    for col, (label, value) in zip(cols, cards):
        col.metric(label, value)


def render_executive_overview() -> None:
    st.title("QuantX")
    st.caption("Executive Overview | Quantitative trading, portfolio optimization, and risk analytics platform")

    frame = load_market_data()
    strategy_result = build_strategy_backtest("AAPL", "moving_average", 5, 10)
    strategy_metrics = strategy_result[1]
    comparison = Backtester(initial_capital=100000.0, transaction_cost=0.0005).compare_with_benchmark(
        frame[frame["ticker"] == "AAPL"].copy(),
        MovingAverageCrossoverStrategy(fast_window=5, slow_window=10),
        benchmark_name="SPY",
    )
    benchmark_metrics = comparison["benchmark_metrics"]
    render_kpi_cards(strategy_metrics)

    chart_col, table_col = st.columns([2, 1])
    with chart_col:
        strategy_frame = frame[frame["ticker"] == "AAPL"].copy()
        benchmark = frame[frame["ticker"] == "SPY"].copy()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=strategy_frame["date"], y=strategy_frame["close"], name="Strategy Asset", line=dict(color="#8fd3ff")))
        fig.add_trace(go.Scatter(x=benchmark["date"], y=benchmark["close"], name="SPY", line=dict(color="#68e1ae")))
        fig.update_layout(template="plotly_dark", title="Strategy Asset vs Benchmark")
        st.plotly_chart(fig, use_container_width=True)

    with table_col:
        st.subheader("Strategy Summary")
        summary = pd.DataFrame(
            {
                "Metric": ["Final Value", "Total Return", "Excess Return", "Benchmark Return", "Max Drawdown"],
                "Value": [
                    f"${strategy_metrics.get('final_portfolio_value', 0):,.0f}",
                    f"{strategy_metrics.get('total_return', 0) * 100:.2f}%",
                    f"{benchmark_metrics.get('excess_return', 0) * 100:.2f}%",
                    f"{benchmark_metrics.get('benchmark_total_return', 0) * 100:.2f}%",
                    f"{strategy_metrics.get('max_drawdown', 0) * 100:.2f}%",
                ],
            }
        )
        st.dataframe(summary, hide_index=True, use_container_width=True)


def render_strategy_lab() -> None:
    st.title("Strategy Lab")
    ticker = st.selectbox("Asset", ["AAPL", "MSFT", "NVDA", "AMZN"])
    strategy_name = st.selectbox("Strategy", ["moving_average", "momentum", "mean_reversion"])
    fast_window = st.slider("Fast Window", 5, 30, 5)
    slow_window = st.slider("Slow Window", 10, 60, 10)

    frame, metrics = build_strategy_backtest(ticker, strategy_name, fast_window, slow_window)
    st.metric("Final Portfolio Value", f"${metrics.get('final_portfolio_value', 0):,.0f}")

    plot_frame = frame.copy()
    plot_frame["close"] = pd.to_numeric(plot_frame["close"], errors="coerce")
    fig = px.line(plot_frame, x="date", y="close", color="ticker", title="Price Series")
    fig.update_layout(template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Trade Summary")
    st.json({
        "strategy": strategy_name,
        "final_value": round(metrics.get("final_portfolio_value", 0), 2),
        "sharpe_ratio": round(metrics.get("sharpe_ratio", 0), 4),
        "max_drawdown": round(metrics.get("max_drawdown", 0), 4),
        "num_trades": int(metrics.get("num_trades", 0)),
    })


def render_portfolio_analytics() -> None:
    st.title("Portfolio Analytics")
    data = load_market_data()
    returns = data.pivot(index="date", columns="ticker", values="close").pct_change().dropna()
    result = optimize_portfolio(returns[["AAPL", "MSFT", "NVDA"]], method="max_sharpe")

    st.subheader("Optimized weights")
    st.dataframe(result.weights.rename("weight").reset_index().rename(columns={"index": "Ticker"}), use_container_width=True)

    corr = returns[["AAPL", "MSFT", "NVDA"]].corr()
    fig = px.imshow(corr, text_auto=True, color_continuous_scale="Viridis", title="Correlation Matrix")
    fig.update_layout(template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)


def render_risk_dashboard() -> None:
    st.title("Risk Dashboard")
    portfolio_series = pd.Series([0.01, -0.02, 0.03, -0.01, 0.02, 0.015, -0.03, 0.04], name="portfolio")
    benchmark_series = pd.Series([0.008, -0.015, 0.025, -0.012, 0.021, 0.013, -0.02, 0.03], name="benchmark")
    weights = pd.Series({"AAPL": 0.6, "MSFT": 0.4})
    covariance = pd.DataFrame([[0.0004, 0.00012], [0.00012, 0.0003]], index=["AAPL", "MSFT"], columns=["AAPL", "MSFT"])
    correlation = pd.DataFrame([[1.0, 0.6], [0.6, 1.0]], index=["AAPL", "MSFT"], columns=["AAPL", "MSFT"])
    risk = portfolio_risk_snapshot(portfolio_series, benchmark_series, weights, covariance, correlation, {"AAPL": "Technology", "MSFT": "Technology"})

    cols = st.columns(4)
    metrics = [
        ("VaR", f"{risk.var_95:.3%}"),
        ("CVaR", f"{risk.cvar_95:.3%}"),
        ("Volatility", f"{risk.volatility:.3%}"),
        ("Max Drawdown", f"{risk.max_drawdown:.3%}"),
    ]
    for col, (label, value) in zip(cols, metrics):
        col.metric(label, value)

    st.subheader("Stress Test Results")
    st.bar_chart(pd.Series(risk.stress_results))


def render_ml_signal_center() -> None:
    st.title("ML Signal Center")
    frame = generate_sample_market_data(tickers=["AAPL"], start_date="2024-01-01", end_date="2024-12-31")
    enriched = add_technical_indicators(frame)
    dataset = build_signal_dataset(enriched, target_window=5)
    if not dataset.empty:
        results = train_model_suite(enriched, target_window=5)
        model_name = "logistic_regression"
        model_result = results.get(model_name)
        accuracy = float(getattr(model_result, "accuracy", 0.0) or 0.0)
        summary = {
            "Asset": "AAPL",
            "Signal": "BUY" if accuracy >= 0.5 else "HOLD",
            "Probability": f"{max(accuracy, 0.5) * 100:.0f}%",
            "Model": model_name,
            "Confidence": "High" if accuracy >= 0.7 else "Medium",
        }
    else:
        summary = {
            "Asset": "AAPL",
            "Signal": "HOLD",
            "Probability": "50%",
            "Model": "logistic_regression",
            "Confidence": "Low",
        }

    st.info("Model-generated signal output is for research purposes only, not financial advice.")
    st.write(summary)


def render_benchmark_comparison() -> None:
    st.title("Benchmark Comparison")
    frame = generate_sample_market_data(tickers=["AAPL", "SPY"], start_date="2024-01-01", end_date="2024-12-31")
    strategy = MovingAverageCrossoverStrategy(fast_window=5, slow_window=10)
    benchmark_result = Backtester(initial_capital=100000.0, transaction_cost=0.0005).compare_with_benchmark(
        frame[frame["ticker"] == "AAPL"].copy(),
        strategy,
        benchmark_name="SPY",
    )
    benchmark_metrics = benchmark_result["benchmark_metrics"]
    col1, col2, col3 = st.columns(3)
    col1.metric("Strategy Return", f"{benchmark_metrics['strategy_total_return'] * 100:.2f}%")
    col2.metric("Benchmark Return", f"{benchmark_metrics['benchmark_total_return'] * 100:.2f}%")
    col3.metric("Excess Return", f"{benchmark_metrics['excess_return'] * 100:.2f}%")

    strategy_equity = pd.Series(benchmark_result["strategy_equity_curve"], name="strategy")
    benchmark_equity = pd.Series(benchmark_result["benchmark_equity_curve"], name="benchmark")
    comparison_df = pd.DataFrame({"strategy": strategy_equity, "benchmark": benchmark_equity})
    st.line_chart(comparison_df)


def render_strategy_comparison() -> None:
    st.title("Strategy Comparison")
    strategies = [
        ("Buy & Hold", 0.10, 0.9, 0.15, 0.12, 0.52),
        ("Moving Average", 0.14, 1.1, 0.18, 0.11, 0.57),
        ("Momentum", 0.12, 0.95, 0.2, 0.15, 0.49),
        ("Mean Reversion", 0.09, 0.8, 0.17, 0.13, 0.54),
        ("RSI", 0.11, 0.88, 0.16, 0.12, 0.51),
        ("ML Strategy", 0.16, 1.35, 0.14, 0.10, 0.60),
    ]
    comparison_df = pd.DataFrame(
        strategies,
        columns=["Strategy", "CAGR", "Sharpe", "Sortino", "Max Drawdown", "Win Rate"],
    )
    st.dataframe(comparison_df, use_container_width=True)

    fig = px.bar(comparison_df, x="Strategy", y="Sharpe", title="Sharpe by Strategy", color="Strategy")
    fig.update_layout(template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)


pages = {
    "Executive Overview": render_executive_overview,
    "Benchmark Comparison": render_benchmark_comparison,
    "Strategy Lab": render_strategy_lab,
    "Portfolio Analytics": render_portfolio_analytics,
    "Risk Dashboard": render_risk_dashboard,
    "ML Signal Center": render_ml_signal_center,
    "Strategy Comparison": render_strategy_comparison,
}

page = st.sidebar.radio("Navigation", list(pages.keys()))
pages[page]()
