from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BenchmarkConfig:
    """Reusable benchmark definitions for research and production-style analytics."""

    name: str
    ticker: str
    description: str = "Benchmark index proxy"
    category: str = "market"


@dataclass
class BacktestResult:
    """Structured results from a single backtest run."""

    equity_curve: pd.Series
    drawdown_curve: pd.Series
    monthly_returns: pd.DataFrame
    metrics: dict[str, float]
    trades: pd.DataFrame


BENCHMARK_REGISTRY: dict[str, BenchmarkConfig] = {
    "SPY": BenchmarkConfig(name="SPY", ticker="SPY", description="S&P 500 benchmark proxy", category="equity"),
    "NIFTY50": BenchmarkConfig(
        name="NIFTY50", ticker="NIFTY50", description="NIFTY 50 benchmark proxy", category="equity"
    ),
    "NIFTY_50": BenchmarkConfig(
        name="NIFTY_50", ticker="NIFTY50", description="NIFTY 50 benchmark proxy", category="equity"
    ),
    "custom": BenchmarkConfig(
        name="custom", ticker="custom", description="User-specified benchmark", category="custom"
    ),
}


def resolve_benchmark_config(
    benchmark: str | BenchmarkConfig | None = None,
    benchmark_ticker: str | None = None,
) -> BenchmarkConfig:
    """Resolve a benchmark selector to a reusable config object."""
    if isinstance(benchmark, BenchmarkConfig):
        return benchmark

    selected = (benchmark or "SPY").strip()
    normalized = selected.upper().replace("-", "_")
    if normalized in {"NIFTY50", "NIFTY_50"}:
        normalized = "NIFTY50"
    if normalized in BENCHMARK_REGISTRY:
        config = BENCHMARK_REGISTRY[normalized]
    elif benchmark_ticker:
        config = BenchmarkConfig(
            name=benchmark_ticker,
            ticker=benchmark_ticker,
            description=f"Custom benchmark: {benchmark_ticker}",
            category="custom",
        )
    else:
        raise ValueError(
            f"Unsupported benchmark: {benchmark}. Available benchmarks: {sorted(BENCHMARK_REGISTRY)}"
        )

    if benchmark_ticker and config.name == "custom":
        config = BenchmarkConfig(
            name=benchmark_ticker,
            ticker=benchmark_ticker,
            description=f"Custom benchmark: {benchmark_ticker}",
            category="custom",
        )
    return config


def resolve_benchmark_returns(
    frame: pd.DataFrame,
    benchmark: str | BenchmarkConfig | None = None,
    benchmark_ticker: str | None = None,
) -> tuple[pd.Series, BenchmarkConfig]:
    """Select a benchmark series from a market-data frame while preserving a reusable config object."""
    config = resolve_benchmark_config(benchmark, benchmark_ticker)
    if "ticker" in frame.columns:
        target_ticker = benchmark_ticker or config.ticker
        matches = frame[frame["ticker"] == target_ticker].sort_values("date").copy()
        if not matches.empty:
            return matches["close"].pct_change().fillna(0.0), config
    if benchmark_ticker and benchmark_ticker != config.ticker:
        matches = frame[frame["ticker"] == benchmark_ticker].sort_values("date").copy()
        if not matches.empty:
            return matches["close"].pct_change().fillna(0.0), BenchmarkConfig(
                name=benchmark_ticker,
                ticker=benchmark_ticker,
                description=f"Custom benchmark: {benchmark_ticker}",
                category="custom",
            )
    if "close" in frame.columns:
        return frame["close"].pct_change().fillna(0.0), config
    raise ValueError(f"Could not resolve benchmark returns for benchmark {config.name!r}.")


def compute_drawdown(series: pd.Series) -> pd.Series:
    """Compute rolling drawdown percentage from an equity series."""
    rolling_max = series.cummax()
    return (series / rolling_max - 1.0) * 100.0


def summarize_monthly_returns(series: pd.Series) -> pd.DataFrame:
    """Aggregate a daily return series into monthly returns."""
    if series.empty:
        return pd.DataFrame(columns=["year", "month", "monthly_return"])

    monthly = pd.DataFrame({"date": series.index, "return": series.to_numpy()})
    monthly["year"] = monthly["date"].dt.year
    monthly["month"] = monthly["date"].dt.month
    return (
        monthly.groupby(["year", "month"], as_index=False)["return"]
        .prod()
        .rename(columns={"return": "monthly_return"})
        .assign(monthly_return=lambda df: df["monthly_return"] - 1.0)
    )


def benchmark_comparison(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
    initial_capital: float = 100_000.0,
    benchmark_name: str = "SPY",
) -> dict[str, float | str]:
    """Compare strategy and benchmark performance using cumulative return and risk metrics."""
    strategy_series = pd.Series(strategy_returns, dtype=float).fillna(0.0)
    benchmark_series = pd.Series(benchmark_returns, dtype=float).fillna(0.0)
    aligned = pd.concat(
        [strategy_series.rename("strategy"), benchmark_series.rename("benchmark")],
        axis=1,
        join="outer",
    ).fillna(0.0)

    strategy_equity = (1.0 + aligned["strategy"]).cumprod() * float(initial_capital)
    benchmark_equity = (1.0 + aligned["benchmark"]).cumprod() * float(initial_capital)

    strategy_total_return = strategy_equity.iloc[-1] / float(initial_capital) - 1.0
    benchmark_total_return = benchmark_equity.iloc[-1] / float(initial_capital) - 1.0
    strategy_cagr = (1.0 + strategy_total_return) ** (252.0 / max(len(strategy_series), 1)) - 1.0
    benchmark_cagr = (1.0 + benchmark_total_return) ** (252.0 / max(len(benchmark_series), 1)) - 1.0

    strategy_vol = aligned["strategy"].std(ddof=1) * np.sqrt(252.0) if len(aligned) > 1 else 0.0
    benchmark_vol = aligned["benchmark"].std(ddof=1) * np.sqrt(252.0) if len(aligned) > 1 else 0.0
    strategy_sharpe = (
        (aligned["strategy"].mean() / aligned["strategy"].std(ddof=1)) * np.sqrt(252.0)
        if aligned["strategy"].std(ddof=1) > 0
        else 0.0
    )
    benchmark_sharpe = (
        (aligned["benchmark"].mean() / aligned["benchmark"].std(ddof=1)) * np.sqrt(252.0)
        if aligned["benchmark"].std(ddof=1) > 0
        else 0.0
    )

    strategy_max_drawdown = abs(compute_drawdown(strategy_equity).min()) if not strategy_equity.empty else 0.0
    benchmark_max_drawdown = abs(compute_drawdown(benchmark_equity).min()) if not benchmark_equity.empty else 0.0

    return {
        "benchmark_name": benchmark_name,
        "strategy_total_return": float(strategy_total_return),
        "benchmark_total_return": float(benchmark_total_return),
        "excess_return": float(strategy_total_return - benchmark_total_return),
        "strategy_cagr": float(strategy_cagr),
        "benchmark_cagr": float(benchmark_cagr),
        "strategy_volatility": float(strategy_vol),
        "benchmark_volatility": float(benchmark_vol),
        "strategy_sharpe": float(strategy_sharpe),
        "benchmark_sharpe": float(benchmark_sharpe),
        "strategy_max_drawdown": float(strategy_max_drawdown),
        "benchmark_max_drawdown": float(benchmark_max_drawdown),
    }


class Backtester:
    """Simple, production-style backtester with transaction-cost-aware returns."""

    def __init__(
        self,
        initial_capital: float = 100_000.0,
        transaction_cost: float = 0.0005,
        risk_free_rate: float = 0.02,
    ) -> None:
        self.initial_capital = float(initial_capital)
        self.transaction_cost = float(transaction_cost)
        self.risk_free_rate = float(risk_free_rate)

    def _positions_from_signal(self, frame: pd.DataFrame, positions: pd.Series | None = None) -> pd.Series:
        if positions is None:
            positions = frame.get("position", pd.Series(0.0, index=frame.index))
        return pd.Series(positions, index=frame.index, dtype=float).fillna(0.0)

    def run(self, frame: pd.DataFrame, strategy: Any, positions: pd.Series | None = None) -> BacktestResult:
        """Backtest a strategy against a price series, including slippage and costs."""
        if frame.empty:
            raise ValueError("Backtest input frame cannot be empty.")
        if "close" not in frame.columns:
            raise KeyError("Input data must contain a 'close' column for the backtest.")

        working = frame.copy().sort_values("date").reset_index(drop=True)
        if positions is None and hasattr(strategy, "generate_positions"):
            positions = strategy.generate_positions(working)
        position_series = self._positions_from_signal(working, positions)

        daily_returns = working["close"].pct_change().fillna(0.0)
        prev_positions = position_series.shift(1).fillna(0.0)
        turnover = (position_series - prev_positions).abs().fillna(0.0)
        net_returns = daily_returns * position_series - self.transaction_cost * turnover

        equity_curve = (1.0 + net_returns).cumprod() * self.initial_capital
        drawdown_curve = compute_drawdown(equity_curve)

        monthly_returns = summarize_monthly_returns(
            pd.Series(net_returns.values, index=working["date"], name="daily_return")
        )

        trades = self._build_trade_log(working, position_series)
        metrics = self._calculate_metrics(equity_curve, net_returns, trades)

        return BacktestResult(
            equity_curve=equity_curve,
            drawdown_curve=drawdown_curve,
            monthly_returns=monthly_returns,
            metrics=metrics,
            trades=trades,
        )

    def _build_trade_log(self, frame: pd.DataFrame, positions: pd.Series) -> pd.DataFrame:
        if frame.empty:
            return pd.DataFrame(columns=["date", "entry", "exit", "return", "pnl"])

        prev_positions = positions.shift(1).fillna(0.0)
        trade_signal = positions != prev_positions
        trade_dates = frame.loc[trade_signal, "date"]
        trade_rows: list[dict[str, Any]] = []

        for current_date in trade_dates:
            idx = frame.index[frame["date"] == current_date][0]
            previous_position = prev_positions.iloc[idx]
            current_position = positions.iloc[idx]
            if idx > 0:
                day_return = float(frame.loc[idx, "close"] / frame.loc[idx - 1, "close"] - 1.0)
            else:
                day_return = 0.0
            pnl = previous_position * day_return
            trade_rows.append(
                {
                    "date": current_date,
                    "prev_position": previous_position,
                    "current_position": current_position,
                    "day_return": day_return,
                    "pnl": pnl,
                }
            )

        return pd.DataFrame(trade_rows)

    def compare_with_benchmark(
        self,
        frame: pd.DataFrame,
        strategy: Any,
        benchmark_returns: pd.Series | None = None,
        benchmark_name: str = "SPY",
        benchmark_ticker: str | None = None,
    ) -> dict[str, Any]:
        """Compare the strategy equity curve with a benchmark while keeping the backtest result structure intact."""
        asset_frame = frame.copy()
        if "ticker" in frame.columns and frame["ticker"].nunique() > 1:
            target_benchmark = benchmark_ticker or benchmark_name
            asset_frame = frame[frame["ticker"] != target_benchmark].sort_values("date").copy()
        if asset_frame.empty:
            asset_frame = frame.copy()

        strategy_result = self.run(asset_frame, strategy)
        if benchmark_returns is None:
            benchmark_returns, config = resolve_benchmark_returns(
                frame,
                benchmark=benchmark_name,
                benchmark_ticker=benchmark_ticker,
            )
            benchmark_name = config.name
        else:
            config = resolve_benchmark_config(benchmark_name, benchmark_ticker)
            benchmark_name = config.name

        strategy_returns = pd.Series(
            strategy_result.equity_curve.pct_change().fillna(0.0),
            index=strategy_result.equity_curve.index,
        )
        benchmark_series = pd.Series(benchmark_returns, dtype=float).fillna(0.0)
        comparison = benchmark_comparison(
            strategy_returns=strategy_returns,
            benchmark_returns=benchmark_series,
            initial_capital=self.initial_capital,
            benchmark_name=benchmark_name,
        )

        return {
            "benchmark_name": benchmark_name,
            "strategy_metrics": strategy_result.metrics,
            "benchmark_metrics": comparison,
            "strategy_equity_curve": strategy_result.equity_curve.tolist(),
            "benchmark_equity_curve": ((1.0 + benchmark_series).cumprod() * self.initial_capital).tolist(),
        }

    def _calculate_metrics(
        self,
        equity_curve: pd.Series,
        strategy_returns: pd.Series,
        trades: pd.DataFrame,
    ) -> dict[str, float]:
        if equity_curve.empty:
            return {
                "initial_capital": self.initial_capital,
                "final_portfolio_value": self.initial_capital,
                "total_return": 0.0,
                "cagr": 0.0,
                "annualized_return": 0.0,
                "annualized_volatility": 0.0,
                "sharpe_ratio": 0.0,
                "sortino_ratio": 0.0,
                "max_drawdown": 0.0,
                "calmar_ratio": 0.0,
                "win_rate": 0.0,
                "num_trades": 0,
                "average_trade_return": 0.0,
                "profit_factor": 0.0,
                "transaction_costs": 0.0,
                "turnover": 0.0,
            }

        daily_returns = strategy_returns.fillna(0.0)
        periods = len(daily_returns)
        total_return = (equity_curve.iloc[-1] / self.initial_capital) - 1.0
        cagr = (equity_curve.iloc[-1] / self.initial_capital) ** (252.0 / max(periods, 1)) - 1.0
        annualized_return = (1.0 + total_return) ** (252.0 / max(periods, 1)) - 1.0
        annualized_volatility = daily_returns.std(ddof=1) * np.sqrt(252.0)

        excess = daily_returns - self.risk_free_rate / 252.0
        sharpe = (excess.mean() / excess.std(ddof=1) * np.sqrt(252.0)) if excess.std(ddof=1) > 0 else 0.0

        downside = daily_returns[daily_returns < 0]
        sortino = (
            (daily_returns.mean() - self.risk_free_rate / 252.0) / downside.std(ddof=1) * np.sqrt(252.0)
            if len(downside) > 0 and downside.std(ddof=1) > 0
            else 0.0
        )

        drawdown = compute_drawdown(equity_curve)
        max_drawdown = abs(drawdown.min())

        if max_drawdown > 0:
            calmar = (annualized_return / max_drawdown) if annualized_return > 0 else 0.0
        else:
            calmar = 0.0

        wins = (daily_returns > 0).sum()
        win_rate = wins / max(len(daily_returns), 1)

        if not trades.empty:
            trade_returns = trades["pnl"].astype(float)
            average_trade_return = float(trade_returns.mean()) if len(trade_returns) else 0.0
            num_trades = int(len(trade_returns))
            positive = trade_returns[trade_returns > 0].sum()
            negative = abs(trade_returns[trade_returns < 0].sum())
            profit_factor = positive / negative if negative > 0 else float("inf")
        else:
            average_trade_return = 0.0
            num_trades = 0
            profit_factor = 0.0

        transaction_costs = (
            (trades["day_return"].abs() * self.transaction_cost).sum()
            if not trades.empty
            else 0.0
        )
        turnover = (
            float((trades["current_position"] - trades["prev_position"]).abs().sum())
            if not trades.empty
            else 0.0
        )

        return {
            "initial_capital": float(self.initial_capital),
            "final_portfolio_value": float(equity_curve.iloc[-1]),
            "total_return": float(total_return),
            "cagr": float(cagr),
            "annualized_return": float(annualized_return),
            "annualized_volatility": float(annualized_volatility),
            "sharpe_ratio": float(sharpe),
            "sortino_ratio": float(sortino),
            "max_drawdown": float(max_drawdown),
            "calmar_ratio": float(calmar),
            "win_rate": float(win_rate),
            "num_trades": int(num_trades),
            "average_trade_return": float(average_trade_return),
            "profit_factor": float(profit_factor),
            "transaction_costs": float(transaction_costs),
            "turnover": float(turnover),
        }
