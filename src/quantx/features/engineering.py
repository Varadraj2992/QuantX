from __future__ import annotations

import pandas as pd


def compute_moving_average(series: pd.Series, window: int = 20) -> pd.Series:
    """Compute a rolling simple moving average without look-ahead bias."""
    return series.rolling(window=window, min_periods=window).mean()


def compute_rolling_volatility(series: pd.Series, window: int = 20) -> pd.Series:
    """Compute rolling realized volatility from percent returns."""
    returns = series.pct_change().fillna(0.0)
    return returns.rolling(window=window, min_periods=window).std().mul(100)


def compute_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    """Compute the Relative Strength Index for a price series."""
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(window=window, min_periods=window).mean()
    loss = (-delta.clip(upper=0)).rolling(window=window, min_periods=window).mean()
    rs = gain / loss.replace(0, pd.NA)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50.0)


def compute_macd(
    series: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Compute MACD, signal, and histogram series."""
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def add_technical_indicators(frame: pd.DataFrame) -> pd.DataFrame:
    """Add a practical feature set for time-series signal research."""
    if frame.empty:
        return frame.copy()

    enriched = frame.copy().sort_values(["ticker", "date"]).reset_index(drop=True)
    enriched["ret_1d"] = enriched.groupby("ticker")["close"].pct_change().fillna(0.0)
    enriched["ret_5d"] = enriched.groupby("ticker")["close"].pct_change(5).fillna(0.0)
    enriched["vol_20d"] = enriched.groupby("ticker")["close"].transform(
        lambda x: compute_rolling_volatility(x, window=20)
    )
    enriched["sma_20"] = enriched.groupby("ticker")["close"].transform(
        lambda x: compute_moving_average(x, window=20)
    )
    enriched["rsi_14"] = enriched.groupby("ticker")["close"].transform(
        lambda x: compute_rsi(x, window=14)
    )
    macd_line, signal_line, hist = compute_macd(enriched["close"])
    enriched["macd"] = macd_line
    enriched["macd_signal"] = signal_line
    enriched["macd_hist"] = hist
    return enriched
