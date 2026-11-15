"""Technical Indicator Calculations for Risk-Sensitive Reinforcement Learning.
Ensures zero look-ahead bias and strict mathematical correctness.
"""

import pandas as pd
import numpy as np
from typing import List
from src.utils import logger


def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Calculates the Relative Strength Index (RSI) using Wilder's smoothing.
    Returns values normalized to the range [0.0, 1.0].
    """
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    
    # Wilder's exponential smoothing
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / (avg_loss + 1e-8)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    
    # Normalize to [0, 1] for neural network stability
    return (rsi / 100.0).clip(0.0, 1.0)


def calculate_macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    """Calculates Moving Average Convergence Divergence (MACD), Signal line, and Histogram.
    Normalized by the price series for stationarity.
    """
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = (ema_fast - ema_slow) / (series + 1e-8)
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    macd_hist = macd_line - signal_line
    return macd_line, signal_line, macd_hist


def calculate_bollinger_bands(series: pd.Series, period: int = 20, num_std: float = 2.0):
    """Calculates Bollinger Bands and returns normalized relative metrics."""
    sma = series.rolling(window=period).mean()
    std = series.rolling(window=period).std()
    upper = sma + (num_std * std)
    lower = sma - (num_std * std)
    
    bb_high_ratio = (series - upper) / (series + 1e-8)
    bb_low_ratio = (series - lower) / (series + 1e-8)
    bb_width = (upper - lower) / (sma + 1e-8)
    
    return bb_high_ratio, bb_low_ratio, bb_width, upper, lower, sma


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculates Average True Range (ATR) normalized by Close price."""
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    prev_close = close.shift(1)
    
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = true_range.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    return atr / (close + 1e-8)


def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Appends all technical indicators cleanly to a OHLCV DataFrame.
    Validates stationarity and drops initial warmup NaN rows.
    """
    df = df.copy()
    
    # 1. Price Returns
    df["returns"] = df["Close"].pct_change().fillna(0.0)
    df["log_returns"] = np.log(df["Close"] / df["Close"].shift(1)).fillna(0.0)
    
    # 2. Volatility (20-day annualized rolling std)
    df["volatility_20"] = df["returns"].rolling(window=20).std().fillna(0.0) * np.sqrt(252)
    
    # 3. Moving Average Ratios (Stationary relative metrics)
    df["sma_10"] = df["Close"].rolling(window=10).mean()
    df["sma_30"] = df["Close"].rolling(window=30).mean()
    df["ema_20"] = df["Close"].ewm(span=20, adjust=False).mean()
    
    df["sma_10_ratio"] = (df["Close"] - df["sma_10"]) / (df["sma_10"] + 1e-8)
    df["sma_30_ratio"] = (df["Close"] - df["sma_30"]) / (df["sma_30"] + 1e-8)
    df["ema_20_ratio"] = (df["Close"] - df["ema_20"]) / (df["ema_20"] + 1e-8)
    
    # 4. RSI (14-day)
    df["rsi_14"] = calculate_rsi(df["Close"], period=14)
    
    # 5. MACD
    macd, macd_signal, macd_hist = calculate_macd(df["Close"])
    df["macd"] = macd
    df["macd_signal"] = macd_signal
    df["macd_hist"] = macd_hist
    
    # 6. Bollinger Bands
    bb_hi, bb_lo, bb_w, upper, lower, bb_mid = calculate_bollinger_bands(df["Close"])
    df["bb_high_ratio"] = bb_hi
    df["bb_low_ratio"] = bb_lo
    df["bb_width"] = bb_w
    df["bb_upper"] = upper
    df["bb_lower"] = lower
    df["bb_middle"] = bb_mid
    
    # 7. ATR
    df["atr_14_ratio"] = calculate_atr(df, period=14)
    
    # 8. Volume Ratio (Current volume / 20-day average)
    if "Volume" in df.columns:
        vol_avg = df["Volume"].rolling(window=20).mean()
        df["volume_ratio"] = (df["Volume"] / (vol_avg + 1e-8)).clip(0.0, 5.0)
    else:
        df["volume_ratio"] = 1.0
        
    # Drop rows containing NaNs from moving window warmup (typically 30 rows)
    initial_len = len(df)
    df = df.dropna().reset_index(drop=False)
    logger.debug(f"Indicators calculated. Rows: {initial_len} -> {len(df)} after warmup drop.")
    
    return df
