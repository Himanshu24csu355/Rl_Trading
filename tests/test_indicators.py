"""Tests for technical indicators and feature engineering.
"""

import pytest
import pandas as pd
import numpy as np
from src.indicators import calculate_rsi, calculate_macd, calculate_bollinger_bands, calculate_atr, add_all_indicators


@pytest.fixture
def sample_ohlcv_data():
    """Generates a synthetic price series for testing."""
    np.random.seed(42)
    n = 100
    dates = pd.date_range("2023-01-01", periods=n, freq="B")
    close = 100.0 * np.exp(np.cumsum(np.random.normal(0.001, 0.02, n)))
    high = close * 1.01
    low = close * 0.99
    open_p = close * 1.002
    volume = np.random.randint(10000, 500000, n)
    
    return pd.DataFrame({
        "Date": dates,
        "Open": open_p,
        "High": high,
        "Low": low,
        "Close": close,
        "Volume": volume
    })


def test_rsi_bounds(sample_ohlcv_data):
    rsi = calculate_rsi(sample_ohlcv_data["Close"], period=14)
    valid_rsi = rsi.dropna()
    assert len(valid_rsi) > 0
    # RSI is normalized to [0, 1]
    assert valid_rsi.min() >= 0.0
    assert valid_rsi.max() <= 1.0


def test_macd_calculation(sample_ohlcv_data):
    macd, signal, hist = calculate_macd(sample_ohlcv_data["Close"])
    assert len(macd) == len(sample_ohlcv_data)
    assert len(signal) == len(sample_ohlcv_data)
    assert len(hist) == len(sample_ohlcv_data)
    # Check hist = macd - signal
    np.testing.assert_allclose(hist.values, (macd - signal).values, atol=1e-7)


def test_bollinger_bands_ordering(sample_ohlcv_data):
    bb_hi, bb_lo, bb_w, upper, lower, sma = calculate_bollinger_bands(sample_ohlcv_data["Close"])
    valid_idx = ~(upper.isna() | lower.isna())
    assert (upper[valid_idx] >= sma[valid_idx]).all()
    assert (sma[valid_idx] >= lower[valid_idx]).all()


def test_add_all_indicators(sample_ohlcv_data):
    df_feat = add_all_indicators(sample_ohlcv_data)
    # Check that warmup NaNs are dropped
    assert not df_feat.isna().any().any()
    assert "returns" in df_feat.columns
    assert "rsi_14" in df_feat.columns
    assert "macd" in df_feat.columns
    assert "volatility_20" in df_feat.columns
    assert len(df_feat) > 50
