"""Data loading, caching, preprocessing, and market calendar pipeline using yfinance.
Supports Historical Backtesting chronological splitting and Current Market Decision Support.
"""

import os
from pathlib import Path
from datetime import datetime, date, timedelta, time
from typing import Tuple, Optional, Dict, Any
import pytz
import pandas as pd
import numpy as np
import yfinance as yf

from config import (
    DATA_DIR,
    DEFAULT_START_DATE,
    DEFAULT_END_DATE,
    DEFAULT_LOOKBACK_YEARS,
    TRAIN_RATIO,
    VAL_RATIO,
    TIMEZONE_INDIA,
    MARKET_OPEN_HOUR,
    MARKET_OPEN_MINUTE,
    MARKET_CLOSE_HOUR,
    MARKET_CLOSE_MINUTE,
)
from src.indicators import add_all_indicators
from src.utils import logger


def get_current_india_time() -> datetime:
    """Returns current timestamp in Indian Standard Time (IST, UTC+5:30)."""
    tz = pytz.timezone(TIMEZONE_INDIA)
    return datetime.now(tz)


def get_market_session_info() -> Dict[str, Any]:
    """Dynamically determines the current trading calendar status for the Indian National Stock Exchange (NSE).
    Never hardcodes dates or trading status.
    """
    now_ist = get_current_india_time()
    current_date = now_ist.date()
    weekday = now_ist.weekday()  # 0=Monday, ..., 5=Saturday, 6=Sunday
    current_time = now_ist.time()
    
    market_open = time(MARKET_OPEN_HOUR, MARKET_OPEN_MINUTE)
    market_close = time(MARKET_CLOSE_HOUR, MARKET_CLOSE_MINUTE)
    
    is_weekend = weekday in [5, 6]
    is_market_hours = (weekday < 5) and (market_open <= current_time <= market_close)
    
    # Determine latest completed trading date
    if weekday == 6:  # Sunday -> Friday was latest session
        latest_trading_date = current_date - timedelta(days=2)
        status_label = "CLOSED (Weekend - Sunday)"
    elif weekday == 5:  # Saturday -> Friday was latest session
        latest_trading_date = current_date - timedelta(days=1)
        status_label = "CLOSED (Weekend - Saturday)"
    else:  # Monday to Friday
        if current_time < market_open:
            # Pre-market: previous session was yesterday (or Friday if Monday)
            offset = 3 if weekday == 0 else 1
            latest_trading_date = current_date - timedelta(days=offset)
            status_label = "CLOSED (Pre-Market Session)"
        elif is_market_hours:
            latest_trading_date = current_date
            status_label = "LIVE / IN-SESSION"
        else:
            # After market hours: today's completed session
            latest_trading_date = current_date
            status_label = "CLOSED (Post-Market EOD)"
            
    return {
        "current_timestamp_ist": now_ist.strftime("%Y-%m-%d %H:%M:%S IST"),
        "current_date": str(current_date),
        "is_weekend": is_weekend,
        "is_market_open": is_market_hours,
        "status_label": status_label,
        "latest_expected_session_date": str(latest_trading_date)
    }


def generate_synthetic_stock_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Generates synthetic geometric Brownian motion stock data as an explicit offline fallback."""
    logger.warning(f"Generating simulated market data for {ticker} as offline fallback.")
    dates = pd.date_range(start=start, end=end, freq="B")  # Business days
    n = len(dates)
    if n == 0:
        dates = pd.date_range(start="2020-01-01", periods=1000, freq="B")
        n = len(dates)
        
    np.random.seed(hash(ticker) % 10000)
    mu = 0.12 / 252.0
    sigma = 0.22 / np.sqrt(252.0)
    returns = np.random.normal(mu, sigma, n)
    
    initial_prices = {
        "RELIANCE.NS": 1400.0,
        "TCS.NS": 2200.0,
        "INFY.NS": 750.0,
        "HDFCBANK.NS": 1100.0
    }
    s0 = initial_prices.get(ticker, 1000.0)
    price_series = s0 * np.exp(np.cumsum(returns))
    
    high = price_series * (1.0 + np.abs(np.random.normal(0, 0.008, n)))
    low = price_series * (1.0 - np.abs(np.random.normal(0, 0.008, n)))
    open_p = price_series * (1.0 + np.random.normal(0, 0.004, n))
    volume = np.random.lognormal(mean=14, sigma=0.5, size=n).astype(int)
    
    df = pd.DataFrame({
        "Date": dates,
        "Open": open_p,
        "High": high,
        "Low": low,
        "Close": price_series,
        "Adj Close": price_series,
        "Volume": volume
    })
    return df


def fetch_stock_data(
    ticker: str,
    start_date: str = DEFAULT_START_DATE,
    end_date: str = DEFAULT_END_DATE,
    use_cache: bool = True,
    allow_synthetic_fallback: bool = True
) -> pd.DataFrame:
    """Fetches historical stock market data with robust local caching.
    Standardizes columns and avoids lookahead bias.
    """
    clean_ticker_name = ticker.replace("^", "").replace(".", "_")
    cache_file = DATA_DIR / f"{clean_ticker_name}_{start_date}_{end_date}.csv"
    
    if use_cache and cache_file.exists():
        logger.info(f"Loading cached stock data from: {cache_file}")
        try:
            df = pd.read_csv(cache_file, parse_dates=["Date"])
            if len(df) > 50:
                return df
        except Exception as e:
            logger.warning(f"Failed to read cache {cache_file}: {e}. Refetching...")

    logger.info(f"Fetching {ticker} data from Yahoo Finance ({start_date} to {end_date})...")
    df = None
    try:
        raw_df = yf.download(
            ticker,
            start=start_date,
            end=end_date,
            progress=False,
            auto_adjust=False
        )
        
        if raw_df is not None and not raw_df.empty:
            # Flatten multi-index columns if present
            if isinstance(raw_df.columns, pd.MultiIndex):
                raw_df.columns = [c[0] for c in raw_df.columns]
                
            raw_df = raw_df.reset_index()
            # Standardize date column
            if "Date" not in raw_df.columns and "Datetime" in raw_df.columns:
                raw_df = raw_df.rename(columns={"Datetime": "Date"})
            raw_df["Date"] = pd.to_datetime(raw_df["Date"]).dt.tz_localize(None)
            
            # Ensure required numeric columns exist
            req_cols = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
            for col in req_cols:
                if col not in raw_df.columns and "Close" in raw_df.columns:
                    raw_df[col] = raw_df["Close"]
                raw_df[col] = pd.to_numeric(raw_df[col], errors="coerce")
                
            raw_df = raw_df.dropna(subset=["Close"]).sort_values("Date").reset_index(drop=True)
            if len(raw_df) > 50:
                df = raw_df
    except Exception as e:
        logger.error(f"Error fetching data via yfinance for {ticker}: {e}")
        
    if df is None or df.empty or len(df) < 50:
        if allow_synthetic_fallback:
            df = generate_synthetic_stock_data(ticker, start_date, end_date)
        else:
            raise RuntimeError(
                f"Could not retrieve verified live market data for {ticker} from Yahoo Finance, "
                f"and synthetic fallback is disabled for Current Market Decision Support."
            )

    # Save to cache
    try:
        df.to_csv(cache_file, index=False)
        logger.info(f"Cached data saved to: {cache_file} ({len(df)} rows)")
    except Exception as e:
        logger.warning(f"Could not write cache file: {e}")

    return df


def prepare_market_dataset(
    ticker: str,
    start_date: str = DEFAULT_START_DATE,
    end_date: str = DEFAULT_END_DATE,
    use_cache: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Downloads historical data, computes technical indicators, and performs
    a strictly chronological split into train, validation, and test datasets.
    Zero look-ahead bias and no data leakage.
    """
    raw_df = fetch_stock_data(ticker, start_date, end_date, use_cache=use_cache, allow_synthetic_fallback=True)
    feature_df = add_all_indicators(raw_df)
    
    total_len = len(feature_df)
    train_end = int(total_len * TRAIN_RATIO)
    val_end = int(total_len * (TRAIN_RATIO + VAL_RATIO))
    
    train_df = feature_df.iloc[:train_end].copy().reset_index(drop=True)
    val_df = feature_df.iloc[train_end:val_end].copy().reset_index(drop=True)
    test_df = feature_df.iloc[val_end:].copy().reset_index(drop=True)
    
    logger.info(
        f"Chronological split for {ticker}: "
        f"Train={len(train_df)} rows ({train_df['Date'].iloc[0].strftime('%Y-%m-%d')} to {train_df['Date'].iloc[-1].strftime('%Y-%m-%d')}), "
        f"Val={len(val_df)} rows ({val_df['Date'].iloc[0].strftime('%Y-%m-%d')} to {val_df['Date'].iloc[-1].strftime('%Y-%m-%d')}), "
        f"Test={len(test_df)} rows ({test_df['Date'].iloc[0].strftime('%Y-%m-%d')} to {test_df['Date'].iloc[-1].strftime('%Y-%m-%d')})"
    )
    
    return train_df, val_df, test_df


def fetch_current_market_observation(
    ticker: str,
    lookback_years: int = DEFAULT_LOOKBACK_YEARS
) -> Dict[str, Any]:
    """Retrieves the latest available verified market session data up to the current date,
    calculates indicator features across the lookback horizon, and formats the state vector
    for immediate model inference.
    """
    session_info = get_market_session_info()
    now_ist = get_current_india_time()
    
    # Calculate lookback start date
    start_dt = now_ist.date() - timedelta(days=lookback_years * 365)
    start_str = start_dt.strftime("%Y-%m-%d")
    
    # Target end date is today (yfinance end parameter is non-inclusive, so +1 day ensures latest closed session is downloaded)
    end_dt = now_ist.date() + timedelta(days=1)
    end_str = end_dt.strftime("%Y-%m-%d")
    
    # Attempt to fetch fresh data (do not rely on stale old cache)
    data_source = "LIVE YAHOO FINANCE API"
    try:
        raw_df = fetch_stock_data(
            ticker=ticker,
            start_date=start_str,
            end_date=end_str,
            use_cache=False,  # Query fresh
            allow_synthetic_fallback=False
        )
    except Exception as e:
        logger.warning(f"Live fetch failed for {ticker}: {e}. Checking local cache...")
        clean_ticker = ticker.replace("^", "").replace(".", "_")
        # Find any recent cache file
        matching_caches = list(DATA_DIR.glob(f"{clean_ticker}_*.csv"))
        if matching_caches:
            latest_cache = max(matching_caches, key=os.path.getmtime)
            raw_df = pd.read_csv(latest_cache, parse_dates=["Date"])
            data_source = f"CACHED HISTORICAL DATA ({latest_cache.name})"
        else:
            raise RuntimeError(
                f"Failed to retrieve current market data for {ticker} from Yahoo Finance: {e}"
            )
            
    # Compute indicators
    feature_df = add_all_indicators(raw_df)
    if feature_df.empty:
        raise ValueError(f"Insufficient historical data to compute indicators for {ticker}")
        
    latest_row = feature_df.iloc[-1]
    latest_session_date = pd.to_datetime(latest_row["Date"]).strftime("%Y-%m-%d")
    latest_price = float(latest_row["Close"])
    
    # Verify data freshness against calendar
    expected_session = session_info["latest_expected_session_date"]
    is_fresh = (latest_session_date == expected_session) or (pd.to_datetime(latest_session_date) >= pd.to_datetime(expected_session) - timedelta(days=3))
    
    freshness_status = "VERIFIED FRESH EOD" if is_fresh else f"DELAYED / STALE (Last Session: {latest_session_date})"
    
    return {
        "ticker": ticker,
        "latest_price": latest_price,
        "latest_session_date": latest_session_date,
        "session_info": session_info,
        "freshness_status": freshness_status,
        "data_source": data_source,
        "feature_df": feature_df,
        "latest_row": latest_row,
        "total_lookback_bars": len(feature_df)
    }
