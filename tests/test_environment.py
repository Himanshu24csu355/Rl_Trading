"""Tests for Gymnasium RiskSensitiveTradingEnv environment dynamics.
"""

import pytest
import pandas as pd
import numpy as np
from src.indicators import add_all_indicators
from src.trading_env import RiskSensitiveTradingEnv


@pytest.fixture
def prepared_features_df():
    np.random.seed(42)
    n = 120
    dates = pd.date_range("2023-01-01", periods=n, freq="B")
    close = 1500.0 * np.exp(np.cumsum(np.random.normal(0.0005, 0.015, n)))
    high = close * 1.01
    low = close * 0.99
    open_p = close * 1.001
    volume = np.random.randint(100000, 1000000, n)
    
    raw_df = pd.DataFrame({
        "Date": dates,
        "Open": open_p,
        "High": high,
        "Low": low,
        "Close": close,
        "Volume": volume
    })
    return add_all_indicators(raw_df)


def test_env_reset_and_dimensions(prepared_features_df):
    env = RiskSensitiveTradingEnv(df=prepared_features_df, initial_capital=50000.0)
    obs, info = env.reset()
    
    assert obs.shape == env.observation_space.shape
    assert info["portfolio_value"] == 50000.0
    assert info["cash"] == 50000.0
    assert info["shares"] == 0


def test_env_step_execution(prepared_features_df):
    env = RiskSensitiveTradingEnv(df=prepared_features_df, initial_capital=100000.0)
    obs, _ = env.reset()
    
    # Action 1 = BUY
    obs, reward, terminated, truncated, info = env.step(1)
    assert not np.isnan(reward)
    assert env.shares > 0
    assert env.cash < 100000.0
    
    # Portfolio accounting invariant: cash + shares * price == portfolio_value
    current_price = float(prepared_features_df.iloc[env.current_step]["Close"])
    expected_val = env.cash + (env.shares * current_price)
    assert pytest.approx(env.portfolio_value, abs=1e-2) == expected_val


def test_env_episode_completion(prepared_features_df):
    env = RiskSensitiveTradingEnv(df=prepared_features_df, initial_capital=100000.0)
    obs, _ = env.reset()
    
    done = False
    step_count = 0
    while not done:
        action = np.random.choice([0, 1, 2])
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        step_count += 1
        
    assert step_count > 0
    history_df = env.get_history_df()
    assert len(history_df) == step_count
