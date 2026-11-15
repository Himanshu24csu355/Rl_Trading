"""Tests for backtesting engine and benchmark comparison logic.
"""

import pytest
import pandas as pd
import numpy as np
from src.indicators import add_all_indicators
from src.backtest import run_buy_and_hold_simulation, run_agent_simulation, explain_ai_decision


@pytest.fixture
def backtest_df():
    np.random.seed(99)
    n = 100
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    close = 1000.0 * np.exp(np.cumsum(np.random.normal(0.001, 0.01, n)))
    high = close * 1.01
    low = close * 0.99
    open_p = close
    volume = np.random.randint(50000, 200000, n)
    
    raw = pd.DataFrame({
        "Date": dates,
        "Open": open_p,
        "High": high,
        "Low": low,
        "Close": close,
        "Volume": volume
    })
    return add_all_indicators(raw)


def test_buy_and_hold_simulation(backtest_df):
    res = run_buy_and_hold_simulation(backtest_df, initial_capital=100000.0)
    assert res["strategy_name"] == "Buy & Hold Benchmark"
    assert res["total_trades"] == 1
    assert res["final_portfolio_value"] > 0
    assert not res["history_df"].empty
    assert "drawdown" in res["history_df"].columns


def test_agent_simulation_runs(backtest_df):
    res = run_agent_simulation(backtest_df, agent=None, initial_capital=100000.0)
    assert res["final_portfolio_value"] > 0
    assert "max_drawdown_pct" in res
    assert "sharpe_ratio" in res


def test_explain_ai_decision(backtest_df):
    row = backtest_df.iloc[-1]
    explanation = explain_ai_decision(
        latest_row=row,
        action=1,
        risk_level="Medium",
        shares_held=10,
        portfolio_value=120000.0,
        current_cash=50000.0
    )
    assert explanation["action"] == "BUY"
    assert len(explanation["primary_reasons"]) > 0
    assert "RSI (14)" in explanation["indicators_summary"]
