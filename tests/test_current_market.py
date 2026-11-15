"""Unit tests for Current Market Decision Support, Trading Calendar,
Virtual Order Sizing, and Policy Distribution.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime

from config import SUPPORTED_TICKERS, DEFAULT_INITIAL_CAPITAL
from src.data_loader import get_market_session_info, fetch_current_market_observation
from src.risk_manager import RiskManager
from src.train_model import get_model_info, get_policy_action_distribution, load_trained_agent
from src.evaluation import generate_empirical_findings


def test_market_session_info_structure():
    """Verifies that the market calendar status helper returns all required fields."""
    info = get_market_session_info()
    assert "current_timestamp_ist" in info
    assert "current_date" in info
    assert "is_weekend" in info
    assert "is_market_open" in info
    assert "status_label" in info
    assert "latest_expected_session_date" in info
    # Ensure date format is YYYY-MM-DD
    datetime.strptime(info["latest_expected_session_date"], "%Y-%m-%d")


def test_virtual_order_sizing_buy_with_sufficient_cash():
    """Tests affordable share calculation and transaction fee on BUY."""
    rm = RiskManager(risk_level="Medium")
    # Virtual capital = ₹10,000, Stock price = ₹2,000
    res = rm.calculate_virtual_order_sizing(
        action=1,  # BUY
        virtual_capital=10000.0,
        current_price=2000.0,
        existing_shares=0
    )
    assert res["raw_action"] == "BUY"
    assert res["executable_action"] == "BUY"
    assert res["status"] == "APPROVED"
    assert res["shares_traded"] >= 1
    assert res["shares_traded"] == int(res["shares_traded"])  # Must be whole shares
    assert res["transaction_fee"] > 0
    assert res["remaining_cash"] < 10000.0
    # Portfolio accounting invariant
    assert pytest.approx(res["portfolio_value"], abs=1e-2) == res["remaining_cash"] + (res["new_shares_held"] * 2000.0)


def test_virtual_order_sizing_buy_insufficient_cash():
    """Tests that BUY action is blocked when cash cannot buy even 1 whole share."""
    rm = RiskManager(risk_level="Medium")
    # Virtual capital = ₹1,000, Stock price = ₹2,500
    res = rm.calculate_virtual_order_sizing(
        action=1,  # BUY
        virtual_capital=1000.0,
        current_price=2500.0,
        existing_shares=0
    )
    assert res["raw_action"] == "BUY"
    assert res["executable_action"] == "HOLD"
    assert "BLOCKED" in res["status"]
    assert res["shares_traded"] == 0
    assert res["remaining_cash"] == 1000.0
    assert res["portfolio_value"] == 1000.0


def test_virtual_order_sizing_sell_no_shares_owned():
    """Tests that SELL action is rejected when no shares are held (no short selling)."""
    rm = RiskManager(risk_level="Low")
    res = rm.calculate_virtual_order_sizing(
        action=2,  # SELL
        virtual_capital=50000.0,
        current_price=1500.0,
        existing_shares=0
    )
    assert res["raw_action"] == "SELL"
    assert res["executable_action"] == "HOLD"
    assert "REJECTED" in res["status"]
    assert res["shares_traded"] == 0
    assert res["remaining_cash"] == 50000.0


def test_virtual_order_sizing_sell_shares_owned():
    """Tests that SELL action executes liquidation when shares are owned."""
    rm = RiskManager(risk_level="Low")
    res = rm.calculate_virtual_order_sizing(
        action=2,  # SELL
        virtual_capital=10000.0,
        current_price=1000.0,
        existing_shares=10
    )
    assert res["raw_action"] == "SELL"
    assert res["executable_action"] == "SELL"
    assert res["shares_traded"] > 0
    assert res["remaining_cash"] > 10000.0


def test_model_info_detection():
    """Tests checking model availability and metadata."""
    # Existing model
    info = get_model_info("RELIANCE.NS", "Low")
    assert info["exists"] is True
    assert "model_path" in info
    
    # Missing model
    info_missing = get_model_info("NONEXISTENT_TICKER.NS", "Low")
    assert info_missing["exists"] is False
    assert "No pre-trained model found" in info_missing["message"]


def test_generate_empirical_findings():
    """Tests that empirical findings are dynamically generated from numbers."""
    dummy_results = [
        {
            "strategy_name": "Buy & Hold Benchmark",
            "initial_capital": 100000.0,
            "final_portfolio_value": 90000.0,
            "total_return_pct": -10.0,
            "max_drawdown_pct": 25.0,
            "sharpe_ratio": -0.8,
            "annualized_volatility_pct": 24.0,
            "total_trades": 1,
            "total_fees_paid": 100.0
        },
        {
            "strategy_name": "Standard PPO (Risk-Neutral)",
            "initial_capital": 100000.0,
            "final_portfolio_value": 95000.0,
            "total_return_pct": -5.0,
            "max_drawdown_pct": 22.0,
            "sharpe_ratio": -0.4,
            "annualized_volatility_pct": 20.0,
            "total_trades": 45,
            "total_fees_paid": 550.0
        },
        {
            "strategy_name": "Risk-Sensitive PPO (Medium)",
            "initial_capital": 100000.0,
            "final_portfolio_value": 98000.0,
            "total_return_pct": -2.0,
            "max_drawdown_pct": 14.0,
            "sharpe_ratio": 0.1,
            "annualized_volatility_pct": 15.0,
            "total_trades": 18,
            "total_fees_paid": 220.0
        }
    ]
    commentary = generate_empirical_findings(dummy_results)
    assert "Drawdown Reduction" in commentary or "Drawdown" in commentary
    assert "Volatility" in commentary
    assert "Transaction Friction" in commentary
