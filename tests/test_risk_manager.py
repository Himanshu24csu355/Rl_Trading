"""Tests for RiskManager mathematical penalties and trade constraints.
"""

import pytest
import numpy as np
from src.risk_manager import RiskManager


def test_risk_manager_initialization():
    rm = RiskManager(risk_level="Low")
    assert rm.lambda_penalty > 0
    assert rm.downside_weight > 0
    assert rm.max_drawdown_limit == 0.12


def test_downside_penalty_asymmetry():
    rm = RiskManager(risk_level="Medium")
    rm.reset(initial_capital=100000.0)
    
    # Positive return: downside penalty should be 0 (only drawdown if any)
    penalty_pos = rm.compute_risk_penalty(step_return=0.03, current_portfolio_value=103000.0)
    
    # Negative return of same magnitude: downside penalty should be positive
    penalty_neg = rm.compute_risk_penalty(step_return=-0.03, current_portfolio_value=97000.0)
    
    assert penalty_pos < penalty_neg
    assert penalty_neg > 0.0


def test_drawdown_calculation():
    rm = RiskManager(risk_level="Low")
    rm.reset(initial_capital=100000.0)
    
    # Portfolio drops from 100,000 to 80,000
    dd = rm.calculate_drawdown(80000.0)
    assert pytest.approx(dd, abs=1e-4) == 0.20
    
    # Portfolio peaks at 120,000 then drops to 90,000
    rm.update_portfolio(120000.0, 0.20)
    dd2 = rm.calculate_drawdown(90000.0)
    assert pytest.approx(dd2, abs=1e-4) == (120000.0 - 90000.0) / 120000.0


def test_action_validation_constraints():
    rm = RiskManager(risk_level="Low")
    rm.reset(initial_capital=100000.0)
    
    # 1. Buy validation with ample cash
    act, qty, msg = rm.validate_action(
        action=1,
        current_cash=100000.0,
        current_shares=0,
        stock_price=2000.0,
        portfolio_value=100000.0
    )
    assert act == 1
    assert qty > 0
    
    # 2. Buy validation with 0 cash
    act, qty, msg = rm.validate_action(
        action=1,
        current_cash=0.0,
        current_shares=10,
        stock_price=2000.0,
        portfolio_value=20000.0
    )
    assert act == 0
    assert qty == 0
    assert "Insufficient" in msg
    
    # 3. Sell validation with 0 shares owned (no short selling)
    act, qty, msg = rm.validate_action(
        action=2,
        current_cash=50000.0,
        current_shares=0,
        stock_price=2000.0,
        portfolio_value=50000.0
    )
    assert act == 0
    assert qty == 0
    assert "No shares" in msg
