"""Backtesting Engine and Current Market Decision Support.
Executes step-by-step simulations, benchmark evaluations, and real-time inference.
"""

from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np
from stable_baselines3 import PPO

from config import DEFAULT_INITIAL_CAPITAL, TRANSACTION_FEE_PERCENT, TECHNICAL_INDICATORS, RISK_LEVELS
from src.trading_env import RiskSensitiveTradingEnv
from src.risk_manager import RiskManager
from src.train_model import get_policy_action_distribution
from src.data_loader import fetch_current_market_observation
from src.utils import logger, calculate_max_drawdown, calculate_sharpe_ratio, calculate_sortino_ratio


def run_agent_simulation(
    df: pd.DataFrame,
    agent: Optional[PPO] = None,
    risk_level: str = "Medium",
    initial_capital: float = DEFAULT_INITIAL_CAPITAL,
    transaction_fee: float = TRANSACTION_FEE_PERCENT,
    strategy_name: str = "Risk-Sensitive PPO"
) -> Dict[str, Any]:
    """Runs a complete chronological backtest simulation for an RL agent on a test dataset."""
    env = RiskSensitiveTradingEnv(
        df=df,
        initial_capital=initial_capital,
        risk_level=risk_level,
        transaction_fee=transaction_fee
    )
    
    obs, info = env.reset()
    done = False
    
    while not done:
        if agent is not None:
            action, _ = agent.predict(obs, deterministic=True)
            action = int(action)
        else:
            action = 0
            
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
    history_df = env.get_history_df()
    
    # Calculate performance metrics
    final_val = float(env.portfolio_value)
    total_return_pct = (final_val - initial_capital) / initial_capital
    portfolio_series = history_df["portfolio_value"].values if not history_df.empty else np.array([initial_capital])
    step_returns = history_df["step_return"].values if not history_df.empty else np.array([0.0])
    
    max_dd = calculate_max_drawdown(portfolio_series)
    sharpe = calculate_sharpe_ratio(step_returns)
    sortino = calculate_sortino_ratio(step_returns)
    volatility = float(np.std(step_returns) * np.sqrt(252)) if len(step_returns) > 1 else 0.0
    
    buys = int(len(history_df[history_df["trade_type"] == "BUY"])) if not history_df.empty else 0
    sells = int(len(history_df[history_df["trade_type"] == "SELL"])) if not history_df.empty else 0
    holds = int(len(history_df[history_df["trade_type"] == "HOLD"])) if not history_df.empty else 0
    total_fees = float(history_df["fee_paid"].sum()) if not history_df.empty else 0.0
    
    return {
        "strategy_name": strategy_name,
        "initial_capital": initial_capital,
        "final_portfolio_value": round(final_val, 2),
        "total_profit_loss": round(final_val - initial_capital, 2),
        "total_return_pct": round(total_return_pct * 100, 2),
        "max_drawdown_pct": round(max_dd * 100, 2),
        "sharpe_ratio": round(sharpe, 2),
        "sortino_ratio": round(sortino, 2),
        "annualized_volatility_pct": round(volatility * 100, 2),
        "total_trades": env.total_trades,
        "buy_trades": buys,
        "sell_trades": sells,
        "hold_steps": holds,
        "total_fees_paid": round(total_fees, 2),
        "final_cash": round(float(env.cash), 2),
        "final_shares": int(env.shares),
        "history_df": history_df
    }


def run_buy_and_hold_simulation(
    df: pd.DataFrame,
    initial_capital: float = DEFAULT_INITIAL_CAPITAL,
    transaction_fee: float = TRANSACTION_FEE_PERCENT
) -> Dict[str, Any]:
    """Runs a standardized Buy & Hold baseline simulation over the exact same period."""
    df = df.copy().reset_index(drop=True)
    if len(df) == 0:
        raise ValueError("DataFrame is empty for Buy & Hold simulation")
        
    p0 = float(df.iloc[0]["Close"])
    cost_per_share = p0 * (1.0 + transaction_fee)
    shares = int(initial_capital // cost_per_share)
    cash = initial_capital - (shares * cost_per_share)
    entry_fee = shares * p0 * transaction_fee
    
    history = []
    portfolio_values = []
    
    for i, row in df.iterrows():
        price = float(row["Close"])
        p_val = cash + (shares * price)
        portfolio_values.append(p_val)
        
        trade_type = "BUY" if i == 0 else "HOLD"
        shares_traded = shares if i == 0 else 0
        fee = entry_fee if i == 0 else 0.0
        
        history.append({
            "step": i,
            "date": row.get("Date", f"Day_{i}"),
            "price": price,
            "trade_type": trade_type,
            "shares_traded": shares_traded,
            "shares_held": shares,
            "fee_paid": fee,
            "cash": cash,
            "portfolio_value": p_val,
            "drawdown": 0.0
        })
        
    history_df = pd.DataFrame(history)
    portfolio_series = np.array(portfolio_values)
    
    peaks = np.maximum.accumulate(portfolio_series)
    drawdowns = (peaks - portfolio_series) / np.where(peaks == 0, 1e-8, peaks)
    history_df["drawdown"] = drawdowns
    
    step_returns = np.diff(portfolio_series) / np.maximum(1e-8, portfolio_series[:-1])
    step_returns = np.concatenate([[0.0], step_returns])
    history_df["step_return"] = step_returns
    
    final_val = float(portfolio_series[-1])
    total_return_pct = (final_val - initial_capital) / initial_capital
    max_dd = calculate_max_drawdown(portfolio_series)
    sharpe = calculate_sharpe_ratio(step_returns)
    sortino = calculate_sortino_ratio(step_returns)
    volatility = float(np.std(step_returns) * np.sqrt(252)) if len(step_returns) > 1 else 0.0
    
    return {
        "strategy_name": "Buy & Hold Benchmark",
        "initial_capital": initial_capital,
        "final_portfolio_value": round(final_val, 2),
        "total_profit_loss": round(final_val - initial_capital, 2),
        "total_return_pct": round(total_return_pct * 100, 2),
        "max_drawdown_pct": round(max_dd * 100, 2),
        "sharpe_ratio": round(sharpe, 2),
        "sortino_ratio": round(sortino, 2),
        "annualized_volatility_pct": round(volatility * 100, 2),
        "total_trades": 1,
        "buy_trades": 1,
        "sell_trades": 0,
        "hold_steps": len(df) - 1,
        "total_fees_paid": round(entry_fee, 2),
        "final_cash": round(float(cash), 2),
        "final_shares": int(shares),
        "history_df": history_df
    }


def explain_ai_decision(
    latest_row: pd.Series,
    action: int,
    risk_level: str,
    shares_held: int,
    portfolio_value: float,
    current_cash: float,
    action_probs: Optional[Dict[str, float]] = None
) -> Dict[str, Any]:
    """Generates an intuitive, factor-based explanation of the AI agent's decision."""
    rsi = float(latest_row.get("rsi_14", 0.5)) * 100.0
    macd = float(latest_row.get("macd", 0.0))
    macd_signal = float(latest_row.get("macd_signal", 0.0))
    vol = float(latest_row.get("volatility_20", 0.2)) * 100.0
    sma_10_ratio = float(latest_row.get("sma_10_ratio", 0.0)) * 100.0
    sma_30_ratio = float(latest_row.get("sma_30_ratio", 0.0)) * 100.0
    
    action_str = ["HOLD", "BUY", "SELL"][action]
    reasons = []
    
    if action == 1:  # BUY
        if rsi < 35.0:
            reasons.append(f"RSI is in oversold territory ({rsi:.1f}), indicating potential price undervaluation.")
        elif rsi < 60.0:
            reasons.append(f"RSI ({rsi:.1f}) indicates healthy upward momentum without overbought risk.")
            
        if macd > macd_signal:
            reasons.append("MACD is trending above the signal line (bullish crossover momentum).")
            
        if sma_10_ratio > 0:
            reasons.append(f"Price is trading {sma_10_ratio:.1f}% above its 10-day SMA, confirming short-term uptrend.")
            
        if not reasons:
            reasons.append("Policy network identified an attractive risk-reward entry opportunity under current market conditions.")
            
    elif action == 2:  # SELL
        if rsi > 70.0:
            reasons.append(f"RSI is overbought ({rsi:.1f}), signaling reversal and mean-reversion risk.")
            
        if macd < macd_signal:
            reasons.append("MACD is trending below the signal line (bearish momentum divergence).")
            
        if vol > 25.0 and risk_level in ["Low", "Medium"]:
            reasons.append(f"Elevated annualized volatility ({vol:.1f}%) triggers defensive capital preservation under {risk_level} risk.")
            
        if sma_10_ratio < 0:
            reasons.append(f"Price broke {abs(sma_10_ratio):.1f}% below the 10-day moving average.")
            
        if not reasons:
            reasons.append("Agent activated risk mitigation to reduce equity exposure and protect portfolio capital.")
            
    else:  # HOLD
        if shares_held == 0:
            reasons.append(f"Neutral observation: Current market indicators do not present a sufficiently strong risk-reward ratio to deploy cash under {risk_level} risk.")
        else:
            reasons.append(f"Maintaining position: Market conditions remain within acceptable risk bounds without clear exit triggers.")
            
    # Calculate confidence based on policy action probability if provided
    if action_probs:
        max_p = action_probs.get(action_str, 0.5)
        conf_label = "High Conviction" if max_p > 0.60 else ("Moderate" if max_p > 0.40 else "Low / Indecisive")
    else:
        conf_label = "Moderate"
        
    return {
        "action": action_str,
        "confidence": conf_label,
        "action_probabilities": action_probs or {},
        "primary_reasons": reasons,
        "indicators_summary": {
            "RSI (14)": f"{rsi:.1f}",
            "MACD vs Signal": f"{macd:.4f} / {macd_signal:.4f}",
            "Rolling Volatility": f"{vol:.1f}%",
            "SMA 10 Distance": f"{sma_10_ratio:+.2f}%",
            "SMA 30 Distance": f"{sma_30_ratio:+.2f}%"
        }
    }


def evaluate_current_market_decision(
    ticker: str,
    agent: PPO,
    risk_level: str = "Medium",
    virtual_capital: float = DEFAULT_INITIAL_CAPITAL,
    existing_shares: int = 0,
    lookback_years: int = 3
) -> Dict[str, Any]:
    """Retrieves the latest verified market observation, executes genuine PPO inference,
    calculates action probabilities, and simulates virtual order execution.
    """
    # 1. Retrieve latest verified market data
    obs_data = fetch_current_market_observation(ticker=ticker, lookback_years=lookback_years)
    latest_row = obs_data["latest_row"]
    latest_price = float(obs_data["latest_price"])
    latest_date = obs_data["latest_session_date"]
    
    # 2. Formulate 18-dimensional state vector matching training observation space exactly
    tech_features = latest_row[TECHNICAL_INDICATORS].values.astype(np.float32)
    tech_features = np.nan_to_num(tech_features, nan=0.0, posinf=5.0, neginf=-5.0)
    
    # Account context for virtual investor
    holdings_value = existing_shares * latest_price
    current_portfolio_value = max(1e-8, virtual_capital + holdings_value)
    cash_ratio = float(np.clip(virtual_capital / current_portfolio_value, 0.0, 1.0))
    holdings_ratio = float(np.clip(holdings_value / current_portfolio_value, 0.0, 1.0))
    current_drawdown = 0.0  # Fresh virtual evaluation starts at peak
    
    account_features = np.array([cash_ratio, holdings_ratio, current_drawdown], dtype=np.float32)
    state_vector = np.concatenate([tech_features, account_features])
    
    # 3. Model Inference & Policy Action Distribution
    raw_action, action_probs, confidence = get_policy_action_distribution(agent, state_vector)
    
    # 4. Risk Manager validation & virtual order sizing
    rm = RiskManager(risk_level=risk_level)
    order_sizing = rm.calculate_virtual_order_sizing(
        action=raw_action,
        virtual_capital=virtual_capital,
        current_price=latest_price,
        existing_shares=existing_shares
    )
    
    # 5. Composite Risk Score
    vol_val = float(latest_row.get("volatility_20", 0.20))
    risk_score = rm.compute_composite_risk_score(
        current_portfolio_value=current_portfolio_value,
        stock_volatility=vol_val,
        holdings_ratio=holdings_ratio
    )
    
    # 6. Explanation
    explanation = explain_ai_decision(
        latest_row=latest_row,
        action=raw_action,
        risk_level=risk_level,
        shares_held=existing_shares,
        portfolio_value=current_portfolio_value,
        current_cash=virtual_capital,
        action_probs=action_probs
    )
    
    return {
        "ticker": ticker,
        "latest_price": latest_price,
        "latest_session_date": latest_date,
        "session_info": obs_data["session_info"],
        "freshness_status": obs_data["freshness_status"],
        "data_source": obs_data["data_source"],
        "total_lookback_bars": obs_data.get("total_lookback_bars", len(obs_data.get("feature_df", []))),
        "raw_action": raw_action,
        "raw_action_name": ["HOLD", "BUY", "SELL"][raw_action],
        "action_probabilities": action_probs,
        "confidence": confidence,
        "order_sizing": order_sizing,
        "risk_score": risk_score,
        "explanation": explanation,
        "virtual_capital": virtual_capital,
        "risk_level": risk_level,
        "latest_row": latest_row,
        "feature_df": obs_data["feature_df"]
    }
