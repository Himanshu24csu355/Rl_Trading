"""Backtesting Engine and Strategy Simulation.
Executes step-by-step trading simulation for RL Agents and Buy & Hold benchmarks.
"""

from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np
from stable_baselines3 import PPO

from config import DEFAULT_INITIAL_CAPITAL, TRANSACTION_FEE_PERCENT
from src.trading_env import RiskSensitiveTradingEnv
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
            # Fallback heuristic if no agent supplied
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
    
    history = []
    portfolio_values = []
    
    for i, row in df.iterrows():
        price = float(row["Close"])
        p_val = cash + (shares * price)
        portfolio_values.append(p_val)
        
        trade_type = "BUY" if i == 0 else "HOLD"
        shares_traded = shares if i == 0 else 0
        fee = (shares * p0 * transaction_fee) if i == 0 else 0.0
        
        history.append({
            "step": i,
            "date": row.get("Date", f"Day_{i}"),
            "price": price,
            "trade_type": trade_type,
            "shares_traded": shares_traded,
            "shares_held": shares,
            "cash": cash,
            "portfolio_value": p_val,
            "drawdown": 0.0  # calculated below
        })
        
    history_df = pd.DataFrame(history)
    portfolio_series = np.array(portfolio_values)
    
    # Calculate drawdowns
    peaks = np.maximum.accumulate(portfolio_series)
    drawdowns = (peaks - portfolio_series) / np.where(peaks == 0, 1e-8, peaks)
    history_df["drawdown"] = drawdowns
    
    # Returns
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
    current_cash: float
) -> Dict[str, Any]:
    """Generates an intuitive, factor-based explanation of the AI agent's decision."""
    rsi = float(latest_row.get("rsi_14", 0.5)) * 100.0
    macd = float(latest_row.get("macd", 0.0))
    macd_signal = float(latest_row.get("macd_signal", 0.0))
    vol = float(latest_row.get("volatility_20", 0.2)) * 100.0
    sma_10_ratio = float(latest_row.get("sma_10_ratio", 0.0)) * 100.0
    
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
            reasons.append("Agent policy identified an attractive risk-reward entry opportunity under current market volatility.")
            
    elif action == 2:  # SELL
        if rsi > 70.0:
            reasons.append(f"RSI is severely overbought ({rsi:.1f}), signaling high reversal risk.")
            
        if macd < macd_signal:
            reasons.append("MACD has crossed below the signal line (bearish momentum divergence).")
            
        if vol > 25.0 and risk_level in ["Low", "Medium"]:
            reasons.append(f"Elevated annualized volatility ({vol:.1f}%) exceeds {risk_level} risk tolerance parameters.")
            
        if not reasons:
            reasons.append("Agent locked in capital preservation mode to mitigate anticipated downside drawdown.")
            
    else:  # HOLD
        if shares_held == 0:
            reasons.append(f"Neutral signal: Market risk/reward profile does not justify capital deployment under {risk_level} risk constraints.")
        else:
            reasons.append(f"Maintaining established position: Trend remains stable and trailing risk metrics are well within safe bounds.")
            
    return {
        "action": action_str,
        "confidence": "High" if abs(macd - macd_signal) > 0.005 or rsi < 30 or rsi > 70 else "Moderate",
        "primary_reasons": reasons,
        "indicators_summary": {
            "RSI (14)": f"{rsi:.1f}",
            "MACD vs Signal": f"{macd:.4f} / {macd_signal:.4f}",
            "Rolling Volatility": f"{vol:.1f}%",
            "SMA 10 Distance": f"{sma_10_ratio:+.2f}%"
        }
    }
