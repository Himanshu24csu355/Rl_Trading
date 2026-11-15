"""Risk Management Module for Portfolio Monitoring, Risk Penalties, and Constraints.
Enforces position limits, downside risk penalties, drawdown tracking, and capital conservation.
Distinguishes raw agent intentions from portfolio-permitted executable actions.
"""

from typing import Dict, Any, Tuple, Optional
import numpy as np
from config import RISK_LEVELS, TRANSACTION_FEE_PERCENT
from src.utils import logger


class RiskManager:
    """Manages risk constraints, downside risk penalties, and real-time risk scores."""

    def __init__(
        self,
        risk_level: str = "Medium",
        transaction_fee: float = TRANSACTION_FEE_PERCENT,
        min_cash_buffer_ratio: float = 0.01  # Keep 1% cash reserve for friction/slippage
    ):
        if risk_level not in RISK_LEVELS:
            raise ValueError(f"Unknown risk level: {risk_level}. Must be one of {list(RISK_LEVELS.keys())}")
            
        self.risk_level = risk_level
        self.config = RISK_LEVELS[risk_level]
        self.lambda_penalty = self.config["lambda"]
        self.downside_weight = self.config["downside_weight"]
        self.drawdown_weight = self.config["drawdown_weight"]
        self.max_drawdown_limit = self.config["max_drawdown_limit"]
        self.transaction_fee = transaction_fee
        self.min_cash_buffer_ratio = min_cash_buffer_ratio
        
        self.peak_portfolio_value = 0.0
        self.rolling_returns = []

    def reset(self, initial_capital: float) -> None:
        """Resets risk tracking for a fresh episode or simulation."""
        self.peak_portfolio_value = max(1e-8, float(initial_capital))
        self.rolling_returns = []

    def update_portfolio(self, current_portfolio_value: float, step_return: float) -> None:
        """Updates internal peak value and historical return buffer."""
        if current_portfolio_value > self.peak_portfolio_value:
            self.peak_portfolio_value = current_portfolio_value
        self.rolling_returns.append(step_return)
        if len(self.rolling_returns) > 60:
            self.rolling_returns.pop(0)

    def calculate_drawdown(self, current_portfolio_value: float) -> float:
        """Calculates current drawdown from historical peak as a fraction [0, 1]."""
        if self.peak_portfolio_value <= 0:
            return 0.0
        drawdown = (self.peak_portfolio_value - current_portfolio_value) / self.peak_portfolio_value
        return max(0.0, float(drawdown))

    def compute_risk_penalty(
        self,
        step_return: float,
        current_portfolio_value: float,
        holdings_ratio: float = 1.0
    ) -> float:
        """Computes the calibrated mathematical risk penalty for the RL reward function:
        Risk Penalty = lambda * [ downside_weight * min(0, step_return)^2 * 5.0 +
                                  drawdown_weight * Drawdown^2 +
                                  breach_penalty * holdings_ratio ]
        
        Note: The breach penalty is scaled by holdings_ratio so that an agent holding cash
        during deep market drawdowns is rewarded for capital preservation rather than penalized.
        """
        # 1. Downside return component (penalizes negative returns quadratically, ignoring upside)
        downside = min(0.0, step_return)
        downside_penalty = (downside ** 2) * self.downside_weight * 5.0

        # 2. Drawdown component (penalizes trailing behind peak portfolio value)
        current_dd = self.calculate_drawdown(current_portfolio_value)
        dd_penalty = (current_dd ** 2) * self.drawdown_weight

        # 3. Breach penalty if drawdown exceeds tolerance limit, scaled by equity exposure
        breach_penalty = 0.0
        if current_dd > self.max_drawdown_limit:
            # Penalizes remaining in high equity when drawdown threshold is breached
            breach_penalty = (current_dd - self.max_drawdown_limit) * 1.5 * max(0.1, holdings_ratio)

        total_penalty = self.lambda_penalty * (downside_penalty + dd_penalty + breach_penalty)
        return float(total_penalty)

    def validate_action(
        self,
        action: int,
        current_cash: float,
        current_shares: int,
        stock_price: float,
        portfolio_value: float
    ) -> Tuple[int, int, str]:
        """Validates raw trading actions against capital constraints, position limits,
        and no-short-selling rules.
        
        Action Space:
          0 = HOLD
          1 = BUY
          2 = SELL
          
        Returns:
          (executable_action, num_shares_to_trade, reason)
        """
        if action == 1:  # BUY
            effective_cash = current_cash * (1.0 - self.min_cash_buffer_ratio)
            cost_per_share = stock_price * (1.0 + self.transaction_fee)
            
            if cost_per_share <= 0 or effective_cash < cost_per_share:
                return 0, 0, "BUY blocked: Insufficient available cash to purchase at least 1 whole share"
                
            # For discrete action, allocate 50% of available effective cash per buy signal
            # to promote prudent dollar-cost averaging and position sizing
            target_spend = effective_cash * 0.50
            shares_to_buy = int(target_spend // cost_per_share)
            if shares_to_buy < 1:
                shares_to_buy = int(effective_cash // cost_per_share)
                
            if shares_to_buy < 1:
                return 0, 0, "BUY blocked: Available capital cannot purchase 1 whole share"
                
            return 1, shares_to_buy, f"BUY approved for {shares_to_buy} shares"

        elif action == 2:  # SELL
            if current_shares <= 0:
                return 0, 0, "SELL blocked: No shares currently owned (Short selling prohibited)"
                
            # Liquidate 50% or all remaining shares if position is small
            shares_to_sell = max(1, int(current_shares * 0.50))
            if current_shares <= 2:
                shares_to_sell = current_shares
                
            return 2, shares_to_sell, f"SELL approved for {shares_to_sell} shares"

        else:  # HOLD
            return 0, 0, "HOLD: Maintain current asset allocation"

    def calculate_virtual_order_sizing(
        self,
        action: int,
        virtual_capital: float,
        current_price: float,
        existing_shares: int = 0
    ) -> Dict[str, Any]:
        """Calculates exact order sizing, affordable whole shares, fees, and cash balances
        for educational virtual investment simulation without real execution.
        """
        raw_action_name = ["HOLD", "BUY", "SELL"][action]
        cost_per_share = current_price * (1.0 + self.transaction_fee)
        
        if action == 1:  # BUY
            effective_capital = virtual_capital * (1.0 - self.min_cash_buffer_ratio)
            # Allocation rule: Invest up to 50% of available cash, or 100% if capital is small (< 2 shares)
            target_outlay = effective_capital * 0.50
            shares_to_buy = int(target_outlay // cost_per_share)
            if shares_to_buy < 1 and effective_capital >= cost_per_share:
                shares_to_buy = int(effective_capital // cost_per_share)
                
            if shares_to_buy >= 1:
                gross_trade_value = shares_to_buy * current_price
                fee = gross_trade_value * self.transaction_fee
                total_outlay = gross_trade_value + fee
                remaining_cash = virtual_capital - total_outlay
                new_shares = existing_shares + shares_to_buy
                portfolio_val = remaining_cash + (new_shares * current_price)
                
                return {
                    "raw_action": raw_action_name,
                    "executable_action": "BUY",
                    "status": "APPROVED",
                    "shares_traded": shares_to_buy,
                    "new_shares_held": new_shares,
                    "gross_trade_value": round(gross_trade_value, 2),
                    "transaction_fee": round(fee, 2),
                    "total_outlay": round(total_outlay, 2),
                    "remaining_cash": round(remaining_cash, 2),
                    "portfolio_value": round(portfolio_val, 2),
                    "reason": f"Allocated virtual funds to acquire {shares_to_buy} whole shares."
                }
            else:
                return {
                    "raw_action": raw_action_name,
                    "executable_action": "HOLD",
                    "status": "BLOCKED (Insufficient Capital)",
                    "shares_traded": 0,
                    "new_shares_held": existing_shares,
                    "gross_trade_value": 0.0,
                    "transaction_fee": 0.0,
                    "total_outlay": 0.0,
                    "remaining_cash": round(virtual_capital, 2),
                    "portfolio_value": round(virtual_capital + (existing_shares * current_price), 2),
                    "reason": f"Virtual capital (₹{virtual_capital:,.2f}) cannot afford at least 1 whole share at ₹{current_price:,.2f}."
                }
                
        elif action == 2:  # SELL
            if existing_shares > 0:
                shares_to_sell = max(1, int(existing_shares * 0.50))
                if existing_shares <= 2:
                    shares_to_sell = existing_shares
                gross_trade_value = shares_to_sell * current_price
                fee = gross_trade_value * self.transaction_fee
                net_inflow = gross_trade_value - fee
                new_shares = existing_shares - shares_to_sell
                remaining_cash = virtual_capital + net_inflow
                portfolio_val = remaining_cash + (new_shares * current_price)
                
                return {
                    "raw_action": raw_action_name,
                    "executable_action": "SELL",
                    "status": "APPROVED",
                    "shares_traded": shares_to_sell,
                    "new_shares_held": new_shares,
                    "gross_trade_value": round(gross_trade_value, 2),
                    "transaction_fee": round(fee, 2),
                    "total_outlay": round(net_inflow, 2),
                    "remaining_cash": round(remaining_cash, 2),
                    "portfolio_value": round(portfolio_val, 2),
                    "reason": f"Liquidated {shares_to_sell} shares to cash to lock in capital preservation."
                }
            else:
                return {
                    "raw_action": raw_action_name,
                    "executable_action": "HOLD",
                    "status": "REJECTED (No Shares Owned)",
                    "shares_traded": 0,
                    "new_shares_held": 0,
                    "gross_trade_value": 0.0,
                    "transaction_fee": 0.0,
                    "total_outlay": 0.0,
                    "remaining_cash": round(virtual_capital, 2),
                    "portfolio_value": round(virtual_capital, 2),
                    "reason": "Model emitted SELL signal, but no shares are owned. Short selling is strictly prohibited. Maintaining 100% cash buffer."
                }
                
        else:  # HOLD
            return {
                "raw_action": raw_action_name,
                "executable_action": "HOLD",
                "status": "APPROVED",
                "shares_traded": 0,
                "new_shares_held": existing_shares,
                "gross_trade_value": 0.0,
                "transaction_fee": 0.0,
                "total_outlay": 0.0,
                "remaining_cash": round(virtual_capital, 2),
                "portfolio_value": round(virtual_capital + (existing_shares * current_price), 2),
                "reason": "Maintaining established asset allocation without incurring transaction friction."
            }

    def compute_composite_risk_score(
        self,
        current_portfolio_value: float,
        stock_volatility: float,
        holdings_ratio: float
    ) -> float:
        """Computes a normalized composite Risk Score between 0 (Lowest Risk) and 100 (Extremely High Risk).
        Incorporates current drawdown, asset volatility, and portfolio exposure.
        """
        current_dd = self.calculate_drawdown(current_portfolio_value)
        
        # Volatility factor (annualized vol around 0.15-0.45 -> scaled to 0-35 pts)
        vol_score = min(35.0, stock_volatility * 90.0)
        
        # Drawdown factor (scaled to 0-45 pts)
        dd_score = min(45.0, current_dd * 150.0)
        
        # Exposure / Holdings factor (scaled to 0-20 pts)
        exposure_score = float(holdings_ratio) * 20.0
        
        composite = vol_score + dd_score + exposure_score
        return round(float(np.clip(composite, 0.0, 100.0)), 1)
