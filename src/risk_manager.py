"""Risk Management Module for Portfolio Monitoring, Risk Penalties, and Constraints.
Enforces position limits, downside risk penalties, drawdown tracking, and capital conservation.
"""

from typing import Dict, Any, Tuple
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
        self.peak_portfolio_value = initial_capital
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

    def compute_risk_penalty(self, step_return: float, current_portfolio_value: float) -> float:
        """Computes the mathematical risk penalty for the RL reward function:
        Risk Penalty = lambda * [ downside_weight * min(0, step_return)^2 + drawdown_weight * Drawdown^2 ]
        """
        # Downside return component (penalizes negative returns quadratically)
        downside = min(0.0, step_return)
        downside_penalty = (downside ** 2) * self.downside_weight * 10.0

        # Drawdown component (penalizes being far below portfolio peak)
        current_dd = self.calculate_drawdown(current_portfolio_value)
        dd_penalty = (current_dd ** 2) * self.drawdown_weight

        # Breach penalty if drawdown exceeds tolerance limit
        breach_penalty = 0.0
        if current_dd > self.max_drawdown_limit:
            breach_penalty = (current_dd - self.max_drawdown_limit) * 2.0

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
        """Validates trading actions against capital constraints and risk rules.
        
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
                return 0, 0, "BUY blocked: Insufficient available cash"
                
            # For discrete action, allocate 50% of available effective cash per buy signal
            # to prevent all-in gambling and promote prudent dollar-cost averaging
            target_spend = effective_cash * 0.50
            shares_to_buy = int(target_spend // cost_per_share)
            if shares_to_buy < 1:
                shares_to_buy = int(effective_cash // cost_per_share)
                
            if shares_to_buy < 1:
                return 0, 0, "BUY blocked: Cash cannot buy at least 1 share"
                
            return 1, shares_to_buy, f"BUY approved for {shares_to_buy} shares"

        elif action == 2:  # SELL
            if current_shares <= 0:
                return 0, 0, "SELL blocked: No shares owned (Short selling prohibited)"
                
            # Liquidate 50% or all remaining shares if small position
            shares_to_sell = max(1, int(current_shares * 0.50))
            if current_shares <= 2:
                shares_to_sell = current_shares
                
            return 2, shares_to_sell, f"SELL approved for {shares_to_sell} shares"

        else:  # HOLD
            return 0, 0, "HOLD: Maintain position"

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
        
        # Volatility factor (annualized vol around 0.20-0.40 -> scaled to 0-35 pts)
        vol_score = min(35.0, stock_volatility * 100.0)
        
        # Drawdown factor (scaled to 0-45 pts)
        dd_score = min(45.0, current_dd * 150.0)
        
        # Exposure / Holdings factor (scaled to 0-20 pts)
        exposure_score = holdings_ratio * 20.0
        
        composite = vol_score + dd_score + exposure_score
        return round(float(np.clip(composite, 0.0, 100.0)), 1)
