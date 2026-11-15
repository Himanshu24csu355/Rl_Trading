"""Gymnasium-compliant Stock Trading Environment with Risk-Sensitive Reward Shaping.
Simulates realistic cash balances, shares held, transaction costs, and portfolio accounting.
"""

from typing import Tuple, Dict, Any, Optional
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

from config import (
    TECHNICAL_INDICATORS,
    DEFAULT_INITIAL_CAPITAL,
    TRANSACTION_FEE_PERCENT,
)
from src.risk_manager import RiskManager
from src.utils import logger


class RiskSensitiveTradingEnv(gym.Env):
    """A realistic single-asset trading environment with risk-penalized reward mechanics."""

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        df: pd.DataFrame,
        initial_capital: float = DEFAULT_INITIAL_CAPITAL,
        risk_level: str = "Medium",
        transaction_fee: float = TRANSACTION_FEE_PERCENT,
        feature_cols: Optional[list] = None
    ):
        super().__init__()
        
        self.df = df.copy().reset_index(drop=True)
        if len(self.df) < 20:
            raise ValueError(f"DataFrame too small for trading simulation ({len(self.df)} rows)")
            
        self.initial_capital = float(initial_capital)
        self.risk_level = risk_level
        self.transaction_fee = float(transaction_fee)
        self.feature_cols = feature_cols or TECHNICAL_INDICATORS
        
        # Verify feature columns exist in DataFrame
        for col in self.feature_cols:
            if col not in self.df.columns:
                raise ValueError(f"Required indicator column '{col}' missing from DataFrame")
                
        # Risk Manager
        self.risk_manager = RiskManager(
            risk_level=self.risk_level,
            transaction_fee=self.transaction_fee
        )
        
        # Action space: 0 = HOLD, 1 = BUY, 2 = SELL
        self.action_space = spaces.Discrete(3)
        
        # Observation space:
        # Technical indicators (len) + Account state: [cash_ratio, holdings_ratio, current_drawdown]
        self.account_features_len = 3
        total_obs_dim = len(self.feature_cols) + self.account_features_len
        
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(total_obs_dim,),
            dtype=np.float32
        )
        
        # Internal state variables
        self.current_step = 0
        self.max_steps = len(self.df) - 1
        self.cash = self.initial_capital
        self.shares = 0
        self.portfolio_value = self.initial_capital
        self.prev_portfolio_value = self.initial_capital
        self.total_trades = 0
        self.history = []

    def _get_observation(self) -> np.ndarray:
        """Constructs the normalized observation vector for the current time step."""
        row = self.df.iloc[self.current_step]
        tech_features = row[self.feature_cols].values.astype(np.float32)
        # Replace any potential inf or nan in tech indicators
        tech_features = np.nan_to_num(tech_features, nan=0.0, posinf=5.0, neginf=-5.0)
        
        current_price = float(row["Close"])
        holdings_val = self.shares * current_price
        total_val = max(1e-8, self.cash + holdings_val)
        
        cash_ratio = np.clip(self.cash / total_val, 0.0, 1.0)
        holdings_ratio = np.clip(holdings_val / total_val, 0.0, 1.0)
        current_dd = self.risk_manager.calculate_drawdown(total_val)
        
        account_features = np.array([cash_ratio, holdings_ratio, current_dd], dtype=np.float32)
        obs = np.concatenate([tech_features, account_features])
        return obs

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Resets the environment for a new trading episode."""
        super().reset(seed=seed)
        
        self.current_step = 0
        self.cash = self.initial_capital
        self.shares = 0
        self.portfolio_value = self.initial_capital
        self.prev_portfolio_value = self.initial_capital
        self.total_trades = 0
        self.history = []
        
        self.risk_manager.reset(self.initial_capital)
        
        obs = self._get_observation()
        info = {
            "step": 0,
            "portfolio_value": self.portfolio_value,
            "cash": self.cash,
            "shares": self.shares,
            "date": str(self.df.iloc[0].get("Date", "N/A"))
        }
        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Executes one trading step, computes risk-adjusted reward, and updates portfolio."""
        current_row = self.df.iloc[self.current_step]
        current_price = float(current_row["Close"])
        current_date = current_row.get("Date", f"Day_{self.current_step}")
        
        # 1. Validate action and determine trade execution
        exec_action, qty, reason = self.risk_manager.validate_action(
            action=action,
            current_cash=self.cash,
            current_shares=self.shares,
            stock_price=current_price,
            portfolio_value=self.portfolio_value
        )
        
        fee_paid = 0.0
        trade_occurred = False
        trade_type = "HOLD"
        
        if exec_action == 1 and qty > 0:  # BUY
            gross_cost = qty * current_price
            fee = gross_cost * self.transaction_fee
            total_outlay = gross_cost + fee
            
            if self.cash >= total_outlay:
                self.cash -= total_outlay
                self.shares += qty
                fee_paid = fee
                trade_occurred = True
                trade_type = "BUY"
                self.total_trades += 1
                
        elif exec_action == 2 and qty > 0:  # SELL
            gross_revenue = qty * current_price
            fee = gross_revenue * self.transaction_fee
            net_inflow = gross_revenue - fee
            
            self.shares -= qty
            self.cash += net_inflow
            fee_paid = fee
            trade_occurred = True
            trade_type = "SELL"
            self.total_trades += 1
            
        # 2. Advance time step
        self.current_step += 1
        terminated = self.current_step >= self.max_steps
        truncated = False
        
        # Next price evaluation
        next_row = self.df.iloc[self.current_step]
        next_price = float(next_row["Close"])
        
        # 3. Update Portfolio Value
        self.portfolio_value = self.cash + (self.shares * next_price)
        
        # Step percentage return
        step_return = (self.portfolio_value - self.prev_portfolio_value) / max(1e-8, self.prev_portfolio_value)
        self.prev_portfolio_value = self.portfolio_value
        
        # 4. Update Risk Manager
        self.risk_manager.update_portfolio(self.portfolio_value, step_return)
        
        # 5. Compute Risk-Sensitive Reward
        # Reward = Step Return - lambda * Risk Penalty - Friction
        holdings_val = self.shares * next_price
        holdings_ratio = float(np.clip(holdings_val / max(1e-8, self.portfolio_value), 0.0, 1.0))
        risk_penalty = self.risk_manager.compute_risk_penalty(
            step_return=step_return,
            current_portfolio_value=self.portfolio_value,
            holdings_ratio=holdings_ratio
        )
        friction_penalty = (fee_paid / self.portfolio_value) if self.portfolio_value > 0 else 0.0
        
        # Penalty for invalid attempted action (discourage agent from attempting prohibited moves)
        invalid_action_penalty = -0.001 if (action != 0 and not trade_occurred) else 0.0
        
        reward = float(step_return - risk_penalty - friction_penalty + invalid_action_penalty)
        
        # Check bankruptcy (portfolio dropped by >= 90%)
        if self.portfolio_value < (self.initial_capital * 0.10):
            terminated = True
            reward -= 0.5  # Severe penalty for capital depletion
            
        # 6. Record step history
        step_record = {
            "step": self.current_step,
            "date": current_date,
            "price": current_price,
            "action_intended": action,
            "action_executed": exec_action,
            "trade_type": trade_type,
            "shares_traded": qty if trade_occurred else 0,
            "shares_held": self.shares,
            "fee_paid": fee_paid,
            "cash": self.cash,
            "portfolio_value": self.portfolio_value,
            "step_return": step_return,
            "drawdown": self.risk_manager.calculate_drawdown(self.portfolio_value),
            "reward": reward,
            "reason": reason
        }
        self.history.append(step_record)
        
        obs = self._get_observation()
        info = {
            "step": self.current_step,
            "date": str(next_row.get("Date", "N/A")),
            "portfolio_value": self.portfolio_value,
            "cash": self.cash,
            "shares": self.shares,
            "total_trades": self.total_trades,
            "drawdown": self.risk_manager.calculate_drawdown(self.portfolio_value),
            "step_record": step_record
        }
        
        return obs, reward, terminated, truncated, info

    def get_history_df(self) -> pd.DataFrame:
        """Returns the full episode history as a pandas DataFrame."""
        return pd.DataFrame(self.history)
