"""Utility functions for logging, currency formatting, random seeding, and calculations.
"""

import os
import random
import logging
from typing import Union
import numpy as np
import torch

def setup_logger(name: str = "rl_trader", level: int = logging.INFO) -> logging.Logger:
    """Configures and returns a standardized logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(level)
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger

logger = setup_logger("RL_Trader")

def set_seed(seed: int = 42) -> None:
    """Sets deterministic seed across random, numpy, and torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    logger.debug(f"Global seed set to {seed}")

def format_inr(val: Union[float, int]) -> str:
    """Formats numeric value into Indian Rupee (INR) representation with commas."""
    if val is None or np.isnan(val):
        return "₹0.00"
    
    is_negative = val < 0
    val = abs(val)
    
    # Split integer and decimal parts
    int_part = int(val)
    dec_part = f"{val - int_part:.2f}"[1:]  # e.g., ".50"
    
    s = str(int_part)
    if len(s) <= 3:
        formatted = s
    else:
        # Indian numbering system: 3 digits rightmost, then groups of 2 digits
        last_three = s[-3:]
        remaining = s[:-3]
        groups = []
        while remaining:
            groups.append(remaining[-2:])
            remaining = remaining[:-2]
        groups.reverse()
        formatted = ",".join(groups) + "," + last_three
        
    result = f"₹{formatted}{dec_part}"
    return f"-{result}" if is_negative else result

def format_pct(val: Union[float, int], include_sign: bool = True) -> str:
    """Formats a decimal into a percentage string."""
    if val is None or np.isnan(val):
        return "0.00%"
    sign = "+" if (include_sign and val > 0) else ""
    return f"{sign}{val * 100:.2f}%"

def calculate_max_drawdown(equity_series: np.ndarray) -> float:
    """Calculates Maximum Drawdown (MDD) as a positive decimal fraction [0, 1]."""
    if len(equity_series) == 0:
        return 0.0
    peaks = np.maximum.accumulate(equity_series)
    drawdowns = (peaks - equity_series) / np.where(peaks == 0, 1e-8, peaks)
    return float(np.max(drawdowns)) if len(drawdowns) > 0 else 0.0

def calculate_sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.065, annual_periods: int = 252) -> float:
    """Calculates Annualized Sharpe Ratio."""
    if len(returns) < 2:
        return 0.0
    rf_daily = (1.0 + risk_free_rate) ** (1.0 / annual_periods) - 1.0
    excess_returns = returns - rf_daily
    std = np.std(excess_returns)
    if std < 1e-8 or np.isnan(std):
        return 0.0
    return float((np.mean(excess_returns) / std) * np.sqrt(annual_periods))

def calculate_sortino_ratio(returns: np.ndarray, risk_free_rate: float = 0.065, annual_periods: int = 252) -> float:
    """Calculates Annualized Sortino Ratio focusing only on downside deviations."""
    if len(returns) < 2:
        return 0.0
    rf_daily = (1.0 + risk_free_rate) ** (1.0 / annual_periods) - 1.0
    excess_returns = returns - rf_daily
    downside_returns = excess_returns[excess_returns < 0]
    if len(downside_returns) < 1:
        return 3.0  # Cap at high positive if zero downside
    downside_std = np.sqrt(np.mean(downside_returns ** 2))
    if downside_std < 1e-8:
        return 0.0
    return float((np.mean(excess_returns) / downside_std) * np.sqrt(annual_periods))
