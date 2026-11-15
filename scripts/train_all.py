"""Script to pre-train and verify genuine PPO models across risk profiles.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import SUPPORTED_TICKERS, FAST_DEMO_TIMESTEPS
from src.train_model import train_ppo_agent
from src.utils import logger


def train_initial_suite():
    """Trains a complete suite of genuine risk-sensitive and benchmark models."""
    logger.info("==================================================")
    logger.info("PRE-TRAINING GENUINE RISK-SENSITIVE PPO MODELS")
    logger.info("==================================================")
    
    # 1. Primary Stock: RELIANCE.NS across all risk levels + Standard
    primary_ticker = "RELIANCE.NS"
    for risk in ["Low", "Medium", "High", "Standard"]:
        logger.info(f"--- Training {primary_ticker} [{risk}] ---")
        train_ppo_agent(
            ticker=primary_ticker,
            risk_level=risk,
            total_timesteps=12000,
            save=True
        )

    # 2. Secondary Key Bluechips: TCS.NS, INFY.NS, HDFCBANK.NS with Medium risk
    for sec_ticker in ["TCS.NS", "INFY.NS", "HDFCBANK.NS"]:
        logger.info(f"--- Training {sec_ticker} [Medium] ---")
        train_ppo_agent(
            ticker=sec_ticker,
            risk_level="Medium",
            total_timesteps=10000,
            save=True
        )

    logger.info("All suite models trained and saved successfully in models/ directory!")


if __name__ == "__main__":
    train_initial_suite()
