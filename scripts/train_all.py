"""Script to pre-train and verify all 16 model combinations across 4 stocks and 4 risk profiles.
Ensures zero missing models for RELIANCE.NS, TCS.NS, INFY.NS, and HDFCBANK.NS.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import SUPPORTED_TICKERS, FAST_DEMO_TIMESTEPS, RISK_LEVELS
from src.train_model import train_ppo_agent, get_model_path
from src.utils import logger


def train_complete_suite():
    """Trains any missing models in the complete 4x4 matrix."""
    logger.info("==================================================")
    logger.info("CHECKING & PRE-TRAINING COMPLETE 16-MODEL SUITE")
    logger.info("==================================================")
    
    tickers = list(SUPPORTED_TICKERS.keys())
    risk_levels = list(RISK_LEVELS.keys())
    
    total_needed = len(tickers) * len(risk_levels)
    trained_count = 0
    
    for ticker in tickers:
        for risk in risk_levels:
            model_path = get_model_path(ticker, risk)
            if model_path.exists():
                logger.info(f"Model already exists: {model_path.name} (Skipping)")
                trained_count += 1
                continue
                
            logger.info(f"--- Training Missing Model: {ticker} [{risk}] ---")
            train_ppo_agent(
                ticker=ticker,
                risk_level=risk,
                total_timesteps=10000,
                save=True
            )
            trained_count += 1
            logger.info(f"Progress: {trained_count}/{total_needed} models ready.")

    logger.info("All 16 models in the complete suite are verified and ready in models/ directory!")


if __name__ == "__main__":
    train_complete_suite()
