"""PPO Reinforcement Learning Training Pipeline using Stable-Baselines3.
Trains risk-sensitive trading agents conditioned on specified risk tolerances.
"""

import os
import json
import argparse
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from config import (
    MODELS_DIR,
    PPO_PARAMS,
    TRAIN_TIMESTEPS,
    FAST_DEMO_TIMESTEPS,
    RISK_LEVELS,
    DEFAULT_TICKER,
)
from src.data_loader import prepare_market_dataset
from src.trading_env import RiskSensitiveTradingEnv
from src.utils import logger, set_seed


class TrainingProgressCallback(BaseCallback):
    """Logs training metrics, episode rewards, and portfolio milestones."""

    def __init__(self, check_freq: int = 2000, verbose: int = 1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.step_count = 0

    def _on_step(self) -> bool:
        self.step_count += 1
        if self.step_count % self.check_freq == 0:
            logger.info(f"Training Progress: Timestep {self.step_count}")
        return True


def get_model_path(ticker: str, risk_level: str) -> Path:
    """Returns the standardized file path for a saved model."""
    clean_ticker = ticker.replace("^", "").replace(".", "_")
    return MODELS_DIR / f"{clean_ticker}_{risk_level}_ppo.zip"


def get_metadata_path(ticker: str, risk_level: str) -> Path:
    """Returns the standardized file path for model training metadata."""
    clean_ticker = ticker.replace("^", "").replace(".", "_")
    return MODELS_DIR / f"{clean_ticker}_{risk_level}_meta.json"


def train_ppo_agent(
    ticker: str = DEFAULT_TICKER,
    risk_level: str = "Medium",
    total_timesteps: int = FAST_DEMO_TIMESTEPS,
    seed: int = 42,
    save: bool = True
) -> Tuple[PPO, Dict[str, Any]]:
    """Trains a PPO reinforcement learning agent on historical stock data
    under the specified risk tolerance penalty.
    """
    set_seed(seed)
    logger.info(f"Starting PPO Agent Training for {ticker} | Risk Level: {risk_level} | Steps: {total_timesteps}")

    # 1. Prepare chronological datasets
    train_df, val_df, test_df = prepare_market_dataset(ticker)
    
    # 2. Instantiate training environment
    env = RiskSensitiveTradingEnv(
        df=train_df,
        risk_level=risk_level
    )
    
    # 3. Configure PPO model
    policy_kwargs = dict(
        net_arch=dict(pi=[128, 128], vf=[128, 128])
    )
    
    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=PPO_PARAMS["learning_rate"],
        n_steps=PPO_PARAMS["n_steps"],
        batch_size=PPO_PARAMS["batch_size"],
        n_epochs=PPO_PARAMS["n_epochs"],
        gamma=PPO_PARAMS["gamma"],
        gae_lambda=PPO_PARAMS["gae_lambda"],
        clip_range=PPO_PARAMS["clip_range"],
        ent_coef=PPO_PARAMS["ent_coef"],
        vf_coef=PPO_PARAMS["vf_coef"],
        max_grad_norm=PPO_PARAMS["max_grad_norm"],
        policy_kwargs=policy_kwargs,
        verbose=0,
        seed=seed
    )
    
    # 4. Train model
    callback = TrainingProgressCallback(check_freq=5000)
    model.learn(total_timesteps=total_timesteps, callback=callback)
    logger.info(f"Training completed successfully for {ticker} ({risk_level}).")
    
    # 5. Evaluate on validation set
    val_env = RiskSensitiveTradingEnv(df=val_df, risk_level=risk_level)
    obs, _ = val_env.reset()
    done = False
    total_val_reward = 0.0
    
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = val_env.step(int(action))
        total_val_reward += reward
        done = terminated or truncated
        
    val_final_portfolio = val_env.portfolio_value
    val_return_pct = (val_final_portfolio - val_env.initial_capital) / val_env.initial_capital
    
    metadata = {
        "ticker": ticker,
        "risk_level": risk_level,
        "lambda_penalty": RISK_LEVELS[risk_level]["lambda"],
        "total_timesteps": total_timesteps,
        "train_rows": len(train_df),
        "val_rows": len(val_df),
        "test_rows": len(test_df),
        "val_final_portfolio": round(val_final_portfolio, 2),
        "val_return_pct": round(val_return_pct * 100, 2),
        "val_total_trades": val_env.total_trades,
        "model_file": str(get_model_path(ticker, risk_level).name)
    }
    
    logger.info(
        f"Validation Evaluation: Return={metadata['val_return_pct']}%, "
        f"Final Value=₹{metadata['val_final_portfolio']}, Trades={metadata['val_total_trades']}"
    )

    # 6. Save model and metadata
    if save:
        model_path = get_model_path(ticker, risk_level)
        meta_path = get_metadata_path(ticker, risk_level)
        model.save(str(model_path))
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=4)
        logger.info(f"Saved model to: {model_path}")
        logger.info(f"Saved metadata to: {meta_path}")
        
    return model, metadata


def load_trained_agent(ticker: str, risk_level: str) -> Optional[PPO]:
    """Loads a previously saved PPO model if it exists, otherwise returns None."""
    model_path = get_model_path(ticker, risk_level)
    if not model_path.exists():
        logger.warning(f"No saved model found at {model_path}")
        return None
        
    try:
        model = PPO.load(str(model_path))
        logger.info(f"Loaded trained agent from {model_path}")
        return model
    except Exception as e:
        logger.error(f"Failed to load model from {model_path}: {e}")
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Risk-Sensitive PPO Trading Agent")
    parser.add_argument("--ticker", type=str, default=DEFAULT_TICKER, help="Stock ticker (e.g. RELIANCE.NS)")
    parser.add_argument("--risk", type=str, default="Medium", choices=list(RISK_LEVELS.keys()), help="Risk Level")
    parser.add_argument("--timesteps", type=int, default=FAST_DEMO_TIMESTEPS, help="Total training timesteps")
    args = parser.parse_args()
    
    train_ppo_agent(ticker=args.ticker, risk_level=args.risk, total_timesteps=args.timesteps)
