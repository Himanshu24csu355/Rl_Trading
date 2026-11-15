"""Configuration settings for Risk-Sensitive RL Stock Trading & Risk Management Application.
Supports both Historical Backtesting and Current Market Decision Support modes.
"""

from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"

DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Application Modes
MODE_BACKTEST = "Historical Backtesting"
MODE_CURRENT = "Current Market Decision Support"
ANALYSIS_MODES = [MODE_BACKTEST, MODE_CURRENT]

# Supported Stock Symbols (NSE Tickers & Popular Assets)
SUPPORTED_TICKERS = {
    "RELIANCE.NS": "Reliance Industries Ltd.",
    "TCS.NS": "Tata Consultancy Services Ltd.",
    "INFY.NS": "Infosys Ltd.",
    "HDFCBANK.NS": "HDFC Bank Ltd.",
}

DEFAULT_TICKER = "RELIANCE.NS"

# Date Ranges & Lookback Windows
DEFAULT_START_DATE = "2020-01-01"
DEFAULT_END_DATE = "2024-12-31"
DEFAULT_LOOKBACK_YEARS = 3  # For Current Market Decision Support indicator calculations
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Market Timing & Exchange Constants (NSE - National Stock Exchange of India)
TIMEZONE_INDIA = "Asia/Kolkata"
MARKET_OPEN_HOUR = 9
MARKET_OPEN_MINUTE = 15
MARKET_CLOSE_HOUR = 15
MARKET_CLOSE_MINUTE = 30

# Trading & Risk Management Defaults
DEFAULT_INITIAL_CAPITAL = 100000.0  # ₹100,000 INR
TRANSACTION_FEE_PERCENT = 0.001    # 0.1% slippage & brokerage fee
ANNUAL_TRADING_DAYS = 252
RISK_FREE_RATE = 0.065              # 6.5% Indian 10Y G-Sec Yield benchmark

# Risk Tolerance Levels and Penalty Weights (Reward: R = Ret - lambda * Risk Penalty - Friction)
# Calibrated so Low Risk preserves capital through selective high-conviction trades without stalling
RISK_LEVELS = {
    "Low": {
        "lambda": 1.8,
        "downside_weight": 1.2,
        "drawdown_weight": 0.8,
        "max_drawdown_limit": 0.12,  # 12% maximum acceptable drawdown threshold
        "description": "Conservative: Prioritizes capital preservation, dampens downside volatility, and manages drawdown strictly.",
        "badge_color": "#00C853"
    },
    "Medium": {
        "lambda": 0.8,
        "downside_weight": 0.8,
        "drawdown_weight": 0.5,
        "max_drawdown_limit": 0.20,  # 20% max drawdown threshold
        "description": "Balanced: Equal focus on competitive capital appreciation while curbing catastrophic market selloffs.",
        "badge_color": "#FFD600"
    },
    "High": {
        "lambda": 0.2,
        "downside_weight": 0.2,
        "drawdown_weight": 0.1,
        "max_drawdown_limit": 0.35,  # 35% max drawdown threshold
        "description": "Aggressive: Maximizes portfolio equity growth with relaxed downside risk penalties.",
        "badge_color": "#FF3D00"
    },
    "Standard": {
        "lambda": 0.0,
        "downside_weight": 0.0,
        "drawdown_weight": 0.0,
        "max_drawdown_limit": 1.0,
        "description": "Risk-Neutral Benchmark: Pure expected return maximization without risk penalty (Standard PPO).",
        "badge_color": "#2979FF"
    }
}

# Technical Indicators Feature List (Must strictly match 15 tech features in observation space)
TECHNICAL_INDICATORS = [
    "returns",
    "log_returns",
    "volatility_20",
    "sma_10_ratio",
    "sma_30_ratio",
    "ema_20_ratio",
    "rsi_14",
    "macd",
    "macd_signal",
    "macd_hist",
    "bb_high_ratio",
    "bb_low_ratio",
    "bb_width",
    "atr_14_ratio",
    "volume_ratio",
]

# Account state features (3): [cash_ratio, holdings_ratio, current_drawdown]
# Total observation dimension = 15 + 3 = 18

# Reinforcement Learning Hyperparameters
PPO_PARAMS = {
    "learning_rate": 3e-4,
    "n_steps": 256,
    "batch_size": 64,
    "n_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.01,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "verbose": 0,
}

TRAIN_TIMESTEPS = 30000
FAST_DEMO_TIMESTEPS = 12000
