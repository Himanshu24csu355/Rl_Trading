# Risk-Sensitive Reinforcement Learning for Intelligent Stock Trading and Risk Management

> **Academic Project:** B.Tech in Artificial Intelligence & Machine Learning  
> **Domain:** Quantitative Finance, Algorithmic Trading, Deep Reinforcement Learning  
> **Frameworks:** Python 3.11, Gymnasium, Stable-Baselines3 (PPO), PyTorch, Streamlit, Plotly, yfinance

---

## 📌 Executive Summary & Problem Formulation

Standard Reinforcement Learning (RL) agents in quantitative trading maximize the expected cumulative reward:

$$\max_{\pi} \mathbb{E}_{\pi} \left[ \sum_{t=0}^T \gamma^t r_t \right]$$

In theoretical environments, this optimizes capital growth. However, in real-world financial markets—characterized by fat-tailed return distributions, regime changes, market crashes, and non-stationarity—pure return maximization produces erratic policies prone to **catastrophic drawdowns** (often exceeding 40–50%). In institutional asset management, excessive drawdowns trigger margin calls, liquidation, and investor panic.

### The Solution: Risk-Sensitive Reinforcement Learning
This project implements a **Risk-Sensitive Proximal Policy Optimization (PPO)** agent. The reward function directly penalizes downside volatility and peak-to-trough drawdowns, parameterized by a risk-aversion coefficient $\lambda$:

$$R_t = r_{\text{portfolio}, t} - \lambda \cdot \Omega_t - c \cdot |\Delta w_t|$$

Where:
- $r_{\text{portfolio}, t} = \frac{V_t - V_{t-1}}{V_{t-1}}$ is the step percentage return of the portfolio.
- $\Omega_t = w_{\text{down}} \cdot \left(\min(0, r_{\text{portfolio}, t})\right)^2 + w_{\text{dd}} \cdot \text{Drawdown}_t^2$ is the risk penalty.
- $\lambda$ is the user's risk tolerance factor:
  - **Low Risk ($\lambda = 2.5$):** Heavy capital protection; aggressive penalty on any downside return or drawdown.
  - **Medium Risk ($\lambda = 1.0$):** Balanced policy reconciling growth and volatility dampening.
  - **High Risk ($\lambda = 0.2$):** Aggressive pursuit of returns with minimal risk damping.
  - **Standard PPO ($\lambda = 0.0$):** Unconstrained benchmark (pure return-seeking).
- $c \cdot |\Delta w_t|$ represents transaction friction ($0.1\%$ per trade) to suppress portfolio churning.

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                       STREAMLIT UI                          │
│  - Dark Theme Financial Terminal                            │
│  - Stock & Capital Selector (INR)                           │
│  - Risk Tolerance Condition (Low, Medium, High)             │
│  - Interactive Plotly Charts (Price, Trades, Equity, MDD)   │
└──────────────────────────────┬──────────────────────────────┘
                               │ User Interactions
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                 BACKTEST & EVALUATION ENGINE                │
│  - Chronological Test Set Runner (Zero Look-ahead Bias)     │
│  - 3-Way Strategy Benchmark (Buy&Hold vs Std PPO vs RS PPO) │
│  - Performance Metrics (Sharpe, Sortino, MDD, Win Rate)     │
└──────────────────────────────┬──────────────────────────────┘
                               │ Observations & Inferences
                               ▼
┌─────────────────────────────────────────────────────────────┐
│           STABLE-BASELINES3 PPO AGENT (ACTOR-CRITIC)        │
│  - Clipped Surrogate Objective                              │
│  - Multi-Layer Perceptron Policy (128x128 pi & vf)          │
│  - Deterministic Action Sampling for Evaluation             │
└──────────────────────────────┬──────────────────────────────┘
                               │ Actions: {0: HOLD, 1: BUY, 2: SELL}
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                 CUSTOM GYMNASIUM ENVIRONMENT                │
│  - 18-Dimensional State Space                               │
│  - Virtual Portfolio Accounting (Cash + Shares * Price)     │
│  - RiskManager Enforcement (No short-selling, Cash buffer)  │
│  - Risk-Penalized Mathematical Reward Function              │
└──────────────────────────────┬──────────────────────────────┘
                               │ OHLCV Data & Stationarity
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                 DATA & INDICATOR PIPELINE                   │
│  - Yahoo Finance (yfinance) Historical Data Ingestion       │
│  - Stationary Indicators (RSI, MACD, Bollinger Bands, ATR)  │
│  - Local Cache & Synthetic Offline Resilience Fallback      │
└─────────────────────────────────────────────────────────────┘
```

---

## 📊 State and Action Space

### 1. State Space ($\mathcal{S} \in \mathbb{R}^{18}$)
All state variables are engineered to maintain statistical stationarity and avoid look-ahead bias:

| Feature Category | Indicators | Description |
|---|---|---|
| **Returns** | `returns`, `log_returns` | 1-day percentage and logarithmic returns |
| **Volatility** | `volatility_20` | 20-day annualized rolling return standard deviation |
| **Trend / Ratios** | `sma_10_ratio`, `sma_30_ratio`, `ema_20_ratio` | $\frac{\text{Close} - \text{MA}}{\text{MA}}$ relative price distances |
| **Momentum** | `rsi_14`, `macd`, `macd_signal`, `macd_hist` | 14-day Wilder RSI ($[0, 1]$), MACD line, signal, and histogram |
| **Volatility Bands** | `bb_high_ratio`, `bb_low_ratio`, `bb_width` | Normalized distances from 20-day $2\sigma$ Bollinger Bands |
| **True Range** | `atr_14_ratio` | 14-day Average True Range normalized by Close price |
| **Volume** | `volume_ratio` | Current volume divided by 20-day moving average volume |
| **Account Context** | `cash_ratio`, `holdings_ratio`, `drawdown` | $\frac{\text{Cash}}{V_t}$, $\frac{\text{Shares} \cdot P_t}{V_t}$, and $\frac{V_{\text{peak}} - V_t}{V_{\text{peak}}}$ |

### 2. Action Space ($\mathcal{A} = \{0, 1, 2\}$)
- `0 = HOLD`: Maintain current positions, incurring zero transaction costs.
- `1 = BUY`: Allocate available cash into shares (subject to risk limits and cash buffers).
- `2 = SELL`: Liquidate equity holdings to cash to preserve gains and curb drawdown.

### 3. Financial Constraints
- **Zero Short-Selling:** If the agent emits a `SELL` signal with $0$ shares, the trade is rejected and treated as `HOLD`.
- **Cash Constraints:** If cash is insufficient to buy at least 1 share, the `BUY` action is converted to `HOLD`.
- **Transaction Costs:** A realistic $0.1\%$ friction ($0.0010$) is applied to all gross trade turnover.

---

## 📁 Repository Structure

```
RL Project/
├── app.py                      # Main Streamlit web application & dark dashboard
├── config.py                   # Central hyperparameters, paths, and risk constants
├── requirements.txt            # Python dependencies (PyTorch, SB3, Gymnasium, etc.)
├── README.md                   # Complete documentation & college viva guide
├── data/                       # Local cache for historical OHLCV stock datasets
├── models/                     # Saved trained PPO neural network weights (.zip & .json)
├── scripts/
│   └── train_all.py            # Pre-training script for all supported bluechip stocks
├── src/
│   ├── __init__.py
│   ├── utils.py                # Logger, currency formatting (INR), math utilities
│   ├── indicators.py           # Technical indicator feature engineering
│   ├── data_loader.py          # yfinance data ingestion and chronological split
│   ├── risk_manager.py         # Downside penalty, drawdown tracking, trade validation
│   ├── trading_env.py          # Custom Gymnasium trading environment
│   ├── train_model.py          # SB3 PPO training pipeline and model persistence
│   ├── backtest.py             # Chronological backtesting engine and AI rationalization
│   └── evaluation.py           # Plotly interactive visualizations and benchmarking
└── tests/
    ├── __init__.py
    ├── test_indicators.py      # Unit tests for indicator mathematics
    ├── test_risk_manager.py    # Unit tests for risk penalties and constraints
    ├── test_environment.py     # Unit tests for Gymnasium dynamics & accounting
    └── test_backtest.py        # Unit tests for backtesting & benchmarking
```

---

## 🚀 Quickstart & Setup Guide

### 1. Prerequisites
- Python 3.11 (or conda environment with Python 3.11)
- Git

### 2. Virtual Environment Setup
```powershell
# Clone or navigate to the project directory
cd "c:\Users\new user\Desktop\RL Project"

# Create virtual environment (if not already created)
python -m venv .venv

# Activate virtual environment
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# Or Command Prompt:
.\.venv\Scripts\activate.bat
```

### 3. Install Dependencies
```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Run Unit Tests
```powershell
python -m pytest tests -v
```
All 14 unit tests will run and verify the mathematical integrity of the indicators, risk penalties, environment step mechanics, and backtesting.

### 5. Launch the Streamlit Application
```powershell
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 🎯 How to Demonstrate the Application (College Presentation)

### Step 1: Sidebar Configuration
1. Select a bluechip stock from the dropdown (e.g. `RELIANCE.NS` - Reliance Industries).
2. Enter a simulated investment amount in Indian Rupees (e.g. `₹1,00,000`).
3. Select the **Risk Tolerance Level**:
   - Start with **Low Risk** ($\lambda = 2.5$) to showcase capital preservation.
4. Click **"⚡ Analyze with AI"**.

### Step 2: Review Main Dashboard
- **KPI Metrics:** Notice the Latest Price, AI Signal (BUY/SELL/HOLD), Dynamic Risk Score (0–100), Final Portfolio Value, and Maximum Drawdown.
- **Interactive Price Chart:** Observe the green `BUY` triangles and red `SELL` triangles overlayed on the price and volume charts.
- **AI Decision Rationalization:** Point out the explanation box explaining why the model took the latest action (RSI oversold/overbought, MACD crossover, rolling volatility threshold).

### Step 3: Strategy Benchmark Tab
- Open the **"⚖️ Strategy Benchmark & Risk Analysis"** tab.
- Show the 3-way side-by-side performance table:
  1. **Buy & Hold Benchmark**
  2. **Standard PPO (Risk-Neutral)**
  3. **Risk-Sensitive PPO**
- Compare the **Equity Curves** and **Underwater Drawdown Profile**.
- Highlight that the Risk-Sensitive agent limits catastrophic drawdowns compared to both unconstrained RL and Buy & Hold!

### Step 4: Audit Trail Tab
- Open the **"📜 Trade Execution Audit Log"** tab.
- Demonstrate that all trades are fully accounted for with timestamp, execution price, shares purchased/sold, transaction fee deducted, and remaining virtual cash.
- Download the trade log as a CSV to show empirical data integrity.

---

## 🔬 Comparison: Standard vs Risk-Sensitive PPO

| Dimension | Standard PPO ($\lambda = 0.0$) | Risk-Sensitive PPO ($\lambda > 0$) |
|---|---|---|
| **Reward Function** | $R_t = r_t$ | $R_t = r_t - \lambda \Omega_t - c |\Delta w_t|$ |
| **Downside Volatility** | High | Strictly Suppressed |
| **Maximum Drawdown** | Frequently > 25% | Actively Contained (10–18%) |
| **Sortino Ratio** | Moderate | Significantly Higher |
| **Behavior in Bear Regimes** | Prone to holding or buying falling knives | Promptly cuts exposure to preserve cash |
| **Transaction Discipline** | May over-trade without cost penalty | Trades only on high-conviction momentum |

---

## 🎓 College Viva Voce Defense Q&A Cheatsheet

### Q1: Why did you choose PPO over DQN or A2C?
**Answer:** PPO (Proximal Policy Optimization) combines the benefits of policy gradient methods with the sample efficiency and stability of trust-region methods (TRPO) without the prohibitive second-order Hessian computation. Financial time series are notoriously noisy; DQN frequently suffers from Q-value overestimation and policy divergence, whereas PPO's clipped surrogate loss prevents excessively large policy updates.

### Q2: How did you ensure there is no look-ahead bias or data leakage?
**Answer:**
1. All technical indicators (RSI, Moving Averages, MACD, ATR, Volatility) use strictly backward-looking rolling windows (`shift(1)` or standard causal rolling mean/std).
2. Data splitting is strictly chronological: the first 70% is used for training, the next 15% for validation, and the final 15% is strictly quarantined for unseen out-of-sample backtesting.

### Q3: What is the significance of the risk parameter $\lambda$?
**Answer:** $\lambda$ is the risk-aversion multiplier in the utility function. It acts as a hyperparameter that interpolates between pure return-seeking behavior ($\lambda = 0$) and extreme capital preservation ($\lambda = 2.5$). Higher $\lambda$ quadratically penalizes negative returns and drawdowns, causing the policy to favor cash preservation during volatile or downtrending market phases.

### Q4: What are the main limitations of historical backtesting?
**Answer:**
1. **Market Impact & Slippage:** Simulated orders do not move the order book. In live institutional trading, large block orders create market impact.
2. **Execution Latency:** Simulated trades execute at historical Close price; live markets have bid-ask spreads and order queue priority.
3. **Regime Shifts:** Future macroeconomic events (e.g. geopolitical shocks, rate hikes) may not match historical distributions.

---

## 📜 Academic Declaration
This project is developed as part of the academic curriculum for the **B.Tech Degree in Artificial Intelligence & Machine Learning**. It serves strictly as an educational and research platform for quantitative reinforcement learning. All trading simulations operate purely on virtual paper capital.
# Rl_Trading
# Rl_Trading
# Rl_Trading
