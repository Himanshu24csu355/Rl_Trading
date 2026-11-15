# Risk-Sensitive Reinforcement Learning for Intelligent Stock Trading and Risk Management

> **Academic Project:** B.Tech in Artificial Intelligence & Machine Learning  
> **Domain:** Quantitative Finance, Algorithmic Decision Support, Deep Reinforcement Learning  
> **Technology Stack:** Python 3.11, Gymnasium, Stable-Baselines3 (PPO), PyTorch, Streamlit, Plotly, yfinance

---

## 📌 Executive Summary & Problem Formulation

Standard Reinforcement Learning (RL) agents in quantitative finance typically optimize the unconstrained **Expected Cumulative Return**:

$$\max_{\pi} \mathbb{E}_{\pi} \left[ \sum_{t=0}^T \gamma^t r_t \right]$$

While effective in toy environments, purely return-seeking agents fail in real-world financial markets characterized by fat-tailed distributions, regime shifts, and unexpected market selloffs. Standard models often exhibit **unbearable drawdowns** (often exceeding 35–50%) and erratic churn during bear regimes. In institutional portfolio management, excessive drawdowns trigger liquidation and catastrophic capital loss.

### The Solution: Risk-Sensitive Reinforcement Learning
This platform formulates a **Risk-Sensitive Proximal Policy Optimization (PPO)** agent. The utility function penalizes downside return deviations and peak-to-trough drawdowns, scaled by a risk-aversion coefficient $\lambda$:

$$R_t = r_{\text{portfolio}, t} - \lambda \cdot \Omega_t - c \cdot |\Delta w_t|$$

Where:
- $r_{\text{portfolio}, t} = \frac{V_t - V_{t-1}}{V_{t-1}}$ is the step percentage portfolio return.
- $\Omega_t = w_{\text{down}} \cdot \left(\min(0, r_{\text{portfolio}, t})\right)^2 \cdot 5.0 + w_{\text{dd}} \cdot \text{Drawdown}_t^2 + \text{breach\_penalty} \cdot \text{holdings\_ratio}$ is the risk penalty.
- $\lambda$ is the user's risk tolerance setting:
  - **Low Risk ($\lambda = 1.8$):** Prioritizes capital preservation, heavily penalizing downside losses and drawdowns while executing selective, high-conviction trades.
  - **Medium Risk ($\lambda = 0.8$):** Balanced growth and volatility dampening.
  - **High Risk ($\lambda = 0.2$):** Aggressive pursuit of returns with relaxed risk penalties.
  - **Standard PPO ($\lambda = 0.0$):** Unconstrained benchmark (pure return-seeking baseline).
- $c \cdot |\Delta w_t|$ represents transaction friction ($0.1\%$ per trade) to suppress portfolio churning.

---

## 🧭 Dual Analysis Architecture

The upgraded platform provides two distinct, fully decoupled operational modes:

```
                               ┌─────────────────────────────────────────────────────────────┐
                               │                     STREAMLIT UI MODE                       │
                               └──────────────┬───────────────────────────────┬──────────────┘
                                              │                               │
                       ┌──────────────────────▼──────┐         ┌──────────────▼──────────────────────┐
                       │ MODE A: HISTORICAL BACKTEST │         │ MODE B: CURRENT MARKET DECISION     │
                       └──────────────┬──────────────┘         └──────────────┬──────────────────────┘
                                      │                                       │
                ┌─────────────────────▼────────────────┐        ┌─────────────▼────────────────┐
                │ 1. Chronological Split (70/15/15)    │        │ 1. Dynamic Market Calendar   │
                │ 2. Unseen Test Evaluation            │        │ 2. Latest Completed Session  │
                │ 3. 3-Way Strategy Benchmark          │        │ 3. PPO Policy Distribution   │
                │ 4. Trade Execution Audit Log         │        │ 4. Whole-Share Order Sizing  │
                │ 5. Data-Driven Empirical Commentary  │        │ 5. Educational Virtual Model │
                └──────────────────────────────────────┘        └──────────────────────────────┘
```

### Mode A: Historical Backtesting
- Evaluates trained policies over strictly quarantined, out-of-sample test periods.
- Displays exact chronological partition dates (Train, Validation, Test) to guarantee zero look-ahead bias and no data leakage.
- Compares:
  1. **Passive Buy & Hold Benchmark**
  2. **Standard PPO Agent ($\lambda = 0.0$)**
  3. **Risk-Sensitive PPO Agent ($\lambda > 0$)**
- Generates 100% data-driven empirical findings reflecting actual numerical outcomes (returns, drawdowns, Sharpe, and fee drag).
- Downloadable trade execution audit log (CSV).

### Mode B: Current Market Decision Support (NEW)
- Dynamically queries current market state (NSE India timezone `Asia/Kolkata`).
- Trading calendar awareness: recognizes weekends, holidays, pre-market, and post-market sessions.
- Automatically retrieves the latest available verified market session (e.g. Friday session when accessed on Sunday) without hardcoded dates.
- Uses a configurable historical lookback window (2 to 5 years, default 3 years) to compute stationary technical indicators.
- **Genuine Policy Inference:** Computes the neural network's softmax action probability distribution:
  $$P(\text{HOLD}), \quad P(\text{BUY}), \quad P(\text{SELL})$$
  and genuine policy confidence score ($\max P$).
- **Raw vs Executable Action:** Distinguishes the model's raw action from portfolio-permitted actions (e.g. if the model emits `SELL` but no shares are owned, short selling is strictly prohibited and the portfolio maintains cash).
- **Virtual Order Sizing:** Calculates affordable whole shares, capital deployed, 0.1% transaction fee, remaining virtual cash, and portfolio value.
- Educational virtual capital disclaimer: Never claims future price prediction or guarantees returns.

---

## 📊 State and Action Space

### 1. State Space ($\mathcal{S} \in \mathbb{R}^{18}$)
All state variables maintain statistical stationarity and avoid look-ahead bias:

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
- `0 = HOLD`: Maintain current allocation.
- `1 = BUY`: Allocate available cash into shares (subject to cash buffer and whole-share lot sizing).
- `2 = SELL`: Liquidate equity to preserve cash and curtail drawdown.

---

## 📁 Repository Structure

```
RL Project/
├── app.py                      # Main Streamlit web application & dark dashboard (Mode A & B)
├── config.py                   # Central hyperparameters, paths, and risk constants
├── requirements.txt            # Python dependencies (PyTorch, SB3, Gymnasium, etc.)
├── README.md                   # Complete documentation & college viva guide
├── data/                       # Local cache for historical OHLCV stock datasets
├── models/                     # Complete suite of 16 pre-trained PPO models (.zip & .json)
├── scripts/
│   └── train_all.py            # Pre-training script verifying all 16 model combinations
├── src/
│   ├── indicators.py           # Technical indicator feature engineering
│   ├── data_loader.py          # yfinance ingestion, market calendar, and data splitting
│   ├── risk_manager.py         # Downside penalty, drawdown tracking, trade validation, sizing
│   ├── trading_env.py          # Custom Gymnasium trading environment
│   ├── train_model.py          # SB3 PPO trainer, model info inspection, policy distribution
│   ├── backtest.py             # Backtesting engine, AI rationalization, Mode B evaluation
│   ├── evaluation.py           # Plotly interactive visualizations, dynamic commentary
│   └── utils.py                # Logger, currency formatting (INR), math utilities
└── tests/
    ├── test_indicators.py      # Technical indicators verification
    ├── test_risk_manager.py    # Risk penalties and action validation
    ├── test_environment.py     # Gymnasium environment step mechanics
    ├── test_backtest.py        # Backtesting engine & benchmark comparison
    └── test_current_market.py  # Mode B calendar, order sizing, and policy distribution
```

---

## 💾 Complete 16-Model Pre-Trained Matrix

All 16 combinations across the 4 major Indian bluechips and 4 risk settings are pre-trained and saved locally in `models/`:

| Ticker | Low Risk ($\lambda = 1.8$) | Medium Risk ($\lambda = 0.8$) | High Risk ($\lambda = 0.2$) | Standard PPO ($\lambda = 0.0$) |
|---|---|---|---|---|
| **RELIANCE.NS** | `RELIANCE_NS_Low_ppo.zip` | `RELIANCE_NS_Medium_ppo.zip` | `RELIANCE_NS_High_ppo.zip` | `RELIANCE_NS_Standard_ppo.zip` |
| **TCS.NS** | `TCS_NS_Low_ppo.zip` | `TCS_NS_Medium_ppo.zip` | `TCS_NS_High_ppo.zip` | `TCS_NS_Standard_ppo.zip` |
| **INFY.NS** | `INFY_NS_Low_ppo.zip` | `INFY_NS_Medium_ppo.zip` | `INFY_NS_High_ppo.zip` | `INFY_NS_Standard_ppo.zip` |
| **HDFCBANK.NS** | `HDFCBANK_NS_Low_ppo.zip` | `HDFCBANK_NS_Medium_ppo.zip` | `HDFCBANK_NS_High_ppo.zip` | `HDFCBANK_NS_Standard_ppo.zip` |

---

## 🚀 Quickstart & Setup Guide

### 1. Virtual Environment Activation
```powershell
# In PowerShell:
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### 2. Run Comprehensive Unit Tests
```powershell
python -m pytest tests -v
```

### 3. Launch Streamlit Application
```powershell
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 🎯 College Demonstration Walkthrough (Reliance Example)

### Scenario: Virtual ₹10,000 Investment Decision Support
1. **Mode Selection:**
   - In the sidebar, select **Mode B: Current Market Decision Support**.
2. **Asset & Capital:**
   - Select `RELIANCE.NS - Reliance Industries Ltd.`
   - Set Simulated Investment to `₹10,000`.
   - Set Risk Tolerance to **Low Risk** ($\lambda = 1.8$).
3. **Run AI Decision:**
   - Click **"🎯 Analyze Market"**.
4. **Observe the Results:**
   - **Market Freshness Banner:** Notice the verified market session date (e.g. 2026-11-13 Friday) and trading status (`CLOSED (Weekend)`).
   - **AI Recommendation:** Observe the Raw Model Action vs Permitted Executable Action.
   - **Confidence & Policy Distribution:** View the horizontal bar chart showing the neural network's exact softmax probabilities ($P(\text{HOLD}), P(\text{BUY}), P(\text{SELL})$).
   - **Order Sizing:** See how many whole shares can be bought with ₹10,000, the 0.1% transaction fee, remaining virtual cash, and total virtual portfolio value.
   - **Switch to Mode A:** Switch to **Mode A: Historical Backtesting** to demonstrate out-of-sample backtest curves, strategy benchmarks, and the audit log.

---

## 🎓 College Viva Voce Defense Cheatsheet

### Q1: What makes your project "Risk-Sensitive"?
**Answer:** Standard RL maximizes expected return $\mathbb{E}[R]$, which often tolerates massive 40-50% drawdowns. Our reward function penalizes downside deviations quadratically:
$$R_t = r_{\text{portfolio}, t} - \lambda \cdot [ w_{\text{down}} \cdot (\min(0, r_t))^2 \cdot 5.0 + w_{\text{dd}} \cdot \text{Drawdown}_t^2 ] - c \cdot |\Delta w_t|$$
This forces the policy to prioritize capital preservation, active drawdown containment, and selective trade timing.

### Q2: Why did Low Risk models sometimes hold cash in earlier versions, and how was it solved?
**Answer:** If the drawdown penalty penalizes the agent even when it holds 100% cash during market drawdowns, holding cash becomes the only zero-penalty option. We calibrated the breach penalty by scaling it with `holdings_ratio`. When the agent holds cash during a market drop, it is protecting capital and receives no penalty; it only gets penalized if it holds high equity exposure while deep in drawdown.

### Q3: How does Mode B differ from Mode A?
**Answer:** Mode A is a retrospective chronological backtest across out-of-sample test data with zero lookahead bias. Mode B is an active virtual decision support tool that fetches the latest available verified market session, evaluates the 18-feature state vector through the trained PPO network, extracts policy action probabilities, and calculates realistic whole-share order allocations.

### Q4: How is policy confidence measured?
**Answer:** Rather than using arbitrary heuristic confidence scores, our system queries the PPO actor network's categorical distribution logits and applies softmax to compute true action probabilities $P(a | s)$. Policy confidence is the maximum probability $\max_a P(a | s)$.
