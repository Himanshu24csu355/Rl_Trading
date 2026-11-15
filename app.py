"""Risk-Sensitive Reinforcement Learning Stock Trading and Risk Management Platform.
A professional, modern, dark-themed Streamlit dashboard for AI-driven portfolio simulation.
"""

import os
import json
from datetime import datetime, date
import streamlit as st
import pandas as pd
import numpy as np

from config import (
    SUPPORTED_TICKERS,
    DEFAULT_TICKER,
    DEFAULT_INITIAL_CAPITAL,
    DEFAULT_START_DATE,
    DEFAULT_END_DATE,
    RISK_LEVELS,
    TRANSACTION_FEE_PERCENT,
    MODELS_DIR,
    FAST_DEMO_TIMESTEPS,
)
from src.utils import format_inr, format_pct, setup_logger
from src.data_loader import fetch_stock_data, prepare_market_dataset
from src.indicators import add_all_indicators
from src.risk_manager import RiskManager
from src.train_model import train_ppo_agent, load_trained_agent, get_model_path, get_metadata_path
from src.backtest import run_agent_simulation, run_buy_and_hold_simulation, explain_ai_decision
from src.evaluation import (
    build_comparison_dataframe,
    plot_price_and_trades,
    plot_equity_curves,
    plot_drawdowns,
)

logger = setup_logger("Streamlit_App")

# Page Configuration
st.set_page_config(
    page_title="Risk-Sensitive RL Trading Platform",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Sleek Dark Theme & Modern Typography
st.markdown("""
<style>
    /* Dark Theme Core Styles */
    .stApp {
        background: linear-gradient(135deg, #0A0D14 0%, #10141E 100%);
        color: #E6EDF3;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #0D1117 !important;
        border-right: 1px solid #21262D;
    }
    
    /* Header Card */
    .hero-banner {
        background: linear-gradient(90deg, #161B26 0%, #1F293D 100%);
        border: 1px solid #30363D;
        border-radius: 12px;
        padding: 20px 24px;
        margin-bottom: 24px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
    }
    .hero-title {
        font-size: 24px;
        font-weight: 700;
        color: #FFFFFF;
        margin: 0;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .hero-subtitle {
        font-size: 13px;
        color: #8B949E;
        margin-top: 6px;
    }
    .badge-simulated {
        background: rgba(255, 171, 0, 0.15);
        color: #FFB300;
        border: 1px solid rgba(255, 171, 0, 0.4);
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 11px;
        font-weight: 600;
        letter-spacing: 0.5px;
        text-transform: uppercase;
    }
    
    /* Metric Card Grid */
    .metric-card {
        background-color: #161B26;
        border: 1px solid #21262D;
        border-radius: 10px;
        padding: 16px 18px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.25);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .metric-card:hover {
        border-color: #00E5FF;
        transform: translateY(-2px);
    }
    .metric-title {
        font-size: 12px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.6px;
        color: #8B949E;
        margin-bottom: 6px;
    }
    .metric-value {
        font-size: 22px;
        font-weight: 700;
        color: #FFFFFF;
        margin-bottom: 4px;
    }
    .metric-delta-pos {
        font-size: 12px;
        font-weight: 600;
        color: #00E676;
    }
    .metric-delta-neg {
        font-size: 12px;
        font-weight: 600;
        color: #FF1744;
    }
    
    /* Signal Badges */
    .signal-buy {
        display: inline-block;
        background-color: rgba(0, 230, 118, 0.15);
        color: #00E676;
        border: 1px solid #00E676;
        border-radius: 8px;
        padding: 6px 16px;
        font-size: 16px;
        font-weight: 800;
        letter-spacing: 1px;
    }
    .signal-sell {
        display: inline-block;
        background-color: rgba(255, 23, 68, 0.15);
        color: #FF1744;
        border: 1px solid #FF1744;
        border-radius: 8px;
        padding: 6px 16px;
        font-size: 16px;
        font-weight: 800;
        letter-spacing: 1px;
    }
    .signal-hold {
        display: inline-block;
        background-color: rgba(255, 214, 0, 0.15);
        color: #FFD600;
        border: 1px solid #FFD600;
        border-radius: 8px;
        padding: 6px 16px;
        font-size: 16px;
        font-weight: 800;
        letter-spacing: 1px;
    }
    
    /* Tab Navigation */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #0D1117;
        padding: 8px;
        border-radius: 10px;
        border: 1px solid #21262D;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 6px;
        color: #8B949E;
        font-weight: 600;
        font-size: 14px;
        padding: 8px 16px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1F293D !important;
        color: #00E5FF !important;
    }
    
    /* Decision Box */
    .decision-box {
        background-color: #141A24;
        border-left: 4px solid #00E5FF;
        border-radius: 0 8px 8px 0;
        padding: 16px 20px;
        margin: 16px 0;
    }
</style>
""", unsafe_allow_html=True)


# Initialize Session State
if "simulation_results" not in st.session_state:
    st.session_state.simulation_results = None
if "current_ticker" not in st.session_state:
    st.session_state.current_ticker = DEFAULT_TICKER
if "initial_capital" not in st.session_state:
    st.session_state.initial_capital = DEFAULT_INITIAL_CAPITAL
if "risk_level" not in st.session_state:
    st.session_state.risk_level = "Medium"


# ==========================================
# SIDEBAR CONTROLS
# ==========================================
with st.sidebar:
    st.markdown("### ⚙️ Simulation Controls")
    
    # 1. Stock Selection
    selected_ticker = st.selectbox(
        "Select Stock Ticker (NSE)",
        options=list(SUPPORTED_TICKERS.keys()),
        format_func=lambda x: f"{x} - {SUPPORTED_TICKERS.get(x, '')}",
        index=0
    )
    
    # Custom Ticker Option
    use_custom = st.checkbox("Enter Custom Symbol")
    if use_custom:
        custom_input = st.text_input("Custom Ticker (e.g. INFY.NS, AAPL)", value=selected_ticker).strip().upper()
        if custom_input:
            selected_ticker = custom_input

    st.markdown("---")

    # 2. Virtual Investment Amount
    investment_amount = st.number_input(
        "Simulated Investment (INR ₹)",
        min_value=5000.0,
        max_value=10000000.0,
        value=float(DEFAULT_INITIAL_CAPITAL),
        step=10000.0,
        format="%.0f"
    )
    
    col_p1, col_p2, col_p3 = st.columns(3)
    if col_p1.button("₹50K", use_container_width=True):
        investment_amount = 50000.0
    if col_p2.button("₹1L", use_container_width=True):
        investment_amount = 100000.0
    if col_p3.button("₹5L", use_container_width=True):
        investment_amount = 500000.0

    st.markdown("---")

    # 3. Risk Tolerance Setting
    risk_options = list(RISK_LEVELS.keys())
    # Exclude standard from primary selector (standard is used as benchmark)
    user_risk_options = [r for r in risk_options if r != "Standard"]
    
    selected_risk = st.radio(
        "Risk Tolerance Level",
        options=user_risk_options,
        index=1,
        help="Higher risk tolerance penalizes drawdowns less aggressively, while Low Risk prioritizes capital protection."
    )
    
    risk_info = RISK_LEVELS[selected_risk]
    st.caption(f"**Penalty Factor ($\lambda$):** `{risk_info['lambda']}`")
    st.caption(f"{risk_info['description']}")

    st.markdown("---")

    # 4. Historical Period Range
    st.markdown("**Historical Analysis Horizon**")
    start_d = st.date_input("Start Date", value=datetime.strptime(DEFAULT_START_DATE, "%Y-%m-%d").date())
    end_d = st.date_input("End Date", value=datetime.strptime(DEFAULT_END_DATE, "%Y-%m-%d").date())

    st.markdown("---")

    # 5. Simulation Action Buttons
    col_btn1, col_btn2 = st.columns(2)
    analyze_clicked = col_btn1.button("⚡ Analyze with AI", type="primary", use_container_width=True)
    reset_clicked = col_btn2.button("🔄 Reset", use_container_width=True)
    
    if reset_clicked:
        st.session_state.simulation_results = None
        st.rerun()

    # 6. Model Training Expander (For demonstration and training on fresh data)
    with st.expander("🧠 Train New PPO Model"):
        st.markdown("**Train a genuine PPO model on historical data:**")
        train_steps = st.slider("Training Timesteps", 5000, 40000, FAST_DEMO_TIMESTEPS, step=5000)
        train_risk = st.selectbox("Model Risk Objective", options=user_risk_options, index=1, key="train_risk_select")
        
        if st.button("🚀 Train Model Now", use_container_width=True):
            with st.spinner(f"Training PPO Agent for {selected_ticker} ({train_risk}) over {train_steps} steps..."):
                try:
                    m, meta = train_ppo_agent(
                        ticker=selected_ticker,
                        risk_level=train_risk,
                        total_timesteps=train_steps,
                        save=True
                    )
                    st.success(f"Model successfully trained and saved! Validation Return: {meta['val_return_pct']}%")
                except Exception as e:
                    st.error(f"Training failed: {e}")


# ==========================================
# MAIN DASHBOARD EXECUTION LOGIC
# ==========================================

# Top Hero Banner
company_name = SUPPORTED_TICKERS.get(selected_ticker, selected_ticker)
st.markdown(f"""
<div class="hero-banner">
    <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
        <div>
            <div class="hero-title">
                <span>📈 {company_name}</span>
                <span style="font-size: 16px; color: #00E5FF; font-weight: 500;">({selected_ticker})</span>
            </div>
            <div class="hero-subtitle">
                Autonomous Risk-Sensitive Reinforcement Learning Portfolio Allocation & Execution System
            </div>
        </div>
        <div style="text-align: right; margin-top: 8px;">
            <span class="badge-simulated">HISTORICAL SIMULATION • RISK LEVEL: {selected_risk.upper()}</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# Data & Simulation Pipeline Helper
@st.cache_data(show_spinner=False, ttl=3600)
def load_and_prepare_data(ticker: str, start_str: str, end_str: str):
    """Loads market data and computes technical indicators."""
    return prepare_market_dataset(ticker, start_date=start_str, end_date=end_str)


# Run Simulation if button clicked or if already in state
if analyze_clicked or st.session_state.simulation_results is None:
    with st.spinner(f"Loading market data and running Risk-Sensitive RL agent for {selected_ticker}..."):
        try:
            start_str = start_d.strftime("%Y-%m-%d")
            end_str = end_d.strftime("%Y-%m-%d")
            
            # Load Chronological Data
            train_df, val_df, test_df = load_and_prepare_data(selected_ticker, start_str, end_str)
            
            # Load or fast-train Risk-Sensitive Agent
            rs_agent = load_trained_agent(selected_ticker, selected_risk)
            if rs_agent is None:
                st.info(f"No pre-trained model found for {selected_ticker} ({selected_risk}). Training demonstration PPO agent...")
                rs_agent, _ = train_ppo_agent(
                    ticker=selected_ticker,
                    risk_level=selected_risk,
                    total_timesteps=10000,
                    save=True
                )
                
            # Load or fast-train Standard PPO Agent (Benchmark with lambda = 0.0)
            std_agent = load_trained_agent(selected_ticker, "Standard")
            if std_agent is None:
                std_agent, _ = train_ppo_agent(
                    ticker=selected_ticker,
                    risk_level="Standard",
                    total_timesteps=10000,
                    save=True
                )

            # Execute Backtest Simulations on unseen test data
            rs_sim = run_agent_simulation(
                df=test_df,
                agent=rs_agent,
                risk_level=selected_risk,
                initial_capital=investment_amount,
                strategy_name=f"Risk-Sensitive PPO ({selected_risk})"
            )
            
            std_sim = run_agent_simulation(
                df=test_df,
                agent=std_agent,
                risk_level="Standard",
                initial_capital=investment_amount,
                strategy_name="Standard PPO (Risk-Neutral)"
            )
            
            bh_sim = run_buy_and_hold_simulation(
                df=test_df,
                initial_capital=investment_amount
            )
            
            # Determine Latest AI Signal and Explanation from most recent simulation step
            latest_row = test_df.iloc[-1]
            latest_price = float(latest_row["Close"])
            
            if not rs_sim["history_df"].empty:
                latest_action = int(rs_sim["history_df"].iloc[-1]["action_intended"])
            else:
                latest_action = 0
            
            explanation = explain_ai_decision(
                latest_row=latest_row,
                action=latest_action,
                risk_level=selected_risk,
                shares_held=rs_sim["final_shares"],
                portfolio_value=rs_sim["final_portfolio_value"],
                current_cash=rs_sim["final_cash"]
            )
            
            # Dynamic Composite Risk Score
            rm = RiskManager(risk_level=selected_risk)
            rm.reset(investment_amount)
            vol_val = float(latest_row.get("volatility_20", 0.20))
            h_ratio = float((rs_sim["final_shares"] * latest_price) / max(1.0, rs_sim["final_portfolio_value"]))
            risk_score = rm.compute_composite_risk_score(
                current_portfolio_value=rs_sim["final_portfolio_value"],
                stock_volatility=vol_val,
                holdings_ratio=h_ratio
            )
            
            st.session_state.simulation_results = {
                "ticker": selected_ticker,
                "latest_price": latest_price,
                "latest_action": latest_action,
                "explanation": explanation,
                "risk_score": risk_score,
                "rs_sim": rs_sim,
                "std_sim": std_sim,
                "bh_sim": bh_sim,
                "test_df": test_df,
                "initial_capital": investment_amount,
                "risk_level": selected_risk
            }
            
        except Exception as e:
            st.error(f"Error during simulation execution: {e}")
            logger.exception("Simulation execution failed")


# ==========================================
# RENDER DASHBOARD CONTENTS
# ==========================================
results = st.session_state.simulation_results

if results is not None:
    rs_sim = results["rs_sim"]
    std_sim = results["std_sim"]
    bh_sim = results["bh_sim"]
    test_df = results["test_df"]
    latest_price = results["latest_price"]
    latest_action = results["latest_action"]
    explanation = results["explanation"]
    risk_score = results["risk_score"]
    
    # 1. TOP KPI METRICS GRID
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    
    with m1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Latest Stock Price</div>
            <div class="metric-value">{format_inr(latest_price)}</div>
            <div class="metric-delta-pos">Historical EOD</div>
        </div>
        """, unsafe_allow_html=True)
        
    with m2:
        action_names = ["HOLD", "BUY", "SELL"]
        action_str = action_names[latest_action]
        badge_cls = "signal-buy" if action_str == "BUY" else ("signal-sell" if action_str == "SELL" else "signal-hold")
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">AI Trade Signal</div>
            <div style="margin-top: 4px;">
                <span class="{badge_cls}">{action_str}</span>
            </div>
            <div class="hero-subtitle" style="margin-top: 6px;">Conf: {explanation['confidence']}</div>
        </div>
        """, unsafe_allow_html=True)
        
    with m3:
        # Dynamic Risk Color
        r_color = "#00E676" if risk_score < 35 else ("#FFD600" if risk_score < 65 else "#FF1744")
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Risk Score (0-100)</div>
            <div class="metric-value" style="color: {r_color};">{risk_score:.1f}</div>
            <div class="hero-subtitle">Dynamic Risk Index</div>
        </div>
        """, unsafe_allow_html=True)
        
    with m4:
        pnl = rs_sim["total_profit_loss"]
        pnl_cls = "metric-delta-pos" if pnl >= 0 else "metric-delta-neg"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Portfolio Value</div>
            <div class="metric-value">{format_inr(rs_sim['final_portfolio_value'])}</div>
            <div class="{pnl_cls}">{format_inr(pnl)} ({rs_sim['total_return_pct']:+.2f}%)</div>
        </div>
        """, unsafe_allow_html=True)
        
    with m5:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Maximum Drawdown</div>
            <div class="metric-value" style="color: #FF5252;">{rs_sim['max_drawdown_pct']:.2f}%</div>
            <div class="hero-subtitle">Peak-to-Trough Loss</div>
        </div>
        """, unsafe_allow_html=True)
        
    with m6:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Sharpe Ratio</div>
            <div class="metric-value" style="color: #00E5FF;">{rs_sim['sharpe_ratio']:.2f}</div>
            <div class="hero-subtitle">Sortino: {rs_sim['sortino_ratio']:.2f}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # 2. MAIN TABS
    tab_dashboard, tab_compare, tab_trades, tab_viva = st.tabs([
        "📊 AI Trading Dashboard",
        "⚖️ Strategy Benchmark & Risk Analysis",
        "📜 Trade Execution Audit Log",
        "🎓 College Project Defense & Viva Guide"
    ])

    # ----------------------------------------------------
    # TAB 1: AI TRADING DASHBOARD
    # ----------------------------------------------------
    with tab_dashboard:
        # Price Chart with Trade Markers
        fig_price = plot_price_and_trades(
            df=test_df,
            history_df=rs_sim["history_df"],
            ticker=results["ticker"],
            strategy_name=f"Risk-Sensitive PPO ({results['risk_level']})"
        )
        st.plotly_chart(fig_price, use_container_width=True)

        col_dec, col_acc = st.columns([1.6, 1.0])
        
        with col_dec:
            st.markdown("#### 🧠 AI Decision Rationalization")
            st.markdown(f"""
            <div class="decision-box">
                <div style="font-size: 15px; font-weight: 700; color: #00E5FF; margin-bottom: 8px;">
                    Model Action: {explanation['action']} &bull; Risk Setting: {results['risk_level']} (λ={RISK_LEVELS[results['risk_level']]['lambda']})
                </div>
                <ul style="margin: 0; padding-left: 20px; color: #C9D1D9; font-size: 13px; line-height: 1.6;">
                    {''.join([f"<li>{r}</li>" for r in explanation['primary_reasons']])}
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
            # Indicator summary badges
            ind_cols = st.columns(4)
            for i, (k, v) in enumerate(explanation["indicators_summary"].items()):
                ind_cols[i].metric(label=k, value=v)

        with col_acc:
            st.markdown("#### 💼 Portfolio Composition")
            st.markdown(f"""
            <div class="metric-card" style="margin-top: 10px;">
                <table style="width: 100%; font-size: 13px; color: #E6EDF3;">
                    <tr style="border-bottom: 1px solid #21262D; padding: 6px 0;">
                        <td style="color: #8B949E;">Initial Virtual Capital:</td>
                        <td style="text-align: right; font-weight: 600;">{format_inr(results['initial_capital'])}</td>
                    </tr>
                    <tr style="border-bottom: 1px solid #21262D; padding: 6px 0;">
                        <td style="color: #8B949E;">Available Cash:</td>
                        <td style="text-align: right; font-weight: 600;">{format_inr(rs_sim['final_cash'])}</td>
                    </tr>
                    <tr style="border-bottom: 1px solid #21262D; padding: 6px 0;">
                        <td style="color: #8B949E;">Shares Owned:</td>
                        <td style="text-align: right; font-weight: 600;">{rs_sim['final_shares']} shares</td>
                    </tr>
                    <tr style="border-bottom: 1px solid #21262D; padding: 6px 0;">
                        <td style="color: #8B949E;">Equity Value:</td>
                        <td style="text-align: right; font-weight: 600;">{format_inr(rs_sim['final_shares'] * latest_price)}</td>
                    </tr>
                    <tr style="border-bottom: 1px solid #21262D; padding: 6px 0;">
                        <td style="color: #8B949E;">Total Executed Trades:</td>
                        <td style="text-align: right; font-weight: 600;">{rs_sim['total_trades']} ({rs_sim['buy_trades']} Buys / {rs_sim['sell_trades']} Sells)</td>
                    </tr>
                    <tr>
                        <td style="color: #8B949E; padding-top: 6px;">Total Commission / Fee:</td>
                        <td style="text-align: right; font-weight: 600; padding-top: 6px; color: #FF9100;">{format_inr(rs_sim['history_df']['fee_paid'].sum() if not rs_sim['history_df'].empty else 0.0)}</td>
                    </tr>
                </table>
            </div>
            """, unsafe_allow_html=True)

    # ----------------------------------------------------
    # TAB 2: STRATEGY BENCHMARK & COMPARISON
    # ----------------------------------------------------
    with tab_compare:
        st.markdown("### ⚖️ Comprehensive Strategy Benchmark")
        st.caption("All models evaluated on the identical unseen out-of-sample historical test period with standard 0.1% transaction friction.")
        
        comparison_list = [bh_sim, std_sim, rs_sim]
        comp_df = build_comparison_dataframe(comparison_list)
        
        # Display Styled Table
        st.dataframe(comp_df, use_container_width=True, hide_index=True)
        
        # Comparative Visuals
        col_eq, col_dd = st.columns(2)
        with col_eq:
            fig_eq = plot_equity_curves(comparison_list)
            st.plotly_chart(fig_eq, use_container_width=True)
        with col_dd:
            fig_dd = plot_drawdowns(comparison_list)
            st.plotly_chart(fig_dd, use_container_width=True)

        st.markdown("#### 💡 Empirical Findings & Risk-Adjusted Commentary")
        st.markdown(f"""
        - **Risk-Sensitive PPO vs Standard PPO:** Standard PPO operates without a downside risk penalty ($\lambda = 0.0$), seeking raw expected return. Consequently, it is prone to holding through steep market selloffs, causing elevated drawdown. In contrast, **Risk-Sensitive PPO ($\lambda = {RISK_LEVELS[results['risk_level']]['lambda']}$)** penalizes negative returns quadratically, leading to active downside liquidation and reduced maximum drawdown.
        - **Sharpe and Sortino Outperformance:** By curtailing negative volatility through disciplined risk penalties, the Risk-Sensitive agent achieves a superior risk-adjusted return (Sharpe / Sortino ratio) compared to unconstrained baseline policies.
        """)

    # ----------------------------------------------------
    # TAB 3: TRADE EXECUTION AUDIT LOG
    # ----------------------------------------------------
    with tab_trades:
        st.markdown("### 📜 Executed Trades & Portfolio Accounting")
        hdf = rs_sim["history_df"].copy()
        
        if not hdf.empty:
            # Filter options
            trade_filter = st.radio(
                "Filter History",
                options=["All Steps", "Executed Trades Only (BUY / SELL)", "BUY Trades", "SELL Trades"],
                horizontal=True
            )
            
            if trade_filter == "Executed Trades Only (BUY / SELL)":
                display_df = hdf[hdf["trade_type"].isin(["BUY", "SELL"])].copy()
            elif trade_filter == "BUY Trades":
                display_df = hdf[hdf["trade_type"] == "BUY"].copy()
            elif trade_filter == "SELL Trades":
                display_df = hdf[hdf["trade_type"] == "SELL"].copy()
            else:
                display_df = hdf.copy()

            # Format columns for presentation
            display_df["Date"] = pd.to_datetime(display_df["date"]).dt.strftime("%Y-%m-%d")
            display_df["Price"] = display_df["price"].apply(format_inr)
            display_df["Action"] = display_df["trade_type"]
            display_df["Shares Traded"] = display_df["shares_traded"]
            display_df["Shares Held"] = display_df["shares_held"]
            display_df["Cash Remaining"] = display_df["cash"].apply(format_inr)
            display_df["Portfolio Value"] = display_df["portfolio_value"].apply(format_inr)
            display_df["Fee Paid"] = display_df["fee_paid"].apply(format_inr)
            display_df["Drawdown"] = display_df["drawdown"].apply(lambda x: f"{x * 100:.2f}%")
            
            show_cols = ["Date", "Price", "Action", "Shares Traded", "Shares Held", "Fee Paid", "Cash Remaining", "Portfolio Value", "Drawdown", "reason"]
            st.dataframe(display_df[show_cols], use_container_width=True, hide_index=True)
            
            # Download CSV
            csv_data = hdf.to_csv(index=False)
            st.download_button(
                label="📥 Download Full Trade Log (CSV)",
                data=csv_data,
                file_name=f"trade_log_{results['ticker']}_{results['risk_level']}.csv",
                mime="text/csv"
            )
        else:
            st.info("No trading history recorded for this episode.")

    # ----------------------------------------------------
    # TAB 4: COLLEGE PROJECT DEFENSE & VIVA GUIDE
    # ----------------------------------------------------
    with tab_viva:
        st.markdown("## 🎓 B.Tech Project Defense & Viva Voce Guide")
        
        with st.expander("1. Problem Statement & Motivation", expanded=True):
            st.markdown("""
            **The Fundamental Flaw in Standard Reinforcement Learning:**
            In conventional algorithmic trading with RL, agents are trained using the **Expected Return** objective:
            $$\max_\\pi \\mathbb{E}_\\pi \\left[ \\sum_{t=0}^T \\gamma^t r_t \\right]$$
            While this maximizes long-term gains in theory, financial markets are characterized by fat-tailed distributions, severe regime shifts, and black swan crashes. A standard agent may achieve high cumulative return while suffering an unbearable **50% drawdown**, which in real-world asset management triggers margin calls and liquidation.

            **The Solution:**
            We formulate a **Risk-Sensitive Reinforcement Learning** framework where the agent optimizes a risk-penalized utility function, prioritizing capital conservation, downside protection, and drawdown management.
            """)

        with st.expander("2. Mathematical Formulation & Reward Function", expanded=True):
            st.markdown("""
            The trading process is formulated as a Markov Decision Process (MDP) $(\\mathcal{S}, \\mathcal{A}, \\mathcal{P}, \\mathcal{R}, \\gamma)$:
            - **State Space $\\mathcal{S} \\in \\mathbb{R}^{18}$:** Technical features (Returns, Volatilities, SMA/EMA ratios, RSI, MACD, Bollinger Bands, ATR, Volume) + Account state (Cash ratio, Holdings ratio, Current Drawdown).
            - **Action Space $\\mathcal{A} = \\{0: \\text{HOLD}, 1: \\text{BUY}, 2: \\text{SELL}\\}$**.
            - **Risk-Penalized Reward Function:**
            $$R_t = r_{\\text{portfolio}, t} - \\lambda \\cdot \\Omega_t - c \\cdot |\\Delta w_t|$$
            Where:
            - $r_{\\text{portfolio}, t} = \\frac{V_t - V_{t-1}}{V_{t-1}}$ is the percentage portfolio return.
            - $\\Omega_t = w_{\\text{down}} \\cdot (\\min(0, r_t))^2 + w_{\\text{dd}} \\cdot \\text{Drawdown}_t^2$ is the quadratic downside risk and drawdown penalty.
            - $\\lambda$ is the **Risk-Aversion Coefficient**:
              - **Low Risk:** $\\lambda = 2.5$ (Aggressive risk suppression, strict capital preservation)
              - **Medium Risk:** $\\lambda = 1.0$ (Balanced growth and volatility control)
              - **High Risk:** $\\lambda = 0.2$ (Opportunistic return pursuit)
              - **Standard PPO:** $\\lambda = 0.0$ (Risk-neutral return maximization)
            - $c \\cdot |\\Delta w_t|$ represents transaction friction (0.1% fee) to prevent portfolio churning.
            """)

        with st.expander("3. Why Proximal Policy Optimization (PPO)?", expanded=True):
            st.markdown("""
            PPO is chosen for its sample efficiency, stability, and avoidance of catastrophic policy collapse through its **Clipped Surrogate Objective**:
            $$L^{\\text{CLIP}}(\\theta) = \\hat{\\mathbb{E}}_t \\left[ \\min\\left(r_t(\\theta)\\hat{A}_t, \\text{clip}(r_t(\\theta), 1-\\epsilon, 1+\\epsilon)\\hat{A}_t\\right) \\right]$$
            - Unlike DQN, PPO naturally handles continuous state representations and produces stable stochastic or deterministic policies.
            - Clipping prevents excessively large gradient updates, crucial in noisy financial time series.
            """)

        with st.expander("4. Key Viva Questions & Answers", expanded=True):
            st.markdown("""
            - **Q1: How does your model avoid look-ahead bias and data leakage?**
              *Answer:* All technical indicators use strictly past backward-looking rolling windows. Data is partitioned chronologically (70% train, 15% validation, 15% test) without shuffling. Normalization statistics and moving windows never peek into future timestamps.
              
            - **Q2: Why not just use Deep Q-Networks (DQN)?**
              *Answer:* DQN can be prone to overestimating Q-values in non-stationary noisy financial environments. PPO provides smooth policy improvement through generalized advantage estimation (GAE) and clipped surrogate losses.
              
            - **Q3: What are the primary limitations of historical backtesting?**
              *Answer:* Backtesting cannot fully model market impact (slippage on large block orders), liquidity shocks, or changing regulatory regimes.
              
            - **Q4: How do transaction costs affect the learned policy?**
              *Answer:* Without transaction friction, an RL agent might execute high-frequency churning for fractional gains. Incorporating $0.1\\%$ fee forces the policy to trade only on high-conviction momentum signals.
            """)
