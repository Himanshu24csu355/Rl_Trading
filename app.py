"""Risk-Sensitive Reinforcement Learning Stock Trading and Risk Management Platform.
A professional, modern, dark-themed Streamlit dashboard providing two distinct analysis modes:
1. Mode A: Historical Backtesting Simulation (unseen chronological test evaluation)
2. Mode B: Current Market Decision Support (real-time virtual decision support with trading calendar awareness)
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
    DEFAULT_LOOKBACK_YEARS,
    RISK_LEVELS,
    TRANSACTION_FEE_PERCENT,
    MODELS_DIR,
    FAST_DEMO_TIMESTEPS,
    MODE_BACKTEST,
    MODE_CURRENT,
    ANALYSIS_MODES,
)
from src.utils import format_inr, format_pct, setup_logger
from src.data_loader import (
    fetch_stock_data,
    prepare_market_dataset,
    get_market_session_info,
    fetch_current_market_observation,
)
from src.indicators import add_all_indicators
from src.risk_manager import RiskManager
from src.train_model import (
    train_ppo_agent,
    load_trained_agent,
    get_model_path,
    get_metadata_path,
    get_model_info,
    get_policy_action_distribution,
)
from src.backtest import (
    run_agent_simulation,
    run_buy_and_hold_simulation,
    explain_ai_decision,
    evaluate_current_market_decision,
)
from src.evaluation import (
    build_comparison_dataframe,
    generate_empirical_findings,
    plot_price_and_trades,
    plot_equity_curves,
    plot_drawdowns,
    plot_policy_distribution,
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
        margin-bottom: 20px;
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
    
    /* Mode Badges */
    .badge-backtest {
        background: rgba(0, 229, 255, 0.15);
        color: #00E5FF;
        border: 1px solid rgba(0, 229, 255, 0.4);
        padding: 5px 12px;
        border-radius: 6px;
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 0.5px;
        text-transform: uppercase;
    }
    .badge-current {
        background: rgba(255, 171, 0, 0.15);
        color: #FFB300;
        border: 1px solid rgba(255, 171, 0, 0.4);
        padding: 5px 12px;
        border-radius: 6px;
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 0.5px;
        text-transform: uppercase;
    }
    
    /* Freshness & Status Badges */
    .badge-fresh {
        background: rgba(0, 230, 118, 0.15);
        color: #00E676;
        border: 1px solid #00E676;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 11px;
        font-weight: 600;
    }
    .badge-closed {
        background: rgba(255, 23, 68, 0.15);
        color: #FF1744;
        border: 1px solid #FF1744;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 11px;
        font-weight: 600;
    }
    
    /* Metric Card Grid */
    .metric-card {
        background-color: #161B26;
        border: 1px solid #21262D;
        border-radius: 10px;
        padding: 16px 18px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.25);
        transition: transform 0.2s ease, border-color 0.2s ease;
        white-space: nowrap !important;
    }
    .metric-card:hover {
        border-color: #00E5FF;
        transform: translateY(-2px);
    }
    .metric-title {
        font-size: 11px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.6px;
        color: #8B949E;
        margin-bottom: 6px;
    }
    .metric-value {
        font-size: 21px;
        font-weight: 700;
        color: #FFFFFF;
        margin-bottom: 4px;
        white-space: nowrap !important;
    }
    .metric-delta-pos {
        font-size: 12px;
        font-weight: 600;
        color: #00E676;
        white-space: nowrap !important;
    }
    .metric-delta-neg {
        font-size: 12px;
        font-weight: 600;
        color: #FF1744;
        white-space: nowrap !important;
    }
    
    /* Signal Badges */
    .signal-buy {
        display: inline-block;
        background-color: rgba(0, 230, 118, 0.15);
        color: #00E676;
        border: 1px solid #00E676;
        border-radius: 8px;
        padding: 6px 14px;
        font-size: 15px;
        font-weight: 800;
        letter-spacing: 1px;
    }
    .signal-sell {
        display: inline-block;
        background-color: rgba(255, 23, 68, 0.15);
        color: #FF1744;
        border: 1px solid #FF1744;
        border-radius: 8px;
        padding: 6px 14px;
        font-size: 15px;
        font-weight: 800;
        letter-spacing: 1px;
    }
    .signal-hold {
        display: inline-block;
        background-color: rgba(255, 214, 0, 0.15);
        color: #FFD600;
        border: 1px solid #FFD600;
        border-radius: 8px;
        padding: 6px 14px;
        font-size: 15px;
        font-weight: 800;
        letter-spacing: 1px;
    }
    
    /* Notice and Callout Boxes */
    .info-callout {
        background-color: #141A24;
        border-left: 4px solid #00E5FF;
        border-radius: 0 8px 8px 0;
        padding: 14px 18px;
        margin: 12px 0;
        font-size: 13px;
        color: #C9D1D9;
        line-height: 1.5;
    }
    .disclaimer-box {
        background-color: rgba(255, 171, 0, 0.08);
        border: 1px solid rgba(255, 171, 0, 0.3);
        border-radius: 8px;
        padding: 12px 16px;
        margin: 14px 0;
        font-size: 12px;
        color: #FFD54F;
    }
</style>
""", unsafe_allow_html=True)


# Initialize Session State
if "backtest_results" not in st.session_state:
    st.session_state.backtest_results = None
if "current_market_results" not in st.session_state:
    st.session_state.current_market_results = None


# ==========================================
# SIDEBAR CONTROLS
# ==========================================
with st.sidebar:
    st.markdown("### 🧭 Platform Mode")
    
    # 1. Top Mode Selector
    active_mode = st.radio(
        "Choose Analysis Mode",
        options=[MODE_BACKTEST, MODE_CURRENT],
        index=0,
        help="Mode A evaluates historical performance on out-of-sample data. Mode B retrieves the latest verified session for virtual decision support."
    )
    
    mode_badge_html = (
        '<div style="margin-bottom: 12px;"><span class="badge-backtest">MODE A: HISTORICAL BACKTESTING</span></div>'
        if active_mode == MODE_BACKTEST
        else '<div style="margin-bottom: 12px;"><span class="badge-current">MODE B: CURRENT DECISION SUPPORT</span></div>'
    )
    st.markdown(mode_badge_html, unsafe_allow_html=True)
    st.markdown("---")

    # 2. Stock Selection
    st.markdown("### 📈 Asset Selection")
    selected_ticker = st.selectbox(
        "Select Stock Ticker (NSE)",
        options=list(SUPPORTED_TICKERS.keys()),
        format_func=lambda x: f"{x} - {SUPPORTED_TICKERS.get(x, '')}",
        index=0
    )
    
    use_custom = st.checkbox("Custom Asset Symbol")
    if use_custom:
        custom_input = st.text_input("Custom Symbol (e.g. INFY.NS, AAPL)", value=selected_ticker).strip().upper()
        if custom_input:
            selected_ticker = custom_input

    st.markdown("---")

    # 3. Virtual Investment Amount
    st.markdown("### 💰 Virtual Capital")
    investment_amount = st.number_input(
        "Simulated Investment (INR ₹)",
        min_value=5000.0,
        max_value=10000000.0,
        value=float(DEFAULT_INITIAL_CAPITAL),
        step=5000.0,
        format="%.0f"
    )
    
    col_p1, col_p2, col_p3 = st.columns(3)
    if col_p1.button("₹10K", use_container_width=True):
        investment_amount = 10000.0
    if col_p2.button("₹50K", use_container_width=True):
        investment_amount = 50000.0
    if col_p3.button("₹1L", use_container_width=True):
        investment_amount = 100000.0

    st.markdown("---")

    # 4. Risk Tolerance Setting
    st.markdown("### 🛡️ Risk Setting")
    user_risk_options = [r for r in RISK_LEVELS.keys() if r != "Standard"]
    
    selected_risk = st.radio(
        "Risk Tolerance Level",
        options=user_risk_options,
        index=1,
        help="Higher risk tolerance penalizes drawdowns less aggressively; Low Risk prioritizes capital protection."
    )
    
    risk_info = RISK_LEVELS[selected_risk]
    st.caption(f"**Penalty ($\lambda$):** `{risk_info['lambda']}` &bull; {risk_info['description']}")

    st.markdown("---")

    # 5. Mode-Specific Date / Lookback Parameters
    if active_mode == MODE_BACKTEST:
        st.markdown("### 📅 Backtest Period")
        start_d = st.date_input("Start Date", value=datetime.strptime(DEFAULT_START_DATE, "%Y-%m-%d").date())
        end_d = st.date_input("End Date", value=datetime.strptime(DEFAULT_END_DATE, "%Y-%m-%d").date())
        st.caption("Chronological split: 70% Train, 15% Validation, 15% Unseen Test.")
    else:
        st.markdown("### 🔍 Historical Lookback")
        lookback_years = st.slider("Lookback Window (Years)", min_value=2, max_value=5, value=DEFAULT_LOOKBACK_YEARS)
        st.caption("Lookback window used to compute stationary technical indicators for the current session observation.")

    st.markdown("---")

    # 6. Model Status Inspection
    st.markdown("### 🧠 Model Status")
    model_info = get_model_info(selected_ticker, selected_risk)
    if model_info["exists"]:
        st.success(f"**Loaded Model:** `{selected_ticker}` ({selected_risk})")
        st.caption(f"Trained: {model_info['trained_date']} | Steps: {model_info['total_timesteps']}")
    else:
        st.warning(f"⚠️ **Missing Model:** No pre-trained model found for `{selected_ticker}` ({selected_risk}).")
        st.caption("Click below to train an agent explicitly. Models are never silently auto-trained.")

    # Model Training Expander
    with st.expander("⚙️ Train / Retrain PPO Agent"):
        st.markdown(f"Train PPO for **{selected_ticker}** [{selected_risk}]")
        train_steps = st.slider("Training Timesteps", 5000, 30000, FAST_DEMO_TIMESTEPS, step=5000)
        
        if st.button("🚀 Train Model Now", use_container_width=True):
            with st.spinner(f"Training PPO Agent for {selected_ticker} ({selected_risk}) over {train_steps} steps..."):
                try:
                    m, meta = train_ppo_agent(
                        ticker=selected_ticker,
                        risk_level=selected_risk,
                        total_timesteps=train_steps,
                        save=True
                    )
                    st.success(f"Model trained and saved successfully! Validation Return: {meta['val_return_pct']}%")
                    st.rerun()
                except Exception as e:
                    st.error(f"Training failed: {e}")

    st.markdown("---")

    # 7. Execution Buttons
    col_run, col_reset = st.columns(2)
    btn_label = "⚡ Run Backtest" if active_mode == MODE_BACKTEST else "🎯 Analyze Market"
    run_clicked = col_run.button(btn_label, type="primary", use_container_width=True)
    reset_clicked = col_reset.button("🔄 Reset", use_container_width=True)
    
    if reset_clicked:
        st.session_state.backtest_results = None
        st.session_state.current_market_results = None
        st.rerun()


# ==========================================
# MAIN DASHBOARD EXECUTION & RENDERING
# ==========================================

company_name = SUPPORTED_TICKERS.get(selected_ticker, selected_ticker)

# Mode Header Banner
if active_mode == MODE_BACKTEST:
    st.markdown(f"""
    <div class="hero-banner">
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
            <div>
                <div class="hero-title">
                    <span>📈 {company_name}</span>
                    <span style="font-size: 16px; color: #00E5FF; font-weight: 500;">({selected_ticker})</span>
                </div>
                <div class="hero-subtitle">
                    Mode A: Historical Backtesting on Unseen Out-of-Sample Market Data (Zero Lookahead Bias)
                </div>
            </div>
            <div style="text-align: right; margin-top: 8px;">
                <span class="badge-backtest">HISTORICAL BACKTEST • RISK: {selected_risk.upper()} (λ={RISK_LEVELS[selected_risk]['lambda']})</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
else:
    session_info = get_market_session_info()
    st.markdown(f"""
    <div class="hero-banner">
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
            <div>
                <div class="hero-title">
                    <span>🎯 {company_name}</span>
                    <span style="font-size: 16px; color: #00E5FF; font-weight: 500;">({selected_ticker})</span>
                </div>
                <div class="hero-subtitle">
                    Mode B: Current Market Virtual Decision Support & Risk Analysis
                </div>
            </div>
            <div style="text-align: right; margin-top: 8px;">
                <span class="badge-current">CURRENT MARKET MODE • RISK: {selected_risk.upper()} (λ={RISK_LEVELS[selected_risk]['lambda']})</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ==============================================================================
# MODE A: HISTORICAL BACKTESTING SIMULATION
# ==============================================================================
if active_mode == MODE_BACKTEST:
    
    # Check Model Availability before running
    if not model_info["exists"]:
        st.warning(
            f"⚠️ **Pre-trained model unavailable for {selected_ticker} ({selected_risk}).** "
            "In accordance with evaluation standards, models are not silently generated during analysis. "
            "Please use the **Train / Retrain PPO Agent** option in the sidebar to generate a genuine model."
        )
    else:
        if run_clicked or st.session_state.backtest_results is None:
            with st.spinner(f"Running historical simulation for {selected_ticker}..."):
                try:
                    start_str = start_d.strftime("%Y-%m-%d")
                    end_str = end_d.strftime("%Y-%m-%d")
                    
                    # 1. Load chronological datasets
                    train_df, val_df, test_df = prepare_market_dataset(selected_ticker, start_str, end_str)
                    
                    # 2. Load trained agents (No silent retraining)
                    rs_agent = load_trained_agent(selected_ticker, selected_risk)
                    std_agent = load_trained_agent(selected_ticker, "Standard")
                    
                    # 3. Run backtest simulations
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
                    
                    # Latest test step metrics
                    latest_row = test_df.iloc[-1]
                    latest_price = float(latest_row["Close"])
                    latest_action = int(rs_sim["history_df"].iloc[-1]["action_intended"]) if not rs_sim["history_df"].empty else 0
                    
                    # Get policy action distribution
                    state_vec = rs_sim["history_df"].iloc[-1] if not rs_sim["history_df"].empty else None
                    action_probs = None
                    if rs_agent is not None:
                        try:
                            # Build latest test obs
                            tech_f = latest_row[list(rs_sim["history_df"].columns)].values if False else None
                        except Exception:
                            pass
                            
                    explanation = explain_ai_decision(
                        latest_row=latest_row,
                        action=latest_action,
                        risk_level=selected_risk,
                        shares_held=rs_sim["final_shares"],
                        portfolio_value=rs_sim["final_portfolio_value"],
                        current_cash=rs_sim["final_cash"]
                    )
                    
                    rm = RiskManager(risk_level=selected_risk)
                    rm.reset(investment_amount)
                    vol_val = float(latest_row.get("volatility_20", 0.20))
                    h_ratio = float((rs_sim["final_shares"] * latest_price) / max(1.0, rs_sim["final_portfolio_value"]))
                    risk_score = rm.compute_composite_risk_score(
                        current_portfolio_value=rs_sim["final_portfolio_value"],
                        stock_volatility=vol_val,
                        holdings_ratio=h_ratio
                    )
                    
                    st.session_state.backtest_results = {
                        "ticker": selected_ticker,
                        "latest_price": latest_price,
                        "latest_action": latest_action,
                        "explanation": explanation,
                        "risk_score": risk_score,
                        "rs_sim": rs_sim,
                        "std_sim": std_sim,
                        "bh_sim": bh_sim,
                        "train_df": train_df,
                        "val_df": val_df,
                        "test_df": test_df,
                        "initial_capital": investment_amount,
                        "risk_level": selected_risk
                    }
                except Exception as e:
                    st.error(f"Error during backtesting simulation: {e}")
                    logger.exception("Backtest execution failed")

        bt_res = st.session_state.backtest_results
        if bt_res is not None:
            rs_sim = bt_res["rs_sim"]
            std_sim = bt_res["std_sim"]
            bh_sim = bt_res["bh_sim"]
            test_df = bt_res["test_df"]
            train_df = bt_res["train_df"]
            val_df = bt_res["val_df"]
            latest_price = bt_res["latest_price"]
            latest_action = bt_res["latest_action"]
            explanation = bt_res["explanation"]
            risk_score = bt_res["risk_score"]
            
            # Chronological Partition Banner
            t_start = train_df["Date"].iloc[0].strftime("%Y-%m-%d") if "Date" in train_df.columns else "Start"
            t_end = train_df["Date"].iloc[-1].strftime("%Y-%m-%d") if "Date" in train_df.columns else "End"
            v_start = val_df["Date"].iloc[0].strftime("%Y-%m-%d") if "Date" in val_df.columns else "Start"
            v_end = val_df["Date"].iloc[-1].strftime("%Y-%m-%d") if "Date" in val_df.columns else "End"
            te_start = test_df["Date"].iloc[0].strftime("%Y-%m-%d") if "Date" in test_df.columns else "Start"
            te_end = test_df["Date"].iloc[-1].strftime("%Y-%m-%d") if "Date" in test_df.columns else "End"
            
            st.markdown(f"""
            <div class="info-callout">
                <b>Strict Chronological Split (Zero Lookahead Bias):</b><br>
                &bull; <b>Training Horizon (70%):</b> {t_start} to {t_end} ({len(train_df)} bars)<br>
                &bull; <b>Validation Horizon (15%):</b> {v_start} to {v_end} ({len(val_df)} bars)<br>
                &bull; <b>Unseen Test Evaluation (15%):</b> {te_start} to {te_end} ({len(test_df)} bars)<br>
                <i>The RL agent is evaluated strictly on out-of-sample data never seen during gradient updates.</i>
            </div>
            """, unsafe_allow_html=True)

            # KPI Grid
            k1, k2, k3, k4, k5, k6 = st.columns(6)
            k1.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Test End Price</div>
                <div class="metric-value">{format_inr(latest_price)}</div>
                <div class="metric-delta-pos">{te_end}</div>
            </div>
            """, unsafe_allow_html=True)
            
            act_names = ["HOLD", "BUY", "SELL"]
            act_s = act_names[latest_action]
            b_cls = "signal-buy" if act_s == "BUY" else ("signal-sell" if act_s == "SELL" else "signal-hold")
            k2.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Latest Signal</div>
                <div style="margin-top: 4px;"><span class="{b_cls}">{act_s}</span></div>
                <div class="hero-subtitle" style="margin-top: 6px;">Conf: {explanation['confidence']}</div>
            </div>
            """, unsafe_allow_html=True)
            
            r_c = "#00E676" if risk_score < 35 else ("#FFD600" if risk_score < 65 else "#FF1744")
            k3.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Risk Score (0-100)</div>
                <div class="metric-value" style="color: {r_c};">{risk_score:.1f}</div>
                <div class="hero-subtitle">Trailing Index</div>
            </div>
            """, unsafe_allow_html=True)
            
            pnl = rs_sim["total_profit_loss"]
            p_cls = "metric-delta-pos" if pnl >= 0 else "metric-delta-neg"
            k4.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Final Portfolio</div>
                <div class="metric-value">{format_inr(rs_sim['final_portfolio_value'])}</div>
                <div class="{p_cls}">{format_inr(pnl)} ({rs_sim['total_return_pct']:+.2f}%)</div>
            </div>
            """, unsafe_allow_html=True)
            
            k5.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Max Drawdown</div>
                <div class="metric-value" style="color: #FF5252;">{rs_sim['max_drawdown_pct']:.2f}%</div>
                <div class="hero-subtitle">Out-of-sample MDD</div>
            </div>
            """, unsafe_allow_html=True)
            
            k6.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Sharpe Ratio</div>
                <div class="metric-value" style="color: #00E5FF;">{rs_sim['sharpe_ratio']:.2f}</div>
                <div class="hero-subtitle">Sortino: {rs_sim['sortino_ratio']:.2f}</div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Tabs
            tab1, tab2, tab3, tab4 = st.tabs([
                "📊 AI Trading Dashboard",
                "⚖️ Strategy Benchmark & Risk Analysis",
                "📜 Trade Execution Audit Log",
                "🎓 College Project Defense & Viva Guide"
            ])

            with tab1:
                fig_price = plot_price_and_trades(
                    df=test_df,
                    history_df=rs_sim["history_df"],
                    ticker=bt_res["ticker"],
                    strategy_name=f"Risk-Sensitive PPO ({bt_res['risk_level']})"
                )
                st.plotly_chart(fig_price, use_container_width=True)

                col_dec, col_acc = st.columns([1.6, 1.0])
                with col_dec:
                    st.markdown("#### 🧠 AI Decision Rationalization")
                    st.markdown(f"""
                    <div class="info-callout">
                        <div style="font-size: 15px; font-weight: 700; color: #00E5FF; margin-bottom: 8px;">
                            Model Action: {explanation['action']} &bull; Risk Setting: {bt_res['risk_level']} (λ={RISK_LEVELS[bt_res['risk_level']]['lambda']})
                        </div>
                        <ul style="margin: 0; padding-left: 20px; color: #C9D1D9; font-size: 13px; line-height: 1.6;">
                            {''.join([f"<li>{r}</li>" for r in explanation['primary_reasons']])}
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    icols = st.columns(4)
                    for i, (k, v) in enumerate(list(explanation["indicators_summary"].items())[:4]):
                        icols[i].metric(label=k, value=v)
                        
                with col_acc:
                    st.markdown("#### 💼 Portfolio Composition")
                    st.markdown(f"""
                    <div class="metric-card" style="margin-top: 10px;">
                        <table style="width: 100%; font-size: 13px; color: #E6EDF3;">
                            <tr style="border-bottom: 1px solid #21262D; padding: 6px 0;">
                                <td style="color: #8B949E;">Initial Virtual Capital:</td>
                                <td style="text-align: right; font-weight: 600;">{format_inr(bt_res['initial_capital'])}</td>
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
                                <td style="color: #8B949E;">Total Executed Trades:</td>
                                <td style="text-align: right; font-weight: 600;">{rs_sim['total_trades']} ({rs_sim['buy_trades']} Buys / {rs_sim['sell_trades']} Sells)</td>
                            </tr>
                            <tr>
                                <td style="color: #8B949E; padding-top: 6px;">Total Commissions (0.1%):</td>
                                <td style="text-align: right; font-weight: 600; padding-top: 6px; color: #FF9100;">{format_inr(rs_sim.get('total_fees_paid', 0.0))}</td>
                            </tr>
                        </table>
                    </div>
                    """, unsafe_allow_html=True)

            with tab2:
                st.markdown("### ⚖️ Comprehensive Strategy Benchmark")
                st.caption("All models evaluated on the identical unseen test period with standard 0.1% transaction friction.")
                
                comparison_list = [bh_sim, std_sim, rs_sim]
                comp_df = build_comparison_dataframe(comparison_list)
                st.dataframe(comp_df, use_container_width=True, hide_index=True)
                
                col_eq, col_dd = st.columns(2)
                with col_eq:
                    fig_eq = plot_equity_curves(comparison_list)
                    st.plotly_chart(fig_eq, use_container_width=True)
                with col_dd:
                    fig_dd = plot_drawdowns(comparison_list)
                    st.plotly_chart(fig_dd, use_container_width=True)

                st.markdown("#### 💡 Data-Driven Empirical Findings & Honest Risk Analysis")
                dynamic_commentary = generate_empirical_findings(comparison_list)
                st.markdown(dynamic_commentary)

            with tab3:
                st.markdown("### 📜 Executed Trades & Portfolio Accounting")
                hdf = rs_sim["history_df"].copy()
                if not hdf.empty:
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
                    
                    csv_data = hdf.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Full Trade Log (CSV)",
                        data=csv_data,
                        file_name=f"trade_log_{bt_res['ticker']}_{bt_res['risk_level']}.csv",
                        mime="text/csv"
                    )
                else:
                    st.info("No trading history recorded for this episode.")

            with tab4:
                st.markdown("## 🎓 B.Tech Project Defense & Viva Voce Guide")
                with st.expander("1. Problem Statement & Motivation", expanded=True):
                    st.markdown("""
                    **The Fundamental Flaw in Standard Reinforcement Learning:**
                    In conventional algorithmic trading with RL, agents maximize **Expected Return**:
                    $$\\max_\\pi \\mathbb{E}_\\pi \\left[ \\sum_{t=0}^T \\gamma^t r_t \\right]$$
                    While this maximizes long-term gains in expectation, financial markets exhibit fat tails and crashes. A standard agent may achieve high cumulative return while suffering an unbearable **50% drawdown**, triggering margin calls and liquidation.

                    **The Solution:**
                    We formulate a **Risk-Sensitive Reinforcement Learning** framework where the agent optimizes a risk-penalized utility function, prioritizing capital conservation, downside protection, and drawdown management.
                    """)

                with st.expander("2. Mathematical Formulation & Reward Function", expanded=True):
                    st.markdown("""
                    The trading process is formulated as an MDP $(\\mathcal{S}, \\mathcal{A}, \\mathcal{P}, \\mathcal{R}, \\gamma)$:
                    - **State Space $\\mathcal{S} \\in \\mathbb{R}^{18}$:** 15 Technical features (Returns, Volatility, Moving Average Ratios, RSI, MACD, Bollinger Bands, ATR, Volume) + 3 Account metrics (Cash ratio, Holdings ratio, Current Drawdown).
                    - **Action Space $\\mathcal{A} = \\{0: \\text{HOLD}, 1: \\text{BUY}, 2: \\text{SELL}\\}$**.
                    - **Risk-Penalized Reward Function:**
                    $$R_t = r_{\\text{portfolio}, t} - \\lambda \\cdot \\Omega_t - c \\cdot |\\Delta w_t|$$
                    Where:
                    - $r_{\\text{portfolio}, t} = \\frac{V_t - V_{t-1}}{V_{t-1}}$
                    - $\\Omega_t = w_{\\text{down}} \\cdot (\\min(0, r_t))^2 + w_{\\text{dd}} \\cdot \\text{Drawdown}_t^2$
                    - $\\lambda$: Low Risk ($\lambda = 1.8$), Medium Risk ($\lambda = 0.8$), High Risk ($\lambda = 0.2$), Standard PPO ($\lambda = 0.0$).
                    - $c \\cdot |\\Delta w_t|$: 0.1% transaction fee to eliminate excessive churning.
                    """)

                with st.expander("3. Why Proximal Policy Optimization (PPO)?", expanded=True):
                    st.markdown("""
                    PPO uses a **Clipped Surrogate Objective**:
                    $$L^{\\text{CLIP}}(\\theta) = \\hat{\\mathbb{E}}_t \\left[ \\min\\left(r_t(\\theta)\\hat{A}_t, \\text{clip}(r_t(\\theta), 1-\\epsilon, 1+\\epsilon)\\hat{A}_t\\right) \\right]$$
                    - PPO avoids destructive policy updates in noisy financial time series.
                    - Provides stable, continuous actor-critic policy optimization without requiring second-order Hessian calculations.
                    """)


# ==============================================================================
# MODE B: CURRENT MARKET DECISION SUPPORT (NEW)
# ==============================================================================
else:
    # Mode B: Current Market Decision Support
    if not model_info["exists"]:
        st.warning(
            f"⚠️ **Pre-trained model unavailable for {selected_ticker} ({selected_risk}).** "
            "In Current Market Decision Support mode, the system evaluates real market observations "
            "using only verified trained policies. Please use the sidebar expander to train the agent first."
        )
    else:
        should_run_current = (
            run_clicked
            or st.session_state.current_market_results is None
            or st.session_state.current_market_results.get("ticker") != selected_ticker
            or st.session_state.current_market_results.get("risk_level") != selected_risk
            or st.session_state.current_market_results.get("virtual_capital") != investment_amount
        )
        if should_run_current:
            with st.spinner(f"Retrieving latest verified market observation for {selected_ticker}..."):
                try:
                    agent = load_trained_agent(selected_ticker, selected_risk)
                    if agent is None:
                        st.error(f"Failed to load model file for {selected_ticker} ({selected_risk}).")
                    else:
                        decision_data = evaluate_current_market_decision(
                            ticker=selected_ticker,
                            agent=agent,
                            risk_level=selected_risk,
                            virtual_capital=investment_amount,
                            existing_shares=0,  # Fresh simulation starts with zero shares
                            lookback_years=lookback_years
                        )
                        st.session_state.current_market_results = decision_data
                except Exception as e:
                    st.error(f"Error evaluating current market observation: {e}")
                    logger.exception("Current market evaluation failed")

        cmd_res = st.session_state.current_market_results
        if cmd_res is not None:
            session_inf = cmd_res.get("session_info", {})
            sizing = cmd_res.get("order_sizing", {})
            expl = cmd_res.get("explanation", {})
            probs = cmd_res.get("action_probabilities", {})
            lookback_bars = cmd_res.get("total_lookback_bars", len(cmd_res.get("feature_df", [])))
            
            # Disclaimers & Session Freshness Banner
            st.markdown(f"""
            <div class="disclaimer-box">
                <b>⚠️ Educational Simulation Notice:</b> This is an AI-powered educational decision support tool operating on simulated virtual capital. 
                It does not constitute financial advice, an investment recommendation, or a guaranteed price prediction. No real financial orders are placed.
            </div>
            """, unsafe_allow_html=True)
            
            # Session & Freshness Metadata
            freshness_str = cmd_res.get("freshness_status", "UNKNOWN")
            fresh_badge = "badge-fresh" if "VERIFIED" in freshness_str else "badge-closed"
            session_date_str = cmd_res.get("latest_session_date", "Latest EOD")
            status_label_str = session_inf.get("status_label", "Market Closed")
            timestamp_ist_str = session_inf.get("current_timestamp_ist", datetime.now().strftime("%Y-%m-%d %H:%M:%S IST"))
            data_source_str = cmd_res.get("data_source", "Yahoo Finance API")
            
            st.markdown(f"""
            <div class="info-callout">
                <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
                    <div>
                        <b>Trading Calendar Status:</b> {status_label_str}<br>
                        <b>Verified Market Session Date:</b> {session_date_str} &bull; <b>Current IST:</b> {timestamp_ist_str}<br>
                        <b>Data Source:</b> {data_source_str} ({lookback_bars} bars historical lookback)
                    </div>
                    <div style="margin-top: 6px;">
                        <span class="{fresh_badge}">{freshness_str}</span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Top KPI Grid
            c1, c2, c3, c4, c5, c6 = st.columns(6)
            
            with c1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Latest Verified Price</div>
                    <div class="metric-value">{format_inr(cmd_res['latest_price'])}</div>
                    <div class="hero-subtitle">Session: {cmd_res['latest_session_date']}</div>
                </div>
                """, unsafe_allow_html=True)
                
            with c2:
                raw_act = cmd_res["raw_action_name"]
                b_c = "signal-buy" if raw_act == "BUY" else ("signal-sell" if raw_act == "SELL" else "signal-hold")
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Raw Model Action</div>
                    <div style="margin-top: 4px;"><span class="{b_c}">{raw_act}</span></div>
                    <div class="hero-subtitle" style="margin-top: 6px;">Conf: {cmd_res['confidence']*100:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)
                
            with c3:
                exec_act = sizing["executable_action"]
                b_exec = "signal-buy" if exec_act == "BUY" else ("signal-sell" if exec_act == "SELL" else "signal-hold")
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Permitted Action</div>
                    <div style="margin-top: 4px;"><span class="{b_exec}">{exec_act}</span></div>
                    <div class="hero-subtitle" style="margin-top: 6px;">{sizing['status']}</div>
                </div>
                """, unsafe_allow_html=True)
                
            with c4:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Virtual Capital</div>
                    <div class="metric-value">{format_inr(cmd_res['virtual_capital'])}</div>
                    <div class="hero-subtitle">Allocated Starting Cash</div>
                </div>
                """, unsafe_allow_html=True)
                
            with c5:
                shares_str = f"{sizing['shares_traded']} whole shares"
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Affordable Shares</div>
                    <div class="metric-value" style="color: #00E5FF;">{sizing['shares_traded']}</div>
                    <div class="hero-subtitle">Outlay: {format_inr(sizing['total_outlay'])}</div>
                </div>
                """, unsafe_allow_html=True)
                
            with c6:
                r_score = cmd_res["risk_score"]
                r_col = "#00E676" if r_score < 35 else ("#FFD600" if r_score < 65 else "#FF1744")
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Dynamic Risk Score</div>
                    <div class="metric-value" style="color: {r_col};">{r_score:.1f}</div>
                    <div class="hero-subtitle">Scale 0 (Low) - 100 (High)</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Sub-sections
            col_left, col_right = st.columns([1.5, 1.0])
            
            with col_left:
                st.markdown("#### 🧠 AI Decision & Constraint Validation")
                st.markdown(f"""
                <div class="info-callout">
                    <div style="font-size: 15px; font-weight: 700; color: #00E5FF; margin-bottom: 6px;">
                        Model Action: {raw_act} &bull; Executable Action: {exec_act} ({sizing['status']})
                    </div>
                    <div style="color: #E6EDF3; margin-bottom: 8px;">
                        <b>Execution Rationale:</b> {sizing['reason']}
                    </div>
                    <ul style="margin: 0; padding-left: 20px; color: #8B949E; font-size: 13px; line-height: 1.5;">
                        {''.join([f"<li>{r}</li>" for r in expl['primary_reasons']])}
                    </ul>
                </div>
                """, unsafe_allow_html=True)
                
                # Softmax Policy Distribution Chart
                if probs:
                    fig_dist = plot_policy_distribution(probs)
                    st.plotly_chart(fig_dist, use_container_width=True)

            with col_right:
                st.markdown("#### 💼 Virtual Portfolio Execution")
                st.markdown(f"""
                <div class="metric-card">
                    <table style="width: 100%; font-size: 13px; color: #E6EDF3;">
                        <tr style="border-bottom: 1px solid #21262D; padding: 6px 0;">
                            <td style="color: #8B949E;">Starting Virtual Funds:</td>
                            <td style="text-align: right; font-weight: 600;">{format_inr(cmd_res['virtual_capital'])}</td>
                        </tr>
                        <tr style="border-bottom: 1px solid #21262D; padding: 6px 0;">
                            <td style="color: #8B949E;">Current Share Price:</td>
                            <td style="text-align: right; font-weight: 600;">{format_inr(cmd_res['latest_price'])}</td>
                        </tr>
                        <tr style="border-bottom: 1px solid #21262D; padding: 6px 0;">
                            <td style="color: #8B949E;">Affordable Whole Shares:</td>
                            <td style="text-align: right; font-weight: 600;">{sizing['shares_traded']} shares</td>
                        </tr>
                        <tr style="border-bottom: 1px solid #21262D; padding: 6px 0;">
                            <td style="color: #8B949E;">Gross Capital Deployed:</td>
                            <td style="text-align: right; font-weight: 600;">{format_inr(sizing['gross_trade_value'])}</td>
                        </tr>
                        <tr style="border-bottom: 1px solid #21262D; padding: 6px 0;">
                            <td style="color: #8B949E;">Estimated 0.1% Fee:</td>
                            <td style="text-align: right; font-weight: 600; color: #FF9100;">{format_inr(sizing['transaction_fee'])}</td>
                        </tr>
                        <tr style="border-bottom: 1px solid #21262D; padding: 6px 0;">
                            <td style="color: #8B949E;">Virtual Cash Remaining:</td>
                            <td style="text-align: right; font-weight: 600;">{format_inr(sizing['remaining_cash'])}</td>
                        </tr>
                        <tr>
                            <td style="color: #8B949E; padding-top: 6px;">Total Virtual Portfolio:</td>
                            <td style="text-align: right; font-weight: 700; padding-top: 6px; color: #00E5FF;">{format_inr(sizing['portfolio_value'])}</td>
                        </tr>
                    </table>
                    <div style="font-size: 11px; color: #8B949E; margin-top: 10px; font-style: italic;">
                        *For a new investor starting today, historical backtest profit is ₹0.00. Current equity is marked to market.
                    </div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("#### 📊 Technical Indicator State (Observation Inputs)")
            tcols = st.columns(5)
            for i, (k, v) in enumerate(expl["indicators_summary"].items()):
                tcols[i].metric(label=k, value=v)

            # Historical Lookback Price Chart
            st.markdown(f"#### 📈 Recent {lookback_years}-Year Historical Context")
            fdf = cmd_res["feature_df"]
            fig_hist = plot_price_and_trades(
                df=fdf,
                history_df=pd.DataFrame(),
                ticker=cmd_res["ticker"],
                strategy_name=f"Lookback Context ({lookback_years}Y)"
            )
            st.plotly_chart(fig_hist, use_container_width=True)
