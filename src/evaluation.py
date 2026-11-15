"""Evaluation and Comparative Strategy Visualization Module.
Produces interactive Plotly charts, dynamic data-driven empirical commentary,
and policy distribution visualizations.
"""

from typing import Dict, Any, List
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.utils import format_inr, format_pct


def build_comparison_dataframe(strategies_results: List[Dict[str, Any]]) -> pd.DataFrame:
    """Builds a formatted tabular summary comparing multiple trading strategies."""
    records = []
    for res in strategies_results:
        records.append({
            "Strategy": res["strategy_name"],
            "Initial Capital": format_inr(res["initial_capital"]),
            "Final Value": format_inr(res["final_portfolio_value"]),
            "Net Profit/Loss": format_inr(res["total_profit_loss"]),
            "Total Return (%)": f"{res['total_return_pct']:+.2f}%",
            "Max Drawdown (%)": f"{res['max_drawdown_pct']:.2f}%",
            "Sharpe Ratio": f"{res['sharpe_ratio']:.2f}",
            "Sortino Ratio": f"{res['sortino_ratio']:.2f}",
            "Annual Volatility": f"{res['annualized_volatility_pct']:.2f}%",
            "Total Trades": res["total_trades"],
            "Total Fees": format_inr(res.get("total_fees_paid", 0.0))
        })
    return pd.DataFrame(records)


def generate_empirical_findings(strategies_results: List[Dict[str, Any]]) -> str:
    """Generates 100% data-driven, objective empirical findings from actual benchmark results.
    Never hardcodes claims that one strategy always outperforms another.
    """
    if len(strategies_results) < 2:
        return "Insufficient strategy data to generate comparative findings."

    # Index by strategy name keywords
    bh = next((s for s in strategies_results if "Buy & Hold" in s["strategy_name"]), None)
    std = next((s for s in strategies_results if "Standard PPO" in s["strategy_name"]), None)
    rs = next((s for s in strategies_results if "Risk-Sensitive" in s["strategy_name"]), None)

    findings = []

    # 1. Compare Risk-Sensitive vs Standard PPO
    if rs and std:
        dd_diff = rs["max_drawdown_pct"] - std["max_drawdown_pct"]
        ret_diff = rs["total_return_pct"] - std["total_return_pct"]
        vol_diff = rs["annualized_volatility_pct"] - std["annualized_volatility_pct"]
        
        # Drawdown analysis
        if dd_diff < -0.5:
            findings.append(
                f"- **Drawdown Reduction:** Risk-Sensitive PPO successfully constrained downside drawdown "
                f"to **{rs['max_drawdown_pct']:.2f}%** compared to Standard PPO's **{std['max_drawdown_pct']:.2f}%** "
                f"(a {abs(dd_diff):.2f}% lower peak-to-trough decline)."
            )
        elif dd_diff > 0.5:
            findings.append(
                f"- **Drawdown Observation:** In this specific historical test regime, Standard PPO experienced lower maximum drawdown "
                f"(**{std['max_drawdown_pct']:.2f}%** vs **{rs['max_drawdown_pct']:.2f}%** for Risk-Sensitive PPO)."
            )
        else:
            findings.append(
                f"- **Drawdown Parity:** Both Risk-Sensitive PPO ({rs['max_drawdown_pct']:.2f}%) and Standard PPO ({std['max_drawdown_pct']:.2f}%) "
                f"experienced comparable maximum drawdown levels."
            )

        # Return analysis
        if ret_diff > 0:
            findings.append(
                f"- **Cumulative Return:** Risk-Sensitive PPO achieved higher net return ({rs['total_return_pct']:+.2f}% vs {std['total_return_pct']:+.2f}% for Standard PPO)."
            )
        elif ret_diff < 0:
            findings.append(
                f"- **Trade-off in Return:** Standard PPO achieved higher cumulative return ({std['total_return_pct']:+.2f}% vs {rs['total_return_pct']:+.2f}% for Risk-Sensitive PPO), "
                f"reflecting the expected theoretical trade-off where risk aversion sacrifices upside capture to limit downside exposure."
            )

        # Volatility & Sharpe
        if vol_diff < -0.5:
            findings.append(
                f"- **Volatility Dampening:** The risk penalty effectively reduced annualized volatility by {abs(vol_diff):.2f}% "
                f"({rs['annualized_volatility_pct']:.2f}% vs {std['annualized_volatility_pct']:.2f}%)."
            )
            
        if rs["sharpe_ratio"] > std["sharpe_ratio"]:
            findings.append(
                f"- **Risk-Adjusted Efficiency:** Risk-Sensitive PPO achieved a superior Sharpe ratio ({rs['sharpe_ratio']:.2f} vs {std['sharpe_ratio']:.2f}), "
                f"confirming more efficient return generation per unit of total risk."
            )

    # 2. Compare with Buy & Hold
    if rs and bh:
        dd_vs_bh = rs["max_drawdown_pct"] - bh["max_drawdown_pct"]
        if dd_vs_bh < -1.0:
            findings.append(
                f"- **Capital Protection vs Passive Holding:** Risk-Sensitive PPO protected capital significantly better than Buy & Hold "
                f"(Max Drawdown: {rs['max_drawdown_pct']:.2f}% vs {bh['max_drawdown_pct']:.2f}%)."
            )
        elif dd_vs_bh > 1.0:
            findings.append(
                f"- **Benchmark Drawdown:** Passive Buy & Hold maintained lower maximum drawdown ({bh['max_drawdown_pct']:.2f}%) than the active RL agent ({rs['max_drawdown_pct']:.2f}%)."
            )

        # Transaction Friction Impact
        total_fees = rs.get("total_fees_paid", 0.0)
        findings.append(
            f"- **Transaction Friction:** Over {rs['total_trades']} active trades, the RL agent paid {format_inr(total_fees)} in 0.1% fees/slippage. "
            f"Active trading frequency should be balanced against commission drag."
        )

    return "\n\n".join(findings)


def plot_price_and_trades(
    df: pd.DataFrame,
    history_df: pd.DataFrame,
    ticker: str,
    strategy_name: str = "Risk-Sensitive PPO"
) -> go.Figure:
    """Creates a dual-panel interactive chart:
    Panel 1: Stock Close Price with SMA 10, SMA 30, Bollinger Bands, and AI BUY/SELL markers.
    Panel 2: Daily trading volume.
    Avoids label collisions using structured hover templates and distinct marker symbols.
    """
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.75, 0.25],
        subplot_titles=[f"{ticker} Market Price & AI Trade Executions ({strategy_name})", "Trading Volume"]
    )
    
    dates = df["Date"] if "Date" in df.columns else np.arange(len(df))
    
    # 1. Close Price Line
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=df["Close"],
            mode="lines",
            name="Close Price",
            line=dict(color="#00E5FF", width=1.8),
            hovertemplate="<b>Date:</b> %{x|%Y-%m-%d}<br><b>Price:</b> ₹%{y:,.2f}<extra></extra>"
        ),
        row=1, col=1
    )
    
    # 2. Moving Averages
    if "sma_10" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=df["sma_10"],
                mode="lines",
                name="SMA 10",
                line=dict(color="#FFD600", width=1.0, dash="dot"),
                opacity=0.7,
                hoverinfo="skip"
            ),
            row=1, col=1
        )
        
    if "sma_30" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=df["sma_30"],
                mode="lines",
                name="SMA 30",
                line=dict(color="#FF4081", width=1.0, dash="dash"),
                opacity=0.7,
                hoverinfo="skip"
            ),
            row=1, col=1
        )
        
    # 3. Bollinger Bands
    if "bb_upper" in df.columns and "bb_lower" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=df["bb_upper"],
                mode="lines",
                name="BB Upper",
                line=dict(color="rgba(180, 180, 180, 0.25)", width=0.8),
                hoverinfo="skip",
                showlegend=False
            ),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=df["bb_lower"],
                mode="lines",
                name="BB Lower",
                line=dict(color="rgba(180, 180, 180, 0.25)", width=0.8),
                fill="tonexty",
                fillcolor="rgba(255, 255, 255, 0.03)",
                hoverinfo="skip",
                showlegend=False
            ),
            row=1, col=1
        )

    # 4. Clean Buy & Sell Markers (Hover-centric to prevent text label collisions)
    if not history_df.empty and "trade_type" in history_df.columns:
        buys = history_df[history_df["trade_type"] == "BUY"]
        sells = history_df[history_df["trade_type"] == "SELL"]
        
        if not buys.empty:
            custom_data = np.stack((buys["shares_traded"], buys["cash"], buys["portfolio_value"]), axis=-1)
            fig.add_trace(
                go.Scatter(
                    x=buys["date"],
                    y=buys["price"],
                    mode="markers",
                    name="AI BUY",
                    marker=dict(
                        symbol="triangle-up",
                        size=11,
                        color="#00E676",
                        line=dict(color="#FFFFFF", width=1.2)
                    ),
                    customdata=custom_data,
                    hovertemplate="<b>Action: BUY</b><br><b>Date:</b> %{x|%Y-%m-%d}<br><b>Price:</b> ₹%{y:,.2f}<br><b>Shares Bought:</b> %{customdata[0]}<br><b>Cash Remaining:</b> ₹%{customdata[1]:,.2f}<br><b>Portfolio Value:</b> ₹%{customdata[2]:,.2f}<extra></extra>"
                ),
                row=1, col=1
            )
            
        if not sells.empty:
            custom_data = np.stack((sells["shares_traded"], sells["cash"], sells["portfolio_value"]), axis=-1)
            fig.add_trace(
                go.Scatter(
                    x=sells["date"],
                    y=sells["price"],
                    mode="markers",
                    name="AI SELL",
                    marker=dict(
                        symbol="triangle-down",
                        size=11,
                        color="#FF1744",
                        line=dict(color="#FFFFFF", width=1.2)
                    ),
                    customdata=custom_data,
                    hovertemplate="<b>Action: SELL</b><br><b>Date:</b> %{x|%Y-%m-%d}<br><b>Price:</b> ₹%{y:,.2f}<br><b>Shares Sold:</b> %{customdata[0]}<br><b>Cash Remaining:</b> ₹%{customdata[1]:,.2f}<br><b>Portfolio Value:</b> ₹%{customdata[2]:,.2f}<extra></extra>"
                ),
                row=1, col=1
            )

    # 5. Volume Bar Chart
    if "Volume" in df.columns:
        fig.add_trace(
            go.Bar(
                x=dates,
                y=df["Volume"],
                name="Volume",
                marker=dict(color="rgba(0, 229, 255, 0.4)"),
                hovertemplate="<b>Volume:</b> %{y:,.0f}<extra></extra>"
            ),
            row=2, col=1
        )

    # Modern Dark Theme Layout
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0E1117",
        plot_bgcolor="#161B22",
        hovermode="x unified",
        margin=dict(l=40, r=40, t=50, b=40),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(0,0,0,0)"
        ),
        xaxis=dict(showgrid=True, gridcolor="#21262D"),
        yaxis=dict(title="Stock Price (INR)", showgrid=True, gridcolor="#21262D"),
        yaxis2=dict(title="Volume", showgrid=False)
    )
    
    return fig


def plot_equity_curves(strategies_results: List[Dict[str, Any]]) -> go.Figure:
    """Plots comparative portfolio equity growth curves across all evaluated strategies."""
    fig = go.Figure()
    palette = ["#00E5FF", "#FFB300", "#76FF03", "#E040FB", "#FF5252"]
    
    for idx, res in enumerate(strategies_results):
        hdf = res.get("history_df", pd.DataFrame())
        if hdf.empty:
            continue
            
        dates = hdf["date"] if "date" in hdf.columns else np.arange(len(hdf))
        color = palette[idx % len(palette)]
        
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=hdf["portfolio_value"],
                mode="lines",
                name=f"{res['strategy_name']} ({res['total_return_pct']:+.1f}%)",
                line=dict(width=2.2, color=color),
                hovertemplate="<b>%{data.name}</b><br>Value: ₹%{y:,.2f}<extra></extra>"
            )
        )
        
    fig.update_layout(
        title="Comparative Strategy Equity Curves (Portfolio Value over Time)",
        template="plotly_dark",
        paper_bgcolor="#0E1117",
        plot_bgcolor="#161B22",
        hovermode="x unified",
        margin=dict(l=40, r=40, t=60, b=40),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(0,0,0,0)"
        ),
        xaxis=dict(title="Date", showgrid=True, gridcolor="#21262D"),
        yaxis=dict(title="Portfolio Value (INR)", showgrid=True, gridcolor="#21262D")
    )
    return fig


def plot_drawdowns(strategies_results: List[Dict[str, Any]]) -> go.Figure:
    """Plots comparative underwater drawdown curves over time."""
    fig = go.Figure()
    palette = ["#00E5FF", "#FFB300", "#76FF03", "#E040FB"]
    
    for idx, res in enumerate(strategies_results):
        hdf = res.get("history_df", pd.DataFrame())
        if hdf.empty or "drawdown" not in hdf.columns:
            continue
            
        dates = hdf["date"] if "date" in hdf.columns else np.arange(len(hdf))
        color = palette[idx % len(palette)]
        dd_pct = -1.0 * hdf["drawdown"] * 100.0  # Display as negative percentage
        
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=dd_pct,
                mode="lines",
                name=f"{res['strategy_name']} (Max: -{res['max_drawdown_pct']:.1f}%)",
                line=dict(width=1.5, color=color),
                fill="tozeroy",
                opacity=0.3,
                hovertemplate="<b>%{data.name}</b><br>Drawdown: %{y:.2f}%<extra></extra>"
            )
        )
        
    fig.update_layout(
        title="Underwater Drawdown Profile (Capital Downside Risk Exposure)",
        template="plotly_dark",
        paper_bgcolor="#0E1117",
        plot_bgcolor="#161B22",
        hovermode="x unified",
        margin=dict(l=40, r=40, t=60, b=40),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(0,0,0,0)"
        ),
        xaxis=dict(title="Date", showgrid=True, gridcolor="#21262D"),
        yaxis=dict(title="Drawdown (%)", showgrid=True, gridcolor="#21262D")
    )
    return fig


def plot_policy_distribution(action_probs: Dict[str, float]) -> go.Figure:
    """Creates an interactive horizontal bar chart displaying the PPO policy's
    softmax action probability distribution.
    """
    actions = list(action_probs.keys())
    probs = [action_probs[a] * 100.0 for a in actions]
    colors = ["#FFD600" if a == "HOLD" else ("#00E676" if a == "BUY" else "#FF1744") for a in actions]
    
    fig = go.Figure(
        go.Bar(
            x=probs,
            y=actions,
            orientation="h",
            marker=dict(color=colors, line=dict(color="#FFFFFF", width=1)),
            text=[f"{p:.1f}%" for p in probs],
            textposition="auto",
            hovertemplate="<b>%{y}:</b> %{x:.2f}% probability<extra></extra>"
        )
    )
    
    fig.update_layout(
        title="PPO Policy Action Probability Distribution (Softmax Output)",
        template="plotly_dark",
        paper_bgcolor="#0E1117",
        plot_bgcolor="#161B22",
        margin=dict(l=40, r=40, t=40, b=30),
        height=220,
        xaxis=dict(title="Confidence Probability (%)", range=[0, 105], showgrid=True, gridcolor="#21262D"),
        yaxis=dict(showgrid=False)
    )
    return fig
