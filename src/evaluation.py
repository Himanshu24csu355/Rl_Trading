"""Evaluation and Comparative Strategy Visualization Module.
Produces interactive Plotly charts and comparative performance tables.
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
            "Total Trades": res["total_trades"]
        })
    return pd.DataFrame(records)


def plot_price_and_trades(
    df: pd.DataFrame,
    history_df: pd.DataFrame,
    ticker: str,
    strategy_name: str = "Risk-Sensitive PPO"
) -> go.Figure:
    """Creates a dual-panel interactive chart:
    Panel 1: Stock Close Price with SMA 10, SMA 30, Bollinger Bands, and AI BUY/SELL markers.
    Panel 2: Daily trading volume.
    """
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.75, 0.25],
        subplot_titles=[f"{ticker} Market Price & AI Trade Executions ({strategy_name})", "Volume"]
    )
    
    dates = df["Date"] if "Date" in df.columns else np.arange(len(df))
    
    # 1. Close Price Line
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=df["Close"],
            mode="lines",
            name="Close Price",
            line=dict(color="#00E5FF", width=1.8)
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
                opacity=0.7
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
                opacity=0.7
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
                line=dict(color="rgba(180, 180, 180, 0.3)", width=0.8),
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
                line=dict(color="rgba(180, 180, 180, 0.3)", width=0.8),
                fill="tonexty",
                fillcolor="rgba(255, 255, 255, 0.03)",
                showlegend=False
            ),
            row=1, col=1
        )

    # 4. Buy & Sell Markers from Strategy History
    if not history_df.empty and "trade_type" in history_df.columns:
        buys = history_df[history_df["trade_type"] == "BUY"]
        sells = history_df[history_df["trade_type"] == "SELL"]
        
        if not buys.empty:
            fig.add_trace(
                go.Scatter(
                    x=buys["date"],
                    y=buys["price"],
                    mode="markers+text",
                    name="AI BUY",
                    marker=dict(
                        symbol="triangle-up",
                        size=12,
                        color="#00E676",
                        line=dict(color="#FFFFFF", width=1)
                    ),
                    text=["BUY" for _ in range(len(buys))],
                    textposition="bottom center",
                    textfont=dict(color="#00E676", size=10)
                ),
                row=1, col=1
            )
            
        if not sells.empty:
            fig.add_trace(
                go.Scatter(
                    x=sells["date"],
                    y=sells["price"],
                    mode="markers+text",
                    name="AI SELL",
                    marker=dict(
                        symbol="triangle-down",
                        size=12,
                        color="#FF1744",
                        line=dict(color="#FFFFFF", width=1)
                    ),
                    text=["SELL" for _ in range(len(sells))],
                    textposition="top center",
                    textfont=dict(color="#FF1744", size=10)
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
                marker=dict(color="rgba(0, 229, 255, 0.4)")
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
                line=dict(width=2.2, color=color)
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
                opacity=0.3
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
