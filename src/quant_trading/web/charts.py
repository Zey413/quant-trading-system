"""Plotly 图表工厂 - 提供各类金融图表

包含: K线图、净值曲线、回撤图、月度收益热力图、相关性矩阵、饼图等。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ---------------------------------------------------------------------------
# K 线图（支持 MA / MACD / RSI 叠加）
# ---------------------------------------------------------------------------

def candlestick_chart(
    df: pd.DataFrame,
    title: str = "K线图",
    ma_periods: list[int] | None = None,
    show_volume: bool = True,
    show_macd: bool = False,
    show_rsi: bool = False,
) -> go.Figure:
    """绘制交互式 K 线图

    Parameters
    ----------
    df : DataFrame
        至少包含 open/high/low/close/volume 列，index 为日期。
    ma_periods : list[int], optional
        需要叠加的均线周期列表，如 [5, 10, 20, 60]。
    show_volume : bool
        是否显示成交量副图。
    show_macd : bool
        是否显示 MACD 副图。
    show_rsi : bool
        是否显示 RSI 副图。
    """
    # 确定子图数量
    rows = 1
    row_heights = [0.6]
    if show_volume:
        rows += 1
        row_heights.append(0.15)
    if show_macd:
        rows += 1
        row_heights.append(0.15)
    if show_rsi:
        rows += 1
        row_heights.append(0.1)

    fig = make_subplots(
        rows=rows, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=row_heights,
    )

    # ---- K 线主图 ----
    colors = ["#ff4444" if c >= o else "#00c851"
              for o, c in zip(df["open"], df["close"])]

    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            increasing_line_color="#ff4444",
            decreasing_line_color="#00c851",
            increasing_fillcolor="#ff4444",
            decreasing_fillcolor="#00c851",
            name="K线",
        ),
        row=1, col=1,
    )

    # ---- 均线 ----
    ma_colors = ["#FFD700", "#FF6347", "#4169E1", "#32CD32", "#FF69B4"]
    if ma_periods:
        for i, period in enumerate(ma_periods):
            ma = df["close"].rolling(period).mean()
            color = ma_colors[i % len(ma_colors)]
            fig.add_trace(
                go.Scatter(
                    x=df.index, y=ma,
                    mode="lines",
                    name=f"MA{period}",
                    line=dict(width=1.2, color=color),
                ),
                row=1, col=1,
            )

    current_row = 2

    # ---- 成交量 ----
    if show_volume:
        fig.add_trace(
            go.Bar(
                x=df.index,
                y=df["volume"],
                marker_color=colors,
                name="成交量",
                showlegend=False,
            ),
            row=current_row, col=1,
        )
        fig.update_yaxes(title_text="成交量", row=current_row, col=1)
        current_row += 1

    # ---- MACD ----
    if show_macd:
        ema12 = df["close"].ewm(span=12, adjust=False).mean()
        ema26 = df["close"].ewm(span=26, adjust=False).mean()
        dif = ema12 - ema26
        dea = dif.ewm(span=9, adjust=False).mean()
        macd_bar = (dif - dea) * 2

        macd_colors = ["#ff4444" if v >= 0 else "#00c851" for v in macd_bar]
        fig.add_trace(
            go.Bar(x=df.index, y=macd_bar, marker_color=macd_colors,
                   name="MACD柱", showlegend=False),
            row=current_row, col=1,
        )
        fig.add_trace(
            go.Scatter(x=df.index, y=dif, mode="lines",
                       name="DIF", line=dict(width=1, color="#FFD700")),
            row=current_row, col=1,
        )
        fig.add_trace(
            go.Scatter(x=df.index, y=dea, mode="lines",
                       name="DEA", line=dict(width=1, color="#4169E1")),
            row=current_row, col=1,
        )
        fig.update_yaxes(title_text="MACD", row=current_row, col=1)
        current_row += 1

    # ---- RSI ----
    if show_rsi:
        delta = df["close"].diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)
        avg_gain = gain.rolling(14).mean()
        avg_loss = loss.rolling(14).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - 100 / (1 + rs)

        fig.add_trace(
            go.Scatter(x=df.index, y=rsi, mode="lines",
                       name="RSI(14)", line=dict(width=1.2, color="#FF6347")),
            row=current_row, col=1,
        )
        fig.add_hline(y=70, line_dash="dash", line_color="gray",
                      annotation_text="超买", row=current_row, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="gray",
                      annotation_text="超卖", row=current_row, col=1)
        fig.update_yaxes(title_text="RSI", range=[0, 100],
                         row=current_row, col=1)

    # ---- 布局 ----
    fig.update_layout(
        title=title,
        height=200 + rows * 180,
        xaxis_rangeslider_visible=False,
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=60, r=30, t=60, b=30),
    )

    return fig


# ---------------------------------------------------------------------------
# 净值曲线
# ---------------------------------------------------------------------------

def equity_curve(
    equity: pd.Series,
    benchmark: pd.Series | None = None,
    title: str = "组合净值曲线",
) -> go.Figure:
    """绘制净值曲线（可含基准对比）"""
    fig = go.Figure()

    # 归一化到 1
    norm_equity = equity / equity.iloc[0]
    fig.add_trace(
        go.Scatter(
            x=norm_equity.index, y=norm_equity.values,
            mode="lines", name="策略净值",
            line=dict(width=2, color="#4169E1"),
            fill="tozeroy", fillcolor="rgba(65,105,225,0.1)",
        )
    )

    if benchmark is not None and not benchmark.empty:
        norm_bm = benchmark / benchmark.iloc[0]
        fig.add_trace(
            go.Scatter(
                x=norm_bm.index, y=norm_bm.values,
                mode="lines", name="基准(沪深300)",
                line=dict(width=1.5, color="#FFD700", dash="dash"),
            )
        )

    fig.update_layout(
        title=title,
        yaxis_title="净值",
        height=400,
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=60, r=30, t=60, b=30),
    )
    return fig


# ---------------------------------------------------------------------------
# 回撤图
# ---------------------------------------------------------------------------

def drawdown_chart(
    drawdown_series: pd.Series,
    title: str = "回撤曲线",
) -> go.Figure:
    """绘制回撤曲线（负值区域填充）"""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=drawdown_series.index,
            y=-drawdown_series.values * 100,
            mode="lines",
            name="回撤(%)",
            line=dict(width=1.5, color="#ff4444"),
            fill="tozeroy",
            fillcolor="rgba(255,68,68,0.2)",
        )
    )
    fig.update_layout(
        title=title,
        yaxis_title="回撤 (%)",
        height=300,
        template="plotly_white",
        margin=dict(l=60, r=30, t=60, b=30),
    )
    return fig


# ---------------------------------------------------------------------------
# 月度收益热力图
# ---------------------------------------------------------------------------

def monthly_returns_heatmap(
    equity: pd.Series,
    title: str = "月度收益率 (%)",
) -> go.Figure:
    """基于净值曲线计算月度收益并绘制热力图"""
    # 计算日收益率，再按月汇总
    daily_ret = equity.pct_change().dropna()
    daily_ret.index = pd.to_datetime(daily_ret.index)

    monthly = daily_ret.groupby([daily_ret.index.year, daily_ret.index.month]).apply(
        lambda x: (1 + x).prod() - 1
    )
    monthly.index.names = ["year", "month"]
    monthly = monthly.reset_index()
    monthly.columns = ["year", "month", "return"]
    pivot = monthly.pivot(index="year", columns="month", values="return")

    # 补全 1-12 月列
    for m in range(1, 13):
        if m not in pivot.columns:
            pivot[m] = np.nan
    pivot = pivot[list(range(1, 13))]

    month_labels = [f"{m}月" for m in range(1, 13)]

    fig = go.Figure(
        go.Heatmap(
            z=pivot.values * 100,
            x=month_labels,
            y=[str(y) for y in pivot.index],
            colorscale=[
                [0, "#00c851"],
                [0.5, "#ffffff"],
                [1, "#ff4444"],
            ],
            zmid=0,
            text=np.where(
                np.isnan(pivot.values),
                "",
                np.round(pivot.values * 100, 2).astype(str),
            ),
            texttemplate="%{text}%",
            textfont=dict(size=11),
            colorbar=dict(title="%"),
        )
    )

    fig.update_layout(
        title=title,
        height=60 + len(pivot) * 50,
        template="plotly_white",
        margin=dict(l=60, r=30, t=60, b=30),
    )
    return fig


# ---------------------------------------------------------------------------
# 相关性矩阵热力图
# ---------------------------------------------------------------------------

def correlation_matrix(
    corr: pd.DataFrame,
    title: str = "资产相关性矩阵",
) -> go.Figure:
    """绘制相关性矩阵热力图"""
    fig = go.Figure(
        go.Heatmap(
            z=corr.values,
            x=corr.columns.tolist(),
            y=corr.index.tolist(),
            colorscale="RdBu_r",
            zmid=0,
            zmin=-1, zmax=1,
            text=np.round(corr.values, 2).astype(str),
            texttemplate="%{text}",
            textfont=dict(size=12),
            colorbar=dict(title="相关系数"),
        )
    )
    fig.update_layout(
        title=title,
        height=100 + len(corr) * 60,
        template="plotly_white",
        margin=dict(l=80, r=30, t=60, b=80),
    )
    return fig


# ---------------------------------------------------------------------------
# 饼图
# ---------------------------------------------------------------------------

def pie_chart(
    labels: list[str],
    values: list[float],
    title: str = "持仓分布",
) -> go.Figure:
    """绘制饼图（用于持仓分布等）"""
    colors = [
        "#4169E1", "#FF6347", "#FFD700", "#32CD32", "#FF69B4",
        "#00CED1", "#FF8C00", "#8A2BE2", "#DC143C", "#20B2AA",
    ]
    fig = go.Figure(
        go.Pie(
            labels=labels,
            values=values,
            hole=0.4,
            marker=dict(colors=colors[:len(labels)]),
            textinfo="label+percent",
            textfont_size=12,
        )
    )
    fig.update_layout(
        title=title,
        height=400,
        template="plotly_white",
        margin=dict(l=30, r=30, t=60, b=30),
    )
    return fig
