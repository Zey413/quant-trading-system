"""Plotly 图表工厂 - 提供各类金融图表

包含: K线图、净值曲线、回撤图、月度收益热力图、相关性矩阵、饼图、
     信号叠加图、多股票对比图、权重柱状图、收益分布直方图等。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ---------------------------------------------------------------------------
# A 股配色
# ---------------------------------------------------------------------------
COLOR_UP = "#ff4444"       # 红涨
COLOR_DOWN = "#00c851"     # 绿跌
COLOR_STRATEGY = "#4169E1" # 策略蓝
COLOR_BENCHMARK = "#FFD700" # 基准金
MA_COLORS = ["#FFD700", "#FF6347", "#4169E1", "#32CD32", "#FF69B4",
             "#00CED1", "#FF8C00", "#8A2BE2"]
PALETTE = [
    "#4169E1", "#FF6347", "#FFD700", "#32CD32", "#FF69B4",
    "#00CED1", "#FF8C00", "#8A2BE2", "#DC143C", "#20B2AA",
]


# ---------------------------------------------------------------------------
# K 线图（支持 MA / MACD / RSI / 信号叠加）
# ---------------------------------------------------------------------------

def candlestick_chart(
    df: pd.DataFrame,
    title: str = "K线图",
    ma_periods: list[int] | None = None,
    show_volume: bool = True,
    show_macd: bool = False,
    show_rsi: bool = False,
    signals: pd.DataFrame | None = None,
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
    signals : DataFrame, optional
        含有 signal 列的 DataFrame (值为 "buy"/"sell"/"hold")，
        索引与 df 对齐，将在 K 线上叠加买卖点标记。
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
    colors = [COLOR_UP if c >= o else COLOR_DOWN
              for o, c in zip(df["open"], df["close"])]

    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            increasing_line_color=COLOR_UP,
            decreasing_line_color=COLOR_DOWN,
            increasing_fillcolor=COLOR_UP,
            decreasing_fillcolor=COLOR_DOWN,
            name="K线",
        ),
        row=1, col=1,
    )

    # ---- 均线 ----
    if ma_periods:
        for i, period in enumerate(ma_periods):
            ma = df["close"].rolling(period).mean()
            color = MA_COLORS[i % len(MA_COLORS)]
            fig.add_trace(
                go.Scatter(
                    x=df.index, y=ma,
                    mode="lines",
                    name=f"MA{period}",
                    line=dict(width=1.2, color=color),
                ),
                row=1, col=1,
            )

    # ---- 买卖信号叠加 ----
    if signals is not None and "signal" in signals.columns:
        buys = signals[signals["signal"] == "buy"]
        sells = signals[signals["signal"] == "sell"]

        if not buys.empty:
            buy_idx = buys.index.intersection(df.index)
            if len(buy_idx) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=buy_idx,
                        y=df.loc[buy_idx, "low"] * 0.98,
                        mode="markers",
                        marker=dict(symbol="triangle-up", size=12,
                                    color=COLOR_UP, line=dict(width=1, color="white")),
                        name="买入信号",
                    ),
                    row=1, col=1,
                )

        if not sells.empty:
            sell_idx = sells.index.intersection(df.index)
            if len(sell_idx) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=sell_idx,
                        y=df.loc[sell_idx, "high"] * 1.02,
                        mode="markers",
                        marker=dict(symbol="triangle-down", size=12,
                                    color=COLOR_DOWN, line=dict(width=1, color="white")),
                        name="卖出信号",
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

        macd_colors = [COLOR_UP if v >= 0 else COLOR_DOWN for v in macd_bar]
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
            line=dict(width=2, color=COLOR_STRATEGY),
            fill="tozeroy", fillcolor="rgba(65,105,225,0.1)",
        )
    )

    if benchmark is not None and not benchmark.empty:
        norm_bm = benchmark / benchmark.iloc[0]
        fig.add_trace(
            go.Scatter(
                x=norm_bm.index, y=norm_bm.values,
                mode="lines", name="基准(沪深300)",
                line=dict(width=1.5, color=COLOR_BENCHMARK, dash="dash"),
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
# 多净值曲线对比 (用于多策略 / 多股票)
# ---------------------------------------------------------------------------

def multi_equity_curves(
    curves: dict[str, pd.Series],
    title: str = "净值对比",
    normalize: bool = True,
) -> go.Figure:
    """绘制多条净值曲线对比

    Parameters
    ----------
    curves : dict[str, pd.Series]
        {名称: 净值曲线}
    normalize : bool
        是否归一化到 1
    """
    fig = go.Figure()
    for i, (name, curve) in enumerate(curves.items()):
        if curve.empty:
            continue
        y = curve / curve.iloc[0] if normalize else curve
        fig.add_trace(go.Scatter(
            x=y.index, y=y.values,
            mode="lines", name=name,
            line=dict(width=2, color=PALETTE[i % len(PALETTE)]),
        ))

    fig.update_layout(
        title=title,
        yaxis_title="净值" if normalize else "价值",
        height=450,
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
            line=dict(width=1.5, color=COLOR_UP),
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
                [0, COLOR_DOWN],
                [0.5, "#ffffff"],
                [1, COLOR_UP],
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
    fig = go.Figure(
        go.Pie(
            labels=labels,
            values=values,
            hole=0.4,
            marker=dict(colors=PALETTE[:len(labels)]),
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


# ---------------------------------------------------------------------------
# 权重柱状图
# ---------------------------------------------------------------------------

def weights_bar_chart(
    labels: list[str],
    weights: list[float],
    title: str = "组合权重分布",
) -> go.Figure:
    """绘制水平柱状图展示组合权重"""
    sorted_pairs = sorted(zip(labels, weights), key=lambda x: x[1])
    sorted_labels = [p[0] for p in sorted_pairs]
    sorted_weights = [p[1] * 100 for p in sorted_pairs]

    colors = [PALETTE[i % len(PALETTE)] for i in range(len(sorted_labels))]

    fig = go.Figure(go.Bar(
        x=sorted_weights,
        y=sorted_labels,
        orientation="h",
        marker_color=colors,
        text=[f"{w:.1f}%" for w in sorted_weights],
        textposition="auto",
    ))
    fig.update_layout(
        title=title,
        xaxis_title="权重 (%)",
        height=max(300, 40 * len(labels)),
        template="plotly_white",
        margin=dict(l=100, r=30, t=60, b=30),
    )
    return fig


# ---------------------------------------------------------------------------
# 收益分布直方图
# ---------------------------------------------------------------------------

def returns_distribution(
    returns: pd.Series,
    title: str = "日收益率分布",
    bins: int = 50,
) -> go.Figure:
    """绘制收益率分布直方图 + 正态分布叠加"""
    fig = go.Figure()

    fig.add_trace(go.Histogram(
        x=returns.values * 100,
        nbinsx=bins,
        name="收益率分布",
        marker_color=COLOR_STRATEGY,
        opacity=0.75,
    ))

    # 叠加正态分布曲线
    mu = returns.mean() * 100
    sigma = returns.std() * 100
    x_range = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 200)
    y_norm = (
        1 / (sigma * np.sqrt(2 * np.pi)) *
        np.exp(-0.5 * ((x_range - mu) / sigma) ** 2)
    )
    # 缩放到匹配直方图高度
    n = len(returns)
    bin_width = (returns.max() - returns.min()) * 100 / bins
    y_scaled = y_norm * n * bin_width

    fig.add_trace(go.Scatter(
        x=x_range, y=y_scaled,
        mode="lines", name="正态分布",
        line=dict(color=COLOR_UP, width=2, dash="dash"),
    ))

    # 零线
    fig.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.5)

    fig.update_layout(
        title=title,
        xaxis_title="日收益率 (%)",
        yaxis_title="频次",
        height=350,
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=60, r=30, t=60, b=30),
    )
    return fig


# ---------------------------------------------------------------------------
# 滚动指标图 (夏普 / 波动率 / 回撤)
# ---------------------------------------------------------------------------

def rolling_metrics_chart(
    equity: pd.Series,
    window: int = 60,
    title: str = "滚动指标",
) -> go.Figure:
    """绘制滚动夏普比率与滚动波动率"""
    daily_ret = equity.pct_change().dropna()

    # 滚动年化波动率
    roll_vol = daily_ret.rolling(window).std() * np.sqrt(244) * 100

    # 滚动年化收益率
    roll_ret = daily_ret.rolling(window).mean() * 244

    # 滚动夏普比率 (假设无风险利率 2.5%)
    rf = 0.025
    roll_sharpe = (roll_ret - rf) / (daily_ret.rolling(window).std() * np.sqrt(244))

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.5, 0.5],
        subplot_titles=[f"{window}日滚动夏普比率", f"{window}日滚动波动率(%)"],
    )

    fig.add_trace(
        go.Scatter(x=roll_sharpe.index, y=roll_sharpe.values,
                   mode="lines", name="滚动夏普",
                   line=dict(width=1.5, color=COLOR_STRATEGY)),
        row=1, col=1,
    )
    fig.add_hline(y=0, line_dash="dash", line_color="gray", row=1, col=1)

    fig.add_trace(
        go.Scatter(x=roll_vol.index, y=roll_vol.values,
                   mode="lines", name="滚动波动率",
                   line=dict(width=1.5, color="#FF6347")),
        row=2, col=1,
    )

    fig.update_layout(
        title=title,
        height=450,
        template="plotly_white",
        showlegend=False,
        margin=dict(l=60, r=30, t=60, b=30),
    )
    return fig


# ---------------------------------------------------------------------------
# 多股票收盘价对比 (归一化)
# ---------------------------------------------------------------------------

def multi_stock_comparison(
    data_dict: dict[str, pd.DataFrame],
    title: str = "多股票走势对比",
) -> go.Figure:
    """绘制多只股票的归一化收盘价走势对比

    Parameters
    ----------
    data_dict : dict[str, DataFrame]
        {股票名称: K线DataFrame (含 close 列)}
    """
    fig = go.Figure()
    for i, (name, df) in enumerate(data_dict.items()):
        if df.empty or "close" not in df.columns:
            continue
        norm = df["close"] / df["close"].iloc[0] * 100
        fig.add_trace(go.Scatter(
            x=df.index, y=norm.values,
            mode="lines", name=name,
            line=dict(width=2, color=PALETTE[i % len(PALETTE)]),
        ))

    fig.add_hline(y=100, line_dash="dash", line_color="gray", opacity=0.5,
                  annotation_text="基准 100")
    fig.update_layout(
        title=title,
        yaxis_title="归一化价格 (基准=100)",
        height=450,
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=60, r=30, t=60, b=30),
    )
    return fig
