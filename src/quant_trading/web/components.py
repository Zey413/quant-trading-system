"""可复用 UI 组件

提供 metric_card / stock_mini_chart / performance_table / trade_table 等。
"""

from __future__ import annotations

import pandas as pd
import streamlit as st


# ---------------------------------------------------------------------------
# 指标卡片
# ---------------------------------------------------------------------------

_CARD_COLORS = {
    "blue":   "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
    "green":  "linear-gradient(135deg, #11998e 0%, #38ef7d 100%)",
    "red":    "linear-gradient(135deg, #eb3349 0%, #f45c43 100%)",
    "orange": "linear-gradient(135deg, #f7971e 0%, #ffd200 100%)",
    "purple": "linear-gradient(135deg, #7F00FF 0%, #E100FF 100%)",
    "dark":   "linear-gradient(135deg, #2c3e50 0%, #4ca1af 100%)",
}


def metric_card(
    label: str,
    value: str,
    delta: str = "",
    delta_positive: bool | None = None,
    color: str = "blue",
) -> None:
    """渲染一个渐变背景的指标卡片

    Parameters
    ----------
    label : str  标签名
    value : str  主要数值
    delta : str  变动值
    delta_positive : bool | None  True=红色(涨), False=绿色(跌)
    color : str  预设颜色名
    """
    bg = _CARD_COLORS.get(color, _CARD_COLORS["blue"])
    delta_html = ""
    if delta:
        css_class = ""
        if delta_positive is True:
            css_class = "delta-up"
        elif delta_positive is False:
            css_class = "delta-down"
        delta_html = f"<div class='delta {css_class}'>{delta}</div>"

    st.markdown(f"""
    <div class='metric-card' style='background:{bg};'>
        <div class='value'>{value}</div>
        <div class='label'>{label}</div>
        {delta_html}
    </div>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# 绩效指标面板
# ---------------------------------------------------------------------------

def performance_table(metrics: dict[str, str | float]) -> None:
    """以两列表格形式展示绩效指标

    Parameters
    ----------
    metrics : dict
        {指标名: 值} 字典
    """
    rows: list[dict] = []
    for k, v in metrics.items():
        if isinstance(v, float):
            if abs(v) < 10:
                display = f"{v:.4f}"
            else:
                display = f"{v:,.2f}"
        else:
            display = str(v)
        rows.append({"指标": k, "值": display})

    df = pd.DataFrame(rows)
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        height=min(40 + len(rows) * 35, 600),
    )


# ---------------------------------------------------------------------------
# 交易记录表格
# ---------------------------------------------------------------------------

def trade_table(trades: pd.DataFrame) -> None:
    """渲染交易记录表格，带涨跌颜色

    Parameters
    ----------
    trades : DataFrame
        至少包含: 日期, 代码, 方向, 数量, 价格, 盈亏
    """
    if trades.empty:
        st.info("暂无交易记录")
        return

    # 格式化
    display_df = trades.copy()
    if "盈亏" in display_df.columns:
        display_df["盈亏"] = display_df["盈亏"].apply(
            lambda x: f"{x:+,.2f}" if isinstance(x, (int, float)) else x
        )

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        height=min(40 + len(display_df) * 35, 500),
    )


# ---------------------------------------------------------------------------
# 涨跌幅标签
# ---------------------------------------------------------------------------

def change_label(pct: float) -> str:
    """返回带颜色的涨跌幅 HTML 文本（A股: 红涨绿跌）"""
    if pct > 0:
        return f"<span class='price-up'>+{pct:.2f}%</span>"
    elif pct < 0:
        return f"<span class='price-down'>{pct:.2f}%</span>"
    return f"{pct:.2f}%"


# ---------------------------------------------------------------------------
# 小型迷你图（sparkline 风格）
# ---------------------------------------------------------------------------

def sparkline(values: list[float], width: int = 120, height: int = 30) -> str:
    """生成 SVG sparkline HTML（用于表格内联展示）"""
    if not values or len(values) < 2:
        return ""
    mn, mx = min(values), max(values)
    rng = mx - mn if mx != mn else 1
    n = len(values)
    points = []
    for i, v in enumerate(values):
        x = i / (n - 1) * width
        y = height - (v - mn) / rng * height
        points.append(f"{x:.1f},{y:.1f}")

    color = "#ff4444" if values[-1] >= values[0] else "#00c851"
    return (
        f"<svg width='{width}' height='{height}'>"
        f"<polyline points='{' '.join(points)}' "
        f"fill='none' stroke='{color}' stroke-width='1.5'/>"
        f"</svg>"
    )


# ---------------------------------------------------------------------------
# 通用空状态提示
# ---------------------------------------------------------------------------

def empty_state(message: str = "暂无数据", icon: str = "📭") -> None:
    """渲染空状态占位图"""
    st.markdown(
        f"<div style='text-align:center;padding:60px 0;color:#888;'>"
        f"<div style='font-size:48px;'>{icon}</div>"
        f"<div style='font-size:16px;margin-top:12px;'>{message}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )
