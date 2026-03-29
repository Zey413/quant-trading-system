"""可复用 UI 组件

提供 metric_card / stock_mini_chart / performance_table / trade_table /
strategy_selector / stock_multi_selector / risk_gauge 等。
"""

from __future__ import annotations

from typing import Any

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
# 绩效指标双列卡片面板
# ---------------------------------------------------------------------------

def performance_cards(metrics: dict[str, str | float], cols: int = 4) -> None:
    """以卡片式网格展示绩效指标

    Parameters
    ----------
    metrics : dict
        {指标名: 值}
    cols : int
        每行列数
    """
    items = list(metrics.items())
    for i in range(0, len(items), cols):
        row_items = items[i:i + cols]
        columns = st.columns(cols)
        for j, (k, v) in enumerate(row_items):
            with columns[j]:
                if isinstance(v, float):
                    st.metric(k, f"{v:.4f}")
                else:
                    st.metric(k, str(v))


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


# ---------------------------------------------------------------------------
# 策略选择器（从 StrategyRegistry 读取可用策略）
# ---------------------------------------------------------------------------

def strategy_selector(
    key: str = "strategy_select",
    label: str = "选择策略",
) -> tuple[str, dict]:
    """从策略注册表获取策略列表并显示选择器

    Returns
    -------
    (strategy_key, params_meta) : 选中的策略键和参数元数据
    """
    # 策略参数定义（供动态表单使用）
    STRATEGY_PARAMS: dict[str, dict] = {
        "ma_crossover": {
            "display_name": "均线交叉策略",
            "description": "短期均线上穿长期均线(金叉)买入，下穿(死叉)卖出",
            "params": {
                "short_window": {"label": "短期均线周期", "type": "int", "default": 5, "min": 2, "max": 60},
                "long_window": {"label": "长期均线周期", "type": "int", "default": 20, "min": 5, "max": 250},
            },
        },
        "rsi": {
            "display_name": "RSI超买超卖策略",
            "description": "RSI从超卖区上穿买入，从超买区下穿卖出",
            "params": {
                "period": {"label": "RSI周期", "type": "int", "default": 14, "min": 2, "max": 50},
                "oversold": {"label": "超卖阈值", "type": "float", "default": 30.0, "min": 5.0, "max": 50.0},
                "overbought": {"label": "超买阈值", "type": "float", "default": 70.0, "min": 50.0, "max": 95.0},
            },
        },
        "macd": {
            "display_name": "MACD策略",
            "description": "MACD金叉买入，死叉卖出",
            "params": {
                "fast_period": {"label": "快线周期", "type": "int", "default": 12, "min": 2, "max": 50},
                "slow_period": {"label": "慢线周期", "type": "int", "default": 26, "min": 5, "max": 100},
                "signal_period": {"label": "信号线周期", "type": "int", "default": 9, "min": 2, "max": 30},
            },
        },
        "bollinger": {
            "display_name": "布林带策略",
            "description": "价格触及下轨买入，触及上轨卖出",
            "params": {
                "period": {"label": "布林带周期", "type": "int", "default": 20, "min": 5, "max": 100},
                "num_std": {"label": "标准差倍数", "type": "float", "default": 2.0, "min": 0.5, "max": 4.0},
            },
        },
        "dual_thrust": {
            "display_name": "Dual Thrust策略",
            "description": "基于前N日价格区间的突破策略",
            "params": {
                "lookback": {"label": "回溯周期", "type": "int", "default": 5, "min": 2, "max": 30},
                "k1": {"label": "上轨系数", "type": "float", "default": 0.5, "min": 0.1, "max": 1.5},
                "k2": {"label": "下轨系数", "type": "float", "default": 0.5, "min": 0.1, "max": 1.5},
            },
        },
        "mean_reversion": {
            "display_name": "均值回归策略",
            "description": "价格偏离均线过远时反向交易",
            "params": {
                "window": {"label": "均线窗口", "type": "int", "default": 20, "min": 5, "max": 120},
                "entry_std": {"label": "入场标准差", "type": "float", "default": 2.0, "min": 0.5, "max": 4.0},
                "exit_std": {"label": "出场标准差", "type": "float", "default": 0.5, "min": 0.0, "max": 2.0},
            },
        },
        "turtle": {
            "display_name": "海龟交易策略",
            "description": "唐奇安通道突破策略",
            "params": {
                "entry_window": {"label": "入场窗口", "type": "int", "default": 20, "min": 5, "max": 60},
                "exit_window": {"label": "出场窗口", "type": "int", "default": 10, "min": 3, "max": 30},
            },
        },
    }

    strategy_names = list(STRATEGY_PARAMS.keys())
    strategy_key = st.selectbox(
        label,
        strategy_names,
        format_func=lambda x: STRATEGY_PARAMS[x]["display_name"],
        key=key,
    )
    return strategy_key, STRATEGY_PARAMS[strategy_key]


# ---------------------------------------------------------------------------
# 动态参数表单
# ---------------------------------------------------------------------------

def param_form(
    params_def: dict[str, dict],
    prefix: str = "param",
) -> dict[str, Any]:
    """根据参数定义动态生成输入表单

    Parameters
    ----------
    params_def : dict
        {参数名: {label, type, default, min, max}}
    prefix : str
        Streamlit key 前缀

    Returns
    -------
    dict : 用户输入的参数值
    """
    user_params: dict[str, Any] = {}
    for pname, pdef in params_def.items():
        if pdef["type"] == "int":
            user_params[pname] = st.number_input(
                pdef["label"],
                min_value=pdef["min"],
                max_value=pdef["max"],
                value=pdef["default"],
                step=1,
                key=f"{prefix}_{pname}",
            )
        elif pdef["type"] == "float":
            user_params[pname] = st.number_input(
                pdef["label"],
                min_value=float(pdef["min"]),
                max_value=float(pdef["max"]),
                value=float(pdef["default"]),
                step=0.1,
                key=f"{prefix}_{pname}",
            )
    return user_params


# ---------------------------------------------------------------------------
# 股票多选器
# ---------------------------------------------------------------------------

# 常用 A 股票池（用于 UI 选择，不依赖网络）
STOCK_POOL: dict[str, str] = {
    "000001": "平安银行",
    "600519": "贵州茅台",
    "000858": "五粮液",
    "601318": "中国平安",
    "600036": "招商银行",
    "300750": "宁德时代",
    "002594": "比亚迪",
    "601012": "隆基绿能",
    "600900": "长江电力",
    "000333": "美的集团",
    "600276": "恒瑞医药",
    "002475": "立讯精密",
    "601888": "中国中免",
    "600809": "山西汾酒",
    "002714": "牧原股份",
}


def stock_multi_selector(
    key: str = "stocks",
    label: str = "选择股票",
    default: list[str] | None = None,
    max_selections: int = 10,
) -> list[str]:
    """多股票选择器

    Returns
    -------
    list[str] : 选中的股票代码列表
    """
    if default is None:
        default = ["600519", "000858", "601318"]

    options = list(STOCK_POOL.keys())
    selected = st.multiselect(
        label,
        options,
        default=[d for d in default if d in options],
        format_func=lambda x: f"{x} {STOCK_POOL.get(x, '')}",
        key=key,
        max_selections=max_selections,
    )
    return selected


# ---------------------------------------------------------------------------
# 风险仪表盘 gauge
# ---------------------------------------------------------------------------

def risk_gauge(
    value: float,
    title: str = "风险水平",
    thresholds: tuple[float, float] = (0.3, 0.7),
) -> None:
    """用进度条 + 颜色展示风险水平

    Parameters
    ----------
    value : float
        0~1 之间的风险值
    title : str
        标题
    thresholds : tuple
        (低风险上限, 中风险上限)
    """
    low, mid = thresholds
    if value <= low:
        color = "green"
        level = "低风险"
    elif value <= mid:
        color = "orange"
        level = "中风险"
    else:
        color = "red"
        level = "高风险"

    color_map = {"green": "#00c851", "orange": "#ff8c00", "red": "#ff4444"}
    bar_color = color_map[color]

    st.markdown(f"**{title}**: {level} ({value:.1%})")
    st.markdown(
        f"<div style='background:#e0e0e0;border-radius:4px;height:12px;'>"
        f"<div style='background:{bar_color};width:{min(value*100,100):.0f}%;"
        f"height:100%;border-radius:4px;'></div></div>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# 信息提示框
# ---------------------------------------------------------------------------

def info_box(title: str, content: str, icon: str = "💡") -> None:
    """带图标的信息提示框"""
    st.markdown(
        f"""<div style='background:#f0f2f6;border-left:4px solid #4169E1;
        padding:12px 16px;border-radius:0 8px 8px 0;margin:8px 0;'>
        <strong>{icon} {title}</strong><br/>
        <span style='color:#555;font-size:14px;'>{content}</span>
        </div>""",
        unsafe_allow_html=True,
    )
