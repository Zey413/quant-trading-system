"""首页概览 - Dashboard

展示大盘指数、持仓概览、组合净值曲线、最近交易、市场涨跌统计。
所有数据使用模拟数据以确保可直接运行。
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import streamlit as st

from quant_trading.web.charts import equity_curve, pie_chart
from quant_trading.web.components import (
    change_label,
    empty_state,
    metric_card,
    sparkline,
    trade_table,
)


# ---------------------------------------------------------------------------
# 模拟数据生成
# ---------------------------------------------------------------------------

def _mock_index_data() -> list[dict]:
    """模拟大盘指数数据"""
    np.random.seed(42)
    indices = [
        {"名称": "上证指数", "代码": "000001.SH", "基准": 3250.0},
        {"名称": "深证成指", "代码": "399001.SZ", "基准": 10800.0},
        {"名称": "创业板指", "代码": "399006.SZ", "基准": 2150.0},
        {"名称": "沪深300",  "代码": "000300.SH", "基准": 3850.0},
        {"名称": "中证500",  "代码": "000905.SH", "基准": 5600.0},
        {"名称": "科创50",   "代码": "000688.SH", "基准": 980.0},
    ]
    result = []
    for idx in indices:
        pct = np.random.uniform(-2.5, 2.5)
        price = idx["基准"] * (1 + pct / 100)
        chg = price - idx["基准"]
        # 最近 30 天走势
        hist = np.cumsum(np.random.randn(30) * 0.3) + idx["基准"]
        result.append({
            "名称": idx["名称"],
            "代码": idx["代码"],
            "最新价": round(price, 2),
            "涨跌": round(chg, 2),
            "涨跌幅": round(pct, 2),
            "走势": hist.tolist(),
        })
    return result


def _mock_portfolio_summary() -> dict:
    """模拟组合摘要"""
    return {
        "总资产": 1_256_800.0,
        "持仓市值": 856_300.0,
        "可用资金": 400_500.0,
        "今日盈亏": 12_350.0,
        "今日收益率": 0.99,
        "总收益率": 25.68,
        "持仓数量": 5,
        "浮动盈亏": 156_800.0,
    }


def _mock_equity_curve() -> tuple[pd.Series, pd.Series]:
    """模拟组合净值曲线和基准"""
    np.random.seed(123)
    days = 365
    dates = pd.date_range(end=datetime.now().date(), periods=days, freq="B")
    # 策略净值
    returns = np.random.normal(0.0005, 0.012, days)
    equity = 1_000_000 * np.cumprod(1 + returns)
    strategy = pd.Series(equity, index=dates)
    # 基准净值
    bm_returns = np.random.normal(0.0002, 0.011, days)
    bm_equity = 1_000_000 * np.cumprod(1 + bm_returns)
    benchmark = pd.Series(bm_equity, index=dates)
    return strategy, benchmark


def _mock_positions() -> pd.DataFrame:
    """模拟持仓明细"""
    data = [
        {"代码": "600519", "名称": "贵州茅台", "持仓": 100, "成本": 1680.0, "现价": 1756.0, "市值": 175600.0, "盈亏": 7600.0, "盈亏比": 4.52},
        {"代码": "000858", "名称": "五粮液", "持仓": 500, "成本": 152.0, "现价": 168.5, "市值": 84250.0, "盈亏": 8250.0, "盈亏比": 10.86},
        {"代码": "601318", "名称": "中国平安", "持仓": 1000, "成本": 48.5, "现价": 52.3, "市值": 52300.0, "盈亏": 3800.0, "盈亏比": 7.84},
        {"代码": "600036", "名称": "招商银行", "持仓": 800, "成本": 35.2, "现价": 37.8, "市值": 30240.0, "盈亏": 2080.0, "盈亏比": 7.39},
        {"代码": "000001", "名称": "平安银行", "持仓": 2000, "成本": 11.5, "现价": 12.1, "市值": 24200.0, "盈亏": 1200.0, "盈亏比": 5.22},
    ]
    return pd.DataFrame(data)


def _mock_recent_trades() -> pd.DataFrame:
    """模拟最近交易"""
    today = datetime.now().date()
    trades = [
        {"日期": today - timedelta(days=1), "代码": "600519", "名称": "贵州茅台", "方向": "买入", "数量": 100, "价格": 1680.0, "金额": 168000.0, "盈亏": 0},
        {"日期": today - timedelta(days=2), "代码": "000858", "名称": "五粮液", "方向": "卖出", "数量": 300, "价格": 170.2, "金额": 51060.0, "盈亏": 2460.0},
        {"日期": today - timedelta(days=3), "代码": "601318", "名称": "中国平安", "方向": "买入", "数量": 500, "价格": 47.8, "金额": 23900.0, "盈亏": 0},
        {"日期": today - timedelta(days=5), "代码": "300750", "名称": "宁德时代", "方向": "卖出", "数量": 200, "价格": 215.6, "金额": 43120.0, "盈亏": -1880.0},
        {"日期": today - timedelta(days=6), "代码": "600036", "名称": "招商银行", "方向": "买入", "数量": 800, "价格": 35.2, "金额": 28160.0, "盈亏": 0},
        {"日期": today - timedelta(days=7), "代码": "002594", "名称": "比亚迪", "方向": "卖出", "数量": 100, "价格": 268.5, "金额": 26850.0, "盈亏": 3650.0},
    ]
    return pd.DataFrame(trades)


def _mock_market_stats() -> dict:
    """模拟市场涨跌统计"""
    np.random.seed(99)
    total = 5200
    up = np.random.randint(1800, 3200)
    down = total - up - np.random.randint(100, 300)
    flat = total - up - down
    return {
        "total": total,
        "up": up,
        "down": down,
        "flat": flat,
        "limit_up": np.random.randint(30, 80),
        "limit_down": np.random.randint(5, 30),
        "avg_change": round(np.random.uniform(-1.5, 1.5), 2),
    }


# ---------------------------------------------------------------------------
# 渲染入口
# ---------------------------------------------------------------------------

def render() -> None:
    st.title("🏠 首页概览")
    st.caption(f"数据更新时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}（模拟数据）")

    # ====== 大盘指数 ======
    st.subheader("📌 大盘指数")
    indices = _mock_index_data()
    cols = st.columns(len(indices))
    for col, idx in zip(cols, indices):
        with col:
            delta_str = f"{idx['涨跌']:+.2f}  ({idx['涨跌幅']:+.2f}%)"
            is_up = idx["涨跌幅"] > 0
            metric_card(
                label=idx["名称"],
                value=f"{idx['最新价']:,.2f}",
                delta=delta_str,
                delta_positive=is_up if idx["涨跌幅"] != 0 else None,
                color="red" if is_up else "green",
            )

    st.markdown("")

    # ====== 组合概览 ======
    st.subheader("💰 组合概览")
    summary = _mock_portfolio_summary()

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("总资产", f"¥{summary['总资产']:,.0f}", color="purple")
    with c2:
        metric_card("持仓市值", f"¥{summary['持仓市值']:,.0f}", color="blue")
    with c3:
        delta_str = f"{summary['今日盈亏']:+,.0f} ({summary['今日收益率']:+.2f}%)"
        metric_card(
            "今日盈亏", f"¥{summary['今日盈亏']:+,.0f}",
            delta=delta_str,
            delta_positive=summary["今日盈亏"] > 0,
            color="red" if summary["今日盈亏"] > 0 else "green",
        )
    with c4:
        metric_card(
            "总收益率", f"{summary['总收益率']:+.2f}%",
            delta=f"浮盈 ¥{summary['浮动盈亏']:+,.0f}",
            delta_positive=summary["浮动盈亏"] > 0,
            color="dark",
        )

    st.markdown("")

    # ====== 净值曲线 + 持仓分布 ======
    col_left, col_right = st.columns([3, 1])

    with col_left:
        st.subheader("📈 组合净值曲线")
        strategy_eq, benchmark_eq = _mock_equity_curve()
        fig = equity_curve(strategy_eq, benchmark_eq, title="")
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.subheader("🥧 持仓分布")
        positions = _mock_positions()
        fig_pie = pie_chart(
            labels=positions["名称"].tolist(),
            values=positions["市值"].tolist(),
            title="",
        )
        fig_pie.update_layout(height=400, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig_pie, use_container_width=True)

    # ====== 市场涨跌统计 ======
    st.subheader("📊 市场涨跌统计")
    stats = _mock_market_stats()
    s1, s2, s3, s4, s5, s6 = st.columns(6)
    with s1:
        st.metric("上涨", f"{stats['up']}", f"{stats['up']/stats['total']*100:.1f}%")
    with s2:
        st.metric("下跌", f"{stats['down']}", f"-{stats['down']/stats['total']*100:.1f}%")
    with s3:
        st.metric("平盘", f"{stats['flat']}")
    with s4:
        st.metric("涨停", f"{stats['limit_up']}")
    with s5:
        st.metric("跌停", f"{stats['limit_down']}")
    with s6:
        st.metric("市场均涨幅", f"{stats['avg_change']:+.2f}%")

    # ====== 持仓明细 ======
    st.subheader("📋 持仓明细")
    st.dataframe(
        positions,
        use_container_width=True,
        hide_index=True,
        column_config={
            "盈亏比": st.column_config.NumberColumn("盈亏比(%)", format="%.2f%%"),
            "市值": st.column_config.NumberColumn("市值", format="¥%,.0f"),
            "盈亏": st.column_config.NumberColumn("盈亏", format="¥%+,.0f"),
        },
    )

    # ====== 最近交易 ======
    st.subheader("📝 最近交易记录")
    trades = _mock_recent_trades()
    trade_table(trades)
