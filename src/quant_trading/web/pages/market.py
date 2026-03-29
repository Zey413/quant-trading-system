"""行情中心 - 股票搜索、K线图、技术指标、实时报价、自选股

全部使用模拟数据，确保可直接运行。
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import streamlit as st

from quant_trading.web.charts import candlestick_chart
from quant_trading.web.components import change_label, metric_card


# ---------------------------------------------------------------------------
# 模拟数据
# ---------------------------------------------------------------------------

_STOCK_POOL: dict[str, dict] = {
    "000001": {"name": "平安银行", "industry": "银行", "pe": 5.2, "pb": 0.56},
    "600519": {"name": "贵州茅台", "industry": "白酒", "pe": 28.3, "pb": 9.8},
    "000858": {"name": "五粮液", "industry": "白酒", "pe": 18.6, "pb": 4.5},
    "601318": {"name": "中国平安", "industry": "保险", "pe": 8.9, "pb": 1.1},
    "600036": {"name": "招商银行", "industry": "银行", "pe": 6.1, "pb": 0.95},
    "300750": {"name": "宁德时代", "industry": "新能源", "pe": 22.5, "pb": 5.3},
    "002594": {"name": "比亚迪", "industry": "汽车", "pe": 25.1, "pb": 4.8},
    "601012": {"name": "隆基绿能", "industry": "光伏", "pe": 12.3, "pb": 2.1},
    "600900": {"name": "长江电力", "industry": "电力", "pe": 20.8, "pb": 3.9},
    "000333": {"name": "美的集团", "industry": "家电", "pe": 12.5, "pb": 3.2},
    "600276": {"name": "恒瑞医药", "industry": "医药", "pe": 45.2, "pb": 8.1},
    "002475": {"name": "立讯精密", "industry": "电子", "pe": 30.1, "pb": 6.5},
    "601888": {"name": "中国中免", "industry": "零售", "pe": 35.6, "pb": 7.2},
    "600809": {"name": "山西汾酒", "industry": "白酒", "pe": 32.4, "pb": 10.2},
    "002714": {"name": "牧原股份", "industry": "农业", "pe": 15.3, "pb": 3.8},
}


def _generate_kline(code: str, days: int = 250) -> pd.DataFrame:
    """为指定股票生成模拟 K 线数据"""
    np.random.seed(hash(code) % 2**31)
    dates = pd.date_range(end=datetime.now().date(), periods=days, freq="B")

    # 根据不同股票设定不同的价格区间
    base_prices = {
        "600519": 1700.0, "000858": 155.0, "601318": 50.0,
        "600036": 36.0, "000001": 12.0, "300750": 210.0,
        "002594": 260.0, "601012": 28.0, "600900": 25.0,
        "000333": 62.0, "600276": 45.0, "002475": 35.0,
        "601888": 85.0, "600809": 220.0, "002714": 42.0,
    }
    base = base_prices.get(code, 50.0)

    # 生成走势
    returns = np.random.normal(0.0003, 0.02, days)
    close = base * np.cumprod(1 + returns)
    high = close * (1 + np.abs(np.random.normal(0, 0.01, days)))
    low = close * (1 - np.abs(np.random.normal(0, 0.01, days)))
    opn = low + (high - low) * np.random.uniform(0.2, 0.8, days)
    volume = np.random.randint(50_000, 500_000, days) * (base / 50)

    df = pd.DataFrame({
        "open": np.round(opn, 2),
        "high": np.round(high, 2),
        "low": np.round(low, 2),
        "close": np.round(close, 2),
        "volume": volume.astype(int),
    }, index=dates)

    return df


def _mock_realtime_quote(code: str) -> dict:
    """模拟实时行情报价"""
    np.random.seed(hash(code + "rt") % 2**31)
    info = _STOCK_POOL.get(code, {"name": code, "industry": "未知", "pe": 0, "pb": 0})
    base_prices = {
        "600519": 1756.0, "000858": 168.5, "601318": 52.3,
        "600036": 37.8, "000001": 12.1, "300750": 215.6,
        "002594": 268.5, "601012": 29.5, "600900": 26.2,
        "000333": 64.5, "600276": 47.8, "002475": 36.2,
        "601888": 88.3, "600809": 225.6, "002714": 43.8,
    }
    price = base_prices.get(code, 50.0)
    pct = np.random.uniform(-3.0, 3.0)
    pre_close = price / (1 + pct / 100)
    chg = price - pre_close

    return {
        "代码": code,
        "名称": info["name"],
        "行业": info["industry"],
        "最新价": round(price, 2),
        "涨跌额": round(chg, 2),
        "涨跌幅": round(pct, 2),
        "今开": round(price * (1 + np.random.uniform(-0.005, 0.005)), 2),
        "最高": round(price * (1 + abs(np.random.normal(0, 0.015))), 2),
        "最低": round(price * (1 - abs(np.random.normal(0, 0.015))), 2),
        "昨收": round(pre_close, 2),
        "成交量": f"{np.random.randint(10, 200):.0f}万手",
        "成交额": f"{np.random.uniform(5, 150):.1f}亿",
        "换手率": f"{np.random.uniform(0.3, 5.0):.2f}%",
        "市盈率": info["pe"],
        "市净率": info["pb"],
        "总市值": f"{np.random.uniform(500, 25000):.0f}亿",
        "流通市值": f"{np.random.uniform(400, 20000):.0f}亿",
    }


# ---------------------------------------------------------------------------
# 渲染入口
# ---------------------------------------------------------------------------

def render() -> None:
    st.title("📊 行情中心")

    # ====== 股票搜索 ======
    col_search, col_btn = st.columns([4, 1])
    with col_search:
        search_input = st.text_input(
            "搜索股票（代码或名称）",
            placeholder="输入股票代码或名称，如 600519 或 贵州茅台",
            label_visibility="collapsed",
        )
    with col_btn:
        st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)

    # 过滤搜索
    if search_input:
        matches = {
            k: v for k, v in _STOCK_POOL.items()
            if search_input in k or search_input in v["name"]
        }
        if matches:
            options = [f"{k} - {v['name']}" for k, v in matches.items()]
            selected = st.selectbox("搜索结果", options)
            if selected:
                selected_code = selected.split(" - ")[0]
        else:
            st.warning("未找到匹配的股票")
            selected_code = "600519"
    else:
        selected_code = st.selectbox(
            "选择股票",
            list(_STOCK_POOL.keys()),
            format_func=lambda x: f"{x} - {_STOCK_POOL[x]['name']}",
        )

    # ====== 加入自选 ======
    col_add, col_period = st.columns([1, 3])
    with col_add:
        if st.button("⭐ 加入自选", use_container_width=True):
            if selected_code not in st.session_state.watchlist:
                st.session_state.watchlist.append(selected_code)
                st.success(f"已添加 {selected_code} 到自选股")
            else:
                st.info("该股票已在自选列表中")
    with col_period:
        period_options = {"近1月": 22, "近3月": 66, "近6月": 132, "近1年": 250, "近2年": 500}
        period_label = st.select_slider(
            "时间范围",
            options=list(period_options.keys()),
            value="近1年",
        )
        period_days = period_options[period_label]

    # ====== 实时报价面板 ======
    st.subheader("📋 实时报价")
    quote = _mock_realtime_quote(selected_code)

    q1, q2, q3, q4 = st.columns(4)
    with q1:
        delta_str = f"{quote['涨跌额']:+.2f} ({quote['涨跌幅']:+.2f}%)"
        is_up = quote["涨跌幅"] > 0
        metric_card(
            label=f"{quote['名称']} ({quote['代码']})",
            value=f"¥{quote['最新价']:.2f}",
            delta=delta_str,
            delta_positive=is_up if quote["涨跌幅"] != 0 else None,
            color="red" if is_up else "green",
        )
    with q2:
        metric_card("今开 / 昨收", f"{quote['今开']} / {quote['昨收']}", color="dark")
    with q3:
        metric_card("最高 / 最低", f"{quote['最高']} / {quote['最低']}", color="blue")
    with q4:
        metric_card("成交量 / 成交额", f"{quote['成交量']} / {quote['成交额']}", color="orange")

    st.markdown("")

    # 基本面指标
    f1, f2, f3, f4, f5 = st.columns(5)
    with f1:
        st.metric("市盈率(PE)", f"{quote['市盈率']:.1f}")
    with f2:
        st.metric("市净率(PB)", f"{quote['市净率']:.2f}")
    with f3:
        st.metric("换手率", quote["换手率"])
    with f4:
        st.metric("总市值", quote["总市值"])
    with f5:
        st.metric("行业", quote["行业"])

    st.markdown("---")

    # ====== K线图 + 技术指标 ======
    st.subheader("📈 K线走势")

    # 技术指标选择器
    col_ma, col_ind = st.columns(2)
    with col_ma:
        ma_options = st.multiselect(
            "均线叠加",
            [5, 10, 20, 30, 60, 120, 250],
            default=[5, 20, 60],
        )
    with col_ind:
        indicators = st.multiselect(
            "技术指标",
            ["成交量", "MACD", "RSI"],
            default=["成交量"],
        )

    # 生成 K 线数据
    kline_data = _generate_kline(selected_code, days=period_days)
    stock_name = _STOCK_POOL.get(selected_code, {}).get("name", selected_code)

    fig = candlestick_chart(
        kline_data,
        title=f"{stock_name} ({selected_code}) K线图",
        ma_periods=ma_options if ma_options else None,
        show_volume="成交量" in indicators,
        show_macd="MACD" in indicators,
        show_rsi="RSI" in indicators,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # ====== 自选股列表 ======
    st.subheader("⭐ 自选股列表")

    if not st.session_state.watchlist:
        st.info("自选股列表为空，请在上方搜索并添加股票")
    else:
        watchlist_data = []
        for code in st.session_state.watchlist:
            q = _mock_realtime_quote(code)
            info = _STOCK_POOL.get(code, {"name": code, "industry": "未知"})
            watchlist_data.append({
                "代码": code,
                "名称": info["name"],
                "最新价": q["最新价"],
                "涨跌幅(%)": q["涨跌幅"],
                "成交额": q["成交额"],
                "市盈率": q["市盈率"],
                "行业": info.get("industry", ""),
            })

        wdf = pd.DataFrame(watchlist_data)
        st.dataframe(
            wdf,
            use_container_width=True,
            hide_index=True,
            column_config={
                "涨跌幅(%)": st.column_config.NumberColumn(
                    "涨跌幅(%)",
                    format="%.2f%%",
                ),
            },
        )

        # 删除自选
        remove_code = st.selectbox(
            "移除自选股",
            ["（不移除）"] + st.session_state.watchlist,
            format_func=lambda x: x if x == "（不移除）" else f"{x} - {_STOCK_POOL.get(x, {}).get('name', x)}",
        )
        if remove_code != "（不移除）":
            if st.button("❌ 确认移除"):
                st.session_state.watchlist.remove(remove_code)
                st.rerun()
