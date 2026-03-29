"""行情中心 - 股票搜索、K线图、技术指标、实时报价、自选股、多股票对比

优先尝试使用真实数据源（DataManager），失败时自动回退到模拟数据。
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import streamlit as st

from quant_trading.web.charts import candlestick_chart, multi_stock_comparison
from quant_trading.web.components import change_label, metric_card, STOCK_POOL, info_box


# ---------------------------------------------------------------------------
# 真实数据获取
# ---------------------------------------------------------------------------

def _try_fetch_kline(code: str, start_date: str, end_date: str) -> pd.DataFrame | None:
    """尝试从真实数据源获取 K 线数据"""
    try:
        from quant_trading.core.config import AppConfig
        from quant_trading.data import DataManager
        config = AppConfig()
        dm = DataManager(config)
        df = dm.fetch_daily(code, start_date, end_date)
        if df.empty:
            return None
        # 设置 date 为索引
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")
        return df
    except Exception:
        return None


# ---------------------------------------------------------------------------
# 模拟数据
# ---------------------------------------------------------------------------

def _generate_kline(code: str, days: int = 250) -> pd.DataFrame:
    """为指定股票生成模拟 K 线数据"""
    np.random.seed(hash(code) % 2**31)
    dates = pd.date_range(end=datetime.now().date(), periods=days, freq="B")

    base_prices = {
        "600519": 1700.0, "000858": 155.0, "601318": 50.0,
        "600036": 36.0, "000001": 12.0, "300750": 210.0,
        "002594": 260.0, "601012": 28.0, "600900": 25.0,
        "000333": 62.0, "600276": 45.0, "002475": 35.0,
        "601888": 85.0, "600809": 220.0, "002714": 42.0,
    }
    base = base_prices.get(code, 50.0)

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


def _get_kline_data(code: str, days: int = 250, use_real: bool = False) -> tuple[pd.DataFrame, bool]:
    """获取 K 线数据，返回 (df, is_real)"""
    if use_real:
        end = datetime.now().strftime("%Y-%m-%d")
        start = (datetime.now() - timedelta(days=int(days * 1.5))).strftime("%Y-%m-%d")
        real_df = _try_fetch_kline(code, start, end)
        if real_df is not None and len(real_df) >= 10:
            return real_df, True
    return _generate_kline(code, days), False


def _mock_realtime_quote(code: str) -> dict:
    """模拟实时行情报价"""
    np.random.seed(hash(code + "rt") % 2**31)
    info = STOCK_POOL.get(code, {"name": code})
    stock_name = info if isinstance(info, str) else info.get("name", code)

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

    # 简化的行业映射
    industry_map = {
        "000001": "银行", "600519": "白酒", "000858": "白酒",
        "601318": "保险", "600036": "银行", "300750": "新能源",
        "002594": "汽车", "601012": "光伏", "600900": "电力",
        "000333": "家电", "600276": "医药", "002475": "电子",
        "601888": "零售", "600809": "白酒", "002714": "农业",
    }
    pe_map = {
        "000001": 5.2, "600519": 28.3, "000858": 18.6,
        "601318": 8.9, "600036": 6.1, "300750": 22.5,
        "002594": 25.1, "601012": 12.3, "600900": 20.8,
        "000333": 12.5, "600276": 45.2, "002475": 30.1,
        "601888": 35.6, "600809": 32.4, "002714": 15.3,
    }
    pb_map = {
        "000001": 0.56, "600519": 9.8, "000858": 4.5,
        "601318": 1.1, "600036": 0.95, "300750": 5.3,
        "002594": 4.8, "601012": 2.1, "600900": 3.9,
        "000333": 3.2, "600276": 8.1, "002475": 6.5,
        "601888": 7.2, "600809": 10.2, "002714": 3.8,
    }

    return {
        "代码": code,
        "名称": stock_name,
        "行业": industry_map.get(code, "未知"),
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
        "市盈率": pe_map.get(code, 15.0),
        "市净率": pb_map.get(code, 2.0),
        "总市值": f"{np.random.uniform(500, 25000):.0f}亿",
        "流通市值": f"{np.random.uniform(400, 20000):.0f}亿",
    }


# ---------------------------------------------------------------------------
# 渲染入口
# ---------------------------------------------------------------------------

def render() -> None:
    st.title("📊 行情中心")

    tab_single, tab_compare, tab_watchlist = st.tabs([
        "📈 个股行情", "📊 多股对比", "⭐ 自选股",
    ])

    # ==================================================================
    # 个股行情
    # ==================================================================
    with tab_single:
        _render_single_stock()

    # ==================================================================
    # 多股对比
    # ==================================================================
    with tab_compare:
        _render_multi_comparison()

    # ==================================================================
    # 自选股
    # ==================================================================
    with tab_watchlist:
        _render_watchlist()


def _render_single_stock() -> None:
    """个股行情页面"""
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
            k: v for k, v in STOCK_POOL.items()
            if search_input in k or search_input in v
        }
        if matches:
            options = [f"{k} - {v}" for k, v in matches.items()]
            selected = st.selectbox("搜索结果", options)
            if selected:
                selected_code = selected.split(" - ")[0]
            else:
                selected_code = "600519"
        else:
            st.warning("未找到匹配的股票")
            selected_code = "600519"
    else:
        selected_code = st.selectbox(
            "选择股票",
            list(STOCK_POOL.keys()),
            format_func=lambda x: f"{x} - {STOCK_POOL[x]}",
        )

    # ====== 加入自选 + 时间范围 ======
    col_add, col_period, col_data = st.columns([1, 2, 1])
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
    with col_data:
        use_real_data = st.checkbox("真实数据", value=False,
                                     help="尝试从数据源获取真实K线数据")

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
    kline_data, is_real = _get_kline_data(selected_code, days=period_days, use_real=use_real_data)

    if is_real:
        st.caption("📡 数据来源: 真实数据")
    else:
        st.caption("📊 数据来源: 模拟数据")

    stock_name = STOCK_POOL.get(selected_code, selected_code)

    fig = candlestick_chart(
        kline_data,
        title=f"{stock_name} ({selected_code}) K线图",
        ma_periods=ma_options if ma_options else None,
        show_volume="成交量" in indicators,
        show_macd="MACD" in indicators,
        show_rsi="RSI" in indicators,
    )
    st.plotly_chart(fig, use_container_width=True)

    # ====== 价格统计 ======
    with st.expander("📊 区间价格统计", expanded=False):
        sc1, sc2, sc3, sc4, sc5, sc6 = st.columns(6)
        with sc1:
            st.metric("区间最高", f"¥{kline_data['high'].max():.2f}")
        with sc2:
            st.metric("区间最低", f"¥{kline_data['low'].min():.2f}")
        with sc3:
            st.metric("区间均价", f"¥{kline_data['close'].mean():.2f}")
        with sc4:
            ret = (kline_data['close'].iloc[-1] / kline_data['close'].iloc[0] - 1) * 100
            st.metric("区间涨幅", f"{ret:.2f}%")
        with sc5:
            vol = kline_data['close'].pct_change().std() * np.sqrt(244) * 100
            st.metric("年化波动率", f"{vol:.1f}%")
        with sc6:
            st.metric("日均成交量", f"{kline_data['volume'].mean():,.0f}")


def _render_multi_comparison() -> None:
    """多股票对比页面"""
    st.subheader("📊 多股票走势对比")

    from quant_trading.web.components import stock_multi_selector

    selected_stocks = stock_multi_selector(
        key="compare_stocks",
        label="选择对比股票（最多5只）",
        default=["600519", "000858", "601318"],
        max_selections=5,
    )

    if len(selected_stocks) < 2:
        st.info("请至少选择 2 只股票进行对比")
        return

    period_options = {"近1月": 22, "近3月": 66, "近6月": 132, "近1年": 250}
    period = st.select_slider(
        "时间范围",
        options=list(period_options.keys()),
        value="近6月",
        key="compare_period",
    )
    days = period_options[period]

    # 获取各股票数据
    data_dict: dict[str, pd.DataFrame] = {}
    for code in selected_stocks:
        name = STOCK_POOL.get(code, code)
        df, _ = _get_kline_data(code, days=days)
        data_dict[f"{name}({code})"] = df

    # 绘制对比图
    fig = multi_stock_comparison(data_dict, title="")
    st.plotly_chart(fig, use_container_width=True)

    # 统计对比表
    st.subheader("📋 区间统计对比")
    compare_rows = []
    for code in selected_stocks:
        name = STOCK_POOL.get(code, code)
        df, _ = _get_kline_data(code, days=days)
        ret = (df["close"].iloc[-1] / df["close"].iloc[0] - 1) * 100
        vol = df["close"].pct_change().std() * np.sqrt(244) * 100
        max_dd = _calc_max_drawdown(df["close"])
        compare_rows.append({
            "股票": f"{name} ({code})",
            "区间涨幅(%)": round(ret, 2),
            "年化波动率(%)": round(vol, 1),
            "最大回撤(%)": round(max_dd * 100, 2),
            "最新收盘价": round(df["close"].iloc[-1], 2),
        })

    compare_df = pd.DataFrame(compare_rows)
    st.dataframe(
        compare_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "区间涨幅(%)": st.column_config.NumberColumn("区间涨幅(%)", format="%.2f%%"),
            "年化波动率(%)": st.column_config.NumberColumn("年化波动率(%)", format="%.1f%%"),
            "最大回撤(%)": st.column_config.NumberColumn("最大回撤(%)", format="%.2f%%"),
        },
    )

    # 相关性分析
    st.subheader("🔗 收益率相关性")
    from quant_trading.web.charts import correlation_matrix

    returns_dict = {}
    for code in selected_stocks:
        name = STOCK_POOL.get(code, code)
        df, _ = _get_kline_data(code, days=days)
        returns_dict[name] = df["close"].pct_change().dropna()

    returns_df = pd.DataFrame(returns_dict).dropna()
    if len(returns_df) > 5:
        corr = returns_df.corr()
        fig_corr = correlation_matrix(corr, title="")
        st.plotly_chart(fig_corr, use_container_width=True)


def _calc_max_drawdown(prices: pd.Series) -> float:
    """计算最大回撤"""
    running_max = prices.cummax()
    drawdown = (running_max - prices) / running_max
    return float(drawdown.max())


def _render_watchlist() -> None:
    """自选股列表"""
    st.subheader("⭐ 自选股管理")

    if not st.session_state.watchlist:
        st.info("自选股列表为空，请在「个股行情」中搜索并添加股票")
        return

    watchlist_data = []
    for code in st.session_state.watchlist:
        q = _mock_realtime_quote(code)
        watchlist_data.append({
            "代码": code,
            "名称": STOCK_POOL.get(code, code),
            "最新价": q["最新价"],
            "涨跌幅(%)": q["涨跌幅"],
            "成交额": q["成交额"],
            "市盈率": q["市盈率"],
            "行业": q["行业"],
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

    # 管理功能
    st.markdown("---")
    col_add, col_remove = st.columns(2)

    with col_add:
        st.markdown("**添加自选股**")
        available = [c for c in STOCK_POOL.keys() if c not in st.session_state.watchlist]
        if available:
            add_code = st.selectbox(
                "选择添加",
                available,
                format_func=lambda x: f"{x} {STOCK_POOL.get(x, '')}",
                key="watch_add",
            )
            if st.button("➕ 添加", key="btn_add_watch"):
                st.session_state.watchlist.append(add_code)
                st.rerun()
        else:
            st.info("所有股票已在自选列表中")

    with col_remove:
        st.markdown("**移除自选股**")
        remove_code = st.selectbox(
            "选择移除",
            st.session_state.watchlist,
            format_func=lambda x: f"{x} {STOCK_POOL.get(x, '')}",
            key="watch_remove",
        )
        if st.button("❌ 移除", key="btn_remove_watch"):
            st.session_state.watchlist.remove(remove_code)
            st.rerun()
