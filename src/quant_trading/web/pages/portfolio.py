"""组合管理 - 持仓分析、多股票组合回测、权重调整、绩效分析

支持:
1. 持仓概览与分布分析
2. 多股票组合回测（尝试真实 PortfolioBacktester，失败回退模拟）
3. 权重方案对比（等权/自定义/市值加权）
4. 交易记录与数据导出
"""

from __future__ import annotations

import io
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import streamlit as st

from quant_trading.web.charts import (
    equity_curve,
    multi_equity_curves,
    pie_chart,
    weights_bar_chart,
    correlation_matrix,
)
from quant_trading.web.components import (
    metric_card,
    trade_table,
    stock_multi_selector,
    info_box,
    STOCK_POOL,
)


# ---------------------------------------------------------------------------
# 模拟数据
# ---------------------------------------------------------------------------

def _mock_holdings() -> pd.DataFrame:
    """模拟持仓数据"""
    data = [
        {"代码": "600519", "名称": "贵州茅台", "行业": "白酒", "持仓": 100, "成本价": 1680.00, "现价": 1756.00, "市值": 175600.0, "浮动盈亏": 7600.0, "盈亏比(%)": 4.52, "仓位(%)": 13.97},
        {"代码": "000858", "名称": "五粮液", "行业": "白酒", "持仓": 500, "成本价": 152.00, "现价": 168.50, "市值": 84250.0, "浮动盈亏": 8250.0, "盈亏比(%)": 10.86, "仓位(%)": 6.70},
        {"代码": "601318", "名称": "中国平安", "行业": "保险", "持仓": 1000, "成本价": 48.50, "现价": 52.30, "市值": 52300.0, "浮动盈亏": 3800.0, "盈亏比(%)": 7.84, "仓位(%)": 4.16},
        {"代码": "600036", "名称": "招商银行", "行业": "银行", "持仓": 800, "成本价": 35.20, "现价": 37.80, "市值": 30240.0, "浮动盈亏": 2080.0, "盈亏比(%)": 7.39, "仓位(%)": 2.41},
        {"代码": "000001", "名称": "平安银行", "行业": "银行", "持仓": 2000, "成本价": 11.50, "现价": 12.10, "市值": 24200.0, "浮动盈亏": 1200.0, "盈亏比(%)": 5.22, "仓位(%)": 1.93},
        {"代码": "300750", "名称": "宁德时代", "行业": "新能源", "持仓": 200, "成本价": 205.00, "现价": 215.60, "市值": 43120.0, "浮动盈亏": 2120.0, "盈亏比(%)": 5.17, "仓位(%)": 3.43},
        {"代码": "002594", "名称": "比亚迪", "行业": "汽车", "持仓": 300, "成本价": 255.00, "现价": 268.50, "市值": 80550.0, "浮动盈亏": 4050.0, "盈亏比(%)": 5.29, "仓位(%)": 6.41},
        {"代码": "600276", "名称": "恒瑞医药", "行业": "医药", "持仓": 600, "成本价": 44.20, "现价": 47.80, "市值": 28680.0, "浮动盈亏": 2160.0, "盈亏比(%)": 8.14, "仓位(%)": 2.28},
    ]
    return pd.DataFrame(data)


def _mock_portfolio_equity() -> pd.Series:
    """模拟组合净值历史"""
    np.random.seed(456)
    days = 365
    dates = pd.date_range(end=datetime.now().date(), periods=days, freq="B")
    returns = np.random.normal(0.0004, 0.011, days)
    equity = 1_000_000 * np.cumprod(1 + returns)
    return pd.Series(equity, index=dates)


def _mock_trade_history() -> pd.DataFrame:
    """模拟历史交易记录"""
    np.random.seed(789)
    today = datetime.now().date()
    n = 50
    stocks = [
        ("600519", "贵州茅台"), ("000858", "五粮液"), ("601318", "中国平安"),
        ("600036", "招商银行"), ("000001", "平安银行"), ("300750", "宁德时代"),
        ("002594", "比亚迪"), ("600276", "恒瑞医药"),
    ]

    records = []
    for i in range(n):
        idx = np.random.randint(0, len(stocks))
        code, name = stocks[idx]
        side = np.random.choice(["买入", "卖出"])
        qty = np.random.choice([100, 200, 300, 500, 1000])
        price = np.random.uniform(10, 1800)
        amount = qty * price
        commission = max(amount * 0.0003, 5)
        tax = amount * 0.001 if side == "卖出" else 0
        pnl = np.random.normal(1500, 6000) if side == "卖出" else 0
        trade_date = today - timedelta(days=np.random.randint(1, 180))

        records.append({
            "日期": trade_date,
            "代码": code,
            "名称": name,
            "方向": side,
            "数量": qty,
            "价格": round(price, 2),
            "金额": round(amount, 2),
            "佣金": round(commission, 2),
            "印花税": round(tax, 2),
            "盈亏": round(pnl, 2),
        })

    df = pd.DataFrame(records)
    df = df.sort_values("日期", ascending=False).reset_index(drop=True)
    return df


def _simulate_portfolio_backtest(
    stocks: list[str],
    weights: dict[str, float],
    strategy_name: str,
    start_date: str,
    end_date: str,
    initial_capital: float,
) -> dict:
    """模拟多股票组合回测结果"""
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    dates = pd.date_range(start=start, end=end, freq="B")
    n_days = len(dates)

    if n_days < 10:
        return {}

    per_stock = {}
    equity_curves = {}

    for stock in stocks:
        np.random.seed(hash(stock + strategy_name) % 2**31)
        vol = np.random.uniform(0.012, 0.022)
        drift = np.random.uniform(0.0001, 0.0006)
        returns = np.random.normal(drift, vol, n_days)
        stock_eq = initial_capital * weights[stock] * np.cumprod(1 + returns)
        eq_series = pd.Series(stock_eq, index=dates)
        equity_curves[stock] = eq_series

        total_return = (stock_eq[-1] - initial_capital * weights[stock]) / (initial_capital * weights[stock])
        per_stock[stock] = {
            "total_return": total_return,
            "volatility": float(pd.Series(returns).std() * np.sqrt(244)),
        }

    # 组合净值
    combined = pd.Series(0.0, index=dates)
    for stock, eq in equity_curves.items():
        combined += eq

    # 组合绩效
    total_return = (combined.iloc[-1] - initial_capital) / initial_capital
    daily_ret = combined.pct_change().dropna()
    ann_vol = float(daily_ret.std() * np.sqrt(244))
    years = n_days / 244
    ann_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
    sharpe = (ann_return - 0.025) / ann_vol if ann_vol > 0 else 0
    running_max = combined.cummax()
    drawdown = (running_max - combined) / running_max
    max_dd = float(drawdown.max())

    # 相关性
    returns_df = pd.DataFrame({s: eq.pct_change().dropna() for s, eq in equity_curves.items()}).dropna()
    corr = returns_df.corr() if len(returns_df) > 5 else pd.DataFrame()

    return {
        "combined_equity": combined,
        "equity_curves": equity_curves,
        "per_stock": per_stock,
        "drawdown": drawdown,
        "correlation": corr,
        "weights": weights,
        "metrics": {
            "总收益率": f"{total_return:.2%}",
            "年化收益率": f"{ann_return:.2%}",
            "年化波动率": f"{ann_vol:.2%}",
            "最大回撤": f"{max_dd:.2%}",
            "夏普比率": round(sharpe, 4),
        },
        "summary_values": {
            "total_return": total_return,
            "ann_return": ann_return,
            "max_dd": max_dd,
            "sharpe": sharpe,
            "ann_vol": ann_vol,
        },
    }


def _try_real_portfolio_backtest(
    stocks: list[str],
    weights: dict[str, float],
    strategy_name: str,
    start_date: str,
    end_date: str,
    initial_capital: float,
) -> dict | None:
    """尝试使用真实 PortfolioBacktester"""
    try:
        from quant_trading.core.config import AppConfig
        from quant_trading.backtest import PortfolioBacktester
        from quant_trading.strategy import StrategyRegistry
        from quant_trading.data import DataManager

        config = AppConfig()
        config.backtest.initial_capital = initial_capital
        config.backtest.start_date = start_date
        config.backtest.end_date = end_date

        strategy = StrategyRegistry.get(strategy_name)
        dm = DataManager(config)

        data_dict = {}
        for stock in stocks:
            df = dm.fetch_daily(stock, start_date, end_date)
            if not df.empty:
                data_dict[stock] = df

        if len(data_dict) < 2:
            return None

        backtester = PortfolioBacktester(config)
        result = backtester.run(strategy, data_dict, weights=weights)

        # 转换格式
        equity_curves = {}
        per_stock_info = {}
        for symbol, res in result.per_stock.items():
            if not res.equity_curve.empty:
                equity_curves[symbol] = res.equity_curve
                per_stock_info[symbol] = {
                    "total_return": res.total_return,
                    "volatility": res.volatility,
                }

        overall = result.overall
        return {
            "combined_equity": overall.equity_curve,
            "equity_curves": equity_curves,
            "per_stock": per_stock_info,
            "drawdown": overall.drawdown_series,
            "correlation": result.correlation_matrix,
            "weights": result.stock_weights,
            "metrics": {
                "总收益率": f"{overall.total_return:.2%}",
                "年化收益率": f"{overall.annualized_return:.2%}",
                "年化波动率": f"{overall.volatility:.2%}",
                "最大回撤": f"{overall.max_drawdown:.2%}",
                "夏普比率": round(overall.sharpe_ratio, 4),
            },
            "summary_values": {
                "total_return": overall.total_return,
                "ann_return": overall.annualized_return,
                "max_dd": overall.max_drawdown,
                "sharpe": overall.sharpe_ratio,
                "ann_vol": overall.volatility,
            },
            "mode": "real",
        }
    except Exception:
        return None


# ---------------------------------------------------------------------------
# 渲染入口
# ---------------------------------------------------------------------------

def render() -> None:
    st.title("💼 组合管理")

    tab_overview, tab_backtest, tab_history = st.tabs([
        "📊 持仓概览", "🔬 组合回测", "📝 交易记录",
    ])

    # ==================================================================
    # 持仓概览
    # ==================================================================
    with tab_overview:
        _render_overview()

    # ==================================================================
    # 组合回测
    # ==================================================================
    with tab_backtest:
        _render_portfolio_backtest()

    # ==================================================================
    # 交易记录
    # ==================================================================
    with tab_history:
        _render_trade_history()


def _render_overview() -> None:
    """持仓概览"""
    holdings = _mock_holdings()
    total_mv = holdings["市值"].sum()
    total_pnl = holdings["浮动盈亏"].sum()
    cash = 400_500.0
    total_assets = total_mv + cash

    # 概览卡片
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        metric_card("总资产", f"¥{total_assets:,.0f}", color="purple")
    with c2:
        metric_card("持仓市值", f"¥{total_mv:,.0f}", color="blue")
    with c3:
        metric_card("可用资金", f"¥{cash:,.0f}", color="dark")
    with c4:
        metric_card(
            "浮动盈亏", f"¥{total_pnl:+,.0f}",
            delta=f"{total_pnl/total_assets*100:+.2f}%",
            delta_positive=total_pnl > 0,
            color="red" if total_pnl > 0 else "green",
        )
    with c5:
        metric_card("持仓数量", f"{len(holdings)} 只", color="orange")

    st.markdown("")

    # 持仓分布
    st.subheader("📊 持仓分析")
    col_pie1, col_pie2 = st.columns(2)

    with col_pie1:
        fig_stock = pie_chart(
            labels=holdings["名称"].tolist(),
            values=holdings["市值"].tolist(),
            title="个股持仓分布",
        )
        st.plotly_chart(fig_stock, use_container_width=True)

    with col_pie2:
        industry_group = holdings.groupby("行业")["市值"].sum()
        industry_labels = industry_group.index.tolist() + ["现金"]
        industry_values = industry_group.values.tolist() + [cash]
        fig_ind = pie_chart(
            labels=industry_labels,
            values=industry_values,
            title="资产配置分布",
        )
        st.plotly_chart(fig_ind, use_container_width=True)

    # 持仓明细
    st.subheader("📋 持仓明细")
    st.dataframe(
        holdings,
        use_container_width=True,
        hide_index=True,
        column_config={
            "市值": st.column_config.NumberColumn("市值", format="¥%,.0f"),
            "浮动盈亏": st.column_config.NumberColumn("浮动盈亏", format="¥%+,.0f"),
            "成本价": st.column_config.NumberColumn("成本价", format="¥%.2f"),
            "现价": st.column_config.NumberColumn("现价", format="¥%.2f"),
            "盈亏比(%)": st.column_config.NumberColumn("盈亏比(%)", format="%.2f%%"),
            "仓位(%)": st.column_config.NumberColumn("仓位(%)", format="%.2f%%"),
        },
    )

    # 组合净值历史
    st.subheader("📈 组合净值历史")
    equity_series = _mock_portfolio_equity()

    period_filter = st.select_slider(
        "时间范围",
        options=["近1月", "近3月", "近6月", "近1年", "全部"],
        value="全部",
        key="port_period",
    )
    period_map = {"近1月": 22, "近3月": 66, "近6月": 132, "近1年": 250, "全部": len(equity_series)}
    days = period_map[period_filter]
    filtered_eq = equity_series.iloc[-days:]

    fig_eq = equity_curve(filtered_eq, title="")
    st.plotly_chart(fig_eq, use_container_width=True)


def _render_portfolio_backtest() -> None:
    """多股票组合回测"""
    st.subheader("🔬 多股票组合回测")

    info_box(
        "组合回测说明",
        "选择多只股票和策略，设置权重方案，运行组合回测。"
        "系统将对每只股票独立运行策略回测，然后按权重加总净值曲线。",
    )

    col_left, col_right = st.columns([1, 2])

    with col_left:
        # 股票选择
        selected_stocks = stock_multi_selector(
            key="port_bt_stocks",
            label="选择组合股票",
            default=["600519", "000858", "601318", "600036"],
            max_selections=8,
        )

        if len(selected_stocks) < 2:
            st.warning("请至少选择 2 只股票")
            return

        st.markdown("---")

        # 策略选择
        from quant_trading.web.pages.backtest import STRATEGY_PARAMS
        strategy_key = st.selectbox(
            "选择策略",
            list(STRATEGY_PARAMS.keys()),
            format_func=lambda x: STRATEGY_PARAMS[x]["display_name"],
            key="port_bt_strategy",
        )

        st.markdown("---")

        # 权重设置
        weight_method = st.radio(
            "权重方案",
            ["等权重", "自定义权重"],
            key="port_weight_method",
        )

        weights: dict[str, float] = {}
        if weight_method == "等权重":
            n = len(selected_stocks)
            weights = {s: 1.0 / n for s in selected_stocks}
        else:
            st.markdown("**设置各股票权重 (%)**")
            raw_weights = {}
            for stock in selected_stocks:
                name = STOCK_POOL.get(stock, stock)
                raw_weights[stock] = st.slider(
                    f"{name} ({stock})",
                    min_value=0,
                    max_value=100,
                    value=100 // len(selected_stocks),
                    key=f"weight_{stock}",
                )
            total = sum(raw_weights.values())
            if total > 0:
                weights = {s: w / total for s, w in raw_weights.items()}
            else:
                weights = {s: 1.0 / len(selected_stocks) for s in selected_stocks}

        st.markdown("---")

        # 回测参数
        start_date = st.date_input("开始日期", value=pd.Timestamp("2023-01-01"), key="port_start")
        end_date = st.date_input("结束日期", value=pd.Timestamp("2025-12-31"), key="port_end")
        initial_capital = st.number_input(
            "初始资金 (元)",
            min_value=100000.0,
            value=1_000_000.0,
            step=100_000.0,
            format="%.0f",
            key="port_capital",
        )

        use_real = st.checkbox("尝试真实回测引擎", value=False, key="port_real")

        run_btn = st.button("🚀 运行组合回测", type="primary", use_container_width=True)

    with col_right:
        if run_btn:
            with st.spinner("正在运行组合回测..."):
                result = None

                if use_real:
                    result = _try_real_portfolio_backtest(
                        selected_stocks, weights, strategy_key,
                        start_date.strftime("%Y-%m-%d"),
                        end_date.strftime("%Y-%m-%d"),
                        initial_capital,
                    )

                if result is None:
                    result = _simulate_portfolio_backtest(
                        selected_stocks, weights, strategy_key,
                        start_date.strftime("%Y-%m-%d"),
                        end_date.strftime("%Y-%m-%d"),
                        initial_capital,
                    )

            if not result:
                st.error("回测失败，请调整参数")
            else:
                mode = result.get("mode", "simulated")
                if mode == "real":
                    st.success("✅ 使用真实回测引擎完成")
                else:
                    st.info("📊 使用模拟数据完成")

                _render_portfolio_result(result)
        else:
            st.info("👈 配置组合参数后点击「运行组合回测」")


def _render_portfolio_result(result: dict) -> None:
    """渲染组合回测结果"""
    sv = result["summary_values"]

    # 顶部指标
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        metric_card("总收益率", f"{sv['total_return']:.2%}",
                     color="red" if sv["total_return"] > 0 else "green")
    with c2:
        metric_card("年化收益率", f"{sv['ann_return']:.2%}",
                     color="red" if sv["ann_return"] > 0 else "green")
    with c3:
        metric_card("年化波动率", f"{sv['ann_vol']:.2%}", color="blue")
    with c4:
        metric_card("最大回撤", f"{sv['max_dd']:.2%}", color="orange")
    with c5:
        metric_card("夏普比率", f"{sv['sharpe']:.4f}", color="purple")

    st.markdown("")

    # 权重分布
    st.subheader("📊 权重分布")
    wt = result["weights"]
    labels = [f"{STOCK_POOL.get(s, s)}({s})" for s in wt.keys()]
    fig_wt = weights_bar_chart(labels, list(wt.values()), title="")
    st.plotly_chart(fig_wt, use_container_width=True)

    # 组合净值 + 个股净值
    st.subheader("📈 净值曲线")
    curves = {"组合": result["combined_equity"]}
    for stock, eq in result["equity_curves"].items():
        name = STOCK_POOL.get(stock, stock)
        curves[name] = eq

    fig_eq = multi_equity_curves(curves, title="")
    st.plotly_chart(fig_eq, use_container_width=True)

    # 相关性矩阵
    corr = result.get("correlation")
    if corr is not None and not corr.empty:
        st.subheader("🔗 股票相关性矩阵")
        # 用股票名称替换代码
        rename_map = {s: STOCK_POOL.get(s, s) for s in corr.columns}
        corr_display = corr.rename(columns=rename_map, index=rename_map)
        fig_corr = correlation_matrix(corr_display, title="")
        st.plotly_chart(fig_corr, use_container_width=True)

    # 个股绩效
    st.subheader("📋 个股绩效")
    stock_perf = []
    for stock, info in result["per_stock"].items():
        name = STOCK_POOL.get(stock, stock)
        w = result["weights"].get(stock, 0)
        stock_perf.append({
            "股票": f"{name} ({stock})",
            "权重": f"{w:.1%}",
            "总收益率": f"{info['total_return']:.2%}",
            "年化波动率": f"{info['volatility']:.2%}",
        })
    st.dataframe(pd.DataFrame(stock_perf), use_container_width=True, hide_index=True)


def _render_trade_history() -> None:
    """交易记录与导出"""
    holdings = _mock_holdings()
    trade_history = _mock_trade_history()
    equity_series = _mock_portfolio_equity()

    st.subheader("📝 交易记录")

    # 筛选
    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        filter_direction = st.selectbox("方向筛选", ["全部", "买入", "卖出"])
    with col_f2:
        stock_codes = ["全部"] + sorted(trade_history["代码"].unique().tolist())
        filter_stock = st.selectbox("股票筛选", stock_codes)
    with col_f3:
        sort_by = st.selectbox("排序", ["日期", "金额", "盈亏"])

    display_trades = trade_history.copy()
    if filter_direction != "全部":
        display_trades = display_trades[display_trades["方向"] == filter_direction]
    if filter_stock != "全部":
        display_trades = display_trades[display_trades["代码"] == filter_stock]

    sort_col_map = {"日期": "日期", "金额": "金额", "盈亏": "盈亏"}
    display_trades = display_trades.sort_values(
        sort_col_map[sort_by], ascending=False
    ).reset_index(drop=True)

    trade_table(display_trades)

    # 交易统计
    st.subheader("📊 交易统计")
    sell_trades = trade_history[trade_history["方向"] == "卖出"]
    if not sell_trades.empty:
        ts1, ts2, ts3, ts4 = st.columns(4)
        with ts1:
            st.metric("总交易笔数", len(trade_history))
        with ts2:
            wins = sell_trades[sell_trades["盈亏"] > 0]
            st.metric("盈利笔数", len(wins))
        with ts3:
            total_pnl = sell_trades["盈亏"].sum()
            st.metric("累计盈亏", f"¥{total_pnl:+,.0f}")
        with ts4:
            avg_pnl = sell_trades["盈亏"].mean()
            st.metric("平均盈亏", f"¥{avg_pnl:+,.0f}")

    # 导出
    st.subheader("📥 数据导出")
    col_e1, col_e2, col_e3 = st.columns(3)
    with col_e1:
        csv_holdings = holdings.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "📄 导出持仓明细 (CSV)",
            data=csv_holdings,
            file_name="holdings.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with col_e2:
        csv_trades = trade_history.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "📄 导出交易记录 (CSV)",
            data=csv_trades,
            file_name="trade_history.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with col_e3:
        eq_df = equity_series.reset_index()
        eq_df.columns = ["日期", "净值"]
        csv_equity = eq_df.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "📄 导出净值曲线 (CSV)",
            data=csv_equity,
            file_name="equity_curve.csv",
            mime="text/csv",
            use_container_width=True,
        )
