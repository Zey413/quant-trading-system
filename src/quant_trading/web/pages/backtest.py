"""策略回测 - 策略选择、参数配置、一键回测、结果展示、多策略对比

支持选择已注册策略并动态生成参数表单，结果包含净值曲线、回撤图、
月度热力图、绩效指标面板、交易明细。
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st

from quant_trading.web.charts import (
    drawdown_chart,
    equity_curve,
    monthly_returns_heatmap,
)
from quant_trading.web.components import metric_card, performance_table, trade_table


# ---------------------------------------------------------------------------
# 策略参数定义（供动态表单使用）
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# 模拟回测结果
# ---------------------------------------------------------------------------

def _simulate_backtest(
    strategy_name: str,
    params: dict,
    start_date: str,
    end_date: str,
    initial_capital: float,
) -> dict:
    """模拟生成回测结果（不依赖真实数据源）"""
    np.random.seed(hash(strategy_name + str(params)) % 2**31)

    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    dates = pd.date_range(start=start, end=end, freq="B")
    n_days = len(dates)

    if n_days < 10:
        return {}

    # 根据策略特征生成不同的收益分布
    volatility_map = {
        "ma_crossover": 0.015,
        "rsi": 0.018,
        "macd": 0.014,
        "bollinger": 0.016,
        "dual_thrust": 0.020,
        "mean_reversion": 0.012,
        "turtle": 0.022,
    }
    drift_map = {
        "ma_crossover": 0.0004,
        "rsi": 0.0003,
        "macd": 0.0005,
        "bollinger": 0.0002,
        "dual_thrust": 0.0006,
        "mean_reversion": 0.0003,
        "turtle": 0.0004,
    }
    vol = volatility_map.get(strategy_name, 0.015)
    drift = drift_map.get(strategy_name, 0.0003)

    daily_returns = np.random.normal(drift, vol, n_days)
    equity_values = initial_capital * np.cumprod(1 + daily_returns)
    equity_series = pd.Series(equity_values, index=dates)

    # 基准净值
    bm_returns = np.random.normal(0.0002, 0.012, n_days)
    bm_values = initial_capital * np.cumprod(1 + bm_returns)
    benchmark_series = pd.Series(bm_values, index=dates)

    # 回撤计算
    running_max = equity_series.cummax()
    drawdown_series = (running_max - equity_series) / running_max

    # 绩效指标
    total_return = (equity_values[-1] - initial_capital) / initial_capital
    years = n_days / 244
    ann_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
    ann_vol = float(pd.Series(daily_returns).std() * np.sqrt(244))
    max_dd = float(drawdown_series.max())
    sharpe = (ann_return - 0.025) / ann_vol if ann_vol > 0 else 0

    # 模拟交易记录
    n_trades = np.random.randint(20, 80)
    trade_dates = np.random.choice(dates, size=n_trades * 2, replace=True)
    trade_dates.sort()
    stocks = ["600519", "000858", "601318", "600036", "000001", "300750"]
    stock_names = ["贵州茅台", "五粮液", "中国平安", "招商银行", "平安银行", "宁德时代"]

    trades_list = []
    for i in range(n_trades):
        idx = np.random.randint(0, len(stocks))
        pnl = np.random.normal(2000, 8000)
        buy_price = np.random.uniform(10, 200)
        sell_price = buy_price * (1 + pnl / (buy_price * 100))
        qty = np.random.choice([100, 200, 300, 500, 1000])

        buy_date = trade_dates[i * 2]
        sell_date = trade_dates[i * 2 + 1]
        if sell_date <= buy_date:
            sell_date = buy_date + pd.Timedelta(days=np.random.randint(1, 30))

        trades_list.append({
            "买入日期": buy_date.strftime("%Y-%m-%d"),
            "卖出日期": sell_date.strftime("%Y-%m-%d"),
            "代码": stocks[idx],
            "名称": stock_names[idx],
            "方向": "卖出",
            "数量": qty,
            "买入价": round(buy_price, 2),
            "卖出价": round(sell_price, 2),
            "盈亏": round(pnl, 2),
        })

    trades_df = pd.DataFrame(trades_list)
    wins = [t for t in trades_list if t["盈亏"] > 0]
    losses = [t for t in trades_list if t["盈亏"] < 0]
    win_rate = len(wins) / n_trades if n_trades > 0 else 0
    avg_win = float(np.mean([w["盈亏"] for w in wins])) if wins else 0
    avg_loss = float(np.mean([l["盈亏"] for l in losses])) if losses else 0
    pl_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else 0

    # 下行偏差
    negative_returns = daily_returns[daily_returns < 0]
    downside_std = float(np.std(negative_returns) * np.sqrt(244)) if len(negative_returns) > 1 else ann_vol
    sortino = (ann_return - 0.025) / downside_std if downside_std > 0 else 0
    calmar = ann_return / max_dd if max_dd > 0 else 0

    max_dd_dur = 0
    cur_dur = 0
    for d in drawdown_series:
        if d > 1e-6:
            cur_dur += 1
            max_dd_dur = max(max_dd_dur, cur_dur)
        else:
            cur_dur = 0

    return {
        "strategy_name": strategy_name,
        "params": params,
        "equity_series": equity_series,
        "benchmark_series": benchmark_series,
        "drawdown_series": drawdown_series,
        "trades_df": trades_df,
        "metrics": {
            "总收益率": f"{total_return:.2%}",
            "年化收益率": f"{ann_return:.2%}",
            "年化波动率": f"{ann_vol:.2%}",
            "最大回撤": f"{max_dd:.2%}",
            "最大回撤持续天数": f"{max_dd_dur}",
            "夏普比率": round(sharpe, 4),
            "Sortino比率": round(sortino, 4),
            "Calmar比率": round(calmar, 4),
            "总交易次数": n_trades,
            "胜率": f"{win_rate:.2%}",
            "盈亏比": round(pl_ratio, 4),
            "平均盈利": round(avg_win, 2),
            "平均亏损": round(avg_loss, 2),
        },
        "summary_values": {
            "total_return": total_return,
            "ann_return": ann_return,
            "max_dd": max_dd,
            "sharpe": sharpe,
        },
    }


# ---------------------------------------------------------------------------
# 渲染入口
# ---------------------------------------------------------------------------

def render() -> None:
    st.title("🔬 策略回测")

    tab_single, tab_compare = st.tabs(["📊 单策略回测", "📈 多策略对比"])

    # ==================================================================
    # 单策略回测
    # ==================================================================
    with tab_single:
        col_config, col_result = st.columns([1, 3])

        with col_config:
            st.subheader("⚙️ 参数配置")

            # 策略选择
            strategy_names = list(STRATEGY_PARAMS.keys())
            strategy_key = st.selectbox(
                "选择策略",
                strategy_names,
                format_func=lambda x: STRATEGY_PARAMS[x]["display_name"],
            )
            st.caption(STRATEGY_PARAMS[strategy_key]["description"])

            st.markdown("---")

            # 动态参数表单
            strategy_config = STRATEGY_PARAMS[strategy_key]
            user_params: dict = {}
            for pname, pdef in strategy_config["params"].items():
                if pdef["type"] == "int":
                    user_params[pname] = st.number_input(
                        pdef["label"],
                        min_value=pdef["min"],
                        max_value=pdef["max"],
                        value=pdef["default"],
                        step=1,
                        key=f"bt_{strategy_key}_{pname}",
                    )
                elif pdef["type"] == "float":
                    user_params[pname] = st.number_input(
                        pdef["label"],
                        min_value=float(pdef["min"]),
                        max_value=float(pdef["max"]),
                        value=float(pdef["default"]),
                        step=0.1,
                        key=f"bt_{strategy_key}_{pname}",
                    )

            st.markdown("---")

            # 回测参数
            start_date = st.date_input("开始日期", value=pd.Timestamp("2023-01-01"))
            end_date = st.date_input("结束日期", value=pd.Timestamp("2025-12-31"))
            initial_capital = st.number_input(
                "初始资金 (元)",
                min_value=10000.0,
                max_value=100_000_000.0,
                value=1_000_000.0,
                step=100_000.0,
                format="%.0f",
            )

            st.markdown("---")

            run_btn = st.button("🚀 一键回测", type="primary", use_container_width=True)

        with col_result:
            if run_btn:
                with st.spinner("正在运行回测..."):
                    result = _simulate_backtest(
                        strategy_key,
                        user_params,
                        start_date.strftime("%Y-%m-%d"),
                        end_date.strftime("%Y-%m-%d"),
                        initial_capital,
                    )

                if not result:
                    st.error("回测失败：日期范围过短，请调整参数")
                else:
                    st.session_state.backtest_results[strategy_key] = result
                    _render_result(result)

            elif strategy_key in st.session_state.backtest_results:
                _render_result(st.session_state.backtest_results[strategy_key])
            else:
                st.info("👈 请在左侧配置策略参数，然后点击「一键回测」")

    # ==================================================================
    # 多策略对比
    # ==================================================================
    with tab_compare:
        _render_comparison()


def _render_result(result: dict) -> None:
    """渲染单策略回测结果"""
    sv = result["summary_values"]

    # 顶部指标卡
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card(
            "总收益率", f"{sv['total_return']:.2%}",
            color="red" if sv["total_return"] > 0 else "green",
        )
    with c2:
        metric_card(
            "年化收益率", f"{sv['ann_return']:.2%}",
            color="red" if sv["ann_return"] > 0 else "green",
        )
    with c3:
        metric_card("最大回撤", f"{sv['max_dd']:.2%}", color="orange")
    with c4:
        metric_card("夏普比率", f"{sv['sharpe']:.4f}", color="purple")

    st.markdown("")

    # 净值曲线
    st.subheader("📈 净值曲线")
    fig_eq = equity_curve(
        result["equity_series"],
        result["benchmark_series"],
        title="",
    )
    st.plotly_chart(fig_eq, use_container_width=True)

    # 回撤 + 绩效
    col_dd, col_perf = st.columns([2, 1])
    with col_dd:
        st.subheader("📉 回撤曲线")
        fig_dd = drawdown_chart(result["drawdown_series"], title="")
        st.plotly_chart(fig_dd, use_container_width=True)
    with col_perf:
        st.subheader("📋 绩效指标")
        performance_table(result["metrics"])

    # 月度热力图
    st.subheader("🗓️ 月度收益热力图")
    fig_hm = monthly_returns_heatmap(result["equity_series"], title="")
    st.plotly_chart(fig_hm, use_container_width=True)

    # 交易明细
    st.subheader("📝 交易明细")
    trade_table(result["trades_df"])


def _render_comparison() -> None:
    """渲染多策略对比"""
    results = st.session_state.backtest_results

    if len(results) < 2:
        st.info("请先在「单策略回测」标签页中至少运行两个不同策略的回测，然后回到此处进行对比。")
        st.caption(f"当前已回测策略: {', '.join(results.keys()) if results else '无'}")
        return

    st.subheader("📈 策略净值对比")
    import plotly.graph_objects as go

    fig = go.Figure()
    for name, res in results.items():
        display_name = STRATEGY_PARAMS.get(name, {}).get("display_name", name)
        eq = res["equity_series"]
        norm = eq / eq.iloc[0]
        fig.add_trace(go.Scatter(
            x=norm.index, y=norm.values,
            mode="lines", name=display_name,
            line=dict(width=2),
        ))
    fig.update_layout(
        yaxis_title="净值",
        height=450,
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig, use_container_width=True)

    # 对比表格
    st.subheader("📊 绩效指标对比")
    compare_data = []
    for name, res in results.items():
        display_name = STRATEGY_PARAMS.get(name, {}).get("display_name", name)
        row = {"策略": display_name}
        row.update(res["metrics"])
        compare_data.append(row)

    compare_df = pd.DataFrame(compare_data)
    st.dataframe(compare_df, use_container_width=True, hide_index=True)

    # 回撤对比
    st.subheader("📉 回撤对比")
    fig_dd = go.Figure()
    for name, res in results.items():
        display_name = STRATEGY_PARAMS.get(name, {}).get("display_name", name)
        dd = res["drawdown_series"]
        fig_dd.add_trace(go.Scatter(
            x=dd.index, y=-dd.values * 100,
            mode="lines", name=display_name,
            fill="tozeroy",
        ))
    fig_dd.update_layout(
        yaxis_title="回撤 (%)",
        height=350,
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig_dd, use_container_width=True)
