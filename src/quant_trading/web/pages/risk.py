"""风险监控 - VaR展示、回撤监控、波动率趋势、相关性矩阵、风控规则配置

全部使用模拟数据。
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from quant_trading.web.charts import correlation_matrix, drawdown_chart
from quant_trading.web.components import metric_card


# ---------------------------------------------------------------------------
# 模拟数据
# ---------------------------------------------------------------------------

def _mock_var_data() -> dict:
    """模拟 VaR 指标"""
    np.random.seed(111)
    return {
        "历史VaR(95%)": -0.0186,
        "历史VaR(99%)": -0.0312,
        "参数VaR(95%)": -0.0195,
        "参数VaR(99%)": -0.0328,
        "预期损失ES(95%)": -0.0254,
        "组合Beta": 0.85,
        "跟踪误差": 0.065,
        "信息比率": 0.42,
    }


def _mock_drawdown_series() -> pd.Series:
    """模拟回撤序列"""
    np.random.seed(222)
    days = 365
    dates = pd.date_range(end=datetime.now().date(), periods=days, freq="B")
    returns = np.random.normal(0.0003, 0.013, days)
    equity = 1_000_000 * np.cumprod(1 + returns)
    running_max = np.maximum.accumulate(equity)
    drawdown = (running_max - equity) / running_max
    return pd.Series(drawdown, index=dates)


def _mock_volatility_series() -> pd.DataFrame:
    """模拟波动率时序数据（20日/60日滚动）"""
    np.random.seed(333)
    days = 365
    dates = pd.date_range(end=datetime.now().date(), periods=days, freq="B")
    returns = np.random.normal(0.0003, 0.013, days)
    ret_series = pd.Series(returns, index=dates)

    vol_20 = ret_series.rolling(20).std() * np.sqrt(244) * 100
    vol_60 = ret_series.rolling(60).std() * np.sqrt(244) * 100

    return pd.DataFrame({"20日波动率(%)": vol_20, "60日波动率(%)": vol_60}, index=dates)


def _mock_correlation() -> pd.DataFrame:
    """模拟资产相关性矩阵"""
    np.random.seed(444)
    stocks = ["贵州茅台", "五粮液", "中国平安", "招商银行", "宁德时代", "比亚迪", "恒瑞医药"]
    n = len(stocks)
    # 生成正定相关矩阵
    A = np.random.randn(n, n) * 0.3
    corr = np.corrcoef(A)
    # 保证对角线为 1
    np.fill_diagonal(corr, 1.0)
    return pd.DataFrame(np.round(corr, 2), index=stocks, columns=stocks)


def _mock_risk_metrics_history() -> pd.DataFrame:
    """模拟风险指标的历史序列"""
    np.random.seed(555)
    days = 180
    dates = pd.date_range(end=datetime.now().date(), periods=days, freq="B")
    var_95 = np.random.normal(-0.018, 0.003, days)
    max_dd = np.random.uniform(0.02, 0.12, days)
    beta = np.random.normal(0.85, 0.08, days)
    return pd.DataFrame({
        "日期": dates,
        "VaR(95%)": np.round(var_95 * 100, 2),
        "当前回撤(%)": np.round(max_dd * 100, 2),
        "Beta": np.round(beta, 2),
    })


# ---------------------------------------------------------------------------
# 渲染入口
# ---------------------------------------------------------------------------

def render() -> None:
    st.title("🛡️ 风险监控")

    # ====== 核心风险指标 ======
    st.subheader("📊 核心风险指标")
    var_data = _mock_var_data()

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card(
            "VaR(95%, 1日)",
            f"{var_data['历史VaR(95%)']:.2%}",
            delta=f"≈ ¥{abs(var_data['历史VaR(95%)']) * 1_256_800:.0f}",
            color="red",
        )
    with c2:
        metric_card(
            "VaR(99%, 1日)",
            f"{var_data['历史VaR(99%)']:.2%}",
            delta=f"≈ ¥{abs(var_data['历史VaR(99%)']) * 1_256_800:.0f}",
            color="orange",
        )
    with c3:
        metric_card(
            "预期损失 ES(95%)",
            f"{var_data['预期损失ES(95%)']:.2%}",
            color="red",
        )
    with c4:
        metric_card(
            "组合 Beta",
            f"{var_data['组合Beta']:.2f}",
            delta=f"跟踪误差: {var_data['跟踪误差']:.2%}",
            color="blue",
        )

    st.markdown("")

    # ====== 回撤监控 ======
    col_dd, col_vol = st.columns(2)

    with col_dd:
        st.subheader("📉 回撤监控")
        dd_series = _mock_drawdown_series()
        fig_dd = drawdown_chart(dd_series, title="")
        st.plotly_chart(fig_dd, use_container_width=True)

        # 回撤统计
        current_dd = dd_series.iloc[-1]
        max_dd = dd_series.max()
        avg_dd = dd_series[dd_series > 0.001].mean()
        dc1, dc2, dc3 = st.columns(3)
        with dc1:
            st.metric("当前回撤", f"{current_dd:.2%}")
        with dc2:
            st.metric("最大回撤", f"{max_dd:.2%}")
        with dc3:
            st.metric("平均回撤", f"{avg_dd:.2%}")

    with col_vol:
        st.subheader("📈 波动率趋势")
        vol_df = _mock_volatility_series().dropna()

        fig_vol = go.Figure()
        fig_vol.add_trace(go.Scatter(
            x=vol_df.index, y=vol_df["20日波动率(%)"],
            mode="lines", name="20日波动率",
            line=dict(width=1.5, color="#FF6347"),
        ))
        fig_vol.add_trace(go.Scatter(
            x=vol_df.index, y=vol_df["60日波动率(%)"],
            mode="lines", name="60日波动率",
            line=dict(width=1.5, color="#4169E1"),
        ))
        fig_vol.update_layout(
            yaxis_title="波动率 (%)",
            height=300,
            template="plotly_white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=50, r=20, t=20, b=30),
        )
        st.plotly_chart(fig_vol, use_container_width=True)

        # 波动率统计
        latest_vol = vol_df["20日波动率(%)"].iloc[-1]
        avg_vol = vol_df["20日波动率(%)"].mean()
        max_vol = vol_df["20日波动率(%)"].max()
        vc1, vc2, vc3 = st.columns(3)
        with vc1:
            st.metric("当前波动率", f"{latest_vol:.1f}%")
        with vc2:
            st.metric("平均波动率", f"{avg_vol:.1f}%")
        with vc3:
            st.metric("最高波动率", f"{max_vol:.1f}%")

    st.markdown("---")

    # ====== 相关性矩阵 ======
    st.subheader("🔗 持仓相关性矩阵")
    corr_df = _mock_correlation()
    fig_corr = correlation_matrix(corr_df, title="")
    st.plotly_chart(fig_corr, use_container_width=True)

    st.markdown("---")

    # ====== 风险指标历史趋势 ======
    st.subheader("📊 风险指标趋势")
    risk_hist = _mock_risk_metrics_history()

    tab_var, tab_dd, tab_beta = st.tabs(["VaR趋势", "回撤趋势", "Beta趋势"])

    with tab_var:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=risk_hist["日期"], y=risk_hist["VaR(95%)"],
            mode="lines", name="VaR(95%)",
            line=dict(color="#FF6347"),
        ))
        fig.add_hline(y=-2.0, line_dash="dash", line_color="red",
                      annotation_text="预警线 -2%")
        fig.update_layout(yaxis_title="VaR (%)", height=300, template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

    with tab_dd:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=risk_hist["日期"], y=risk_hist["当前回撤(%)"],
            mode="lines", name="回撤",
            fill="tozeroy", fillcolor="rgba(255,68,68,0.15)",
            line=dict(color="#FF6347"),
        ))
        fig.add_hline(y=10, line_dash="dash", line_color="red",
                      annotation_text="预警线 10%")
        fig.update_layout(yaxis_title="回撤 (%)", height=300, template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

    with tab_beta:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=risk_hist["日期"], y=risk_hist["Beta"],
            mode="lines", name="Beta",
            line=dict(color="#4169E1"),
        ))
        fig.add_hline(y=1.0, line_dash="dash", line_color="gray",
                      annotation_text="Beta=1")
        fig.update_layout(yaxis_title="Beta", height=300, template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # ====== 风控规则配置 ======
    st.subheader("⚙️ 风控规则配置")

    with st.expander("📋 当前风控规则", expanded=True):
        rc1, rc2 = st.columns(2)
        with rc1:
            max_pos = st.slider("单只最大仓位 (%)", 5, 50, 30, key="risk_max_pos")
            max_total = st.slider("最大总仓位 (%)", 30, 100, 80, key="risk_max_total")
            stop_loss = st.slider("止损比例 (%)", 1, 20, 5, key="risk_stop_loss")
        with rc2:
            take_profit = st.slider("止盈比例 (%)", 5, 50, 15, key="risk_take_profit")
            max_drawdown_limit = st.slider("最大回撤限制 (%)", 5, 30, 15, key="risk_max_dd")
            var_limit = st.number_input(
                "单日VaR限额 (元)",
                min_value=10000.0,
                max_value=500000.0,
                value=50000.0,
                step=5000.0,
                key="risk_var_limit",
            )

        if st.button("💾 保存风控规则", type="primary"):
            st.success("风控规则已更新！")
            st.json({
                "单只最大仓位": f"{max_pos}%",
                "最大总仓位": f"{max_total}%",
                "止损比例": f"{stop_loss}%",
                "止盈比例": f"{take_profit}%",
                "最大回撤限制": f"{max_drawdown_limit}%",
                "单日VaR限额": f"¥{var_limit:,.0f}",
            })

    # ====== 风控事件日志 ======
    with st.expander("📜 风控事件日志"):
        events = [
            {"时间": "2025-12-20 14:32:00", "类型": "止损触发", "标的": "300750 宁德时代", "详情": "跌幅达 5.2%，已自动平仓"},
            {"时间": "2025-12-18 10:15:00", "类型": "仓位预警", "标的": "600519 贵州茅台", "详情": "单只仓位达 28%，接近 30% 上限"},
            {"时间": "2025-12-15 09:35:00", "类型": "VaR预警", "标的": "组合", "详情": "日VaR达 ¥48,500，接近 ¥50,000 限额"},
            {"时间": "2025-12-10 14:50:00", "类型": "止盈触发", "标的": "002594 比亚迪", "详情": "涨幅达 15.8%，已自动减仓 50%"},
            {"时间": "2025-12-05 11:20:00", "类型": "回撤预警", "标的": "组合", "详情": "组合回撤达 12.3%，接近 15% 上限"},
        ]
        st.dataframe(
            pd.DataFrame(events),
            use_container_width=True,
            hide_index=True,
        )
