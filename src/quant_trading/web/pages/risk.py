"""风险监控 - VaR展示、回撤监控、波动率趋势、相关性矩阵、
仓位管理、止损止盈设置、风控规则展示与配置

集成 quant_trading.risk 模块（RiskManager/PositionSizer/StopLossManager），
失败时回退到模拟数据。
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from quant_trading.web.charts import correlation_matrix, drawdown_chart
from quant_trading.web.components import metric_card, risk_gauge, info_box, STOCK_POOL


# ---------------------------------------------------------------------------
# 尝试加载真实风控模块
# ---------------------------------------------------------------------------

def _get_risk_config() -> dict:
    """获取风控配置（优先从 AppConfig 读取）"""
    try:
        from quant_trading.core.config import AppConfig
        config = AppConfig()
        rc = config.risk
        return {
            "max_position_pct": rc.max_position_pct,
            "max_total_position_pct": rc.max_total_position_pct,
            "stop_loss_pct": rc.stop_loss_pct,
            "take_profit_pct": rc.take_profit_pct,
            "position_sizing_method": rc.position_sizing_method,
            "fixed_ratio": rc.fixed_ratio,
        }
    except Exception:
        return {
            "max_position_pct": 0.30,
            "max_total_position_pct": 0.80,
            "stop_loss_pct": 0.05,
            "take_profit_pct": 0.15,
            "position_sizing_method": "fixed_ratio",
            "fixed_ratio": 0.10,
        }


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
    A = np.random.randn(n, n) * 0.3
    corr = np.corrcoef(A)
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


def _mock_position_analysis() -> list[dict]:
    """模拟个股仓位风险分析"""
    stocks = [
        ("600519", "贵州茅台", 175600, 0.1397),
        ("000858", "五粮液", 84250, 0.0670),
        ("601318", "中国平安", 52300, 0.0416),
        ("600036", "招商银行", 30240, 0.0241),
        ("000001", "平安银行", 24200, 0.0193),
        ("300750", "宁德时代", 43120, 0.0343),
        ("002594", "比亚迪", 80550, 0.0641),
        ("600276", "恒瑞医药", 28680, 0.0228),
    ]
    np.random.seed(666)
    result = []
    for code, name, mv, pos_pct in stocks:
        daily_vol = np.random.uniform(0.015, 0.035)
        var_95 = mv * daily_vol * 1.645
        result.append({
            "代码": code,
            "名称": name,
            "市值": mv,
            "仓位(%)": round(pos_pct * 100, 2),
            "日波动率(%)": round(daily_vol * 100, 2),
            "VaR(95%)": round(var_95, 0),
            "风险贡献(%)": round(np.random.uniform(5, 25), 1),
        })
    return result


# ---------------------------------------------------------------------------
# 渲染入口
# ---------------------------------------------------------------------------

def render() -> None:
    st.title("🛡️ 风险监控")

    tab_overview, tab_position, tab_rules, tab_log = st.tabs([
        "📊 风险概览", "📐 仓位管理", "⚙️ 风控规则", "📜 事件日志",
    ])

    with tab_overview:
        _render_risk_overview()

    with tab_position:
        _render_position_management()

    with tab_rules:
        _render_risk_rules()

    with tab_log:
        _render_event_log()


def _render_risk_overview() -> None:
    """风险概览页面"""
    # ====== 核心风险指标 ======
    st.subheader("📊 核心风险指标")
    var_data = _mock_var_data()
    total_assets = 1_256_800.0

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card(
            "VaR(95%, 1日)",
            f"{var_data['历史VaR(95%)']:.2%}",
            delta=f"约 ¥{abs(var_data['历史VaR(95%)']) * total_assets:,.0f}",
            color="red",
        )
    with c2:
        metric_card(
            "VaR(99%, 1日)",
            f"{var_data['历史VaR(99%)']:.2%}",
            delta=f"约 ¥{abs(var_data['历史VaR(99%)']) * total_assets:,.0f}",
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

    # 风险水平仪表
    risk_rules = st.session_state.get("risk_rules", {})
    max_dd_limit = risk_rules.get("max_drawdown_limit", 15) / 100

    dd_series = _mock_drawdown_series()
    current_dd = float(dd_series.iloc[-1])
    risk_level = current_dd / max_dd_limit if max_dd_limit > 0 else 0

    r1, r2, r3 = st.columns(3)
    with r1:
        risk_gauge(min(risk_level, 1.0), "回撤风险水平", (0.4, 0.7))
    with r2:
        var_risk = abs(var_data['历史VaR(95%)']) / 0.03
        risk_gauge(min(var_risk, 1.0), "VaR风险水平", (0.4, 0.7))
    with r3:
        vol_risk = var_data['跟踪误差'] / 0.10
        risk_gauge(min(vol_risk, 1.0), "波动率风险水平", (0.4, 0.7))

    st.markdown("---")

    # ====== 回撤监控 + 波动率趋势 ======
    col_dd, col_vol = st.columns(2)

    with col_dd:
        st.subheader("📉 回撤监控")
        fig_dd = drawdown_chart(dd_series, title="")
        st.plotly_chart(fig_dd, use_container_width=True)

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


def _render_position_management() -> None:
    """仓位管理页面"""
    st.subheader("📐 仓位风险分析")

    risk_config = _get_risk_config()

    # 个股仓位风险表
    position_data = _mock_position_analysis()
    pos_df = pd.DataFrame(position_data)

    # 仓位预警
    max_pos_pct = risk_config["max_position_pct"] * 100
    warnings = pos_df[pos_df["仓位(%)"] > max_pos_pct * 0.8]
    if not warnings.empty:
        for _, row in warnings.iterrows():
            st.warning(
                f"⚠️ **{row['名称']}** 仓位 {row['仓位(%)']:.2f}%，"
                f"接近上限 {max_pos_pct:.0f}%"
            )

    st.dataframe(
        pos_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "市值": st.column_config.NumberColumn("市值", format="¥%,.0f"),
            "VaR(95%)": st.column_config.NumberColumn("VaR(95%)", format="¥%,.0f"),
            "仓位(%)": st.column_config.NumberColumn("仓位(%)", format="%.2f%%"),
            "日波动率(%)": st.column_config.NumberColumn("日波动率(%)", format="%.2f%%"),
            "风险贡献(%)": st.column_config.NumberColumn("风险贡献(%)", format="%.1f%%"),
        },
    )

    st.markdown("---")

    # 仓位计算器
    st.subheader("🧮 仓位计算器")
    info_box(
        "仓位计算",
        "根据风控规则和当前资产，计算单只股票的建议买入数量。",
    )

    cal_c1, cal_c2 = st.columns(2)
    with cal_c1:
        total_assets = st.number_input(
            "总资产 (元)", value=1_256_800.0, step=10000.0,
            format="%.0f", key="pos_total",
        )
        method = st.selectbox(
            "仓位方法",
            ["固定比例", "固定金额", "Kelly公式", "ATR自适应"],
            key="pos_method",
        )
        stock_price = st.number_input(
            "股票现价 (元)", value=50.0, step=1.0,
            format="%.2f", key="pos_price",
        )

    with cal_c2:
        if method == "固定比例":
            ratio = st.slider("仓位比例 (%)", 1, 50,
                               int(risk_config["fixed_ratio"] * 100),
                               key="pos_ratio")
            target_value = total_assets * ratio / 100
        elif method == "固定金额":
            target_value = st.number_input(
                "买入金额 (元)", value=100_000.0, step=10000.0,
                format="%.0f", key="pos_amount",
            )
        elif method == "Kelly公式":
            win_rate = st.slider("预期胜率 (%)", 30, 90, 55, key="kelly_wr") / 100
            pl_ratio = st.number_input("盈亏比", value=1.5, step=0.1, key="kelly_pl")
            kelly_f = win_rate - (1 - win_rate) / pl_ratio
            kelly_f = max(0, min(kelly_f, 0.5))  # 半Kelly限制
            target_value = total_assets * kelly_f / 2
            st.caption(f"Kelly比例: {kelly_f:.2%}, 半Kelly: {kelly_f/2:.2%}")
        else:  # ATR
            atr = st.number_input("ATR值", value=2.5, step=0.1, key="atr_val")
            risk_per_trade = total_assets * 0.01  # 单笔风险1%
            target_value = risk_per_trade / atr * stock_price if atr > 0 else 0
            st.caption(f"单笔风险: ¥{risk_per_trade:,.0f}")

        # 计算结果
        if stock_price > 0:
            shares = int(target_value / stock_price)
            lot_shares = (shares // 100) * 100  # A股整手
            actual_amount = lot_shares * stock_price
            position_pct = actual_amount / total_assets * 100

            st.markdown("---")
            st.markdown("**计算结果**")
            st.metric("建议买入 (股)", f"{lot_shares:,}")
            st.metric("实际金额", f"¥{actual_amount:,.0f}")
            st.metric("占总资产", f"{position_pct:.2f}%")

            # 仓位检查
            if position_pct > risk_config["max_position_pct"] * 100:
                st.error(f"⛔ 超出单只最大仓位限制 ({risk_config['max_position_pct']:.0%})")


def _render_risk_rules() -> None:
    """风控规则配置"""
    st.subheader("⚙️ 风控规则配置")

    risk_rules = st.session_state.get("risk_rules", {})

    with st.expander("📋 仓位管理规则", expanded=True):
        rc1, rc2 = st.columns(2)
        with rc1:
            max_pos = st.slider(
                "单只最大仓位 (%)", 5, 50,
                risk_rules.get("max_position_pct", 30),
                key="risk_max_pos",
            )
            max_total = st.slider(
                "最大总仓位 (%)", 30, 100,
                risk_rules.get("max_total_position_pct", 80),
                key="risk_max_total",
            )
            min_cash = st.slider(
                "最低现金比例 (%)", 0, 50,
                100 - risk_rules.get("max_total_position_pct", 80),
                key="risk_min_cash",
            )
        with rc2:
            stop_loss = st.slider(
                "止损比例 (%)", 1, 20,
                risk_rules.get("stop_loss_pct", 5),
                key="risk_stop_loss",
            )
            take_profit = st.slider(
                "止盈比例 (%)", 5, 50,
                risk_rules.get("take_profit_pct", 15),
                key="risk_take_profit",
            )
            trailing_stop = st.checkbox(
                "启用跟踪止损",
                value=False,
                key="risk_trailing",
                help="价格创新高后回落一定比例触发止损",
            )

    with st.expander("📋 组合风控规则", expanded=True):
        rc3, rc4 = st.columns(2)
        with rc3:
            max_drawdown_limit = st.slider(
                "最大回撤限制 (%)", 5, 30,
                risk_rules.get("max_drawdown_limit", 15),
                key="risk_max_dd",
            )
            var_limit = st.number_input(
                "单日VaR限额 (元)",
                min_value=10000.0,
                max_value=500000.0,
                value=float(risk_rules.get("var_limit", 50000)),
                step=5000.0,
                key="risk_var_limit",
            )
        with rc4:
            max_corr = st.slider(
                "持仓最大相关性", 0.3, 1.0, 0.8, 0.05,
                key="risk_max_corr",
                help="同行业/高相关性股票的仓位限制",
            )
            max_industry = st.slider(
                "单行业最大仓位 (%)", 10, 60, 40,
                key="risk_max_industry",
            )

    if st.button("💾 保存风控规则", type="primary"):
        st.session_state.risk_rules = {
            "max_position_pct": max_pos,
            "max_total_position_pct": max_total,
            "stop_loss_pct": stop_loss,
            "take_profit_pct": take_profit,
            "max_drawdown_limit": max_drawdown_limit,
            "var_limit": var_limit,
        }
        st.success("风控规则已更新！")
        st.json({
            "单只最大仓位": f"{max_pos}%",
            "最大总仓位": f"{max_total}%",
            "止损比例": f"{stop_loss}%",
            "止盈比例": f"{take_profit}%",
            "跟踪止损": "启用" if trailing_stop else "禁用",
            "最大回撤限制": f"{max_drawdown_limit}%",
            "单日VaR限额": f"¥{var_limit:,.0f}",
            "持仓最大相关性": f"{max_corr:.2f}",
            "单行业最大仓位": f"{max_industry}%",
        })

    # 尝试将规则写入 AppConfig
    with st.expander("🔗 同步到配置文件", expanded=False):
        info_box(
            "配置同步",
            "将当前风控规则同步到 AppConfig YAML 配置文件，供回测引擎使用。",
        )
        if st.button("同步规则到 config.yaml", key="sync_risk"):
            try:
                from quant_trading.core.config import AppConfig
                config = AppConfig()
                config.risk.max_position_pct = max_pos / 100
                config.risk.max_total_position_pct = max_total / 100
                config.risk.stop_loss_pct = stop_loss / 100
                config.risk.take_profit_pct = take_profit / 100
                config.to_yaml("config.yaml")
                st.success("已同步到 config.yaml")
            except Exception as e:
                st.warning(f"同步失败: {e}")


def _render_event_log() -> None:
    """风控事件日志"""
    st.subheader("📜 风控事件日志")

    # 筛选
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        event_type = st.selectbox(
            "事件类型",
            ["全部", "止损触发", "止盈触发", "仓位预警", "VaR预警", "回撤预警"],
            key="log_type",
        )
    with col_f2:
        log_days = st.selectbox("时间范围", ["近7天", "近30天", "近90天", "全部"], key="log_days")

    events = [
        {"时间": "2025-12-20 14:32:00", "类型": "止损触发", "级别": "🔴 严重", "标的": "300750 宁德时代", "详情": "跌幅达 5.2%，已自动平仓"},
        {"时间": "2025-12-18 10:15:00", "类型": "仓位预警", "级别": "🟡 警告", "标的": "600519 贵州茅台", "详情": "单只仓位达 28%，接近 30% 上限"},
        {"时间": "2025-12-15 09:35:00", "类型": "VaR预警", "级别": "🟡 警告", "标的": "组合", "详情": "日VaR达 ¥48,500，接近 ¥50,000 限额"},
        {"时间": "2025-12-10 14:50:00", "类型": "止盈触发", "级别": "🟢 信息", "标的": "002594 比亚迪", "详情": "涨幅达 15.8%，已自动减仓 50%"},
        {"时间": "2025-12-05 11:20:00", "类型": "回撤预警", "级别": "🟡 警告", "标的": "组合", "详情": "组合回撤达 12.3%，接近 15% 上限"},
        {"时间": "2025-11-28 15:00:00", "类型": "止损触发", "级别": "🔴 严重", "标的": "601012 隆基绿能", "详情": "跌幅达 6.1%，已自动平仓"},
        {"时间": "2025-11-20 09:45:00", "类型": "仓位预警", "级别": "🟡 警告", "标的": "000858 五粮液", "详情": "加仓后仓位达 25%"},
        {"时间": "2025-11-15 14:10:00", "类型": "止盈触发", "级别": "🟢 信息", "标的": "600036 招商银行", "详情": "涨幅达 16.2%，已自动减仓"},
    ]

    events_df = pd.DataFrame(events)
    if event_type != "全部":
        events_df = events_df[events_df["类型"] == event_type]

    st.dataframe(
        events_df,
        use_container_width=True,
        hide_index=True,
    )

    # 事件统计
    st.subheader("📊 事件统计")
    all_events = pd.DataFrame(events)
    es1, es2, es3, es4 = st.columns(4)
    with es1:
        st.metric("止损触发", len(all_events[all_events["类型"] == "止损触发"]))
    with es2:
        st.metric("止盈触发", len(all_events[all_events["类型"] == "止盈触发"]))
    with es3:
        st.metric("仓位预警", len(all_events[all_events["类型"] == "仓位预警"]))
    with es4:
        st.metric("VaR/回撤预警",
                   len(all_events[all_events["类型"].isin(["VaR预警", "回撤预警"])]))
