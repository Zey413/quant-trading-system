"""系统设置 - 交易参数、数据源、风险参数、系统信息"""

from __future__ import annotations

import streamlit as st


def render() -> None:
    st.title("⚙️ 系统设置")

    settings = st.session_state.settings

    tab_trade, tab_data, tab_risk, tab_about = st.tabs([
        "💰 交易参数", "📡 数据源", "🛡️ 风险参数", "ℹ️ 关于系统",
    ])

    # ==================================================================
    # 交易参数
    # ==================================================================
    with tab_trade:
        st.subheader("交易成本与规则")

        c1, c2 = st.columns(2)
        with c1:
            initial_capital = st.number_input(
                "初始资金 (元)",
                min_value=10_000.0,
                max_value=100_000_000.0,
                value=float(settings.get("initial_capital", 1_000_000)),
                step=100_000.0,
                format="%.0f",
                key="set_capital",
            )
            commission_rate = st.number_input(
                "佣金率",
                min_value=0.0,
                max_value=0.01,
                value=float(settings.get("commission_rate", 0.0003)),
                step=0.0001,
                format="%.4f",
                key="set_commission",
                help="一般为万三(0.0003)",
            )
            min_commission = st.number_input(
                "最低佣金 (元)",
                min_value=0.0,
                max_value=50.0,
                value=5.0,
                step=1.0,
                key="set_min_comm",
            )
        with c2:
            stamp_tax = st.number_input(
                "印花税率 (仅卖出)",
                min_value=0.0,
                max_value=0.01,
                value=float(settings.get("stamp_tax_rate", 0.001)),
                step=0.0001,
                format="%.4f",
                key="set_stamp",
                help="当前为千分之一(0.001)",
            )
            slippage = st.number_input(
                "滑点",
                min_value=0.0,
                max_value=0.01,
                value=float(settings.get("slippage", 0.001)),
                step=0.0001,
                format="%.4f",
                key="set_slippage",
            )
            lot_size = st.selectbox(
                "最小交易单位",
                [100, 200, 500, 1000],
                index=0,
                key="set_lot",
                help="A股一手=100股",
            )

        t_plus_1 = st.checkbox("T+1 规则", value=True, key="set_t1",
                               help="A股实行T+1，当日买入不可卖出")

        if st.button("💾 保存交易参数", key="save_trade"):
            settings["initial_capital"] = initial_capital
            settings["commission_rate"] = commission_rate
            settings["stamp_tax_rate"] = stamp_tax
            settings["slippage"] = slippage
            st.session_state.settings = settings
            st.success("交易参数已保存!")

    # ==================================================================
    # 数据源配置
    # ==================================================================
    with tab_data:
        st.subheader("数据源配置")

        data_source = st.selectbox(
            "默认数据源",
            ["akshare", "tushare"],
            index=0 if settings.get("data_source", "akshare") == "akshare" else 1,
            key="set_source",
        )

        if data_source == "tushare":
            tushare_token = st.text_input(
                "Tushare Token",
                type="password",
                key="set_tushare_token",
                help="前往 tushare.pro 注册获取",
            )
        else:
            st.info("AKShare 为免费数据源，无需配置 Token。")

        st.markdown("---")

        st.subheader("缓存设置")
        cache_enabled = st.checkbox("启用数据缓存", value=True, key="set_cache")
        cache_dir = st.text_input("缓存目录", value="data_cache", key="set_cache_dir")

        if st.button("💾 保存数据源配置", key="save_data"):
            settings["data_source"] = data_source
            st.session_state.settings = settings
            st.success("数据源配置已保存!")

    # ==================================================================
    # 风险参数
    # ==================================================================
    with tab_risk:
        st.subheader("风险管理参数")

        c1, c2 = st.columns(2)
        with c1:
            benchmark = st.text_input(
                "基准指数",
                value=settings.get("benchmark", "000300"),
                key="set_benchmark",
                help="沪深300: 000300, 上证50: 000016",
            )
            risk_free_rate = st.number_input(
                "无风险利率",
                min_value=0.0,
                max_value=0.10,
                value=float(settings.get("risk_free_rate", 0.025)),
                step=0.005,
                format="%.3f",
                key="set_rf",
                help="10年期国债利率",
            )
            trading_days = st.number_input(
                "年交易日数",
                min_value=200,
                max_value=260,
                value=244,
                key="set_td",
            )
        with c2:
            position_method = st.selectbox(
                "仓位管理方法",
                ["固定比例", "固定金额", "Kelly公式", "ATR自适应"],
                key="set_pos_method",
            )
            fixed_ratio = st.slider(
                "固定仓位比例 (%)",
                min_value=1,
                max_value=50,
                value=10,
                key="set_fixed_ratio",
            )
            stop_loss_method = st.selectbox(
                "止损方法",
                ["固定百分比", "ATR跟踪止损", "支撑位止损"],
                key="set_sl_method",
            )

        if st.button("💾 保存风险参数", key="save_risk"):
            settings["benchmark"] = benchmark
            settings["risk_free_rate"] = risk_free_rate
            st.session_state.settings = settings
            st.success("风险参数已保存!")

    # ==================================================================
    # 关于系统
    # ==================================================================
    with tab_about:
        st.subheader("关于量化交易系统")

        st.markdown("""
        ### 📈 A股量化交易系统 v0.3.0

        **功能特性:**
        - 🔌 可插拔数据源 (AKShare / Tushare)
        - 📊 多策略回测引擎
        - 🛡️ 全面风险管理
        - 📈 交互式可视化 Dashboard
        - 💼 组合管理与交易记录

        **内置策略:**
        - 均线交叉策略 (MA Crossover)
        - RSI 超买超卖策略
        - MACD 策略
        - 布林带策略 (Bollinger Bands)
        - Dual Thrust 策略
        - 均值回归策略 (Mean Reversion)
        - 海龟交易策略 (Turtle Trading)
        - 组合策略 (Composite)

        **技术栈:**
        - Python 3.10+
        - Pandas / NumPy / Plotly
        - Streamlit
        - Pydantic
        """)

        st.markdown("---")
        st.subheader("🔧 当前配置概览")
        st.json(st.session_state.settings)

        st.markdown("---")
        st.caption("作者: zey413 | 协议: MIT")
