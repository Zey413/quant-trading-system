"""系统设置 - 交易参数、数据源、风险参数、系统信息

集成 AppConfig 进行 YAML 配置的加载/保存。
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st


# ---------------------------------------------------------------------------
# 尝试加载真实配置
# ---------------------------------------------------------------------------

def _try_load_config() -> dict | None:
    """尝试从 YAML 文件加载 AppConfig"""
    try:
        from quant_trading.core.config import AppConfig
        config_path = Path("config.yaml")
        config = AppConfig.from_yaml(config_path) if config_path.exists() else AppConfig()
        return {
            "initial_capital": config.backtest.initial_capital,
            "commission_rate": config.broker.commission_rate,
            "min_commission": config.broker.min_commission,
            "stamp_tax_rate": config.broker.stamp_tax_rate,
            "slippage": config.broker.slippage,
            "lot_size": config.broker.lot_size,
            "t_plus_1": config.broker.t_plus_1,
            "data_source": config.data_source.default_source,
            "tushare_token": config.data_source.tushare_token,
            "cache_enabled": config.data_source.cache_enabled,
            "cache_dir": config.data_source.cache_dir,
            "benchmark": config.backtest.benchmark,
            "risk_free_rate": config.backtest.risk_free_rate,
            "trading_days": config.broker.trading_days_per_year,
            "position_method": config.risk.position_sizing_method,
            "fixed_ratio": config.risk.fixed_ratio,
            "stop_loss_method": "fixed_percent",
        }
    except Exception:
        return None


def _try_save_config(**kwargs) -> bool:
    """尝试保存配置到 YAML"""
    try:
        from quant_trading.core.config import AppConfig
        config_path = Path("config.yaml")
        config = AppConfig.from_yaml(config_path) if config_path.exists() else AppConfig()

        # 更新配置
        if "initial_capital" in kwargs:
            config.backtest.initial_capital = kwargs["initial_capital"]
        if "commission_rate" in kwargs:
            config.broker.commission_rate = kwargs["commission_rate"]
        if "stamp_tax_rate" in kwargs:
            config.broker.stamp_tax_rate = kwargs["stamp_tax_rate"]
        if "slippage" in kwargs:
            config.broker.slippage = kwargs["slippage"]
        if "data_source" in kwargs:
            config.data_source.default_source = kwargs["data_source"]
        if "benchmark" in kwargs:
            config.backtest.benchmark = kwargs["benchmark"]
        if "risk_free_rate" in kwargs:
            config.backtest.risk_free_rate = kwargs["risk_free_rate"]

        config.to_yaml(config_path)
        return True
    except Exception:
        return False


def render() -> None:
    st.title("⚙️ 系统设置")

    settings = st.session_state.settings

    # 尝试从配置文件加载
    file_config = _try_load_config()

    tab_trade, tab_data, tab_risk, tab_cache, tab_about = st.tabs([
        "💰 交易参数", "📡 数据源", "🛡️ 风险参数", "💾 缓存管理", "ℹ️ 关于系统",
    ])

    # ==================================================================
    # 交易参数
    # ==================================================================
    with tab_trade:
        st.subheader("交易成本与规则")

        if file_config:
            st.caption("已从配置文件 config.yaml 加载默认值")

        c1, c2 = st.columns(2)
        with c1:
            initial_capital = st.number_input(
                "初始资金 (元)",
                min_value=10_000.0,
                max_value=100_000_000.0,
                value=float(file_config["initial_capital"] if file_config else settings.get("initial_capital", 1_000_000)),
                step=100_000.0,
                format="%.0f",
                key="set_capital",
            )
            commission_rate = st.number_input(
                "佣金率",
                min_value=0.0,
                max_value=0.01,
                value=float(file_config["commission_rate"] if file_config else settings.get("commission_rate", 0.0003)),
                step=0.0001,
                format="%.4f",
                key="set_commission",
                help="一般为万三(0.0003)",
            )
            min_commission = st.number_input(
                "最低佣金 (元)",
                min_value=0.0,
                max_value=50.0,
                value=float(file_config["min_commission"] if file_config else 5.0),
                step=1.0,
                key="set_min_comm",
            )
        with c2:
            stamp_tax = st.number_input(
                "印花税率 (仅卖出)",
                min_value=0.0,
                max_value=0.01,
                value=float(file_config["stamp_tax_rate"] if file_config else settings.get("stamp_tax_rate", 0.001)),
                step=0.0001,
                format="%.4f",
                key="set_stamp",
                help="当前为千分之一(0.001)",
            )
            slippage = st.number_input(
                "滑点",
                min_value=0.0,
                max_value=0.01,
                value=float(file_config["slippage"] if file_config else settings.get("slippage", 0.001)),
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

        # 交易成本预估
        with st.expander("📊 交易成本预估", expanded=False):
            est_amount = st.number_input("预估成交金额 (元)", value=100000.0,
                                          step=10000.0, format="%.0f", key="est_amount")
            buy_commission = max(est_amount * commission_rate, min_commission)
            sell_commission = max(est_amount * commission_rate, min_commission)
            sell_tax = est_amount * stamp_tax
            total_cost = buy_commission + sell_commission + sell_tax
            slippage_cost = est_amount * slippage * 2

            ec1, ec2, ec3, ec4 = st.columns(4)
            with ec1:
                st.metric("买入佣金", f"¥{buy_commission:.2f}")
            with ec2:
                st.metric("卖出佣金", f"¥{sell_commission:.2f}")
            with ec3:
                st.metric("印花税", f"¥{sell_tax:.2f}")
            with ec4:
                st.metric("总成本(含滑点)", f"¥{total_cost + slippage_cost:.2f}")

        if st.button("💾 保存交易参数", key="save_trade"):
            settings["initial_capital"] = initial_capital
            settings["commission_rate"] = commission_rate
            settings["stamp_tax_rate"] = stamp_tax
            settings["slippage"] = slippage
            st.session_state.settings = settings

            # 尝试保存到文件
            saved = _try_save_config(
                initial_capital=initial_capital,
                commission_rate=commission_rate,
                stamp_tax_rate=stamp_tax,
                slippage=slippage,
            )
            if saved:
                st.success("交易参数已保存! (含 config.yaml)")
            else:
                st.success("交易参数已保存到 session!")

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

        # 数据源状态检测
        st.subheader("📡 数据源状态")
        if st.button("🔍 检测数据源", key="check_source"):
            with st.spinner("正在检测..."):
                try:
                    from quant_trading.data import DataSourceRegistry
                    source = DataSourceRegistry.get(data_source)
                    st.success(f"✅ {data_source} 数据源可用: {source.get_name()}")

                    # 尝试获取测试数据
                    try:
                        from quant_trading.core.config import AppConfig
                        from quant_trading.data import DataManager
                        config = AppConfig()
                        dm = DataManager(config)
                        test_df = dm.fetch_daily("000001", "20250101", "20250301")
                        if not test_df.empty:
                            st.success(f"✅ 数据获取正常，测试获取 {len(test_df)} 条记录")
                        else:
                            st.warning("⚠️ 数据源连接正常但未获取到数据")
                    except Exception as e:
                        st.warning(f"⚠️ 数据获取测试失败: {e}")
                except Exception as e:
                    st.error(f"❌ 数据源不可用: {e}")

        st.markdown("---")

        st.subheader("缓存设置")
        cache_enabled = st.checkbox("启用数据缓存", value=True, key="set_cache")
        cache_dir = st.text_input("缓存目录", value="data_cache", key="set_cache_dir")

        if st.button("💾 保存数据源配置", key="save_data"):
            settings["data_source"] = data_source
            st.session_state.settings = settings
            saved = _try_save_config(data_source=data_source)
            if saved:
                st.success("数据源配置已保存! (含 config.yaml)")
            else:
                st.success("数据源配置已保存到 session!")

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
            saved = _try_save_config(
                benchmark=benchmark,
                risk_free_rate=risk_free_rate,
            )
            if saved:
                st.success("风险参数已保存! (含 config.yaml)")
            else:
                st.success("风险参数已保存到 session!")

    # ==================================================================
    # 缓存管理
    # ==================================================================
    with tab_cache:
        st.subheader("💾 缓存管理")

        # 尝试获取真实缓存信息
        cache_info = None
        try:
            from quant_trading.core.config import AppConfig
            from quant_trading.data import DataManager
            config = AppConfig()
            dm = DataManager(config)
            cache_info = dm.cache_info()
        except Exception:
            pass

        if cache_info:
            ci1, ci2, ci3 = st.columns(3)
            with ci1:
                st.metric("缓存状态", "启用" if cache_info["enabled"] else "禁用")
            with ci2:
                st.metric("缓存文件数", cache_info["file_count"])
            with ci3:
                st.metric("缓存大小", f"{cache_info['total_size_mb']:.1f} MB")
            st.caption(f"缓存目录: {cache_info['directory']}")
        else:
            st.info("缓存信息不可用（DataManager 未初始化）")

        st.markdown("---")

        col_cache1, col_cache2 = st.columns(2)
        with col_cache1:
            if st.button("🗑️ 清除全部缓存", type="secondary"):
                try:
                    from quant_trading.core.config import AppConfig
                    from quant_trading.data import DataManager
                    config = AppConfig()
                    dm = DataManager(config)
                    count = dm.clear_cache()
                    st.success(f"已清除 {count} 个缓存文件")
                except Exception as e:
                    st.warning(f"清除失败: {e}")

        with col_cache2:
            clear_symbol = st.text_input("指定股票代码清除缓存", placeholder="如 000001")
            if clear_symbol and st.button("🗑️ 清除指定缓存"):
                try:
                    from quant_trading.core.config import AppConfig
                    from quant_trading.data import DataManager
                    config = AppConfig()
                    dm = DataManager(config)
                    count = dm.clear_cache(clear_symbol)
                    st.success(f"已清除 {clear_symbol} 的 {count} 个缓存文件")
                except Exception as e:
                    st.warning(f"清除失败: {e}")

    # ==================================================================
    # 关于系统
    # ==================================================================
    with tab_about:
        st.subheader("关于量化交易系统")

        st.markdown("""
        ### 📈 A股量化交易系统 v0.3.0

        **功能特性:**
        - 🔌 可插拔数据源 (AKShare / Tushare)
        - 📊 多策略回测引擎（支持真实/模拟双模式）
        - 🛡️ 全面风险管理（VaR、回撤、仓位控制）
        - 📈 交互式可视化 Dashboard
        - 💼 多股票组合管理与回测
        - 📥 数据导出 (CSV)

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

        # 模块状态检测
        st.markdown("---")
        st.subheader("📦 模块状态")
        modules = [
            ("quant_trading.core", "核心模块"),
            ("quant_trading.data", "数据模块"),
            ("quant_trading.strategy", "策略模块"),
            ("quant_trading.backtest", "回测模块"),
            ("quant_trading.risk", "风控模块"),
            ("quant_trading.indicators", "指标模块"),
        ]
        for mod_name, display_name in modules:
            try:
                __import__(mod_name)
                st.markdown(f"✅ **{display_name}** ({mod_name}) - 可用")
            except ImportError as e:
                st.markdown(f"❌ **{display_name}** ({mod_name}) - 不可用: {e}")

        # 已注册策略
        st.markdown("---")
        st.subheader("📋 已注册策略")
        try:
            from quant_trading.strategy import StrategyRegistry
            strategies = StrategyRegistry.list_strategies()
            if strategies:
                for s in strategies:
                    st.markdown(f"  - `{s}`")
            else:
                st.info("无已注册策略")
        except Exception:
            st.info("策略注册表不可用")

        st.markdown("---")
        st.caption("作者: zey413 | 协议: MIT")
