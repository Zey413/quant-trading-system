"""量化交易系统 - Streamlit Web Dashboard 主入口

多页面应用，侧边栏导航，全局 session_state 管理。
启动方式: streamlit run src/quant_trading/web/app.py
"""

from __future__ import annotations

import streamlit as st


# ---------------------------------------------------------------------------
# 页面配置（必须是第一个 Streamlit 调用）
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="量化交易系统",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# 全局 session_state 初始化
# ---------------------------------------------------------------------------

_DEFAULT_STATE: dict = {
    # 当前选中页面
    "current_page": "首页概览",
    # 自选股列表
    "watchlist": ["000001", "600519", "000858", "601318", "600036"],
    # 回测结果缓存
    "backtest_results": {},
    # 组合持仓（模拟）
    "portfolio_positions": {},
    # 系统设置
    "settings": {
        "initial_capital": 1_000_000.0,
        "commission_rate": 0.0003,
        "stamp_tax_rate": 0.001,
        "slippage": 0.001,
        "benchmark": "000300",
        "risk_free_rate": 0.025,
        "data_source": "akshare",
    },
    # 风控规则
    "risk_rules": {
        "max_position_pct": 30,
        "max_total_position_pct": 80,
        "stop_loss_pct": 5,
        "take_profit_pct": 15,
        "max_drawdown_limit": 15,
        "var_limit": 50000.0,
    },
}


def _init_session_state() -> None:
    """初始化所有 session_state 默认值"""
    for key, default in _DEFAULT_STATE.items():
        if key not in st.session_state:
            st.session_state[key] = default


_init_session_state()


# ---------------------------------------------------------------------------
# 页面注册表
# ---------------------------------------------------------------------------

PAGE_MAP: dict[str, dict] = {
    "首页概览": {"icon": "🏠", "module": "dashboard"},
    "行情中心": {"icon": "📊", "module": "market"},
    "策略回测": {"icon": "🔬", "module": "backtest"},
    "组合管理": {"icon": "💼", "module": "portfolio"},
    "风险监控": {"icon": "🛡️", "module": "risk"},
    "系统设置": {"icon": "⚙️", "module": "settings"},
}


# ---------------------------------------------------------------------------
# 侧边栏导航
# ---------------------------------------------------------------------------

def _render_sidebar() -> str:
    """渲染侧边栏，返回当前选中的页面名称"""
    with st.sidebar:
        st.markdown(
            "<h1 style='text-align:center;'>📈 量化交易系统</h1>",
            unsafe_allow_html=True,
        )
        st.markdown("---")

        for page_name, meta in PAGE_MAP.items():
            label = f"{meta['icon']}  {page_name}"
            if st.button(
                label,
                key=f"nav_{page_name}",
                use_container_width=True,
                type="primary" if st.session_state.current_page == page_name else "secondary",
            ):
                st.session_state.current_page = page_name
                st.rerun()

        st.markdown("---")

        # 快捷信息面板
        settings = st.session_state.settings
        st.markdown("**快捷信息**")
        st.caption(f"初始资金: ¥{settings['initial_capital']:,.0f}")
        st.caption(f"数据源: {settings['data_source']}")
        st.caption(f"自选股: {len(st.session_state.watchlist)} 只")

        st.markdown("---")
        st.caption("v0.3.0  |  A股量化交易系统")

    return st.session_state.current_page


# ---------------------------------------------------------------------------
# 页面路由
# ---------------------------------------------------------------------------

def _route(page_name: str) -> None:
    """根据页面名称渲染对应页面"""
    module_name = PAGE_MAP[page_name]["module"]

    if module_name == "dashboard":
        from quant_trading.web.pages.dashboard import render
    elif module_name == "market":
        from quant_trading.web.pages.market import render
    elif module_name == "backtest":
        from quant_trading.web.pages.backtest import render
    elif module_name == "portfolio":
        from quant_trading.web.pages.portfolio import render
    elif module_name == "risk":
        from quant_trading.web.pages.risk import render
    elif module_name == "settings":
        from quant_trading.web.pages.settings import render
    else:
        st.error(f"未知页面: {page_name}")
        return

    render()


# ---------------------------------------------------------------------------
# 自定义 CSS
# ---------------------------------------------------------------------------

def _inject_css() -> None:
    st.markdown("""
    <style>
    /* 指标卡片 */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 12px;
        padding: 20px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        margin-bottom: 10px;
    }
    .metric-card .value { font-size: 28px; font-weight: bold; }
    .metric-card .label { font-size: 14px; opacity: 0.85; margin-top: 5px; }
    .metric-card .delta { font-size: 14px; margin-top: 3px; }
    .delta-up { color: #ff4444; }
    .delta-down { color: #00c851; }
    /* 红涨绿跌（A股配色） */
    .price-up { color: #ff4444; }
    .price-down { color: #00c851; }
    /* 表格优化 */
    .stDataFrame { border-radius: 8px; }
    /* 标签页美化 */
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 8px 16px;
    }
    /* 信息框 */
    .info-box {
        background: #f0f2f6;
        border-left: 4px solid #4169E1;
        padding: 12px 16px;
        border-radius: 0 8px 8px 0;
        margin: 8px 0;
    }
    </style>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

def main() -> None:
    _inject_css()
    current_page = _render_sidebar()
    _route(current_page)


if __name__ == "__main__":
    main()
