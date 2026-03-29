"""组合管理 - 持仓饼图、明细表、净值历史、交易记录导出

所有数据使用模拟数据。
"""

from __future__ import annotations

import io
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import streamlit as st

from quant_trading.web.charts import equity_curve, pie_chart
from quant_trading.web.components import metric_card, trade_table


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


# ---------------------------------------------------------------------------
# 渲染入口
# ---------------------------------------------------------------------------

def render() -> None:
    st.title("💼 组合管理")

    # ====== 组合概览 ======
    holdings = _mock_holdings()
    total_mv = holdings["市值"].sum()
    total_pnl = holdings["浮动盈亏"].sum()
    cash = 400_500.0
    total_assets = total_mv + cash

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

    # ====== 持仓分布 + 行业分布 ======
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
        # 加入现金
        industry_labels = industry_group.index.tolist() + ["现金"]
        industry_values = industry_group.values.tolist() + [cash]
        fig_ind = pie_chart(
            labels=industry_labels,
            values=industry_values,
            title="资产配置分布",
        )
        st.plotly_chart(fig_ind, use_container_width=True)

    # ====== 持仓明细 ======
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

    # ====== 组合净值历史 ======
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

    # ====== 交易记录 ======
    st.subheader("📝 交易记录")
    trade_history = _mock_trade_history()

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

    # ====== 导出按钮 ======
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
        # 导出组合净值
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
