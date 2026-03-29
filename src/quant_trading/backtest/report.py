"""回测报告生成器 - 生成完整的文本/CSV回测报告

提供:
- 文本格式的完整回测报告（含策略信息、绩效指标、月度统计、交易明细、风险分析）
- 交易记录 CSV 导出
- 净值曲线 CSV 导出
"""

from __future__ import annotations

import csv
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from quant_trading.backtest.metrics import PerformanceResult
from quant_trading.core.config import AppConfig
from quant_trading.core.models import TradeRecord

logger = logging.getLogger(__name__)


class ReportGenerator:
    """回测报告生成器

    生成包含以下内容的完整回测报告:
    - 策略信息和参数
    - 绩效指标摘要 (表格)
    - 月度/年度收益统计
    - 交易明细列表 (Top 5 盈利/亏损)
    - 风险分析 (回撤期间)
    """

    @staticmethod
    def generate_text_report(
        result: PerformanceResult,
        strategy_name: str = "",
        symbol: str = "",
        config: Optional[AppConfig] = None,
        trades: Optional[list[TradeRecord]] = None,
    ) -> str:
        """生成文本格式的完整报告

        Args:
            result: 回测绩效结果
            strategy_name: 策略名称
            symbol: 股票代码
            config: 应用配置（用于显示资金信息等）
            trades: 交易记录列表

        Returns:
            str: 格式化的文本报告
        """
        lines: list[str] = []
        sep = "=" * 70
        thin_sep = "-" * 70

        # ---- 1. 报告头部 ----
        lines.append(sep)
        lines.append("                      回测分析报告")
        lines.append(sep)
        lines.append(f"  生成时间:    {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        if strategy_name:
            lines.append(f"  策略名称:    {strategy_name}")
        if symbol:
            lines.append(f"  股票代码:    {symbol}")

        # 日期范围（从净值曲线推断）
        if not result.equity_curve.empty:
            eq_index = result.equity_curve.index
            start_str = str(eq_index[0])[:10]
            end_str = str(eq_index[-1])[:10]
            lines.append(f"  回测区间:    {start_str} ~ {end_str}")
            lines.append(f"  交易天数:    {len(result.equity_curve)}")

        lines.append("")

        # ---- 2. 资金信息 ----
        if config is not None:
            initial_capital = config.backtest.initial_capital
            if not result.equity_curve.empty:
                final_value = result.equity_curve.iloc[-1]
                net_profit = final_value - initial_capital
            else:
                final_value = initial_capital
                net_profit = 0.0

            lines.append(thin_sep)
            lines.append("【资金概况】")
            lines.append(thin_sep)
            lines.append(f"  初始资金:          {initial_capital:>16,.2f} 元")
            lines.append(f"  期末资产:          {final_value:>16,.2f} 元")
            profit_sign = "+" if net_profit >= 0 else ""
            lines.append(
                f"  净利润:            {profit_sign}{net_profit:>15,.2f} 元"
            )
            lines.append("")

        # ---- 3. 绩效指标表 ----
        lines.append(thin_sep)
        lines.append("【绩效指标】")
        lines.append(thin_sep)
        lines.append(f"  总收益率:          {result.total_return:>12.2%}")
        lines.append(f"  年化收益率:        {result.annualized_return:>12.2%}")
        lines.append(f"  最大回撤:          {result.max_drawdown:>12.2%}")
        lines.append(f"  最大回撤持续天数:  {result.max_drawdown_duration:>12d}")
        lines.append(f"  年化波动率:        {result.volatility:>12.2%}")
        lines.append(f"  夏普比率:          {result.sharpe_ratio:>12.4f}")
        lines.append(f"  Sortino比率:       {result.sortino_ratio:>12.4f}")
        lines.append(f"  Calmar比率:        {result.calmar_ratio:>12.4f}")
        lines.append("")

        # ---- 4. 交易统计 ----
        lines.append(thin_sep)
        lines.append("【交易统计】")
        lines.append(thin_sep)
        lines.append(f"  总交易次数:        {result.total_trades:>12d}")
        lines.append(f"  胜率:              {result.win_rate:>12.2%}")
        lines.append(f"  盈亏比:            {result.profit_loss_ratio:>12.4f}")
        lines.append(f"  平均盈利:          {result.avg_win:>12.2f} 元")
        lines.append(f"  平均亏损:          {result.avg_loss:>12.2f} 元")
        lines.append("")

        # ---- 5. 月度收益表 ----
        monthly_table = ReportGenerator._build_monthly_returns_table(
            result.equity_curve
        )
        if monthly_table:
            lines.append(thin_sep)
            lines.append("【月度收益率】")
            lines.append(thin_sep)
            lines.extend(monthly_table)
            lines.append("")

        # ---- 6. Top 5 盈利/亏损交易 ----
        if trades:
            closed = [t for t in trades if t.side == "sell"]
            if closed:
                lines.append(thin_sep)
                lines.append("【交易明细 - Top 5 盈利】")
                lines.append(thin_sep)
                sorted_by_pnl = sorted(closed, key=lambda t: t.pnl, reverse=True)
                top_wins = sorted_by_pnl[:5]
                for t in top_wins:
                    lines.append(
                        f"  {t.date}  {t.symbol}  "
                        f"数量: {t.quantity:>6d}  "
                        f"价格: {t.price:>8.2f}  "
                        f"盈亏: {t.pnl:>+10.2f}"
                    )
                lines.append("")

                lines.append(thin_sep)
                lines.append("【交易明细 - Top 5 亏损】")
                lines.append(thin_sep)
                top_losses = sorted_by_pnl[-5:][::-1]
                for t in top_losses:
                    lines.append(
                        f"  {t.date}  {t.symbol}  "
                        f"数量: {t.quantity:>6d}  "
                        f"价格: {t.price:>8.2f}  "
                        f"盈亏: {t.pnl:>+10.2f}"
                    )
                lines.append("")

        # ---- 7. 风险分析（回撤期间） ----
        drawdown_periods = ReportGenerator._find_drawdown_periods(
            result.drawdown_series
        )
        if drawdown_periods:
            lines.append(thin_sep)
            lines.append("【风险分析 - 主要回撤期间】")
            lines.append(thin_sep)
            for i, period in enumerate(drawdown_periods[:5], 1):
                lines.append(
                    f"  {i}. {period['start']} ~ {period['end']}  "
                    f"持续 {period['duration']:>3d} 天  "
                    f"最大回撤: {period['max_drawdown']:>8.2%}"
                )
            lines.append("")

        lines.append(sep)
        lines.append("                      报告结束")
        lines.append(sep)

        return "\n".join(lines)

    @staticmethod
    def generate_csv_trades(trades: list[TradeRecord], filepath: str) -> None:
        """导出交易记录为CSV

        Args:
            trades: 交易记录列表
            filepath: 输出文件路径
        """
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        headers = [
            "trade_id",
            "date",
            "symbol",
            "side",
            "quantity",
            "price",
            "commission",
            "tax",
            "pnl",
            "strategy_name",
        ]

        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            for t in trades:
                writer.writerow([
                    t.trade_id,
                    str(t.date),
                    t.symbol,
                    t.side,
                    t.quantity,
                    f"{t.price:.4f}",
                    f"{t.commission:.4f}",
                    f"{t.tax:.4f}",
                    f"{t.pnl:.4f}",
                    t.strategy_name,
                ])

        logger.info("交易记录已导出至: %s (%d 条)", filepath, len(trades))

    @staticmethod
    def generate_equity_csv(equity_curve: pd.Series, filepath: str) -> None:
        """导出净值曲线为CSV

        Args:
            equity_curve: 净值曲线 Series（index为日期，值为组合总价值）
            filepath: 输出文件路径
        """
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        df = pd.DataFrame({
            "date": equity_curve.index,
            "equity": equity_curve.values,
        })

        # 计算日收益率
        df["daily_return"] = equity_curve.pct_change().values

        # 计算累计收益率
        if len(equity_curve) > 0:
            df["cumulative_return"] = (
                equity_curve.values / equity_curve.iloc[0] - 1
            )
        else:
            df["cumulative_return"] = 0.0

        df.to_csv(path, index=False, encoding="utf-8-sig")
        logger.info("净值曲线已导出至: %s (%d 条)", filepath, len(df))

    # ------------------------------------------------------------------ #
    #  内部辅助方法
    # ------------------------------------------------------------------ #

    @staticmethod
    def _build_monthly_returns_table(equity_curve: pd.Series) -> list[str]:
        """构建月度收益率文本表格（年 x 月网格）

        Returns:
            list[str]: 表格行列表，如果数据不足则返回空列表
        """
        if equity_curve.empty or len(equity_curve) < 2:
            return []

        eq = equity_curve.copy()
        if not isinstance(eq.index, pd.DatetimeIndex):
            eq.index = pd.to_datetime(eq.index)

        daily_returns = eq.pct_change().dropna()
        if daily_returns.empty:
            return []

        # 按 (年, 月) 分组计算月度收益率
        monthly = daily_returns.groupby(
            [daily_returns.index.year, daily_returns.index.month]
        ).apply(lambda x: (1 + x).prod() - 1)

        if not isinstance(monthly.index, pd.MultiIndex):
            return []

        years = sorted(monthly.index.get_level_values(0).unique())
        months = list(range(1, 13))

        # 构建矩阵
        matrix: dict[int, dict[int, float | None]] = {}
        for year in years:
            matrix[year] = {}
            for month in months:
                if (year, month) in monthly.index:
                    matrix[year][month] = float(monthly.loc[(year, month)])
                else:
                    matrix[year][month] = None

        # 生成文本表格
        lines: list[str] = []
        # 表头
        header = "  年份  " + " ".join(f"{m:>6d}月" for m in months) + "    年度"
        lines.append(header)
        lines.append("  " + "-" * (len(header) - 2))

        for year in years:
            cols = []
            year_return = 0.0
            has_data = False
            for month in months:
                val = matrix[year][month]
                if val is not None:
                    cols.append(f"{val:>7.1%}")
                    year_return = (1 + year_return) * (1 + val) - 1
                    has_data = True
                else:
                    cols.append("      -")
            year_str = f"{year_return:>7.1%}" if has_data else "      -"
            lines.append(f"  {year}  " + " ".join(cols) + f"  {year_str}")

        return lines

    @staticmethod
    def _find_drawdown_periods(
        drawdown_series: pd.Series,
    ) -> list[dict]:
        """识别主要回撤期间

        Returns:
            按最大回撤降序排列的回撤期间列表，每项包含:
            - start: 回撤开始日期
            - end: 回撤结束日期
            - duration: 持续天数
            - max_drawdown: 期间最大回撤幅度
        """
        if drawdown_series.empty:
            return []

        dd = drawdown_series.copy()
        if not isinstance(dd.index, pd.DatetimeIndex):
            try:
                dd.index = pd.to_datetime(dd.index)
            except Exception:
                return []

        periods: list[dict] = []
        in_drawdown = False
        start_idx = None
        max_dd_in_period = 0.0

        for i, (dt, val) in enumerate(dd.items()):
            if val > 1e-10:  # 处于回撤中
                if not in_drawdown:
                    in_drawdown = True
                    start_idx = dt
                    max_dd_in_period = val
                else:
                    max_dd_in_period = max(max_dd_in_period, val)
            else:
                if in_drawdown:
                    periods.append({
                        "start": str(start_idx)[:10],
                        "end": str(dt)[:10],
                        "duration": i - list(dd.index).index(start_idx),
                        "max_drawdown": max_dd_in_period,
                    })
                    in_drawdown = False

        # 如果序列结束时仍在回撤中
        if in_drawdown and start_idx is not None:
            periods.append({
                "start": str(start_idx)[:10],
                "end": str(dd.index[-1])[:10],
                "duration": len(dd) - list(dd.index).index(start_idx),
                "max_drawdown": max_dd_in_period,
            })

        # 按最大回撤降序排列
        periods.sort(key=lambda p: p["max_drawdown"], reverse=True)

        return periods
