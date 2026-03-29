"""回测绩效指标计算模块

提供完整的回测绩效评估，包括：
- 收益率指标：总收益率、年化收益率
- 风险指标：最大回撤、波动率
- 风险调整收益：夏普比率、Sortino比率、Calmar比率
- 交易统计：胜率、盈亏比、平均盈亏
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from quant_trading.core.config import BacktestConfig
from quant_trading.core.models import TradeRecord

# A股年交易日数
TRADING_DAYS_PER_YEAR = 244


@dataclass
class PerformanceResult:
    """回测绩效结果"""

    # Returns
    total_return: float  # 总收益率
    annualized_return: float  # 年化收益率

    # Risk
    max_drawdown: float  # 最大回撤
    max_drawdown_duration: int  # 最大回撤持续天数（交易日）
    volatility: float  # 年化波动率

    # Risk-adjusted
    sharpe_ratio: float  # 夏普比率
    sortino_ratio: float  # Sortino比率
    calmar_ratio: float  # Calmar比率

    # Trading
    total_trades: int  # 总交易次数（平仓次数）
    win_rate: float  # 胜率
    profit_loss_ratio: float  # 盈亏比
    avg_win: float  # 平均盈利
    avg_loss: float  # 平均亏损

    # Series
    equity_curve: pd.Series  # 净值曲线
    drawdown_series: pd.Series  # 回撤序列

    def summary(self) -> str:
        """格式化输出绩效摘要"""
        lines = [
            "=" * 60,
            "                  回测绩效报告",
            "=" * 60,
            "",
            "【收益指标】",
            f"  总收益率:          {self.total_return:>10.2%}",
            f"  年化收益率:        {self.annualized_return:>10.2%}",
            "",
            "【风险指标】",
            f"  最大回撤:          {self.max_drawdown:>10.2%}",
            f"  最大回撤持续天数:  {self.max_drawdown_duration:>10d}",
            f"  年化波动率:        {self.volatility:>10.2%}",
            "",
            "【风险调整收益】",
            f"  夏普比率:          {self.sharpe_ratio:>10.4f}",
            f"  Sortino比率:       {self.sortino_ratio:>10.4f}",
            f"  Calmar比率:        {self.calmar_ratio:>10.4f}",
            "",
            "【交易统计】",
            f"  总交易次数:        {self.total_trades:>10d}",
            f"  胜率:              {self.win_rate:>10.2%}",
            f"  盈亏比:            {self.profit_loss_ratio:>10.4f}",
            f"  平均盈利:          {self.avg_win:>10.2f}",
            f"  平均亏损:          {self.avg_loss:>10.2f}",
            "",
            "=" * 60,
        ]
        return "\n".join(lines)


class PerformanceMetrics:
    """绩效指标计算器

    基于净值曲线和交易记录计算完整的回测绩效指标。

    所有年化计算使用A股年交易日数（244天）。
    """

    @staticmethod
    def calculate(
        equity_curve: pd.Series,
        trades: list[TradeRecord],
        config: BacktestConfig,
    ) -> PerformanceResult:
        """计算完整的绩效指标

        Args:
            equity_curve: 净值曲线，index为日期，值为组合总价值
            trades: 平仓交易记录列表
            config: 回测配置（含初始资金、无风险利率等）

        Returns:
            PerformanceResult: 完整的绩效结果
        """
        if equity_curve.empty:
            return PerformanceMetrics._empty_result()

        # --- 收益指标 ---
        initial_value = equity_curve.iloc[0]
        final_value = equity_curve.iloc[-1]
        total_return = (final_value - initial_value) / initial_value

        trading_days = len(equity_curve)
        if trading_days > 1:
            annualized_return = (
                (final_value / initial_value)
                ** (TRADING_DAYS_PER_YEAR / trading_days)
                - 1
            )
        else:
            annualized_return = 0.0

        # --- 日收益率序列 ---
        daily_returns = equity_curve.pct_change().dropna()

        # --- 风险指标 ---
        volatility = PerformanceMetrics._calc_volatility(daily_returns)
        max_drawdown, max_dd_duration, drawdown_series = (
            PerformanceMetrics._calc_drawdown(equity_curve)
        )

        # --- 风险调整收益 ---
        risk_free_rate = config.risk_free_rate
        sharpe = PerformanceMetrics._calc_sharpe(
            annualized_return, volatility, risk_free_rate
        )
        sortino = PerformanceMetrics._calc_sortino(
            daily_returns, annualized_return, risk_free_rate
        )
        calmar = PerformanceMetrics._calc_calmar(annualized_return, max_drawdown)

        # --- 交易统计 ---
        trade_stats = PerformanceMetrics._calc_trade_stats(trades)

        return PerformanceResult(
            total_return=total_return,
            annualized_return=annualized_return,
            max_drawdown=max_drawdown,
            max_drawdown_duration=max_dd_duration,
            volatility=volatility,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            calmar_ratio=calmar,
            total_trades=trade_stats["total_trades"],
            win_rate=trade_stats["win_rate"],
            profit_loss_ratio=trade_stats["profit_loss_ratio"],
            avg_win=trade_stats["avg_win"],
            avg_loss=trade_stats["avg_loss"],
            equity_curve=equity_curve,
            drawdown_series=drawdown_series,
        )

    @staticmethod
    def _calc_volatility(daily_returns: pd.Series) -> float:
        """计算年化波动率

        年化波动率 = 日收益率标准差 * sqrt(244)
        """
        if daily_returns.empty or len(daily_returns) < 2:
            return 0.0
        return float(daily_returns.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR))

    @staticmethod
    def _calc_drawdown(
        equity_curve: pd.Series,
    ) -> tuple[float, int, pd.Series]:
        """计算最大回撤、最大回撤持续天数、回撤序列

        回撤 = (前期最高点 - 当前值) / 前期最高点

        Returns:
            (max_drawdown, max_drawdown_duration, drawdown_series)
        """
        if equity_curve.empty:
            return 0.0, 0, pd.Series(dtype=float)

        # 累计最高点
        running_max = equity_curve.cummax()
        # 回撤序列（正值表示回撤幅度）
        drawdown_series = (running_max - equity_curve) / running_max
        max_drawdown = float(drawdown_series.max())

        # 计算最大回撤持续天数
        # 找到回撤为0的点（即创新高的点），计算相邻创新高之间的最大间隔
        max_dd_duration = PerformanceMetrics._calc_max_drawdown_duration(
            drawdown_series
        )

        return max_drawdown, max_dd_duration, drawdown_series

    @staticmethod
    def _calc_max_drawdown_duration(drawdown_series: pd.Series) -> int:
        """计算最大回撤持续天数

        从进入回撤（离开0）到恢复（回到0）的最长交易日数。
        如果到序列末尾仍未恢复，则计算到末尾的天数。
        """
        if drawdown_series.empty:
            return 0

        # 标记是否处于回撤中（回撤>0）
        is_drawdown = drawdown_series > 1e-10  # 使用小容差避免浮点精度问题

        max_duration = 0
        current_duration = 0

        for in_drawdown in is_drawdown:
            if in_drawdown:
                current_duration += 1
                max_duration = max(max_duration, current_duration)
            else:
                current_duration = 0

        return max_duration

    @staticmethod
    def _calc_sharpe(
        annualized_return: float,
        volatility: float,
        risk_free_rate: float,
    ) -> float:
        """计算夏普比率

        Sharpe = (年化收益率 - 无风险利率) / 年化波动率
        """
        if volatility == 0 or np.isnan(volatility):
            return 0.0
        return (annualized_return - risk_free_rate) / volatility

    @staticmethod
    def _calc_sortino(
        daily_returns: pd.Series,
        annualized_return: float,
        risk_free_rate: float,
    ) -> float:
        """计算Sortino比率

        Sortino = (年化收益率 - 无风险利率) / 下行标准差（年化）

        下行标准差只考虑负收益的波动。
        """
        if daily_returns.empty or len(daily_returns) < 2:
            return 0.0

        # 日无风险收益率
        daily_rf = (1 + risk_free_rate) ** (1 / TRADING_DAYS_PER_YEAR) - 1
        # 下行偏差：只取低于无风险利率的部分
        downside = daily_returns[daily_returns < daily_rf] - daily_rf

        if downside.empty or len(downside) < 2:
            # 没有下行日，Sortino为正无穷大，返回一个大数
            if annualized_return > risk_free_rate:
                return 99.99
            return 0.0

        # 下行标准差使用全部样本数做分母（因为非下行日贡献0）
        downside_squared = downside**2
        downside_deviation = float(
            np.sqrt(downside_squared.sum() / len(daily_returns))
            * np.sqrt(TRADING_DAYS_PER_YEAR)
        )

        if downside_deviation == 0:
            return 0.0

        return (annualized_return - risk_free_rate) / downside_deviation

    @staticmethod
    def _calc_calmar(annualized_return: float, max_drawdown: float) -> float:
        """计算Calmar比率

        Calmar = 年化收益率 / 最大回撤
        """
        if max_drawdown == 0:
            if annualized_return > 0:
                return 99.99
            return 0.0
        return annualized_return / max_drawdown

    @staticmethod
    def _calc_trade_stats(trades: list[TradeRecord]) -> dict:
        """计算交易统计指标

        只统计卖出（平仓）交易的盈亏。

        Returns:
            dict with keys: total_trades, win_rate, profit_loss_ratio,
                           avg_win, avg_loss
        """
        # 筛选平仓交易（卖出的有pnl的记录）
        closed_trades = [t for t in trades if t.side == "sell"]
        total_trades = len(closed_trades)

        if total_trades == 0:
            return {
                "total_trades": 0,
                "win_rate": 0.0,
                "profit_loss_ratio": 0.0,
                "avg_win": 0.0,
                "avg_loss": 0.0,
            }

        pnl_list = [t.pnl for t in closed_trades]
        wins = [p for p in pnl_list if p > 0]
        losses = [p for p in pnl_list if p < 0]

        win_count = len(wins)
        win_rate = win_count / total_trades

        avg_win = float(np.mean(wins)) if wins else 0.0
        avg_loss = float(np.mean(losses)) if losses else 0.0

        # 盈亏比 = 平均盈利 / |平均亏损|
        if avg_loss != 0:
            profit_loss_ratio = abs(avg_win / avg_loss)
        else:
            profit_loss_ratio = 99.99 if avg_win > 0 else 0.0

        return {
            "total_trades": total_trades,
            "win_rate": win_rate,
            "profit_loss_ratio": profit_loss_ratio,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
        }

    @staticmethod
    def _empty_result() -> PerformanceResult:
        """返回空的绩效结果（无数据时使用）"""
        return PerformanceResult(
            total_return=0.0,
            annualized_return=0.0,
            max_drawdown=0.0,
            max_drawdown_duration=0,
            volatility=0.0,
            sharpe_ratio=0.0,
            sortino_ratio=0.0,
            calmar_ratio=0.0,
            total_trades=0,
            win_rate=0.0,
            profit_loss_ratio=0.0,
            avg_win=0.0,
            avg_loss=0.0,
            equity_curve=pd.Series(dtype=float),
            drawdown_series=pd.Series(dtype=float),
        )
