"""信号质量分析器 - 评估策略信号的有效性

提供独立于完整回测的信号质量评估工具，包括：
- 买卖信号的前瞻收益分析
- 信号胜率与频率统计
- 连续盈亏分析
- 信号分布可视化
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class SignalAnalysisResult:
    """信号分析结果"""

    total_signals: int  # 信号总数
    buy_signals: int  # 买入信号数
    sell_signals: int  # 卖出信号数
    avg_return_after_buy: float  # 买入信号后N日平均收益
    avg_return_after_sell: float  # 卖出信号后N日平均收益
    buy_win_rate: float  # 买入信号正确率
    sell_win_rate: float  # 卖出信号正确率
    signal_frequency: float  # 信号频率 (signals per trading day)
    avg_holding_period: float  # 平均持仓天数
    consecutive_wins: int  # 最大连胜
    consecutive_losses: int  # 最大连亏

    def summary(self) -> str:
        """格式化输出信号分析摘要"""
        lines = [
            "=" * 60,
            "                  信号质量分析报告",
            "=" * 60,
            "",
            "【信号统计】",
            f"  信号总数:          {self.total_signals:>10d}",
            f"  买入信号数:        {self.buy_signals:>10d}",
            f"  卖出信号数:        {self.sell_signals:>10d}",
            f"  信号频率:          {self.signal_frequency:>10.4f}",
            "",
            "【收益分析】",
            f"  买入后平均收益:    {self.avg_return_after_buy:>10.4%}",
            f"  卖出后平均收益:    {self.avg_return_after_sell:>10.4%}",
            "",
            "【胜率分析】",
            f"  买入信号正确率:    {self.buy_win_rate:>10.2%}",
            f"  卖出信号正确率:    {self.sell_win_rate:>10.2%}",
            "",
            "【持仓与连续性】",
            f"  平均持仓天数:      {self.avg_holding_period:>10.1f}",
            f"  最大连胜:          {self.consecutive_wins:>10d}",
            f"  最大连亏:          {self.consecutive_losses:>10d}",
            "",
            "=" * 60,
        ]
        return "\n".join(lines)


class SignalAnalyzer:
    """信号质量分析器

    分析策略生成的信号质量，评估信号的预测能力。

    该分析器不依赖完整回测引擎，仅通过信号与后续价格变动之间的
    关系来评估信号有效性。

    信号列约定:
        - ``signal`` 列: 1 = 买入, -1 = 卖出, 0 = 无信号
        - ``close`` 列: 收盘价

    Example::

        analyzer = SignalAnalyzer()
        signals_df = strategy.generate_signals(data)
        result = analyzer.analyze(signals_df, forward_periods=[1, 3, 5, 10])
    """

    _DEFAULT_FORWARD_PERIODS = [1, 5, 10]

    def analyze(
        self,
        data: pd.DataFrame,
        forward_periods: list[int] | None = None,
        signal_column: str = "signal",
    ) -> SignalAnalysisResult:
        """分析信号质量

        Parameters
        ----------
        data : pd.DataFrame
            至少包含 ``close`` 和 ``signal`` 列的 DataFrame。
            signal: 1 = 买入, -1 = 卖出, 0 = 无信号。
        forward_periods : list[int], optional
            前瞻周期列表，用于计算信号发出后的收益。
            默认 [1, 5, 10]，胜率和平均收益使用列表中第一个周期。
        signal_column : str
            信号列名，默认 ``"signal"``。

        Returns
        -------
        SignalAnalysisResult
        """
        self._validate_data(data, signal_column)

        if forward_periods is None:
            forward_periods = self._DEFAULT_FORWARD_PERIODS.copy()
        if not forward_periods:
            forward_periods = [1]

        # 使用第一个前瞻周期作为主要评估周期
        primary_period = forward_periods[0]

        signals = data[signal_column]
        close = data["close"]

        buy_mask = signals == 1
        sell_mask = signals == -1

        buy_signals = int(buy_mask.sum())
        sell_signals = int(sell_mask.sum())
        total_signals = buy_signals + sell_signals
        total_bars = len(data)

        # --- 信号频率 ---
        signal_frequency = total_signals / total_bars if total_bars > 0 else 0.0

        # --- 前瞻收益 ---
        forward_return = close.pct_change(periods=primary_period).shift(
            -primary_period
        )

        # 买入信号分析
        buy_returns = forward_return[buy_mask].dropna()
        avg_return_after_buy = (
            float(buy_returns.mean()) if len(buy_returns) > 0 else 0.0
        )
        buy_win_rate = (
            float((buy_returns > 0).sum() / len(buy_returns))
            if len(buy_returns) > 0
            else 0.0
        )

        # 卖出信号分析（卖出后价格下跌即"正确"）
        sell_returns = forward_return[sell_mask].dropna()
        avg_return_after_sell = (
            float(sell_returns.mean()) if len(sell_returns) > 0 else 0.0
        )
        sell_win_rate = (
            float((sell_returns < 0).sum() / len(sell_returns))
            if len(sell_returns) > 0
            else 0.0
        )

        # --- 平均持仓天数 ---
        avg_holding_period = self._calc_avg_holding_period(signals)

        # --- 连续盈亏 ---
        consecutive_wins, consecutive_losses = self._calc_consecutive_streaks(
            data, signal_column, primary_period
        )

        return SignalAnalysisResult(
            total_signals=total_signals,
            buy_signals=buy_signals,
            sell_signals=sell_signals,
            avg_return_after_buy=avg_return_after_buy,
            avg_return_after_sell=avg_return_after_sell,
            buy_win_rate=buy_win_rate,
            sell_win_rate=sell_win_rate,
            signal_frequency=signal_frequency,
            avg_holding_period=avg_holding_period,
            consecutive_wins=consecutive_wins,
            consecutive_losses=consecutive_losses,
        )

    def plot_signal_distribution(
        self,
        data: pd.DataFrame,
        save_path: str | None = None,
        signal_column: str = "signal",
    ) -> None:
        """绘制信号分布图 - 买卖信号的时间分布和收益分布

        Parameters
        ----------
        data : pd.DataFrame
            至少包含 ``close`` 和 ``signal`` 列。
        save_path : str, optional
            图片保存路径，为 None 时调用 ``plt.show()``。
        signal_column : str
            信号列名，默认 ``"signal"``。
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:
            raise ImportError(
                "绘图功能需要 matplotlib，请先安装: pip install matplotlib"
            ) from exc

        self._validate_data(data, signal_column)

        fig, axes = plt.subplots(2, 1, figsize=(14, 8))

        signals = data[signal_column]
        close = data["close"]
        buy_mask = signals == 1
        sell_mask = signals == -1

        # 子图1：价格走势 + 信号标记
        ax1 = axes[0]
        ax1.plot(close.index, close.values, label="收盘价", color="gray", alpha=0.7)
        if buy_mask.any():
            ax1.scatter(
                close.index[buy_mask],
                close[buy_mask].values,
                marker="^",
                color="red",
                label="买入信号",
                s=60,
                zorder=5,
            )
        if sell_mask.any():
            ax1.scatter(
                close.index[sell_mask],
                close[sell_mask].values,
                marker="v",
                color="green",
                label="卖出信号",
                s=60,
                zorder=5,
            )
        ax1.set_title("信号时间分布")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 子图2：信号后收益分布
        ax2 = axes[1]
        forward_return = close.pct_change(periods=5).shift(-5)
        buy_returns = forward_return[buy_mask].dropna()
        sell_returns = forward_return[sell_mask].dropna()

        if len(buy_returns) > 0:
            ax2.hist(
                buy_returns,
                bins=30,
                alpha=0.5,
                label="买入信号后5日收益",
                color="red",
            )
        if len(sell_returns) > 0:
            ax2.hist(
                sell_returns,
                bins=30,
                alpha=0.5,
                label="卖出信号后5日收益",
                color="green",
            )
        ax2.axvline(x=0, color="black", linestyle="--", alpha=0.5)
        ax2.set_title("信号后收益分布 (5日)")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        else:
            plt.show()
        plt.close(fig)

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_data(data: pd.DataFrame, signal_column: str) -> None:
        """校验输入数据。"""
        if data.empty:
            raise ValueError("输入数据为空")
        required = ["close", signal_column]
        missing = [c for c in required if c not in data.columns]
        if missing:
            raise ValueError(
                f"输入数据缺少必需列: {missing}。"
                f"现有列: {list(data.columns)}"
            )

    @staticmethod
    def _calc_avg_holding_period(signals: pd.Series) -> float:
        """计算平均持仓天数。

        从买入信号到下一个卖出信号之间的天数，取所有配对的平均值。
        如果没有完整的买入→卖出配对，返回 0。
        """
        holding_periods: list[int] = []
        in_position = False
        entry_idx = 0

        for i, sig in enumerate(signals):
            if sig == 1 and not in_position:
                in_position = True
                entry_idx = i
            elif sig == -1 and in_position:
                holding_periods.append(i - entry_idx)
                in_position = False

        if not holding_periods:
            return 0.0
        return float(np.mean(holding_periods))

    @staticmethod
    def _calc_consecutive_streaks(
        data: pd.DataFrame,
        signal_column: str,
        forward_period: int,
    ) -> tuple[int, int]:
        """计算最大连胜和最大连亏。

        根据信号发出后的前瞻收益判定每笔信号的盈亏：
        - 买入信号后价格上涨 = 盈利
        - 卖出信号后价格下跌 = 盈利

        Returns
        -------
        (max_consecutive_wins, max_consecutive_losses)
        """
        signals = data[signal_column]
        close = data["close"]
        forward_return = close.pct_change(periods=forward_period).shift(
            -forward_period
        )

        # 构建盈亏序列
        outcomes: list[bool] = []
        for i in range(len(data)):
            sig = signals.iloc[i]
            ret = forward_return.iloc[i]
            if sig == 0 or np.isnan(ret):
                continue
            if sig == 1:
                outcomes.append(ret > 0)
            elif sig == -1:
                outcomes.append(ret < 0)

        if not outcomes:
            return 0, 0

        max_wins = 0
        max_losses = 0
        current_wins = 0
        current_losses = 0

        for win in outcomes:
            if win:
                current_wins += 1
                current_losses = 0
                max_wins = max(max_wins, current_wins)
            else:
                current_losses += 1
                current_wins = 0
                max_losses = max(max_losses, current_losses)

        return max_wins, max_losses
