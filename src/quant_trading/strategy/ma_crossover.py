"""均线交叉策略 (Moving Average Crossover)

金叉买入、死叉卖出。
"""

from __future__ import annotations

import pandas as pd

from quant_trading.indicators.trend import SMA
from quant_trading.strategy.base import Strategy, StrategyRegistry


@StrategyRegistry.register("ma_crossover")
class MACrossoverStrategy(Strategy):
    """均线交叉策略。

    当短期均线上穿长期均线（金叉）时产生买入信号；
    当短期均线下穿长期均线（死叉）时产生卖出信号；
    其余时刻为持仓不变（hold）。

    Parameters
    ----------
    short_window : int
        短期均线周期，默认 5。
    long_window : int
        长期均线周期，默认 20。
    """

    def __init__(
        self,
        short_window: int = 5,
        long_window: int = 20,
        name: str = "",
    ) -> None:
        super().__init__(name=name)
        if short_window >= long_window:
            raise ValueError(
                f"short_window ({short_window}) 必须小于 long_window ({long_window})"
            )
        self.short_window = short_window
        self.long_window = long_window
        self._sma_short = SMA(window=short_window)
        self._sma_long = SMA(window=long_window)

    def get_params(self) -> dict:
        return {
            "short_window": self.short_window,
            "long_window": self.long_window,
        }

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        # 计算均线（如果尚未存在）
        short_col = f"sma_{self.short_window}"
        long_col = f"sma_{self.long_window}"
        if short_col not in data.columns:
            data = self._sma_short.calculate(data)
        if long_col not in data.columns:
            data = self._sma_long.calculate(data)

        short_ma = data[short_col]
        long_ma = data[long_col]

        # 当前 bar 短均线在长均线之上/之下
        above = short_ma > long_ma
        below_or_equal = short_ma <= long_ma
        # 前一 bar 的关系
        prev_above = above.shift(1, fill_value=False)
        prev_below_or_equal = below_or_equal.shift(1, fill_value=True)

        # 金叉：前一 bar 短 <= 长，当前 bar 短 > 长
        golden_cross = prev_below_or_equal & above
        # 死叉：前一 bar 短 >= 长，当前 bar 短 < 长
        death_cross = prev_above & below_or_equal

        data["signal"] = "hold"
        data.loc[golden_cross, "signal"] = "buy"
        data.loc[death_cross, "signal"] = "sell"
        return data
