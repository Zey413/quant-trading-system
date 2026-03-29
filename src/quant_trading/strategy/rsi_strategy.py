"""RSI 超买超卖策略

RSI 从超卖区上穿时买入，从超买区下穿时卖出。
"""

from __future__ import annotations

import pandas as pd

from quant_trading.indicators.oscillator import RSI
from quant_trading.strategy.base import Strategy, StrategyRegistry


@StrategyRegistry.register("rsi")
class RSIStrategy(Strategy):
    """RSI 超买超卖策略。

    - 当 RSI 从超卖线以下上穿 *oversold* 时产生买入信号。
    - 当 RSI 从超买线以上下穿 *overbought* 时产生卖出信号。
    - 其余为 hold。

    Parameters
    ----------
    period : int
        RSI 计算周期，默认 14。
    oversold : float
        超卖阈值，默认 30。
    overbought : float
        超买阈值，默认 70。
    """

    def __init__(
        self,
        period: int = 14,
        oversold: float = 30.0,
        overbought: float = 70.0,
        name: str = "",
    ) -> None:
        super().__init__(name=name)
        if oversold >= overbought:
            raise ValueError(
                f"oversold ({oversold}) 必须小于 overbought ({overbought})"
            )
        self.period = period
        self.oversold = oversold
        self.overbought = overbought
        self._rsi = RSI(period=period)

    def get_params(self) -> dict:
        return {
            "period": self.period,
            "oversold": self.oversold,
            "overbought": self.overbought,
        }

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        rsi_col = f"rsi_{self.period}"
        if rsi_col not in data.columns:
            data = self._rsi.calculate(data)

        rsi = data[rsi_col]
        prev_rsi = rsi.shift(1)

        # 买入: 前一 bar RSI < oversold，当前 bar RSI >= oversold（从下方上穿）
        buy_signal = (prev_rsi < self.oversold) & (rsi >= self.oversold)
        # 卖出: 前一 bar RSI > overbought，当前 bar RSI <= overbought（从上方下穿）
        sell_signal = (prev_rsi > self.overbought) & (rsi <= self.overbought)

        data["signal"] = "hold"
        data.loc[buy_signal, "signal"] = "buy"
        data.loc[sell_signal, "signal"] = "sell"
        return data
