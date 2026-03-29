"""MACD 策略

MACD 柱状线由负转正时买入（金叉），由正转负时卖出（死叉）。
"""

from __future__ import annotations

import pandas as pd

from quant_trading.indicators.trend import MACD
from quant_trading.strategy.base import Strategy, StrategyRegistry


@StrategyRegistry.register("macd")
class MACDStrategy(Strategy):
    """MACD 策略。

    - 当 MACD 柱状线由负转正（即 MACD 线上穿信号线 — 金叉）时买入。
    - 当 MACD 柱状线由正转负（即 MACD 线下穿信号线 — 死叉）时卖出。
    - 其余为 hold。

    Parameters
    ----------
    fast : int
        快线周期，默认 12。
    slow : int
        慢线周期，默认 26。
    signal : int
        信号线周期，默认 9。
    """

    def __init__(
        self,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
        name: str = "",
    ) -> None:
        super().__init__(name=name)
        self.fast = fast
        self.slow = slow
        self.signal = signal
        self._macd = MACD(fast=fast, slow=slow, signal=signal)

    def get_params(self) -> dict:
        return {
            "fast": self.fast,
            "slow": self.slow,
            "signal": self.signal,
        }

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        if "macd_hist" not in data.columns:
            data = self._macd.calculate(data)

        hist = data["macd_hist"]
        prev_hist = hist.shift(1)

        # 金叉：前一 bar 柱状线 <= 0，当前 bar > 0
        buy_signal = (prev_hist <= 0) & (hist > 0)
        # 死叉：前一 bar 柱状线 >= 0，当前 bar < 0
        sell_signal = (prev_hist >= 0) & (hist < 0)

        data["signal"] = "hold"
        data.loc[buy_signal, "signal"] = "buy"
        data.loc[sell_signal, "signal"] = "sell"
        return data
