"""布林带策略 (Bollinger Bands Strategy)

价格触及下轨后回升买入，触及上轨后回落卖出。
"""

from __future__ import annotations

import pandas as pd

from quant_trading.indicators.volatility import BollingerBands
from quant_trading.strategy.base import Strategy, StrategyRegistry


@StrategyRegistry.register("bollinger")
class BollingerStrategy(Strategy):
    """布林带策略。

    - 当价格在前一 bar 触及或跌破下轨（close <= bb_lower），
      且当前 bar 回升至下轨之上时，产生买入信号。
    - 当价格在前一 bar 触及或突破上轨（close >= bb_upper），
      且当前 bar 回落至上轨之下时，产生卖出信号。
    - 其余为 hold。

    Parameters
    ----------
    window : int
        布林带窗口，默认 20。
    num_std : float
        标准差倍数，默认 2。
    """

    def __init__(
        self,
        window: int = 20,
        num_std: float = 2.0,
        name: str = "",
    ) -> None:
        super().__init__(name=name)
        self.window = window
        self.num_std = num_std
        self._bb = BollingerBands(window=window, num_std=num_std)

    def get_params(self) -> dict:
        return {
            "window": self.window,
            "num_std": self.num_std,
        }

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        if "bb_upper" not in data.columns:
            data = self._bb.calculate(data)

        close = data["close"]
        prev_close = close.shift(1)
        bb_lower = data["bb_lower"]
        bb_upper = data["bb_upper"]
        prev_lower = bb_lower.shift(1)
        prev_upper = bb_upper.shift(1)

        # 买入: 前一 bar 收盘 <= 下轨，当前 bar 收盘 > 下轨（触底回升）
        buy_signal = (prev_close <= prev_lower) & (close > bb_lower)
        # 卖出: 前一 bar 收盘 >= 上轨，当前 bar 收盘 < 上轨（冲顶回落）
        sell_signal = (prev_close >= prev_upper) & (close < bb_upper)

        data["signal"] = "hold"
        data.loc[buy_signal, "signal"] = "buy"
        data.loc[sell_signal, "signal"] = "sell"
        return data
