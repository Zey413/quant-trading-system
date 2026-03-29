"""Dual Thrust 突破策略

经典动量突破策略，在国内期货市场广受欢迎。
"""

from __future__ import annotations

import pandas as pd

from quant_trading.strategy.base import Strategy, StrategyRegistry


@StrategyRegistry.register("dual_thrust")
class DualThrustStrategy(Strategy):
    """Dual Thrust 突破策略。

    原理: 利用前N日的价格区间计算上下轨
    - Range = max(HH-LC, HC-LL)  (N日最高最高-最低收盘, 最高收盘-最低最低)
    - 上轨 = Open + K1 * Range
    - 下轨 = Open - K2 * Range
    - 突破上轨买入, 突破下轨卖出

    Parameters
    ----------
    lookback : int
        回看周期，用于计算价格区间，默认 4。
    k1 : float
        上轨系数，默认 0.5。
    k2 : float
        下轨系数，默认 0.5。
    """

    def __init__(
        self,
        lookback: int = 4,
        k1: float = 0.5,
        k2: float = 0.5,
        name: str = "",
    ) -> None:
        super().__init__(name=name)
        if lookback < 1:
            raise ValueError(f"lookback 必须 >= 1，收到 {lookback}")
        if k1 <= 0:
            raise ValueError(f"k1 必须 > 0，收到 {k1}")
        if k2 <= 0:
            raise ValueError(f"k2 必须 > 0，收到 {k2}")
        self.lookback = lookback
        self.k1 = k1
        self.k2 = k2

    def get_params(self) -> dict:
        return {
            "lookback": self.lookback,
            "k1": self.k1,
            "k2": self.k2,
        }

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        # 计算前 N 日的 HH、LL、HC、LC
        hh = data["high"].rolling(window=self.lookback, min_periods=self.lookback).max()
        ll = data["low"].rolling(window=self.lookback, min_periods=self.lookback).min()
        hc = data["close"].rolling(window=self.lookback, min_periods=self.lookback).max()
        lc = data["close"].rolling(window=self.lookback, min_periods=self.lookback).min()

        # 将 rolling 结果偏移一行，使用"前 N 日"的数据（不含当日）
        hh = hh.shift(1)
        ll = ll.shift(1)
        hc = hc.shift(1)
        lc = lc.shift(1)

        # Range = max(HH - LC, HC - LL)
        range1 = hh - lc
        range2 = hc - ll
        price_range = pd.concat([range1, range2], axis=1).max(axis=1)

        # 上下轨
        open_price = data["open"]
        data["dt_upper"] = open_price + self.k1 * price_range
        data["dt_lower"] = open_price - self.k2 * price_range
        data["dt_range"] = price_range

        # 生成信号：收盘价突破上轨买入，突破下轨卖出
        close = data["close"]
        data["signal"] = "hold"
        data.loc[close > data["dt_upper"], "signal"] = "buy"
        data.loc[close < data["dt_lower"], "signal"] = "sell"

        return data
