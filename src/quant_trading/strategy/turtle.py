"""海龟交易策略 (Turtle Trading - 简化版)

经典的趋势跟踪策略，基于通道突破。
"""

from __future__ import annotations

import pandas as pd

from quant_trading.indicators.volatility import ATR
from quant_trading.strategy.base import Strategy, StrategyRegistry


@StrategyRegistry.register("turtle")
class TurtleStrategy(Strategy):
    """海龟交易策略（简化版）。

    原理: 经典的趋势跟踪策略
    - 20日通道突破买入（价格创20日新高）
    - 10日通道突破卖出（价格创10日新低）
    - 用ATR计算仓位（ATR值存入列供外部使用）

    Parameters
    ----------
    entry_period : int
        入场通道周期（创新高买入），默认 20。
    exit_period : int
        出场通道周期（创新低卖出），默认 10。
    atr_period : int
        ATR 周期，用于仓位计算，默认 20。
    """

    def __init__(
        self,
        entry_period: int = 20,
        exit_period: int = 10,
        atr_period: int = 20,
        name: str = "",
    ) -> None:
        super().__init__(name=name)
        if entry_period < 1:
            raise ValueError(f"entry_period 必须 >= 1，收到 {entry_period}")
        if exit_period < 1:
            raise ValueError(f"exit_period 必须 >= 1，收到 {exit_period}")
        if atr_period < 1:
            raise ValueError(f"atr_period 必须 >= 1，收到 {atr_period}")
        self.entry_period = entry_period
        self.exit_period = exit_period
        self.atr_period = atr_period
        self._atr = ATR(period=atr_period)

    def get_params(self) -> dict:
        return {
            "entry_period": self.entry_period,
            "exit_period": self.exit_period,
            "atr_period": self.atr_period,
        }

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        # 计算 ATR（如果尚未存在）
        atr_col = f"atr_{self.atr_period}"
        if atr_col not in data.columns:
            data = self._atr.calculate(data)

        # 入场通道：前 entry_period 日的最高价（不含当日）
        data["turtle_upper"] = data["high"].rolling(
            window=self.entry_period, min_periods=self.entry_period,
        ).max().shift(1)

        # 出场通道：前 exit_period 日的最低价（不含当日）
        data["turtle_lower"] = data["low"].rolling(
            window=self.exit_period, min_periods=self.exit_period,
        ).min().shift(1)

        # 生成信号
        close = data["close"]
        data["signal"] = "hold"
        # 收盘价突破入场通道上轨 -> 买入
        data.loc[close > data["turtle_upper"], "signal"] = "buy"
        # 收盘价跌破出场通道下轨 -> 卖出
        data.loc[close < data["turtle_lower"], "signal"] = "sell"

        return data
