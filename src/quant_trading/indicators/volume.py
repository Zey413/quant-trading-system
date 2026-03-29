"""成交量指标 - OBV / VWAP"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_trading.indicators.base import Indicator


class OBV(Indicator):
    """能量潮指标 (On-Balance Volume)。

    计算方式：
        - 当日收盘价 > 昨日收盘价: OBV += volume
        - 当日收盘价 < 昨日收盘价: OBV -= volume
        - 当日收盘价 == 昨日收盘价: OBV 不变

    添加的列: ``obv``。
    """

    @property
    def name(self) -> str:
        return "obv"

    @property
    def required_columns(self) -> list[str]:
        return ["close", "volume"]

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        self._validate(data)
        close = data["close"]
        volume = data["volume"]

        # 计算价格变动方向: +1 / -1 / 0
        direction = np.sign(close.diff())
        # 第一个值没有前一天的收盘价，设方向为 0
        direction.iloc[0] = 0

        data["obv"] = (direction * volume).cumsum()
        return data


class VWAP(Indicator):
    """成交量加权平均价格 (Volume Weighted Average Price)。

    计算方式：
        - typical_price = (high + low + close) / 3
        - VWAP = cumsum(typical_price * volume) / cumsum(volume)

    添加的列: ``vwap``。

    Notes
    -----
    此实现为全序列累积 VWAP。如需日内重置，应在外部按交易日分组后
    分别调用。
    """

    @property
    def name(self) -> str:
        return "vwap"

    @property
    def required_columns(self) -> list[str]:
        return ["high", "low", "close", "volume"]

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        self._validate(data)
        typical_price = (data["high"] + data["low"] + data["close"]) / 3.0
        cum_tp_vol = (typical_price * data["volume"]).cumsum()
        cum_vol = data["volume"].cumsum()

        # 防止除以零
        data["vwap"] = np.where(cum_vol != 0, cum_tp_vol / cum_vol, 0.0)
        return data
