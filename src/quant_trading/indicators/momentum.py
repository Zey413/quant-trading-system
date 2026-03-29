"""动量类指标 - ROC / Williams %R / CCI"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_trading.indicators.base import Indicator


class ROC(Indicator):
    """变动率 (Rate of Change)。

    计算方式：
        ROC = (close - close_n) / close_n * 100

    添加的列: ``roc_{period}``。

    Parameters
    ----------
    period : int
        回看周期，默认 12。
    column : str
        源列名，默认 ``"close"``。
    """

    def __init__(self, period: int = 12, column: str = "close") -> None:
        if period < 1:
            raise ValueError(f"period 必须 >= 1，收到 {period}")
        self.period = period
        self.column = column

    @property
    def name(self) -> str:
        return f"roc_{self.period}"

    @property
    def required_columns(self) -> list[str]:
        return [self.column]

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        self._validate(data)
        close = data[self.column]
        close_n = close.shift(self.period)
        # 防止除以零
        col_name = f"roc_{self.period}"
        data[col_name] = np.where(
            close_n != 0,
            (close - close_n) / close_n * 100.0,
            np.nan,
        )
        return data


class WilliamsR(Indicator):
    """威廉指标 (Williams %R)。

    计算方式：
        %R = (highest_high - close) / (highest_high - lowest_low) * -100

    范围: -100 ~ 0
    添加的列: ``williams_r_{period}``。

    Parameters
    ----------
    period : int
        回看周期，默认 14。
    """

    def __init__(self, period: int = 14) -> None:
        if period < 1:
            raise ValueError(f"period 必须 >= 1，收到 {period}")
        self.period = period

    @property
    def name(self) -> str:
        return f"williams_r_{self.period}"

    @property
    def required_columns(self) -> list[str]:
        return ["high", "low", "close"]

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        self._validate(data)
        highest_high = data["high"].rolling(
            window=self.period, min_periods=self.period
        ).max()
        lowest_low = data["low"].rolling(
            window=self.period, min_periods=self.period
        ).min()

        denom = highest_high - lowest_low
        col_name = f"williams_r_{self.period}"
        data[col_name] = np.where(
            denom != 0,
            (highest_high - data["close"]) / denom * -100.0,
            -50.0,  # 当高低点相同时取中值
        )
        # 仅在rolling窗口有效时保留值
        data.loc[data.index[: self.period - 1], col_name] = np.nan
        return data


class CCI(Indicator):
    """顺势指标 (Commodity Channel Index)。

    计算方式：
        TP = (high + low + close) / 3
        CCI = (TP - SMA(TP, period)) / (0.015 * mean_deviation)

    添加的列: ``cci_{period}``。

    Parameters
    ----------
    period : int
        计算周期，默认 20。
    """

    def __init__(self, period: int = 20) -> None:
        if period < 1:
            raise ValueError(f"period 必须 >= 1，收到 {period}")
        self.period = period

    @property
    def name(self) -> str:
        return f"cci_{self.period}"

    @property
    def required_columns(self) -> list[str]:
        return ["high", "low", "close"]

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        self._validate(data)
        tp = (data["high"] + data["low"] + data["close"]) / 3.0
        tp_sma = tp.rolling(window=self.period, min_periods=self.period).mean()

        # 平均绝对偏差 (mean deviation)
        mean_dev = tp.rolling(window=self.period, min_periods=self.period).apply(
            lambda x: np.mean(np.abs(x - x.mean())), raw=True
        )

        col_name = f"cci_{self.period}"
        data[col_name] = np.where(
            mean_dev != 0,
            (tp - tp_sma) / (0.015 * mean_dev),
            0.0,
        )
        # 仅在rolling窗口有效时保留值
        data.loc[data.index[: self.period - 1], col_name] = np.nan
        return data
