"""趋势类指标 - SMA / EMA / MACD"""

from __future__ import annotations

import pandas as pd

from quant_trading.indicators.base import Indicator


class SMA(Indicator):
    """简单移动平均线 (Simple Moving Average)。

    Parameters
    ----------
    window : int
        移动窗口大小。
    column : str
        用于计算的源列名，默认 ``"close"``。
    """

    def __init__(self, window: int = 20, column: str = "close") -> None:
        if window < 1:
            raise ValueError(f"window 必须 >= 1，收到 {window}")
        self.window = window
        self.column = column

    @property
    def name(self) -> str:
        return f"sma_{self.window}"

    @property
    def required_columns(self) -> list[str]:
        return [self.column]

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        self._validate(data)
        col_name = f"sma_{self.window}"
        data[col_name] = data[self.column].rolling(window=self.window, min_periods=self.window).mean()
        return data


class EMA(Indicator):
    """指数移动平均线 (Exponential Moving Average)。

    Parameters
    ----------
    window : int
        衰减跨度 (span)。
    column : str
        用于计算的源列名，默认 ``"close"``。
    """

    def __init__(self, window: int = 20, column: str = "close") -> None:
        if window < 1:
            raise ValueError(f"window 必须 >= 1，收到 {window}")
        self.window = window
        self.column = column

    @property
    def name(self) -> str:
        return f"ema_{self.window}"

    @property
    def required_columns(self) -> list[str]:
        return [self.column]

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        self._validate(data)
        col_name = f"ema_{self.window}"
        data[col_name] = data[self.column].ewm(span=self.window, adjust=False).mean()
        return data


class MACD(Indicator):
    """MACD 指标 (Moving Average Convergence Divergence)。

    计算方式：
        - DIF  = EMA(close, fast) - EMA(close, slow)
        - DEA  = EMA(DIF, signal)
        - HIST = 2 * (DIF - DEA)

    添加的列：``macd`` (DIF), ``macd_signal`` (DEA), ``macd_hist`` (HIST)。

    Parameters
    ----------
    fast : int
        快线周期，默认 12。
    slow : int
        慢线周期，默认 26。
    signal : int
        信号线周期，默认 9。
    column : str
        用于计算的源列名，默认 ``"close"``。
    """

    def __init__(
        self,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
        column: str = "close",
    ) -> None:
        if fast >= slow:
            raise ValueError(f"fast ({fast}) 必须小于 slow ({slow})")
        self.fast = fast
        self.slow = slow
        self.signal = signal
        self.column = column

    @property
    def name(self) -> str:
        return f"macd_{self.fast}_{self.slow}_{self.signal}"

    @property
    def required_columns(self) -> list[str]:
        return [self.column]

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        self._validate(data)
        ema_fast = data[self.column].ewm(span=self.fast, adjust=False).mean()
        ema_slow = data[self.column].ewm(span=self.slow, adjust=False).mean()
        dif = ema_fast - ema_slow
        dea = dif.ewm(span=self.signal, adjust=False).mean()
        data["macd"] = dif
        data["macd_signal"] = dea
        data["macd_hist"] = 2.0 * (dif - dea)
        return data
