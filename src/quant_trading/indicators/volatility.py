"""波动率指标 - BollingerBands / ATR"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_trading.indicators.base import Indicator


class BollingerBands(Indicator):
    """布林带 (Bollinger Bands)。

    计算方式：
        - 中轨 (middle) = SMA(close, window)
        - 上轨 (upper)  = middle + num_std * std(close, window)
        - 下轨 (lower)  = middle - num_std * std(close, window)
        - 带宽 (width)  = (upper - lower) / middle

    添加的列: ``bb_upper``, ``bb_middle``, ``bb_lower``, ``bb_width``。

    Parameters
    ----------
    window : int
        移动窗口大小，默认 20。
    num_std : float
        标准差倍数，默认 2。
    column : str
        源列名，默认 ``"close"``。
    """

    def __init__(self, window: int = 20, num_std: float = 2.0, column: str = "close") -> None:
        if window < 1:
            raise ValueError(f"window 必须 >= 1，收到 {window}")
        if num_std <= 0:
            raise ValueError(f"num_std 必须 > 0，收到 {num_std}")
        self.window = window
        self.num_std = num_std
        self.column = column

    @property
    def name(self) -> str:
        return f"bollinger_{self.window}_{self.num_std}"

    @property
    def required_columns(self) -> list[str]:
        return [self.column]

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        self._validate(data)
        rolling = data[self.column].rolling(window=self.window, min_periods=self.window)
        middle = rolling.mean()
        std = rolling.std(ddof=0)

        data["bb_middle"] = middle
        data["bb_upper"] = middle + self.num_std * std
        data["bb_lower"] = middle - self.num_std * std
        # 带宽：防止除以零
        data["bb_width"] = np.where(
            middle != 0,
            (data["bb_upper"] - data["bb_lower"]) / middle,
            0.0,
        )
        return data


class ATR(Indicator):
    """平均真实波幅 (Average True Range)。

    计算方式：
        - TR = max(high - low, |high - prev_close|, |low - prev_close|)
        - ATR = EMA(TR, period)  （Wilder 平滑, alpha=1/period）

    添加的列: ``atr_{period}``。

    Parameters
    ----------
    period : int
        计算周期，默认 14。
    """

    def __init__(self, period: int = 14) -> None:
        if period < 1:
            raise ValueError(f"period 必须 >= 1，收到 {period}")
        self.period = period

    @property
    def name(self) -> str:
        return f"atr_{self.period}"

    @property
    def required_columns(self) -> list[str]:
        return ["high", "low", "close"]

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        self._validate(data)
        high = data["high"]
        low = data["low"]
        prev_close = data["close"].shift(1)

        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # Wilder 平滑
        col_name = f"atr_{self.period}"
        data[col_name] = tr.ewm(alpha=1.0 / self.period, min_periods=self.period, adjust=False).mean()
        return data
