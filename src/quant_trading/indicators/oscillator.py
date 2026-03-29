"""振荡类指标 - RSI / KDJ"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_trading.indicators.base import Indicator


class RSI(Indicator):
    """RSI 相对强弱指标 (Relative Strength Index)。

    计算方式（Wilder 平滑）：
        - delta = close.diff()
        - gain  = delta.clip(lower=0)
        - loss  = (-delta).clip(lower=0)
        - avg_gain = gain.ewm(alpha=1/period, adjust=False)
        - avg_loss = loss.ewm(alpha=1/period, adjust=False)
        - RS  = avg_gain / avg_loss
        - RSI = 100 - 100 / (1 + RS)

    添加的列: ``rsi_{period}``，范围 [0, 100]。

    Parameters
    ----------
    period : int
        计算周期，默认 14。
    column : str
        源列名，默认 ``"close"``。
    """

    def __init__(self, period: int = 14, column: str = "close") -> None:
        if period < 1:
            raise ValueError(f"period 必须 >= 1，收到 {period}")
        self.period = period
        self.column = column

    @property
    def name(self) -> str:
        return f"rsi_{self.period}"

    @property
    def required_columns(self) -> list[str]:
        return [self.column]

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        self._validate(data)
        delta = data[self.column].diff()
        gain = delta.clip(lower=0.0)
        loss = (-delta).clip(lower=0.0)

        # Wilder 平滑 (等效于 alpha=1/period 的EMA)
        avg_gain = gain.ewm(alpha=1.0 / self.period, min_periods=self.period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1.0 / self.period, min_periods=self.period, adjust=False).mean()

        rs = avg_gain / avg_loss
        rsi = 100.0 - 100.0 / (1.0 + rs)

        # 处理 avg_loss == 0（全涨）或 avg_gain == 0（全跌）的极端情况
        rsi = rsi.where(avg_loss != 0, 100.0)
        rsi = rsi.where(avg_gain != 0, np.where(avg_loss != 0, 0.0, 50.0))

        col_name = f"rsi_{self.period}"
        data[col_name] = rsi
        return data


class KDJ(Indicator):
    """KDJ 随机指标 (Stochastic Oscillator)。

    计算方式：
        - RSV = (close - lowest_low_n) / (highest_high_n - lowest_low_n) * 100
        - K   = SMA(RSV, m1)  （此处的SMA为递推平滑: K_t = (1/m1)*RSV_t + (1-1/m1)*K_{t-1}）
        - D   = SMA(K, m2)
        - J   = 3*K - 2*D

    添加的列: ``k``, ``d``, ``j``。

    Parameters
    ----------
    n : int
        RSV 的回看周期，默认 9。
    m1 : int
        K 值的平滑周期，默认 3。
    m2 : int
        D 值的平滑周期，默认 3。
    """

    def __init__(self, n: int = 9, m1: int = 3, m2: int = 3) -> None:
        if n < 1 or m1 < 1 or m2 < 1:
            raise ValueError(f"n, m1, m2 必须 >= 1，收到 n={n}, m1={m1}, m2={m2}")
        self.n = n
        self.m1 = m1
        self.m2 = m2

    @property
    def name(self) -> str:
        return f"kdj_{self.n}_{self.m1}_{self.m2}"

    @property
    def required_columns(self) -> list[str]:
        return ["high", "low", "close"]

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        self._validate(data)
        low_n = data["low"].rolling(window=self.n, min_periods=self.n).min()
        high_n = data["high"].rolling(window=self.n, min_periods=self.n).max()

        # 防止除以零
        denom = high_n - low_n
        rsv = pd.Series(
            np.where(denom == 0, 50.0, (data["close"] - low_n) / denom * 100.0),
            index=data.index,
        )

        # 递推 SMA 平滑: val_t = (1/m)*src_t + (1-1/m)*val_{t-1}
        # 初始值取 50
        k_values = np.empty(len(data))
        d_values = np.empty(len(data))
        k_values[:] = np.nan
        d_values[:] = np.nan

        rsv_arr = rsv.to_numpy()
        alpha_k = 1.0 / self.m1
        alpha_d = 1.0 / self.m2

        # 找到第一个非 NaN 的 RSV 位置作为起点
        first_valid = rsv.first_valid_index()
        if first_valid is None:
            data["k"] = np.nan
            data["d"] = np.nan
            data["j"] = np.nan
            return data

        start = data.index.get_loc(first_valid)
        if isinstance(start, slice):
            start = start.start

        k_values[start] = 50.0  # 初始 K = 50
        d_values[start] = 50.0  # 初始 D = 50

        for i in range(start, len(data)):
            rsv_val = rsv_arr[i]
            if np.isnan(rsv_val):
                k_values[i] = np.nan
                d_values[i] = np.nan
                continue
            if i == start:
                k_values[i] = alpha_k * rsv_val + (1 - alpha_k) * 50.0
            else:
                prev_k = k_values[i - 1]
                if np.isnan(prev_k):
                    prev_k = 50.0
                k_values[i] = alpha_k * rsv_val + (1 - alpha_k) * prev_k

            if i == start:
                d_values[i] = alpha_d * k_values[i] + (1 - alpha_d) * 50.0
            else:
                prev_d = d_values[i - 1]
                if np.isnan(prev_d):
                    prev_d = 50.0
                d_values[i] = alpha_d * k_values[i] + (1 - alpha_d) * prev_d

        data["k"] = k_values
        data["d"] = d_values
        data["j"] = 3.0 * data["k"] - 2.0 * data["d"]
        return data
