"""动量因子

提供:
- MomentumFactor: 经典动量因子（过去N日收益率）
- ReverseReturnFactor: 反转因子（短期反转）
"""

from __future__ import annotations

import pandas as pd

from quant_trading.factor.base import Factor, FactorDirection, FactorRegistry


@FactorRegistry.register("momentum")
class MomentumFactor(Factor):
    """动量因子 - 过去N日收益率

    动量因子 = (当前价格 / N日前价格) - 1

    动量效应：过去表现好的股票在未来一段时间内倾向于继续表现好。

    Parameters
    ----------
    window : int
        回看窗口期（交易日），默认 20 日
    name : str
        因子名称
    """

    def __init__(self, window: int = 20, name: str = "") -> None:
        super().__init__(
            name=name or f"momentum_{window}",
            direction=FactorDirection.POSITIVE,
        )
        self.window = window

    def compute(self, data: pd.DataFrame) -> pd.Series:
        """计算动量因子值

        Parameters
        ----------
        data : pd.DataFrame
            必须包含 ``close`` 列

        Returns
        -------
        pd.Series
            动量因子值（过去N日收益率）
        """
        close = data["close"]
        momentum = close / close.shift(self.window) - 1
        return momentum

    def get_params(self) -> dict:
        return {"window": self.window}


@FactorRegistry.register("reverse_return")
class ReverseReturnFactor(Factor):
    """反转因子 - 短期收益反转

    反转因子 = -(过去N日收益率)

    短期反转效应：过去短期内下跌较多的股票倾向于反弹。

    Parameters
    ----------
    window : int
        回看窗口期（交易日），默认 5 日
    name : str
        因子名称
    """

    def __init__(self, window: int = 5, name: str = "") -> None:
        super().__init__(
            name=name or f"reverse_{window}",
            direction=FactorDirection.POSITIVE,
        )
        self.window = window

    def compute(self, data: pd.DataFrame) -> pd.Series:
        """计算反转因子值

        Parameters
        ----------
        data : pd.DataFrame
            必须包含 ``close`` 列

        Returns
        -------
        pd.Series
            反转因子值（负的过去N日收益率）
        """
        close = data["close"]
        momentum = close / close.shift(self.window) - 1
        return -momentum

    def get_params(self) -> dict:
        return {"window": self.window}
