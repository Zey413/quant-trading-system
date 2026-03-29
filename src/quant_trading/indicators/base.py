"""指标基类 - 所有技术指标的抽象基础类"""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class Indicator(ABC):
    """技术指标抽象基类。

    所有指标必须实现 ``calculate`` 方法，该方法接收包含OHLCV数据的DataFrame，
    在其中添加计算后的指标列并返回。
    """

    @abstractmethod
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算指标，将结果列添加到 *data* 中并返回。

        Parameters
        ----------
        data : pd.DataFrame
            至少包含 ``required_columns`` 中声明的列。

        Returns
        -------
        pd.DataFrame
            添加了指标列的 DataFrame（原地修改并返回）。
        """
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """指标名称，用于日志与展示。"""
        ...

    @property
    @abstractmethod
    def required_columns(self) -> list[str]:
        """计算该指标所需的最少列名列表。"""
        ...

    def _validate(self, data: pd.DataFrame) -> None:
        """校验输入数据是否包含所需列。"""
        missing = [c for c in self.required_columns if c not in data.columns]
        if missing:
            raise ValueError(
                f"指标 {self.name} 缺少必需列: {missing}。"
                f"DataFrame 现有列: {list(data.columns)}"
            )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"
