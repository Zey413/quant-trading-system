"""因子基类与因子注册表

提供:
- Factor: 因子抽象基类，定义统一的计算接口
- FactorRegistry: 因子注册表，装饰器模式自动注册
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from enum import Enum

import pandas as pd

logger = logging.getLogger(__name__)


class FactorDirection(str, Enum):
    """因子方向 - 表示因子值与预期收益的关系"""
    POSITIVE = "positive"   # 因子值越大，预期收益越高
    NEGATIVE = "negative"   # 因子值越小，预期收益越高


class Factor(ABC):
    """因子抽象基类

    所有因子必须实现 ``compute`` 方法，返回因子值 Series 或 DataFrame。

    Attributes
    ----------
    name : str
        因子名称
    direction : FactorDirection
        因子方向，用于分层回测排序
    """

    def __init__(
        self,
        name: str = "",
        direction: FactorDirection = FactorDirection.POSITIVE,
    ) -> None:
        self.name: str = name or self.__class__.__name__
        self.direction = direction

    @abstractmethod
    def compute(self, data: pd.DataFrame) -> pd.Series:
        """计算因子值

        Parameters
        ----------
        data : pd.DataFrame
            包含行情/财务数据的 DataFrame。
            对于单只股票，index 为日期；
            对于截面数据，可包含多只股票的面板数据。

        Returns
        -------
        pd.Series
            因子值，index 与输入 data 对齐
        """
        ...

    def compute_cross_section(
        self,
        panel_data: dict[str, pd.DataFrame],
    ) -> pd.DataFrame:
        """计算截面因子值（多只股票同时计算）

        Parameters
        ----------
        panel_data : dict[str, pd.DataFrame]
            {股票代码: 行情DataFrame} 的字典

        Returns
        -------
        pd.DataFrame
            index 为日期，columns 为股票代码，值为因子值
        """
        factor_values = {}
        for symbol, df in panel_data.items():
            try:
                factor_values[symbol] = self.compute(df)
            except Exception as e:
                logger.warning("计算因子 %s 对 %s 失败: %s", self.name, symbol, e)
        return pd.DataFrame(factor_values)

    def get_params(self) -> dict:
        """返回因子参数，用于序列化/日志"""
        return {}

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}"
            f"(name={self.name!r}, direction={self.direction.value}, "
            f"params={self.get_params()})"
        )


class FactorRegistry:
    """全局因子注册表

    通过装饰器注册因子类，并提供按名称获取、枚举等能力。

    Example
    -------
    >>> @FactorRegistry.register("momentum_20")
    ... class MyMomentum(Factor):
    ...     ...
    >>> factor = FactorRegistry.get("momentum_20", window=20)
    """

    _factors: dict[str, type[Factor]] = {}

    @classmethod
    def register(cls, name: str):
        """返回一个装饰器，将因子类以 *name* 注册到注册表中。"""

        def decorator(factor_cls: type[Factor]) -> type[Factor]:
            if not (isinstance(factor_cls, type) and issubclass(factor_cls, Factor)):
                raise TypeError(
                    f"只能注册 Factor 的子类，收到 {factor_cls}"
                )
            cls._factors[name] = factor_cls
            return factor_cls

        return decorator

    @classmethod
    def get(cls, name: str, **kwargs) -> Factor:
        """按名称获取已注册的因子实例。

        Parameters
        ----------
        name : str
            注册时使用的因子名称。
        **kwargs
            传递给因子构造函数的参数。

        Raises
        ------
        KeyError
            未找到对应的因子。
        """
        if name not in cls._factors:
            available = ", ".join(cls._factors.keys()) or "(无)"
            raise KeyError(
                f"未找到因子 {name!r}。可用因子: {available}"
            )
        return cls._factors[name](**kwargs)

    @classmethod
    def list_factors(cls) -> list[str]:
        """列出所有已注册的因子名称。"""
        return list(cls._factors.keys())

    @classmethod
    def clear(cls) -> None:
        """清空注册表（用于测试）"""
        cls._factors.clear()
