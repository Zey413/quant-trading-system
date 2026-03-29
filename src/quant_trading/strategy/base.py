"""策略基类与策略注册表"""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class Strategy(ABC):
    """交易策略抽象基类。

    所有策略必须实现 ``generate_signals`` 方法，该方法接收包含行情数据
    （以及可能已经计算好的指标列）的 DataFrame，在其中添加 ``signal`` 列
    （值为 ``"buy"`` / ``"sell"`` / ``"hold"``）并返回。
    """

    def __init__(self, name: str = "") -> None:
        self.name: str = name or self.__class__.__name__

    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """生成交易信号。

        Parameters
        ----------
        data : pd.DataFrame
            至少包含 OHLCV 列的行情数据。

        Returns
        -------
        pd.DataFrame
            添加了 ``signal`` 列的 DataFrame。
        """
        ...

    def get_params(self) -> dict:
        """返回策略参数字典，用于序列化/日志。"""
        return {}

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r}, params={self.get_params()})"


class StrategyRegistry:
    """全局策略注册表。

    通过装饰器注册策略类，并提供按名称获取、枚举等能力。

    Example
    -------
    >>> @StrategyRegistry.register("my_strategy")
    ... class MyStrategy(Strategy):
    ...     ...
    >>> strategy = StrategyRegistry.get("my_strategy", param1=10)
    """

    _strategies: dict[str, type[Strategy]] = {}

    @classmethod
    def register(cls, name: str):
        """返回一个装饰器，将策略类以 *name* 注册到注册表中。"""

        def decorator(strategy_cls: type[Strategy]) -> type[Strategy]:
            if not (isinstance(strategy_cls, type) and issubclass(strategy_cls, Strategy)):
                raise TypeError(
                    f"只能注册 Strategy 的子类，收到 {strategy_cls}"
                )
            cls._strategies[name] = strategy_cls
            return strategy_cls

        return decorator

    @classmethod
    def get(cls, name: str, **kwargs) -> Strategy:
        """按名称获取已注册的策略实例。

        Parameters
        ----------
        name : str
            注册时使用的策略名称。
        **kwargs
            传递给策略构造函数的参数。

        Raises
        ------
        KeyError
            未找到对应的策略。
        """
        if name not in cls._strategies:
            available = ", ".join(cls._strategies.keys()) or "(无)"
            raise KeyError(
                f"未找到策略 {name!r}。可用策略: {available}"
            )
        return cls._strategies[name](**kwargs)

    @classmethod
    def list_strategies(cls) -> list[str]:
        """列出所有已注册的策略名称。"""
        return list(cls._strategies.keys())
