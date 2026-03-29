"""数据源基类和注册表 - 可插拔数据源架构的基石

提供:
- DataSource: 数据源抽象基类，定义统一接口
- DataSourceRegistry: 数据源注册表，装饰器模式实现自动发现
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import pandas as pd

logger = logging.getLogger(__name__)


class DataSource(ABC):
    """数据源抽象基类

    所有数据源（AKShare、Tushare等）必须继承此类，
    实现 fetch_daily() 和 fetch_stock_list() 方法。

    返回的 DataFrame 必须包含标准化列名:
        date, open, high, low, close, volume
    可选列:
        amount (成交额)
    """

    @abstractmethod
    def get_name(self) -> str:
        """返回数据源名称标识符"""
        ...

    @abstractmethod
    def fetch_daily(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        adjust: str = "qfq",
    ) -> pd.DataFrame:
        """获取日线行情数据

        Parameters
        ----------
        symbol : str
            股票代码，纯数字格式，例如 "000001", "600519"
        start_date : str
            开始日期，格式 "YYYYMMDD" 或 "YYYY-MM-DD"
        end_date : str
            结束日期，格式 "YYYYMMDD" 或 "YYYY-MM-DD"
        adjust : str
            复权类型: "qfq"(前复权), "hfq"(后复权), "none"(不复权)

        Returns
        -------
        pd.DataFrame
            包含列: date, open, high, low, close, volume[, amount]
            按日期升序排列
        """
        ...

    @abstractmethod
    def fetch_stock_list(self) -> pd.DataFrame:
        """获取A股股票列表

        Returns
        -------
        pd.DataFrame
            至少包含列: code (股票代码), name (股票名称)
        """
        ...

    def fetch_realtime(self, symbol: str) -> dict:
        """获取实时行情（可选实现）

        Parameters
        ----------
        symbol : str
            股票代码

        Returns
        -------
        dict
            包含 price, open, high, low, volume 等字段

        Raises
        ------
        NotImplementedError
            子类未实现此方法时抛出
        """
        raise NotImplementedError(
            f"{self.get_name()} does not support realtime data"
        )


class DataSourceRegistry:
    """数据源注册表 - 装饰器模式

    使用方式::

        @DataSourceRegistry.register("akshare")
        class AKShareDataSource(DataSource):
            ...

        # 获取数据源实例
        source = DataSourceRegistry.get("akshare")

    注意: _sources 是类级别的可变字典。在同一进程中，
    所有通过 register 注册的数据源都会被保留。
    """

    _sources: dict[str, type[DataSource]] = {}

    @classmethod
    def register(cls, name: str):
        """装饰器：注册数据源类

        Parameters
        ----------
        name : str
            数据源名称标识符，例如 "akshare", "tushare"

        Returns
        -------
        Callable
            装饰器函数
        """

        def decorator(source_cls: type[DataSource]) -> type[DataSource]:
            if name in cls._sources:
                logger.warning(
                    "Data source '%s' is being re-registered: %s -> %s",
                    name,
                    cls._sources[name].__name__,
                    source_cls.__name__,
                )
            cls._sources[name] = source_cls
            logger.debug("Registered data source: %s -> %s", name, source_cls.__name__)
            return source_cls

        return decorator

    @classmethod
    def get(cls, name: str) -> DataSource:
        """获取数据源实例

        Parameters
        ----------
        name : str
            已注册的数据源名称

        Returns
        -------
        DataSource
            数据源实例

        Raises
        ------
        ValueError
            数据源未注册时抛出
        """
        if name not in cls._sources:
            available = ", ".join(sorted(cls._sources.keys())) or "(none)"
            raise ValueError(
                f"Unknown data source: '{name}'. Available: {available}"
            )
        return cls._sources[name]()

    @classmethod
    def get_class(cls, name: str) -> type[DataSource]:
        """获取数据源类（不实例化）

        Parameters
        ----------
        name : str
            已注册的数据源名称

        Returns
        -------
        type[DataSource]
            数据源类
        """
        if name not in cls._sources:
            available = ", ".join(sorted(cls._sources.keys())) or "(none)"
            raise ValueError(
                f"Unknown data source: '{name}'. Available: {available}"
            )
        return cls._sources[name]

    @classmethod
    def list_sources(cls) -> list[str]:
        """列出所有已注册的数据源名称"""
        return sorted(cls._sources.keys())

    @classmethod
    def clear(cls) -> None:
        """清除所有已注册的数据源（主要用于测试）"""
        cls._sources.clear()
