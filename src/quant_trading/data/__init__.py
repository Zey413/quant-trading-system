"""数据源模块 - 可插拔数据源架构

提供统一的数据获取接口，支持多种数据源（AKShare、Tushare等）。

用法::

    from quant_trading.core.config import AppConfig
    from quant_trading.data import DataManager

    config = AppConfig()
    dm = DataManager(config)
    df = dm.fetch_daily("000001", "20230101", "20231231")
"""

from quant_trading.data.base import DataSource, DataSourceRegistry
from quant_trading.data.export import DataExporter
from quant_trading.data.manager import DataManager

# 导入具体数据源，触发 @DataSourceRegistry.register() 装饰器注册
import quant_trading.data.akshare_source as _akshare_source  # noqa: F401
import quant_trading.data.tushare_source as _tushare_source  # noqa: F401

__all__ = ["DataSource", "DataSourceRegistry", "DataExporter", "DataManager"]
