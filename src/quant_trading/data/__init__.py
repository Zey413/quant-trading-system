"""数据源模块 - 可插拔数据源架构

提供统一的数据获取接口，支持多种数据源（AKShare、Tushare等）。
包含实时行情、财务数据、数据缓存和速率限制等子模块。

用法::

    from quant_trading.core.config import AppConfig
    from quant_trading.data import DataManager

    config = AppConfig()
    dm = DataManager(config)
    df = dm.fetch_daily("000001", "20230101", "20231231")

    # 实时行情
    from quant_trading.data import RealtimeQuoteProvider
    provider = RealtimeQuoteProvider()
    quote = provider.get_realtime_quote("000001")

    # 财务数据
    from quant_trading.data import FinancialDataProvider
    fp = FinancialDataProvider()
    info = fp.get_stock_info("000001")

    # 数据缓存
    from quant_trading.data import DataCache, cached
    cache = DataCache()

    # 速率限制
    from quant_trading.data import RateLimiter, rate_limited
"""

from quant_trading.data.base import DataSource, DataSourceRegistry
from quant_trading.data.cache import DataCache, cached
from quant_trading.data.export import DataExporter
from quant_trading.data.financial import FinancialDataProvider
from quant_trading.data.manager import DataManager
from quant_trading.data.rate_limiter import RateLimiter, rate_limited
from quant_trading.data.realtime import RealtimeQuote, RealtimeQuoteProvider, RealtimeDataStream

# 导入具体数据源，触发 @DataSourceRegistry.register() 装饰器注册
import quant_trading.data.akshare_source as _akshare_source  # noqa: F401
import quant_trading.data.tushare_source as _tushare_source  # noqa: F401

__all__ = [
    "DataSource",
    "DataSourceRegistry",
    "DataExporter",
    "DataManager",
    "RealtimeQuote",
    "RealtimeQuoteProvider",
    "RealtimeDataStream",
    "FinancialDataProvider",
    "DataCache",
    "cached",
    "RateLimiter",
    "rate_limited",
]
