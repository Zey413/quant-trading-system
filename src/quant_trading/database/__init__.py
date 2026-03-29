"""数据库模块 - SQLite存储引擎

提供量化交易系统的持久化层:
- 行情数据存储 (StockDaily, StockInfo)
- 交易记录 (TradeRecord)
- 回测结果 (BacktestResult)
- 策略参数 (StrategyParam)
- 投资组合快照 (PortfolioSnapshot)
- 自选股 (WatchlistItem)
- 告警 (Alert)

所有数据库操作使用纯 sqlite3，参数化查询防注入，线程安全。
"""

from quant_trading.database.connection import DatabaseManager
from quant_trading.database.models import (
    Alert,
    BacktestResult,
    PortfolioSnapshot,
    StockDaily,
    StockInfo,
    StrategyParam,
    TradeRecord,
    WatchlistItem,
)
from quant_trading.database.repositories import (
    BacktestRepository,
    PortfolioRepository,
    StockDataRepository,
    TradeRepository,
    WatchlistRepository,
)

__all__ = [
    # 连接管理
    "DatabaseManager",
    # 数据模型
    "StockDaily",
    "StockInfo",
    "TradeRecord",
    "BacktestResult",
    "StrategyParam",
    "PortfolioSnapshot",
    "WatchlistItem",
    "Alert",
    # 仓库
    "StockDataRepository",
    "TradeRepository",
    "BacktestRepository",
    "PortfolioRepository",
    "WatchlistRepository",
]
