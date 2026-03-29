"""A股量化交易系统 - 支持可插拔数据源、多策略回测、风险管理"""

__version__ = "0.1.0"

from quant_trading.core.config import AppConfig
from quant_trading.core.enums import (
    AdjustType,
    FrequencyType,
    OrderSide,
    OrderStatus,
    OrderType,
    PositionSizingMethod,
    SignalType,
    StopLossMethod,
)
from quant_trading.core.models import Order, Portfolio, Position, Signal, TradeRecord
from quant_trading.strategy.base import Strategy, StrategyRegistry

__all__ = [
    # Version
    "__version__",
    # Config
    "AppConfig",
    # Enums
    "AdjustType",
    "FrequencyType",
    "OrderSide",
    "OrderStatus",
    "OrderType",
    "PositionSizingMethod",
    "SignalType",
    "StopLossMethod",
    # Models
    "Order",
    "Portfolio",
    "Position",
    "Signal",
    "TradeRecord",
    # Strategy
    "Strategy",
    "StrategyRegistry",
]
