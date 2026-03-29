"""交易引擎模块 - 实时/模拟交易执行

提供完整的交易执行链路：
- OrderManager: 订单生命周期管理
- PositionTracker: 持仓跟踪与盈亏计算
- ExecutionEngine: 信号执行与撮合引擎
- RiskMonitor: 实时风控监控
- TradeLogger: 交易日志与统计
"""

from quant_trading.trading.execution_engine import (
    ExecutionEngine,
    LiveExecutor,
    SimulatedExecutor,
)
from quant_trading.trading.order_manager import OrderManager
from quant_trading.trading.position_tracker import PositionTracker
from quant_trading.trading.risk_monitor import (
    ConcentrationRule,
    DailyLossLimitRule,
    MaxDrawdownRule,
    MaxPositionRule,
    RiskMonitor,
    RiskRule,
)
from quant_trading.trading.trade_logger import TradeLogger

__all__ = [
    "OrderManager",
    "PositionTracker",
    "ExecutionEngine",
    "SimulatedExecutor",
    "LiveExecutor",
    "RiskMonitor",
    "RiskRule",
    "MaxPositionRule",
    "MaxDrawdownRule",
    "DailyLossLimitRule",
    "ConcentrationRule",
    "TradeLogger",
]
