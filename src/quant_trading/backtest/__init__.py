"""回测引擎模块"""

from quant_trading.backtest.broker import SimulatedBroker
from quant_trading.backtest.comparator import ComparisonResult, StrategyComparator
from quant_trading.backtest.engine import BacktestEngine
from quant_trading.backtest.metrics import PerformanceMetrics, PerformanceResult
from quant_trading.backtest.optimizer import GridSearchOptimizer, OptimizationResult
from quant_trading.backtest.portfolio import PortfolioManager
from quant_trading.backtest.portfolio_backtest import (
    PortfolioBacktester,
    PortfolioBacktestResult,
)
from quant_trading.backtest.report import ReportGenerator
from quant_trading.backtest.signal_analyzer import SignalAnalysisResult, SignalAnalyzer
from quant_trading.backtest.walk_forward import (
    WalkForwardAnalyzer,
    WalkForwardResult,
    WalkForwardWindow,
)

__all__ = [
    "BacktestEngine",
    "ComparisonResult",
    "GridSearchOptimizer",
    "OptimizationResult",
    "PerformanceMetrics",
    "PerformanceResult",
    "PortfolioBacktester",
    "PortfolioBacktestResult",
    "PortfolioManager",
    "ReportGenerator",
    "SignalAnalysisResult",
    "SignalAnalyzer",
    "SimulatedBroker",
    "StrategyComparator",
    "WalkForwardAnalyzer",
    "WalkForwardResult",
    "WalkForwardWindow",
]
