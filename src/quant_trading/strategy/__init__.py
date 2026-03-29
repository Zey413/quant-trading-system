"""交易策略模块

提供策略基类、注册表，以及多种内置策略。
"""

from quant_trading.strategy.base import Strategy, StrategyRegistry

# 导入具体策略以触发 @register 装饰器注册
from quant_trading.strategy.bollinger_strategy import BollingerStrategy
from quant_trading.strategy.composite import CompositeStrategy
from quant_trading.strategy.dual_thrust import DualThrustStrategy
from quant_trading.strategy.ma_crossover import MACrossoverStrategy
from quant_trading.strategy.macd_strategy import MACDStrategy
from quant_trading.strategy.mean_reversion import MeanReversionStrategy
from quant_trading.strategy.rsi_strategy import RSIStrategy
from quant_trading.strategy.turtle import TurtleStrategy

__all__ = [
    "Strategy",
    "StrategyRegistry",
    "MACrossoverStrategy",
    "RSIStrategy",
    "MACDStrategy",
    "BollingerStrategy",
    "CompositeStrategy",
    "DualThrustStrategy",
    "MeanReversionStrategy",
    "TurtleStrategy",
]
