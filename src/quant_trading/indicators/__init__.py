"""技术指标模块

提供趋势、振荡、波动率、成交量四大类技术指标。
"""

from quant_trading.indicators.base import Indicator
from quant_trading.indicators.oscillator import KDJ, RSI
from quant_trading.indicators.trend import EMA, MACD, SMA
from quant_trading.indicators.volatility import ATR, BollingerBands
from quant_trading.indicators.volume import OBV, VWAP

__all__ = [
    "Indicator",
    # 趋势
    "SMA",
    "EMA",
    "MACD",
    # 振荡
    "RSI",
    "KDJ",
    # 波动率
    "BollingerBands",
    "ATR",
    # 成交量
    "OBV",
    "VWAP",
]
