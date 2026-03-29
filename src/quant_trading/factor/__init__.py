"""因子分析模块 - 提供因子定义、计算、分析和评估

包含:
- Factor: 因子抽象基类
- FactorRegistry: 因子注册表
- 动量因子 (MomentumFactor)
- 价值因子 (PEFactor, PBFactor, PSFactor)
- 质量因子 (ROEFactor, ROAFactor)
- FactorAnalyzer: 因子分析器 (IC/IR/分层回测)
"""

from quant_trading.factor.base import Factor, FactorRegistry
from quant_trading.factor.momentum import MomentumFactor, ReverseReturnFactor
from quant_trading.factor.value import PBFactor, PEFactor, PSFactor
from quant_trading.factor.quality import ROAFactor, ROEFactor
from quant_trading.factor.analyzer import FactorAnalyzer, FactorAnalysisResult

__all__ = [
    "Factor",
    "FactorRegistry",
    "MomentumFactor",
    "ReverseReturnFactor",
    "PEFactor",
    "PBFactor",
    "PSFactor",
    "ROEFactor",
    "ROAFactor",
    "FactorAnalyzer",
    "FactorAnalysisResult",
]
