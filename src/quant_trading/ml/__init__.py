"""机器学习策略模块

提供特征工程、ML模型（纯numpy实现）、训练流水线、模型评估，
以及与策略注册表集成的ML交易策略。
"""

from quant_trading.ml.evaluation import ModelEvaluator
from quant_trading.ml.features import FeatureEngineer
from quant_trading.ml.models import (
    DecisionTreeModel,
    EnsembleModel,
    GradientBoostingModel,
    LinearModel,
    LSTMModel,
    MLModelBase,
    RandomForestModel,
)
from quant_trading.ml.pipeline import MLPipeline, MinMaxScaler, StandardScaler

__all__ = [
    # 特征工程
    "FeatureEngineer",
    # 模型
    "MLModelBase",
    "DecisionTreeModel",
    "RandomForestModel",
    "LinearModel",
    "GradientBoostingModel",
    "LSTMModel",
    "EnsembleModel",
    # 流水线
    "MLPipeline",
    "StandardScaler",
    "MinMaxScaler",
    # 评估
    "ModelEvaluator",
]
