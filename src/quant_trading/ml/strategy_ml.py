"""ML交易策略模块

将机器学习模型集成到策略系统中，继承 StrategyBase 并注册到 StrategyRegistry。
提供 GradientBoostingStrategy、RandomForestStrategy 和 EnsembleMLStrategy。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_trading.ml.features import FeatureEngineer
from quant_trading.ml.models import (
    EnsembleModel,
    GradientBoostingModel,
    LinearModel,
    RandomForestModel,
)
from quant_trading.ml.pipeline import MLPipeline, StandardScaler
from quant_trading.strategy.base import Strategy, StrategyRegistry


class _MLStrategyMixin:
    """ML策略的通用Mixin，提供训练和预测流程。

    所有ML策略共享特征工程、Pipeline构建和信号生成逻辑。
    """

    def _init_ml(
        self,
        retrain_interval: int = 60,
        lookback: int = 200,
        signal_threshold: float = 0.55,
        label_method: str = "direction",
        label_threshold: float = 0.0,
        label_horizon: int = 1,
    ) -> None:
        """初始化ML相关参数。

        Parameters
        ----------
        retrain_interval : int
            每隔多少个bar重新训练模型。
        lookback : int
            训练窗口大小（bar数量）。
        signal_threshold : float
            概率阈值，高于此值发出买入信号，低于(1-此值)发出卖出信号。
        label_method : str
            标签生成方法。
        label_threshold : float
            标签阈值。
        label_horizon : int
            标签前瞻周期。
        """
        self.retrain_interval = retrain_interval
        self.lookback = lookback
        self.signal_threshold = signal_threshold
        self._feature_engineer = FeatureEngineer(
            label_method=label_method,
            label_threshold=label_threshold,
            label_horizon=label_horizon,
        )
        self._pipeline: MLPipeline | None = None
        self._last_train_idx: int = 0
        self._is_ml_trained = False

    def _build_pipeline(self, model) -> MLPipeline:
        """构建含Scaler和模型的Pipeline。"""
        pipeline = MLPipeline()
        pipeline.add_step("scaler", StandardScaler())
        pipeline.add_step("model", model)
        return pipeline

    def _generate_ml_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """通用ML信号生成逻辑。

        Parameters
        ----------
        data : pd.DataFrame
            至少包含OHLCV列的行情数据。

        Returns
        -------
        pd.DataFrame
            添加了 signal 列的 DataFrame。
        """
        data = data.copy()
        data["signal"] = "hold"

        n = len(data)
        if n < self.lookback + 10:
            return data

        # 判断是否需要重新训练
        should_train = (
            not self._is_ml_trained
            or (n - self._last_train_idx) >= self.retrain_interval
        )

        if should_train:
            # 使用最近的lookback条数据训练
            train_start = max(0, n - self.lookback - self._feature_engineer.label_horizon)
            train_data = data.iloc[train_start:n - self._feature_engineer.label_horizon].copy()

            try:
                X, y, feature_names = self._feature_engineer.build_feature_matrix(
                    train_data
                )
                if len(X) > 20 and len(np.unique(y)) >= 2:
                    self._pipeline = self._build_pipeline(self._create_model())
                    self._pipeline.fit(X, y)
                    self._is_ml_trained = True
                    self._last_train_idx = n
            except Exception:
                # 训练失败时保持hold
                return data

        if not self._is_ml_trained or self._pipeline is None:
            return data

        # 对最新数据进行预测
        try:
            # 构建最新特征
            recent_data = data.iloc[max(0, n - 100):].copy()
            X_recent, _, _ = self._feature_engineer.build_feature_matrix(
                recent_data
            )

            if len(X_recent) == 0:
                return data

            proba = self._pipeline.predict_proba(X_recent)

            # 最后一行的预测
            last_proba = proba[-1]

            if len(last_proba) >= 2:
                # 二分类: last_proba[1] 是上涨概率
                buy_prob = last_proba[1] if len(last_proba) == 2 else max(last_proba)

                if buy_prob > self.signal_threshold:
                    data.iloc[-1, data.columns.get_loc("signal")] = "buy"
                elif buy_prob < (1 - self.signal_threshold):
                    data.iloc[-1, data.columns.get_loc("signal")] = "sell"

        except Exception:
            pass

        return data

    def _create_model(self):
        """子类实现，返回具体ML模型实例。"""
        raise NotImplementedError


# ======================================================================
# 梯度提升策略
# ======================================================================


@StrategyRegistry.register("ml_gradient_boosting")
class GradientBoostingStrategy(Strategy, _MLStrategyMixin):
    """基于梯度提升树的ML交易策略。

    使用GradientBoostingModel预测价格方向，产生买卖信号。

    Parameters
    ----------
    n_estimators : int
        提升轮数，默认 30。
    max_depth : int
        树深度，默认 3。
    learning_rate : float
        学习率，默认 0.1。
    retrain_interval : int
        重训练间隔（bar数），默认 60。
    lookback : int
        训练窗口大小，默认 200。
    signal_threshold : float
        信号概率阈值，默认 0.55。
    """

    def __init__(
        self,
        n_estimators: int = 30,
        max_depth: int = 3,
        learning_rate: float = 0.1,
        retrain_interval: int = 60,
        lookback: int = 200,
        signal_threshold: float = 0.55,
        name: str = "",
    ) -> None:
        Strategy.__init__(self, name=name)
        self._init_ml(
            retrain_interval=retrain_interval,
            lookback=lookback,
            signal_threshold=signal_threshold,
        )
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.gb_learning_rate = learning_rate

    def _create_model(self):
        return GradientBoostingModel(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.gb_learning_rate,
            random_state=42,
        )

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """生成交易信号。"""
        return self._generate_ml_signals(data)

    def get_params(self) -> dict:
        return {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "learning_rate": self.gb_learning_rate,
            "retrain_interval": self.retrain_interval,
            "lookback": self.lookback,
            "signal_threshold": self.signal_threshold,
        }


# ======================================================================
# 随机森林策略
# ======================================================================


@StrategyRegistry.register("ml_random_forest")
class RandomForestStrategy(Strategy, _MLStrategyMixin):
    """基于随机森林的ML交易策略。

    Parameters
    ----------
    n_estimators : int
        树的数量，默认 10。
    max_depth : int
        树深度，默认 5。
    retrain_interval : int
        重训练间隔（bar数），默认 60。
    lookback : int
        训练窗口大小，默认 200。
    signal_threshold : float
        信号概率阈值，默认 0.55。
    """

    def __init__(
        self,
        n_estimators: int = 10,
        max_depth: int = 5,
        retrain_interval: int = 60,
        lookback: int = 200,
        signal_threshold: float = 0.55,
        name: str = "",
    ) -> None:
        Strategy.__init__(self, name=name)
        self._init_ml(
            retrain_interval=retrain_interval,
            lookback=lookback,
            signal_threshold=signal_threshold,
        )
        self.n_estimators = n_estimators
        self.max_depth = max_depth

    def _create_model(self):
        return RandomForestModel(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=42,
        )

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """生成交易信号。"""
        return self._generate_ml_signals(data)

    def get_params(self) -> dict:
        return {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "retrain_interval": self.retrain_interval,
            "lookback": self.lookback,
            "signal_threshold": self.signal_threshold,
        }


# ======================================================================
# 集成策略
# ======================================================================


@StrategyRegistry.register("ml_ensemble")
class EnsembleMLStrategy(Strategy, _MLStrategyMixin):
    """基于模型集成的ML交易策略。

    集成随机森林、梯度提升和逻辑回归三个模型，通过概率加权平均
    生成更稳健的交易信号。

    Parameters
    ----------
    rf_n_estimators : int
        随机森林树数量，默认 8。
    gb_n_estimators : int
        梯度提升轮数，默认 20。
    weights : list[float] | None
        各模型权重 [RF, GB, LR]，默认等权。
    retrain_interval : int
        重训练间隔。
    lookback : int
        训练窗口大小。
    signal_threshold : float
        信号概率阈值。
    """

    def __init__(
        self,
        rf_n_estimators: int = 8,
        gb_n_estimators: int = 20,
        weights: list[float] | None = None,
        retrain_interval: int = 60,
        lookback: int = 200,
        signal_threshold: float = 0.55,
        name: str = "",
    ) -> None:
        Strategy.__init__(self, name=name)
        self._init_ml(
            retrain_interval=retrain_interval,
            lookback=lookback,
            signal_threshold=signal_threshold,
        )
        self.rf_n_estimators = rf_n_estimators
        self.gb_n_estimators = gb_n_estimators
        self.ensemble_weights = weights or [1.0, 1.0, 1.0]

    def _create_model(self):
        ensemble = EnsembleModel(method="average")
        ensemble.add_model(
            RandomForestModel(
                n_estimators=self.rf_n_estimators,
                max_depth=4,
                random_state=42,
            ),
            weight=self.ensemble_weights[0],
        )
        ensemble.add_model(
            GradientBoostingModel(
                n_estimators=self.gb_n_estimators,
                max_depth=3,
                random_state=42,
            ),
            weight=self.ensemble_weights[1],
        )
        ensemble.add_model(
            LinearModel(n_iterations=500, regularization=0.1),
            weight=self.ensemble_weights[2],
        )
        return ensemble

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """生成交易信号。"""
        return self._generate_ml_signals(data)

    def get_params(self) -> dict:
        return {
            "rf_n_estimators": self.rf_n_estimators,
            "gb_n_estimators": self.gb_n_estimators,
            "weights": self.ensemble_weights,
            "retrain_interval": self.retrain_interval,
            "lookback": self.lookback,
            "signal_threshold": self.signal_threshold,
        }
