"""ML训练流水线模块

提供 MLPipeline 类用于组装数据预处理和模型训练步骤，
以及 StandardScaler、MinMaxScaler 等纯numpy实现的预处理器。
支持交叉验证和Walk-Forward验证。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from quant_trading.ml.models import MLModelBase


# ======================================================================
# 预处理器协议
# ======================================================================


class Transformer(Protocol):
    """预处理器协议，所有预处理步骤需实现 fit / transform / inverse_transform。"""

    def fit(self, X: np.ndarray) -> None: ...
    def transform(self, X: np.ndarray) -> np.ndarray: ...
    def inverse_transform(self, X: np.ndarray) -> np.ndarray: ...


# ======================================================================
# Scaler
# ======================================================================


class StandardScaler:
    """标准化 (Z-Score)，纯numpy实现。

    将特征标准化为均值0、标准差1。

    Example
    -------
    >>> scaler = StandardScaler()
    >>> scaler.fit(X_train)
    >>> X_scaled = scaler.transform(X_train)
    """

    def __init__(self) -> None:
        self._mean: np.ndarray | None = None
        self._std: np.ndarray | None = None
        self._is_fitted = False

    def fit(self, X: np.ndarray) -> None:
        """计算均值和标准差。

        Parameters
        ----------
        X : np.ndarray
            训练数据, shape (n_samples, n_features)。
        """
        self._mean = np.nanmean(X, axis=0)
        self._std = np.nanstd(X, axis=0)
        # 防止除以零
        self._std[self._std < 1e-12] = 1.0
        self._is_fitted = True

    def transform(self, X: np.ndarray) -> np.ndarray:
        """标准化转换。

        Parameters
        ----------
        X : np.ndarray
            输入数据。

        Returns
        -------
        np.ndarray
            标准化后的数据。
        """
        if not self._is_fitted:
            raise RuntimeError("Scaler 尚未fit。请先调用 fit()。")
        return (X - self._mean) / self._std

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """逆标准化。

        Parameters
        ----------
        X : np.ndarray
            标准化后的数据。

        Returns
        -------
        np.ndarray
            原始尺度的数据。
        """
        if not self._is_fitted:
            raise RuntimeError("Scaler 尚未fit。请先调用 fit()。")
        return X * self._std + self._mean

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """fit后立即transform。"""
        self.fit(X)
        return self.transform(X)


class MinMaxScaler:
    """最小-最大归一化，纯numpy实现。

    将特征缩放到 [feature_range[0], feature_range[1]] 区间。

    Parameters
    ----------
    feature_range : tuple[float, float]
        目标值域，默认 (0, 1)。

    Example
    -------
    >>> scaler = MinMaxScaler(feature_range=(0, 1))
    >>> scaler.fit(X_train)
    >>> X_scaled = scaler.transform(X_train)
    """

    def __init__(self, feature_range: tuple[float, float] = (0.0, 1.0)) -> None:
        self.feature_range = feature_range
        self._min: np.ndarray | None = None
        self._max: np.ndarray | None = None
        self._is_fitted = False

    def fit(self, X: np.ndarray) -> None:
        """计算最小值和最大值。

        Parameters
        ----------
        X : np.ndarray
            训练数据, shape (n_samples, n_features)。
        """
        self._min = np.nanmin(X, axis=0)
        self._max = np.nanmax(X, axis=0)
        # 防止除以零
        diff = self._max - self._min
        diff[diff < 1e-12] = 1.0
        self._max = self._min + diff
        self._is_fitted = True

    def transform(self, X: np.ndarray) -> np.ndarray:
        """归一化转换。

        Parameters
        ----------
        X : np.ndarray
            输入数据。

        Returns
        -------
        np.ndarray
            归一化后的数据。
        """
        if not self._is_fitted:
            raise RuntimeError("Scaler 尚未fit。请先调用 fit()。")
        lo, hi = self.feature_range
        X_std = (X - self._min) / (self._max - self._min)
        return X_std * (hi - lo) + lo

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """逆归一化。

        Parameters
        ----------
        X : np.ndarray
            归一化后的数据。

        Returns
        -------
        np.ndarray
            原始尺度的数据。
        """
        if not self._is_fitted:
            raise RuntimeError("Scaler 尚未fit。请先调用 fit()。")
        lo, hi = self.feature_range
        X_std = (X - lo) / (hi - lo)
        return X_std * (self._max - self._min) + self._min

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """fit后立即transform。"""
        self.fit(X)
        return self.transform(X)


# ======================================================================
# 验证结果模型
# ======================================================================


@dataclass
class CVResult:
    """交叉验证结果。

    Attributes
    ----------
    scores : list[float]
        每折的评估分数。
    mean_score : float
        平均分数。
    std_score : float
        分数标准差。
    fold_details : list[dict]
        每折的详细信息。
    """

    scores: list[float] = field(default_factory=list)
    mean_score: float = 0.0
    std_score: float = 0.0
    fold_details: list[dict[str, Any]] = field(default_factory=list)

    def summary(self) -> str:
        """返回摘要字符串。"""
        return (
            f"CV Score: {self.mean_score:.4f} (+/- {self.std_score:.4f}), "
            f"folds: {len(self.scores)}"
        )


@dataclass
class WFResult:
    """Walk-Forward验证结果。

    Attributes
    ----------
    scores : list[float]
        每个窗口的评估分数。
    mean_score : float
        平均分数。
    std_score : float
        分数标准差。
    train_sizes : list[int]
        每个窗口的训练集大小。
    test_sizes : list[int]
        每个窗口的测试集大小。
    predictions : list[np.ndarray]
        每个窗口的预测结果。
    """

    scores: list[float] = field(default_factory=list)
    mean_score: float = 0.0
    std_score: float = 0.0
    train_sizes: list[int] = field(default_factory=list)
    test_sizes: list[int] = field(default_factory=list)
    predictions: list[np.ndarray] = field(default_factory=list)

    def summary(self) -> str:
        """返回摘要字符串。"""
        return (
            f"WF Score: {self.mean_score:.4f} (+/- {self.std_score:.4f}), "
            f"windows: {len(self.scores)}"
        )


# ======================================================================
# ML Pipeline
# ======================================================================


class MLPipeline:
    """机器学习训练流水线。

    支持添加预处理步骤和模型，提供 fit/predict/cross_validate/
    walk_forward_validate 等完整训练流程。

    Example
    -------
    >>> pipeline = MLPipeline()
    >>> pipeline.add_step("scaler", StandardScaler())
    >>> pipeline.add_step("model", RandomForestModel(n_estimators=10))
    >>> pipeline.fit(X_train, y_train)
    >>> predictions = pipeline.predict(X_test)
    """

    def __init__(self) -> None:
        self._steps: list[tuple[str, Any]] = []
        self._is_fitted = False

    def add_step(self, name: str, step: Any) -> MLPipeline:
        """添加流水线步骤。

        步骤可以是 Transformer（预处理器）或 MLModelBase（模型）。
        模型必须作为最后一个步骤。

        Parameters
        ----------
        name : str
            步骤名称。
        step : Any
            预处理器或模型实例。

        Returns
        -------
        MLPipeline
            返回自身，支持链式调用。
        """
        self._steps.append((name, step))
        return self

    def get_step(self, name: str) -> Any:
        """按名称获取步骤。

        Parameters
        ----------
        name : str
            步骤名称。

        Returns
        -------
        Any
            对应的步骤实例。

        Raises
        ------
        KeyError
            未找到步骤。
        """
        for step_name, step in self._steps:
            if step_name == name:
                return step
        raise KeyError(f"未找到步骤: {name!r}")

    @property
    def _transformers(self) -> list[tuple[str, Any]]:
        """返回所有预处理步骤（不含最后的模型）。"""
        if not self._steps:
            return []
        # 最后一个是模型
        return self._steps[:-1]

    @property
    def _model(self) -> MLModelBase:
        """返回最后一个步骤（模型）。"""
        if not self._steps:
            raise RuntimeError("Pipeline为空，请先添加步骤。")
        _, last = self._steps[-1]
        if not isinstance(last, MLModelBase):
            raise RuntimeError(
                f"Pipeline最后一步应为MLModelBase，收到 {type(last).__name__}"
            )
        return last

    def fit(self, X: np.ndarray, y: np.ndarray) -> MLPipeline:
        """训练整个流水线。

        依次对每个Transformer执行 fit+transform，最后对模型执行 train。

        Parameters
        ----------
        X : np.ndarray
            特征矩阵。
        y : np.ndarray
            标签向量。

        Returns
        -------
        MLPipeline
            返回自身。
        """
        X_transformed = X.copy()
        for name, step in self._transformers:
            step.fit(X_transformed)
            X_transformed = step.transform(X_transformed)

        self._model.train(X_transformed, y)
        self._is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测。

        Parameters
        ----------
        X : np.ndarray
            特征矩阵。

        Returns
        -------
        np.ndarray
            预测结果。
        """
        if not self._is_fitted:
            raise RuntimeError("Pipeline尚未训练。请先调用 fit()。")
        X_transformed = X.copy()
        for name, step in self._transformers:
            X_transformed = step.transform(X_transformed)
        return self._model.predict(X_transformed)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """预测概率。

        Parameters
        ----------
        X : np.ndarray
            特征矩阵。

        Returns
        -------
        np.ndarray
            预测概率矩阵。
        """
        if not self._is_fitted:
            raise RuntimeError("Pipeline尚未训练。请先调用 fit()。")
        X_transformed = X.copy()
        for name, step in self._transformers:
            X_transformed = step.transform(X_transformed)
        return self._model.predict_proba(X_transformed)

    def cross_validate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_folds: int = 5,
        metric: str = "accuracy",
        shuffle: bool = True,
        random_state: int | None = 42,
    ) -> CVResult:
        """K折交叉验证。

        Parameters
        ----------
        X : np.ndarray
            特征矩阵。
        y : np.ndarray
            标签向量。
        n_folds : int
            折数，默认 5。
        metric : str
            评估指标: 'accuracy', 'f1'。
        shuffle : bool
            是否打乱数据。
        random_state : int | None
            随机种子。

        Returns
        -------
        CVResult
            交叉验证结果。
        """
        n_samples = X.shape[0]
        indices = np.arange(n_samples)

        if shuffle:
            rng = np.random.RandomState(random_state)
            rng.shuffle(indices)

        fold_size = n_samples // n_folds
        result = CVResult()

        for fold in range(n_folds):
            # 划分训练集和验证集
            test_start = fold * fold_size
            test_end = (
                test_start + fold_size
                if fold < n_folds - 1
                else n_samples
            )
            test_idx = indices[test_start:test_end]
            train_idx = np.concatenate([
                indices[:test_start], indices[test_end:]
            ])

            X_train, y_train = X[train_idx], y[train_idx]
            X_test, y_test = X[test_idx], y[test_idx]

            # 克隆pipeline (重新创建步骤)
            temp_pipeline = self._clone()
            temp_pipeline.fit(X_train, y_train)
            y_pred = temp_pipeline.predict(X_test)

            score = self._compute_metric(y_test, y_pred, metric)
            result.scores.append(score)
            result.fold_details.append({
                "fold": fold,
                "train_size": len(train_idx),
                "test_size": len(test_idx),
                "score": score,
            })

        result.mean_score = float(np.mean(result.scores))
        result.std_score = float(np.std(result.scores))
        return result

    def walk_forward_validate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_splits: int = 5,
        train_ratio: float = 0.6,
        metric: str = "accuracy",
    ) -> WFResult:
        """Walk-Forward (滚动窗口) 验证。

        模拟真实时间序列预测：训练集始终在测试集之前。

        Parameters
        ----------
        X : np.ndarray
            特征矩阵 (按时间排序)。
        y : np.ndarray
            标签向量。
        n_splits : int
            滚动窗口数量。
        train_ratio : float
            初始训练集占总数据的比例。
        metric : str
            评估指标。

        Returns
        -------
        WFResult
            Walk-Forward验证结果。
        """
        n_samples = X.shape[0]
        init_train_size = int(n_samples * train_ratio)
        remaining = n_samples - init_train_size
        step_size = max(1, remaining // n_splits)

        result = WFResult()

        for i in range(n_splits):
            train_end = init_train_size + i * step_size
            test_end = min(train_end + step_size, n_samples)

            if train_end >= n_samples or test_end <= train_end:
                break

            X_train, y_train = X[:train_end], y[:train_end]
            X_test, y_test = X[train_end:test_end], y[train_end:test_end]

            if len(X_test) == 0:
                break

            temp_pipeline = self._clone()
            temp_pipeline.fit(X_train, y_train)
            y_pred = temp_pipeline.predict(X_test)

            score = self._compute_metric(y_test, y_pred, metric)
            result.scores.append(score)
            result.train_sizes.append(len(X_train))
            result.test_sizes.append(len(X_test))
            result.predictions.append(y_pred)

        if result.scores:
            result.mean_score = float(np.mean(result.scores))
            result.std_score = float(np.std(result.scores))
        return result

    def _clone(self) -> MLPipeline:
        """克隆Pipeline (创建新的步骤实例)。"""
        new_pipeline = MLPipeline()
        for name, step in self._steps:
            if isinstance(step, StandardScaler):
                new_step = StandardScaler()
            elif isinstance(step, MinMaxScaler):
                new_step = MinMaxScaler(feature_range=step.feature_range)
            elif isinstance(step, MLModelBase):
                # 使用相同参数创建新模型实例
                new_step = step.__class__(**self._extract_init_params(step))
            else:
                new_step = step  # fallback
            new_pipeline.add_step(name, new_step)
        return new_pipeline

    @staticmethod
    def _extract_init_params(model: MLModelBase) -> dict:
        """从模型实例提取构造参数。"""
        import inspect

        sig = inspect.signature(model.__class__.__init__)
        params = {}
        for p_name, param in sig.parameters.items():
            if p_name == "self":
                continue
            if hasattr(model, p_name):
                params[p_name] = getattr(model, p_name)
        return params

    @staticmethod
    def _compute_metric(
        y_true: np.ndarray, y_pred: np.ndarray, metric: str
    ) -> float:
        """计算评估指标。

        Parameters
        ----------
        y_true : np.ndarray
            真实标签。
        y_pred : np.ndarray
            预测标签。
        metric : str
            指标名称。

        Returns
        -------
        float
            指标值。
        """
        if metric == "accuracy":
            return float(np.mean(y_true == y_pred))
        elif metric == "f1":
            # 宏平均F1
            classes = np.unique(np.concatenate([y_true, y_pred]))
            f1_scores = []
            for c in classes:
                tp = np.sum((y_pred == c) & (y_true == c))
                fp = np.sum((y_pred == c) & (y_true != c))
                fn = np.sum((y_pred != c) & (y_true == c))
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1 = (
                    2 * precision * recall / (precision + recall)
                    if (precision + recall) > 0
                    else 0.0
                )
                f1_scores.append(f1)
            return float(np.mean(f1_scores)) if f1_scores else 0.0
        else:
            raise ValueError(f"不支持的评估指标: {metric!r}")

    def __repr__(self) -> str:
        steps_str = ", ".join(
            f"({name!r}, {type(step).__name__})" for name, step in self._steps
        )
        status = "fitted" if self._is_fitted else "unfitted"
        return f"MLPipeline([{steps_str}], {status})"
