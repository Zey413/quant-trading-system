"""ML Pipeline 单元测试

至少10个测试用例，覆盖 StandardScaler、MinMaxScaler、MLPipeline
的 fit/predict/cross_validate/walk_forward_validate 功能。
"""

from __future__ import annotations

import numpy as np
import pytest

from quant_trading.ml.models import (
    DecisionTreeModel,
    LinearModel,
    RandomForestModel,
)
from quant_trading.ml.pipeline import (
    CVResult,
    MLPipeline,
    MinMaxScaler,
    StandardScaler,
    WFResult,
)


# ======================================================================
# 辅助函数
# ======================================================================


def _make_data(
    n: int = 200, n_features: int = 5, seed: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    """生成简单的二分类数据。"""
    rng = np.random.RandomState(seed)
    X = rng.randn(n, n_features)
    y = (X[:, 0] + X[:, 1] + rng.randn(n) * 0.3 > 0).astype(float)
    return X, y


# ======================================================================
# StandardScaler 测试
# ======================================================================


class TestStandardScaler:

    def test_transform_zero_mean_unit_std(self):
        """标准化后均值应接近0，标准差接近1。"""
        rng = np.random.RandomState(42)
        X = rng.randn(100, 5) * 10 + 50
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        np.testing.assert_allclose(X_scaled.mean(axis=0), 0.0, atol=1e-10)
        np.testing.assert_allclose(X_scaled.std(axis=0), 1.0, atol=0.05)

    def test_inverse_transform(self):
        """逆变换应恢复原始数据。"""
        rng = np.random.RandomState(42)
        X = rng.randn(50, 3) * 5 + 10
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_recovered = scaler.inverse_transform(X_scaled)
        np.testing.assert_allclose(X_recovered, X, atol=1e-10)

    def test_unfitted_raises(self):
        """未fit的scaler调用transform应报错。"""
        scaler = StandardScaler()
        with pytest.raises(RuntimeError, match="尚未fit"):
            scaler.transform(np.array([[1, 2, 3]]))

    def test_constant_column_handled(self):
        """常数列应不导致除零错误。"""
        X = np.ones((50, 3))
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        assert not np.any(np.isnan(X_scaled))


# ======================================================================
# MinMaxScaler 测试
# ======================================================================


class TestMinMaxScaler:

    def test_transform_range(self):
        """归一化后值应在[0,1]范围内。"""
        rng = np.random.RandomState(42)
        X = rng.randn(100, 5) * 100
        scaler = MinMaxScaler(feature_range=(0, 1))
        X_scaled = scaler.fit_transform(X)
        assert np.all(X_scaled >= -1e-10)
        assert np.all(X_scaled <= 1.0 + 1e-10)

    def test_custom_range(self):
        """自定义范围应生效。"""
        rng = np.random.RandomState(42)
        X = rng.randn(100, 3)
        scaler = MinMaxScaler(feature_range=(-1, 1))
        X_scaled = scaler.fit_transform(X)
        np.testing.assert_allclose(X_scaled.min(axis=0), -1.0, atol=1e-10)
        np.testing.assert_allclose(X_scaled.max(axis=0), 1.0, atol=1e-10)

    def test_inverse_transform(self):
        """逆变换应恢复原始数据。"""
        rng = np.random.RandomState(42)
        X = rng.randn(50, 3) * 5 + 10
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)
        X_recovered = scaler.inverse_transform(X_scaled)
        np.testing.assert_allclose(X_recovered, X, atol=1e-10)


# ======================================================================
# MLPipeline 测试
# ======================================================================


class TestMLPipeline:

    def test_fit_and_predict(self):
        """Pipeline应能正常fit和predict。"""
        X, y = _make_data()
        pipeline = MLPipeline()
        pipeline.add_step("scaler", StandardScaler())
        pipeline.add_step("model", DecisionTreeModel(max_depth=3, random_state=42))
        pipeline.fit(X, y)
        preds = pipeline.predict(X)
        assert preds.shape == (len(X),)
        acc = np.mean(preds == y)
        assert acc > 0.6

    def test_predict_before_fit_raises(self):
        """未fit的pipeline预测应报错。"""
        pipeline = MLPipeline()
        pipeline.add_step("scaler", StandardScaler())
        pipeline.add_step("model", DecisionTreeModel())
        with pytest.raises(RuntimeError, match="尚未训练"):
            pipeline.predict(np.array([[1, 2, 3, 4, 5]]))

    def test_chain_add_step(self):
        """add_step应支持链式调用。"""
        pipeline = (
            MLPipeline()
            .add_step("scaler", StandardScaler())
            .add_step("model", LinearModel(n_iterations=100))
        )
        assert len(pipeline._steps) == 2

    def test_get_step(self):
        """应能按名称获取步骤。"""
        pipeline = MLPipeline()
        scaler = StandardScaler()
        pipeline.add_step("scaler", scaler)
        pipeline.add_step("model", DecisionTreeModel())
        assert pipeline.get_step("scaler") is scaler

    def test_get_step_not_found(self):
        """不存在的步骤名应抛出KeyError。"""
        pipeline = MLPipeline()
        pipeline.add_step("model", DecisionTreeModel())
        with pytest.raises(KeyError, match="未找到步骤"):
            pipeline.get_step("nonexistent")

    def test_cross_validate(self):
        """交叉验证应返回有效结果。"""
        X, y = _make_data(n=200)
        pipeline = MLPipeline()
        pipeline.add_step("scaler", StandardScaler())
        pipeline.add_step("model", DecisionTreeModel(max_depth=3, random_state=42))

        result = pipeline.cross_validate(X, y, n_folds=3, metric="accuracy")
        assert isinstance(result, CVResult)
        assert len(result.scores) == 3
        assert 0.0 <= result.mean_score <= 1.0
        assert result.std_score >= 0.0
        assert len(result.fold_details) == 3

    def test_cross_validate_f1(self):
        """F1指标的交叉验证应工作。"""
        X, y = _make_data(n=200)
        pipeline = MLPipeline()
        pipeline.add_step("scaler", StandardScaler())
        pipeline.add_step("model", DecisionTreeModel(max_depth=3, random_state=42))

        result = pipeline.cross_validate(X, y, n_folds=3, metric="f1")
        assert isinstance(result, CVResult)
        assert all(0.0 <= s <= 1.0 for s in result.scores)

    def test_walk_forward_validate(self):
        """Walk-Forward验证应返回有效结果。"""
        X, y = _make_data(n=200)
        pipeline = MLPipeline()
        pipeline.add_step("scaler", StandardScaler())
        pipeline.add_step("model", DecisionTreeModel(max_depth=3, random_state=42))

        result = pipeline.walk_forward_validate(
            X, y, n_splits=3, train_ratio=0.5
        )
        assert isinstance(result, WFResult)
        assert len(result.scores) >= 1
        assert all(0.0 <= s <= 1.0 for s in result.scores)
        assert len(result.train_sizes) == len(result.scores)
        assert len(result.test_sizes) == len(result.scores)

    def test_walk_forward_train_before_test(self):
        """Walk-Forward中训练集应始终在测试集之前。"""
        X, y = _make_data(n=100)
        pipeline = MLPipeline()
        pipeline.add_step("scaler", StandardScaler())
        pipeline.add_step("model", DecisionTreeModel(max_depth=2, random_state=42))

        result = pipeline.walk_forward_validate(
            X, y, n_splits=3, train_ratio=0.5
        )
        # 训练集大小应递增
        for i in range(1, len(result.train_sizes)):
            assert result.train_sizes[i] >= result.train_sizes[i - 1]

    def test_repr(self):
        """repr应显示步骤信息。"""
        pipeline = MLPipeline()
        pipeline.add_step("scaler", StandardScaler())
        pipeline.add_step("model", DecisionTreeModel())
        r = repr(pipeline)
        assert "StandardScaler" in r
        assert "DecisionTreeModel" in r
        assert "unfitted" in r

    def test_cv_result_summary(self):
        """CVResult.summary 应返回格式化字符串。"""
        result = CVResult(
            scores=[0.8, 0.85, 0.9],
            mean_score=0.85,
            std_score=0.05,
        )
        summary = result.summary()
        assert "0.85" in summary
        assert "0.05" in summary

    def test_wf_result_summary(self):
        """WFResult.summary 应返回格式化字符串。"""
        result = WFResult(
            scores=[0.7, 0.75],
            mean_score=0.725,
            std_score=0.025,
        )
        summary = result.summary()
        assert "0.725" in summary
