"""ML模型单元测试

至少15个测试用例，覆盖 DecisionTreeModel、RandomForestModel、LinearModel、
GradientBoostingModel、LSTMModel 和 EnsembleModel。测试数据使用numpy生成。
"""

from __future__ import annotations

import json
import os
import tempfile

import numpy as np
import pytest

from quant_trading.ml.models import (
    DecisionTreeModel,
    EnsembleModel,
    GradientBoostingModel,
    LinearModel,
    LSTMModel,
    RandomForestModel,
)


# ======================================================================
# 辅助函数
# ======================================================================


def _make_binary_data(
    n: int = 200, n_features: int = 5, seed: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    """生成二分类数据（线性可分+噪声）。"""
    rng = np.random.RandomState(seed)
    X = rng.randn(n, n_features)
    # 基于前两个特征的线性组合
    y = (X[:, 0] + X[:, 1] + rng.randn(n) * 0.3 > 0).astype(float)
    return X, y


def _make_multiclass_data(
    n: int = 300, n_features: int = 5, n_classes: int = 3, seed: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    """生成多分类数据。"""
    rng = np.random.RandomState(seed)
    X = rng.randn(n, n_features)
    boundaries = np.linspace(-1.5, 1.5, n_classes + 1)[1:-1]
    y = np.digitize(X[:, 0] + rng.randn(n) * 0.3, boundaries).astype(float)
    return X, y


# ======================================================================
# DecisionTreeModel 测试
# ======================================================================


class TestDecisionTreeModel:

    def test_train_and_predict(self):
        """训练后应能预测。"""
        X, y = _make_binary_data()
        model = DecisionTreeModel(max_depth=3, random_state=42)
        model.train(X, y)
        preds = model.predict(X)
        assert preds.shape == (len(X),)
        # 训练集准确率应显著高于随机
        acc = np.mean(preds == y)
        assert acc > 0.6

    def test_predict_proba_sums_to_one(self):
        """预测概率每行应归一。"""
        X, y = _make_binary_data()
        model = DecisionTreeModel(max_depth=3, random_state=42)
        model.train(X, y)
        proba = model.predict_proba(X)
        row_sums = proba.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-10)

    def test_feature_importance(self):
        """特征重要性应可用且归一。"""
        X, y = _make_binary_data()
        model = DecisionTreeModel(max_depth=3, random_state=42)
        model.train(X, y)
        imp = model.feature_importance()
        assert len(imp) == X.shape[1]
        np.testing.assert_allclose(imp.sum(), 1.0, atol=1e-6)

    def test_save_and_load(self):
        """保存并加载后预测应一致。"""
        X, y = _make_binary_data()
        model = DecisionTreeModel(max_depth=3, random_state=42)
        model.train(X, y)
        preds_before = model.predict(X[:10])

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            model.save(path)
            model2 = DecisionTreeModel()
            model2.load(path)
            preds_after = model2.predict(X[:10])
            np.testing.assert_array_equal(preds_before, preds_after)
        finally:
            os.unlink(path)


# ======================================================================
# RandomForestModel 测试
# ======================================================================


class TestRandomForestModel:

    def test_train_and_predict(self):
        """随机森林应能训练和预测。"""
        X, y = _make_binary_data()
        model = RandomForestModel(n_estimators=5, max_depth=3, random_state=42)
        model.train(X, y)
        preds = model.predict(X)
        acc = np.mean(preds == y)
        assert acc > 0.6

    def test_better_than_single_tree(self):
        """随机森林通常应比单棵树更好或相当。"""
        X, y = _make_binary_data()
        tree = DecisionTreeModel(max_depth=3, random_state=42)
        tree.train(X, y)
        tree_acc = np.mean(tree.predict(X) == y)

        forest = RandomForestModel(n_estimators=10, max_depth=3, random_state=42)
        forest.train(X, y)
        forest_acc = np.mean(forest.predict(X) == y)

        # 森林准确率应不低于单棵树太多
        assert forest_acc >= tree_acc - 0.1

    def test_feature_importance_shape(self):
        """特征重要性维度应正确。"""
        X, y = _make_binary_data(n_features=8)
        model = RandomForestModel(n_estimators=5, max_depth=3, random_state=42)
        model.train(X, y)
        imp = model.feature_importance()
        assert len(imp) == 8


# ======================================================================
# LinearModel 测试
# ======================================================================


class TestLinearModel:

    def test_binary_classification(self):
        """二分类应能工作。"""
        X, y = _make_binary_data(n=300)
        model = LinearModel(learning_rate=0.1, n_iterations=500, regularization=0.01)
        model.train(X, y)
        preds = model.predict(X)
        acc = np.mean(preds == y)
        assert acc > 0.6

    def test_multiclass_classification(self):
        """多分类应能工作。"""
        X, y = _make_multiclass_data()
        model = LinearModel(learning_rate=0.1, n_iterations=500)
        model.train(X, y)
        preds = model.predict(X)
        # 三分类随机准确率约1/3
        acc = np.mean(preds == y)
        assert acc > 0.35

    def test_predict_proba_shape(self):
        """概率矩阵形状应正确。"""
        X, y = _make_binary_data()
        model = LinearModel(n_iterations=100)
        model.train(X, y)
        proba = model.predict_proba(X)
        assert proba.shape == (len(X), 2)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)


# ======================================================================
# GradientBoostingModel 测试
# ======================================================================


class TestGradientBoostingModel:

    def test_train_and_predict(self):
        """GBDT应能训练和预测。"""
        X, y = _make_binary_data()
        model = GradientBoostingModel(
            n_estimators=20, max_depth=2, learning_rate=0.1, random_state=42
        )
        model.train(X, y)
        preds = model.predict(X)
        acc = np.mean(preds == y)
        assert acc > 0.65

    def test_multiclass(self):
        """多分类GBDT应能工作。"""
        X, y = _make_multiclass_data()
        model = GradientBoostingModel(
            n_estimators=15, max_depth=2, learning_rate=0.1, random_state=42
        )
        model.train(X, y)
        preds = model.predict(X)
        acc = np.mean(preds == y)
        assert acc > 0.35

    def test_save_and_load(self):
        """GBDT保存加载后预测应一致。"""
        X, y = _make_binary_data(n=100)
        model = GradientBoostingModel(
            n_estimators=10, max_depth=2, random_state=42
        )
        model.train(X, y)
        preds_before = model.predict(X[:10])

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            model.save(path)
            model2 = GradientBoostingModel()
            model2.load(path)
            preds_after = model2.predict(X[:10])
            np.testing.assert_array_equal(preds_before, preds_after)
        finally:
            os.unlink(path)


# ======================================================================
# LSTMModel 测试
# ======================================================================


class TestLSTMModel:

    def test_train_and_predict(self):
        """LSTM应能训练和预测。"""
        rng = np.random.RandomState(42)
        X = rng.randn(50, 3)
        y = (X[:, 0] > 0).astype(float)

        model = LSTMModel(hidden_size=4, seq_len=5, n_epochs=5, learning_rate=0.01)
        model.train(X, y)
        preds = model.predict(X)
        assert len(preds) > 0

    def test_predict_proba_shape(self):
        """LSTM概率输出形状应正确。"""
        rng = np.random.RandomState(42)
        X = rng.randn(50, 3)
        y = (X[:, 0] > 0).astype(float)

        model = LSTMModel(hidden_size=4, seq_len=5, n_epochs=3)
        model.train(X, y)
        proba = model.predict_proba(X)
        assert proba.ndim == 2
        assert proba.shape[1] == 2


# ======================================================================
# EnsembleModel 测试
# ======================================================================


class TestEnsembleModel:

    def test_average_ensemble(self):
        """概率平均集成应能工作。"""
        X, y = _make_binary_data()
        ensemble = EnsembleModel(method="average")
        ensemble.add_model(DecisionTreeModel(max_depth=3, random_state=42))
        ensemble.add_model(LinearModel(n_iterations=200))
        ensemble.train(X, y)
        preds = ensemble.predict(X)
        acc = np.mean(preds == y)
        assert acc > 0.55

    def test_vote_ensemble(self):
        """投票集成应能工作。"""
        X, y = _make_binary_data()
        ensemble = EnsembleModel(method="vote")
        ensemble.add_model(
            DecisionTreeModel(max_depth=3, random_state=42), weight=1.0
        )
        ensemble.add_model(
            RandomForestModel(n_estimators=5, max_depth=3, random_state=42),
            weight=2.0,
        )
        ensemble.train(X, y)
        preds = ensemble.predict(X)
        acc = np.mean(preds == y)
        assert acc > 0.55

    def test_ensemble_feature_importance(self):
        """集成模型应能返回特征重要性。"""
        X, y = _make_binary_data(n_features=5)
        ensemble = EnsembleModel(method="average")
        ensemble.add_model(DecisionTreeModel(max_depth=3, random_state=42))
        ensemble.add_model(RandomForestModel(n_estimators=3, max_depth=2, random_state=42))
        ensemble.train(X, y)
        imp = ensemble.feature_importance()
        assert len(imp) == 5


# ======================================================================
# 通用测试
# ======================================================================


class TestModelGeneral:

    def test_untrained_model_raises(self):
        """未训练模型预测应报错。"""
        model = DecisionTreeModel()
        with pytest.raises(RuntimeError):
            model.predict(np.array([[1, 2, 3]]))

    def test_repr(self):
        """repr应显示训练状态。"""
        model = DecisionTreeModel(name="test_tree")
        assert "untrained" in repr(model)
        X, y = _make_binary_data(n=50)
        model.train(X, y)
        assert "trained" in repr(model)
