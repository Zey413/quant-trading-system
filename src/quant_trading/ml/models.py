"""机器学习模型模块 (纯Python+numpy实现)

提供决策树、随机森林、逻辑回归、梯度提升树、简单LSTM和模型集成，
所有模型均不依赖sklearn/tensorflow等外部ML库。
"""

from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from typing import Any

import numpy as np


# ======================================================================
# 抽象基类
# ======================================================================


class MLModelBase(ABC):
    """ML模型抽象基类。

    所有模型必须实现 train、predict、predict_proba 方法，
    并支持 save/load 序列化。
    """

    def __init__(self, name: str = "") -> None:
        self.name = name or self.__class__.__name__
        self._is_trained = False
        self._feature_importances: np.ndarray | None = None

    @abstractmethod
    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """训练模型。

        Parameters
        ----------
        X : np.ndarray
            特征矩阵, shape (n_samples, n_features)。
        y : np.ndarray
            标签向量, shape (n_samples,)。
        """
        ...

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测类别标签。

        Parameters
        ----------
        X : np.ndarray
            特征矩阵, shape (n_samples, n_features)。

        Returns
        -------
        np.ndarray
            预测标签, shape (n_samples,)。
        """
        ...

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """预测类别概率。

        Parameters
        ----------
        X : np.ndarray
            特征矩阵, shape (n_samples, n_features)。

        Returns
        -------
        np.ndarray
            预测概率, shape (n_samples, n_classes)。
        """
        ...

    def feature_importance(self) -> np.ndarray:
        """返回特征重要性。

        Returns
        -------
        np.ndarray
            特征重要性向量，shape (n_features,)。
        """
        if self._feature_importances is None:
            raise RuntimeError("模型尚未训练或不支持特征重要性。")
        return self._feature_importances

    def save(self, path: str) -> None:
        """将模型参数保存为JSON文件。

        Parameters
        ----------
        path : str
            保存路径。
        """
        state = self._get_state()
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)

    def load(self, path: str) -> None:
        """从JSON文件加载模型参数。

        Parameters
        ----------
        path : str
            模型文件路径。
        """
        with open(path, "r", encoding="utf-8") as f:
            state = json.load(f)
        self._set_state(state)
        self._is_trained = True

    @abstractmethod
    def _get_state(self) -> dict:
        """序列化模型内部状态为字典。"""
        ...

    @abstractmethod
    def _set_state(self, state: dict) -> None:
        """从字典恢复模型内部状态。"""
        ...

    def __repr__(self) -> str:
        status = "trained" if self._is_trained else "untrained"
        return f"{self.__class__.__name__}(name={self.name!r}, {status})"


# ======================================================================
# 决策树
# ======================================================================


class _TreeNode:
    """决策树节点。"""

    __slots__ = [
        "feature_idx", "threshold", "left", "right",
        "value", "n_samples", "impurity",
    ]

    def __init__(self) -> None:
        self.feature_idx: int = -1
        self.threshold: float = 0.0
        self.left: _TreeNode | None = None
        self.right: _TreeNode | None = None
        self.value: np.ndarray | None = None  # 叶子节点的类别分布
        self.n_samples: int = 0
        self.impurity: float = 0.0


class DecisionTreeModel(MLModelBase):
    """CART决策树分类器 (纯numpy实现)。

    使用基尼系数进行节点分裂，支持最大深度和最小叶样本数限制。

    Parameters
    ----------
    max_depth : int
        最大树深度，默认 5。
    min_samples_split : int
        节点分裂所需最小样本数，默认 5。
    min_samples_leaf : int
        叶节点最小样本数，默认 2。
    max_features : int | None
        每次分裂时考虑的最大特征数（随机选择），None表示使用全部。
    random_state : int | None
        随机种子。
    """

    def __init__(
        self,
        max_depth: int = 5,
        min_samples_split: int = 5,
        min_samples_leaf: int = 2,
        max_features: int | None = None,
        random_state: int | None = None,
        name: str = "",
    ) -> None:
        super().__init__(name=name)
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state
        self._root: _TreeNode | None = None
        self._n_classes: int = 0
        self._classes: np.ndarray = np.array([])
        self._rng = np.random.RandomState(random_state)

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """训练决策树。"""
        self._classes = np.unique(y)
        self._n_classes = len(self._classes)
        # 将y映射为0..n_classes-1的整数索引
        self._class_map = {c: i for i, c in enumerate(self._classes)}
        y_mapped = np.array([self._class_map[v] for v in y], dtype=int)
        self._root = self._build_tree(X, y_mapped, depth=0)
        self._compute_feature_importance(X.shape[1])
        self._is_trained = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测类别。"""
        proba = self.predict_proba(X)
        indices = np.argmax(proba, axis=1)
        return self._classes[indices]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """预测类别概率。"""
        if self._root is None:
            raise RuntimeError("模型尚未训练。")
        n_samples = X.shape[0]
        proba = np.zeros((n_samples, self._n_classes))
        for i in range(n_samples):
            proba[i] = self._predict_one(X[i], self._root)
        return proba

    def _predict_one(self, x: np.ndarray, node: _TreeNode) -> np.ndarray:
        """对单个样本沿树进行预测。"""
        if node.left is None and node.right is None:
            return node.value  # type: ignore[return-value]
        if x[node.feature_idx] <= node.threshold:
            return self._predict_one(x, node.left)  # type: ignore[arg-type]
        return self._predict_one(x, node.right)  # type: ignore[arg-type]

    def _build_tree(
        self, X: np.ndarray, y: np.ndarray, depth: int
    ) -> _TreeNode:
        """递归构建决策树。"""
        node = _TreeNode()
        node.n_samples = len(y)
        node.impurity = self._gini(y)

        # 计算叶子值 (类别分布)
        counts = np.bincount(y, minlength=self._n_classes).astype(float)
        node.value = counts / counts.sum()

        # 终止条件
        if (
            depth >= self.max_depth
            or node.n_samples < self.min_samples_split
            or node.impurity < 1e-10
        ):
            return node

        # 寻找最佳分裂
        best_gain = 0.0
        best_feature = -1
        best_threshold = 0.0
        best_left_mask: np.ndarray | None = None

        n_features = X.shape[1]
        if self.max_features is not None:
            feature_indices = self._rng.choice(
                n_features, min(self.max_features, n_features), replace=False
            )
        else:
            feature_indices = np.arange(n_features)

        for f_idx in feature_indices:
            thresholds = np.unique(X[:, f_idx])
            if len(thresholds) <= 1:
                continue
            # 子采样阈值加速
            if len(thresholds) > 20:
                thresholds = np.percentile(
                    X[:, f_idx], np.linspace(5, 95, 20)
                )

            for t in thresholds:
                left_mask = X[:, f_idx] <= t
                right_mask = ~left_mask
                n_left = left_mask.sum()
                n_right = right_mask.sum()

                if n_left < self.min_samples_leaf or n_right < self.min_samples_leaf:
                    continue

                gini_left = self._gini(y[left_mask])
                gini_right = self._gini(y[right_mask])
                weighted_gini = (
                    n_left * gini_left + n_right * gini_right
                ) / node.n_samples
                gain = node.impurity - weighted_gini

                if gain > best_gain:
                    best_gain = gain
                    best_feature = int(f_idx)
                    best_threshold = float(t)
                    best_left_mask = left_mask

        if best_left_mask is None or best_gain <= 0:
            return node

        node.feature_idx = best_feature
        node.threshold = best_threshold
        left_mask = best_left_mask
        right_mask = ~left_mask
        node.left = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        node.right = self._build_tree(X[right_mask], y[right_mask], depth + 1)

        return node

    @staticmethod
    def _gini(y: np.ndarray) -> float:
        """计算基尼系数。"""
        if len(y) == 0:
            return 0.0
        counts = np.bincount(y)
        probs = counts / len(y)
        return float(1.0 - np.sum(probs ** 2))

    def _compute_feature_importance(self, n_features: int) -> None:
        """基于节点分裂增益累积计算特征重要性。"""
        importances = np.zeros(n_features)
        self._accumulate_importance(self._root, importances)
        total = importances.sum()
        if total > 0:
            importances /= total
        self._feature_importances = importances

    def _accumulate_importance(
        self, node: _TreeNode | None, importances: np.ndarray
    ) -> None:
        """递归累积特征重要性。"""
        if node is None or node.left is None or node.right is None:
            return
        # 加权信息增益
        n = node.n_samples
        n_left = node.left.n_samples
        n_right = node.right.n_samples
        gain = (
            n * node.impurity
            - n_left * node.left.impurity
            - n_right * node.right.impurity
        )
        importances[node.feature_idx] += gain
        self._accumulate_importance(node.left, importances)
        self._accumulate_importance(node.right, importances)

    def _get_state(self) -> dict:
        return {
            "model_type": "DecisionTree",
            "max_depth": self.max_depth,
            "min_samples_split": self.min_samples_split,
            "min_samples_leaf": self.min_samples_leaf,
            "classes": self._classes.tolist(),
            "tree": self._serialize_node(self._root),
            "feature_importances": (
                self._feature_importances.tolist()
                if self._feature_importances is not None
                else None
            ),
        }

    def _set_state(self, state: dict) -> None:
        self.max_depth = state["max_depth"]
        self.min_samples_split = state["min_samples_split"]
        self.min_samples_leaf = state["min_samples_leaf"]
        self._classes = np.array(state["classes"])
        self._n_classes = len(self._classes)
        self._root = self._deserialize_node(state["tree"])
        if state.get("feature_importances") is not None:
            self._feature_importances = np.array(state["feature_importances"])

    def _serialize_node(self, node: _TreeNode | None) -> dict | None:
        if node is None:
            return None
        return {
            "feature_idx": node.feature_idx,
            "threshold": node.threshold,
            "value": node.value.tolist() if node.value is not None else None,
            "n_samples": node.n_samples,
            "impurity": node.impurity,
            "left": self._serialize_node(node.left),
            "right": self._serialize_node(node.right),
        }

    def _deserialize_node(self, data: dict | None) -> _TreeNode | None:
        if data is None:
            return None
        node = _TreeNode()
        node.feature_idx = data["feature_idx"]
        node.threshold = data["threshold"]
        node.value = (
            np.array(data["value"]) if data["value"] is not None else None
        )
        node.n_samples = data["n_samples"]
        node.impurity = data["impurity"]
        node.left = self._deserialize_node(data.get("left"))
        node.right = self._deserialize_node(data.get("right"))
        return node


# ======================================================================
# 随机森林
# ======================================================================


class RandomForestModel(MLModelBase):
    """随机森林分类器（Bagging + 随机特征子集）。

    Parameters
    ----------
    n_estimators : int
        树的数量，默认 10。
    max_depth : int
        每棵树最大深度，默认 5。
    min_samples_split : int
        节点分裂最小样本数，默认 5。
    max_features_ratio : float
        每棵树使用的特征比例，默认 0.7 (sqrt策略的近似)。
    bootstrap_ratio : float
        每棵树采样比例，默认 1.0。
    random_state : int | None
        随机种子。
    """

    def __init__(
        self,
        n_estimators: int = 10,
        max_depth: int = 5,
        min_samples_split: int = 5,
        max_features_ratio: float = 0.7,
        bootstrap_ratio: float = 1.0,
        random_state: int | None = None,
        name: str = "",
    ) -> None:
        super().__init__(name=name)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features_ratio = max_features_ratio
        self.bootstrap_ratio = bootstrap_ratio
        self.random_state = random_state
        self._trees: list[DecisionTreeModel] = []
        self._rng = np.random.RandomState(random_state)

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """训练随机森林。"""
        n_samples, n_features = X.shape
        max_feat = max(1, int(n_features * self.max_features_ratio))
        bootstrap_size = max(1, int(n_samples * self.bootstrap_ratio))

        self._trees = []
        for i in range(self.n_estimators):
            # Bootstrap采样
            indices = self._rng.choice(n_samples, bootstrap_size, replace=True)
            X_boot = X[indices]
            y_boot = y[indices]

            tree = DecisionTreeModel(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                max_features=max_feat,
                random_state=self._rng.randint(0, 2**31),
            )
            tree.train(X_boot, y_boot)
            self._trees.append(tree)

        # 平均特征重要性
        importances = np.zeros(n_features)
        for tree in self._trees:
            imp = tree.feature_importance()
            if len(imp) == n_features:
                importances += imp
        importances /= len(self._trees)
        self._feature_importances = importances
        self._is_trained = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        """多数投票预测。"""
        proba = self.predict_proba(X)
        classes = self._trees[0]._classes
        return classes[np.argmax(proba, axis=1)]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """平均概率预测。"""
        if not self._trees:
            raise RuntimeError("模型尚未训练。")
        proba_sum = None
        for tree in self._trees:
            p = tree.predict_proba(X)
            if proba_sum is None:
                proba_sum = p.copy()
            else:
                proba_sum += p
        return proba_sum / len(self._trees)  # type: ignore[operator]

    def _get_state(self) -> dict:
        return {
            "model_type": "RandomForest",
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "trees": [t._get_state() for t in self._trees],
            "feature_importances": (
                self._feature_importances.tolist()
                if self._feature_importances is not None
                else None
            ),
        }

    def _set_state(self, state: dict) -> None:
        self.n_estimators = state["n_estimators"]
        self.max_depth = state["max_depth"]
        self._trees = []
        for tree_state in state["trees"]:
            tree = DecisionTreeModel()
            tree._set_state(tree_state)
            tree._is_trained = True
            self._trees.append(tree)
        if state.get("feature_importances") is not None:
            self._feature_importances = np.array(state["feature_importances"])


# ======================================================================
# 逻辑回归
# ======================================================================


class LinearModel(MLModelBase):
    """逻辑回归分类器 (梯度下降实现)。

    支持L2正则化和学习率衰减。对于多分类使用 one-vs-rest。

    Parameters
    ----------
    learning_rate : float
        初始学习率，默认 0.01。
    n_iterations : int
        最大迭代次数，默认 1000。
    regularization : float
        L2正则化系数，默认 0.01。
    tol : float
        收敛阈值，默认 1e-6。
    """

    def __init__(
        self,
        learning_rate: float = 0.01,
        n_iterations: int = 1000,
        regularization: float = 0.01,
        tol: float = 1e-6,
        name: str = "",
    ) -> None:
        super().__init__(name=name)
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.regularization = regularization
        self.tol = tol
        self._weights: np.ndarray | None = None
        self._bias: np.ndarray | None = None
        self._classes: np.ndarray = np.array([])
        self._n_classes: int = 0

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        """数值稳定的sigmoid函数。"""
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """训练逻辑回归。"""
        self._classes = np.unique(y)
        self._n_classes = len(self._classes)
        n_samples, n_features = X.shape

        if self._n_classes == 2:
            # 二分类
            y_binary = (y == self._classes[1]).astype(float)
            w, b = self._train_binary(X, y_binary)
            self._weights = w.reshape(1, -1)
            self._bias = np.array([b])
        else:
            # One-vs-Rest
            self._weights = np.zeros((self._n_classes, n_features))
            self._bias = np.zeros(self._n_classes)
            for k in range(self._n_classes):
                y_binary = (y == self._classes[k]).astype(float)
                w, b = self._train_binary(X, y_binary)
                self._weights[k] = w
                self._bias[k] = b

        # 特征重要性 = |权重|的平均
        self._feature_importances = np.mean(np.abs(self._weights), axis=0)
        total = self._feature_importances.sum()
        if total > 0:
            self._feature_importances /= total
        self._is_trained = True

    def _train_binary(
        self, X: np.ndarray, y: np.ndarray
    ) -> tuple[np.ndarray, float]:
        """训练单个二分类逻辑回归。"""
        n_samples, n_features = X.shape
        w = np.zeros(n_features)
        b = 0.0
        lr = self.learning_rate

        prev_loss = float("inf")
        for iteration in range(self.n_iterations):
            z = X @ w + b
            y_hat = self._sigmoid(z)

            # 梯度
            error = y_hat - y
            dw = (X.T @ error) / n_samples + self.regularization * w
            db = np.mean(error)

            w -= lr * dw
            b -= lr * db

            # 损失 (交叉熵 + L2)
            eps = 1e-12
            loss = -np.mean(
                y * np.log(y_hat + eps) + (1 - y) * np.log(1 - y_hat + eps)
            ) + 0.5 * self.regularization * np.sum(w ** 2)

            if abs(prev_loss - loss) < self.tol:
                break
            prev_loss = loss

        return w, b

    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测类别。"""
        proba = self.predict_proba(X)
        return self._classes[np.argmax(proba, axis=1)]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """预测概率。"""
        if self._weights is None:
            raise RuntimeError("模型尚未训练。")

        if self._n_classes == 2:
            z = X @ self._weights[0] + self._bias[0]
            p1 = self._sigmoid(z)
            return np.column_stack([1 - p1, p1])
        else:
            # One-vs-Rest: softmax归一化
            logits = X @ self._weights.T + self._bias
            exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
            return exp_logits / exp_logits.sum(axis=1, keepdims=True)

    def _get_state(self) -> dict:
        return {
            "model_type": "Linear",
            "learning_rate": self.learning_rate,
            "n_iterations": self.n_iterations,
            "regularization": self.regularization,
            "classes": self._classes.tolist(),
            "weights": self._weights.tolist() if self._weights is not None else None,
            "bias": self._bias.tolist() if self._bias is not None else None,
            "feature_importances": (
                self._feature_importances.tolist()
                if self._feature_importances is not None
                else None
            ),
        }

    def _set_state(self, state: dict) -> None:
        self.learning_rate = state["learning_rate"]
        self.n_iterations = state["n_iterations"]
        self.regularization = state["regularization"]
        self._classes = np.array(state["classes"])
        self._n_classes = len(self._classes)
        self._weights = (
            np.array(state["weights"]) if state["weights"] is not None else None
        )
        self._bias = (
            np.array(state["bias"]) if state["bias"] is not None else None
        )
        if state.get("feature_importances") is not None:
            self._feature_importances = np.array(state["feature_importances"])


# ======================================================================
# 梯度提升树
# ======================================================================


class GradientBoostingModel(MLModelBase):
    """梯度提升树分类器 (GBDT)。

    使用对数损失函数（log loss）进行梯度提升，每一轮拟合负梯度（伪残差）。
    对于二分类直接使用log loss，多分类使用one-vs-rest。

    Parameters
    ----------
    n_estimators : int
        提升轮数（树的数量），默认 50。
    max_depth : int
        每棵树最大深度，默认 3。
    learning_rate : float
        学习率（收缩因子），默认 0.1。
    min_samples_split : int
        节点分裂最小样本数，默认 5。
    subsample : float
        每轮的样本采样比例，默认 0.8。
    random_state : int | None
        随机种子。
    """

    def __init__(
        self,
        n_estimators: int = 50,
        max_depth: int = 3,
        learning_rate: float = 0.1,
        min_samples_split: int = 5,
        subsample: float = 0.8,
        random_state: int | None = None,
        name: str = "",
    ) -> None:
        super().__init__(name=name)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.min_samples_split = min_samples_split
        self.subsample = subsample
        self.random_state = random_state
        self._classes: np.ndarray = np.array([])
        self._n_classes: int = 0
        self._init_pred: float = 0.0
        self._stumps: list[list[_GBStump]] = []
        self._rng = np.random.RandomState(random_state)

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """训练梯度提升模型。"""
        self._classes = np.unique(y)
        self._n_classes = len(self._classes)
        n_samples, n_features = X.shape

        if self._n_classes == 2:
            # 二分类: 直接用log loss
            y_binary = (y == self._classes[1]).astype(float)
            self._init_pred = float(np.log(
                (y_binary.mean() + 1e-10) / (1 - y_binary.mean() + 1e-10)
            ))
            F = np.full(n_samples, self._init_pred)

            self._stumps = [[]]
            for m in range(self.n_estimators):
                p = self._sigmoid(F)
                residual = y_binary - p

                # 子采样
                if self.subsample < 1.0:
                    sub_size = max(1, int(n_samples * self.subsample))
                    indices = self._rng.choice(n_samples, sub_size, replace=False)
                else:
                    indices = np.arange(n_samples)

                stump = _GBStump(max_depth=self.max_depth, min_samples_split=self.min_samples_split)
                stump.fit(X[indices], residual[indices])
                pred = stump.predict(X)
                F += self.learning_rate * pred
                self._stumps[0].append(stump)
        else:
            # 多分类: one-vs-rest
            self._stumps = [[] for _ in range(self._n_classes)]
            self._init_preds: list[float] = []
            for k in range(self._n_classes):
                y_binary = (y == self._classes[k]).astype(float)
                init_p = float(np.log(
                    (y_binary.mean() + 1e-10) / (1 - y_binary.mean() + 1e-10)
                ))
                self._init_preds.append(init_p)

            F_all = np.array([
                np.full(n_samples, self._init_preds[k])
                for k in range(self._n_classes)
            ])  # shape: (n_classes, n_samples)

            for m in range(self.n_estimators):
                # 子采样
                if self.subsample < 1.0:
                    sub_size = max(1, int(n_samples * self.subsample))
                    indices = self._rng.choice(n_samples, sub_size, replace=False)
                else:
                    indices = np.arange(n_samples)

                for k in range(self._n_classes):
                    y_binary = (y == self._classes[k]).astype(float)
                    p = self._sigmoid(F_all[k])
                    residual = y_binary - p

                    stump = _GBStump(
                        max_depth=self.max_depth,
                        min_samples_split=self.min_samples_split,
                    )
                    stump.fit(X[indices], residual[indices])
                    pred = stump.predict(X)
                    F_all[k] += self.learning_rate * pred
                    self._stumps[k].append(stump)

        # 特征重要性
        importances = np.zeros(n_features)
        for class_stumps in self._stumps:
            for stump in class_stumps:
                importances += stump.feature_importances(n_features)
        total = importances.sum()
        if total > 0:
            importances /= total
        self._feature_importances = importances
        self._is_trained = True

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))

    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测类别。"""
        proba = self.predict_proba(X)
        return self._classes[np.argmax(proba, axis=1)]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """预测概率。"""
        if not self._stumps:
            raise RuntimeError("模型尚未训练。")

        n_samples = X.shape[0]

        if self._n_classes == 2:
            F = np.full(n_samples, self._init_pred)
            for stump in self._stumps[0]:
                F += self.learning_rate * stump.predict(X)
            p1 = self._sigmoid(F)
            return np.column_stack([1 - p1, p1])
        else:
            F_all = np.array([
                np.full(n_samples, self._init_preds[k])
                for k in range(self._n_classes)
            ])
            for k in range(self._n_classes):
                for stump in self._stumps[k]:
                    F_all[k] += self.learning_rate * stump.predict(X)

            # Softmax
            exp_F = np.exp(F_all - np.max(F_all, axis=0, keepdims=True))
            proba = exp_F / exp_F.sum(axis=0, keepdims=True)
            return proba.T  # shape: (n_samples, n_classes)

    def _get_state(self) -> dict:
        state: dict[str, Any] = {
            "model_type": "GradientBoosting",
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "classes": self._classes.tolist(),
            "n_classes": self._n_classes,
            "init_pred": self._init_pred,
            "stumps": [
                [s.to_dict() for s in class_stumps]
                for class_stumps in self._stumps
            ],
            "feature_importances": (
                self._feature_importances.tolist()
                if self._feature_importances is not None
                else None
            ),
        }
        if self._n_classes > 2:
            state["init_preds"] = self._init_preds
        return state

    def _set_state(self, state: dict) -> None:
        self.n_estimators = state["n_estimators"]
        self.max_depth = state["max_depth"]
        self.learning_rate = state["learning_rate"]
        self._classes = np.array(state["classes"])
        self._n_classes = state["n_classes"]
        self._init_pred = state["init_pred"]
        if "init_preds" in state:
            self._init_preds = state["init_preds"]
        self._stumps = [
            [_GBStump.from_dict(s) for s in class_stumps]
            for class_stumps in state["stumps"]
        ]
        if state.get("feature_importances") is not None:
            self._feature_importances = np.array(state["feature_importances"])


class _GBStump:
    """梯度提升中使用的回归树桩。

    与DecisionTreeModel不同，这里是回归树 (拟合连续残差)。
    """

    def __init__(
        self,
        max_depth: int = 3,
        min_samples_split: int = 5,
    ) -> None:
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self._root: _GBNode | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """拟合回归树。"""
        self._root = self._build(X, y, depth=0)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测连续值。"""
        if self._root is None:
            return np.zeros(X.shape[0])
        return np.array([self._predict_one(x, self._root) for x in X])

    def _predict_one(self, x: np.ndarray, node: _GBNode) -> float:
        if node.left is None and node.right is None:
            return node.value
        if x[node.feature_idx] <= node.threshold:
            return self._predict_one(x, node.left)  # type: ignore[arg-type]
        return self._predict_one(x, node.right)  # type: ignore[arg-type]

    def _build(
        self, X: np.ndarray, y: np.ndarray, depth: int
    ) -> _GBNode:
        node = _GBNode()
        node.value = float(np.mean(y)) if len(y) > 0 else 0.0
        node.n_samples = len(y)

        if (
            depth >= self.max_depth
            or len(y) < self.min_samples_split
            or np.std(y) < 1e-12
        ):
            return node

        best_reduction = 0.0
        best_feature = -1
        best_threshold = 0.0
        best_left_mask: np.ndarray | None = None
        total_var = np.var(y) * len(y)

        n_features = X.shape[1]
        for f_idx in range(n_features):
            thresholds = np.unique(X[:, f_idx])
            if len(thresholds) <= 1:
                continue
            if len(thresholds) > 20:
                thresholds = np.percentile(X[:, f_idx], np.linspace(5, 95, 20))

            for t in thresholds:
                left_mask = X[:, f_idx] <= t
                right_mask = ~left_mask
                n_l = left_mask.sum()
                n_r = right_mask.sum()
                if n_l < 2 or n_r < 2:
                    continue

                var_l = np.var(y[left_mask]) * n_l
                var_r = np.var(y[right_mask]) * n_r
                reduction = total_var - var_l - var_r

                if reduction > best_reduction:
                    best_reduction = reduction
                    best_feature = int(f_idx)
                    best_threshold = float(t)
                    best_left_mask = left_mask

        if best_left_mask is None or best_reduction <= 0:
            return node

        node.feature_idx = best_feature
        node.threshold = best_threshold
        node.left = self._build(X[best_left_mask], y[best_left_mask], depth + 1)
        node.right = self._build(X[~best_left_mask], y[~best_left_mask], depth + 1)
        return node

    def feature_importances(self, n_features: int) -> np.ndarray:
        """计算特征重要性。"""
        importances = np.zeros(n_features)
        self._acc_imp(self._root, importances)
        return importances

    def _acc_imp(self, node: _GBNode | None, imp: np.ndarray) -> None:
        if node is None or node.left is None or node.right is None:
            return
        imp[node.feature_idx] += node.n_samples
        self._acc_imp(node.left, imp)
        self._acc_imp(node.right, imp)

    def to_dict(self) -> dict:
        return {
            "max_depth": self.max_depth,
            "min_samples_split": self.min_samples_split,
            "root": self._serialize_node(self._root),
        }

    @classmethod
    def from_dict(cls, data: dict) -> _GBStump:
        stump = cls(
            max_depth=data["max_depth"],
            min_samples_split=data["min_samples_split"],
        )
        stump._root = cls._deserialize_node(data["root"])
        return stump

    def _serialize_node(self, node: _GBNode | None) -> dict | None:
        if node is None:
            return None
        return {
            "feature_idx": node.feature_idx,
            "threshold": node.threshold,
            "value": node.value,
            "n_samples": node.n_samples,
            "left": self._serialize_node(node.left),
            "right": self._serialize_node(node.right),
        }

    @classmethod
    def _deserialize_node(cls, data: dict | None) -> _GBNode | None:
        if data is None:
            return None
        node = _GBNode()
        node.feature_idx = data["feature_idx"]
        node.threshold = data["threshold"]
        node.value = data["value"]
        node.n_samples = data["n_samples"]
        node.left = cls._deserialize_node(data.get("left"))
        node.right = cls._deserialize_node(data.get("right"))
        return node


class _GBNode:
    """梯度提升回归树节点。"""

    __slots__ = ["feature_idx", "threshold", "left", "right", "value", "n_samples"]

    def __init__(self) -> None:
        self.feature_idx: int = -1
        self.threshold: float = 0.0
        self.left: _GBNode | None = None
        self.right: _GBNode | None = None
        self.value: float = 0.0
        self.n_samples: int = 0


# ======================================================================
# LSTM
# ======================================================================


class LSTMModel(MLModelBase):
    """简单LSTM分类器 (纯numpy实现)。

    将输入特征视为长度为 seq_len 的时间序列（通过滑动窗口切分），
    使用单层LSTM网络学习时序模式，最后一个时刻的隐状态接全连接层分类。

    Parameters
    ----------
    hidden_size : int
        LSTM隐藏单元数，默认 16。
    seq_len : int
        输入序列长度，默认 10。
    learning_rate : float
        学习率，默认 0.001。
    n_epochs : int
        训练轮数，默认 50。
    """

    def __init__(
        self,
        hidden_size: int = 16,
        seq_len: int = 10,
        learning_rate: float = 0.001,
        n_epochs: int = 50,
        name: str = "",
    ) -> None:
        super().__init__(name=name)
        self.hidden_size = hidden_size
        self.seq_len = seq_len
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        # LSTM参数 (将在train中初始化)
        self._Wf: np.ndarray | None = None
        self._Wi: np.ndarray | None = None
        self._Wc: np.ndarray | None = None
        self._Wo: np.ndarray | None = None
        self._bf: np.ndarray | None = None
        self._bi: np.ndarray | None = None
        self._bc: np.ndarray | None = None
        self._bo: np.ndarray | None = None
        # 输出层
        self._Wy: np.ndarray | None = None
        self._by: np.ndarray | None = None
        self._classes: np.ndarray = np.array([])
        self._n_classes: int = 0
        self._input_size: int = 0

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))

    @staticmethod
    def _tanh(z: np.ndarray) -> np.ndarray:
        return np.tanh(z)

    def _init_params(self, input_size: int, n_classes: int) -> None:
        """Xavier初始化LSTM参数。"""
        self._input_size = input_size
        h = self.hidden_size
        scale = np.sqrt(2.0 / (input_size + h))

        rng = np.random.RandomState(42)
        # 遗忘门
        self._Wf = rng.randn(h, input_size + h).astype(np.float64) * scale
        self._bf = np.zeros(h)
        # 输入门
        self._Wi = rng.randn(h, input_size + h).astype(np.float64) * scale
        self._bi = np.zeros(h)
        # 候选记忆
        self._Wc = rng.randn(h, input_size + h).astype(np.float64) * scale
        self._bc = np.zeros(h)
        # 输出门
        self._Wo = rng.randn(h, input_size + h).astype(np.float64) * scale
        self._bo = np.zeros(h)
        # 全连接输出
        scale_y = np.sqrt(2.0 / (h + n_classes))
        self._Wy = rng.randn(n_classes, h).astype(np.float64) * scale_y
        self._by = np.zeros(n_classes)

    def _lstm_forward(self, x_seq: np.ndarray) -> np.ndarray:
        """LSTM前向传播，返回最后时刻的隐状态。

        Parameters
        ----------
        x_seq : np.ndarray
            输入序列, shape (seq_len, input_size)。

        Returns
        -------
        np.ndarray
            最后时刻的隐状态, shape (hidden_size,)。
        """
        h = np.zeros(self.hidden_size)
        c = np.zeros(self.hidden_size)

        for t in range(x_seq.shape[0]):
            x_t = x_seq[t]
            concat = np.concatenate([h, x_t])

            f_t = self._sigmoid(self._Wf @ concat + self._bf)
            i_t = self._sigmoid(self._Wi @ concat + self._bi)
            c_hat = self._tanh(self._Wc @ concat + self._bc)
            c = f_t * c + i_t * c_hat
            o_t = self._sigmoid(self._Wo @ concat + self._bo)
            h = o_t * self._tanh(c)

        return h

    def _softmax(self, z: np.ndarray) -> np.ndarray:
        exp_z = np.exp(z - np.max(z))
        return exp_z / exp_z.sum()

    def _create_sequences(
        self, X: np.ndarray, y: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """将平面特征矩阵转为LSTM序列输入。

        Parameters
        ----------
        X : np.ndarray
            shape (n_samples, n_features)
        y : np.ndarray
            shape (n_samples,)

        Returns
        -------
        tuple
            (sequences, labels) where sequences shape is
            (n_sequences, seq_len, n_features)
        """
        n_samples = X.shape[0]
        if n_samples < self.seq_len:
            raise ValueError(
                f"样本数 ({n_samples}) 不足以构建长度为 {self.seq_len} 的序列。"
            )
        sequences = []
        labels = []
        for i in range(n_samples - self.seq_len + 1):
            sequences.append(X[i : i + self.seq_len])
            labels.append(y[i + self.seq_len - 1])
        return np.array(sequences), np.array(labels)

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """训练LSTM。使用简化的数值梯度+SGD。"""
        self._classes = np.unique(y)
        self._n_classes = len(self._classes)
        class_map = {c: i for i, c in enumerate(self._classes)}

        sequences, labels = self._create_sequences(X, y)
        y_mapped = np.array([class_map[v] for v in labels], dtype=int)
        n_seq = len(sequences)
        input_size = X.shape[1]

        self._init_params(input_size, self._n_classes)

        # 简化训练: 使用随机梯度下降 + 数值梯度
        rng = np.random.RandomState(42)
        params = self._get_flat_params()
        eps = 1e-4

        for epoch in range(self.n_epochs):
            perm = rng.permutation(n_seq)
            # 每个epoch只用一个mini-batch (简化)
            batch_size = min(32, n_seq)
            batch_idx = perm[:batch_size]

            # 数值梯度 (参数维度较大时只更新子集)
            grad = np.zeros_like(params)
            base_loss = self._compute_loss(sequences[batch_idx], y_mapped[batch_idx])

            # 随机选取一部分参数计算梯度 (减少计算量)
            n_params = len(params)
            param_indices = rng.choice(
                n_params, min(100, n_params), replace=False
            )

            for idx in param_indices:
                params[idx] += eps
                self._set_flat_params(params)
                loss_plus = self._compute_loss(
                    sequences[batch_idx], y_mapped[batch_idx]
                )
                params[idx] -= eps
                grad[idx] = (loss_plus - base_loss) / eps

            self._set_flat_params(params)
            params -= self.learning_rate * grad
            self._set_flat_params(params)

        self._feature_importances = np.zeros(input_size)
        self._is_trained = True

    def _compute_loss(
        self, sequences: np.ndarray, y: np.ndarray
    ) -> float:
        """计算交叉熵损失。"""
        total_loss = 0.0
        for i in range(len(sequences)):
            h = self._lstm_forward(sequences[i])
            logits = self._Wy @ h + self._by
            proba = self._softmax(logits)
            total_loss -= np.log(proba[y[i]] + 1e-12)
        return total_loss / len(sequences)

    def _get_flat_params(self) -> np.ndarray:
        """将所有参数展平为一维向量。"""
        parts = [
            self._Wf.ravel(), self._bf, self._Wi.ravel(), self._bi,
            self._Wc.ravel(), self._bc, self._Wo.ravel(), self._bo,
            self._Wy.ravel(), self._by,
        ]
        return np.concatenate(parts)

    def _set_flat_params(self, params: np.ndarray) -> None:
        """从一维向量恢复所有参数。"""
        h = self.hidden_size
        d = self._input_size
        sizes = [
            h * (d + h), h, h * (d + h), h,
            h * (d + h), h, h * (d + h), h,
            self._n_classes * h, self._n_classes,
        ]
        shapes = [
            (h, d + h), (h,), (h, d + h), (h,),
            (h, d + h), (h,), (h, d + h), (h,),
            (self._n_classes, h), (self._n_classes,),
        ]
        attrs = [
            "_Wf", "_bf", "_Wi", "_bi",
            "_Wc", "_bc", "_Wo", "_bo",
            "_Wy", "_by",
        ]
        offset = 0
        for size, shape, attr in zip(sizes, shapes, attrs):
            setattr(self, attr, params[offset : offset + size].reshape(shape))
            offset += size

    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测类别。"""
        proba = self.predict_proba(X)
        return self._classes[np.argmax(proba, axis=1)]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """预测概率。"""
        if self._Wy is None:
            raise RuntimeError("模型尚未训练。")

        # 构建序列
        n_samples = X.shape[0]
        if n_samples < self.seq_len:
            # 不足序列长度时pad
            pad = np.zeros((self.seq_len - n_samples, X.shape[1]))
            X_padded = np.vstack([pad, X])
            sequences = [X_padded]
        else:
            sequences = []
            for i in range(n_samples - self.seq_len + 1):
                sequences.append(X[i : i + self.seq_len])

        proba_list = []
        for seq in sequences:
            h = self._lstm_forward(seq)
            logits = self._Wy @ h + self._by
            proba_list.append(self._softmax(logits))

        return np.array(proba_list)

    def _get_state(self) -> dict:
        return {
            "model_type": "LSTM",
            "hidden_size": self.hidden_size,
            "seq_len": self.seq_len,
            "learning_rate": self.learning_rate,
            "n_epochs": self.n_epochs,
            "input_size": self._input_size,
            "classes": self._classes.tolist(),
            "params": self._get_flat_params().tolist() if self._Wf is not None else None,
        }

    def _set_state(self, state: dict) -> None:
        self.hidden_size = state["hidden_size"]
        self.seq_len = state["seq_len"]
        self.learning_rate = state["learning_rate"]
        self.n_epochs = state["n_epochs"]
        self._input_size = state["input_size"]
        self._classes = np.array(state["classes"])
        self._n_classes = len(self._classes)
        if state["params"] is not None:
            self._init_params(self._input_size, self._n_classes)
            self._set_flat_params(np.array(state["params"]))


# ======================================================================
# 模型集成
# ======================================================================


class EnsembleModel(MLModelBase):
    """模型集成（投票/加权平均）。

    将多个基模型的预测概率进行加权平均或投票。

    Parameters
    ----------
    models : list[MLModelBase]
        基模型列表。
    weights : list[float] | None
        各模型权重，None表示等权。
    method : str
        集成方法: 'average'(概率平均) 或 'vote'(多数投票)。
    """

    def __init__(
        self,
        models: list[MLModelBase] | None = None,
        weights: list[float] | None = None,
        method: str = "average",
        name: str = "",
    ) -> None:
        super().__init__(name=name)
        self.models: list[MLModelBase] = models or []
        self.weights = weights
        self.method = method
        self._classes: np.ndarray = np.array([])

    def add_model(self, model: MLModelBase, weight: float = 1.0) -> None:
        """添加一个基模型。

        Parameters
        ----------
        model : MLModelBase
            基模型实例。
        weight : float
            模型权重。
        """
        self.models.append(model)
        if self.weights is None:
            self.weights = [1.0] * len(self.models)
        else:
            self.weights.append(weight)

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """训练所有基模型。"""
        for model in self.models:
            model.train(X, y)
        self._classes = self.models[0]._classes if hasattr(self.models[0], "_classes") else np.unique(y)

        # 聚合特征重要性
        n_features = X.shape[1]
        importances = np.zeros(n_features)
        count = 0
        for model in self.models:
            try:
                imp = model.feature_importance()
                if len(imp) == n_features:
                    importances += imp
                    count += 1
            except RuntimeError:
                pass
        if count > 0:
            importances /= count
        self._feature_importances = importances
        self._is_trained = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测类别。"""
        proba = self.predict_proba(X)
        return self._classes[np.argmax(proba, axis=1)]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """预测概率。"""
        if not self.models:
            raise RuntimeError("集成模型中没有基模型。")

        weights = self.weights or [1.0] * len(self.models)
        weight_sum = sum(weights)

        if self.method == "average":
            proba_sum = None
            for model, w in zip(self.models, weights):
                p = model.predict_proba(X)
                if proba_sum is None:
                    proba_sum = p * w
                else:
                    # 确保维度匹配
                    min_len = min(proba_sum.shape[0], p.shape[0])
                    proba_sum = proba_sum[:min_len] + p[:min_len] * w
            return proba_sum / weight_sum  # type: ignore[operator]

        elif self.method == "vote":
            # 硬投票
            n_samples = X.shape[0]
            n_classes = len(self._classes)
            votes = np.zeros((n_samples, n_classes))
            for model, w in zip(self.models, weights):
                preds = model.predict(X)
                for i, pred in enumerate(preds):
                    class_idx = np.where(self._classes == pred)[0]
                    if len(class_idx) > 0:
                        votes[i, class_idx[0]] += w
            return votes / weight_sum
        else:
            raise ValueError(f"不支持的集成方法: {self.method!r}")

    def _get_state(self) -> dict:
        return {
            "model_type": "Ensemble",
            "method": self.method,
            "weights": self.weights,
            "classes": self._classes.tolist(),
            "models": [m._get_state() for m in self.models],
        }

    def _set_state(self, state: dict) -> None:
        self.method = state["method"]
        self.weights = state["weights"]
        self._classes = np.array(state["classes"])
        # 注意: 反序列化需要知道模型类型
        _model_registry: dict[str, type[MLModelBase]] = {
            "DecisionTree": DecisionTreeModel,
            "RandomForest": RandomForestModel,
            "Linear": LinearModel,
            "GradientBoosting": GradientBoostingModel,
            "LSTM": LSTMModel,
        }
        self.models = []
        for m_state in state["models"]:
            model_type = m_state.get("model_type", "DecisionTree")
            cls = _model_registry.get(model_type, DecisionTreeModel)
            model = cls()
            model._set_state(m_state)
            model._is_trained = True
            self.models.append(model)
