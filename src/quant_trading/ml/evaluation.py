"""模型评估模块

提供 ModelEvaluator 类，用于计算分类指标（准确率、精确率、召回率、F1）、
混淆矩阵、特征重要性可视化和过拟合检测。
"""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_trading.ml.models import MLModelBase


class ModelEvaluator:
    """ML模型评估器。

    提供一系列静态方法用于评估分类模型的性能。

    Example
    -------
    >>> evaluator = ModelEvaluator()
    >>> report = evaluator.full_report(y_true, y_pred)
    """

    @staticmethod
    def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """计算准确率。

        Parameters
        ----------
        y_true : np.ndarray
            真实标签。
        y_pred : np.ndarray
            预测标签。

        Returns
        -------
        float
            准确率 [0, 1]。
        """
        if len(y_true) == 0:
            return 0.0
        return float(np.mean(y_true == y_pred))

    @staticmethod
    def precision(
        y_true: np.ndarray, y_pred: np.ndarray, average: str = "macro"
    ) -> float:
        """计算精确率 (Precision)。

        Parameters
        ----------
        y_true : np.ndarray
            真实标签。
        y_pred : np.ndarray
            预测标签。
        average : str
            聚合方式: 'macro' (宏平均), 'micro' (微平均), 'binary' (二分类)。

        Returns
        -------
        float
            精确率。
        """
        classes = np.unique(np.concatenate([y_true, y_pred]))

        if average == "binary":
            pos_class = classes[-1] if len(classes) > 0 else 1
            tp = np.sum((y_pred == pos_class) & (y_true == pos_class))
            fp = np.sum((y_pred == pos_class) & (y_true != pos_class))
            return float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0

        elif average == "micro":
            tp_total = np.sum(y_true == y_pred)
            return float(tp_total / len(y_true)) if len(y_true) > 0 else 0.0

        else:  # macro
            precisions = []
            for c in classes:
                tp = np.sum((y_pred == c) & (y_true == c))
                fp = np.sum((y_pred == c) & (y_true != c))
                p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                precisions.append(p)
            return float(np.mean(precisions)) if precisions else 0.0

    @staticmethod
    def recall(
        y_true: np.ndarray, y_pred: np.ndarray, average: str = "macro"
    ) -> float:
        """计算召回率 (Recall)。

        Parameters
        ----------
        y_true : np.ndarray
            真实标签。
        y_pred : np.ndarray
            预测标签。
        average : str
            聚合方式: 'macro', 'micro', 'binary'。

        Returns
        -------
        float
            召回率。
        """
        classes = np.unique(np.concatenate([y_true, y_pred]))

        if average == "binary":
            pos_class = classes[-1] if len(classes) > 0 else 1
            tp = np.sum((y_pred == pos_class) & (y_true == pos_class))
            fn = np.sum((y_pred != pos_class) & (y_true == pos_class))
            return float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0

        elif average == "micro":
            return ModelEvaluator.accuracy(y_true, y_pred)

        else:  # macro
            recalls = []
            for c in classes:
                tp = np.sum((y_pred == c) & (y_true == c))
                fn = np.sum((y_pred != c) & (y_true == c))
                r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                recalls.append(r)
            return float(np.mean(recalls)) if recalls else 0.0

    @staticmethod
    def f1_score(
        y_true: np.ndarray, y_pred: np.ndarray, average: str = "macro"
    ) -> float:
        """计算F1分数。

        Parameters
        ----------
        y_true : np.ndarray
            真实标签。
        y_pred : np.ndarray
            预测标签。
        average : str
            聚合方式: 'macro', 'micro', 'binary'。

        Returns
        -------
        float
            F1分数。
        """
        p = ModelEvaluator.precision(y_true, y_pred, average=average)
        r = ModelEvaluator.recall(y_true, y_pred, average=average)
        if (p + r) == 0:
            return 0.0
        return float(2 * p * r / (p + r))

    @staticmethod
    def confusion_matrix(
        y_true: np.ndarray, y_pred: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """计算混淆矩阵。

        Parameters
        ----------
        y_true : np.ndarray
            真实标签。
        y_pred : np.ndarray
            预测标签。

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            (混淆矩阵, 类别标签)
            混淆矩阵 shape (n_classes, n_classes)，
            cm[i][j] = 真实为 classes[i] 但预测为 classes[j] 的数量。
        """
        classes = np.unique(np.concatenate([y_true, y_pred]))
        n_classes = len(classes)
        class_to_idx = {c: i for i, c in enumerate(classes)}

        cm = np.zeros((n_classes, n_classes), dtype=int)
        for true, pred in zip(y_true, y_pred):
            i = class_to_idx[true]
            j = class_to_idx[pred]
            cm[i, j] += 1

        return cm, classes

    @staticmethod
    def classification_report(
        y_true: np.ndarray, y_pred: np.ndarray
    ) -> dict[str, Any]:
        """生成完整分类报告。

        Parameters
        ----------
        y_true : np.ndarray
            真实标签。
        y_pred : np.ndarray
            预测标签。

        Returns
        -------
        dict
            包含每个类别和总体指标的字典。
        """
        classes = np.unique(np.concatenate([y_true, y_pred]))
        report: dict[str, Any] = {}

        for c in classes:
            tp = np.sum((y_pred == c) & (y_true == c))
            fp = np.sum((y_pred == c) & (y_true != c))
            fn = np.sum((y_pred != c) & (y_true == c))
            support = int(np.sum(y_true == c))

            p = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
            r = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
            f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0

            report[str(c)] = {
                "precision": p,
                "recall": r,
                "f1-score": f1,
                "support": support,
            }

        report["accuracy"] = ModelEvaluator.accuracy(y_true, y_pred)
        report["macro_avg"] = {
            "precision": ModelEvaluator.precision(y_true, y_pred, "macro"),
            "recall": ModelEvaluator.recall(y_true, y_pred, "macro"),
            "f1-score": ModelEvaluator.f1_score(y_true, y_pred, "macro"),
        }

        return report

    @staticmethod
    def plot_feature_importance(
        model: MLModelBase,
        feature_names: list[str] | None = None,
        top_k: int = 20,
    ) -> dict[str, float]:
        """获取特征重要性排名（以字典形式返回，便于绘图）。

        Parameters
        ----------
        model : MLModelBase
            已训练的模型。
        feature_names : list[str] | None
            特征名列表。
        top_k : int
            返回前k个最重要的特征。

        Returns
        -------
        dict[str, float]
            特征名 -> 重要性值 的有序字典。
        """
        importances = model.feature_importance()
        n = len(importances)

        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(n)]

        # 排序
        indices = np.argsort(importances)[::-1][:top_k]
        result = {}
        for idx in indices:
            if idx < len(feature_names):
                result[feature_names[idx]] = float(importances[idx])

        return result

    @staticmethod
    def overfitting_check(
        model: MLModelBase,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        threshold: float = 0.1,
    ) -> dict[str, Any]:
        """过拟合检测。

        比较训练集和测试集的表现差异，判断是否存在过拟合。

        Parameters
        ----------
        model : MLModelBase
            已训练的模型。
        X_train : np.ndarray
            训练集特征。
        y_train : np.ndarray
            训练集标签。
        X_test : np.ndarray
            测试集特征。
        y_test : np.ndarray
            测试集标签。
        threshold : float
            过拟合判定阈值：训练准确率 - 测试准确率 > threshold 则视为过拟合。

        Returns
        -------
        dict
            包含训练/测试指标和过拟合判定结果。
        """
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

        train_acc = ModelEvaluator.accuracy(y_train, y_pred_train)
        test_acc = ModelEvaluator.accuracy(y_test, y_pred_test)
        train_f1 = ModelEvaluator.f1_score(y_train, y_pred_train)
        test_f1 = ModelEvaluator.f1_score(y_test, y_pred_test)

        acc_gap = train_acc - test_acc
        f1_gap = train_f1 - test_f1

        is_overfitting = acc_gap > threshold

        return {
            "train_accuracy": train_acc,
            "test_accuracy": test_acc,
            "accuracy_gap": acc_gap,
            "train_f1": train_f1,
            "test_f1": test_f1,
            "f1_gap": f1_gap,
            "is_overfitting": is_overfitting,
            "threshold": threshold,
            "verdict": (
                "可能存在过拟合" if is_overfitting
                else "未检测到明显过拟合"
            ),
        }

    @staticmethod
    def full_report(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """生成完整评估报告。

        Parameters
        ----------
        y_true : np.ndarray
            真实标签。
        y_pred : np.ndarray
            预测标签。
        y_proba : np.ndarray | None
            预测概率（用于AUC等指标）。

        Returns
        -------
        dict
            完整评估报告。
        """
        cm, classes = ModelEvaluator.confusion_matrix(y_true, y_pred)
        report = ModelEvaluator.classification_report(y_true, y_pred)

        result: dict[str, Any] = {
            "classification_report": report,
            "confusion_matrix": cm.tolist(),
            "classes": classes.tolist(),
            "accuracy": ModelEvaluator.accuracy(y_true, y_pred),
            "precision_macro": ModelEvaluator.precision(y_true, y_pred, "macro"),
            "recall_macro": ModelEvaluator.recall(y_true, y_pred, "macro"),
            "f1_macro": ModelEvaluator.f1_score(y_true, y_pred, "macro"),
            "n_samples": len(y_true),
        }

        # 如果有概率，计算AUC (二分类)
        if y_proba is not None and len(classes) == 2:
            result["auc"] = ModelEvaluator._compute_auc(
                y_true, y_proba[:, 1], classes[-1]
            )

        return result

    @staticmethod
    def _compute_auc(
        y_true: np.ndarray, y_scores: np.ndarray, pos_class: Any
    ) -> float:
        """计算AUC (梯形法则)。

        Parameters
        ----------
        y_true : np.ndarray
            真实标签。
        y_scores : np.ndarray
            正类预测概率。
        pos_class : Any
            正类标签值。

        Returns
        -------
        float
            AUC值。
        """
        y_binary = (y_true == pos_class).astype(int)

        # 按分数降序排列
        sorted_indices = np.argsort(-y_scores)
        y_sorted = y_binary[sorted_indices]

        n_pos = y_binary.sum()
        n_neg = len(y_binary) - n_pos

        if n_pos == 0 or n_neg == 0:
            return 0.5

        # 梯形法则计算AUC
        tpr_prev = 0.0
        fpr_prev = 0.0
        auc = 0.0
        tp = 0
        fp = 0

        for i in range(len(y_sorted)):
            if y_sorted[i] == 1:
                tp += 1
            else:
                fp += 1
            tpr = tp / n_pos
            fpr = fp / n_neg
            auc += (fpr - fpr_prev) * (tpr + tpr_prev) / 2
            tpr_prev = tpr
            fpr_prev = fpr

        return float(auc)
