"""
==============================================================================
意图识别 — 带 Platt Scaling 的 One-vs-All SVM 分类器
==============================================================================
采用 sklearn.svm.SVC + OneVsRestClassifier 实现 4 类机动意图分类。

关键要求 (对齐理论参考):
    - 核函数:      RBF (高斯核)
    - C:           5
    - gamma:       0.01
    - probability: True  (启用 Platt Scaling，输出连续概率分布)
    - 策略:        One-vs-All (OvA) — 使用 OneVsRestClassifier 封装

输出: 长度为 4 的概率分布向量，可直接拼接到 GRU 输入特征后方。

【公平性修正】
    新增 fit_predict_proba_oof() 方法，使用 K-Fold 交叉验证生成
    训练集的 out-of-fold (OOF) 概率，消除训练/测试 SVM 概率
    分布偏移 (covariate shift)，确保 GRU 训练时看到的意图概率
    与推理时看到的分布一致。
==============================================================================
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC


class IntentSVMClassifier:
    """
    OvA + Platt Scaling 的 SVM 意图分类器封装。

    用法:
        clf = IntentSVMClassifier(n_classes=4, C=5.0, gamma=0.01)
        clf.fit(X_train, y_train)
        proba = clf.predict_proba(X_test)   # shape (N, 4)
    """

    def __init__(
        self,
        n_classes: int = 4,
        C: float = 5.0,
        gamma: float = 0.01,
        kernel: str = "rbf",
        random_state: int = 42,
        n_jobs: int = -1,
        class_weight: Optional[str] = "balanced",
    ):
        self.n_classes = n_classes
        self.C = C
        self.gamma = gamma
        self.kernel = kernel
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.class_weight = class_weight

        # 基础 SVC + Platt Scaling (probability=True)
        base_svc = SVC(
            C=self.C,
            kernel=self.kernel,
            gamma=self.gamma,
            probability=True,           # 启用 Platt Scaling
            class_weight=self.class_weight,
            random_state=self.random_state,
        )

        # OvA 包装: 对每个类别训练一个 "vs 其余" 的二分类器
        self.clf = OneVsRestClassifier(base_svc, n_jobs=self.n_jobs)

        self._is_fitted = False
        self._classes_seen: Optional[np.ndarray] = None

    # ---------------------------------------------------------------- #
    def fit(self, X: np.ndarray, y: np.ndarray) -> "IntentSVMClassifier":
        """在 (X, y) 上训练 OvA SVM，每个子分类器均启用 Platt Scaling。"""
        self.clf.fit(X, y)
        self._is_fitted = True
        self._classes_seen = np.asarray(self.clf.classes_)
        return self

    # ---------------------------------------------------------------- #
    def fit_predict_proba_oof(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_splits: int = 5,
        verbose: bool = True,
    ) -> Tuple[np.ndarray, "IntentSVMClassifier"]:
        """
        使用 K-Fold 交叉验证生成训练集的 out-of-fold (OOF) 概率，
        然后在全量训练集上重新 fit 最终模型。

        【公平性修正】核心逻辑:
            - 训练集的 SVM 概率通过 OOF 方式生成 (每个样本仅由
              未见过它的子模型预测)，与测试集的 out-of-sample
              预测分布一致，消除 covariate shift。
            - 最终模型仍在全量训练集上 fit，用于测试集推理。

        参数:
            X:         shape (N, n_features), 训练集特征
            y:         shape (N,), 训练集标签
            n_splits:  折数 (默认 5)
            verbose:   是否打印进度

        返回:
            oof_proba: shape (N, n_classes), OOF 概率矩阵
            self:      已在全量训练集上 fit 的分类器实例
        """
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
        oof_proba = np.zeros((len(X), self.n_classes), dtype=np.float64)

        if verbose:
            print(f"  [SVM OOF] {n_splits}-Fold 交叉验证生成训练集概率...")

        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr = y[train_idx]

            fold_clf = IntentSVMClassifier(
                n_classes=self.n_classes,
                C=self.C,
                gamma=self.gamma,
                kernel=self.kernel,
                random_state=self.random_state,
                n_jobs=self.n_jobs,
                class_weight=self.class_weight,
            )
            fold_clf.fit(X_tr, y_tr)
            oof_proba[val_idx] = fold_clf.predict_proba(X_val)

            if verbose:
                fold_acc = fold_clf.score(X_val, y[val_idx])
                print(f"    Fold {fold_idx + 1}/{n_splits}: "
                      f"val_acc = {fold_acc:.4f}, "
                      f"val_size = {len(val_idx)}")

        if verbose:
            oof_preds = np.argmax(oof_proba, axis=1)
            oof_acc = np.mean(oof_preds == y)
            print(f"  [SVM OOF] 整体 OOF 准确率 = {oof_acc:.4f}")

        self.fit(X, y)

        return oof_proba, self

    # ---------------------------------------------------------------- #
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        返回 shape (N, n_classes) 的概率矩阵。

        sklearn 的 OneVsRestClassifier.predict_proba() 返回的是对各 OvA
        子分类器概率做归一化之后的结果。当训练集中某些类别从未出现时，
        我们会在输出矩阵中自动补上全零列以保证输出维度恒为 n_classes。
        """
        self._assert_fitted()
        raw_proba = self.clf.predict_proba(X)   # shape (N, len(classes_seen))

        if raw_proba.shape[1] == self.n_classes:
            return raw_proba

        # 否则补齐缺失类别列
        full = np.zeros((raw_proba.shape[0], self.n_classes), dtype=raw_proba.dtype)
        for j, cls in enumerate(self._classes_seen):
            if 0 <= int(cls) < self.n_classes:
                full[:, int(cls)] = raw_proba[:, j]
        return full

    # ---------------------------------------------------------------- #
    def predict(self, X: np.ndarray) -> np.ndarray:
        """硬分类 (返回最可能的类别编号)。"""
        self._assert_fitted()
        return self.clf.predict(X).astype(np.int64)

    # ---------------------------------------------------------------- #
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """在 (X, y) 上计算准确率。"""
        self._assert_fitted()
        return float(self.clf.score(X, y))

    # ---------------------------------------------------------------- #
    def _assert_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError("IntentSVMClassifier 尚未调用 fit()，无法推理。")
