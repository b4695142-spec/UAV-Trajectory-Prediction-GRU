"""
==============================================================================
意图识别 — 随机森林特征筛选
==============================================================================
基于 sklearn.ensemble.RandomForestClassifier 对意图分类任务中的特征进行
重要性评估 (基于基尼不纯度 / 信息增益的默认 Gini impurity)，保留累计
贡献率达到指定阈值 (默认 80%) 的核心特征。

返回的特征索引可直接作用于原始特征矩阵，以实现降维。
==============================================================================
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np
from sklearn.ensemble import RandomForestClassifier


def _resolve_exclude_indices(
    feature_names: Optional[Sequence[str]],
    exclude_names: Optional[Sequence[str]],
    exclude_indices: Optional[Sequence[int]],
    n_features: int,
) -> np.ndarray:
    """
    将 `exclude_names` / `exclude_indices` 规范化为一维整数索引数组。
    """
    idx_set: set = set()
    if exclude_indices:
        for idx in exclude_indices:
            idx = int(idx)
            if 0 <= idx < n_features:
                idx_set.add(idx)
    if exclude_names and feature_names is not None:
        name2idx = {name: i for i, name in enumerate(feature_names)}
        for name in exclude_names:
            if name in name2idx:
                idx_set.add(name2idx[name])
    return np.array(sorted(idx_set), dtype=np.int64)


def fit_random_forest(
    X: np.ndarray,
    y: np.ndarray,
    n_estimators: int = 200,
    max_depth: int | None = None,
    random_state: int = 42,
    n_jobs: int = -1,
    criterion: str = "gini",
    verbose: bool = True,
    feature_names: Optional[Sequence[str]] = None,
    exclude_names: Optional[Sequence[str]] = None,
    exclude_indices: Optional[Sequence[int]] = None,
) -> Tuple[RandomForestClassifier, np.ndarray]:
    """
    训练随机森林分类器，用于提取特征重要性。

    默认使用 Gini 不纯度作为分裂依据；如需信息增益可传 criterion="entropy"。

    参数:
        X:               shape (N, n_features)，标准化后的特征矩阵
        y:               shape (N,)，整型标签
        n_estimators:    决策树数量
        max_depth:       单棵树最大深度 (None 表示不限制)
        random_state:    随机种子
        n_jobs:          并行线程数 (-1 表示使用所有核)
        criterion:       分裂判据 ("gini" / "entropy"，后者对应信息增益)
        verbose:         是否打印进度
        feature_names:   原始特征名称 (用于解析 exclude_names / 打印日志)
        exclude_names:   需要从 RF 候选池中剔除的列名列表 (例如标签生成特征，
                         防止数据泄漏)
        exclude_indices: 需要从 RF 候选池中剔除的列索引列表

    返回:
        rf:             已 fit 的 RandomForestClassifier 实例
                        注意: rf 只在 "保留列" 上训练，因此其 feature_importances_
                        的长度可能小于原始 X 的列数。
        kept_indices:   shape (k,)，保留下来的列在原始 X 中的索引 (升序)。
                        若调用方未指定 exclude_*，则等价于 np.arange(n_features)。
    """
    n_features_orig = X.shape[1]
    excl = _resolve_exclude_indices(
        feature_names, exclude_names, exclude_indices, n_features_orig,
    )
    mask = np.ones(n_features_orig, dtype=bool)
    if len(excl) > 0:
        mask[excl] = False
    kept_indices = np.arange(n_features_orig)[mask]

    if verbose:
        print("=" * 60)
        print("  [intent] Random Forest 特征重要性拟合")
        print("=" * 60)
        print(f"  n_estimators = {n_estimators}, criterion = {criterion}")
        if len(excl) > 0:
            excl_names = (
                [feature_names[i] for i in excl] if feature_names is not None
                else [str(i) for i in excl]
            )
            print(f"  ⚠️  已从 RF 候选池中剔除 {len(excl)} 个\"标签生成/泄漏\" 特征:")
            print(f"     {excl_names}")
        print(f"  参与 RF 拟合的特征数: {len(kept_indices)} / {n_features_orig}")

    X_kept = X[:, kept_indices]

    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        criterion=criterion,
        random_state=random_state,
        n_jobs=n_jobs,
        class_weight="balanced",   # 缓解伪标签可能存在的类别不平衡
    )
    rf.fit(X_kept, y)

    if verbose:
        oob_score = getattr(rf, "oob_score_", None)
        print(f"  ✅ RF 训练完成 (训练集准确率 = {rf.score(X_kept, y):.4f}"
              + (f", OOB score = {oob_score:.4f}" if oob_score is not None else "")
              + ")")
        print()

    return rf, kept_indices


def select_features_by_cumulative_importance(
    rf: RandomForestClassifier,
    feature_names: List[str],
    cumulative_threshold: float = 0.80,
    kept_indices: Optional[np.ndarray] = None,
    verbose: bool = True,
) -> Tuple[np.ndarray, List[str], np.ndarray]:
    """
    基于 RF.feature_importances_，按累计贡献率选择核心特征。

    支持"RF 只在子集列上训练"的场景 (见 fit_random_forest 的 exclude_* 参数)。
    此时 `rf.feature_importances_` 的长度等于 `kept_indices` 的长度，
    本函数会自动把相对索引映射回原始特征索引。

    参数:
        rf:                    已拟合的 RandomForestClassifier
        feature_names:         所有原始特征名称列表 (与原始 X 的列顺序一致)
        cumulative_threshold:  累计贡献率阈值 (0-1)，默认 0.8 即 80%
        kept_indices:          RF 训练时保留的原始列索引。若为 None，则视为
                               RF 在全部特征上训练 (旧行为)
        verbose:               是否打印日志

    返回:
        selected_indices:   shape (k,)，保留特征在 *原始* 特征矩阵中的列索引
                            (按原始列顺序升序)
        selected_names:     长度 k 的列表，与 selected_indices 对应的列名
        importances_full:   shape (n_features,) 的原始特征重要性数组；对于
                            被 RF 剔除的列，其值为 0。
    """
    if verbose:
        print("=" * 60)
        print(f"  [intent] 累计 {cumulative_threshold * 100:.0f}% 重要性特征筛选")
        print("=" * 60)

    n_features_orig = len(feature_names)
    raw_imp = rf.feature_importances_.copy()

    if kept_indices is None:
        kept_indices = np.arange(n_features_orig)

    if len(raw_imp) != len(kept_indices):
        raise ValueError(
            f"rf.feature_importances_ ({len(raw_imp)}) 与 kept_indices "
            f"({len(kept_indices)}) 长度不一致"
        )

    # 把 RF 的相对重要性映射回完整特征空间 (剔除列重要性=0)
    importances_full = np.zeros(n_features_orig, dtype=raw_imp.dtype)
    importances_full[kept_indices] = raw_imp

    # 按 RF 原始列 (即 kept 列) 重要性降序排列
    order_kept = np.argsort(raw_imp)[::-1]          # 相对 kept 的索引
    cum_kept = np.cumsum(raw_imp[order_kept])

    k = int(np.searchsorted(cum_kept, cumulative_threshold) + 1)
    k = max(1, min(k, len(raw_imp)))

    # 相对 kept 的 top-k 索引 → 映射到原始列索引
    selected_relative = order_kept[:k]
    selected_absolute = np.sort(kept_indices[selected_relative])
    selected_names = [feature_names[i] for i in selected_absolute]

    if verbose:
        print(f"  RF 参与列数 = {len(kept_indices)}, "
              f"原始特征列数 = {n_features_orig}")
        print(f"  特征重要性排序 (降序, 仅展示 RF 参与列):")
        selected_set = set(selected_absolute.tolist())
        for rank, rel_idx in enumerate(order_kept):
            abs_idx = int(kept_indices[rel_idx])
            marker = "★" if abs_idx in selected_set else " "
            print(f"    {marker} [{rank + 1:>2d}] {feature_names[abs_idx]:<18s} "
                  f"importance = {raw_imp[rel_idx]:.6f}  "
                  f"cum = {cum_kept[rank]:.4f}")
        print(f"\n  ✅ 保留前 {k} 个特征，累计贡献率 = {cum_kept[k - 1]:.4f} "
              f"(阈值 {cumulative_threshold:.2f})")
        print(f"  保留的列索引 (按原顺序): {selected_absolute.tolist()}")
        print(f"  保留的列名称:             {selected_names}")
        print()

    return selected_absolute, selected_names, importances_full
