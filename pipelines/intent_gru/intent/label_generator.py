"""
==============================================================================
意图识别 — 机动类别标签生成
==============================================================================
本数据集 (UMAV/AGZ OnboardGPS.csv) 未提供显式的机动类别标注。
本模块基于运动学启发式规则 (alt_rate + heading_rate) 自动生成 4 类伪标签，
用于后续 SVM 分类器的监督训练。

四种基础机动类别:
    0 - 平飞   (Level flight):  |alt_rate| 小 且 |heading_rate| 小
    1 - 转弯   (Turn):          |alt_rate| 小 且 |heading_rate| 大
    2 - 爬升   (Climb):         alt_rate 大 (正向高度变化)
    3 - 俯冲   (Dive/Descent):  alt_rate 小 (负向高度变化)

阈值默认使用数据自适应的分位数 (而非硬编码常量)，避免因数据量纲差异
导致某一类别完全无样本。
==============================================================================
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


# 4 类基础机动的标签映射 (同时暴露为 zh / en 两套名称)
MANEUVER_LABEL_MAP = {
    0: ("平飞", "level"),
    1: ("转弯", "turn"),
    2: ("爬升", "climb"),
    3: ("俯冲", "dive"),
}

# 单纯的类别数值数组，供下游代码引用
MANEUVER_CLASSES = np.array([0, 1, 2, 3], dtype=np.int64)

# ---------------------------------------------------------------------
# 标签生成特征 (label-source features)
#
# 下列特征被用于"启发式生成伪标签"。是否将它们排除出 RF / SVM 候选池
# 由调用方决定 —— 通过 `fit_random_forest(exclude_names=...)` 与
# `select_features_by_cumulative_importance(leak_feature_names=...)` 的可选
# 参数控制:
#
#   - 排除 (intent_gru/prepare_intent.py): 防止数据泄漏压垮 RF 重要性分布,
#     RF 才能挑出真正有判别力的"非泄漏"特征。
#   - 不排除 (attention_bigru/prepare_data.py 默认): 让 SVM 输入与标签定义
#     保持一致, 接受 SVM 准确率被显著拉高的事实, 用于受控对比。
#
# 名称保留为 LABEL_LEAK_FEATURES 仅出于兼容历史代码; 在不排除场景下其
# 语义等价于 "label-source features"。
# ---------------------------------------------------------------------
LABEL_LEAK_FEATURES = ("alt_rate", "heading_rate")


def _quantile_threshold(
    values: np.ndarray,
    upper_quantile: float = 0.75,
) -> float:
    """
    基于绝对值的分位数计算阈值。
    例如 upper_quantile=0.75 → 返回 |values| 的 75% 分位数。
    """
    abs_vals = np.abs(values)
    return float(np.quantile(abs_vals, upper_quantile))


def generate_maneuver_labels(
    features: np.ndarray,
    feature_names: list,
    alt_rate_threshold: Optional[float] = None,
    heading_rate_threshold: Optional[float] = None,
    turn_quantile: float = 0.75,
    climb_quantile: float = 0.70,
    verbose: bool = True,
) -> Tuple[np.ndarray, dict]:
    """
    基于派生的 alt_rate 与 heading_rate，生成 4 分类机动伪标签。

    规则 (按优先级自上而下判断):
        - 若 alt_rate >=  +alt_th         → 2 (爬升)
        - 若 alt_rate <=  -alt_th         → 3 (俯冲)
        - 若 |heading_rate| >=  head_th   → 1 (转弯)
        - 否则                            → 0 (平飞)

    参数:
        features:                 shape (N, n_features), 必须包含
                                  alt_rate / heading_rate 两列 (由 feature_extractor 产生)
        feature_names:            与 features 列对应的列名列表
        alt_rate_threshold:       若为 None，则自适应取 |alt_rate| 的 `climb_quantile` 分位数
        heading_rate_threshold:   若为 None，则自适应取 |heading_rate| 的 `turn_quantile` 分位数
        turn_quantile:            转弯阈值的自适应分位数
        climb_quantile:           爬升/俯冲阈值的自适应分位数
        verbose:                  是否打印日志

    返回:
        labels:  shape (N,), dtype=int64，取值在 {0, 1, 2, 3}
        info:    dict 记录阈值与各类别样本计数，便于外部报告
    """
    if verbose:
        print("=" * 60)
        print("  [intent] 机动类别伪标签生成")
        print("=" * 60)

    # --- 定位 alt_rate / heading_rate 列 --------------------------------
    if "alt_rate" not in feature_names:
        raise ValueError("特征矩阵中未找到 'alt_rate' 列，无法生成机动标签。")
    if "heading_rate" not in feature_names:
        raise ValueError("特征矩阵中未找到 'heading_rate' 列，无法生成机动标签。")

    alt_idx = feature_names.index("alt_rate")
    head_idx = feature_names.index("heading_rate")

    alt_rate = features[:, alt_idx]
    heading_rate = features[:, head_idx]

    # --- 自适应阈值 -----------------------------------------------------
    if alt_rate_threshold is None:
        alt_rate_threshold = _quantile_threshold(alt_rate, upper_quantile=climb_quantile)
    if heading_rate_threshold is None:
        heading_rate_threshold = _quantile_threshold(heading_rate, upper_quantile=turn_quantile)

    if verbose:
        print(f"  alt_rate 分布:      min={alt_rate.min():.4f}, max={alt_rate.max():.4f}, "
              f"std={alt_rate.std():.4f}")
        print(f"  heading_rate 分布:  min={heading_rate.min():.4f}, max={heading_rate.max():.4f}, "
              f"std={heading_rate.std():.4f}")
        print(f"  使用阈值 (可能为自适应):")
        print(f"    alt_rate  阈值 = ±{alt_rate_threshold:.6f}")
        print(f"    heading_rate 阈值 = ±{heading_rate_threshold:.6f}")

    # --- 按优先级规则赋标签 ---------------------------------------------
    labels = np.zeros(len(features), dtype=np.int64)   # 默认 0 (平飞)

    # 转弯 — 水平面明显转向 (高度变化不大)
    is_turn = (np.abs(heading_rate) >= heading_rate_threshold) & \
              (np.abs(alt_rate) < alt_rate_threshold)
    labels[is_turn] = 1

    # 爬升 — 正向高度变化明显
    is_climb = alt_rate >= alt_rate_threshold
    labels[is_climb] = 2

    # 俯冲 — 负向高度变化明显 (放在爬升后判断，互斥)
    is_dive = alt_rate <= -alt_rate_threshold
    labels[is_dive] = 3

    # --- 统计 / 报告 ----------------------------------------------------
    class_counts = {int(c): int(np.sum(labels == c)) for c in MANEUVER_CLASSES}
    total = len(labels)

    if verbose:
        print(f"\n  ✅ 标签生成完成 (总样本 {total}):")
        for c, (zh, en) in MANEUVER_LABEL_MAP.items():
            n = class_counts[c]
            print(f"    [{c}] {zh:>4s} ({en:>5s}): {n:>6d}  ({n/total*100:.2f}%)")
        print()

    info = {
        "alt_rate_threshold": alt_rate_threshold,
        "heading_rate_threshold": heading_rate_threshold,
        "class_counts": class_counts,
        "total_samples": total,
        # 显式暴露"标签生成所使用"的特征名称，供下游 (prepare_intent.py)
        # 从 RF 候选池中剔除以防止数据泄漏
        "leak_features": list(LABEL_LEAK_FEATURES),
    }

    return labels, info
