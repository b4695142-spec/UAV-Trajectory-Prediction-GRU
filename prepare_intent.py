"""
==============================================================================
UAV 轨迹预测 — 机动意图识别数据准备 (级联架构, 步骤 1.5)
==============================================================================
本脚本是"意图识别增强版" GRU 流水线的入口，负责：

    1. 动态提取 OnboardGPS.csv 中的数值特征 + 派生运动学特征
    2. 分组归一化 (位置列 MinMax / 其他列 StandardScaler)
    3. 自动生成 4 类机动伪标签 (平飞 / 转弯 / 爬升 / 俯冲)
    4. Random Forest 特征重要性评估  → 保留累计贡献率 80% 的核心特征
       ★ 自动剔除"标签生成特征" (alt_rate / heading_rate) 防止数据泄漏
    5. OvA SVM + Platt Scaling (C=5, gamma=0.01, probability=True) 训练
       ★ 使用 5-Fold OOF 生成训练集概率，消除训练/测试分布偏移
    6. 对全部时间步计算 4 维意图概率向量
    7. 构建增广滑动窗口序列:
            GRU 输入 = [MinMax(lat,lon,alt)  ⊕  StdScale(RF 筛选的非位置特征)  ⊕  SVM 4 维概率]
       Y 仍为 MinMax 归一化后的 (lat, lon, alt)，以便与纯 GRU 直接对齐比较

所有产物保存至 `processed_data/intent/`，不会影响任何纯 GRU 文件。

【公平性修正 — 综合落地】
    A. 始终保留 lat/lon/alt 到 GRU 输入 (不受 RF 取舍影响)，避免位置信息丢失
    B. RF 阈值保持 0.80 (符合论文), 但语义更健康 (作用于非泄漏列)
    C. 从 RF 候选池中剔除 alt_rate / heading_rate 这两个"标签生成"特征，
       防止数据泄漏压垮 RF 的特征重要性分布
    D. 输入归一化与目标对齐: 位置列走 MinMax, 其他列走 StandardScaler
    E. MinMaxScaler 仅在训练集上 fit (与修正后的纯 GRU 管线一致)
    F. SVM 训练集概率使用 5-Fold OOF 生成，消除训练/测试分布偏移
==============================================================================
"""

from __future__ import annotations

import json
import os
import pickle
from typing import List, Tuple

import numpy as np
from sklearn.metrics import classification_report
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from intent import (
    IntentSVMClassifier,
    LABEL_LEAK_FEATURES,
    MANEUVER_LABEL_MAP,
    build_augmented_sequences,
    extract_intent_features,
    fit_random_forest,
    generate_maneuver_labels,
    select_features_by_cumulative_importance,
)


# ============================================================================
# 配置参数
# ============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 原始数据
RAW_CSV_PATH = os.path.join(BASE_DIR, "Log Files", "OnboardGPS.csv")

# 时间 / 下采样参数 (与 preprocess_uav.py 完全一致，保证时间轴对齐)
ORIGINAL_INTERVAL_S = 0.033333
TARGET_INTERVAL_S = 0.1
DOWNSAMPLE_FACTOR = round(TARGET_INTERVAL_S / ORIGINAL_INTERVAL_S)   # = 3
TRAIN_RATIO = 0.8

# 滑动窗口参数 (与 build_sequences.py 保持一致)
LOOK_BACK = 50
FORWARD_LENGTH = 0

# RF 特征筛选
RF_N_ESTIMATORS = 200
RF_CRITERION = "gini"                   # 可改为 "entropy" (信息增益)
RF_CUMULATIVE_THRESHOLD = 0.80          # 累计贡献率 80% (严格按论文)
RF_RANDOM_STATE = 42

# SVM 参数 (严格按照理论参考)
SVM_C = 5.0
SVM_GAMMA = 0.01
SVM_KERNEL = "rbf"
SVM_N_CLASSES = 4

# 必须保留到 GRU 输入中的"核心位置列" (方案 A) — 不受 RF 筛选影响
POSITION_COLS = ["lat", "lon", "alt"]

# 目录
OUTPUT_DIR = os.path.join(BASE_DIR, "processed_data", "intent")


# ============================================================================
# 工具函数
# ============================================================================
def _split_by_time(arr: np.ndarray, train_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    """按时间顺序严格切分 (不打乱)。"""
    n_total = len(arr)
    n_train = int(n_total * train_ratio)
    return arr[:n_train], arr[n_train:]


def _union_preserve_order(
    always_keep: List[int],
    rf_selected: np.ndarray,
) -> np.ndarray:
    """
    合并"始终保留列"与"RF 筛选列"为一个升序、去重的索引数组。

    合并语义:
        最终特征列 = 位置列 ∪ RF 筛选列
    """
    all_idx = sorted(set(always_keep) | set(map(int, rf_selected.tolist())))
    return np.array(all_idx, dtype=np.int64)


def _format_selected_layout(
    feature_names: List[str],
    final_indices: np.ndarray,
    position_indices: List[int],
    rf_selected: np.ndarray,
) -> List[str]:
    """构造一个人类友好的"保留列来源"描述列表。"""
    pos_set = set(position_indices)
    rf_set = set(int(i) for i in rf_selected)
    descs = []
    for i in final_indices:
        i = int(i)
        tags = []
        if i in pos_set:
            tags.append("POS")
        if i in rf_set:
            tags.append("RF")
        descs.append(f"{feature_names[i]}[{'+'.join(tags)}]")
    return descs


# ============================================================================
# 主流程
# ============================================================================
def main():
    print("\n" + "▓" * 60)
    print("  UAV 意图识别数据准备 (级联架构 — 步骤 1.5) — 方案 A/B/C/D 综合版")
    print("▓" * 60 + "\n")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ------------------------------------------------------------------
    # 1) 动态特征提取 + 降采样
    # ------------------------------------------------------------------
    features, feature_names, df_down = extract_intent_features(
        csv_path=RAW_CSV_PATH,
        downsample_factor=DOWNSAMPLE_FACTOR,
        target_interval_s=TARGET_INTERVAL_S,
        verbose=True,
    )
    # features: (N, n_features_raw)

    # ------------------------------------------------------------------
    # 2) 构造回归目标 (lat, lon, alt) 并做 MinMax 归一化
    # ------------------------------------------------------------------
    print("=" * 60)
    print("  [intent] 构造回归目标 — MinMax 归一化 (lat, lon, alt)")
    print("=" * 60)

    missing = [c for c in POSITION_COLS if c not in df_down.columns]
    if missing:
        raise ValueError(f"目标位置列缺失: {missing}")

    position = df_down[POSITION_COLS].to_numpy(dtype=np.float64)   # (N, 3)

    position_train, position_test = _split_by_time(position, TRAIN_RATIO)
    mm_scaler = MinMaxScaler(feature_range=(0, 1))
    mm_scaler.fit(position_train)
    target_all = mm_scaler.transform(position)                     # (N, 3)

    print(f"  position 原始 shape: {position.shape}")
    print(f"  train/test 切分:     {len(position_train)} / {len(position_test)}")
    print(f"  data_min: {mm_scaler.data_min_}")
    print(f"  data_max: {mm_scaler.data_max_}")
    print()

    # ------------------------------------------------------------------
    # 3) 生成机动伪标签 (基于全量派生特征, 依赖 alt_rate / heading_rate)
    # ------------------------------------------------------------------
    labels_all, label_info = generate_maneuver_labels(
        features=features,
        feature_names=feature_names,
        verbose=True,
    )
    labels_train, labels_test = _split_by_time(labels_all, TRAIN_RATIO)
    print(f"  标签生成使用的特征 (已记入 leak 清单): "
          f"{label_info['leak_features']}\n")

    # ------------------------------------------------------------------
    # 4) 分组归一化 (方案 D)
    #    - 位置列 (lat, lon, alt): 使用 MinMaxScaler (与目标对齐)
    #    - 其他列: StandardScaler (按用户规格)
    #    位置列在标准化 / 目标里均使用 "仅 train fit" 的 MinMax，
    #    因此 GRU 输入位置列与目标列的数值完全一致。
    # ------------------------------------------------------------------
    print("=" * 60)
    print("  [intent] 分组归一化 — 位置 MinMax / 其他 StandardScaler (方案 D)")
    print("=" * 60)

    # 位置列在 features 矩阵中的索引
    pos_indices: List[int] = [feature_names.index(c) for c in POSITION_COLS]
    other_indices: List[int] = [i for i in range(len(feature_names)) if i not in set(pos_indices)]

    print(f"  位置列索引 (MinMax): {pos_indices}  名称: {POSITION_COLS}")
    print(f"  其他列索引 (StandardScaler): 共 {len(other_indices)} 列")

    features_train, features_test = _split_by_time(features, TRAIN_RATIO)

    # -- 位置子矩阵: 使用 MinMax (与前面 mm_scaler 完全一致)
    pos_train = features_train[:, pos_indices]
    pos_test = features_test[:, pos_indices]
    pos_all = features[:, pos_indices]
    pos_mm = MinMaxScaler(feature_range=(0, 1))
    pos_mm.fit(pos_train)
    pos_train_norm = pos_mm.transform(pos_train)
    pos_test_norm = pos_mm.transform(pos_test)
    pos_all_norm = pos_mm.transform(pos_all)

    # -- 其他列子矩阵: StandardScaler
    if len(other_indices) > 0:
        other_train = features_train[:, other_indices]
        other_test = features_test[:, other_indices]
        other_all = features[:, other_indices]
        std_scaler = StandardScaler()
        std_scaler.fit(other_train)
        other_train_norm = std_scaler.transform(other_train)
        other_test_norm = std_scaler.transform(other_test)
        other_all_norm = std_scaler.transform(other_all)
    else:
        std_scaler = None
        other_train_norm = np.zeros((features_train.shape[0], 0))
        other_test_norm = np.zeros((features_test.shape[0], 0))
        other_all_norm = np.zeros((features.shape[0], 0))

    # 拼回完整的 "归一化后特征矩阵" (与 features 列顺序对齐)
    n_train, n_test, n_all = len(features_train), len(features_test), len(features)
    n_feats = features.shape[1]
    features_scaled_train = np.empty((n_train, n_feats), dtype=np.float64)
    features_scaled_test = np.empty((n_test, n_feats), dtype=np.float64)
    features_scaled_all = np.empty((n_all, n_feats), dtype=np.float64)
    features_scaled_train[:, pos_indices] = pos_train_norm
    features_scaled_test[:, pos_indices] = pos_test_norm
    features_scaled_all[:, pos_indices] = pos_all_norm
    if len(other_indices) > 0:
        features_scaled_train[:, other_indices] = other_train_norm
        features_scaled_test[:, other_indices] = other_test_norm
        features_scaled_all[:, other_indices] = other_all_norm

    print(f"  features_scaled_train: {features_scaled_train.shape}")
    print(f"  features_scaled_test:  {features_scaled_test.shape}\n")

    # ------------------------------------------------------------------
    # 5) RF 特征重要性 (方案 C: 排除标签生成特征)
    # ------------------------------------------------------------------
    rf, rf_kept_indices = fit_random_forest(
        X=features_scaled_train,
        y=labels_train,
        n_estimators=RF_N_ESTIMATORS,
        criterion=RF_CRITERION,
        random_state=RF_RANDOM_STATE,
        verbose=True,
        feature_names=feature_names,
        exclude_names=list(LABEL_LEAK_FEATURES),       # ★ 方案 C
    )

    rf_selected_indices, rf_selected_names, importances_full = \
        select_features_by_cumulative_importance(
            rf=rf,
            feature_names=feature_names,
            cumulative_threshold=RF_CUMULATIVE_THRESHOLD,
            kept_indices=rf_kept_indices,
            verbose=True,
        )

    # ------------------------------------------------------------------
    # 6) 合并"强制保留的位置列" ∪ "RF 筛选列" → 最终 GRU 特征索引 (方案 A)
    # ------------------------------------------------------------------
    print("=" * 60)
    print("  [intent] 合并位置列 ∪ RF 筛选列  (方案 A: 防止位置信息丢失)")
    print("=" * 60)

    final_feature_indices = _union_preserve_order(pos_indices, rf_selected_indices)
    final_layout_desc = _format_selected_layout(
        feature_names, final_feature_indices, pos_indices, rf_selected_indices,
    )
    print(f"  位置列 (POS): {[feature_names[i] for i in pos_indices]}")
    print(f"  RF 筛选列 (RF): {rf_selected_names}")
    print(f"  合并后最终 {len(final_feature_indices)} 列: {final_layout_desc}")
    print()

    feat_final_train = features_scaled_train[:, final_feature_indices]
    feat_final_test = features_scaled_test[:, final_feature_indices]
    feat_final_all = features_scaled_all[:, final_feature_indices]

    # ------------------------------------------------------------------
    # 7) OvA SVM + Platt Scaling — 使用 K-Fold OOF 消除分布偏移
    #    【公平性修正】训练集概率通过 5-Fold OOF 生成 (out-of-sample)，
    #    与测试集的 out-of-sample 推理分布一致，消除 covariate shift。
    # ------------------------------------------------------------------
    print("=" * 60)
    print(f"  [intent] OvA SVM 训练 (RBF, C={SVM_C}, gamma={SVM_GAMMA}, "
          f"probability=True) — K-Fold OOF 公平性修正")
    print("=" * 60)

    feat_svm_train = features_scaled_train[:, rf_selected_indices]
    feat_svm_test = features_scaled_test[:, rf_selected_indices]
    feat_svm_all = features_scaled_all[:, rf_selected_indices]

    svm_clf = IntentSVMClassifier(
        n_classes=SVM_N_CLASSES,
        C=SVM_C,
        gamma=SVM_GAMMA,
        kernel=SVM_KERNEL,
        random_state=42,
    )

    probs_train, svm_clf = svm_clf.fit_predict_proba_oof(
        feat_svm_train, labels_train, n_splits=5, verbose=True,
    )

    train_acc = svm_clf.score(feat_svm_train, labels_train)
    test_acc = svm_clf.score(feat_svm_test, labels_test)
    print(f"  ✅ SVM 训练完成 (全量训练集最终模型)")
    print(f"  SVM 输入维度 = {feat_svm_train.shape[1]} (RF 筛选列)")
    print(f"  训练集准确率 (最终模型 in-sample) = {train_acc:.4f}")
    print(f"  测试集准确率 (最终模型 out-of-sample) = {test_acc:.4f}")

    preds_test = svm_clf.predict(feat_svm_test)
    target_names = [f"{c}-{MANEUVER_LABEL_MAP[c][1]}" for c in range(SVM_N_CLASSES)]
    print("\n  测试集分类报告:")
    report_str = classification_report(
        labels_test, preds_test,
        target_names=target_names,
        labels=list(range(SVM_N_CLASSES)),
        zero_division=0,
    )
    print(report_str)

    # ------------------------------------------------------------------
    # 8) 对全部时间步计算 4 维意图概率向量
    # ------------------------------------------------------------------
    print("=" * 60)
    print("  [intent] 计算意图概率 (训练集=OOF, 测试集=out-of-sample)")
    print("=" * 60)

    probs_test = svm_clf.predict_proba(feat_svm_test)

    print(f"  probs_train shape: {probs_train.shape}   "
          f"(样本 0 概率: {np.round(probs_train[0], 4)})")
    print(f"  probs_test  shape: {probs_test.shape}   "
          f"(样本 0 概率: {np.round(probs_test[0], 4)})")
    print()

    # ------------------------------------------------------------------
    # 9) 构建增广滑动窗口序列: 位置列 ∪ RF 列  ⊕  SVM 概率
    # ------------------------------------------------------------------
    print("=" * 60)
    print("  [intent] 构建增广滑动窗口: [位置 ∪ RF 筛选列  ⊕  4 维 SVM 概率]")
    print("=" * 60)

    target_train, target_test = _split_by_time(target_all, TRAIN_RATIO)

    X_train, Y_train = build_augmented_sequences(
        feat_selected=feat_final_train,
        intent_probs=probs_train,
        target=target_train,
        look_back=LOOK_BACK,
        forward_length=FORWARD_LENGTH,
        dataset_name="训练集",
        verbose=True,
    )
    X_test, Y_test = build_augmented_sequences(
        feat_selected=feat_final_test,
        intent_probs=probs_test,
        target=target_test,
        look_back=LOOK_BACK,
        forward_length=FORWARD_LENGTH,
        dataset_name="测试集",
        verbose=True,
    )

    # ------------------------------------------------------------------
    # 10) 保存全部产物
    # ------------------------------------------------------------------
    print("=" * 60)
    print("  [intent] 保存产物")
    print("=" * 60)

    np.save(os.path.join(OUTPUT_DIR, "X_train_intent.npy"), X_train)
    np.save(os.path.join(OUTPUT_DIR, "Y_train_intent.npy"), Y_train)
    np.save(os.path.join(OUTPUT_DIR, "X_test_intent.npy"), X_test)
    np.save(os.path.join(OUTPUT_DIR, "Y_test_intent.npy"), Y_test)
    np.save(os.path.join(OUTPUT_DIR, "selected_indices.npy"), final_feature_indices)
    np.save(os.path.join(OUTPUT_DIR, "rf_selected_indices.npy"), rf_selected_indices)
    np.save(os.path.join(OUTPUT_DIR, "rf_importances.npy"), importances_full)
    np.save(os.path.join(OUTPUT_DIR, "labels_train.npy"), labels_train)
    np.save(os.path.join(OUTPUT_DIR, "labels_test.npy"), labels_test)
    np.savez(
        os.path.join(OUTPUT_DIR, "minmax_scaler_params.npz"),
        data_min=mm_scaler.data_min_,
        data_max=mm_scaler.data_max_,
        feature_range=np.array(mm_scaler.feature_range),
    )

    with open(os.path.join(OUTPUT_DIR, "position_minmax_scaler.pkl"), "wb") as fp:
        pickle.dump(pos_mm, fp)
    if std_scaler is not None:
        with open(os.path.join(OUTPUT_DIR, "std_scaler.pkl"), "wb") as fp:
            pickle.dump(std_scaler, fp)
    with open(os.path.join(OUTPUT_DIR, "rf_model.pkl"), "wb") as fp:
        pickle.dump(rf, fp)
    with open(os.path.join(OUTPUT_DIR, "svm_model.pkl"), "wb") as fp:
        pickle.dump(svm_clf, fp)

    # 汇总 JSON (便于人工查看)
    report = {
        "version": "v3-fairness-ABCDEF",
        "look_back": LOOK_BACK,
        "forward_length": FORWARD_LENGTH,
        "downsample_factor": DOWNSAMPLE_FACTOR,
        "train_ratio": TRAIN_RATIO,
        "feature_names": feature_names,
        "position_cols": POSITION_COLS,
        "position_indices": pos_indices,
        "leak_features": list(LABEL_LEAK_FEATURES),
        "rf_cumulative_threshold": RF_CUMULATIVE_THRESHOLD,
        "rf_kept_indices": rf_kept_indices.tolist(),
        "rf_selected_indices": rf_selected_indices.tolist(),
        "rf_selected_names": rf_selected_names,
        "final_feature_indices": final_feature_indices.tolist(),
        "final_feature_names": [feature_names[i] for i in final_feature_indices],
        "final_feature_sources": final_layout_desc,
        "rf_feature_importances": {
            name: float(importance)
            for name, importance in zip(feature_names, importances_full.tolist())
        },
        "svm": {
            "kernel": SVM_KERNEL,
            "C": SVM_C,
            "gamma": SVM_GAMMA,
            "probability": True,
            "strategy": "OneVsRest",
            "input_columns": rf_selected_names,
            "train_accuracy": train_acc,
            "test_accuracy": test_acc,
            "oof_enabled": True,
            "oof_n_splits": 5,
        },
        "label_info": label_info,
        "maneuver_classes": {
            str(c): {"zh": zh, "en": en}
            for c, (zh, en) in MANEUVER_LABEL_MAP.items()
        },
        "augmented_input_size": int(X_train.shape[-1]),
        "shapes": {
            "X_train": list(X_train.shape),
            "Y_train": list(Y_train.shape),
            "X_test": list(X_test.shape),
            "Y_test": list(Y_test.shape),
        },
    }
    with open(os.path.join(OUTPUT_DIR, "intent_report.json"), "w", encoding="utf-8") as fp:
        json.dump(report, fp, ensure_ascii=False, indent=2)

    print("  ✅ 产物已保存至:")
    print(f"     {OUTPUT_DIR}/")
    for name in [
        "X_train_intent.npy", "Y_train_intent.npy",
        "X_test_intent.npy",  "Y_test_intent.npy",
        "selected_indices.npy", "rf_selected_indices.npy",
        "rf_importances.npy",
        "labels_train.npy", "labels_test.npy",
        "minmax_scaler_params.npz",
        "position_minmax_scaler.pkl", "std_scaler.pkl",
        "rf_model.pkl", "svm_model.pkl",
        "intent_report.json",
    ]:
        path = os.path.join(OUTPUT_DIR, name)
        size = os.path.getsize(path) if os.path.exists(path) else 0
        print(f"       - {name:<30s}  ({size / 1024:>8.1f} KiB)")

    print("\n" + "▓" * 60)
    print("  意图识别数据准备完成! (公平性修正 A/B/C/D/E/F 综合版) 后续可运行:")
    print("    python train_with_intent.py")
    print("    python visualize_with_intent.py")
    print("    python compare_models.py")
    print("▓" * 60 + "\n")


if __name__ == "__main__":
    main()
