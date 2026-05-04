"""
==============================================================================
UAV 轨迹预测 — Attention-Bi-GRU 数据准备 (StandardScaler + 意图识别)
==============================================================================
本脚本是 Attention-Bi-GRU 管线的入口，负责:

    1. 动态提取 OnboardGPS.csv 中的数值特征 + 派生运动学特征
    2. ★ 全部特征使用 StandardScaler 归一化 (论文 Equation 8 Z-score)
       —— 这是与 intent_gru/prepare_intent.py 的核心差异 (后者位置列用 MinMax)
    3. 自动生成 4 类机动伪标签 (平飞 / 转弯 / 爬升 / 俯冲)
    4. Random Forest 特征重要性评估 → 保留累计贡献率 80% 的核心特征
       ★ 是否剔除"标签生成特征" (alt_rate / heading_rate) 由配置项
         EXCLUDE_LABEL_LEAK_FEATURES 控制; 本管线默认 False, 即不剔除,
         让意图分类器输入与标签定义一致 (会显著拉高 SVM 准确率, 知情接受)。
    5. OvA SVM + Platt Scaling (C=5, gamma=0.01, probability=True) 训练
       ★ 使用 5-Fold OOF 生成训练集概率，消除训练/测试分布偏移
    6. 对全部时间步计算 4 维意图概率向量
    7. 构建增广滑动窗口序列:
            模型输入 = [StandardScaler(lat,lon,alt) ⊕ StandardScaler(RF 筛选的非位置特征) ⊕ SVM 4 维概率]
       Y 为 StandardScaler 归一化后的 (lat, lon, alt)，反归一化时使用 target_scaler.inverse_transform()。

所有产物保存至 `processed_data/attention_bigru/`，与 intent_gru 完全解耦。

【与 intent_gru/prepare_intent.py 的核心差异】
    | 维度       | intent_gru/prepare_intent.py | attention_bigru/prepare_data.py    |
    | -------- | ---------------------------- | ---------------------------------- |
    | 位置列归一化   | MinMaxScaler [0, 1]          | ★ StandardScaler (论文 Eq.8)         |
    | 其他列归一化   | StandardScaler               | StandardScaler (一致)                |
    | 目标 Y 归一化 | MinMaxScaler [0, 1]          | ★ StandardScaler (论文 Eq.8)         |
    | 产物目录     | processed_data/intent/       | processed_data/attention_bigru/    |
    | 意图识别子模块  | 直接调用                       | import 复用 (pipelines.intent_gru.intent) |
==============================================================================
"""

from __future__ import annotations

import json
import os
import pickle
import sys
from typing import List, Tuple

import numpy as np
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler

from config import (
    DOWNSAMPLE_FACTOR,
    EXCLUDE_LABEL_LEAK_FEATURES,
    FORWARD_LENGTH,
    LOOK_BACK,
    OUTPUT_DIR,
    POSITION_COLS,
    PROJECT_ROOT,
    RAW_CSV_PATH,
    RF_CRITERION,
    RF_CUMULATIVE_THRESHOLD,
    RF_N_ESTIMATORS,
    RF_RANDOM_STATE,
    SVM_C,
    SVM_GAMMA,
    SVM_KERNEL,
    SVM_N_CLASSES,
    SVM_OOF_SPLITS,
    TARGET_INTERVAL_S,
    TRAIN_RATIO,
)

# ============================================================================
# 复用 intent_gru/intent/ 子模块 (代码逻辑复用, 不复用产物)
# —— 使用包路径 pipelines.intent_gru.intent，避免运行时改 sys.path 且便于静态分析
# ============================================================================
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
from pipelines.intent_gru.intent import (  # noqa: E402
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
# 工具函数
# ============================================================================
def _split_by_time(arr: np.ndarray, train_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    """按时间顺序严格切分 (不打乱)，与现有管线完全一致。"""
    n_total = len(arr)
    n_train = int(n_total * train_ratio)
    return arr[:n_train], arr[n_train:]


def _union_preserve_order(
    always_keep: List[int],
    rf_selected: np.ndarray,
) -> np.ndarray:
    """合并"始终保留列"与"RF 筛选列"为一个升序、去重的索引数组。"""
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
    print("  Attention-Bi-GRU 数据准备 — StandardScaler (论文 Eq.8) 版本")
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

    # ------------------------------------------------------------------
    # 2) 构造回归目标 (lat, lon, alt) 并做 StandardScaler 归一化
    # ★ 论文 Equation 8: X_scaled(i,j) = (X(i,j) - μ_j) / σ_j
    # ------------------------------------------------------------------
    print("=" * 60)
    print("  [attn_bigru] 构造回归目标 — StandardScaler 归一化 (lat, lon, alt)")
    print("=" * 60)

    missing = [c for c in POSITION_COLS if c not in df_down.columns]
    if missing:
        raise ValueError(f"目标位置列缺失: {missing}")

    position = df_down[POSITION_COLS].to_numpy(dtype=np.float64)   # (N, 3)
    position_train, position_test = _split_by_time(position, TRAIN_RATIO)

    # ★ 仅在训练集上 fit StandardScaler, 消除测试集泄漏
    target_scaler = StandardScaler()
    target_scaler.fit(position_train)
    target_all = target_scaler.transform(position)                 # (N, 3)

    print(f"  position 原始 shape: {position.shape}")
    print(f"  train/test 切分:     {len(position_train)} / {len(position_test)}")
    print(f"  target μ (mean): {target_scaler.mean_}")
    print(f"  target σ (std):  {target_scaler.scale_}")
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
    # 4) ★ 全部特征列均使用 StandardScaler 归一化 (论文 Equation 8)
    #    与 intent_gru/prepare_intent.py 的差异:
    #      - intent_gru: 位置列 MinMaxScaler / 其他列 StandardScaler
    #      - attn_bigru: 位置列 StandardScaler / 其他列 StandardScaler  (统一)
    # ------------------------------------------------------------------
    print("=" * 60)
    print("  [attn_bigru] 全部特征 StandardScaler 归一化 (论文 Equation 8)")
    print("=" * 60)

    # 位置列在 features 矩阵中的索引 (用于产物报告)
    pos_indices: List[int] = [feature_names.index(c) for c in POSITION_COLS]
    other_indices: List[int] = [
        i for i in range(len(feature_names)) if i not in set(pos_indices)
    ]

    print(f"  位置列索引 (StandardScaler): {pos_indices}  名称: {POSITION_COLS}")
    print(f"  其他列索引 (StandardScaler): 共 {len(other_indices)} 列")

    features_train, features_test = _split_by_time(features, TRAIN_RATIO)

    # ★ 一次性对全部列 fit + transform (统一使用 StandardScaler)
    feat_scaler = StandardScaler()
    feat_scaler.fit(features_train)
    features_scaled_train = feat_scaler.transform(features_train)
    features_scaled_test = feat_scaler.transform(features_test)
    features_scaled_all = feat_scaler.transform(features)

    print(f"  features_scaled_train: {features_scaled_train.shape}")
    print(f"  features_scaled_test:  {features_scaled_test.shape}")
    print(f"  feature μ[0..3]: {feat_scaler.mean_[:3]}")
    print(f"  feature σ[0..3]: {feat_scaler.scale_[:3]}\n")

    # ------------------------------------------------------------------
    # 5) RF 特征重要性
    #    EXCLUDE_LABEL_LEAK_FEATURES = True  → 排除 alt_rate / heading_rate
    #                                          (与 intent_gru 行为一致)
    #    EXCLUDE_LABEL_LEAK_FEATURES = False → 不排除 (本管线默认):
    #                                          RF 与 SVM 都能看到这两列,
    #                                          保证 SVM 输入包含标签定义所用特征。
    # ------------------------------------------------------------------
    rf_exclude_names = (
        list(LABEL_LEAK_FEATURES) if EXCLUDE_LABEL_LEAK_FEATURES else None
    )
    if not EXCLUDE_LABEL_LEAK_FEATURES:
        print(f"  ℹ️  EXCLUDE_LABEL_LEAK_FEATURES=False — "
              f"alt_rate / heading_rate 将进入 RF/SVM 候选池 "
              f"(知情接受其对 SVM 准确率的拉高效应)")

    rf, rf_kept_indices = fit_random_forest(
        X=features_scaled_train,
        y=labels_train,
        n_estimators=RF_N_ESTIMATORS,
        criterion=RF_CRITERION,
        random_state=RF_RANDOM_STATE,
        verbose=True,
        feature_names=feature_names,
        exclude_names=rf_exclude_names,
    )

    rf_selected_indices, rf_selected_names, importances_full = \
        select_features_by_cumulative_importance(
            rf=rf,
            feature_names=feature_names,
            cumulative_threshold=RF_CUMULATIVE_THRESHOLD,
            kept_indices=rf_kept_indices,
            # 仅在排除模式下启用泄漏特征校验; 不排除模式下这两列被允许出现
            leak_feature_names=rf_exclude_names,
            verbose=True,
        )

    # ------------------------------------------------------------------
    # 6) 合并"位置列 ∪ RF 筛选列" → 最终模型特征索引
    # ------------------------------------------------------------------
    print("=" * 60)
    print("  [attn_bigru] 合并位置列 ∪ RF 筛选列 (防止位置信息丢失)")
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

    # ------------------------------------------------------------------
    # 7) OvA SVM + Platt Scaling (5-Fold OOF 训练集概率, 消除分布偏移)
    #    ★ 当 EXCLUDE_LABEL_LEAK_FEATURES = False 时, 显式确保 alt_rate /
    #      heading_rate 出现在 SVM 输入中 (满足"SVM 必须包含标签生成特征"约束),
    #      即使 RF 累计阈值碰巧没有挑到其中之一也兜底。
    # ------------------------------------------------------------------
    print("=" * 60)
    print(f"  [attn_bigru] OvA SVM 训练 (RBF, C={SVM_C}, gamma={SVM_GAMMA}, "
          f"probability=True) — {SVM_OOF_SPLITS}-Fold OOF")
    print("=" * 60)

    if EXCLUDE_LABEL_LEAK_FEATURES:
        svm_input_indices = rf_selected_indices
    else:
        leak_idx = [
            feature_names.index(name)
            for name in LABEL_LEAK_FEATURES
            if name in feature_names
        ]
        svm_input_indices = np.array(
            sorted(set(rf_selected_indices.tolist()) | set(leak_idx)),
            dtype=np.int64,
        )
    svm_input_names = [feature_names[i] for i in svm_input_indices]
    print(f"  SVM 输入列 ({len(svm_input_indices)}): {svm_input_names}")

    feat_svm_train = features_scaled_train[:, svm_input_indices]
    feat_svm_test = features_scaled_test[:, svm_input_indices]

    svm_clf = IntentSVMClassifier(
        n_classes=SVM_N_CLASSES,
        C=SVM_C,
        gamma=SVM_GAMMA,
        kernel=SVM_KERNEL,
        random_state=42,
    )
    probs_train, svm_clf = svm_clf.fit_predict_proba_oof(
        feat_svm_train, labels_train, n_splits=SVM_OOF_SPLITS, verbose=True,
    )

    train_acc = svm_clf.score(feat_svm_train, labels_train)
    test_acc = svm_clf.score(feat_svm_test, labels_test)
    print(f"  ✅ SVM 训练完成 (全量训练集最终模型)")
    print(f"  SVM 输入维度 = {feat_svm_train.shape[1]} "
          f"(= RF 筛选列"
          f"{' ∪ 标签生成特征' if not EXCLUDE_LABEL_LEAK_FEATURES else ''})")
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
    # 8) 对测试集计算意图概率向量 (训练集已通过 OOF 生成)
    # ------------------------------------------------------------------
    probs_test = svm_clf.predict_proba(feat_svm_test)
    print(f"  probs_train shape: {probs_train.shape}   "
          f"(样本 0 概率: {np.round(probs_train[0], 4)})")
    print(f"  probs_test  shape: {probs_test.shape}   "
          f"(样本 0 概率: {np.round(probs_test[0], 4)})\n")

    # ------------------------------------------------------------------
    # 9) 构建增广滑动窗口序列: [位置 ∪ RF 筛选列  ⊕  4 维 SVM 概率]
    # ------------------------------------------------------------------
    print("=" * 60)
    print("  [attn_bigru] 构建增广滑动窗口: [位置 ∪ RF 筛选列 ⊕ 4 维 SVM 概率]")
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
    print("  [attn_bigru] 保存产物")
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

    # ★ StandardScaler 参数保存为 .npz (供 visualize / compare 反归一化)
    np.savez(
        os.path.join(OUTPUT_DIR, "scaler_params.npz"),
        mean=target_scaler.mean_,
        scale=target_scaler.scale_,
        feat_mean=feat_scaler.mean_,
        feat_scale=feat_scaler.scale_,
    )

    # 同时保存完整的 sklearn StandardScaler 实例 (.pkl) 便于完整反演
    with open(os.path.join(OUTPUT_DIR, "target_std_scaler.pkl"), "wb") as fp:
        pickle.dump(target_scaler, fp)
    with open(os.path.join(OUTPUT_DIR, "feature_std_scaler.pkl"), "wb") as fp:
        pickle.dump(feat_scaler, fp)
    with open(os.path.join(OUTPUT_DIR, "rf_model.pkl"), "wb") as fp:
        pickle.dump(rf, fp)
    with open(os.path.join(OUTPUT_DIR, "svm_model.pkl"), "wb") as fp:
        pickle.dump(svm_clf, fp)

    # 汇总 JSON
    report = {
        "version": "attn_bigru-v1-standardscaler",
        "look_back": LOOK_BACK,
        "forward_length": FORWARD_LENGTH,
        "downsample_factor": DOWNSAMPLE_FACTOR,
        "train_ratio": TRAIN_RATIO,
        "normalization": {
            "method": "StandardScaler",
            "paper_reference": "Equation 8 (Section 3.1)",
            "scope": "全部特征 + 目标 Y 均使用 StandardScaler",
            "fit_strategy": "仅在训练集上 fit, 避免数据泄漏",
        },
        "feature_names": feature_names,
        "position_cols": POSITION_COLS,
        "position_indices": pos_indices,
        # alt_rate / heading_rate 是用来生成伪标签的源特征。
        # 是否在 RF/SVM 中排除它们由 EXCLUDE_LABEL_LEAK_FEATURES 控制。
        "label_source_features": list(LABEL_LEAK_FEATURES),
        "leak_features": list(LABEL_LEAK_FEATURES),  # (兼容字段, 同 label_source_features)
        "exclude_label_leak_features_in_rf": EXCLUDE_LABEL_LEAK_FEATURES,
        "rf_cumulative_threshold": RF_CUMULATIVE_THRESHOLD,
        "rf_kept_indices": rf_kept_indices.tolist(),
        "rf_selected_indices": rf_selected_indices.tolist(),
        "rf_selected_names": rf_selected_names,
        # final_feature_indices 表示最终送入 GRU 的"上下文"特征列索引,
        # 等于 位置列 ∪ rf_selected_indices; 当不排除标签生成特征时,
        # 它们会因 RF 高重要性而进入这里 (即同时被 GRU 看到)。
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
            # SVM 实际输入列 = RF 筛选列 (排除模式)
            #              或 RF 筛选列 ∪ 标签生成特征 (不排除模式, 兜底保证)
            "input_indices": svm_input_indices.tolist(),
            "input_columns": svm_input_names,
            "input_includes_label_source_features": (
                not EXCLUDE_LABEL_LEAK_FEATURES
            ),
            "train_accuracy": train_acc,
            "test_accuracy": test_acc,
            "oof_enabled": True,
            "oof_n_splits": SVM_OOF_SPLITS,
        },
        "label_info": label_info,
        "maneuver_classes": {
            str(c): {"zh": zh, "en": en}
            for c, (zh, en) in MANEUVER_LABEL_MAP.items()
        },
        "augmented_input_size": int(X_train.shape[-1]),
        "n_intent": int(SVM_N_CLASSES),
        "shapes": {
            "X_train": list(X_train.shape),
            "Y_train": list(Y_train.shape),
            "X_test":  list(X_test.shape),
            "Y_test":  list(Y_test.shape),
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
        "scaler_params.npz",
        "target_std_scaler.pkl", "feature_std_scaler.pkl",
        "rf_model.pkl", "svm_model.pkl",
        "intent_report.json",
    ]:
        path = os.path.join(OUTPUT_DIR, name)
        size = os.path.getsize(path) if os.path.exists(path) else 0
        print(f"       - {name:<30s}  ({size / 1024:>8.1f} KiB)")

    print("\n" + "▓" * 60)
    print("  Attention-Bi-GRU 数据准备完成! 后续可运行:")
    print("    python pipelines/attention_bigru/train.py")
    print("    python pipelines/attention_bigru/visualize.py")
    print("    python pipelines/attention_bigru/compare_all.py")
    print("▓" * 60 + "\n")


if __name__ == "__main__":
    main()
