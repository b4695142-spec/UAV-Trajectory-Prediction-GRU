"""
==============================================================================
UAV 机动意图识别 (Maneuver Intent Recognition) 模块
==============================================================================
本包提供独立的"级联意图识别"子系统，用于增强主 GRU 轨迹预测模型：

    原始序列特征  →  [RF 特征筛选] → [OvA SVM + Platt Scaling]  →  4 维概率向量
                                                                      │
                                                                      ↓
                          GRU 输入 = [筛选后的特征, 4 维概率] (拼接)

本包与现有纯 GRU 管线完全解耦：
    - 不修改任何原始脚本 (preprocess_uav.py / build_sequences.py / train.py / visualize.py)
    - 所有产物存放于 processed_data/intent/ 子目录
    - 随时可通过运行原始 train.py 回退到纯 GRU 版本
==============================================================================
"""

from .feature_extractor import (
    extract_intent_features,
    DERIVED_FEATURE_NAMES,
    DROP_COLUMNS,
)
from .label_generator import (
    generate_maneuver_labels,
    MANEUVER_CLASSES,
    MANEUVER_LABEL_MAP,
    LABEL_LEAK_FEATURES,
)
from .rf_selector import (
    fit_random_forest,
    select_features_by_cumulative_importance,
)
from .svm_classifier import IntentSVMClassifier
from .intent_dataset import IntentAugmentedDataset, build_augmented_sequences

__all__ = [
    "extract_intent_features",
    "DERIVED_FEATURE_NAMES",
    "DROP_COLUMNS",
    "generate_maneuver_labels",
    "MANEUVER_CLASSES",
    "MANEUVER_LABEL_MAP",
    "LABEL_LEAK_FEATURES",
    "fit_random_forest",
    "select_features_by_cumulative_importance",
    "IntentSVMClassifier",
    "IntentAugmentedDataset",
    "build_augmented_sequences",
]
