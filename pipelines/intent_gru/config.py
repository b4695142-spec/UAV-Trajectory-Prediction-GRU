"""
==============================================================================
意图增强 GRU 管线 — 统一配置
==============================================================================
将分散在 prepare_intent.py / train_with_intent.py / visualize_with_intent.py
/ compare_models.py 中的所有超参数与路径配置集中管理。

与纯 GRU 管线共享的参数 (如 LOOK_BACK, TRAIN_RATIO) 在此独立定义，
确保意图管线可独立运行，同时便于与纯 GRU 管线对齐校验。
==============================================================================
"""

import os

# ============================================================================
# 项目路径
# ============================================================================
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

# ============================================================================
# 数据路径 — 纯 GRU 产物 (compare_models.py 依赖)
# ============================================================================
PURE_DATA_DIR = os.path.join(PROJECT_ROOT, "processed_data")
PURE_X_TEST = os.path.join(PURE_DATA_DIR, "X_test.npy")
PURE_Y_TEST = os.path.join(PURE_DATA_DIR, "Y_test.npy")
PURE_MODEL = os.path.join(PURE_DATA_DIR, "best_gru_model.pth")
PURE_SCALER = os.path.join(PURE_DATA_DIR, "scaler_params.npz")

# ============================================================================
# 数据路径 — 意图增强产物
# ============================================================================
INTENT_DIR = os.path.join(PROJECT_ROOT, "processed_data", "intent")

X_TRAIN_PATH = os.path.join(INTENT_DIR, "X_train_intent.npy")
Y_TRAIN_PATH = os.path.join(INTENT_DIR, "Y_train_intent.npy")
X_TEST_PATH = os.path.join(INTENT_DIR, "X_test_intent.npy")
Y_TEST_PATH = os.path.join(INTENT_DIR, "Y_test_intent.npy")
SCALER_PATH = os.path.join(INTENT_DIR, "minmax_scaler_params.npz")
MODEL_SAVE_PATH = os.path.join(INTENT_DIR, "best_gru_model_intent.pth")
MODEL_PATH = MODEL_SAVE_PATH

# compare_models.py 使用的意图管线路径/超参别名
INTENT_X_TEST = X_TEST_PATH
INTENT_Y_TEST = Y_TEST_PATH
INTENT_SCALER = SCALER_PATH
INTENT_MODEL = MODEL_PATH

# ============================================================================
# 原始数据路径 (prepare_intent.py)
# ============================================================================
RAW_CSV_PATH = os.path.join(PROJECT_ROOT, "Log Files", "OnboardGPS.csv")

# ============================================================================
# 输出路径
# ============================================================================
OUTPUT_DIR = INTENT_DIR
OUTPUT_IMG_DIR = os.path.join(PROJECT_ROOT, "output", "intent_gru")
OUTPUT_CMP_DIR = os.path.join(PROJECT_ROOT, "output", "comparison")

OUTPUT_IMG_PATH = os.path.join(OUTPUT_IMG_DIR, "trajectory_3d_plot_intent.png")
OUTPUT_2D_ERROR_PATH = os.path.join(OUTPUT_IMG_DIR, "trajectory_2d_error_intent.png")
OUTPUT_TIME_PATH = os.path.join(OUTPUT_IMG_DIR, "inference_time_plot_intent.png")

OUT_CMP_ERR = os.path.join(OUTPUT_CMP_DIR, "compare_2d_error.png")
OUT_CMP_3D = os.path.join(OUTPUT_CMP_DIR, "compare_3d_trajectory.png")
OUT_CMP_CDF = os.path.join(OUTPUT_CMP_DIR, "compare_error_cdf.png")
OUT_METRICS = os.path.join(OUTPUT_CMP_DIR, "compare_metrics.json")

# ============================================================================
# 数据预处理参数 (与纯 GRU 管线一致，保证时间轴对齐)
# ============================================================================
ORIGINAL_INTERVAL_S = 0.033333
TARGET_INTERVAL_S = 0.1
DOWNSAMPLE_FACTOR = round(TARGET_INTERVAL_S / ORIGINAL_INTERVAL_S)
TRAIN_RATIO = 0.8

# ============================================================================
# 滑动窗口参数 (与纯 GRU 管线一致)
# ============================================================================
LOOK_BACK = 50
FORWARD_LENGTH = 0

# ============================================================================
# RF 特征筛选参数 (prepare_intent.py)
# ============================================================================
RF_N_ESTIMATORS = 200
RF_CRITERION = "entropy"
RF_CUMULATIVE_THRESHOLD = 0.80
RF_RANDOM_STATE = 42

# ============================================================================
# SVM 参数 (prepare_intent.py)
# ============================================================================
SVM_C = 5.0
SVM_GAMMA = 0.01
SVM_KERNEL = "rbf"
SVM_N_CLASSES = 4

# ============================================================================
# 位置列配置 (方案 A: 强制保留)
# ============================================================================
POSITION_COLS = ["lat", "lon", "alt"]

# ============================================================================
# 模型超参数 (与纯 GRU 管线一致，确保对比公平)
# ============================================================================
PURE_INPUT_SIZE = 3
HIDDEN_SIZE = 64
NUM_LAYERS = 2
OUTPUT_SIZE = 3
DROPOUT = 0.0
INTENT_DROPOUT = DROPOUT

# ============================================================================
# 训练超参数 (与纯 GRU 管线严格一致)
# ============================================================================
BATCH_SIZE = 70
LEARNING_RATE = 1e-3
MAX_EPOCHS = 500
PATIENCE = 15

# ============================================================================
# 可视化参数 (visualize_with_intent.py)
# ============================================================================
PLOT_START = 0
PLOT_END = None
