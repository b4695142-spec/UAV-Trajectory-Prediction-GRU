"""
==============================================================================
纯 GRU 管线 — 统一配置
==============================================================================
将分散在 preprocess_uav.py / build_sequences.py / train.py / visualize.py
中的所有超参数与路径配置集中管理，便于维护和参数调优。

修改参数时只需编辑本文件，无需逐一修改各脚本。
==============================================================================
"""

import os

# ============================================================================
# 项目路径
# ============================================================================
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

# ============================================================================
# 数据路径
# ============================================================================
RAW_CSV_PATH = os.path.join(PROJECT_ROOT, "Log Files", "OnboardGPS.csv")
DATA_DIR = os.path.join(PROJECT_ROOT, "processed_data")

TRAIN_DATA_PATH = os.path.join(DATA_DIR, "train_data.npy")
TEST_DATA_PATH = os.path.join(DATA_DIR, "test_data.npy")
SCALER_PATH = os.path.join(DATA_DIR, "scaler_params.npz")

X_TRAIN_PATH = os.path.join(DATA_DIR, "X_train.npy")
Y_TRAIN_PATH = os.path.join(DATA_DIR, "Y_train.npy")
X_TEST_PATH = os.path.join(DATA_DIR, "X_test.npy")
Y_TEST_PATH = os.path.join(DATA_DIR, "Y_test.npy")

MODEL_SAVE_PATH = os.path.join(DATA_DIR, "best_gru_model.pth")
MODEL_PATH = MODEL_SAVE_PATH

# ============================================================================
# 输出路径
# ============================================================================
OUTPUT_DIR = DATA_DIR
OUTPUT_IMG_PATH = os.path.join(PROJECT_ROOT, "trajectory_3d_plot.png")
OUTPUT_2D_ERROR_PATH = os.path.join(PROJECT_ROOT, "trajectory_2d_error.png")
OUTPUT_TIME_PATH = os.path.join(PROJECT_ROOT, "inference_time_plot.png")

# ============================================================================
# 数据预处理参数 (preprocess_uav.py)
# ============================================================================
ORIGINAL_INTERVAL_S = 0.033333
TARGET_INTERVAL_S = 0.1
DOWNSAMPLE_FACTOR = round(TARGET_INTERVAL_S / ORIGINAL_INTERVAL_S)
TRAIN_RATIO = 0.8

# ============================================================================
# 滑动窗口参数 (build_sequences.py)
# ============================================================================
LOOK_BACK = 50
FORWARD_LENGTH = 0

# ============================================================================
# 模型超参数 (train.py / visualize.py)
# ============================================================================
INPUT_SIZE = 3
HIDDEN_SIZE = 64
NUM_LAYERS = 2
OUTPUT_SIZE = 3
DROPOUT = 0.0

# ============================================================================
# 训练超参数 (train.py)
# ============================================================================
BATCH_SIZE = 70
LEARNING_RATE = 1e-3
MAX_EPOCHS = 500
PATIENCE = 15

# ============================================================================
# 可视化参数 (visualize.py)
# ============================================================================
PLOT_START = 0
PLOT_END = None
