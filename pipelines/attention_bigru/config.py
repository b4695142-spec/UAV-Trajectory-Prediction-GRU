"""
==============================================================================
Attention-Bi-GRU 意图增强管线 — 统一配置
==============================================================================
集中管理本管线的全部超参数与路径配置:
    - 数据准备 (prepare_data.py)
    - 训练   (train.py)
    - 可视化 (visualize.py)
    - 对比   (compare_all.py)

★ 与 pure_gru 的关键差异:
    - 模型: Bi-GRU + Attention + 意图融合 (编码器直出架构)
    - 输入: 位置 + RF 筛选特征 + SVM 4 维意图概率
    - 归一化: MinMaxScaler [0,1] (与 pure_gru 对齐)
==============================================================================
"""

import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

RAW_CSV_PATH = os.path.join(PROJECT_ROOT, "Log Files", "OnboardGPS.csv")

DATA_DIR = os.path.join(PROJECT_ROOT, "processed_data", "attention_bigru")

X_TRAIN_PATH = os.path.join(DATA_DIR, "X_train_intent.npy")
Y_TRAIN_PATH = os.path.join(DATA_DIR, "Y_train_intent.npy")
X_TEST_PATH  = os.path.join(DATA_DIR, "X_test_intent.npy")
Y_TEST_PATH  = os.path.join(DATA_DIR, "Y_test_intent.npy")

SCALER_PATH = os.path.join(DATA_DIR, "scaler_params.npz")

OUTPUT_DIR = DATA_DIR
MODEL_SAVE_PATH = os.path.join(OUTPUT_DIR, "best_attention_bigru_model.pth")
MODEL_PATH = MODEL_SAVE_PATH

OUTPUT_IMG_DIR = os.path.join(PROJECT_ROOT, "output", "attention_bigru")
OUTPUT_CMP_DIR = os.path.join(PROJECT_ROOT, "output", "comparison")

OUTPUT_IMG_PATH = os.path.join(OUTPUT_IMG_DIR, "trajectory_3d_plot_attn_bigru.png")
OUTPUT_2D_ERROR_PATH = os.path.join(OUTPUT_IMG_DIR, "trajectory_2d_error_attn_bigru.png")
OUTPUT_TIME_PATH = os.path.join(OUTPUT_IMG_DIR, "inference_time_plot_attn_bigru.png")
LOSS_CURVE_PATH = os.path.join(OUTPUT_IMG_DIR, "loss_curve_attn_bigru.png")

OUT_CMP_ERR = os.path.join(OUTPUT_CMP_DIR, "compare_2d_error_all.png")
OUT_CMP_3D  = os.path.join(OUTPUT_CMP_DIR, "compare_3d_trajectory_all.png")
OUT_CMP_CDF = os.path.join(OUTPUT_CMP_DIR, "compare_error_cdf_all.png")
OUT_CMP_DIR = OUTPUT_CMP_DIR
OUT_METRICS = os.path.join(OUTPUT_CMP_DIR, "compare_metrics_all.json")

PURE_DATA_DIR = os.path.join(PROJECT_ROOT, "processed_data")
PURE_X_TEST  = os.path.join(PURE_DATA_DIR, "X_test.npy")
PURE_Y_TEST  = os.path.join(PURE_DATA_DIR, "Y_test.npy")
PURE_MODEL   = os.path.join(PURE_DATA_DIR, "best_gru_model.pth")
PURE_SCALER  = os.path.join(PURE_DATA_DIR, "scaler_params.npz")

ORIGINAL_INTERVAL_S = 0.033333
TARGET_INTERVAL_S = 0.1
DOWNSAMPLE_FACTOR = round(TARGET_INTERVAL_S / ORIGINAL_INTERVAL_S)
TRAIN_RATIO = 0.8

LOOK_BACK = 50
FORWARD_LENGTH = 1

RF_N_ESTIMATORS = 200
RF_CRITERION = "entropy"
RF_CUMULATIVE_THRESHOLD = 0.80
RF_RANDOM_STATE = 42

EXCLUDE_LABEL_LEAK_FEATURES = False

SVM_C = 5.0
SVM_GAMMA = 0.01
SVM_KERNEL = "rbf"
SVM_N_CLASSES = 4
SVM_OOF_SPLITS = 5

POSITION_COLS = ["lat", "lon", "alt"]

HIDDEN_SIZE = 64
N_ENC_LAYERS = 2
N_DEC_LAYERS = 2
D_FF = 256
DROPOUT = 0.0
OUTPUT_SIZE = 3
N_INTENT = 4
N_DECODE_STEPS = 1

BATCH_SIZE = 70
LEARNING_RATE = 1e-3
MAX_EPOCHS = 500
PATIENCE = 15
CLIP_GRAD_NORM = 1.0

PLOT_START = 0
PLOT_END = None
