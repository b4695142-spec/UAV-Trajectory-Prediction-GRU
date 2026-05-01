"""
==============================================================================
Attention-Bi-GRU 意图增强管线 — 统一配置
==============================================================================
集中管理本管线的全部超参数与路径配置:
    - 数据准备 (prepare_data.py)
    - 训练   (train.py)
    - 可视化 (visualize.py)
    - 对比   (compare_all.py)

★ 与 pure_gru / intent_gru 的关键差异:
    - 归一化: StandardScaler (论文 Equation 8 Z-score), 而非 MinMaxScaler
    - 模型: Attention-Bi-GRU (动态 Attention + 每层 GeLU 意图融合)
    - 训练超参数: lr=5e-4, batch=64, dropout=0.2, epochs=300 (论文 Table 3)
==============================================================================
"""

import os

# ============================================================================
# 项目路径
# ============================================================================
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

# ============================================================================
# 原始数据路径
# ============================================================================
RAW_CSV_PATH = os.path.join(PROJECT_ROOT, "Log Files", "OnboardGPS.csv")

# ============================================================================
# 数据路径 — 本管线独立产物 (StandardScaler 归一化, 由 prepare_data.py 生成)
# ============================================================================
DATA_DIR = os.path.join(PROJECT_ROOT, "processed_data", "attention_bigru")

X_TRAIN_PATH = os.path.join(DATA_DIR, "X_train_intent.npy")
Y_TRAIN_PATH = os.path.join(DATA_DIR, "Y_train_intent.npy")
X_TEST_PATH  = os.path.join(DATA_DIR, "X_test_intent.npy")
Y_TEST_PATH  = os.path.join(DATA_DIR, "Y_test_intent.npy")

# StandardScaler 参数 (压缩 .npz 形式, 与 pure_gru 风格保持一致, 便于 compare_all)
SCALER_PATH = os.path.join(DATA_DIR, "scaler_params.npz")

# ============================================================================
# 模型权重保存路径
# ============================================================================
OUTPUT_DIR = DATA_DIR
MODEL_SAVE_PATH = os.path.join(OUTPUT_DIR, "best_attention_bigru_model.pth")
MODEL_PATH = MODEL_SAVE_PATH

# ============================================================================
# 单管线可视化输出路径
# ============================================================================
OUTPUT_IMG_DIR = os.path.join(PROJECT_ROOT, "output", "attention_bigru")
OUTPUT_CMP_DIR = os.path.join(PROJECT_ROOT, "output", "comparison")

OUTPUT_IMG_PATH = os.path.join(OUTPUT_IMG_DIR, "trajectory_3d_plot_attn_bigru.png")
OUTPUT_2D_ERROR_PATH = os.path.join(OUTPUT_IMG_DIR, "trajectory_2d_error_attn_bigru.png")
OUTPUT_TIME_PATH = os.path.join(OUTPUT_IMG_DIR, "inference_time_plot_attn_bigru.png")

# ============================================================================
# 双管线对比输出路径 (compare_all.py)
# ============================================================================
OUT_CMP_ERR = os.path.join(OUTPUT_CMP_DIR, "compare_2d_error_all.png")
OUT_CMP_3D  = os.path.join(OUTPUT_CMP_DIR, "compare_3d_trajectory_all.png")
OUT_CMP_CDF = os.path.join(OUTPUT_CMP_DIR, "compare_error_cdf_all.png")
OUT_METRICS = os.path.join(OUTPUT_CMP_DIR, "compare_metrics_all.json")

# ============================================================================
# pure_gru 管线产物路径 (compare_all.py 加载基线模型)
# ============================================================================
PURE_DATA_DIR = os.path.join(PROJECT_ROOT, "processed_data")
PURE_X_TEST  = os.path.join(PURE_DATA_DIR, "X_test.npy")
PURE_Y_TEST  = os.path.join(PURE_DATA_DIR, "Y_test.npy")
PURE_MODEL   = os.path.join(PURE_DATA_DIR, "best_gru_model.pth")
PURE_SCALER  = os.path.join(PURE_DATA_DIR, "scaler_params.npz")

# ============================================================================
# 数据预处理参数 (与 pure_gru / intent_gru 一致, 保证时间轴对齐)
# ============================================================================
ORIGINAL_INTERVAL_S = 0.033333
TARGET_INTERVAL_S = 0.1
DOWNSAMPLE_FACTOR = round(TARGET_INTERVAL_S / ORIGINAL_INTERVAL_S)
TRAIN_RATIO = 0.8

# ============================================================================
# 滑动窗口参数 (与 pure_gru 一致, 保证对比公平)
# ============================================================================
LOOK_BACK = 50
FORWARD_LENGTH = 0

# ============================================================================
# RF 特征筛选参数 (复用 intent_gru 设定)
# ============================================================================
RF_N_ESTIMATORS = 200
RF_CRITERION = "entropy"
RF_CUMULATIVE_THRESHOLD = 0.80
RF_RANDOM_STATE = 42

# ----------------------------------------------------------------------------
# 是否在 RF / SVM 候选池中排除"标签生成特征" (alt_rate / heading_rate)
#   False (默认, 本管线推荐): 不排除 — 允许 RF 与 SVM 把 alt_rate / heading_rate
#                            作为输入。伪标签由这两列定义, 因而会显著拉高 SVM 准
#                            确率, 但可让意图分类器输入与标签定义一致, 便于与
#                            intent_gru 做受控对比。
#   True:                    与 intent_gru 行为一致, 防止数据泄漏污染 RF 重要性
#                            分布。
# ----------------------------------------------------------------------------
EXCLUDE_LABEL_LEAK_FEATURES = False

# ============================================================================
# SVM 参数 (复用 intent_gru 设定)
# ============================================================================
SVM_C = 5.0
SVM_GAMMA = 0.01
SVM_KERNEL = "rbf"
SVM_N_CLASSES = 4
SVM_OOF_SPLITS = 5

# ============================================================================
# 位置列配置 (始终保留: 防止位置信息丢失)
# ============================================================================
POSITION_COLS = ["lat", "lon", "alt"]

# ============================================================================
# 模型超参数 (论文 Table 3)
# ============================================================================
HIDDEN_SIZE = 64
N_ENC_LAYERS = 4
N_DEC_LAYERS = 4
D_FF = 128
DROPOUT = 0.2
OUTPUT_SIZE = 3
N_INTENT = 4
# 解码步数:
#   训练 / 推理 时模型自回归输出 N_DECODE_STEPS 步 (论文 Section 4.2 Seq2Seq 设计的核心,
#   只有 > 1 才会真正触发 output_proj 自回归循环 + 动态 Q/α 的"动态"特性);
#   与 pure_gru 对比时 (compare_all.py) 仅取首步 [:, 0, :], 保证两者预测的目标时刻
#   完全对齐 (target[t]), 任务定义一致 → 公平。
N_DECODE_STEPS = 5

# ============================================================================
# 训练超参数 (论文 Table 3)
# ============================================================================
BATCH_SIZE = 64
LEARNING_RATE = 5e-4
MAX_EPOCHS = 300
PATIENCE = 15

# ============================================================================
# 可视化参数
# ============================================================================
PLOT_START = 0
PLOT_END = None
