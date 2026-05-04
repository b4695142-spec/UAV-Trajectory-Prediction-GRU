"""
==============================================================================
UAV 轨迹预测 — Attention-Bi-GRU 推理与可视化
==============================================================================
功能:
    1. 加载 prepare_data.py 生成的 StandardScaler 归一化测试集 + 训练好的模型权重
    2. 执行推理 (拆分增广 X 中的结构性特征 / 意图概率)
    3. ★ 使用 StandardScaler 反归一化 (论文 Equation 8 对应的逆变换)
       —— 与 pure_gru / intent_gru 使用 MinMaxScaler 反归一化的关键差异
    4. 绘制 3D 轨迹对比图 / 2D 误差曲线 / 单次推理耗时图

输出:
    - trajectory_3d_plot_attn_bigru.png
    - trajectory_2d_error_attn_bigru.png
    - inference_time_plot_attn_bigru.png
==============================================================================
"""

from __future__ import annotations

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.font_manager import FontManager

from config import (
    D_FF,
    DROPOUT,
    HIDDEN_SIZE,
    MODEL_PATH,
    N_DEC_LAYERS,
    N_DECODE_STEPS,
    N_ENC_LAYERS,
    N_INTENT,
    OUTPUT_IMG_DIR,
    OUTPUT_IMG_PATH,
    OUTPUT_SIZE,
    PLOT_END,
    PLOT_START,
    PROJECT_ROOT,
    SCALER_PATH,
    X_TEST_PATH,
    Y_TEST_PATH,
)

if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)
from core.attention_bigru_model import AttentionBiGRU


# ============================================================================
# 中文字体配置
# ============================================================================
def setup_chinese_font():
    font_candidates = {
        'win32': ['SimHei', 'Microsoft YaHei', 'SimSun'],
        'linux': ['WenQuanYi Micro Hei', 'WenQuanYi Zen Hei',
                  'Noto Sans CJK SC', 'DejaVu Sans'],
        'darwin': ['PingFang SC', 'Heiti TC', 'STHeiti', 'Arial Unicode MS'],
    }
    platform = sys.platform
    if platform not in font_candidates:
        print(f"⚠️  警告: 未识别的操作系统 '{platform}'")
        return
    available_fonts = set(FontManager().get_font_names())
    for font_name in font_candidates[platform]:
        if font_name in available_fonts:
            plt.rcParams['font.sans-serif'] = [font_name]
            plt.rcParams['axes.unicode_minus'] = False
            print(f"✅ 已配置中文字体: {font_name}")
            return
    print(f"⚠️  警告: 系统中未找到可用的中文字体")


setup_chinese_font()


# ============================================================================
# 步骤 1: 数据 + 模型加载 + 推理
# ============================================================================
def load_and_predict() -> tuple:
    print("=" * 60)
    print("  步骤 1: 数据与模型加载 + 推理 (Attention-Bi-GRU)")
    print("=" * 60)

    for p in (X_TEST_PATH, Y_TEST_PATH, MODEL_PATH, SCALER_PATH):
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"未找到 {p}\n"
                f"请先运行:\n"
                f"  python pipelines/attention_bigru/prepare_data.py\n"
                f"  python pipelines/attention_bigru/train.py"
            )

    X_test = np.load(X_TEST_PATH)
    Y_test = np.load(Y_TEST_PATH)
    print(f"  X_test: {X_test.shape}  Y_test: {Y_test.shape}")

    # 拆分: 最后 N_INTENT 列为 SVM 概率, 其余为结构性特征
    input_size = X_test.shape[-1] - N_INTENT
    X_feat_test = X_test[:, :, :-N_INTENT]
    X_intent_test = X_test[:, :, -N_INTENT:]

    print(f"  动态 input_size = {input_size}, n_intent = {N_INTENT}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  设备: {device}")

    model = AttentionBiGRU(
        input_size=input_size,
        hidden_size=HIDDEN_SIZE,
        output_size=OUTPUT_SIZE,
        n_enc_layers=N_ENC_LAYERS,
        n_dec_layers=N_DEC_LAYERS,
        d_ff=D_FF,
        n_intent=N_INTENT,
        n_decode_steps=N_DECODE_STEPS,
        dropout=DROPOUT,
    )
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    print(f"  ✅ Attention-Bi-GRU 模型权重已加载: {MODEL_PATH}")

    # 张量化 + 推理 (分块以避免单次张量过大)
    X_feat_t = torch.tensor(X_feat_test, dtype=torch.float32).to(device)
    X_intent_t = torch.tensor(X_intent_test, dtype=torch.float32).to(device)

    Y_pred_list = []
    chunk = 512
    with torch.no_grad():
        for i in range(0, len(X_feat_t), chunk):
            x_f = X_feat_t[i:i + chunk]
            x_i = X_intent_t[i:i + chunk]
            out = model(x_f, x_i)                # (chunk, output_size) [N_DECODE_STEPS=1]
            Y_pred_list.append(out.cpu().numpy())
    Y_pred = np.concatenate(Y_pred_list, axis=0)

    print(f"  ✅ 推理完成! Y_pred shape: {Y_pred.shape}\n")

    return Y_pred, Y_test


# ============================================================================
# 步骤 2: StandardScaler 反归一化
# ============================================================================
def inverse_standard(data: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """
    StandardScaler 反归一化:  X_original = X_scaled * σ + μ

    参数:
        data:  归一化后的数据, shape (n, 3)
        mean:  各特征均值 μ, shape (3,)
        scale: 各特征标准差 σ, shape (3,)

    返回:
        反归一化后的真实坐标 (lat, lon, alt), shape (n, 3)
    """
    return data * scale + mean


def denormalize(Y_pred: np.ndarray, Y_test: np.ndarray) -> tuple:
    """加载 StandardScaler 参数, 对预测和真实值执行反归一化, 输出误差指标。"""
    print("=" * 60)
    print("  步骤 2: 反归一化 (StandardScaler — 论文 Equation 8 逆变换)")
    print("=" * 60)

    params = np.load(SCALER_PATH)
    mean = params["mean"]
    scale = params["scale"]

    print(f"  target μ (mean):  {mean}")
    print(f"  target σ (scale): {scale}")

    Y_pred_real = inverse_standard(Y_pred, mean, scale)
    Y_test_real = inverse_standard(Y_test, mean, scale)

    labels = ["Latitude", "Longitude", "Altitude"]
    mae = np.mean(np.abs(Y_pred_real - Y_test_real), axis=0)
    rmse = np.sqrt(np.mean((Y_pred_real - Y_test_real) ** 2, axis=0))
    avg_rmse = float(np.mean(rmse))
    euclid = np.sqrt(np.sum((Y_pred_real - Y_test_real) ** 2, axis=1))
    mean_euc = float(np.mean(euclid))
    max_euc = float(np.max(euclid))
    p95_euc = float(np.percentile(euclid, 95))
    p99_euc = float(np.percentile(euclid, 99))

    print(f"\n  [Attn-Bi-GRU] 各维度 MAE:")
    for i, name in enumerate(labels):
        unit = "°" if i < 2 else "m"
        print(f"    {name:>10}: {mae[i]:.8f} {unit}")

    print(f"\n  [Attn-Bi-GRU] 各维度 RMSE:")
    for i, name in enumerate(labels):
        unit = "°" if i < 2 else "m"
        print(f"    {name:>10}: {rmse[i]:.8f} {unit}")

    print(f"\n  综合误差:")
    print(f"    Average RMSE        : {avg_rmse:.8f}")
    print(f"    Mean 3D Euclidean   : {mean_euc:.4f} m")
    print(f"    Max  3D Euclidean   : {max_euc:.4f} m  ← 长视距离群值指标")
    print(f"    P95  3D Euclidean   : {p95_euc:.4f} m")
    print(f"    P99  3D Euclidean   : {p99_euc:.4f} m")
    print()

    return Y_pred_real, Y_test_real


# ============================================================================
# 步骤 3: 3D 轨迹对比图
# ============================================================================
def plot_3d_trajectory(Y_pred_real: np.ndarray, Y_test_real: np.ndarray) -> None:
    print("=" * 60)
    print("  步骤 3: 3D 轨迹图 (Attention-Bi-GRU)")
    print("=" * 60)

    plot_end = len(Y_test_real) if PLOT_END is None else PLOT_END
    actual = Y_test_real[PLOT_START:plot_end]
    pred = Y_pred_real[PLOT_START:plot_end]

    fig = plt.figure(figsize=(14, 10), dpi=150)
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(actual[:, 1], actual[:, 0], actual[:, 2],
            linestyle="--", color="gray", linewidth=1.5,
            label="实际轨迹", alpha=0.8)
    ax.plot(pred[:, 1], pred[:, 0], pred[:, 2],
            linestyle="-", color="royalblue", linewidth=1.5,
            label="预测轨迹（Attention-Bi-GRU + 意图）", alpha=0.9)
    ax.scatter(actual[0, 1], actual[0, 0], actual[0, 2],
               color="green", s=80, marker="o", zorder=5, label="起点")
    ax.scatter(actual[-1, 1], actual[-1, 0], actual[-1, 2],
               color="blue", s=80, marker="^", zorder=5, label="终点")

    ax.set_xlabel("经度 (°)", fontsize=12, labelpad=10)
    ax.set_ylabel("纬度 (°)", fontsize=12, labelpad=10)
    ax.set_zlabel("高度 (m)", fontsize=12, labelpad=10)
    ax.set_title("无人机三维轨迹: 实际轨迹 vs Attention-Bi-GRU + 意图 预测",
                 fontsize=14, fontweight="bold", pad=20)
    ax.legend(loc="upper left", fontsize=10, framealpha=0.9)
    ax.view_init(elev=25, azim=135)
    plt.tight_layout()
    fig.savefig(OUTPUT_IMG_PATH, dpi=300, bbox_inches="tight", pad_inches=0.3)
    print(f"  ✅ 已保存: {OUTPUT_IMG_PATH}\n")


# ============================================================================
# 主函数
# ============================================================================
def main():
    print("\n" + "▓" * 60)
    print("  UAV 轨迹预测 — Attention-Bi-GRU 测试集推理与可视化")
    print("▓" * 60 + "\n")

    os.makedirs(OUTPUT_IMG_DIR, exist_ok=True)

    Y_pred, Y_test = load_and_predict()
    Y_pred_real, Y_test_real = denormalize(Y_pred, Y_test)
    plot_3d_trajectory(Y_pred_real, Y_test_real)

    print("▓" * 60)
    print("  Attention-Bi-GRU 可视化完成!")
    print("▓" * 60 + "\n")
    plt.show()


if __name__ == "__main__":
    main()
