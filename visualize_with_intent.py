"""
==============================================================================
UAV 轨迹预测 GRU (意图识别增强版) — 推理与可视化
==============================================================================
与 visualize.py 功能等价，但从 processed_data/intent/ 加载增广数据集
与增广模型权重，输出加 `_intent` 后缀的可视化图:

    - trajectory_3d_plot_intent.png
    - trajectory_2d_error_intent.png
    - inference_time_plot_intent.png

本脚本不触碰任何纯 GRU 产物。
==============================================================================
"""

from __future__ import annotations

import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.font_manager import FontManager

from gru_model import UAVTrajectoryGRU


# ============================================================================
# 中文字体配置 (与 visualize.py 一致)
# ============================================================================
def setup_chinese_font():
    font_candidates = {
        'win32': ['SimHei', 'Microsoft YaHei', 'SimSun'],
        'linux': ['WenQuanYi Micro Hei', 'WenQuanYi Zen Hei', 'Noto Sans CJK SC', 'DejaVu Sans'],
        'darwin': ['PingFang SC', 'Heiti TC', 'STHeiti', 'Arial Unicode MS'],
    }
    platform = sys.platform
    if platform not in font_candidates:
        print(f"⚠️  警告: 未识别的操作系统 '{platform}'，尝试使用默认字体配置")
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
# 路径配置
# ============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INTENT_DIR = os.path.join(BASE_DIR, "processed_data", "intent")

X_TEST_PATH = os.path.join(INTENT_DIR, "X_test_intent.npy")
Y_TEST_PATH = os.path.join(INTENT_DIR, "Y_test_intent.npy")
SCALER_PATH = os.path.join(INTENT_DIR, "minmax_scaler_params.npz")
MODEL_PATH = os.path.join(INTENT_DIR, "best_gru_model_intent.pth")

OUTPUT_IMG_PATH = os.path.join(BASE_DIR, "trajectory_3d_plot_intent.png")
OUTPUT_2D_ERROR_PATH = os.path.join(BASE_DIR, "trajectory_2d_error_intent.png")
OUTPUT_TIME_PATH = os.path.join(BASE_DIR, "inference_time_plot_intent.png")

# 模型超参数 (与训练时一致)
HIDDEN_SIZE = 64
NUM_LAYERS = 2
OUTPUT_SIZE = 3
DROPOUT = 0.0

# 绘图区间
PLOT_START = 0
PLOT_END = None


# ============================================================================
# 推理
# ============================================================================
def load_and_predict() -> tuple:
    print("=" * 60)
    print("  步骤 1: 数据与模型加载 + 推理 (意图增广版)")
    print("=" * 60)

    for p in (X_TEST_PATH, Y_TEST_PATH, MODEL_PATH, SCALER_PATH):
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"未找到 {p}\n请先运行:\n  python prepare_intent.py\n  python train_with_intent.py"
            )

    X_test = np.load(X_TEST_PATH)
    Y_test = np.load(Y_TEST_PATH)
    print(f"  X_test: {X_test.shape}  Y_test: {Y_test.shape}")

    input_size = int(X_test.shape[-1])
    print(f"  动态 input_size = {input_size}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  设备: {device}")

    model = UAVTrajectoryGRU(
        input_size=input_size,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        output_size=OUTPUT_SIZE,
        dropout=DROPOUT,
    )
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    print(f"  ✅ 意图增广版模型权重已加载: {MODEL_PATH}")

    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    with torch.no_grad():
        Y_pred_tensor = model(X_test_tensor)
    Y_pred = Y_pred_tensor.cpu().numpy()

    print(f"  ✅ 推理完成! Y_pred shape: {Y_pred.shape}\n")
    return Y_pred, Y_test, model, X_test_tensor


# ============================================================================
# 反归一化 + 误差指标
# ============================================================================
def inverse_transform(data, data_min, data_max):
    return data * (data_max - data_min) + data_min


def denormalize(Y_pred, Y_test):
    print("=" * 60)
    print("  步骤 2: 反归一化 + 误差指标")
    print("=" * 60)

    params = np.load(SCALER_PATH)
    data_min = params["data_min"]
    data_max = params["data_max"]

    print(f"  data_min: {data_min}")
    print(f"  data_max: {data_max}")

    Y_pred_real = inverse_transform(Y_pred, data_min, data_max)
    Y_test_real = inverse_transform(Y_test, data_min, data_max)

    labels = ["Latitude", "Longitude", "Altitude"]
    mae = np.mean(np.abs(Y_pred_real - Y_test_real), axis=0)
    rmse = np.sqrt(np.mean((Y_pred_real - Y_test_real) ** 2, axis=0))
    avg_rmse = float(np.mean(rmse))
    euclid = np.sqrt(np.sum((Y_pred_real - Y_test_real) ** 2, axis=1))
    mean_euc = float(np.mean(euclid))
    max_euc = float(np.max(euclid))
    p95_euc = float(np.percentile(euclid, 95))

    print(f"\n  [intent 增广版] 各维度 MAE:")
    for i, name in enumerate(labels):
        unit = "°" if i < 2 else "m"
        print(f"    {name:>10}: {mae[i]:.8f} {unit}")

    print(f"\n  [intent 增广版] 各维度 RMSE:")
    for i, name in enumerate(labels):
        unit = "°" if i < 2 else "m"
        print(f"    {name:>10}: {rmse[i]:.8f} {unit}")

    print(f"\n  综合误差:")
    print(f"    Average RMSE        : {avg_rmse:.8f}")
    print(f"    Mean 3D Euclidean   : {mean_euc:.4f} m")
    print(f"    Max  3D Euclidean   : {max_euc:.4f} m  ← 长视距离群值指标")
    print(f"    P95  3D Euclidean   : {p95_euc:.4f} m")
    print()

    return Y_pred_real, Y_test_real


# ============================================================================
# 3D 轨迹绘图
# ============================================================================
def plot_3d_trajectory(Y_pred_real, Y_test_real):
    print("=" * 60)
    print("  步骤 3: 3D 轨迹图 (意图增广版)")
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
            linestyle="-", color="orange", linewidth=1.5,
            label="预测轨迹（GRU + 意图）", alpha=0.9)
    ax.scatter(actual[0, 1], actual[0, 0], actual[0, 2],
               color="green", s=80, marker="o", zorder=5, label="起点")
    ax.scatter(actual[-1, 1], actual[-1, 0], actual[-1, 2],
               color="blue", s=80, marker="^", zorder=5, label="终点")

    ax.set_xlabel("经度 (°)", fontsize=12, labelpad=10)
    ax.set_ylabel("纬度 (°)", fontsize=12, labelpad=10)
    ax.set_zlabel("高度 (m)", fontsize=12, labelpad=10)
    ax.set_title("无人机三维轨迹：实际轨迹 vs GRU+意图 预测轨迹",
                 fontsize=14, fontweight="bold", pad=20)
    ax.legend(loc="upper left", fontsize=10, framealpha=0.9)
    ax.view_init(elev=25, azim=135)
    plt.tight_layout()
    fig.savefig(OUTPUT_IMG_PATH, dpi=300, bbox_inches="tight", pad_inches=0.3)
    print(f"  ✅ 已保存: {OUTPUT_IMG_PATH}\n")


# ============================================================================
# 2D 误差曲线
# ============================================================================
def plot_2d_error_chart(Y_pred_real, Y_test_real):
    print("=" * 60)
    print("  步骤 4: 2D 综合误差曲线 (意图增广版)")
    print("=" * 60)

    plot_end = len(Y_test_real) if PLOT_END is None else PLOT_END
    actual = Y_test_real[PLOT_START:plot_end]
    pred = Y_pred_real[PLOT_START:plot_end]
    err = np.sqrt(np.sum((pred - actual) ** 2, axis=1))

    fig, ax = plt.subplots(figsize=(12, 6), dpi=150)
    time_steps = np.arange(PLOT_START, plot_end)
    ax.plot(time_steps, err, linestyle="-", color="orange",
            linewidth=2.0, label="综合真实误差 (GRU+意图)")

    ax.set_title("GRU+意图 综合预测误差随时间变化曲线",
                 fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("时间步", fontsize=12, labelpad=10)
    ax.set_ylabel("综合真实误差", fontsize=12, labelpad=10)
    ax.grid(True, linestyle="--", alpha=0.7)
    ax.legend(loc="upper right", fontsize=11, framealpha=0.9)
    plt.tight_layout()
    fig.savefig(OUTPUT_2D_ERROR_PATH, dpi=300, bbox_inches="tight", pad_inches=0.1)
    print(f"  ✅ 已保存: {OUTPUT_2D_ERROR_PATH}\n")


# ============================================================================
# 推理耗时
# ============================================================================
def plot_inference_time(model, X_test_tensor):
    print("=" * 60)
    print("  步骤 5: 单次预测耗时评估 (意图增广版)")
    print("=" * 60)

    model.eval()
    times = []

    plot_end = len(X_test_tensor) if PLOT_END is None else PLOT_END
    X_subset = X_test_tensor[PLOT_START:plot_end]
    print(f"  正在计算单次预测耗时，共 {len(X_subset)} 个点...")

    with torch.no_grad():
        _ = model(X_subset[0].unsqueeze(0))      # 预热
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        for i in range(len(X_subset)):
            x = X_subset[i].unsqueeze(0)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            s = time.perf_counter()
            _ = model(x)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            e = time.perf_counter()
            times.append((e - s) * 1000.0)

    times = np.array(times)
    avg = float(times.mean())
    print(f"  单次耗时 (ms): 平均 {avg:.4f}, 最小 {times.min():.4f}, 最大 {times.max():.4f}")

    fig, ax = plt.subplots(figsize=(12, 6), dpi=150)
    time_steps = np.arange(PLOT_START, plot_end)
    ax.plot(time_steps, times, linestyle="-", color="teal",
            linewidth=1.5, label="单次预测耗时 (GRU+意图)", alpha=0.8)
    ax.axhline(avg, color="red", linestyle="--", linewidth=1.5,
               label=f"平均耗时: {avg:.4f} ms")
    ax.set_title("GRU+意图 单次预测耗时随时间步变化",
                 fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("时间步", fontsize=12, labelpad=10)
    ax.set_ylabel("耗时 (毫秒 / ms)", fontsize=12, labelpad=10)
    ax.grid(True, linestyle="--", alpha=0.7)
    ax.legend(loc="upper right", fontsize=11, framealpha=0.9)
    plt.tight_layout()
    fig.savefig(OUTPUT_TIME_PATH, dpi=300, bbox_inches="tight", pad_inches=0.1)
    print(f"  ✅ 已保存: {OUTPUT_TIME_PATH}\n")


# ============================================================================
# 主函数
# ============================================================================
def main():
    print("\n" + "▓" * 60)
    print("  UAV 轨迹预测 — 意图增广版测试集推理与可视化")
    print("▓" * 60 + "\n")

    Y_pred, Y_test, model, X_test_tensor = load_and_predict()
    Y_pred_real, Y_test_real = denormalize(Y_pred, Y_test)
    plot_3d_trajectory(Y_pred_real, Y_test_real)
    plot_2d_error_chart(Y_pred_real, Y_test_real)
    plot_inference_time(model, X_test_tensor)

    print("▓" * 60)
    print("  意图增广版可视化完成!")
    print("▓" * 60 + "\n")
    plt.show()


if __name__ == "__main__":
    main()
