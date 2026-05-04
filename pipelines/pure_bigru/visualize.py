"""
==============================================================================
UAV 轨迹预测 Bi-GRU 模型 — 测试集推理与 3D 轨迹可视化
==============================================================================
功能:
    1. 加载测试集数据与训练好的最佳 Bi-GRU 模型权重
    2. 执行推理，得到预测坐标
    3. 对预测值和真实值进行反归一化，还原为真实世界坐标
    4. 绘制 3D 轨迹对比图 (Actual vs Predicted)，保存为高清 PNG
==============================================================================
"""

import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontManager

from config import (
    HIDDEN_SIZE,
    INPUT_SIZE,
    MODEL_PATH,
    NUM_LAYERS,
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
    print(f"   候选字体列表: {', '.join(font_candidates[platform])}")
    print(f"   图表中的中文可能显示为方块，建议安装上述字体之一")


setup_chinese_font()


from core.bigru_model import UAVTrajectoryBiGRU


def load_and_predict() -> tuple:
    print("=" * 60)
    print("  步骤 1: 数据与模型加载 + 推理")
    print("=" * 60)

    X_test = np.load(X_TEST_PATH)
    Y_test = np.load(Y_TEST_PATH)
    print(f"  X_test: {X_test.shape}  Y_test: {Y_test.shape}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  设备: {device}")

    model = UAVTrajectoryBiGRU(
        input_size=INPUT_SIZE,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        output_size=OUTPUT_SIZE,
    )

    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    print(f"  ✅ 模型权重已加载: {MODEL_PATH}")

    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)

    with torch.no_grad():
        Y_pred_tensor = model(X_test_tensor)

    Y_pred = Y_pred_tensor.cpu().numpy()

    print(f"  ✅ 推理完成! 预测结果形状: {Y_pred.shape}")
    print()

    return Y_pred, Y_test


def inverse_transform(data: np.ndarray, data_min: np.ndarray, data_max: np.ndarray) -> np.ndarray:
    return data * (data_max - data_min) + data_min


def denormalize(Y_pred: np.ndarray, Y_test: np.ndarray) -> tuple:
    print("=" * 60)
    print("  步骤 2: 反归一化 (Inverse Transform)")
    print("=" * 60)

    scaler_params = np.load(SCALER_PATH)
    data_min = scaler_params["data_min"]
    data_max = scaler_params["data_max"]

    print(f"  data_min (lat, lon, alt): {data_min}")
    print(f"  data_max (lat, lon, alt): {data_max}")
    print(f"  data_range:               {data_max - data_min}")

    Y_pred_real = inverse_transform(Y_pred, data_min, data_max)
    Y_test_real = inverse_transform(Y_test, data_min, data_max)

    print(f"\n  反归一化后 — 真实值 (Y_test) 统计:")
    labels = ["Latitude", "Longitude", "Altitude"]
    for i, label in enumerate(labels):
        print(f"    {label:>10}: min={Y_test_real[:, i].min():.6f}, "
              f"max={Y_test_real[:, i].max():.6f}, "
              f"mean={Y_test_real[:, i].mean():.6f}")

    print(f"\n  反归一化后 — 预测值 (Y_pred) 统计:")
    for i, label in enumerate(labels):
        print(f"    {label:>10}: min={Y_pred_real[:, i].min():.6f}, "
              f"max={Y_pred_real[:, i].max():.6f}, "
              f"mean={Y_pred_real[:, i].mean():.6f}")

    mae = np.mean(np.abs(Y_pred_real - Y_test_real), axis=0)
    rmse = np.sqrt(np.mean((Y_pred_real - Y_test_real) ** 2, axis=0))
    avg_rmse = np.mean(rmse)

    euclidean_distances = np.sqrt(np.sum((Y_pred_real - Y_test_real) ** 2, axis=1))
    mean_euclidean = np.mean(euclidean_distances)

    print(f"\n  反归一化后 — 误差指标统计:")
    print(f"  各维度平均绝对误差 (MAE):")
    for i, label in enumerate(labels):
        unit = "°" if i < 2 else "m"
        if i < 2:
            print(f"    {label:>10}: {mae[i]:.8f} {unit}")
        else:
            print(f"    {label:>10}: {mae[i]:.4f} {unit}")

    print(f"\n  各维度均方根误差 (RMSE):")
    for i, label in enumerate(labels):
        unit = "°" if i < 2 else "m"
        if i < 2:
            print(f"    {label:>10}: {rmse[i]:.8f} {unit}")
        else:
            print(f"    {label:>10}: {rmse[i]:.4f} {unit}")

    print(f"\n  综合预测误差:")
    print(f"    Average RMSE    : {avg_rmse:.8f}")
    print(f"    平均欧氏距离误差: {mean_euclidean:.4f} (3D 空间)")
    print()

    return Y_pred_real, Y_test_real


def plot_3d_trajectory(Y_pred_real: np.ndarray, Y_test_real: np.ndarray) -> None:
    print("=" * 60)
    print("  步骤 3: 3D 轨迹对比图绘制")
    print("=" * 60)

    plot_end = len(Y_test_real) if PLOT_END is None else PLOT_END

    actual = Y_test_real[PLOT_START:plot_end]
    predicted = Y_pred_real[PLOT_START:plot_end]
    n_points = len(actual)
    print(f"  绘图区间: [{PLOT_START}, {plot_end}), 共 {n_points} 个点")

    lat_actual, lon_actual, alt_actual = actual[:, 0], actual[:, 1], actual[:, 2]
    lat_pred, lon_pred, alt_pred = predicted[:, 0], predicted[:, 1], predicted[:, 2]

    fig = plt.figure(figsize=(14, 10), dpi=150)
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(
        lon_actual, lat_actual, alt_actual,
        linestyle="--", color="gray", linewidth=1.5,
        label="实际轨迹", alpha=0.8,
    )

    ax.plot(
        lon_pred, lat_pred, alt_pred,
        linestyle="-", color="red", linewidth=1.5,
        label="预测轨迹（Bi-GRU）", alpha=0.9,
    )

    ax.scatter(actual[0][1], actual[0][0], actual[0][2], color="green", s=80, marker="o", zorder=5, label="起点")
    ax.scatter(actual[-1][1], actual[-1][0], actual[-1][2], color="blue", s=80, marker="^", zorder=5, label="终点")

    ax.set_xlabel("经度 (°)", fontsize=12, labelpad=10)
    ax.set_ylabel("纬度 (°)", fontsize=12, labelpad=10)
    ax.set_zlabel("高度 (m)", fontsize=12, labelpad=10)
    ax.set_title(
        f"无人机三维轨迹：实际轨迹与 Bi-GRU 预测轨迹对比",
        fontsize=14, fontweight="bold", pad=20,
    )

    ax.legend(loc="upper left", fontsize=10, framealpha=0.9)
    ax.view_init(elev=25, azim=135)
    plt.tight_layout()

    fig.savefig(OUTPUT_IMG_PATH, dpi=300, bbox_inches="tight", pad_inches=0.3)
    print(f"  ✅ 3D 轨迹图已保存: {OUTPUT_IMG_PATH}")
    print()


def main():
    print("\n" + "▓" * 60)
    print("  UAV 轨迹预测 Bi-GRU — 测试集推理与 3D 可视化")
    print("▓" * 60 + "\n")

    os.makedirs(OUTPUT_IMG_DIR, exist_ok=True)

    Y_pred, Y_test = load_and_predict()

    Y_pred_real, Y_test_real = denormalize(Y_pred, Y_test)

    plot_3d_trajectory(Y_pred_real, Y_test_real)

    print("▓" * 60)
    print("  全部任务完成! 正在弹窗显示可视化图表...")
    print("▓" * 60 + "\n")

    plt.show()


if __name__ == "__main__":
    main()
