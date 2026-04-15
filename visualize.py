"""
==============================================================================
UAV 轨迹预测 GRU 模型 — 测试集推理与 3D 轨迹可视化
==============================================================================
功能:
    1. 加载测试集数据与训练好的最佳 GRU 模型权重
    2. 执行推理，得到预测坐标
    3. 对预测值和真实值进行反归一化，还原为真实世界坐标
    4. 绘制 3D 轨迹对比图 (Actual vs Predicted)，保存为高清 PNG
==============================================================================
"""

import os
import time
import numpy as np
import torch
import matplotlib.pyplot as plt

# 设置中文字体 (解决中文显示问题)
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
plt.rcParams['axes.unicode_minus'] = False     # 解决保存图像是负号'-'显示为方块的问题


# 从同目录导入模型类
from gru_model import UAVTrajectoryGRU


# ============================================================================
# 配置参数
# ============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "processed_data")

# 数据路径
X_TEST_PATH = os.path.join(DATA_DIR, "X_test.npy")
Y_TEST_PATH = os.path.join(DATA_DIR, "Y_test.npy")
SCALER_PATH = os.path.join(DATA_DIR, "scaler_params.npz")
MODEL_PATH = os.path.join(DATA_DIR, "best_gru_model.pth")

# 输出图片路径
OUTPUT_IMG_PATH = os.path.join(BASE_DIR, "trajectory_3d_plot.png")
OUTPUT_2D_ERROR_PATH = os.path.join(BASE_DIR, "trajectory_2d_error.png")

# 模型超参数 (与训练时一致)
INPUT_SIZE = 3
HIDDEN_SIZE = 64
NUM_LAYERS = 2
OUTPUT_SIZE = 3

# 可视化参数: 截取测试集中的连续片段用于绘图
PLOT_START = 0       # 绘图起始索引
PLOT_END = None      # 绘图结束索引 (设为 None 时表示绘制到最后所有的点)


# ============================================================================
# 步骤 1: 数据与模型加载 + 推理
# ============================================================================
def load_and_predict() -> tuple:
    """
    加载测试集与模型权重，执行推理。

    返回:
        Y_pred: numpy.ndarray, shape (n_test, 3), 模型预测的归一化坐标
        Y_test: numpy.ndarray, shape (n_test, 3), 真实的归一化坐标
        model: 加载权重的网络模型
        X_test_tensor: 测试集输入张量
    """
    print("=" * 60)
    print("  步骤 1: 数据与模型加载 + 推理")
    print("=" * 60)

    # 加载测试集
    X_test = np.load(X_TEST_PATH)
    Y_test = np.load(Y_TEST_PATH)
    print(f"  X_test: {X_test.shape}  Y_test: {Y_test.shape}")

    # 选择设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  设备: {device}")

    # 实例化模型
    model = UAVTrajectoryGRU(
        input_size=INPUT_SIZE,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        output_size=OUTPUT_SIZE,
    )

    # 加载最佳模型权重
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    print(f"  ✅ 模型权重已加载: {MODEL_PATH}")

    # 转换为张量并送入模型
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)

    with torch.no_grad():
        Y_pred_tensor = model(X_test_tensor)  # (n_test, 3)

    # 转回 numpy
    Y_pred = Y_pred_tensor.cpu().numpy()

    print(f"  ✅ 推理完成! 预测结果形状: {Y_pred.shape}")
    print()

    return Y_pred, Y_test, model, X_test_tensor


# ============================================================================
# 步骤 2: 反归一化 (Inverse Transform)
# ============================================================================
def inverse_transform(data: np.ndarray, data_min: np.ndarray, data_max: np.ndarray) -> np.ndarray:
    """
    MinMaxScaler 反归一化。

    公式: X_original = X_scaled * (data_max - data_min) + data_min

    参数:
        data:      归一化后的数据, shape (n, 3)
        data_min:  各特征最小值, shape (3,)
        data_max:  各特征最大值, shape (3,)

    返回:
        还原后的真实世界坐标, shape (n, 3)
    """
    return data * (data_max - data_min) + data_min


def denormalize(Y_pred: np.ndarray, Y_test: np.ndarray) -> tuple:
    """
    加载 scaler 参数并对预测值和真实值执行反归一化。

    返回:
        Y_pred_real: 反归一化后的预测坐标 (lat, lon, alt)
        Y_test_real: 反归一化后的真实坐标 (lat, lon, alt)
    """
    print("=" * 60)
    print("  步骤 2: 反归一化 (Inverse Transform)")
    print("=" * 60)

    # 加载归一化参数
    scaler_params = np.load(SCALER_PATH)
    data_min = scaler_params["data_min"]  # shape (3,)
    data_max = scaler_params["data_max"]  # shape (3,)

    print(f"  data_min (lat, lon, alt): {data_min}")
    print(f"  data_max (lat, lon, alt): {data_max}")
    print(f"  data_range:               {data_max - data_min}")

    # 反归一化
    Y_pred_real = inverse_transform(Y_pred, data_min, data_max)
    Y_test_real = inverse_transform(Y_test, data_min, data_max)

    # 打印反归一化后的统计信息
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

    # 计算各维度的平均绝对误差 (MAE) 和均方根误差 (RMSE)
    mae = np.mean(np.abs(Y_pred_real - Y_test_real), axis=0)
    rmse = np.sqrt(np.mean((Y_pred_real - Y_test_real) ** 2, axis=0))
    avg_rmse = np.mean(rmse)
    
    # 整体测试集的 3D 平均欧氏距离误差
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


# ============================================================================
# 步骤 3: 3D 轨迹对比图绘制
# ============================================================================
def plot_3d_trajectory(Y_pred_real: np.ndarray, Y_test_real: np.ndarray) -> None:
    """
    绘制 3D 轨迹对比图: 真实轨迹 (灰色虚线) vs 预测轨迹 (红色实线)。

    截取测试集中连续的一段 (PLOT_START:PLOT_END) 以清晰展示机动细节。
    """
    print("=" * 60)
    print("  步骤 3: 3D 轨迹对比图绘制")
    print("=" * 60)

    plot_end = len(Y_test_real) if PLOT_END is None else PLOT_END

    # 截取绘图片段
    actual = Y_test_real[PLOT_START:plot_end]
    predicted = Y_pred_real[PLOT_START:plot_end]
    n_points = len(actual)
    print(f"  绘图区间: [{PLOT_START}, {plot_end}), 共 {n_points} 个点")

    # 提取各维度
    lat_actual, lon_actual, alt_actual = actual[:, 0], actual[:, 1], actual[:, 2]
    lat_pred, lon_pred, alt_pred = predicted[:, 0], predicted[:, 1], predicted[:, 2]

    # ------------------------------------------------------------------
    # 创建 3D 图
    # ------------------------------------------------------------------
    fig = plt.figure(figsize=(14, 10), dpi=150)
    ax = fig.add_subplot(111, projection="3d")

    # 真实轨迹 — 灰色虚线
    ax.plot(
        lon_actual, lat_actual, alt_actual,
        linestyle="--", color="gray", linewidth=1.5,
        label="实际轨迹", alpha=0.8,
    )

    # 预测轨迹 — 红色实线
    ax.plot(
        lon_pred, lat_pred, alt_pred,
        linestyle="-", color="red", linewidth=1.5,
        label="预测轨迹（GRU）", alpha=0.9,
    )

    # 标记起点和终点
    ax.scatter(actual[0][1], actual[0][0], actual[0][2], color="green", s=80, marker="o", zorder=5, label="起点")
    ax.scatter(actual[-1][1], actual[-1][0], actual[-1][2], color="blue", s=80, marker="^", zorder=5, label="终点")

    # ------------------------------------------------------------------
    # 坐标轴标签与标题
    # ------------------------------------------------------------------
    ax.set_xlabel("经度 (°)", fontsize=12, labelpad=10)
    ax.set_ylabel("纬度 (°)", fontsize=12, labelpad=10)
    ax.set_zlabel("高度 (m)", fontsize=12, labelpad=10)
    ax.set_title(
        f"无人机三维轨迹：实际轨迹与 GRU 预测轨迹对比",
        fontsize=14, fontweight="bold", pad=20,
    )

    # 图例
    ax.legend(loc="upper left", fontsize=10, framealpha=0.9)

    # 调整视角使轨迹更直观
    ax.view_init(elev=25, azim=135)

    # 紧凑布局
    plt.tight_layout()

    # ------------------------------------------------------------------
    # 保存高清图片
    # ------------------------------------------------------------------
    fig.savefig(OUTPUT_IMG_PATH, dpi=300, bbox_inches="tight", pad_inches=0.3)
    print(f"  ✅ 3D 轨迹图已保存: {OUTPUT_IMG_PATH}")
    print()


# ============================================================================
# 步骤 4: 2D 综合误差折线图绘制
# ============================================================================
def plot_2d_error_chart(Y_pred_real: np.ndarray, Y_test_real: np.ndarray) -> None:
    """
    绘制截取片段内，每个时间步的预测点与真实点之间的 3D 欧氏距离真实误差，保存为 2D 折线图。
    """
    print("=" * 60)
    print("  步骤 4: 2D 综合误差折线图绘制")
    print("=" * 60)
    
    plot_end = len(Y_test_real) if PLOT_END is None else PLOT_END

    # 截取绘图片段
    actual = Y_test_real[PLOT_START:plot_end]
    predicted = Y_pred_real[PLOT_START:plot_end]
    
    # 计算每一个时间步下，真实点与预测点之间的 3D 欧氏距离 (综合误差)
    real_error = np.sqrt(np.sum((predicted - actual) ** 2, axis=1))

    # 创建 2D 折线图
    fig, ax = plt.subplots(figsize=(12, 6), dpi=150)
    
    time_steps = np.arange(PLOT_START, plot_end)
    
    # 红色实线
    ax.plot(
        time_steps, real_error,
        linestyle="-", color="red", linewidth=2.0,
        label="综合真实误差"
    )

    # 坐标轴标签与标题
    ax.set_title(
        f"GRU 综合预测误差随时间变化曲线", 
        fontsize=14, fontweight="bold", pad=15
    )
    ax.set_xlabel("时间步", fontsize=12, labelpad=10)
    ax.set_ylabel("综合真实误差", fontsize=12, labelpad=10)
    
    # 网格线
    ax.grid(True, linestyle="--", alpha=0.7)
    
    # 图例
    ax.legend(loc="upper right", fontsize=11, framealpha=0.9)

    plt.tight_layout()

    # 保存图片
    fig.savefig(OUTPUT_2D_ERROR_PATH, dpi=300, bbox_inches="tight", pad_inches=0.1)
    print(f"  ✅ 2D 误差折线图已保存: {OUTPUT_2D_ERROR_PATH}")
    print()


# ============================================================================
# 步骤 5: 单次预测耗时评估及绘图
# ============================================================================
def plot_inference_time(model, X_test_tensor) -> None:
    """
    评估模型在测试集中样本的单次预测耗时，并将其绘制为折线图。
    """
    print("=" * 60)
    print("  步骤 5: 单次预测耗时评估及绘图")
    print("=" * 60)
    
    # 将模型设置为评估模式
    model.eval()
    times = []
    
    plot_end = len(X_test_tensor) if PLOT_END is None else PLOT_END
    X_subset = X_test_tensor[PLOT_START:plot_end]
    
    print(f"  正在计算单次预测耗时，共 {len(X_subset)} 个点...")
    
    with torch.no_grad():
        # GPU预热（如果使用 GPU）防止首次推理过慢影响结果
        _ = model(X_subset[0].unsqueeze(0))
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        for i in range(len(X_subset)):
            x = X_subset[i].unsqueeze(0)  # 单个样本，变成 (1, seq_len, features)
            
            # 使用高精度计时器
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start_t = time.perf_counter()
            
            _ = model(x)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            end_t = time.perf_counter()
            
            times.append((end_t - start_t) * 1000)  # 转换为毫秒 (ms)
            
    avg_time = np.mean(times)
    max_time = np.max(times)
    min_time = np.min(times)
    
    print(f"  单次预测耗时 (ms) - 平均: {avg_time:.4f}, 最小: {min_time:.4f}, 最大: {max_time:.4f}")
    
    # 绘制耗时折线图
    fig, ax = plt.subplots(figsize=(12, 6), dpi=150)
    time_steps = np.arange(PLOT_START, plot_end)
    
    ax.plot(
        time_steps, times,
        linestyle="-", color="purple", linewidth=1.5,
        label="单次预测耗时", alpha=0.8
    )
    
    # 绘制平均耗时基准线
    ax.axhline(avg_time, color='red', linestyle='--', linewidth=1.5, label=f'平均耗时: {avg_time:.4f} ms')
    
    ax.set_title("GRU 模型单次预测耗时随时间步变化", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("时间步", fontsize=12, labelpad=10)
    ax.set_ylabel("耗时 (毫秒 / ms)", fontsize=12, labelpad=10)
    
    ax.grid(True, linestyle="--", alpha=0.7)
    ax.legend(loc="upper right", fontsize=11, framealpha=0.9)
    plt.tight_layout()
    
    OUTPUT_TIME_PATH = os.path.join(BASE_DIR, "inference_time_plot.png")
    fig.savefig(OUTPUT_TIME_PATH, dpi=300, bbox_inches="tight", pad_inches=0.1)
    print(f"  ✅ 单次预测耗时图已保存: {OUTPUT_TIME_PATH}")
    print()


# ============================================================================
# 主函数
# ============================================================================
def main():
    print("\n" + "▓" * 60)
    print("  UAV 轨迹预测 — 测试集推理与 3D/2D 可视化")
    print("▓" * 60 + "\n")

    # 步骤 1: 加载数据 + 推理
    Y_pred, Y_test, model, X_test_tensor = load_and_predict()

    # 步骤 2: 反归一化并计算完整维度误差
    Y_pred_real, Y_test_real = denormalize(Y_pred, Y_test)

    # 步骤 3: 绘制 3D 轨迹对比图
    plot_3d_trajectory(Y_pred_real, Y_test_real)

    # 步骤 4: 绘制 2D 综合误差折线图
    plot_2d_error_chart(Y_pred_real, Y_test_real)
    
    # 步骤 5: 绘制单次预测耗时图
    plot_inference_time(model, X_test_tensor)

    print("▓" * 60)
    print("  全部任务完成! 正在弹窗显示所有可视化图表...")
    print("▓" * 60 + "\n")
    
    # 统一在这里弹窗，这样可以同时打开 3D 和 2D 窗口
    plt.show()


if __name__ == "__main__":
    main()
