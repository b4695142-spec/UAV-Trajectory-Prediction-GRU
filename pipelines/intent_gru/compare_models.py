"""
==============================================================================
UAV 轨迹预测 — 纯 GRU  vs.  GRU + 意图识别  对比评估 (公平性修正版)
==============================================================================
本脚本在"同一测试集"上同时推理两个模型并做定量 / 定性对比，用于验证
机动意图识别模块对长预测视距下离群值的抑制效果。

对比内容:
    1. 定量指标表 (MAE、RMSE、Average RMSE、Mean / Max / P95 / P99 欧氏距离)
       — 特别关注 Max 与 P95/P99，这是"离群值"的直接体现。
    2. 2D 误差曲线对比 (红: 纯 GRU, 橙: GRU+意图) — 可直观比较误差波动。
    3. 3D 轨迹对比 (灰虚线: 真值, 红: 纯 GRU, 橙: GRU+意图)。
    4. 误差直方图 + 累计分布函数 (CDF) 对比 — 量化离群值尾部分布。

【公平性修正】
    - 两条管线均使用"仅在训练集上 fit"的 MinMaxScaler，消除数据泄漏
    - 意图版 dropout=0.0，与纯 GRU 一致，排除正则化差异
    - 意图版 SVM 概率使用 OOF 生成，消除训练/测试分布偏移
    - 反归一化时使用各自的 scaler 参数 (已对齐)

【前置条件】
    - 必须已运行纯 GRU 管线:
        python preprocess_uav.py
        python build_sequences.py
        python train.py
    - 以及意图增广管线:
        python prepare_intent.py
        python train_with_intent.py
==============================================================================
"""

from __future__ import annotations

import json
import os
import sys
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.font_manager import FontManager

from config import (
    HIDDEN_SIZE,
    INTENT_DROPOUT,
    INTENT_MODEL,
    INTENT_SCALER,
    INTENT_X_TEST,
    INTENT_Y_TEST,
    NUM_LAYERS,
    OUT_CMP_3D,
    OUT_CMP_CDF,
    OUT_CMP_ERR,
    OUT_METRICS,
    OUTPUT_SIZE,
    PURE_INPUT_SIZE,
    PURE_MODEL,
    PURE_SCALER,
    PURE_X_TEST,
    PURE_Y_TEST,
    PROJECT_ROOT,
)
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)
from core.gru_model import UAVTrajectoryGRU


# ============================================================================
# 中文字体
# ============================================================================
def setup_chinese_font():
    font_candidates = {
        'win32': ['SimHei', 'Microsoft YaHei', 'SimSun'],
        'linux': ['WenQuanYi Micro Hei', 'WenQuanYi Zen Hei', 'Noto Sans CJK SC', 'DejaVu Sans'],
        'darwin': ['PingFang SC', 'Heiti TC', 'STHeiti', 'Arial Unicode MS'],
    }
    platform = sys.platform
    if platform not in font_candidates:
        return
    available_fonts = set(FontManager().get_font_names())
    for font_name in font_candidates[platform]:
        if font_name in available_fonts:
            plt.rcParams['font.sans-serif'] = [font_name]
            plt.rcParams['axes.unicode_minus'] = False
            return


setup_chinese_font()


# ============================================================================
# 工具函数
# ============================================================================
def _inverse_mm(data: np.ndarray, data_min: np.ndarray, data_max: np.ndarray) -> np.ndarray:
    return data * (data_max - data_min) + data_min


def _check_paths(paths):
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        print("❌ 缺少以下文件:")
        for p in missing:
            print(f"    - {p}")
        print("\n请先完成以下流水线:")
        print("  [纯 GRU]       python preprocess_uav.py && python build_sequences.py && python train.py")
        print("  [意图增广版]   python prepare_intent.py && python train_with_intent.py")
        sys.exit(1)


@torch.no_grad()
def _infer(model: UAVTrajectoryGRU, X: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    x = torch.tensor(X, dtype=torch.float32).to(device)
    # 分块推理以避免单次张量过大
    out_list = []
    chunk = 1024
    for i in range(0, len(x), chunk):
        out_list.append(model(x[i: i + chunk]).cpu().numpy())
    return np.concatenate(out_list, axis=0)


def _compute_metrics(Y_pred: np.ndarray, Y_true: np.ndarray) -> Dict:
    """计算各维度与综合误差指标 (全部在反归一化后的真实坐标下)。"""
    mae = np.mean(np.abs(Y_pred - Y_true), axis=0)
    rmse = np.sqrt(np.mean((Y_pred - Y_true) ** 2, axis=0))
    euc = np.sqrt(np.sum((Y_pred - Y_true) ** 2, axis=1))
    return {
        "mae_lat": float(mae[0]),
        "mae_lon": float(mae[1]),
        "mae_alt": float(mae[2]),
        "rmse_lat": float(rmse[0]),
        "rmse_lon": float(rmse[1]),
        "rmse_alt": float(rmse[2]),
        "avg_rmse": float(np.mean(rmse)),
        "mean_euclidean": float(np.mean(euc)),
        "median_euclidean": float(np.median(euc)),
        "max_euclidean": float(np.max(euc)),
        "p95_euclidean": float(np.percentile(euc, 95)),
        "p99_euclidean": float(np.percentile(euc, 99)),
        "euclidean_errors": euc,              # 用于绘图
    }


def _apply_outlier_counts_vs_pure_p95(pure_m: Dict, intent_m: Dict) -> None:
    """
    以「纯 GRU」在对齐样本上的 P95 三维欧氏误差为阈值：
    - pure_m：统计纯 GRU 自身超过该阈值的样本数（约尾部 5%，因并列值可能略有偏差）
    - intent_m：统计意图版在**同一阈值**下超过的样本数（用于对比尾部是否被压低）
    """
    e_p = pure_m["euclidean_errors"]
    e_i = intent_m["euclidean_errors"]
    n = min(len(e_p), len(e_i))
    if n == 0:
        pure_m["n_outliers_over_p95_pure"] = 0
        intent_m["n_outliers_over_p95_pure"] = 0
        return
    thr = float(np.percentile(e_p[:n], 95))
    pure_m["n_outliers_over_p95_pure"] = int(np.sum(e_p[:n] > thr))
    intent_m["n_outliers_over_p95_pure"] = int(np.sum(e_i[:n] > thr))


# ============================================================================
# 1) 加载两个模型 + 在各自测试集上推理 → 反归一化
# ============================================================================
def run_pure_gru():
    print("=" * 60)
    print("  [1/2] 加载纯 GRU 模型并推理")
    print("=" * 60)

    _check_paths([PURE_X_TEST, PURE_Y_TEST, PURE_MODEL, PURE_SCALER])

    X_test = np.load(PURE_X_TEST)
    Y_test = np.load(PURE_Y_TEST)
    params = np.load(PURE_SCALER)
    data_min, data_max = params["data_min"], params["data_max"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UAVTrajectoryGRU(
        input_size=PURE_INPUT_SIZE,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        output_size=OUTPUT_SIZE,
    )
    model.load_state_dict(torch.load(PURE_MODEL, map_location=device, weights_only=True))
    model.to(device)

    Y_pred = _infer(model, X_test, device)
    Y_pred_real = _inverse_mm(Y_pred, data_min, data_max)
    Y_test_real = _inverse_mm(Y_test, data_min, data_max)

    print(f"  X_test: {X_test.shape}   Y_pred_real: {Y_pred_real.shape}")
    return Y_pred_real, Y_test_real


def run_intent_gru():
    print("=" * 60)
    print("  [2/2] 加载意图增广版 GRU 模型并推理")
    print("=" * 60)

    _check_paths([INTENT_X_TEST, INTENT_Y_TEST, INTENT_MODEL, INTENT_SCALER])

    X_test = np.load(INTENT_X_TEST)
    Y_test = np.load(INTENT_Y_TEST)
    params = np.load(INTENT_SCALER)
    data_min, data_max = params["data_min"], params["data_max"]

    input_size = int(X_test.shape[-1])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UAVTrajectoryGRU(
        input_size=input_size,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        output_size=OUTPUT_SIZE,
        dropout=INTENT_DROPOUT,
    )
    model.load_state_dict(torch.load(INTENT_MODEL, map_location=device, weights_only=True))
    model.to(device)

    Y_pred = _infer(model, X_test, device)
    Y_pred_real = _inverse_mm(Y_pred, data_min, data_max)
    Y_test_real = _inverse_mm(Y_test, data_min, data_max)

    print(f"  X_test: {X_test.shape} (input_size={input_size})   "
          f"Y_pred_real: {Y_pred_real.shape}")
    return Y_pred_real, Y_test_real


# ============================================================================
# 2) 指标对比表
# ============================================================================
def print_metrics_table(pure_m: Dict, intent_m: Dict):
    def fmt(v, unit=""):
        return f"{v:.6f}{unit}"

    def diff_pct(a, b):
        """返回 'intent 相对 pure' 的变化百分比 (负值=降低=更好)。"""
        if a == 0:
            return 0.0
        return (b - a) / a * 100.0

    print("\n" + "=" * 78)
    print(f"  {'定量指标对比 (反归一化后的真实坐标)':<72s}")
    print("=" * 78)
    header = f"{'指标':<24s}  {'纯 GRU':>16s}  {'GRU+意图':>16s}  {'变化 (%)':>14s}"
    print(header)
    print("-" * 78)

    rows = [
        ("MAE  Latitude",        "mae_lat",         "°"),
        ("MAE  Longitude",       "mae_lon",         "°"),
        ("MAE  Altitude",        "mae_alt",         " m"),
        ("RMSE Latitude",        "rmse_lat",        "°"),
        ("RMSE Longitude",       "rmse_lon",        "°"),
        ("RMSE Altitude",        "rmse_alt",        " m"),
        ("Average RMSE",         "avg_rmse",        ""),
        ("Mean 3D Euclidean",    "mean_euclidean",  " m"),
        ("Median 3D Euclidean",  "median_euclidean"," m"),
        ("Max  3D Euclidean",    "max_euclidean",   " m"),  # ← 离群值
        ("P95  3D Euclidean",    "p95_euclidean",   " m"),
        ("P99  3D Euclidean",    "p99_euclidean",   " m"),
        ("> 纯 GRU P95 样本数",  "n_outliers_over_p95_pure", ""),
    ]

    for name, key, unit in rows:
        pure_v = pure_m[key]
        int_v = intent_m[key]
        delta = diff_pct(float(pure_v), float(int_v))
        better = "↓" if delta < 0 else ("↑" if delta > 0 else "=")
        if key == "n_outliers_over_p95_pure":
            pure_s = f"{int(pure_v):>16d}"
            int_s = f"{int(int_v):>16d}"
        else:
            pure_s = f"{fmt(pure_v, unit):>16s}"
            int_s = f"{fmt(int_v, unit):>16s}"
        print(f"{name:<24s}  {pure_s}  {int_s}  {delta:+11.2f}% {better}")

    print("=" * 78)
    print("  说明: '变化 (%)' 为 [意图版 vs 纯 GRU]，负号 = 误差下降 = 效果更好。")
    print("        Max / P95 / P99 欧氏距离与「> 纯 GRU P95 样本数」反映离群尾部抑制效果。")
    print("  【公平性修正】两条管线均使用 train-only scaler, dropout=0.0, SVM OOF 概率")
    print()


# ============================================================================
# 3) 绘图: 2D 误差曲线 / 3D 轨迹 / CDF
# ============================================================================
def plot_compare_2d_error(pure_err: np.ndarray, intent_err: np.ndarray):
    print("=" * 60)
    print("  绘制 2D 误差曲线对比图")
    print("=" * 60)

    # 两个测试集长度可能因 look_back 配置相同而一致; 若不一致对齐到较短者
    n = min(len(pure_err), len(intent_err))
    t = np.arange(n)

    fig, ax = plt.subplots(figsize=(14, 6), dpi=150)
    ax.plot(t, pure_err[:n], linestyle="-", color="red",
            linewidth=1.6, alpha=0.8, label="纯 GRU")
    ax.plot(t, intent_err[:n], linestyle="-", color="darkorange",
            linewidth=1.6, alpha=0.8, label="GRU + 意图识别")

    # 高亮"离群值区间" (误差超过纯 GRU P95)
    p95 = float(np.percentile(pure_err[:n], 95))
    outlier_mask = pure_err[:n] > p95
    ax.fill_between(t, 0, np.max([pure_err[:n].max(), intent_err[:n].max()]),
                    where=outlier_mask, color="red", alpha=0.08,
                    label=f"纯 GRU 离群区 (> P95 = {p95:.2f} m)")

    ax.set_title("预测误差随时间对比：纯 GRU vs GRU+意图",
                 fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("时间步", fontsize=12, labelpad=10)
    ax.set_ylabel("3D 综合欧氏距离误差 (m)", fontsize=12, labelpad=10)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="upper right", fontsize=11, framealpha=0.9)

    plt.tight_layout()
    fig.savefig(OUT_CMP_ERR, dpi=300, bbox_inches="tight", pad_inches=0.1)
    print(f"  ✅ 已保存: {OUT_CMP_ERR}\n")


def plot_compare_3d_trajectory(pure_pred, pure_true, intent_pred, intent_true):
    print("=" * 60)
    print("  绘制 3D 轨迹对比图")
    print("=" * 60)

    n = min(len(pure_pred), len(intent_pred), len(pure_true), len(intent_true))

    fig = plt.figure(figsize=(14, 10), dpi=150)
    ax = fig.add_subplot(111, projection="3d")

    # 真值 (统一使用纯 GRU 的 Y_test，因为两者时间对齐)
    truth = pure_true[:n]
    ax.plot(truth[:, 1], truth[:, 0], truth[:, 2],
            linestyle="--", color="gray", linewidth=1.3,
            label="实际轨迹", alpha=0.75)

    ax.plot(pure_pred[:n, 1], pure_pred[:n, 0], pure_pred[:n, 2],
            linestyle="-", color="red", linewidth=1.5,
            label="纯 GRU 预测", alpha=0.85)
    ax.plot(intent_pred[:n, 1], intent_pred[:n, 0], intent_pred[:n, 2],
            linestyle="-", color="darkorange", linewidth=1.5,
            label="GRU+意图 预测", alpha=0.85)

    ax.scatter(truth[0, 1], truth[0, 0], truth[0, 2],
               color="green", s=80, marker="o", zorder=5, label="起点")
    ax.scatter(truth[-1, 1], truth[-1, 0], truth[-1, 2],
               color="blue", s=80, marker="^", zorder=5, label="终点")

    ax.set_xlabel("经度 (°)", fontsize=12, labelpad=10)
    ax.set_ylabel("纬度 (°)", fontsize=12, labelpad=10)
    ax.set_zlabel("高度 (m)", fontsize=12, labelpad=10)
    ax.set_title("三维轨迹对比：实际 / 纯 GRU / GRU+意图",
                 fontsize=14, fontweight="bold", pad=20)
    ax.legend(loc="upper left", fontsize=10, framealpha=0.9)
    ax.view_init(elev=25, azim=135)
    plt.tight_layout()
    fig.savefig(OUT_CMP_3D, dpi=300, bbox_inches="tight", pad_inches=0.3)
    print(f"  ✅ 已保存: {OUT_CMP_3D}\n")


def plot_error_cdf(pure_err: np.ndarray, intent_err: np.ndarray):
    print("=" * 60)
    print("  绘制误差 CDF 对比图 (量化离群值尾部)")
    print("=" * 60)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=150)

    # 左: 直方图
    bins = np.linspace(0, max(pure_err.max(), intent_err.max()), 60)
    axes[0].hist(pure_err, bins=bins, alpha=0.55, color="red",
                 label="纯 GRU", density=True)
    axes[0].hist(intent_err, bins=bins, alpha=0.55, color="darkorange",
                 label="GRU+意图", density=True)
    axes[0].set_title("误差分布直方图", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("3D 欧氏距离误差 (m)")
    axes[0].set_ylabel("概率密度")
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[0].legend()

    # 右: CDF
    def _cdf(x):
        xs = np.sort(x)
        ys = np.arange(1, len(xs) + 1) / len(xs)
        return xs, ys

    xp, yp = _cdf(pure_err)
    xi, yi = _cdf(intent_err)
    axes[1].plot(xp, yp, color="red", linewidth=2, label="纯 GRU")
    axes[1].plot(xi, yi, color="darkorange", linewidth=2, label="GRU+意图")
    axes[1].axhline(0.95, color="gray", linestyle=":", linewidth=1, label="P95 基线")
    axes[1].set_title("误差累积分布函数 (CDF)", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("3D 欧氏距离误差 (m)")
    axes[1].set_ylabel("累积概率")
    axes[1].grid(True, linestyle="--", alpha=0.5)
    axes[1].legend()

    plt.tight_layout()
    fig.savefig(OUT_CMP_CDF, dpi=300, bbox_inches="tight", pad_inches=0.15)
    print(f"  ✅ 已保存: {OUT_CMP_CDF}\n")


# ============================================================================
# 主函数
# ============================================================================
def main():
    print("\n" + "▓" * 60)
    print("  UAV 轨迹预测 — 纯 GRU vs GRU+意图 对比评估 (公平性修正版)")
    print("▓" * 60 + "\n")

    print("  【公平性修正确认】")
    print("    ✅ 两条管线 MinMaxScaler 均仅在训练集上 fit")
    print("    ✅ 两条管线 dropout = 0.0")
    print("    ✅ 意图版 SVM 训练集概率使用 5-Fold OOF 生成")
    print()

    pure_pred, pure_true = run_pure_gru()
    intent_pred, intent_true = run_intent_gru()

    pure_m = _compute_metrics(pure_pred, pure_true)
    intent_m = _compute_metrics(intent_pred, intent_true)
    _apply_outlier_counts_vs_pure_p95(pure_m, intent_m)

    print_metrics_table(pure_m, intent_m)

    plot_compare_2d_error(pure_m["euclidean_errors"], intent_m["euclidean_errors"])
    plot_compare_3d_trajectory(pure_pred, pure_true, intent_pred, intent_true)
    plot_error_cdf(pure_m["euclidean_errors"], intent_m["euclidean_errors"])

    def _strip(m):
        return {k: v for k, v in m.items() if k != "euclidean_errors"}

    summary = {
        "fairness_corrections": {
            "scaler_fit_on": "train_only",
            "dropout": 0.0,
            "svm_train_proba": "5-fold_OOF",
        },
        "pure_gru": _strip(pure_m),
        "intent_gru": _strip(intent_m),
    }
    with open(OUT_METRICS, "w", encoding="utf-8") as fp:
        json.dump(summary, fp, ensure_ascii=False, indent=2)
    print(f"  📝 指标汇总 JSON: {OUT_METRICS}\n")

    print("▓" * 60)
    print("  对比评估完成! 生成的对比图:")
    print(f"    - {OUT_CMP_ERR}")
    print(f"    - {OUT_CMP_3D}")
    print(f"    - {OUT_CMP_CDF}")
    print(f"    - {OUT_METRICS}")
    print("▓" * 60 + "\n")

    plt.show()


if __name__ == "__main__":
    main()
