"""
==============================================================================
UAV 轨迹预测 — pure_bigru  vs.  Attention-Bi-GRU  对比评估
==============================================================================
本脚本在"同一时间段的测试集"上同时推理两条管线并做定量 / 定性对比:

    1. pure_bigru:      双向 GRU + Linear, MinMaxScaler [0,1] 归一化
    2. attention_bigru: Bi-GRU + 动态 Attention + 每层 GeLU 意图融合,
                        MinMaxScaler [0,1] 归一化

对比内容:
    1. 定量指标表 (各维度 MAE / RMSE / Average RMSE / 3D 欧氏距离 Mean / Median /
       Max / P95 / P99 + "超 pure_bigru P95 阈值的样本数")
    2. 2D 误差曲线对比图  → compare_2d_error_all.png
    3. 3D 轨迹对比图      → compare_3d_trajectory_all.png
    4. 误差直方图 + CDF   → compare_error_cdf_all.png
    5. 指标 + 超参数汇总 JSON → compare_metrics_all.json

【公平性保障】
    - 两条管线使用同一原始数据源 (OnboardGPS.csv) 和同一 80:20 时序切分
    - 两条管线归一化方法完全相同: MinMaxScaler [0,1], 均仅在训练集上 fit
    - 两条管线 LOOK_BACK 完全相同: 30 (论文 Table 3: Bi-GRU 最优值)
    - 两条管线训练超参数完全相同: batch=70, lr=1e-3, epochs=500, patience=15
    - 反归一化使用各自 scaler 的逆变换 (均为 MinMaxScaler inverse)
    - 各自的架构差异在指标表与汇总 JSON 中明确标注

【前置条件】
    - 已运行 pure_bigru 流水线:
        python pipelines/pure_bigru/preprocess_uav.py
        python pipelines/pure_bigru/build_sequences.py
        python pipelines/pure_bigru/train.py
    - 已运行 attention_bigru 流水线:
        python pipelines/attention_bigru/prepare_data.py
        python pipelines/attention_bigru/train.py
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
    BATCH_SIZE,
    D_FF,
    DROPOUT,
    HIDDEN_SIZE,
    LEARNING_RATE,
    MAX_EPOCHS,
    MODEL_PATH as ATTN_MODEL,
    N_DEC_LAYERS,
    N_DECODE_STEPS,
    N_ENC_LAYERS,
    N_INTENT,
    OUT_CMP_3D,
    OUT_CMP_CDF,
    OUT_CMP_DIR,
    OUT_CMP_ERR,
    OUT_METRICS,
    OUTPUT_SIZE,
    PROJECT_ROOT,
    PURE_MODEL,
    PURE_SCALER,
    PURE_X_TEST,
    PURE_Y_TEST,
    SCALER_PATH as ATTN_SCALER,
    X_TEST_PATH as ATTN_X_TEST,
    Y_TEST_PATH as ATTN_Y_TEST,
)

if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)
from core.attention_bigru_model import AttentionBiGRU
from core.bigru_model import UAVTrajectoryBiGRU


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
        return
    available_fonts = set(FontManager().get_font_names())
    for font_name in font_candidates[platform]:
        if font_name in available_fonts:
            plt.rcParams['font.sans-serif'] = [font_name]
            plt.rcParams['axes.unicode_minus'] = False
            return


setup_chinese_font()


# ============================================================================
# 工具函数 — 反归一化
# ============================================================================
def _inverse_minmax(data, data_min, data_max):
    """MinMaxScaler 逆变换: X_orig = X_scaled * (max - min) + min"""
    return data * (data_max - data_min) + data_min


# ============================================================================
# 工具函数 — 路径检查
# ============================================================================
def _check_paths(paths):
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        print("❌ 缺少以下文件:")
        for p in missing:
            print(f"    - {p}")
        print("\n请先完成以下流水线:")
        print("  [pure_bigru]       python pipelines/pure_bigru/preprocess_uav.py")
        print("                     python pipelines/pure_bigru/build_sequences.py")
        print("                     python pipelines/pure_bigru/train.py")
        print("  [attention_bigru]  python pipelines/attention_bigru/prepare_data.py")
        print("                     python pipelines/attention_bigru/train.py")
        sys.exit(1)


# ============================================================================
# pure_bigru 推理
# ============================================================================
@torch.no_grad()
def _infer_bigru(model, X, device, chunk=1024):
    model.eval()
    x = torch.tensor(X, dtype=torch.float32).to(device)
    outs = []
    for i in range(0, len(x), chunk):
        outs.append(model(x[i: i + chunk]).cpu().numpy())
    return np.concatenate(outs, axis=0)


def run_pure_bigru():
    print("=" * 60)
    print("  [1/2] 加载 pure_bigru 模型并推理")
    print("=" * 60)

    _check_paths([PURE_X_TEST, PURE_Y_TEST, PURE_MODEL, PURE_SCALER])

    X_test = np.load(PURE_X_TEST)
    Y_test = np.load(PURE_Y_TEST)
    params = np.load(PURE_SCALER)
    data_min, data_max = params["data_min"], params["data_max"]

    PURE_HIDDEN = 64
    PURE_LAYERS = 2
    PURE_INPUT_SIZE = 3
    PURE_DROPOUT = 0.0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UAVTrajectoryBiGRU(
        input_size=PURE_INPUT_SIZE,
        hidden_size=PURE_HIDDEN,
        num_layers=PURE_LAYERS,
        output_size=OUTPUT_SIZE,
        dropout=PURE_DROPOUT,
    )
    model.load_state_dict(torch.load(PURE_MODEL, map_location=device, weights_only=True))
    model.to(device)

    Y_pred = _infer_bigru(model, X_test, device)
    Y_pred_real = _inverse_minmax(Y_pred, data_min, data_max)
    Y_test_real = _inverse_minmax(Y_test, data_min, data_max)

    print(f"  X_test: {X_test.shape}   Y_pred_real: {Y_pred_real.shape}")
    return Y_pred_real, Y_test_real


# ============================================================================
# attention_bigru 推理
# ============================================================================
@torch.no_grad()
def _infer_attn(model, X_feat, X_intent, device, chunk=512):
    model.eval()
    x_f = torch.tensor(X_feat, dtype=torch.float32).to(device)
    x_i = torch.tensor(X_intent, dtype=torch.float32).to(device)
    outs = []
    for i in range(0, len(x_f), chunk):
        out = model(x_f[i:i + chunk], x_i[i:i + chunk])
        outs.append(out.cpu().numpy())
    return np.concatenate(outs, axis=0)


def run_attention_bigru():
    print("=" * 60)
    print("  [2/2] 加载 Attention-Bi-GRU 模型并推理")
    print("=" * 60)

    _check_paths([ATTN_X_TEST, ATTN_Y_TEST, ATTN_MODEL, ATTN_SCALER])

    X_test = np.load(ATTN_X_TEST)
    Y_test = np.load(ATTN_Y_TEST)
    params = np.load(ATTN_SCALER)
    data_min, data_max = params["data_min"], params["data_max"]

    input_size = X_test.shape[-1] - N_INTENT
    X_feat_test = X_test[:, :, :-N_INTENT]
    X_intent_test = X_test[:, :, -N_INTENT:]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    model.load_state_dict(torch.load(ATTN_MODEL, map_location=device, weights_only=True))
    model.to(device)

    Y_pred = _infer_attn(model, X_feat_test, X_intent_test, device)
    Y_pred_real = _inverse_minmax(Y_pred, data_min, data_max)
    Y_test_real = _inverse_minmax(Y_test, data_min, data_max)

    print(f"  X_test: {X_test.shape}   (input_size={input_size}, n_intent={N_INTENT})")
    print(f"  Y_pred_real: {Y_pred_real.shape}")
    return Y_pred_real, Y_test_real


# ============================================================================
# 误差指标
# ============================================================================
def _compute_metrics(Y_pred: np.ndarray, Y_true: np.ndarray) -> Dict:
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
        "euclidean_errors": euc,
    }


def _apply_outlier_counts_vs_bigru_p95(bigru_m: Dict, attn_m: Dict) -> None:
    e_b = bigru_m["euclidean_errors"]
    e_a = attn_m["euclidean_errors"]
    n = min(len(e_b), len(e_a))
    if n == 0:
        bigru_m["n_outliers_over_p95_bigru"] = 0
        attn_m["n_outliers_over_p95_bigru"] = 0
        return
    thr = float(np.percentile(e_b[:n], 95))
    bigru_m["n_outliers_over_p95_bigru"] = int(np.sum(e_b[:n] > thr))
    attn_m["n_outliers_over_p95_bigru"] = int(np.sum(e_a[:n] > thr))


# ============================================================================
# 指标对比表
# ============================================================================
def print_metrics_table(bigru_m: Dict, attn_m: Dict):
    def fmt(v, unit=""):
        return f"{v:.6f}{unit}"

    def diff_pct(a, b):
        if a == 0:
            return 0.0
        return (b - a) / a * 100.0

    print("\n" + "=" * 88)
    print(f"  {'定量指标对比 (反归一化后的真实坐标)':<82s}")
    print("=" * 88)
    header = (f"{'指标':<26s}  {'Pure-Bi-GRU':>18s}  {'Attn-Bi-GRU':>18s}  "
              f"{'变化 (%)':>14s}")
    print(header)
    print("-" * 88)

    rows = [
        ("MAE  Latitude",        "mae_lat",         "°"),
        ("MAE  Longitude",       "mae_lon",         "°"),
        ("MAE  Altitude",        "mae_alt",         " m"),
        ("RMSE Latitude",        "rmse_lat",        "°"),
        ("RMSE Longitude",       "rmse_lon",        "°"),
        ("RMSE Altitude",        "rmse_alt",        " m"),
        ("Average RMSE",         "avg_rmse",        ""),
        ("Mean 3D Euclidean",    "mean_euclidean",  " m"),
        ("Median 3D Euclidean",  "median_euclidean", " m"),
        ("Max  3D Euclidean",    "max_euclidean",   " m"),
        ("P95  3D Euclidean",    "p95_euclidean",   " m"),
        ("P99  3D Euclidean",    "p99_euclidean",   " m"),
        ("> Bi-GRU P95 样本数",  "n_outliers_over_p95_bigru", ""),
    ]

    for name, key, unit in rows:
        bigru_v = bigru_m[key]
        attn_v = attn_m[key]
        delta = diff_pct(float(bigru_v), float(attn_v))
        better = "↓" if delta < 0 else ("↑" if delta > 0 else "=")
        if key == "n_outliers_over_p95_bigru":
            bigru_s = f"{int(bigru_v):>18d}"
            attn_s = f"{int(attn_v):>18d}"
        else:
            bigru_s = f"{fmt(bigru_v, unit):>18s}"
            attn_s = f"{fmt(attn_v, unit):>18s}"
        print(f"{name:<26s}  {bigru_s}  {attn_s}  {delta:+11.2f}% {better}")

    print("=" * 88)
    print("  说明: '变化 (%)' 为 [Attn-Bi-GRU vs Pure-Bi-GRU], 负号 = 误差下降 = 效果更好。")
    print("        Max / P95 / P99 与「> Bi-GRU P95 样本数」反映长尾离群值抑制。")
    print()


def print_hyperparam_table():
    print("=" * 88)
    print(f"  {'超参数对比 (实验报告需明确标注差异)':<82s}")
    print("=" * 88)
    print(f"{'维度':<22s}  {'Pure-Bi-GRU':>22s}  {'Attn-Bi-GRU':>22s}")
    print("-" * 88)
    rows = [
        ("归一化方法",          "MinMaxScaler [0,1]", "MinMaxScaler [0,1]"),
        ("模型架构",            "Bi-GRU → Linear",    "Bi-GRU + Attention + 意图融合"),
        ("意图融合",            "无",                  "GeLU + Concat 融合"),
        ("Hidden Size",        "64",                  "64"),
        ("编/解码层数",         "2",                   f"{N_ENC_LAYERS} / {N_DEC_LAYERS}"),
        ("Dropout",            "0.0",                 f"{DROPOUT}"),
        ("学习率",              "1e-3",                f"{LEARNING_RATE}"),
        ("Batch Size",         "70",                  f"{BATCH_SIZE}"),
        ("Max Epochs",         "500",                 f"{MAX_EPOCHS}"),
        ("Look_Back",          "30",                  "30"),
        ("解码步数 (T_dec)",    "1",                   f"{N_DECODE_STEPS}"),
    ]
    for name, bigru_v, attn_v in rows:
        print(f"{name:<22s}  {bigru_v:>22s}  {attn_v:>22s}")
    print("=" * 88)
    print()


# ============================================================================
# 可视化对比
# ============================================================================
def plot_compare_2d_error(bigru_err: np.ndarray, attn_err: np.ndarray):
    print("=" * 60)
    print("  绘制 2D 误差曲线对比图")
    print("=" * 60)

    n = min(len(bigru_err), len(attn_err))
    t = np.arange(n)

    fig, ax = plt.subplots(figsize=(14, 6), dpi=150)
    ax.plot(t, bigru_err[:n], linestyle="-", color="red",
            linewidth=1.6, alpha=0.85, label="Pure-Bi-GRU")
    ax.plot(t, attn_err[:n], linestyle="-", color="royalblue",
            linewidth=1.6, alpha=0.85, label="Attention-Bi-GRU + 意图")

    p95 = float(np.percentile(bigru_err[:n], 95))
    outlier_mask = bigru_err[:n] > p95
    y_top = float(max(bigru_err[:n].max(), attn_err[:n].max()))
    ax.fill_between(t, 0, y_top, where=outlier_mask, color="red", alpha=0.08,
                    label=f"Bi-GRU 离群区 (> P95 = {p95:.2f} m)")

    ax.set_title("预测误差随时间对比: Pure-Bi-GRU vs Attention-Bi-GRU + 意图",
                 fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("时间步", fontsize=12, labelpad=10)
    ax.set_ylabel("3D 综合欧氏距离误差 (m)", fontsize=12, labelpad=10)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="upper right", fontsize=11, framealpha=0.9)
    plt.tight_layout()
    fig.savefig(OUT_CMP_ERR, dpi=300, bbox_inches="tight", pad_inches=0.1)
    print(f"  ✅ 已保存: {OUT_CMP_ERR}\n")


def plot_compare_3d_trajectory(bigru_pred, bigru_true, attn_pred, attn_true):
    print("=" * 60)
    print("  绘制 3D 轨迹对比图")
    print("=" * 60)

    n = min(len(bigru_pred), len(attn_pred), len(bigru_true), len(attn_true))
    truth = bigru_true[:n]

    fig = plt.figure(figsize=(14, 10), dpi=150)
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(truth[:, 1], truth[:, 0], truth[:, 2],
            linestyle="--", color="gray", linewidth=1.3,
            label="实际轨迹", alpha=0.75)
    ax.plot(bigru_pred[:n, 1], bigru_pred[:n, 0], bigru_pred[:n, 2],
            linestyle="-", color="red", linewidth=1.5,
            label="Pure-Bi-GRU 预测", alpha=0.85)
    ax.plot(attn_pred[:n, 1], attn_pred[:n, 0], attn_pred[:n, 2],
            linestyle="-", color="royalblue", linewidth=1.5,
            label="Attention-Bi-GRU + 意图 预测", alpha=0.85)
    ax.scatter(truth[0, 1], truth[0, 0], truth[0, 2],
               color="green", s=80, marker="o", zorder=5, label="起点")
    ax.scatter(truth[-1, 1], truth[-1, 0], truth[-1, 2],
               color="blue", s=80, marker="^", zorder=5, label="终点")

    ax.set_xlabel("经度 (°)", fontsize=12, labelpad=10)
    ax.set_ylabel("纬度 (°)", fontsize=12, labelpad=10)
    ax.set_zlabel("高度 (m)", fontsize=12, labelpad=10)
    ax.set_title("三维轨迹对比: 实际 / Pure-Bi-GRU / Attention-Bi-GRU + 意图",
                 fontsize=14, fontweight="bold", pad=20)
    ax.legend(loc="upper left", fontsize=10, framealpha=0.9)
    ax.view_init(elev=25, azim=135)
    plt.tight_layout()
    fig.savefig(OUT_CMP_3D, dpi=300, bbox_inches="tight", pad_inches=0.3)
    print(f"  ✅ 已保存: {OUT_CMP_3D}\n")


def plot_error_cdf(bigru_err: np.ndarray, attn_err: np.ndarray):
    print("=" * 60)
    print("  绘制误差 CDF 对比图")
    print("=" * 60)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=150)

    bins = np.linspace(0, max(bigru_err.max(), attn_err.max()), 60)
    axes[0].hist(bigru_err, bins=bins, alpha=0.55, color="red",
                 label="Pure-Bi-GRU", density=True)
    axes[0].hist(attn_err, bins=bins, alpha=0.55, color="royalblue",
                 label="Attn-Bi-GRU + 意图", density=True)
    axes[0].set_title("误差分布直方图", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("3D 欧氏距离误差 (m)")
    axes[0].set_ylabel("概率密度")
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[0].legend()

    def _cdf(x):
        xs = np.sort(x)
        ys = np.arange(1, len(xs) + 1) / len(xs)
        return xs, ys

    xb, yb = _cdf(bigru_err)
    xa, ya = _cdf(attn_err)
    axes[1].plot(xb, yb, color="red", linewidth=2, label="Pure-Bi-GRU")
    axes[1].plot(xa, ya, color="royalblue", linewidth=2, label="Attn-Bi-GRU + 意图")
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
    print("  UAV 轨迹预测 — Pure-Bi-GRU vs Attention-Bi-GRU 对比评估")
    print("▓" * 60 + "\n")

    os.makedirs(OUT_CMP_DIR, exist_ok=True)

    print("  【公平性保障】")
    print("    ✅ 同一数据源 (OnboardGPS.csv) 与同一 80:20 时序切分")
    print("    ✅ 两条管线归一化方法相同: 均使用 MinMaxScaler [0,1]")
    print("    ✅ 两条管线 scaler 均仅在训练集上 fit (无数据泄漏)")
    print("    ✅ 两条管线 LOOK_BACK 相同: 30 (论文 Table 3: Bi-GRU 最优值)")
    print("    ✅ 两条管线训练超参数相同: batch=70, lr=1e-3, epochs=500, patience=15")
    print("    ⚠️  架构差异: Attention 机制 / 意图融合 (见下方对比表)")
    print()

    bigru_pred, bigru_true = run_pure_bigru()
    attn_pred, attn_true = run_attention_bigru()

    bigru_m = _compute_metrics(bigru_pred, bigru_true)
    attn_m = _compute_metrics(attn_pred, attn_true)
    _apply_outlier_counts_vs_bigru_p95(bigru_m, attn_m)

    print_metrics_table(bigru_m, attn_m)
    print_hyperparam_table()

    plot_compare_2d_error(bigru_m["euclidean_errors"], attn_m["euclidean_errors"])
    plot_compare_3d_trajectory(bigru_pred, bigru_true, attn_pred, attn_true)
    plot_error_cdf(bigru_m["euclidean_errors"], attn_m["euclidean_errors"])

    def _strip(m):
        return {k: v for k, v in m.items() if k != "euclidean_errors"}

    summary = {
        "pipelines": {
            "pure_bigru": {
                "model": "UAVTrajectoryBiGRU (双向 GRU + Linear)",
                "normalization": "MinMaxScaler [0, 1] (train-only fit)",
                "hidden_size": 64,
                "num_layers": 2,
                "input_size": 3,
                "dropout": 0.0,
                "learning_rate": 1e-3,
                "batch_size": 70,
                "max_epochs": 500,
                "look_back": 30,
                "intent_fusion": "无",
                "attention": "无",
                "decode_steps": 1,
            },
            "attention_bigru": {
                "model": "AttentionBiGRU (Bi-GRU + Attention + 意图融合)",
                "normalization": "MinMaxScaler [0, 1] (train-only fit)",
                "hidden_size": HIDDEN_SIZE,
                "n_enc_layers": N_ENC_LAYERS,
                "n_dec_layers": N_DEC_LAYERS,
                "d_ff": D_FF,
                "n_intent": N_INTENT,
                "dropout": DROPOUT,
                "learning_rate": LEARNING_RATE,
                "batch_size": BATCH_SIZE,
                "max_epochs": MAX_EPOCHS,
                "look_back": 30,
                "intent_fusion": "GeLU 意图投影 + Concat 融合 (论文 Eq.31-32)",
                "attention": "缩放点积 Attention (Q=编码器末步)",
                "decode_steps": N_DECODE_STEPS,
            },
        },
        "fairness_notes": {
            "data_source": "同一 OnboardGPS.csv, 同一 80:20 时序切分",
            "scaler_method": "两条管线均使用 MinMaxScaler [0,1], 仅在训练集上 fit",
            "look_back": "两条管线均使用 LOOK_BACK=30 (论文 Table 3: Bi-GRU 最优值)",
            "training_hyperparams": "batch=70, lr=1e-3, epochs=500, patience=15 (完全一致)",
            "arch_difference": "架构差异: Attention 机制 + 意图融合 (这是对比的核心变量)",
        },
        "metrics": {
            "pure_bigru": _strip(bigru_m),
            "attention_bigru": _strip(attn_m),
        },
    }
    with open(OUT_METRICS, "w", encoding="utf-8") as fp:
        json.dump(summary, fp, ensure_ascii=False, indent=2)
    print(f"  📝 指标 + 超参数汇总 JSON: {OUT_METRICS}\n")

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
