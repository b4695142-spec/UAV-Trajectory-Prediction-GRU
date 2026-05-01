"""
==============================================================================
UAV 轨迹预测 — pure_gru  vs.  Attention-Bi-GRU  对比评估
==============================================================================
本脚本在"同一时间段的测试集"上同时推理两条管线并做定量 / 定性对比:

    1. pure_gru:        单向 GRU + Linear, MinMaxScaler 归一化
    2. attention_bigru: Bi-GRU + 动态 Attention + 每层 GeLU 意图融合,
                        StandardScaler (论文 Equation 8) 归一化

对比内容:
    1. 定量指标表 (各维度 MAE / RMSE / Average RMSE / 3D 欧氏距离 Mean / Median /
       Max / P95 / P99 + "超 pure_gru P95 阈值的样本数")
    2. 2D 误差曲线对比图  → compare_2d_error_all.png
    3. 3D 轨迹对比图      → compare_3d_trajectory_all.png
    4. 误差直方图 + CDF   → compare_error_cdf_all.png
    5. 指标 + 超参数汇总 JSON → compare_metrics_all.json

【公平性保障】
    - 两条管线使用同一原始数据源 (OnboardGPS.csv) 和同一 80:20 时序切分
    - 两条管线归一化均仅在训练集上 fit (无数据泄漏)
    - 反归一化使用各自 scaler 的逆变换 (MinMax 对 pure_gru, Standard 对 attn_bigru)
    - 各自的超参数差异在指标表与汇总 JSON 中明确标注

【前置条件】
    - 已运行 pure_gru 流水线:
        python pipelines/pure_gru/preprocess_uav.py
        python pipelines/pure_gru/build_sequences.py
        python pipelines/pure_gru/train.py
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
from core.gru_model import UAVTrajectoryGRU


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


def _inverse_standard(data, mean, scale):
    """StandardScaler 逆变换: X_orig = X_scaled * σ + μ  (论文 Equation 8 反演)"""
    return data * scale + mean


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
        print("  [pure_gru]         python pipelines/pure_gru/preprocess_uav.py")
        print("                     python pipelines/pure_gru/build_sequences.py")
        print("                     python pipelines/pure_gru/train.py")
        print("  [attention_bigru]  python pipelines/attention_bigru/prepare_data.py")
        print("                     python pipelines/attention_bigru/train.py")
        sys.exit(1)


# ============================================================================
# pure_gru 推理
# ============================================================================
@torch.no_grad()
def _infer_gru(model, X, device, chunk=1024):
    model.eval()
    x = torch.tensor(X, dtype=torch.float32).to(device)
    outs = []
    for i in range(0, len(x), chunk):
        outs.append(model(x[i: i + chunk]).cpu().numpy())
    return np.concatenate(outs, axis=0)


def run_pure_gru():
    print("=" * 60)
    print("  [1/2] 加载 pure_gru 模型并推理")
    print("=" * 60)

    _check_paths([PURE_X_TEST, PURE_Y_TEST, PURE_MODEL, PURE_SCALER])

    X_test = np.load(PURE_X_TEST)
    Y_test = np.load(PURE_Y_TEST)
    params = np.load(PURE_SCALER)
    data_min, data_max = params["data_min"], params["data_max"]

    # ── pure_gru 超参数 (与 pipelines/pure_gru/config.py 保持一致) ──
    PURE_HIDDEN = 64
    PURE_LAYERS = 2
    PURE_INPUT_SIZE = 3
    PURE_DROPOUT = 0.0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UAVTrajectoryGRU(
        input_size=PURE_INPUT_SIZE,
        hidden_size=PURE_HIDDEN,
        num_layers=PURE_LAYERS,
        output_size=OUTPUT_SIZE,
        dropout=PURE_DROPOUT,
    )
    model.load_state_dict(torch.load(PURE_MODEL, map_location=device, weights_only=True))
    model.to(device)

    Y_pred = _infer_gru(model, X_test, device)
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
    mean, scale = params["mean"], params["scale"]

    # 拆分增广 X: 最后 N_INTENT 列为 SVM 概率, 其余为结构性特征
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
    # Y_pred shape:
    #   (n, 3)                    当 N_DECODE_STEPS == 1
    #   (n, N_DECODE_STEPS, 3)    当 N_DECODE_STEPS  > 1
    # Y_test 同上 (由 prepare_data.py 按相同 N_DECODE_STEPS 生成)
    if N_DECODE_STEPS > 1:
        if Y_pred.ndim == 3:
            Y_pred = Y_pred[:, 0, :]
        if Y_test.ndim == 3:
            Y_test = Y_test[:, 0, :]
        print(f"  ℹ️  T_dec={N_DECODE_STEPS}: 仅取首步与 pure_gru 对齐 "
              f"(两者预测目标都是 target[t], 任务定义一致)")

    Y_pred_real = _inverse_standard(Y_pred, mean, scale)
    Y_test_real = _inverse_standard(Y_test, mean, scale)

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


def _apply_outlier_counts_vs_pure_p95(pure_m: Dict, attn_m: Dict) -> None:
    """以 pure_gru 的 P95 阈值统计两条管线超过该阈值的样本数。"""
    e_p = pure_m["euclidean_errors"]
    e_a = attn_m["euclidean_errors"]
    n = min(len(e_p), len(e_a))
    if n == 0:
        pure_m["n_outliers_over_p95_pure"] = 0
        attn_m["n_outliers_over_p95_pure"] = 0
        return
    thr = float(np.percentile(e_p[:n], 95))
    pure_m["n_outliers_over_p95_pure"] = int(np.sum(e_p[:n] > thr))
    attn_m["n_outliers_over_p95_pure"] = int(np.sum(e_a[:n] > thr))


# ============================================================================
# 指标对比表
# ============================================================================
def print_metrics_table(pure_m: Dict, attn_m: Dict):
    def fmt(v, unit=""):
        return f"{v:.6f}{unit}"

    def diff_pct(a, b):
        if a == 0:
            return 0.0
        return (b - a) / a * 100.0

    print("\n" + "=" * 88)
    print(f"  {'定量指标对比 (反归一化后的真实坐标)':<82s}")
    print("=" * 88)
    header = (f"{'指标':<26s}  {'pure_gru':>18s}  {'Attn-Bi-GRU':>18s}  "
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
        ("> pure_gru P95 样本数", "n_outliers_over_p95_pure", ""),
    ]

    for name, key, unit in rows:
        pure_v = pure_m[key]
        attn_v = attn_m[key]
        delta = diff_pct(float(pure_v), float(attn_v))
        better = "↓" if delta < 0 else ("↑" if delta > 0 else "=")
        if key == "n_outliers_over_p95_pure":
            pure_s = f"{int(pure_v):>18d}"
            attn_s = f"{int(attn_v):>18d}"
        else:
            pure_s = f"{fmt(pure_v, unit):>18s}"
            attn_s = f"{fmt(attn_v, unit):>18s}"
        print(f"{name:<26s}  {pure_s}  {attn_s}  {delta:+11.2f}% {better}")

    print("=" * 88)
    print("  说明: '变化 (%)' 为 [Attn-Bi-GRU vs pure_gru], 负号 = 误差下降 = 效果更好。")
    print("        Max / P95 / P99 与「> pure_gru P95 样本数」反映长视距离群尾部抑制。")
    print()


def print_hyperparam_table():
    """打印两条管线的超参数差异 (用于实验报告标注)。"""
    print("=" * 88)
    print(f"  {'超参数对比 (实验报告需明确标注差异)':<82s}")
    print("=" * 88)
    print(f"{'维度':<22s}  {'pure_gru':>22s}  {'Attn-Bi-GRU':>22s}")
    print("-" * 88)
    rows = [
        ("归一化方法",          "MinMaxScaler [0,1]", "StandardScaler (论文 Eq.8)"),
        ("模型架构",            "单向 GRU → Linear", "Bi-GRU 编/解码器 + 动态 Attn"),
        ("意图融合",            "无", "每层每步 GeLU + Concat 融合"),
        ("Hidden Size",        "64",                "64"),
        ("编/解码层数",         "2",                 f"{N_ENC_LAYERS} / {N_DEC_LAYERS}"),
        ("Dropout",            "0.0",               f"{DROPOUT}"),
        ("学习率",              "1e-3",              f"{LEARNING_RATE}"),
        ("Batch Size",         "70",                f"{BATCH_SIZE}"),
        ("Max Epochs",         "500",               f"{MAX_EPOCHS}"),
        ("解码步数 (T_dec)",    "1",
         (f"{N_DECODE_STEPS} (训练) / 取首步评估"
          if N_DECODE_STEPS > 1 else f"{N_DECODE_STEPS}")),
    ]
    for name, pure_v, attn_v in rows:
        print(f"{name:<22s}  {pure_v:>22s}  {attn_v:>22s}")
    print("=" * 88)
    print("  ⚠️  超参数差异引入混淆变量, 性能差异不能完全归因于架构改进。")
    print()


# ============================================================================
# 可视化对比
# ============================================================================
def plot_compare_2d_error(pure_err: np.ndarray, attn_err: np.ndarray):
    print("=" * 60)
    print("  绘制 2D 误差曲线对比图")
    print("=" * 60)

    n = min(len(pure_err), len(attn_err))
    t = np.arange(n)

    fig, ax = plt.subplots(figsize=(14, 6), dpi=150)
    ax.plot(t, pure_err[:n], linestyle="-", color="red",
            linewidth=1.6, alpha=0.85, label="pure_gru")
    ax.plot(t, attn_err[:n], linestyle="-", color="royalblue",
            linewidth=1.6, alpha=0.85, label="Attention-Bi-GRU + 意图")

    p95 = float(np.percentile(pure_err[:n], 95))
    outlier_mask = pure_err[:n] > p95
    y_top = float(max(pure_err[:n].max(), attn_err[:n].max()))
    ax.fill_between(t, 0, y_top, where=outlier_mask, color="red", alpha=0.08,
                    label=f"pure_gru 离群区 (> P95 = {p95:.2f} m)")

    ax.set_title("预测误差随时间对比: pure_gru vs Attention-Bi-GRU + 意图",
                 fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("时间步", fontsize=12, labelpad=10)
    ax.set_ylabel("3D 综合欧氏距离误差 (m)", fontsize=12, labelpad=10)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="upper right", fontsize=11, framealpha=0.9)
    plt.tight_layout()
    fig.savefig(OUT_CMP_ERR, dpi=300, bbox_inches="tight", pad_inches=0.1)
    print(f"  ✅ 已保存: {OUT_CMP_ERR}\n")


def plot_compare_3d_trajectory(pure_pred, pure_true, attn_pred, attn_true):
    print("=" * 60)
    print("  绘制 3D 轨迹对比图")
    print("=" * 60)

    n = min(len(pure_pred), len(attn_pred), len(pure_true), len(attn_true))
    truth = pure_true[:n]

    fig = plt.figure(figsize=(14, 10), dpi=150)
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(truth[:, 1], truth[:, 0], truth[:, 2],
            linestyle="--", color="gray", linewidth=1.3,
            label="实际轨迹", alpha=0.75)
    ax.plot(pure_pred[:n, 1], pure_pred[:n, 0], pure_pred[:n, 2],
            linestyle="-", color="red", linewidth=1.5,
            label="pure_gru 预测", alpha=0.85)
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
    ax.set_title("三维轨迹对比: 实际 / pure_gru / Attention-Bi-GRU + 意图",
                 fontsize=14, fontweight="bold", pad=20)
    ax.legend(loc="upper left", fontsize=10, framealpha=0.9)
    ax.view_init(elev=25, azim=135)
    plt.tight_layout()
    fig.savefig(OUT_CMP_3D, dpi=300, bbox_inches="tight", pad_inches=0.3)
    print(f"  ✅ 已保存: {OUT_CMP_3D}\n")


def plot_error_cdf(pure_err: np.ndarray, attn_err: np.ndarray):
    print("=" * 60)
    print("  绘制误差 CDF 对比图")
    print("=" * 60)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=150)

    # 左: 直方图
    bins = np.linspace(0, max(pure_err.max(), attn_err.max()), 60)
    axes[0].hist(pure_err, bins=bins, alpha=0.55, color="red",
                 label="pure_gru", density=True)
    axes[0].hist(attn_err, bins=bins, alpha=0.55, color="royalblue",
                 label="Attn-Bi-GRU + 意图", density=True)
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
    xa, ya = _cdf(attn_err)
    axes[1].plot(xp, yp, color="red", linewidth=2, label="pure_gru")
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
    print("  UAV 轨迹预测 — pure_gru vs Attention-Bi-GRU 对比评估")
    print("▓" * 60 + "\n")

    os.makedirs(OUT_CMP_DIR, exist_ok=True)

    print("  【公平性保障与混淆变量声明】")
    print("    ✅ 同一数据源 (OnboardGPS.csv) 与同一 80:20 时序切分")
    print("    ✅ 两条管线 scaler 均仅在训练集上 fit")
    print("    ⚠️  归一化方法不同: pure_gru=MinMax, Attn-Bi-GRU=Standard (论文 Eq.8)")
    print("    ⚠️  超参数差异: lr / batch / dropout / epochs / 架构 (见下方对比表)")
    print()

    pure_pred, pure_true = run_pure_gru()
    attn_pred, attn_true = run_attention_bigru()

    pure_m = _compute_metrics(pure_pred, pure_true)
    attn_m = _compute_metrics(attn_pred, attn_true)
    _apply_outlier_counts_vs_pure_p95(pure_m, attn_m)

    print_metrics_table(pure_m, attn_m)
    print_hyperparam_table()

    plot_compare_2d_error(pure_m["euclidean_errors"], attn_m["euclidean_errors"])
    plot_compare_3d_trajectory(pure_pred, pure_true, attn_pred, attn_true)
    plot_error_cdf(pure_m["euclidean_errors"], attn_m["euclidean_errors"])

    # ── 保存指标 + 超参数汇总 JSON ──
    def _strip(m):
        return {k: v for k, v in m.items() if k != "euclidean_errors"}

    summary = {
        "pipelines": {
            "pure_gru": {
                "model": "UAVTrajectoryGRU (单向 GRU + Linear)",
                "normalization": "MinMaxScaler [0, 1] (train-only fit)",
                "hidden_size": 64,
                "num_layers": 2,
                "input_size": 3,
                "dropout": 0.0,
                "learning_rate": 1e-3,
                "batch_size": 70,
                "max_epochs": 500,
                "intent_fusion": "无",
                "attention": "无",
                "decode_steps": 1,
            },
            "attention_bigru": {
                "model": "AttentionBiGRU (Bi-GRU 编/解码器 + 动态 Attention)",
                "normalization": "StandardScaler (论文 Equation 8, train-only fit)",
                "hidden_size": HIDDEN_SIZE,
                "n_enc_layers": N_ENC_LAYERS,
                "n_dec_layers": N_DEC_LAYERS,
                "d_ff": D_FF,
                "n_intent": N_INTENT,
                "dropout": DROPOUT,
                "learning_rate": LEARNING_RATE,
                "batch_size": BATCH_SIZE,
                "max_epochs": MAX_EPOCHS,
                "intent_fusion": "每层每步 GeLU + Concat 融合 (论文 Eq.31-32)",
                "attention": "动态 Seq2Seq 缩放点积 Attention (Q 每步重新计算)",
                "decode_steps_train": N_DECODE_STEPS,
                "decode_steps_eval": (
                    1 if N_DECODE_STEPS == 1
                    else f"取首步与 pure_gru 对齐 (训练时仍是 {N_DECODE_STEPS} 步)"
                ),
            },
        },
        "fairness_notes": {
            "data_source": "同一 OnboardGPS.csv, 同一 80:20 时序切分",
            "scaler_fit": "两条管线均仅在训练集上 fit",
            "confounders": [
                "归一化方法不同 (MinMax vs Standard)",
                "学习率 / batch_size / dropout / max_epochs 不同",
                "架构整体不同 (含 Bi-GRU / 动态 Attn / 意图融合 三处差异)",
            ],
        },
        "metrics": {
            "pure_gru": _strip(pure_m),
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
