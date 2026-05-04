"""
==============================================================================
UAV 轨迹数据预处理脚本 — 用于 Bi-GRU 时间序列预测模型
==============================================================================
数据来源: 苏黎世城市微型飞行器 (UMAV/AGZ) 数据集 — OnboardGPS.csv
处理流程:
    1. 核心特征提取 (纬度、经度、海拔)
    2. 数据降采样 (~0.03s → 0.1s)
    3. 按时间顺序 80:20 切分训练集 / 测试集
    4. Min-Max 归一化缩放 ([0, 1]) — 仅在训练集上 fit，消除数据泄漏

与 pure_gru 管线完全一致，输出保存到 processed_data/pure_bigru/ 目录。
==============================================================================
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from config import (
    DOWNSAMPLE_FACTOR,
    ORIGINAL_INTERVAL_S,
    OUTPUT_DIR,
    RAW_CSV_PATH,
    TARGET_INTERVAL_S,
    TRAIN_RATIO,
)


def load_raw_data(csv_path: str) -> pd.DataFrame:
    print("=" * 60)
    print("步骤 1: 核心特征提取")
    print("=" * 60)

    df_raw = pd.read_csv(csv_path, skipinitialspace=True)
    df_raw.columns = df_raw.columns.str.strip()

    print(f"  原始数据维度: {df_raw.shape}")
    print(f"  原始列名: {list(df_raw.columns)}")

    timestamp_col = "Timpstemp"
    feature_cols = ["lat", "lon", "alt"]

    required_cols = [timestamp_col] + feature_cols
    for col in required_cols:
        if col not in df_raw.columns:
            raise ValueError(f"  ❌ 缺少必要列: '{col}'。可用列: {list(df_raw.columns)}")

    df_selected = df_raw[required_cols].copy()

    n_before = len(df_selected)
    df_selected.dropna(inplace=True)
    n_after = len(df_selected)
    if n_before != n_after:
        print(f"  ⚠️  丢弃了 {n_before - n_after} 行含 NaN 的数据")

    print(f"  ✅ 提取后数据维度: {df_selected.shape}")
    print(f"  保留的特征列: {feature_cols}")
    print(f"  丢弃的列: {[c for c in df_raw.columns if c not in required_cols]}")
    print(f"  数据样例 (前 5 行):\n{df_selected.head()}\n")

    return df_selected


def downsample(df: pd.DataFrame, factor: int) -> pd.DataFrame:
    print("=" * 60)
    print("步骤 2: 数据降采样")
    print("=" * 60)

    n_original = len(df)
    print(f"  降采样前数据量:  {n_original} 个点")
    print(f"  降采样因子:      {factor} (每 {factor} 个点取 1 个)")
    print(f"  原始间隔 ≈ {ORIGINAL_INTERVAL_S:.6f}s → 目标间隔 = {TARGET_INTERVAL_S}s")

    df_down = df.iloc[::factor].reset_index(drop=True)

    n_downsampled = len(df_down)
    print(f"  ✅ 降采样后数据量: {n_downsampled} 个点")
    print(f"  数据缩减比例:    {n_downsampled / n_original:.4f} ({n_original} → {n_downsampled})")

    if "Timpstemp" in df_down.columns:
        ts = df_down["Timpstemp"].values
        intervals = np.diff(ts)
        mean_interval_us = np.mean(intervals)
        mean_interval_s = mean_interval_us / 1e6
        print(f"  降采样后平均时间间隔: {mean_interval_s:.6f}s (目标: {TARGET_INTERVAL_S}s)")

    print()
    return df_down


def normalize_features(
    train_data: np.ndarray,
    test_data: np.ndarray,
    feature_cols: list,
) -> tuple:
    print("=" * 60)
    print("步骤 4: Min-Max 归一化 (仅在训练集上 fit — 消除数据泄漏)")
    print("=" * 60)

    print(f"  归一化前各特征统计 (训练集):")
    for i, col in enumerate(feature_cols):
        col_data = train_data[:, i]
        print(f"    {col}: min={col_data.min():.6f}, max={col_data.max():.6f}, "
              f"mean={col_data.mean():.6f}, range={col_data.max() - col_data.min():.6f}")

    scaler = MinMaxScaler(feature_range=(0, 1))
    train_scaled = scaler.fit_transform(train_data)
    test_scaled = scaler.transform(test_data)

    print(f"\n  ✅ 归一化完成! 映射区间: [0, 1]")
    print(f"  归一化后各特征统计 (训练集):")
    for i, col in enumerate(feature_cols):
        col_data = train_scaled[:, i]
        print(f"    {col}: min={col_data.min():.6f}, max={col_data.max():.6f}, "
              f"mean={col_data.mean():.6f}")

    print(f"\n  归一化后各特征统计 (测试集):")
    for i, col in enumerate(feature_cols):
        col_data = test_scaled[:, i]
        print(f"    {col}: min={col_data.min():.6f}, max={col_data.max():.6f}, "
              f"mean={col_data.mean():.6f}")

    print(f"\n  提示: 使用 scaler.inverse_transform() 可将预测结果还原为原始坐标。\n")

    return train_scaled, test_scaled, scaler


def split_train_test(data: np.ndarray, train_ratio: float) -> tuple:
    print("=" * 60)
    print("步骤 3: 训练集 / 测试集划分 (时序切分，无打乱)")
    print("=" * 60)

    n_total = len(data)
    n_train = int(n_total * train_ratio)
    n_test = n_total - n_train

    train_data = data[:n_train]
    test_data = data[n_train:]

    print(f"  总样本数:  {n_total}")
    print(f"  训练集:    {n_train} 个样本 ({n_train / n_total * 100:.1f}%)")
    print(f"  测试集:    {n_test} 个样本 ({n_test / n_total * 100:.1f}%)")
    print(f"  训练集形状: {train_data.shape}")
    print(f"  测试集形状: {test_data.shape}")
    print()

    return train_data, test_data


def main():
    print("\n" + "▓" * 60)
    print("  UAV 轨迹数据预处理 — Bi-GRU 模型训练数据准备")
    print("▓" * 60 + "\n")

    df = load_raw_data(RAW_CSV_PATH)

    df_downsampled = downsample(df, factor=DOWNSAMPLE_FACTOR)

    feature_cols = ["lat", "lon", "alt"]
    raw_data = df_downsampled[feature_cols].to_numpy(dtype=np.float64)

    train_raw, test_raw = split_train_test(raw_data, TRAIN_RATIO)

    train_data, test_data, scaler = normalize_features(train_raw, test_raw, feature_cols)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    train_path = os.path.join(OUTPUT_DIR, "train_data.npy")
    test_path = os.path.join(OUTPUT_DIR, "test_data.npy")
    scaler_path = os.path.join(OUTPUT_DIR, "scaler_params.npz")

    np.save(train_path, train_data)
    np.save(test_path, test_data)

    np.savez(scaler_path,
             data_min=scaler.data_min_,
             data_max=scaler.data_max_,
             feature_range=scaler.feature_range)

    print("=" * 60)
    print("✅ 预处理完成! 文件已保存:")
    print("=" * 60)
    print(f"  训练集: {train_path}")
    print(f"  测试集: {test_path}")
    print(f"  Scaler: {scaler_path}")
    print()

    return train_data, test_data, scaler


if __name__ == "__main__":
    train_data, test_data, scaler = main()
