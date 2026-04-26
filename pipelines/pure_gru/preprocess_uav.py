"""
==============================================================================
UAV 轨迹数据预处理脚本 — 用于 GRU 时间序列预测模型
==============================================================================
数据来源: 苏黎世城市微型飞行器 (UMAV/AGZ) 数据集 — OnboardGPS.csv
处理流程:
    1. 核心特征提取 (纬度、经度、海拔)
    2. 数据降采样 (~0.03s → 0.1s)
    3. 按时间顺序 80:20 切分训练集 / 测试集
    4. Min-Max 归一化缩放 ([0, 1]) — 仅在训练集上 fit，消除数据泄漏

【公平性修正】
    原始版本在全量数据上 fit MinMaxScaler 后再切分，导致测试集的
    min/max 信息泄漏到 scaler 中。修正后改为先切分再归一化，
    仅在训练集上 fit scaler，与意图增强版管线保持一致。
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
    """
    步骤 1: 读取原始 OnboardGPS.csv，提取核心位置特征。

    OnboardGPS.csv 列结构 (参考 readme.txt):
        Timpstemp, imgid, lat, lon, alt, s_variance_m_s, c_variance_rad,
        fix_type, eph_m, epv_m, vel_n_m_s, vel_e_m_s, vel_d_m_s, num_sat

    仅保留: lat (纬度), lon (经度), alt (海拔)
    丢弃:   速度、航向、精度估计等所有非位置参数
    """
    print("=" * 60)
    print("步骤 1: 核心特征提取")
    print("=" * 60)

    # 读取 CSV（跳过末尾多余的空列）
    df_raw = pd.read_csv(csv_path, skipinitialspace=True)

    # 清理列名中可能存在的空格
    df_raw.columns = df_raw.columns.str.strip()

    print(f"  原始数据维度: {df_raw.shape}")
    print(f"  原始列名: {list(df_raw.columns)}")

    # 保留时间戳（用于降采样参考）和三个核心位置特征
    # 注意：原始数据中时间戳列名拼写为 "Timpstemp"（原始数据的 typo）
    timestamp_col = "Timpstemp"
    feature_cols = ["lat", "lon", "alt"]

    # 检查必要列是否存在
    required_cols = [timestamp_col] + feature_cols
    for col in required_cols:
        if col not in df_raw.columns:
            raise ValueError(f"  ❌ 缺少必要列: '{col}'。可用列: {list(df_raw.columns)}")

    # 提取并只保留需要的列
    df_selected = df_raw[required_cols].copy()

    # 丢弃包含 NaN 的行（如有）
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
    """
    步骤 2: 数据降采样。

    原始采样频率 ≈ 30Hz (~0.033s/点)，目标频率 = 10Hz (0.1s/点)。
    方法: 等间隔抽取，每 `factor` 个点保留 1 个。

    对于完整数据集 (81,169 点):
        降采样后 ≈ ceil(81169 / 3) ≈ 27,057 点
    """
    print("=" * 60)
    print("步骤 2: 数据降采样")
    print("=" * 60)

    n_original = len(df)
    print(f"  降采样前数据量:  {n_original} 个点")
    print(f"  降采样因子:      {factor} (每 {factor} 个点取 1 个)")
    print(f"  原始间隔 ≈ {ORIGINAL_INTERVAL_S:.6f}s → 目标间隔 = {TARGET_INTERVAL_S}s")

    # 等间隔降采样: 从第 0 行开始，每隔 factor 行取一个
    df_down = df.iloc[::factor].reset_index(drop=True)

    n_downsampled = len(df_down)
    print(f"  ✅ 降采样后数据量: {n_downsampled} 个点")
    print(f"  数据缩减比例:    {n_downsampled / n_original:.4f} ({n_original} → {n_downsampled})")

    # 验证时间间隔 (使用时间戳列)
    if "Timpstemp" in df_down.columns:
        ts = df_down["Timpstemp"].values
        intervals = np.diff(ts)
        # 时间戳单位为微秒 (µs)
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
    """
    步骤 4: Min-Max 归一化 (仅在训练集上 fit)。

    使用 sklearn.preprocessing.MinMaxScaler，将纬度/经度/海拔
    统一映射到 [0, 1] 区间，消除量级差异:
        - 纬度 (lat):  ~47.38°    (十位数量级)
        - 经度 (lon):  ~8.55°     (个位数量级)
        - 海拔 (alt):  ~460-480m  (百位数量级)

    ⚠️ 公平性修正: 仅在训练集上 fit scaler，再 transform 测试集，
    避免测试集的 min/max 信息泄漏到归一化参数中。

    参数:
        train_data:   numpy.ndarray, shape = (n_train, 3), 训练集原始数据
        test_data:    numpy.ndarray, shape = (n_test, 3),  测试集原始数据
        feature_cols: list, 特征列名 (仅用于日志)

    返回:
        train_scaled: numpy.ndarray, shape = (n_train, 3), 归一化后的训练集
        test_scaled:  numpy.ndarray, shape = (n_test, 3),  归一化后的测试集
        scaler:       MinMaxScaler 实例 (用于后续逆变换还原预测值)
    """
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
    """
    步骤 3: 按时间顺序划分训练集与测试集 (80:20)。

    ⚠️ 重要: 这是连续时间序列轨迹预测任务，
    绝对不能使用 sklearn.train_test_split() 打乱数据!
    必须严格按照时间先后顺序进行切分。

    ⚠️ 公平性修正: 切分在归一化之前执行，确保归一化参数
    仅从训练集计算，避免测试集信息泄漏。

    返回:
        train_data: numpy.ndarray, 前 80% 的数据 (原始值)
        test_data:  numpy.ndarray, 后 20% 的数据 (原始值)
    """
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


# ============================================================================
# 主流程
# ============================================================================
def main():
    """
    执行完整的数据预处理流水线:
        原始 CSV → 特征提取 → 降采样 → 训练/测试划分 → 归一化 (仅 train fit)

    【公平性修正】流程顺序调整为先切分再归一化，
    与意图增强版管线 (prepare_intent.py) 保持一致。
    """
    print("\n" + "▓" * 60)
    print("  UAV 轨迹数据预处理 — GRU 模型训练数据准备")
    print("▓" * 60 + "\n")

    # ------------------------------------------------------------------
    # 步骤 1: 读取原始数据，提取核心位置特征 (lat, lon, alt)
    # ------------------------------------------------------------------
    df = load_raw_data(RAW_CSV_PATH)

    # ------------------------------------------------------------------
    # 步骤 2: 降采样至 0.1 秒间隔
    # ------------------------------------------------------------------
    df_downsampled = downsample(df, factor=DOWNSAMPLE_FACTOR)

    feature_cols = ["lat", "lon", "alt"]
    raw_data = df_downsampled[feature_cols].to_numpy(dtype=np.float64)

    # ------------------------------------------------------------------
    # 步骤 3: 按时间顺序 80:20 划分训练集 / 测试集 (先切分!)
    # ------------------------------------------------------------------
    train_raw, test_raw = split_train_test(raw_data, TRAIN_RATIO)

    # ------------------------------------------------------------------
    # 步骤 4: Min-Max 归一化 [0, 1] (仅在训练集上 fit)
    # ------------------------------------------------------------------
    train_data, test_data, scaler = normalize_features(train_raw, test_raw, feature_cols)

    # ------------------------------------------------------------------
    # 保存处理结果
    # ------------------------------------------------------------------
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


# ============================================================================
# 入口
# ============================================================================
if __name__ == "__main__":
    train_data, test_data, scaler = main()
