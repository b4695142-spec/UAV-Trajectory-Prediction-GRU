"""
==============================================================================
意图识别 — 动态特征提取模块
==============================================================================
从 Log Files/OnboardGPS.csv 中动态提取所有可用的数值特征，并额外计算若干
刻画机动状态的派生特征 (derived features)，用于后续随机森林特征筛选
与 SVM 意图分类。

设计原则:
    1. 动态列探测: 不硬编码列名，避免因数据源字段变化而失效。
    2. 丢弃无意义列: 时间戳、图像编号等标识符不参与分类建模。
    3. 衍生运动学特征: 根据 GPS 速度分量额外计算水平速度、航向、
       航向角速率、高度变化率、水平加速度等特征，以显式暴露机动信息。
    4. 下采样: 与主管线 preprocess_uav.py 保持一致 (每 3 个点保留 1 个)，
       保证时间轴对齐。
==============================================================================
"""

from __future__ import annotations

import os
from typing import Tuple

import numpy as np
import pandas as pd


# 默认降采样因子 (与 preprocess_uav.py 保持一致: 0.1s / 0.033333s ≈ 3)
DEFAULT_DOWNSAMPLE_FACTOR = 3

# 标识符类列 — 不作为机动判别特征
DROP_COLUMNS = {
    "Timpstemp",   # 时间戳 (原始数据 typo)
    "Timestamp",   # 时间戳 (正确拼写兼容)
    "Timestemp",   # readme.md 中的拼写兼容
    "imgid",       # 图像编号
}

# 从原始列额外派生的特征名称 (若基础列存在则会被生成)
DERIVED_FEATURE_NAMES = [
    "h_speed",       # 水平速度 = sqrt(vel_n^2 + vel_e^2)
    "total_speed",   # 总速度 = sqrt(vel_n^2 + vel_e^2 + vel_d^2)
    "heading",       # 航向角 = atan2(vel_e, vel_n)
    "heading_rate",  # 航向角速率 = d(heading)/dt (已做 unwrap)
    "alt_rate",      # 高度变化率 = d(alt)/dt
    "h_accel",       # 水平加速度 = d(h_speed)/dt
    "climb_angle",   # 爬升角 = atan2(-vel_d, h_speed) (NED 坐标系下 vel_d>0 为下降)
]


def _read_raw_csv(csv_path: str) -> pd.DataFrame:
    """读取原始 CSV，并清理列名末尾空格与空尾列。"""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV 文件不存在: {csv_path}")

    df = pd.read_csv(csv_path, skipinitialspace=True)
    df.columns = df.columns.str.strip()

    # 丢弃完全由 NaN 构成的尾列 (原始文件末尾有多余逗号)
    df = df.dropna(axis=1, how="all")
    # 丢弃任何包含 NaN 的行
    df = df.dropna(axis=0, how="any").reset_index(drop=True)

    return df


def _pick_numeric_columns(df: pd.DataFrame) -> list:
    """
    从 DataFrame 中动态选出所有数值型列，并排除标识符类列。
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    picked = [c for c in numeric_cols if c not in DROP_COLUMNS]
    return picked


def _compute_derived_features(
    df: pd.DataFrame,
    dt: float,
) -> pd.DataFrame:
    """
    基于原始列尝试派生一系列运动学特征。
    仅在所需基础列存在时才生成相应的派生列。

    参数:
        df: 输入 DataFrame (必须已按时间排好序)
        dt: 相邻采样点的时间间隔 (秒)

    返回:
        仅包含派生列的 DataFrame (行数与 df 对齐)
    """
    derived = {}

    vel_n = df["vel_n_m_s"].to_numpy() if "vel_n_m_s" in df.columns else None
    vel_e = df["vel_e_m_s"].to_numpy() if "vel_e_m_s" in df.columns else None
    vel_d = df["vel_d_m_s"].to_numpy() if "vel_d_m_s" in df.columns else None
    alt = df["alt"].to_numpy() if "alt" in df.columns else None

    if vel_n is not None and vel_e is not None:
        h_speed = np.sqrt(vel_n ** 2 + vel_e ** 2)
        derived["h_speed"] = h_speed
        heading = np.arctan2(vel_e, vel_n)
        derived["heading"] = heading
        # 航向角速率 (弧度/秒) — 使用 unwrap 消除 ±π 跳变
        heading_unwrapped = np.unwrap(heading)
        heading_rate = np.gradient(heading_unwrapped, dt)
        derived["heading_rate"] = heading_rate
        # 水平加速度
        h_accel = np.gradient(h_speed, dt)
        derived["h_accel"] = h_accel

        if vel_d is not None:
            derived["total_speed"] = np.sqrt(vel_n ** 2 + vel_e ** 2 + vel_d ** 2)
            # NED 下 vel_d > 0 表示下降，因此爬升角 = atan2(-vel_d, h_speed)
            eps = 1e-6
            derived["climb_angle"] = np.arctan2(-vel_d, h_speed + eps)

    if alt is not None:
        derived["alt_rate"] = np.gradient(alt, dt)

    return pd.DataFrame(derived)


def extract_intent_features(
    csv_path: str,
    downsample_factor: int = DEFAULT_DOWNSAMPLE_FACTOR,
    target_interval_s: float = 0.1,
    verbose: bool = True,
) -> Tuple[np.ndarray, list, pd.DataFrame]:
    """
    从 OnboardGPS.csv 提取完整的意图识别特征矩阵。

    流程:
        1. 读取原始 CSV，清理列名和 NaN。
        2. 等间隔降采样 (factor=3 → 10Hz)。
        3. 动态筛选所有数值列作为基础特征 (排除标识符)。
        4. 计算若干派生运动学特征。
        5. 最终返回特征矩阵 + 特征列名 + 完整 DataFrame (供标签生成等用)。

    参数:
        csv_path:           OnboardGPS.csv 路径
        downsample_factor:  降采样因子 (默认 3，与主管线保持一致)
        target_interval_s:  目标采样间隔 (秒)，用于派生特征中的 d/dt 计算
        verbose:            是否打印日志

    返回:
        features:   np.ndarray, shape (N, n_features), 尚未标准化
        feature_names: list[str], 与 features 列对应的名称
        df_full:    pd.DataFrame, 降采样后的完整数据 (含标识符列, 便于标签生成)
    """
    if verbose:
        print("=" * 60)
        print("  [intent] 动态特征提取")
        print("=" * 60)

    # ------------------------------------------------------------------
    # 1. 读取 + 清洗
    # ------------------------------------------------------------------
    df = _read_raw_csv(csv_path)
    if verbose:
        print(f"  原始数据维度: {df.shape}")
        print(f"  原始列名:     {list(df.columns)}")

    # ------------------------------------------------------------------
    # 2. 降采样
    # ------------------------------------------------------------------
    df_down = df.iloc[::downsample_factor].reset_index(drop=True)
    if verbose:
        print(f"  降采样 (factor={downsample_factor}) 后: {df_down.shape}")

    # ------------------------------------------------------------------
    # 3. 挑选基础数值特征 (排除标识符列)
    # ------------------------------------------------------------------
    base_cols = _pick_numeric_columns(df_down)
    if verbose:
        print(f"  保留的基础数值列 ({len(base_cols)}): {base_cols}")

    # ------------------------------------------------------------------
    # 4. 计算派生运动学特征
    # ------------------------------------------------------------------
    derived_df = _compute_derived_features(df_down, dt=target_interval_s)
    derived_cols = list(derived_df.columns)
    if verbose:
        print(f"  派生特征列 ({len(derived_cols)}): {derived_cols}")

    # ------------------------------------------------------------------
    # 5. 合并为最终特征矩阵
    # ------------------------------------------------------------------
    # 基础列 + 派生列
    feat_df = pd.concat([df_down[base_cols], derived_df], axis=1)

    # 丢弃零方差列 (对分类器没有贡献且会污染 RF 重要性)
    keep_mask = feat_df.var(axis=0) > 1e-12
    if (~keep_mask).any():
        dropped = feat_df.columns[~keep_mask].tolist()
        if verbose:
            print(f"  丢弃零方差列: {dropped}")
        feat_df = feat_df.loc[:, keep_mask]

    feature_names = list(feat_df.columns)
    features = feat_df.to_numpy(dtype=np.float64)

    if verbose:
        print(f"  ✅ 最终特征矩阵: {features.shape}")
        print(f"  最终特征列 ({len(feature_names)}): {feature_names}")
        print()

    return features, feature_names, df_down
