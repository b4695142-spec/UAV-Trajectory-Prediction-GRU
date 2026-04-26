"""
==============================================================================
滑动窗口序列构建脚本 — 用于 GRU 无人机轨迹预测模型
==============================================================================
输入: 预处理后的 train_data.npy / test_data.npy，形状 (N, 3)
输出: X_train, Y_train, X_test, Y_test 滑动窗口序列

论文定义:
    对于任意有效时间点 t:
        Input_t  = data[t - Look_Back + 1 : t + 1]   → 形状 (Look_Back, 3)
        Output_t = data[t + Forward_Length]            → 形状 (3,)

    边界约束:
        t - Look_Back + 1 >= 0        (窗口起始不越界)
        t + Forward_Length < N         (预测目标不越界)

    因此有效 t 的范围: Look_Back - 1 <= t <= N - Forward_Length - 1
==============================================================================
"""

import os
import numpy as np

from config import (
    DATA_DIR,
    FORWARD_LENGTH,
    LOOK_BACK,
    OUTPUT_DIR,
    TEST_DATA_PATH,
    TRAIN_DATA_PATH,
)


def build_sliding_window_sequences(
    data: np.ndarray,
    look_back: int = LOOK_BACK,
    forward_length: int = FORWARD_LENGTH,
    dataset_name: str = "dataset"
) -> tuple:
    """
    根据论文公式构建滑动窗口序列。

    数学定义:
        对于时间点 t (Look_Back-1 <= t <= N-Forward_Length-1):
            X[t] = data[t - Look_Back + 1 : t + 1]    # 过去 Look_Back 个连续点
            Y[t] = data[t + Forward_Length]             # 未来第 Forward_Length 个点

    参数:
        data:           np.ndarray, 形状 (N, 3), 归一化后的轨迹数据
        look_back:      int, 历史观测窗口长度
        forward_length: int, 未来预测步长
        dataset_name:   str, 数据集名称 (仅用于日志输出)

    返回:
        X: np.ndarray, 形状 (有效样本数, look_back, 3)
        Y: np.ndarray, 形状 (有效样本数, 3)
    """
    N = len(data)  # 数据集总长度

    print(f"\n  [{dataset_name}] 数据长度 N = {N}")
    print(f"  [{dataset_name}] Look_Back = {look_back}, Forward_Length = {forward_length}")

    # ------------------------------------------------------------------
    # 计算有效时间点 t 的范围
    # ------------------------------------------------------------------
    # 约束 1: 窗口起始 t - Look_Back + 1 >= 0  →  t >= Look_Back - 1
    t_min = look_back - 1

    # 约束 2: 预测目标 t + Forward_Length < N   →  t <= N - Forward_Length - 1
    t_max = N - forward_length - 1

    # 边界检查: 如果数据量不足以构成哪怕一个有效样本
    if t_min > t_max:
        raise ValueError(
            f"  ❌ [{dataset_name}] 数据长度不足! "
            f"需要至少 {look_back + forward_length} 个点, "
            f"但只有 {N} 个点。"
        )

    n_valid = t_max - t_min + 1  # 有效样本总数
    print(f"  [{dataset_name}] 有效 t 范围: [{t_min}, {t_max}]")
    print(f"  [{dataset_name}] 跳过的边界样本数: {N - n_valid} "
          f"(前 {t_min} 个 + 后 {forward_length} 个)")

    # ------------------------------------------------------------------
    # 构建滑动窗口序列
    # ------------------------------------------------------------------
    X_list = []
    Y_list = []

    for t in range(t_min, t_max + 1):
        # 输入序列: 从 t-Look_Back+1 到 t (含)，共 Look_Back 个连续点
        # 即 data[t - look_back + 1 : t + 1]
        input_seq = data[t - look_back + 1: t + 1]   # shape: (look_back, 3)

        # 目标标签: 未来第 Forward_Length 个时间点的坐标
        target = data[t + forward_length]              # shape: (3,)

        X_list.append(input_seq)
        Y_list.append(target)

    # 转换为 numpy 数组
    X = np.array(X_list)  # shape: (n_valid, look_back, 3)
    Y = np.array(Y_list)  # shape: (n_valid, 3)

    print(f"  [{dataset_name}] ✅ 序列构建完成!")
    print(f"  [{dataset_name}] X 形状: {X.shape}  (样本数, Look_Back, 特征维度)")
    print(f"  [{dataset_name}] Y 形状: {Y.shape}  (样本数, 特征维度)")

    return X, Y


# ============================================================================
# 主流程
# ============================================================================
def main():
    """
    对 train_data 和 test_data 分别构建滑动窗口序列，
    生成 GRU 模型所需的 X_train, Y_train, X_test, Y_test。
    """
    print("▓" * 60)
    print("  滑动窗口序列构建 — GRU 轨迹预测模型")
    print("▓" * 60)

    print(f"\n  核心超参数:")
    print(f"    Look_Back       = {LOOK_BACK}  (历史观测步长)")
    print(f"    Forward_Length  = {FORWARD_LENGTH}  (未来预测步长, 即 {FORWARD_LENGTH * 0.1:.1f}s)")

    # ------------------------------------------------------------------
    # 加载预处理数据
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("加载预处理数据")
    print("=" * 60)

    train_data = np.load(TRAIN_DATA_PATH)
    test_data = np.load(TEST_DATA_PATH)

    print(f"  train_data 形状: {train_data.shape}")
    print(f"  test_data  形状: {test_data.shape}")

    # ------------------------------------------------------------------
    # 构建训练集滑动窗口
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("构建训练集滑动窗口序列")
    print("=" * 60)

    X_train, Y_train = build_sliding_window_sequences(
        data=train_data,
        look_back=LOOK_BACK,
        forward_length=FORWARD_LENGTH,
        dataset_name="训练集"
    )

    # ------------------------------------------------------------------
    # 构建测试集滑动窗口
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("构建测试集滑动窗口序列")
    print("=" * 60)

    X_test, Y_test = build_sliding_window_sequences(
        data=test_data,
        look_back=LOOK_BACK,
        forward_length=FORWARD_LENGTH,
        dataset_name="测试集"
    )

    # ------------------------------------------------------------------
    # 形状验证 (断言检查)
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("形状验证")
    print("=" * 60)

    assert X_train.shape[1] == LOOK_BACK, f"X_train 第2维应为 {LOOK_BACK}, 实际 {X_train.shape[1]}"
    assert X_train.shape[2] == 3, f"X_train 第3维应为 3, 实际 {X_train.shape[2]}"
    assert Y_train.shape[1] == 3, f"Y_train 第2维应为 3, 实际 {Y_train.shape[1]}"
    assert X_test.shape[1] == LOOK_BACK, f"X_test 第2维应为 {LOOK_BACK}, 实际 {X_test.shape[1]}"
    assert X_test.shape[2] == 3, f"X_test 第3维应为 3, 实际 {X_test.shape[2]}"
    assert Y_test.shape[1] == 3, f"Y_test 第2维应为 3, 实际 {Y_test.shape[1]}"

    print("  ✅ 所有形状验证通过!")
    print(f"    X_train: {X_train.shape}  →  (样本数, {LOOK_BACK}, 3)")
    print(f"    Y_train: {Y_train.shape}  →  (样本数, 3)")
    print(f"    X_test:  {X_test.shape}  →  (样本数, {LOOK_BACK}, 3)")
    print(f"    Y_test:  {Y_test.shape}  →  (样本数, 3)")

    # ------------------------------------------------------------------
    # 保存结果
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("保存序列数据")
    print("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    paths = {
        "X_train": os.path.join(OUTPUT_DIR, "X_train.npy"),
        "Y_train": os.path.join(OUTPUT_DIR, "Y_train.npy"),
        "X_test":  os.path.join(OUTPUT_DIR, "X_test.npy"),
        "Y_test":  os.path.join(OUTPUT_DIR, "Y_test.npy"),
    }

    np.save(paths["X_train"], X_train)
    np.save(paths["Y_train"], Y_train)
    np.save(paths["X_test"], X_test)
    np.save(paths["Y_test"], Y_test)

    print(f"  ✅ 文件已保存至 {OUTPUT_DIR}/")
    for name, path in paths.items():
        print(f"    {name}: {path}")

    # ------------------------------------------------------------------
    # 汇总信息
    # ------------------------------------------------------------------
    print("\n" + "▓" * 60)
    print("  汇总")
    print("▓" * 60)
    print(f"""
    ┌──────────────────────────────────────────────────┐
    │  超参数                                          │
    │    Look_Back       = {LOOK_BACK:<26d}  │
    │    Forward_Length  = {FORWARD_LENGTH:<26d}  │
    │    预测时间跨度    = {FORWARD_LENGTH * 0.1:<26.1f}  │
    ├──────────────────────────────────────────────────┤
    │  训练集                                          │
    │    X_train  {str(X_train.shape):<36s} │
    │    Y_train  {str(Y_train.shape):<36s} │
    ├──────────────────────────────────────────────────┤
    │  测试集                                          │
    │    X_test   {str(X_test.shape):<36s} │
    │    Y_test   {str(Y_test.shape):<36s} │
    └──────────────────────────────────────────────────┘
    """)

    return X_train, Y_train, X_test, Y_test


# ============================================================================
# 入口
# ============================================================================
if __name__ == "__main__":
    X_train, Y_train, X_test, Y_test = main()
