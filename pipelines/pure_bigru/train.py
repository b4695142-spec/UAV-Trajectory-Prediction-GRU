"""
==============================================================================
UAV 轨迹预测 Bi-GRU 模型 — 训练与验证脚本
==============================================================================
功能:
    1. 加载预处理后的滑动窗口数据 (X_train, Y_train, X_test, Y_test)
    2. 使用 Glorot (Xavier) 初始化模型权重
    3. 以 MSE 损失 + Adam 优化器训练 Bi-GRU 模型
    4. 内置 Early Stopping 机制 (patience=15)，自动保存最佳模型

训练参数 (严格按照文献要求):
    - Batch Size:   70
    - 学习率:       1e-3 (Adam)
    - 最大 Epochs:  500
    - 早停 patience: 15
    - 损失函数:     MSELoss
==============================================================================
"""

import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from config import (
    BATCH_SIZE,
    DATA_DIR,
    HIDDEN_SIZE,
    INPUT_SIZE,
    LEARNING_RATE,
    LOSS_CURVE_PATH,
    MAX_EPOCHS,
    MODEL_SAVE_PATH,
    NUM_LAYERS,
    OUTPUT_SIZE,
    PATIENCE,
    PROJECT_ROOT,
    X_TEST_PATH,
    X_TRAIN_PATH,
    Y_TEST_PATH,
    Y_TRAIN_PATH,
)
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)
from core.bigru_model import UAVTrajectoryBiGRU
from core.training_utils import (
    EarlyStopping,
    evaluate,
    get_device,
    init_weights,
    plot_loss_curves,
    train_one_epoch,
)


def load_data() -> tuple:
    print("=" * 60)
    print("  数据加载")
    print("=" * 60)

    X_train = np.load(X_TRAIN_PATH)
    Y_train = np.load(Y_TRAIN_PATH)
    X_test = np.load(X_TEST_PATH)
    Y_test = np.load(Y_TEST_PATH)

    print(f"  X_train: {X_train.shape}  Y_train: {Y_train.shape}")
    print(f"  X_test:  {X_test.shape}   Y_test:  {Y_test.shape}")

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    Y_train_t = torch.tensor(Y_train, dtype=torch.float32)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    Y_test_t = torch.tensor(Y_test, dtype=torch.float32)

    train_dataset = TensorDataset(X_train_t, Y_train_t)
    test_dataset = TensorDataset(X_test_t, Y_test_t)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False
    )

    print(f"  训练集 batches: {len(train_loader)} (batch_size={BATCH_SIZE})")
    print(f"  测试集 batches: {len(test_loader)} (batch_size={BATCH_SIZE})")
    print()

    return train_loader, test_loader


def train(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
) -> None:
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    early_stopping = EarlyStopping(patience=PATIENCE)

    train_losses = []
    test_losses = []

    print("=" * 60)
    print("  开始训练")
    print("=" * 60)
    print(f"  损失函数:  MSELoss")
    print(f"  优化器:    Adam (lr={LEARNING_RATE})")
    print(f"  最大 Epochs: {MAX_EPOCHS}")
    print(f"  早停 Patience: {PATIENCE}")
    print(f"  Batch Size:  {BATCH_SIZE}")
    print("-" * 60)
    print(f"  {'Epoch':>6}  |  {'Train Loss':>14}  |  {'Test Loss':>14}  |  备注")
    print("-" * 60)

    start_time = time.time()

    for epoch in range(1, MAX_EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        test_loss = evaluate(model, test_loader, criterion, device)
        early_stopping(test_loss, model, epoch)

        train_losses.append(train_loss)
        test_losses.append(test_loss)

        remark = ""
        if test_loss <= early_stopping.best_loss:
            remark = "★ Best"
        elif early_stopping.counter > 0:
            remark = f"patience {early_stopping.counter}/{PATIENCE}"

        print(
            f"  {epoch:>6}  |  {train_loss:>14.8f}  |  {test_loss:>14.8f}  |  {remark}"
        )

        if early_stopping.early_stop:
            break

    elapsed = time.time() - start_time

    print()
    print("=" * 60)
    print("  训练结束")
    print("=" * 60)
    print(f"  完成 Epoch 数:    {epoch}")
    print(f"  最佳 Epoch:       {early_stopping.best_epoch}")
    print(f"  最佳 Test Loss:   {early_stopping.best_loss:.8f}")
    print(f"  总训练时间:       {elapsed:.2f} 秒 ({elapsed / 60:.2f} 分钟)")

    if early_stopping.best_state is not None:
        model.load_state_dict(early_stopping.best_state)
        print(f"\n  ✅ 已加载最佳模型权重 (Epoch {early_stopping.best_epoch})")

    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"  ✅ 最佳模型已保存至: {MODEL_SAVE_PATH}")

    plot_loss_curves(
        train_losses=train_losses,
        test_losses=test_losses,
        best_epoch=early_stopping.best_epoch,
        save_path=LOSS_CURVE_PATH,
        title="Pure Bi-GRU — 训练损失曲线",
    )


def main():
    print("\n" + "▓" * 60)
    print("  UAV 轨迹预测 Bi-GRU 模型 — 训练与验证")
    print("▓" * 60 + "\n")

    device = get_device()
    print()

    train_loader, test_loader = load_data()

    print("=" * 60)
    print("  模型初始化")
    print("=" * 60)

    model = UAVTrajectoryBiGRU(
        input_size=INPUT_SIZE,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        output_size=OUTPUT_SIZE,
    )

    init_weights(model)
    print("  ✅ Xavier Uniform 权重初始化完成")

    model.to(device)
    print(f"  ✅ 模型已移至 {device}")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  总参数量: {total_params:,}")
    print(f"\n{model}\n")

    train(model, train_loader, test_loader, device)

    print("\n" + "▓" * 60)
    print("  全部完成!")
    print("▓" * 60 + "\n")


if __name__ == "__main__":
    main()
