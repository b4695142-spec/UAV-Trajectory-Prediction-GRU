"""
==============================================================================
UAV 轨迹预测 GRU (意图识别增强版) — 训练脚本
==============================================================================
与 train.py 功能等价，唯一区别在于:

    - 输入序列来自 processed_data/intent/X_{train,test}_intent.npy
      每个时间步的特征 = [筛选后的标准化特征, 4 维 SVM 意图概率]
    - 输入维度 input_size = X.shape[-1] (由 prepare_intent.py 动态决定)
    - 模型权重保存至 processed_data/intent/best_gru_model_intent.pth

此文件不触碰任何纯 GRU 相关产物，可以与原始 train.py 并存并随时回退。
==============================================================================
"""

from __future__ import annotations

import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from gru_model import UAVTrajectoryGRU


# ============================================================================
# 配置参数
# ============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INTENT_DIR = os.path.join(BASE_DIR, "processed_data", "intent")

X_TRAIN_PATH = os.path.join(INTENT_DIR, "X_train_intent.npy")
Y_TRAIN_PATH = os.path.join(INTENT_DIR, "Y_train_intent.npy")
X_TEST_PATH = os.path.join(INTENT_DIR, "X_test_intent.npy")
Y_TEST_PATH = os.path.join(INTENT_DIR, "Y_test_intent.npy")

MODEL_SAVE_PATH = os.path.join(INTENT_DIR, "best_gru_model_intent.pth")

# 训练超参数 — 与 train.py 严格一致，确保对比公平
BATCH_SIZE = 70
LEARNING_RATE = 1e-3
MAX_EPOCHS = 500
PATIENCE = 15

# 模型超参数 — hidden_size / num_layers / output_size 与原模型一致
# input_size 会根据 X 的最后一维动态确定 (位置 ∪ RF 筛选列 + 4 意图概率)
HIDDEN_SIZE = 64
NUM_LAYERS = 2
OUTPUT_SIZE = 3
# 【公平性修正】dropout 设为 0.0，与纯 GRU 管线严格一致
# 确保对比时唯一变量为"是否加入意图特征"，排除正则化差异干扰
DROPOUT = 0.0


# ============================================================================
# 工具函数 (从 train.py 复用等价逻辑，保持自包含以方便对比)
# ============================================================================
def get_device() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"  使用设备: {torch.cuda.get_device_name(0)} (CUDA)")
    else:
        device = torch.device("cpu")
        print(f"  使用设备: CPU")
    return device


def load_data() -> tuple:
    """加载意图增广数据集。"""
    print("=" * 60)
    print("  数据加载 (意图增广版)")
    print("=" * 60)

    for p in (X_TRAIN_PATH, Y_TRAIN_PATH, X_TEST_PATH, Y_TEST_PATH):
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"未找到 {p}\n请先运行: python prepare_intent.py"
            )

    X_train = np.load(X_TRAIN_PATH)
    Y_train = np.load(Y_TRAIN_PATH)
    X_test = np.load(X_TEST_PATH)
    Y_test = np.load(Y_TEST_PATH)

    print(f"  X_train: {X_train.shape}  Y_train: {Y_train.shape}")
    print(f"  X_test:  {X_test.shape}   Y_test:  {Y_test.shape}")

    input_size = X_train.shape[-1]
    print(f"  动态输入维度 (input_size) = {input_size}")

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    Y_train_t = torch.tensor(Y_train, dtype=torch.float32)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    Y_test_t = torch.tensor(Y_test, dtype=torch.float32)

    train_ds = TensorDataset(X_train_t, Y_train_t)
    test_ds = TensorDataset(X_test_t, Y_test_t)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

    print(f"  训练集 batches: {len(train_loader)} (batch_size={BATCH_SIZE})")
    print(f"  测试集 batches: {len(test_loader)} (batch_size={BATCH_SIZE})")
    print()

    return train_loader, test_loader, input_size


def init_weights(model: nn.Module) -> None:
    """Glorot/Xavier Uniform 权重初始化。"""
    for name, param in model.named_parameters():
        if "weight" in name:
            nn.init.xavier_uniform_(param.data)
        elif "bias" in name:
            nn.init.zeros_(param.data)


class EarlyStopping:
    """与 train.py 中一致的 Early Stopping 机制。"""

    def __init__(self, patience: int = 15):
        self.patience = patience
        self.best_loss = float("inf")
        self.best_epoch = 0
        self.counter = 0
        self.best_state = None
        self.early_stop = False

    def __call__(self, val_loss: float, model: nn.Module, epoch: int):
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.best_epoch = epoch
            self.counter = 0
            self.best_state = {k: v.clone().cpu() for k, v in model.state_dict().items()}
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                print(f"\n  ⏹ 早停触发! 连续 {self.patience} 个 Epoch 验证损失无改善。")
                print(f"    最佳 Epoch: {self.best_epoch}, 最佳 Test Loss: {self.best_loss:.8f}")


# ============================================================================
# 训练 / 验证 (等同于 train.py)
# ============================================================================
def train_one_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    total, n = 0.0, 0
    for X_batch, Y_batch in train_loader:
        X_batch = X_batch.to(device)
        Y_batch = Y_batch.to(device)
        pred = model(X_batch)
        loss = criterion(pred, Y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += loss.item()
        n += 1
    return total / n


@torch.no_grad()
def evaluate(model, test_loader, criterion, device):
    model.eval()
    total, n = 0.0, 0
    for X_batch, Y_batch in test_loader:
        X_batch = X_batch.to(device)
        Y_batch = Y_batch.to(device)
        pred = model(X_batch)
        loss = criterion(pred, Y_batch)
        total += loss.item()
        n += 1
    return total / n


def train(model, train_loader, test_loader, device):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    early_stopping = EarlyStopping(patience=PATIENCE)

    print("=" * 60)
    print("  开始训练 (意图增广版 GRU)")
    print("=" * 60)
    print(f"  损失函数:    MSELoss")
    print(f"  优化器:      Adam (lr={LEARNING_RATE})")
    print(f"  最大 Epochs: {MAX_EPOCHS}")
    print(f"  早停 Patience: {PATIENCE}")
    print(f"  Batch Size:  {BATCH_SIZE}")
    print("-" * 60)
    print(f"  {'Epoch':>6}  |  {'Train Loss':>14}  |  {'Test Loss':>14}  |  备注")
    print("-" * 60)

    start = time.time()
    for epoch in range(1, MAX_EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        test_loss = evaluate(model, test_loader, criterion, device)
        early_stopping(test_loss, model, epoch)

        remark = ""
        if test_loss <= early_stopping.best_loss:
            remark = "★ Best"
        elif early_stopping.counter > 0:
            remark = f"patience {early_stopping.counter}/{PATIENCE}"

        print(f"  {epoch:>6}  |  {train_loss:>14.8f}  |  {test_loss:>14.8f}  |  {remark}")

        if early_stopping.early_stop:
            break

    elapsed = time.time() - start
    print()
    print("=" * 60)
    print("  训练结束")
    print("=" * 60)
    print(f"  完成 Epoch 数:   {epoch}")
    print(f"  最佳 Epoch:      {early_stopping.best_epoch}")
    print(f"  最佳 Test Loss:  {early_stopping.best_loss:.8f}")
    print(f"  总训练时间:      {elapsed:.2f} 秒 ({elapsed / 60:.2f} 分钟)")

    if early_stopping.best_state is not None:
        model.load_state_dict(early_stopping.best_state)
        print(f"\n  ✅ 已加载最佳模型权重 (Epoch {early_stopping.best_epoch})")

    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"  ✅ 最佳模型已保存至: {MODEL_SAVE_PATH}")


# ============================================================================
# 主函数
# ============================================================================
def main():
    print("\n" + "▓" * 60)
    print("  UAV 轨迹预测 GRU (意图识别增强版) — 训练")
    print("▓" * 60 + "\n")

    device = get_device()
    print()

    train_loader, test_loader, input_size = load_data()

    print("=" * 60)
    print("  模型初始化")
    print("=" * 60)

    model = UAVTrajectoryGRU(
        input_size=input_size,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        output_size=OUTPUT_SIZE,
        dropout=DROPOUT,
    )
    init_weights(model)
    print(f"  ✅ Xavier Uniform 权重初始化完成 (GRU dropout = {DROPOUT})")

    model.to(device)
    print(f"  ✅ 模型已移至 {device}")

    total = sum(p.numel() for p in model.parameters())
    print(f"  总参数量: {total:,}")
    print(f"\n{model}\n")

    train(model, train_loader, test_loader, device)

    print("\n" + "▓" * 60)
    print("  意图增广版训练完成!")
    print("▓" * 60 + "\n")


if __name__ == "__main__":
    main()
