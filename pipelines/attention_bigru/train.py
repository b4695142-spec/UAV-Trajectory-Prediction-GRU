"""
==============================================================================
UAV 轨迹预测 — Attention-Bi-GRU 训练脚本
==============================================================================
功能:
    1. 加载 prepare_data.py 生成的 StandardScaler 归一化增广数据
       (X 包含 "[位置 ∪ RF 筛选列] ⊕ 4 维 SVM 概率")
    2. 拆分增广特征 X 中的 "结构性特征" 与 "意图概率" 两部分
    3. 实例化 AttentionBiGRU 模型 (论文 Table 3 超参数)
    4. Xavier Uniform 权重初始化
    5. MSELoss + Adam(lr=5e-4) + EarlyStopping(patience=15) 训练
    6. 保存最佳模型至 processed_data/attention_bigru/best_attention_bigru_model.pth

训练参数 (严格遵循论文 Table 3):
    - Batch Size:  64
    - 学习率:      5e-4 (Adam)
    - 最大 Epochs: 300
    - 早停 patience: 15
    - 损失函数:    MSELoss
==============================================================================
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from config import (
    BATCH_SIZE,
    D_FF,
    DROPOUT,
    HIDDEN_SIZE,
    LEARNING_RATE,
    MAX_EPOCHS,
    MODEL_SAVE_PATH,
    N_DEC_LAYERS,
    N_DECODE_STEPS,
    N_ENC_LAYERS,
    N_INTENT,
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
from core.attention_bigru_model import AttentionBiGRU


# ============================================================================
# 自定义 Dataset — 拆分增广 X 中的结构性特征 / 意图概率
# ============================================================================
class AttentionBiGRUDataset(Dataset):
    """
    每个样本返回 (x_feat, x_intent, y) 三元组:
        x_feat   : (look_back, input_size)              结构性特征 (位置 + RF 筛选列)
        x_intent : (look_back, n_intent)                SVM 概率向量序列
        y        : (output_size,)        当 decode_steps == 1
                   (decode_steps, out)   当 decode_steps  > 1
                   目标 (lat, lon, alt, StandardScaler), 形状由 prepare_data.py
                   生成的 .npy 决定 (即 N_DECODE_STEPS 配置)。
    """

    def __init__(
        self,
        X_feat: np.ndarray,
        X_intent: np.ndarray,
        Y: np.ndarray,
    ):
        self.X_feat = torch.tensor(X_feat, dtype=torch.float32)
        self.X_intent = torch.tensor(X_intent, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.Y)

    def __getitem__(self, idx: int):
        return self.X_feat[idx], self.X_intent[idx], self.Y[idx]


# ============================================================================
# 工具函数
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
    """加载 prepare_data.py 生成的增广数据, 拆分结构性特征与意图概率。"""
    print("=" * 60)
    print("  数据加载 (Attention-Bi-GRU 增广版)")
    print("=" * 60)

    for p in (X_TRAIN_PATH, Y_TRAIN_PATH, X_TEST_PATH, Y_TEST_PATH):
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"未找到 {p}\n请先运行: python pipelines/attention_bigru/prepare_data.py"
            )

    X_train = np.load(X_TRAIN_PATH)
    Y_train = np.load(Y_TRAIN_PATH)
    X_test = np.load(X_TEST_PATH)
    Y_test = np.load(Y_TEST_PATH)

    print(f"  X_train: {X_train.shape}  Y_train: {Y_train.shape}")
    print(f"  X_test:  {X_test.shape}   Y_test:  {Y_test.shape}")

    # ── Y 形状 / 解码步数一致性校验 ─────────────────────────────────────
    if N_DECODE_STEPS == 1:
        if Y_train.ndim != 2:
            raise ValueError(
                f"N_DECODE_STEPS=1 但 Y_train 维度为 {Y_train.ndim} (期望 2D); "
                f"请重新运行 prepare_data.py"
            )
    else:
        if Y_train.ndim != 3 or Y_train.shape[1] != N_DECODE_STEPS:
            raise ValueError(
                f"N_DECODE_STEPS={N_DECODE_STEPS} 但 Y_train shape={Y_train.shape} "
                f"(期望 (n, {N_DECODE_STEPS}, 3)); 请重新运行 prepare_data.py"
            )

    # ── 拆分: 最后 N_INTENT 列为 SVM 概率, 其余为结构性特征 ──
    input_size = X_train.shape[-1] - N_INTENT
    X_feat_train = X_train[:, :, :-N_INTENT]
    X_intent_train = X_train[:, :, -N_INTENT:]
    X_feat_test = X_test[:, :, :-N_INTENT]
    X_intent_test = X_test[:, :, -N_INTENT:]

    print(f"  动态输入维度 input_size = {input_size}  (结构性特征)")
    print(f"  意图概率维度 n_intent  = {N_INTENT}     (SVM 4 维概率)")
    print(f"  解码步数 T_dec        = {N_DECODE_STEPS}")

    train_ds = AttentionBiGRUDataset(X_feat_train, X_intent_train, Y_train)
    test_ds = AttentionBiGRUDataset(X_feat_test, X_intent_test, Y_test)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False
    )
    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=False
    )

    print(f"  训练集 batches: {len(train_loader)} (batch_size={BATCH_SIZE})")
    print(f"  测试集 batches: {len(test_loader)} (batch_size={BATCH_SIZE})")
    print()

    return train_loader, test_loader, input_size


def init_weights(model: nn.Module) -> None:
    """
    Xavier Uniform 权重初始化。

    策略:
        - 所有 *.weight (Linear / GRU 的 weight_ih / weight_hh): Xavier Uniform
        - 所有 *.bias:                                         零
        - LayerNorm 的 weight: 保持默认 (全 1), 否则会破坏 LayerNorm 数学语义
    """
    for name, param in model.named_parameters():
        if "weight" in name:
            # LayerNorm 的 weight 必须保持 1, 否则归一化会失效
            if param.dim() < 2:
                # 1D 权重 (如 LayerNorm.weight, GRU.bias_ih_*) 跳过 Xavier
                continue
            nn.init.xavier_uniform_(param.data)
        elif "bias" in name:
            nn.init.zeros_(param.data)


# ============================================================================
# 早停机制
# ============================================================================
class EarlyStopping:
    """与现有管线一致的 Early Stopping 机制。"""

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
            self.best_state = {
                k: v.clone().cpu() for k, v in model.state_dict().items()
            }
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                print(f"\n  ⏹ 早停触发! 连续 {self.patience} 个 Epoch 验证损失无改善。")
                print(f"    最佳 Epoch: {self.best_epoch}, "
                      f"最佳 Test Loss: {self.best_loss:.8f}")


# ============================================================================
# 训练 / 验证
# ============================================================================
def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total, n = 0.0, 0
    for x_feat, x_intent, y in train_loader:
        x_feat = x_feat.to(device)
        x_intent = x_intent.to(device)
        y = y.to(device)

        pred = model(x_feat, x_intent)
        loss = criterion(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total += loss.item()
        n += 1
    return total / max(n, 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    test_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.eval()
    total, n = 0.0, 0
    for x_feat, x_intent, y in test_loader:
        x_feat = x_feat.to(device)
        x_intent = x_intent.to(device)
        y = y.to(device)

        pred = model(x_feat, x_intent)
        loss = criterion(pred, y)

        total += loss.item()
        n += 1
    return total / max(n, 1)


def train(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
) -> None:
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    early_stopping = EarlyStopping(patience=PATIENCE)

    print("=" * 60)
    print("  开始训练 (Attention-Bi-GRU)")
    print("=" * 60)
    print(f"  损失函数:      MSELoss")
    print(f"  优化器:        Adam (lr={LEARNING_RATE})")
    print(f"  最大 Epochs:   {MAX_EPOCHS}")
    print(f"  早停 Patience: {PATIENCE}")
    print(f"  Batch Size:    {BATCH_SIZE}")
    print("-" * 60)
    print(f"  {'Epoch':>6}  |  {'Train Loss':>14}  |  {'Test Loss':>14}  |  备注")
    print("-" * 60)

    start = time.time()
    last_epoch = 0
    for epoch in range(1, MAX_EPOCHS + 1):
        last_epoch = epoch
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
    print(f"  完成 Epoch 数:   {last_epoch}")
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
    print("  UAV 轨迹预测 — Attention-Bi-GRU (意图增强) 训练")
    print("▓" * 60 + "\n")

    device = get_device()
    print()

    train_loader, test_loader, input_size = load_data()

    print("=" * 60)
    print("  模型初始化 (Attention-Bi-GRU)")
    print("=" * 60)

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
    init_weights(model)
    print(f"  ✅ Xavier Uniform 权重初始化完成 (dropout = {DROPOUT})")
    print(f"  编码器层数 N_enc = {N_ENC_LAYERS}, 解码器层数 N_dec = {N_DEC_LAYERS}")
    if N_DECODE_STEPS == 1:
        print(f"  解码步数 T_dec = 1 (单步预测; 注意自回归循环未触发, 模型动态特性退化)")
    else:
        print(f"  解码步数 T_dec = {N_DECODE_STEPS} "
              f"(自回归 Seq2Seq, 训练目标 Y shape = (B, {N_DECODE_STEPS}, 3); "
              f"评估时 compare_all 仅取首步与 pure_gru 对齐)")

    model.to(device)
    print(f"  ✅ 模型已移至 {device}")

    total = sum(p.numel() for p in model.parameters())
    print(f"  总参数量: {total:,}")
    print(f"\n{model}\n")

    train(model, train_loader, test_loader, device)

    print("\n" + "▓" * 60)
    print("  Attention-Bi-GRU 训练完成!")
    print("▓" * 60 + "\n")


if __name__ == "__main__":
    main()
