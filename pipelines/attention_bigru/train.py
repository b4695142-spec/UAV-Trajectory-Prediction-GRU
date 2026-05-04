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
    LOSS_CURVE_PATH,
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
from core.training_utils import (
    EarlyStopping,
    evaluate,
    get_device,
    init_weights,
    plot_loss_curves,
    train_one_epoch,
)


# ============================================================================
# 自定义 Dataset — 拆分增广 X 中的结构性特征 / 意图概率
# ============================================================================
class AttentionBiGRUDataset(Dataset):
    """
    每个样本返回 (x_feat, x_intent, y) 三元组:
        x_feat   : (look_back, input_size)   结构性特征 (位置 + RF 筛选列)
        x_intent : (look_back, n_intent)     SVM 概率向量序列
        y        : (output_size,)            目标 (lat, lon, alt, StandardScaler)
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

    # ── 拆分: 最后 N_INTENT 列为 SVM 概率, 其余为结构性特征 ──
    input_size = X_train.shape[-1] - N_INTENT
    X_feat_train = X_train[:, :, :-N_INTENT]
    X_intent_train = X_train[:, :, -N_INTENT:]
    X_feat_test = X_test[:, :, :-N_INTENT]
    X_intent_test = X_test[:, :, -N_INTENT:]

    print(f"  动态输入维度 input_size = {input_size}  (结构性特征)")
    print(f"  意图概率维度 n_intent  = {N_INTENT}     (SVM 4 维概率)")

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

        train_losses.append(train_loss)
        test_losses.append(test_loss)

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

    plot_loss_curves(
        train_losses=train_losses,
        test_losses=test_losses,
        best_epoch=early_stopping.best_epoch,
        save_path=LOSS_CURVE_PATH,
        title="Attention-Bi-GRU — 训练损失曲线",
    )


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
    print(f"  解码步数 T_dec = {N_DECODE_STEPS} (与 pure_gru 对齐)")

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
