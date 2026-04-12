"""
==============================================================================
UAV 轨迹预测 GRU 模型 — 训练与验证脚本
==============================================================================
功能:
    1. 加载预处理后的滑动窗口数据 (X_train, Y_train, X_test, Y_test)
    2. 使用 Glorot (Xavier) 初始化模型权重
    3. 以 MSE 损失 + Adam 优化器训练 GRU 模型
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
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# 从同目录导入模型类
from gru_model import UAVTrajectoryGRU


# ============================================================================
# 配置参数
# ============================================================================
# 数据路径
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "processed_data")
X_TRAIN_PATH = os.path.join(DATA_DIR, "X_train.npy")
Y_TRAIN_PATH = os.path.join(DATA_DIR, "Y_train.npy")
X_TEST_PATH = os.path.join(DATA_DIR, "X_test.npy")
Y_TEST_PATH = os.path.join(DATA_DIR, "Y_test.npy")

# 模型保存路径
MODEL_SAVE_PATH = os.path.join(DATA_DIR, "best_gru_model.pth")

# 训练超参数
BATCH_SIZE = 70          # 批次大小 (严格按文献要求)
LEARNING_RATE = 1e-3     # Adam 学习率
MAX_EPOCHS = 500         # 最大训练轮数
PATIENCE = 15            # 早停耐心值: 连续 15 个 Epoch 无改善则停止

# 模型超参数
INPUT_SIZE = 3           # 输入特征 (lat, lon, alt)
HIDDEN_SIZE = 64         # GRU 隐藏层维度
NUM_LAYERS = 2           # GRU 层数
OUTPUT_SIZE = 3          # 输出维度 (lat, lon, alt)


# ============================================================================
# 工具函数
# ============================================================================
def get_device() -> torch.device:
    """选择最优计算设备: CUDA GPU > CPU"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"  使用设备: {torch.cuda.get_device_name(0)} (CUDA)")
    else:
        device = torch.device("cpu")
        print(f"  使用设备: CPU")
    return device


def load_data() -> tuple:
    """
    加载预处理后的 .npy 数据文件，转换为 PyTorch DataLoader。

    返回:
        train_loader: 训练集 DataLoader (batch_size=70)
        test_loader:  测试集 DataLoader (batch_size=70)
    """
    print("=" * 60)
    print("  数据加载")
    print("=" * 60)

    # 读取 numpy 数组
    X_train = np.load(X_TRAIN_PATH)
    Y_train = np.load(Y_TRAIN_PATH)
    X_test = np.load(X_TEST_PATH)
    Y_test = np.load(Y_TEST_PATH)

    print(f"  X_train: {X_train.shape}  Y_train: {Y_train.shape}")
    print(f"  X_test:  {X_test.shape}   Y_test:  {Y_test.shape}")

    # 转换为 PyTorch 张量 (float32)
    # 原始数据为 float64，需要转为 float32 以匹配模型参数精度
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    Y_train_t = torch.tensor(Y_train, dtype=torch.float32)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    Y_test_t = torch.tensor(Y_test, dtype=torch.float32)

    # 构建 TensorDataset
    train_dataset = TensorDataset(X_train_t, Y_train_t)
    test_dataset = TensorDataset(X_test_t, Y_test_t)

    # 构建 DataLoader
    # 训练集: shuffle=True (每个 epoch 打乱 batch 顺序，但不影响序列内部时序)
    # 测试集: shuffle=False (保持原始顺序，用于验证)
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


def init_weights(model: nn.Module) -> None:
    """
    Glorot (Xavier Uniform) 权重初始化。

    策略:
        - GRU 权重矩阵 (weight_ih_*, weight_hh_*): Xavier Uniform
        - Linear 权重 (fc.weight): Xavier Uniform
        - 所有偏置项 (bias): 初始化为 0
    """
    for name, param in model.named_parameters():
        if "weight" in name:
            # 对所有权重矩阵应用 Glorot/Xavier 均匀初始化
            # GRU 的 weight_ih / weight_hh 和 Linear 的 weight 均适用
            nn.init.xavier_uniform_(param.data)
        elif "bias" in name:
            # 偏置项全部置零
            nn.init.zeros_(param.data)


class EarlyStopping:
    """
    早停机制: 监控验证损失，在指定 patience 内无改善时终止训练。

    属性:
        patience    (int):   连续无改善的最大容忍 Epoch 数
        best_loss   (float): 历史最佳验证损失
        best_epoch  (int):   最佳验证损失对应的 Epoch
        counter     (int):   当前连续无改善的计数器
        best_state  (dict):  最佳模型的权重副本
        early_stop  (bool):  是否触发早停
    """

    def __init__(self, patience: int = 15):
        self.patience = patience
        self.best_loss = float("inf")
        self.best_epoch = 0
        self.counter = 0
        self.best_state = None
        self.early_stop = False

    def __call__(self, val_loss: float, model: nn.Module, epoch: int):
        """
        每个 Epoch 结束时调用，判断是否触发早停。

        参数:
            val_loss: 当前 Epoch 的验证损失
            model:    当前模型 (用于保存最佳权重)
            epoch:    当前 Epoch 编号
        """
        if val_loss < self.best_loss:
            # 验证损失有改善 → 更新最佳记录，重置计数器
            self.best_loss = val_loss
            self.best_epoch = epoch
            self.counter = 0
            # 深拷贝当前模型权重 (移到 CPU 以节省 GPU 显存)
            self.best_state = {
                k: v.clone().cpu() for k, v in model.state_dict().items()
            }
        else:
            # 无改善 → 计数器 +1
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                print(
                    f"\n  ⏹ 早停触发! 连续 {self.patience} 个 Epoch 验证损失无改善。"
                )
                print(f"    最佳 Epoch: {self.best_epoch}, 最佳 Test Loss: {self.best_loss:.8f}")


# ============================================================================
# 训练与验证
# ============================================================================
def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """
    执行一个 Epoch 的训练。

    返回:
        avg_train_loss: 该 Epoch 的平均训练损失
    """
    model.train()
    total_loss = 0.0
    n_batches = 0

    for X_batch, Y_batch in train_loader:
        # 将数据移到目标设备
        X_batch = X_batch.to(device)
        Y_batch = Y_batch.to(device)

        # 前向传播
        predictions = model(X_batch)           # (batch_size, 3)
        loss = criterion(predictions, Y_batch)

        # 反向传播 + 参数更新
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    avg_train_loss = total_loss / n_batches
    return avg_train_loss


@torch.no_grad()
def evaluate(
    model: nn.Module,
    test_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """
    在测试集上评估模型。

    返回:
        avg_test_loss: 平均测试损失
    """
    model.eval()
    total_loss = 0.0
    n_batches = 0

    for X_batch, Y_batch in test_loader:
        X_batch = X_batch.to(device)
        Y_batch = Y_batch.to(device)

        predictions = model(X_batch)
        loss = criterion(predictions, Y_batch)

        total_loss += loss.item()
        n_batches += 1

    avg_test_loss = total_loss / n_batches
    return avg_test_loss


def train(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
) -> None:
    """
    完整训练流程: 训练循环 + Early Stopping + 最佳模型保存。
    """
    # ------------------------------------------------------------------
    # 损失函数与优化器
    # ------------------------------------------------------------------
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # ------------------------------------------------------------------
    # 早停机制
    # ------------------------------------------------------------------
    early_stopping = EarlyStopping(patience=PATIENCE)

    # ------------------------------------------------------------------
    # 训练主循环
    # ------------------------------------------------------------------
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
        # 训练一个 Epoch
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)

        # 在测试集上评估
        test_loss = evaluate(model, test_loader, criterion, device)

        # 早停检查
        early_stopping(test_loss, model, epoch)

        # 打印进度
        remark = ""
        if test_loss <= early_stopping.best_loss:
            remark = "★ Best"
        elif early_stopping.counter > 0:
            remark = f"patience {early_stopping.counter}/{PATIENCE}"

        print(
            f"  {epoch:>6}  |  {train_loss:>14.8f}  |  {test_loss:>14.8f}  |  {remark}"
        )

        # 触发早停则退出循环
        if early_stopping.early_stop:
            break

    elapsed = time.time() - start_time

    # ------------------------------------------------------------------
    # 训练结束: 加载并保存最佳模型
    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    print("  训练结束")
    print("=" * 60)
    print(f"  完成 Epoch 数:    {epoch}")
    print(f"  最佳 Epoch:       {early_stopping.best_epoch}")
    print(f"  最佳 Test Loss:   {early_stopping.best_loss:.8f}")
    print(f"  总训练时间:       {elapsed:.2f} 秒 ({elapsed / 60:.2f} 分钟)")

    # 加载最佳模型权重
    if early_stopping.best_state is not None:
        model.load_state_dict(early_stopping.best_state)
        print(f"\n  ✅ 已加载最佳模型权重 (Epoch {early_stopping.best_epoch})")

    # 保存到磁盘
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"  ✅ 最佳模型已保存至: {MODEL_SAVE_PATH}")


# ============================================================================
# 主函数
# ============================================================================
def main():
    print("\n" + "▓" * 60)
    print("  UAV 轨迹预测 GRU 模型 — 训练与验证")
    print("▓" * 60 + "\n")

    # ------------------------------------------------------------------
    # 选择设备
    # ------------------------------------------------------------------
    device = get_device()
    print()

    # ------------------------------------------------------------------
    # 加载数据
    # ------------------------------------------------------------------
    train_loader, test_loader = load_data()

    # ------------------------------------------------------------------
    # 初始化模型
    # ------------------------------------------------------------------
    print("=" * 60)
    print("  模型初始化")
    print("=" * 60)

    model = UAVTrajectoryGRU(
        input_size=INPUT_SIZE,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        output_size=OUTPUT_SIZE,
    )

    # 应用 Glorot (Xavier Uniform) 权重初始化
    init_weights(model)
    print("  ✅ Xavier Uniform 权重初始化完成")

    # 将模型移到目标设备
    model.to(device)
    print(f"  ✅ 模型已移至 {device}")

    # 打印模型结构
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  总参数量: {total_params:,}")
    print(f"\n{model}\n")

    # ------------------------------------------------------------------
    # 开始训练
    # ------------------------------------------------------------------
    train(model, train_loader, test_loader, device)

    print("\n" + "▓" * 60)
    print("  全部完成!")
    print("▓" * 60 + "\n")


# ============================================================================
# 入口
# ============================================================================
if __name__ == "__main__":
    main()
