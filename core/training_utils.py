"""
==============================================================================
共享训练基础设施模块
==============================================================================
提供统一的训练工具函数和类，供所有训练管线使用：
    - EarlyStopping: 早停机制
    - train_one_epoch: 单轮训练
    - evaluate: 模型评估
    - get_device: 设备选择
    - init_weights: 权重初始化
==============================================================================
"""

from __future__ import annotations

import os
import sys

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from matplotlib.font_manager import FontManager
from torch.utils.data import DataLoader
from typing import Optional, Dict, Any, List


def get_device() -> torch.device:
    """选择最优计算设备: CUDA GPU > CPU"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"  使用设备: {torch.cuda.get_device_name(0)} (CUDA)")
    else:
        device = torch.device("cpu")
        print(f"  使用设备: CPU")
    return device


def init_weights(model: nn.Module) -> None:
    """
    Glorot (Xavier Uniform) 权重初始化。

    策略:
        - 所有 *.weight (Linear / GRU 的 weight_ih / weight_hh): Xavier Uniform
        - 所有 *.bias: 零
        - LayerNorm 的 weight: 保持默认 (全 1), 否则会破坏 LayerNorm 数学语义
    """
    for name, param in model.named_parameters():
        if "weight" in name:
            # LayerNorm 的 weight 必须保持 1, 否则归一化会失效
            if param.dim() < 2:
                continue
            nn.init.xavier_uniform_(param.data)
        elif "bias" in name:
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
        self.best_state: Optional[Dict[str, torch.Tensor]] = None
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
                print(
                    f"\n  ⏹ 早停触发! 连续 {self.patience} 个 Epoch 验证损失无改善。"
                )
                print(f"    最佳 Epoch: {self.best_epoch}, 最佳 Test Loss: {self.best_loss:.8f}")


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """
    执行一个 Epoch 的训练。

    参数:
        model: 训练模型
        train_loader: 训练数据加载器
        criterion: 损失函数
        optimizer: 优化器
        device: 计算设备

    返回:
        avg_train_loss: 该 Epoch 的平均训练损失
    """
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch_data in train_loader:
        # 支持两种数据格式: (X, Y) 和 (X_feat, X_intent, Y)
        if len(batch_data) == 2:
            X_batch, Y_batch = batch_data
            X_batch = X_batch.to(device)
            Y_batch = Y_batch.to(device)
            pred = model(X_batch)
        elif len(batch_data) == 3:
            X_feat, X_intent, Y_batch = batch_data
            X_feat = X_feat.to(device)
            X_intent = X_intent.to(device)
            Y_batch = Y_batch.to(device)
            pred = model(X_feat, X_intent)
        else:
            raise ValueError(f"不支持的 batch 数据格式: {len(batch_data)}")

        loss = criterion(pred, Y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


def _setup_chinese_font():
    font_candidates = {
        'win32': ['SimHei', 'Microsoft YaHei', 'SimSun'],
        'linux': ['WenQuanYi Micro Hei', 'WenQuanYi Zen Hei', 'Noto Sans CJK SC', 'DejaVu Sans'],
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


def plot_loss_curves(
    train_losses: List[float],
    test_losses: List[float],
    best_epoch: int,
    save_path: str,
    title: str = "训练损失曲线",
) -> None:
    _setup_chinese_font()

    epochs = list(range(1, len(train_losses) + 1))

    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)

    ax.plot(epochs, train_losses, color='#1f77b4', linewidth=1.8, label='训练损失 (Train Loss)', alpha=0.9)
    ax.plot(epochs, test_losses, color='#d62728', linewidth=1.8, label='验证损失 (Val Loss)', alpha=0.9)

    ax.axvline(x=best_epoch, color='#2ca02c', linestyle='--', linewidth=1.5, alpha=0.8, label=f'最佳 Epoch ({best_epoch})')

    best_val_loss = test_losses[best_epoch - 1]
    ax.scatter([best_epoch], [best_val_loss], color='#2ca02c', s=100, zorder=5, edgecolors='white', linewidths=1.5)
    ax.annotate(
        f'Best: {best_val_loss:.6f}',
        xy=(best_epoch, best_val_loss),
        xytext=(best_epoch + len(epochs) * 0.05, best_val_loss * 1.1),
        fontsize=9,
        color='#2ca02c',
        arrowprops=dict(arrowstyle='->', color='#2ca02c', lw=1.2),
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#2ca02c', alpha=0.9),
    )

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('MSE Loss', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold', pad=12)
    ax.legend(loc='upper right', fontsize=10, framealpha=0.9, edgecolor='#cccccc')
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.set_xlim(1, len(epochs))
    ax.tick_params(labelsize=10)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  ✅ 损失曲线已保存至: {save_path}")


@torch.no_grad()
def evaluate(
    model: nn.Module,
    test_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """
    在测试集上评估模型。

    参数:
        model: 待评估模型
        test_loader: 测试数据加载器
        criterion: 损失函数
        device: 计算设备

    返回:
        avg_test_loss: 平均测试损失
    """
    model.eval()
    total_loss = 0.0
    n_batches = 0

    for batch_data in test_loader:
        # 支持两种数据格式: (X, Y) 和 (X_feat, X_intent, Y)
        if len(batch_data) == 2:
            X_batch, Y_batch = batch_data
            X_batch = X_batch.to(device)
            Y_batch = Y_batch.to(device)
            pred = model(X_batch)
        elif len(batch_data) == 3:
            X_feat, X_intent, Y_batch = batch_data
            X_feat = X_feat.to(device)
            X_intent = X_intent.to(device)
            Y_batch = Y_batch.to(device)
            pred = model(X_feat, X_intent)
        else:
            raise ValueError(f"不支持的 batch 数据格式: {len(batch_data)}")

        loss = criterion(pred, Y_batch)
        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)
