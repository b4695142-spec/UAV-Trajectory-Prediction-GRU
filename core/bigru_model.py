"""
==============================================================================
UAV 轨迹预测 Bi-GRU 模型
==============================================================================
基于 PyTorch 实现的双向 GRU (Bidirectional GRU) 网络，用于预测无人机
在当前 t 时刻的三维坐标 (纬度, 经度, 海拔)。

论文: Yoon et al., "GRU-based deep learning framework for real-time,
      accurate, and scalable UAV trajectory prediction",
      Drones 2025, 9(2): 142-168, Section 2.4.1

原文描述:
    "The Bi-GRU architecture ultimately integrates bidirectional processing
     with the GRU's streamlined gating mechanism. This model employs two
     bidirectional GRU layers, each with 64 hidden units per direction,
     to forecast latitude, longitude, and altitude via a fully linked layer
     with three output neurons."

网络结构:
    输入 (batch_size, Look_Back, 3)
        ↓
    Bi-GRU × 2 层 (hidden_size=64 per direction)
        ↓  ← 取最后一个时间步的双向隐藏状态拼接
    Linear (128 → 3)    [128 = 2 × hidden_size]
        ↓  ← 无激活函数，线性投影
    输出 (batch_size, 3)
==============================================================================
"""

import torch
import torch.nn as nn


class UAVTrajectoryBiGRU(nn.Module):
    """
    双向 GRU (Bi-GRU) 轨迹预测模型。

    参数:
        input_size  (int):   输入特征维度，默认 3 (纬度, 经度, 海拔)
        hidden_size (int):   Bi-GRU 每个方向的隐藏层维度，默认 64
        num_layers  (int):   Bi-GRU 堆叠层数，默认 2
        output_size (int):   输出维度，默认 3 (预测的纬度, 经度, 海拔)
        dropout     (float): GRU 层间 dropout 概率，默认 0.0 (禁用)。
                             仅当 num_layers >= 2 时生效 (PyTorch 规范)。
    """

    def __init__(
        self,
        input_size: int = 3,
        hidden_size: int = 64,
        num_layers: int = 2,
        output_size: int = 3,
        dropout: float = 0.0,
    ):
        super(UAVTrajectoryBiGRU, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.dropout = float(dropout)

        effective_dropout = self.dropout if self.num_layers > 1 else 0.0
        self.bigru = nn.GRU(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=effective_dropout,
        )

        self.fc = nn.Linear(
            in_features=self.hidden_size * 2,
            out_features=self.output_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        h_0 = torch.zeros(
            self.num_layers * 2, batch_size, self.hidden_size,
            device=x.device, dtype=x.dtype
        )

        gru_out, h_n = self.bigru(x, h_0)
        last_hidden = gru_out[:, -1, :]
        out = self.fc(last_hidden)

        return out
