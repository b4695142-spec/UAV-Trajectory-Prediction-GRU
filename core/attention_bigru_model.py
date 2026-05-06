"""
==============================================================================
UAV 轨迹预测 — Attention-Bi-GRU 意图增强模型 (优化版)
==============================================================================
在原始 Seq2Seq 架构基础上优化为编码器直出架构，解决小数据集上的严重过拟合问题。

整体架构 (编码器直出):
    输入 x ∈ R^(B × L × in_size),  P_S ∈ R^(B × L × n_intent)
        │
        ▼
    Bi-GRU 编码器 (2 层, hidden_size, bidirectional=True)
        │
        ▼  gru_out ∈ R^(B × L × d_model),  d_model = 2 * hidden_size
    ┌──── 缩放点积注意力 ─────────────────────────────┐
    │   Q = W_Q · gru_out[:, -1, :]   (最后时间步)     │
    │   K = W_K · gru_out             (全部时间步)      │
    │   V = W_V · gru_out             (全部时间步)      │
    │   α = softmax(Q·K^T / √d_K) · V                  │
    └────────────────────────────────────────────────┘
        │
        ▼  α ∈ R^(B × d_attn)
    ┌──── 意图融合 (论文公式 31-32) ──────────────────┐
    │   z_S = GeLU(W_P · P_S_last)                     │
    │   h_fused = proj(Concat(gru_last, α, z_S))       │
    └────────────────────────────────────────────────┘
        │
        ▼
    输出层: Linear(d_fused → output_size)
        │
        ▼
    输出 y ∈ R^(B × output_size)

★ 与原始 Seq2Seq 版本的关键差异:
    1. 移除了解码器 (消除单步预测下 Bi-GRU 解码的冗余)
    2. 移除了自回归循环 (T_dec=1 时无需自回归)
    3. 使用标准多层 Bi-GRU 替代每层独立的 Bi-GRU + AddNorm
    4. 注意力 Q 直接从编码器最后时间步生成 (无需解码器隐藏状态)
    5. 意图融合采用拼接+投影方式 (保留论文公式 31-32 的 GeLU 变换)
    6. 参数量大幅减少，有效抑制小数据集上的过拟合
==============================================================================
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k: int, dropout: float = 0.1):
        super().__init__()
        self.scale = 1.0 / math.sqrt(d_k)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
    ) -> torch.Tensor:
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        weights = F.softmax(scores, dim=-1)
        weights = self.dropout(weights)
        context = torch.matmul(weights, V)
        return context.squeeze(1)


class AttentionBiGRU(nn.Module):
    """
    编码器直出的 Attention-Bi-GRU 意图增强轨迹预测模型。

    架构: Bi-GRU 编码器 → 缩放点积注意力 → 意图融合 → 输出投影

    参数:
        input_size:    结构性特征维度 (位置 + RF 筛选列, 不含意图概率)
        hidden_size:   Bi-GRU 隐藏维度 (单向); d_model = 2 * hidden_size
        output_size:   输出维度 (lat, lon, alt → 3)
        n_enc_layers:  Bi-GRU 编码器层数
        n_dec_layers:  保留接口兼容, 内部不使用
        d_ff:          保留接口兼容, 内部不使用
        n_intent:      意图概率维度 (4 类机动)
        n_decode_steps:保留接口兼容, 内部固定为 1
        dropout:       Dropout 概率
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        output_size: int = 3,
        n_enc_layers: int = 2,
        n_dec_layers: int = 2,
        d_ff: int = 256,
        n_intent: int = 4,
        n_decode_steps: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_enc_layers = n_enc_layers
        self.n_dec_layers = n_dec_layers
        self.n_intent = n_intent
        self.n_decode_steps = n_decode_steps
        self.dropout = float(dropout)

        self.d_model = 2 * hidden_size

        self.bigru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=n_enc_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if n_enc_layers > 1 else 0.0,
        )

        d_attn = hidden_size
        self.d_v = d_attn
        self.W_Q = nn.Linear(self.d_model, d_attn)
        self.W_K = nn.Linear(self.d_model, d_attn)
        self.W_V = nn.Linear(self.d_model, d_attn)
        self.attention = ScaledDotProductAttention(d_attn, dropout)

        self.intent_proj = nn.Linear(n_intent, d_attn)

        d_fused = self.d_model + d_attn + d_attn
        self.fusion_proj = nn.Sequential(
            nn.Linear(d_fused, self.d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.fc = nn.Linear(self.d_model, output_size)

    def forward(
        self,
        x: torch.Tensor,
        intent_probs: torch.Tensor,
    ) -> torch.Tensor:
        """
        参数:
            x:            (batch, seq_len, input_size) — 结构性特征序列
            intent_probs: (batch, seq_len, n_intent)   — SVM 概率序列

        返回:
            (batch, output_size)
        """
        gru_out, _ = self.bigru(x)

        last_hidden = gru_out[:, -1, :]

        Q = self.W_Q(last_hidden).unsqueeze(1)
        K = self.W_K(gru_out)
        V = self.W_V(gru_out)
        alpha = self.attention(Q, K, V)

        intent_last = intent_probs[:, -1, :]
        z_S = F.gelu(self.intent_proj(intent_last))

        fused = torch.cat([last_hidden, alpha, z_S], dim=-1)
        fused_out = self.fusion_proj(fused)

        out = self.fc(fused_out)

        if self.n_decode_steps == 1:
            return out
        return out.unsqueeze(1)
