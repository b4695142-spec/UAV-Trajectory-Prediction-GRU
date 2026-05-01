"""
==============================================================================
UAV 轨迹预测 — Attention-Bi-GRU 意图增强模型
==============================================================================
严格遵循论文 *"Research on trajectory prediction algorithm based on unmanned
aerial vehicles behavioral intentions"* (Drones 2025, 9, 640) 的 Section 4.2
公式 (24)-(32) 与 Figure 9/10。

整体架构:
    输入 x ∈ R^(B × L × in_size),  P_S ∈ R^(B × L × n_intent)
        │
        ▼
    ┌──── 编码器 (N_enc 层, 每层双残差 Add & Norm) ────┐
    │   FFN + Add & Norm → Bi-GRU + Add & Norm        │
    │   (论文 Figure 10)                                │
    └────────────────────────────────────────────────┘
        │
        ▼  H_enc ∈ R^(B × L × d_model),  enc_h_final
    ┌──── K, V 投影 (仅一次) ────────────────────────┐
    │   K = W_K · H_enc   (28)                        │
    │   V = W_V · H_enc   (29)                        │
    └────────────────────────────────────────────────┘
        │
        ▼
    ┌──── 解码器初始化 (映射编码器最终隐藏状态) ────┐
    │   dec_h0 = proj(enc_final_hidden)                │
    └────────────────────────────────────────────────┘
        │
        ▼
    ┌──── 自回归解码循环 (t = 1 .. T_dec) ──────────┐
    │   ★ 动态 Q_t = W_Q · h_dec_t   (27)              │
    │   ★ 动态 α_t = Attention(Q_t, K, V)   (30)       │
    │                                                 │
    │   for layer in 解码器各层:                     │
    │       z_S = GeLU(W_P · P_S)  (31)               │
    │       h_fused = proj(Concat(α_t, z_S))  (32)    │
    │       条件注入: x ← x + h_fused                 │
    │       FFN + Add & Norm → Bi-GRU + Add & Norm    │
    │                                                 │
    │   y_t = fc(dec_out_t)                           │
    │   if t < T_dec: dec_input_{t+1} ← output_proj(y_t)
    └────────────────────────────────────────────────┘
        │
        ▼
    输出 y ∈ R^(B × T_dec × out_size)  [若 T_dec=1 则压缩为 (B × out_size)]

★ 关键修正记录 (vs 初版方案):
    1. Q 在每步解码时从解码器当前隐藏状态动态生成 (论文公式 27)，
       而非在循环前用 pre_dec_bigru 生成静态 Q
    2. α 在每步解码时随 Q_t 动态重新计算，而非静态共享
    3. 编码器每层包含双残差 Add & Norm (论文 Figure 10)
    4. 解码器每层与编码器架构一致 (论文 Section 4.2.3)
    5. 每层解码器独立接收 α_t 和 P_S (论文 Section 4.2.3)
    6. 支持自回归多步解码 (T_dec 可配置)
==============================================================================
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# 子模块 1: 前馈网络 (论文公式 24)
# ============================================================================
class FeedForward(nn.Module):
    """
    两层前馈网络: ReLU(W1·x + b1)·W2 + b2  (论文公式 24)

    输入维度与输出维度均为 d_model，便于残差连接。
    编码器与解码器共用此结构 (论文 Section 4.2.3 指出
    "Each layer of the decoder is consistent with the encoder")。
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.2):
        super().__init__()
        self.layer1 = nn.Linear(d_model, d_ff)
        self.layer2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.layer2(F.relu(self.layer1(x))))


# ============================================================================
# 子模块 2: 编码器单层 (论文 Figure 10)
# ============================================================================
class EncoderLayer(nn.Module):
    """
    单层编码器: FFN + Add & Norm → Bi-GRU + Add & Norm  (论文 Figure 10)

    论文 Figure 10 明确展示每层编码器包含两个 Add & Norm 模块:
        1. 第一个位于 FFN 之后: norm1_out = LayerNorm(x + ffn(x))
        2. 第二个位于 Bi-GRU 之后: out = LayerNorm(norm1_out + proj(bigru(norm1_out)))

    Bi-GRU 输出维度为 2*hidden_size，需通过 bigru_proj 线性映射回 d_model
    才能与残差分支 (norm1_out) 相加。
    """

    def __init__(
        self,
        d_model: int,
        hidden_size: int,
        d_ff: int,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)

        self.bigru = nn.GRU(
            input_size=d_model,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.bigru_proj = nn.Linear(2 * hidden_size, d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, h_0: torch.Tensor | None = None):
        # ── 第一段: FFN + Add & Norm ──
        ffn_out = self.ffn(x)
        norm1_out = self.norm1(x + ffn_out)

        # ── 第二段: Bi-GRU + Add & Norm ──
        bigru_out, h_n = self.bigru(norm1_out, h_0)
        bigru_proj = self.bigru_proj(bigru_out)
        out = self.norm2(norm1_out + self.dropout_layer(bigru_proj))

        return out, h_n


# ============================================================================
# 子模块 3: 解码器单层 (与编码器架构一致 + 每层意图融合)
# ============================================================================
class DecoderLayer(nn.Module):
    """
    单层解码器: 意图融合 + FFN + Add & Norm → Bi-GRU + Add & Norm

    ★ 论文 Section 4.2.3:
        "Each layer of the decoder is consistent with the encoder
         in architecture design."
    ★ 论文 Section 4.2.3:
        "Each layer of the decoder receives two key inputs: α and P_S."

    每层解码器的处理流程:
        1. 意图融合 (每层独立参数, 每步重新计算):
           z_S       = GeLU(W_P · P_S + b_P)         (31)
           h_fused   = proj(Concat(α_t, z_S))         (32)
           x_cond    = x + h_fused                     ← 条件注入 (残差式)
        2. FFN + Add & Norm (与编码器一致)
        3. Bi-GRU + Add & Norm (与编码器一致, 隐藏状态自回归更新)

    ★ 每层有独立的 intent_proj (W_P, b_P), 每步重新生成 z_S
    ★ α_t 在每步解码时由外层动态重新计算后传入
    """

    def __init__(
        self,
        d_model: int,
        hidden_size: int,
        d_ff: int,
        d_v: int,
        n_intent: int,
        dropout: float = 0.2,
    ):
        super().__init__()
        # ── 意图融合 (每层独立参数) ──
        self.intent_proj = nn.Linear(n_intent, d_v)
        self.fusion_proj = nn.Linear(2 * d_v, d_model)

        # ── FFN + Add & Norm (与编码器一致) ──
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # ── Bi-GRU + Add & Norm (与编码器一致) ──
        self.bigru = nn.GRU(
            input_size=d_model,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.bigru_proj = nn.Linear(2 * hidden_size, d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout_layer = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        alpha: torch.Tensor,
        intent_probs: torch.Tensor,
        h_0: torch.Tensor | None = None,
    ):
        """
        参数:
            x:            (batch, seq_len, d_model)   — 上一层输出
            alpha:        (batch, d_v)                — 当前解码步的注意力上下文 (动态)
            intent_probs: (batch, n_intent)           — SVM 概率向量
            h_0:          (2, batch, hidden_size)     — Bi-GRU 初始隐藏状态

        返回:
            out:          (batch, seq_len, d_model)
            h_n:          (2, batch, hidden_size)     — 更新后的 Bi-GRU 隐藏状态
        """
        # ── 意图融合: GeLU(W_P · P_S) → Concat(α, z_S) → proj ──
        z_S = F.gelu(self.intent_proj(intent_probs))     # (batch, d_v)
        h_fused = torch.cat([alpha, z_S], dim=-1)        # (batch, 2*d_v)
        h_proj = self.fusion_proj(h_fused)               # (batch, d_model)
        # 残差式条件注入: 在 seq_len 维度广播
        x_conditioned = x + h_proj.unsqueeze(1)

        # ── 第一段: FFN + Add & Norm ──
        ffn_out = self.ffn(x_conditioned)
        norm1_out = self.norm1(x_conditioned + ffn_out)

        # ── 第二段: Bi-GRU + Add & Norm ──
        bigru_out, h_n = self.bigru(norm1_out, h_0)
        bigru_proj = self.bigru_proj(bigru_out)
        out = self.norm2(norm1_out + self.dropout_layer(bigru_proj))

        return out, h_n


# ============================================================================
# 子模块 4: 缩放点积注意力 (论文公式 30)
# ============================================================================
class ScaledDotProductAttention(nn.Module):
    """
    缩放点积注意力: α = softmax(Q·K^T / √d_K) · V   (论文公式 30)

    输入:
        Q: (batch, 1,       d_attn)   — 解码器当前隐藏状态投影
        K: (batch, seq_len, d_attn)   — 编码器全部时间步的键
        V: (batch, seq_len, d_attn)   — 编码器全部时间步的值

    输出:
        context: (batch, d_attn)       — 当前解码步的注意力上下文向量
    """

    def __init__(self, d_k: int, dropout: float = 0.2):
        super().__init__()
        self.scale = 1.0 / math.sqrt(d_k)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
    ) -> torch.Tensor:
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale   # (B, 1, L)
        weights = F.softmax(scores, dim=-1)
        weights = self.dropout(weights)
        context = torch.matmul(weights, V)                           # (B, 1, d_attn)
        return context.squeeze(1)                                    # (B, d_attn)


# ============================================================================
# 主模型: AttentionBiGRU
# ============================================================================
class AttentionBiGRU(nn.Module):
    """
    完整的 Attention-Bi-GRU 意图增强轨迹预测模型。

    严格遵循论文 Section 4.2 的 Seq2Seq 架构:
        编码器 (FFN + Bi-GRU + 双残差 Add & Norm) × N_enc 层
            → 生成 K, V (仅计算一次)
        解码器初始化 (编码器最终隐藏状态映射)
        自回归解码循环 (t = 1 .. T_dec):
            动态 Q_t = W_Q(h_t^d)  ← 每步从解码器隐藏状态重新生成
            动态 α_t = Attention(Q_t, K, V)  ← 每步重新计算
            解码器 (FFN + Bi-GRU + 双残差 Add & Norm + 每层意图融合) × N_dec 层
            输出投影 → y_t
            自回归: dec_input_{t+1} = output_proj(y_t)
        线性输出层

    参数:
        input_size:    增广特征维度 (位置 + RF 筛选列, 不含意图概率)
        hidden_size:   Bi-GRU 隐藏维度 (单向); d_model = 2 * hidden_size
        output_size:   输出维度 (lat, lon, alt → 3)
        n_enc_layers:  编码器层数 (论文 Table 3 = 4)
        n_dec_layers:  解码器层数 (论文 Table 3 = 4)
        d_ff:          FFN 中间层维度 (默认 2 * hidden_size = 128)
        n_intent:      意图概率维度 (4 类机动)
        n_decode_steps:解码步数 (默认 1, 与 pure_gru 对齐; 架构支持多步)
        dropout:       Dropout 概率 (论文 Table 3 = 0.2)
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        output_size: int = 3,
        n_enc_layers: int = 4,
        n_dec_layers: int = 4,
        d_ff: int = 128,
        n_intent: int = 4,
        n_decode_steps: int = 1,
        dropout: float = 0.2,
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

        # ── 统一维度 (双向 GRU 拼接后维度 = 2 * hidden_size) ──
        self.d_model = 2 * hidden_size

        # ── 输入投影层 (将 input_size → d_model) ──
        self.input_proj = nn.Linear(input_size, self.d_model)

        # ── 编码器: N_enc 层 ──
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(self.d_model, hidden_size, d_ff, dropout)
            for _ in range(n_enc_layers)
        ])

        # ── 注意力 Q / K / V 投影 ──
        d_attn = hidden_size
        self.d_v = d_attn
        self.W_Q = nn.Linear(self.d_model, d_attn)
        self.W_K = nn.Linear(self.d_model, d_attn)
        self.W_V = nn.Linear(self.d_model, d_attn)
        self.attention = ScaledDotProductAttention(d_attn, dropout)

        # ── 解码器初始化映射 ──
        # 编码器最终隐藏 (来自最后一个 EncoderLayer 的 Bi-GRU): (2, batch, hidden)
        # 解码器各层初始隐藏: 每层 (2, batch, hidden), 共 n_dec_layers 层
        # 因此输入维度 = 2 * hidden, 输出维度 = n_dec_layers * 2 * hidden
        self.dec_h0_proj = nn.Linear(
            2 * hidden_size,
            n_dec_layers * 2 * hidden_size,
        )

        # ── 解码器: N_dec 层 ──
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(
                self.d_model, hidden_size, d_ff, self.d_v, n_intent, dropout
            )
            for _ in range(n_dec_layers)
        ])

        # ── 自回归输出投影 (将预测 y → d_model 作为下一步输入) ──
        self.output_proj = nn.Linear(output_size, self.d_model)

        # ── 输出层 (d_model → output_size) ──
        self.fc = nn.Linear(self.d_model, output_size)

    # ------------------------------------------------------------------
    def _get_decoder_hidden_repr(
        self,
        layer_h_list: list,
    ) -> torch.Tensor:
        """
        从解码器各层隐藏状态中提取用于计算动态 Q 的表示。
        使用最后一层的前向 + 后向隐藏状态拼接作为 h_dec_t (与 d_model 维度对齐)。

        参数:
            layer_h_list: List[Tensor], 长度 = n_dec_layers
                          每个元素形状 (2, batch, hidden_size)
        返回:
            h_dec_t: (batch, d_model) — 解码器当前时刻的隐藏状态表示
        """
        last_layer_h = layer_h_list[-1]                       # (2, batch, hidden)
        h_dec_t = torch.cat([last_layer_h[0], last_layer_h[1]], dim=-1)
        return h_dec_t                                        # (batch, d_model)

    # ------------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,
        intent_probs: torch.Tensor,
    ) -> torch.Tensor:
        """
        前向传播 — 自回归解码 + 动态注意力 + 每层意图融合。

        参数:
            x:            (batch, seq_len, input_size) — 增广特征序列
            intent_probs: (batch, seq_len, n_intent)   — SVM 概率序列
                          (会取最后一个时间步作为当前意图)

        返回:
            若 n_decode_steps == 1:  (batch, output_size)
            若 n_decode_steps  > 1:  (batch, n_decode_steps, output_size)
        """
        batch_size = x.size(0)

        # ══════════════════════════════════════════════════
        # 步骤 1: 输入投影 — 把 (input_size) → (d_model)
        # ══════════════════════════════════════════════════
        enc_input = self.input_proj(x)                        # (B, L, d_model)

        # ══════════════════════════════════════════════════
        # 步骤 2: 编码器 — 逐层处理 (每层双残差 Add & Norm)
        # ══════════════════════════════════════════════════
        enc_out = enc_input
        enc_h_n = None
        for layer in self.encoder_layers:
            enc_out, enc_h_n = layer(enc_out)
        # 此处 enc_h_n 为最后一层 Bi-GRU 的最终隐藏状态: (2, batch, hidden)

        # ══════════════════════════════════════════════════
        # 步骤 3: K, V 投影 (从编码器输出, 仅计算一次)
        # ══════════════════════════════════════════════════
        K = self.W_K(enc_out)                                 # (B, L, d_attn)
        V = self.W_V(enc_out)                                 # (B, L, d_attn)

        # ══════════════════════════════════════════════════
        # 步骤 4: 解码器初始化 — 映射编码器最终隐藏到各解码层
        # ══════════════════════════════════════════════════
        # enc_h_n: (2, batch, hidden) → flatten → (batch, 2*hidden)
        enc_h_flat = enc_h_n.permute(1, 0, 2).reshape(batch_size, -1)
        dec_h0_flat = self.dec_h0_proj(enc_h_flat)
        # → reshape (batch, n_dec_layers*2, hidden) → permute (n_dec_layers*2, batch, hidden)
        dec_h0 = dec_h0_flat.reshape(
            batch_size, self.n_dec_layers * 2, self.hidden_size
        ).permute(1, 0, 2).contiguous()

        # 拆分为每层独立的初始隐藏状态: 每层 (2, batch, hidden)
        layer_h = [
            dec_h0[i * 2:(i + 1) * 2].contiguous()
            for i in range(self.n_dec_layers)
        ]

        # ══════════════════════════════════════════════════
        # 步骤 5: 自回归解码循环
        # ══════════════════════════════════════════════════
        # 取意图概率序列的最后一步作为当前意图条件
        intent_last = intent_probs[:, -1, :]                  # (B, n_intent)

        # 初始解码输入: 编码器最后时间步的输出 (Bridge)
        dec_input = enc_out[:, -1:, :]                        # (B, 1, d_model)

        outputs = []
        for t in range(self.n_decode_steps):
            # ── 5a. 动态 Q: 从解码器当前隐藏状态生成 (论文公式 27) ──
            h_dec_t = self._get_decoder_hidden_repr(layer_h)  # (B, d_model)
            Q_t = self.W_Q(h_dec_t).unsqueeze(1)              # (B, 1, d_attn)

            # ── 5b. 动态 α: 每步重新计算 (论文公式 30) ──
            alpha_t = self.attention(Q_t, K, V)               # (B, d_v)

            # ── 5c. 逐层解码 (每层独立融合 α_t 和 P_S, 隐藏状态自回归更新) ──
            for i, layer in enumerate(self.decoder_layers):
                dec_input, new_h = layer(
                    dec_input, alpha_t, intent_last, layer_h[i]
                )
                layer_h[i] = new_h

            # ── 5d. 输出投影: d_model → output_size ──
            out_t = self.fc(dec_input.squeeze(1))             # (B, output_size)
            outputs.append(out_t)

            # ── 5e. 自回归: 投影预测结果作为下一步解码输入 ──
            if t < self.n_decode_steps - 1:
                dec_input = self.output_proj(out_t).unsqueeze(1)   # (B, 1, d_model)

        # ══════════════════════════════════════════════════
        # 步骤 6: 返回
        # ══════════════════════════════════════════════════
        if self.n_decode_steps == 1:
            return outputs[0]                                  # (B, output_size)
        return torch.stack(outputs, dim=1)                     # (B, T_dec, output_size)
