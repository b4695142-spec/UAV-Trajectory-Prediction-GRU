"""
==============================================================================
UAV 轨迹预测 GRU 模型
==============================================================================
基于 PyTorch 实现的 GRU (Gated Recurrent Unit) 网络，用于预测无人机
在当前 t 时刻的三维坐标 (纬度, 经度, 海拔)。

网络结构:
    输入 (batch_size, Look_Back, 3)
        ↓
    GRU × 2 层 (hidden_size=64)
        ↓  ← 取最后一个时间步的隐藏状态
    Linear (64 → 3)
        ↓  ← 无激活函数，线性投影
    输出 (batch_size, 3)
==============================================================================
"""

import torch
import torch.nn as nn


class UAVTrajectoryGRU(nn.Module):
    """
    面向对象的 GRU 轨迹预测模型。

    参数:
        input_size  (int): 输入特征维度，默认 3 (纬度, 经度, 海拔)
        hidden_size (int): GRU 隐藏层维度，默认 64
        num_layers  (int): GRU 堆叠层数，默认 2
        output_size (int): 输出维度，默认 3 (预测的纬度, 经度, 海拔)
    """

    def __init__(
        self,
        input_size: int = 3,
        hidden_size: int = 64,
        num_layers: int = 2,
        output_size: int = 3,
    ):
        super(UAVTrajectoryGRU, self).__init__()

        # 保存超参数，方便后续访问
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size

        # ---------------------------------------------------------------
        # GRU 层
        # ---------------------------------------------------------------
        # - input_size=3:      每个时间步输入 3 个特征 (lat, lon, alt)
        # - hidden_size=64:    隐藏状态维度为 64
        # - num_layers=2:      堆叠 2 层 GRU
        # - batch_first=True:  输入张量形状为 (batch_size, seq_len, input_size)
        #                      而非默认的 (seq_len, batch_size, input_size)
        self.gru = nn.GRU(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
        )

        # ---------------------------------------------------------------
        # 全连接输出层 (线性投影)
        # ---------------------------------------------------------------
        # - 输入维度:  hidden_size (64)，即 GRU 最后时间步的隐藏状态
        # - 输出维度:  output_size (3)，对应 t 时刻的 (lat, lon, alt)
        # - 不使用激活函数: 回归任务直接线性映射即可
        self.fc = nn.Linear(
            in_features=self.hidden_size,
            out_features=self.output_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播。

        参数:
            x: 输入张量，形状 (batch_size, Look_Back, 3)
               - batch_size: 批次大小
               - Look_Back:  滑动窗口长度 (历史时间步数)
               - 3:          特征维度 (lat, lon, alt)

        返回:
            out: 预测结果，形状 (batch_size, 3)
                 对应 t 时刻的 (纬度, 经度, 海拔)
        """
        # -----------------------------------------------------------------
        # 1. 初始化 GRU 隐藏状态为全零
        # -----------------------------------------------------------------
        #    h_0 形状: (num_layers, batch_size, hidden_size) = (2, batch_size, 64)
        #    使用与输入相同的设备 (CPU/GPU) 和数据类型
        batch_size = x.size(0)
        h_0 = torch.zeros(
            self.num_layers, batch_size, self.hidden_size,
            device=x.device, dtype=x.dtype
        )

        # -----------------------------------------------------------------
        # 2. 将输入序列送入 GRU 层
        # -----------------------------------------------------------------
        #    gru_out 形状: (batch_size, Look_Back, hidden_size)
        #                   包含每个时间步的输出
        #    h_n     形状: (num_layers, batch_size, hidden_size)
        #                   最后一个时间步各层的隐藏状态
        gru_out, h_n = self.gru(x, h_0)

        # -----------------------------------------------------------------
        # 3. 提取最后一个时间步的输出
        # -----------------------------------------------------------------
        #    gru_out[:, -1, :] → 形状: (batch_size, hidden_size) = (batch_size, 64)
        #    这是 GRU 最顶层在序列最后一个时间步的隐藏状态，
        #    编码了整个输入序列的时序信息
        last_hidden = gru_out[:, -1, :]

        # -----------------------------------------------------------------
        # 4. 全连接层: 线性投影到输出空间
        # -----------------------------------------------------------------
        #    (batch_size, 64) → (batch_size, 3)
        #    不经过激活函数，直接返回回归预测值
        out = self.fc(last_hidden)

        return out


# ============================================================================
# 测试代码: 实例化模型并验证
# ============================================================================
if __name__ == "__main__":
    # 模型超参数
    INPUT_SIZE = 3       # 输入特征数 (lat, lon, alt)
    HIDDEN_SIZE = 64     # GRU 隐藏层维度
    NUM_LAYERS = 2       # GRU 层数
    OUTPUT_SIZE = 3      # 输出维度 (lat, lon, alt)
    LOOK_BACK = 10       # 滑动窗口长度 (历史时间步数)
    BATCH_SIZE = 32      # 批次大小

    # -----------------------------------------------------------------
    # 实例化模型
    # -----------------------------------------------------------------
    model = UAVTrajectoryGRU(
        input_size=INPUT_SIZE,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        output_size=OUTPUT_SIZE,
    )

    # -----------------------------------------------------------------
    # 打印模型结构
    # -----------------------------------------------------------------
    print("=" * 60)
    print("  UAVTrajectoryGRU 模型结构")
    print("=" * 60)
    print(model)
    print()

    # 统计模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  总参数量:      {total_params:,}")
    print(f"  可训练参数量:  {trainable_params:,}")
    print()

    # -----------------------------------------------------------------
    # 模拟前向传播，验证输入输出形状
    # -----------------------------------------------------------------
    print("=" * 60)
    print("  前向传播测试")
    print("=" * 60)

    # 构造随机输入: (batch_size, Look_Back, 3)
    dummy_input = torch.randn(BATCH_SIZE, LOOK_BACK, INPUT_SIZE)
    print(f"  输入形状:  {dummy_input.shape}  "
          f"→ (batch_size={BATCH_SIZE}, Look_Back={LOOK_BACK}, features={INPUT_SIZE})")

    # 前向传播
    with torch.no_grad():
        prediction = model(dummy_input)

    print(f"  输出形状:  {prediction.shape}  "
          f"→ (batch_size={BATCH_SIZE}, output={OUTPUT_SIZE})")
    print(f"  输出样例 (前 3 个样本):")
    for i in range(min(3, BATCH_SIZE)):
        lat, lon, alt = prediction[i].tolist()
        print(f"    样本 {i}: lat={lat:.6f}, lon={lon:.6f}, alt={alt:.6f}")

    print()
    print("✅ 模型结构与前向传播验证通过!")
