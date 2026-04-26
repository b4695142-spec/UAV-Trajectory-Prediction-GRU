"""
==============================================================================
UAVTrajectoryGRU 模型结构与前向传播验证
==============================================================================
将原 gru_model.py 中的 if __name__ == "__main__" 测试代码提取至此文件，
保持核心模型文件的纯净性。
==============================================================================
"""

import sys
from pathlib import Path

# 允许直接运行本文件：将仓库根目录加入 path，以便 `import core.*`
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch
from core.gru_model import UAVTrajectoryGRU


def main():
    INPUT_SIZE = 3
    HIDDEN_SIZE = 64
    NUM_LAYERS = 2
    OUTPUT_SIZE = 3
    LOOK_BACK = 10
    BATCH_SIZE = 32

    model = UAVTrajectoryGRU(
        input_size=INPUT_SIZE,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        output_size=OUTPUT_SIZE,
    )

    print("=" * 60)
    print("  UAVTrajectoryGRU 模型结构")
    print("=" * 60)
    print(model)
    print()

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  总参数量:      {total_params:,}")
    print(f"  可训练参数量:  {trainable_params:,}")
    print()

    print("=" * 60)
    print("  前向传播测试")
    print("=" * 60)

    dummy_input = torch.randn(BATCH_SIZE, LOOK_BACK, INPUT_SIZE)
    print(f"  输入形状:  {dummy_input.shape}  "
          f"→ (batch_size={BATCH_SIZE}, Look_Back={LOOK_BACK}, features={INPUT_SIZE})")

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


if __name__ == "__main__":
    main()
