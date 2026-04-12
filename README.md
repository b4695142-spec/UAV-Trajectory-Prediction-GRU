# UAV 轨迹预测项目 (UAV Trajectory Prediction with GRU)

本项目基于苏黎世城市微型飞行器 (UMAV/AGZ) 数据集，使用 PyTorch 框架构建了一个门控循环单元 (GRU) 神经网络，用于对无人机在未来的三维飞行轨迹（纬度、经度和海拔）进行连续时间序列预测。

## 📋 项目简介

该项目实现了一个完整的端到端时序预测工作流，包含了如下模块：
- **数据清洗与预处理**：从原始 GPS 数据中提取关键特征，进行了降采样与归一化。
- **序列构建**：使用滑动窗口机制构造 GRU 模型所需的时序输入数据。
- **模型搭建与训练**：定义了两层 GRU 叠加的网络，配合 Early Stopping 自动保存最佳模型。
- **直观可视化**：自动运行推理过程，并进行坐标系的反归一化，最终生成真实与预测的 3D 飞行轨迹对比图。

## 📂 核心文件目录结构

```text
AGZ_subset/
│
├── Log Files/
│   └── OnboardGPS.csv        # 原始数据集 (UMAV/AGZ 无人机飞行日志)
│
├── preprocess_uav.py         # 1. 核心特征提取、降采样 (0.03s -> 0.1s)、归一化、切分数据集
├── build_sequences.py        # 2. 根据 Look_Back 和 Forward_Length 构造滑动窗口序列
├── gru_model.py              # 3. PyTorch GRU 模型架构定义 (面向对象结构)
├── train.py                  # 4. 训练核心代码：包含 Xavier 初始化、MSE 损失和早停机制
├── visualize.py              # 5. 模型推理、反归一化还原真实坐标与 3D 绘图
│
└── processed_data/           # 预处理全流程自动生成的中间数据目录
    ├── train_data.npy        # 步骤 1 输出的训练集
    ├── test_data.npy         # 步骤 1 输出的测试集
    ├── scaler_params.npz     # 步骤 1 输出的 MinMax 归一化参数 (用于反向推导)
    ├── X_train.npy, Y_train.npy, ...  # 步骤 2 序列化后的建模数据
    └── best_gru_model.pth    # 步骤 4 训练完成后保存的最佳模型权重
```

## 🛠️ 环境依赖

请确保您的环境中安装了以下基础依赖：
- Python 3.8+
- `torch` (PyTorch，支持 CPU/GPU 自动切换)
- `numpy`
- `pandas`
- `scikit-learn`
- `matplotlib`

## 🚀 快速开始指引 (Workflow)

请严格按照以下顺序执行脚本进行测试及训练。

### 1. 数据预处理
运行预处理脚本以清洗原始 `OnboardGPS.csv` 并归一化至 `[0,1]` 范围：
```bash
python preprocess_uav.py
```
> *操作说明*：该脚本会将大约 30Hz 的高频信号降采样为 10Hz (0.1s点距)，并根据连续时间切分为 80% 训练集与 20% 测试集。

### 2. 构造滑动窗口数据
根据论文约束及定义自动切分过去观察长度及未来预测长度：
```bash
python build_sequences.py
```
> *核心参数*：默认过去观察窗口 `Look_Back=50` (5秒跨度的数据)，目标预测长度为 `Forward_Length=5` (预测未来 0.5s 时刻的位置)。

### 3. 模型训练
启动 PyTorch GRU 模型的训练过程：
```bash
python train.py
```
> *操作说明*：模型会自动通过 MSE 损失搭配 Adam 优化器在最佳显卡（优先使用 CUDA）上进行训练。如果在 15 个 epoch 内验证集没有改善则会立刻触发早停并保存当前最佳的参数集。

### 4. 结果可视化与推理
使用训练好的最佳模型进行测试验证并生成图片：
```bash
python visualize.py
```
> *操作说明*：会自动在项目根目录输出一张全高清的 3D 轨迹图 `trajectory_3d_plot.png`。由于数据需要真实还原，这里进行了一次 `inverse_transform` (反归一化) 绘制真实经纬度与海拔。

## ⚙️ 关键超参数总结

| 参数 | 默认设定值 | 描述说明 |
|:---|:---|:---|
| `Downsample Factor` | 3 | 获取 10Hz 数据的降低采样倍率 |
| `Look_Back` | 50 | 窗口滑动观测长度 (用于构成 GRU 每个 batch ) |
| `Forward_Length` | 5 | 往后预测第 N 个点的时间序列 |
| `Hidden_Size` | 64 | GRU 网络内层特征维度大小 |
| `Num_Layers` | 2 | GRU 网络叠加的层数 |
| `Batch_Size` | 70 | 单批次处理的大小 (严格控制以匹配论文) |
| `Learning_Rate`| 1e-3 | Adam 网络学习率 |

---
**注意**: 生成的数据缓存和日志大文件已在本地 `.gitignore` 中进行了配置忽略以保持代码库整洁。 
