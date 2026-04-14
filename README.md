# UAV-Trajectory-Prediction-GRU

基于门控循环单元 (GRU) 神经网络的无人机 (UAV) 三维飞行轨迹预测系统。本项目针对苏黎世城市微型飞行器 (UMAV/AGZ) 数据集进行建模，实现对无人机未来位置（纬度、经度、海拔）的精准预测。

---

## 🚀 项目亮点

- **完整序列工程**：涵盖从原始 GPS 数据清洗、降采样、归一化到滑动窗口构建的全流程。
- **高性能 GRU 架构**：采用双层叠加的 GRU 网络，具备更强的时序特征捕捉能力。
- **精准评估体系**：除常规 MSE 损失外，还计算了物理含义明确的 MAE、RMSE 以及 **3D 空间欧氏距离误差**。
- **双维度可视化**：
    - **3D 轨迹图**：直观对比真实轨迹与预测轨迹的重合度。
    - **2D 误差图**：实时分析预测误差随时间步的变化趋势。

---

## 📂 目录结构

```text
UAV-Trajectory-Prediction-GRU/
├── Log Files/               # 数据源目录
│   └── OnboardGPS.csv       # 原始飞行日志 (30Hz)
├── processed_data/          # 自动生成的中间件与产物
│   ├── train_data.npy       # 归一化后的训练集
│   ├── test_data.npy        # 归一化后的测试集
│   ├── scaler_params.npz    # Min-Max 缩放参数 (用于反归一化)
│   ├── X_train.npy/Y_train.npy  # 滑动窗口训练序列
│   └── best_gru_model.pth    # 性能最优的模型权重文件
├── preprocess_uav.py        # [步骤1] 数据预处理与归一化
├── build_sequences.py       # [步骤2] 滑动窗口序列构建
├── gru_model.py             # [核心] GRU 模型类定义
├── train.py                 # [步骤3] 模型训练与早停优化
├── visualize.py             # [步骤4] 测试集推理与多维可视化
└── trajectory_*.png         # 最终输出的轨迹图与误差图
```

---

## 🛠️ 环境准备

请确保 Python 版本 >= 3.8，并安装以下依赖：

```bash
pip install torch numpy pandas scikit-learn matplotlib
```

---

## 🏃 运行流程 (Quick Start)

为保证模型训练效果，请严格遵循以下流水线：

### 1. 数据预处理
```bash
python preprocess_uav.py
```
- **逻辑**：提取 `lat, lon, alt`，将 ~30Hz 降采样至 **10Hz** (0.1s 间隔)，并执行 Min-Max 归一化。

### 2. 构造序列数据
```bash
python build_sequences.py
```
- **配置**：默认观测长度 `Look_Back = 50` (5.0s 历史)，预测目标为未来第 5 个点 (`Forward_Length = 5`，即 0.5s 后)。

### 3. 执行模型训练
```bash
python train.py
```
- **参数**：Batch Size = 70，学习率 = 1e-3。采用 **Early Stopping** (Patience=15) 机制防止过拟合，并保存测试集表现最佳的模型。

### 4. 结果验证与绘图
```bash
python visualize.py
```
- **输出**：在反归一化真实坐标系下计算误差，并生成 `trajectory_3d_plot.png` 和 `trajectory_2d_error.png`。

---

## 🧠 模型细节

### 神经网络架构
- **输入层**：(Batch, 50, 3) -> 包含 50 个连续时刻的三维坐标序列。
- **隐藏层**：2 层叠加 GRU，Hidden Size = 64。
- **输出层**：Linear 层 (64 -> 3)，输出未来指定时刻的 `(lat, lon, alt)`。
- **初始化**：权重采用 **Xavier Uniform (Glorot)** 初始化。

### 误差评估指标
系统在推理阶段会自动输出以下物理指标：
- **MAE / RMSE**：分别针对经度、纬度、高度计算平均绝对误差和均方根误差。
- **Average Euclidean Error**：在 3D 物理空间中的预测点与真实点之间的平均欧几里得距离。

---

## 📊 可视化示例

项目运行结束后将获得：
1. **3D 轨迹图**：展示无人机在三维空间中的实际机动路径与 GRU 预测路径。
2. **2D 误差变化曲线**：横轴为时间步，纵轴为 3D 欧氏距离误差，用于分析预测稳定性。

---
**数据集引用**：Zurich Urban Micro Aerial Vehicle (UMAV) Dataset.
