# UAV-Trajectory-Prediction-GRU

基于门控循环单元 (GRU) 神经网络的无人机 (UAV) 三维飞行轨迹预测系统。本项目针对苏黎世城市微型飞行器 (UMAV/AGZ) 数据集进行建模，基于历史观测序列实现对无人机位置（纬度、经度、海拔）的精准预测。

---

## 🚀 项目亮点

- **完整序列工程**：涵盖从原始 GPS 数据清洗、降采样、归一化到滑动窗口构建的全流程。
- **高性能 GRU 架构**：采用双层叠加的 GRU 网络，具备更强的时序特征捕捉能力。
- **精准评估体系**：除常规 MSE 损失外，还计算了物理含义明确的 MAE、RMSE、Average RMSE 以及 **3D 空间欧氏距离误差**。
- **三维度可视化**：
    - **3D 轨迹图**：直观对比真实轨迹与预测轨迹在三维空间中的重合度。
    - **2D 误差图**：实时分析 3D 欧氏距离综合预测误差随时间步的变化趋势。
    - **推理耗时图**：评估模型单次预测延迟，分析推理性能随时间步的波动情况。

---

## 📂 目录结构

```text
UAV-Trajectory-Prediction-GRU/
├── Log Files/                    # 数据源目录 (需自行放置)
│   └── OnboardGPS.csv            # 原始飞行日志 (~30Hz)
├── processed_data/               # [自动生成] 预处理中间件与模型产物
│   ├── train_data.npy            # 归一化后的训练集 (shape: N_train × 3)
│   ├── test_data.npy             # 归一化后的测试集 (shape: N_test × 3)
│   ├── scaler_params.npz         # Min-Max 缩放参数 (data_min, data_max, feature_range)
│   ├── X_train.npy               # 训练集滑动窗口输入 (shape: n_train × 50 × 3)
│   ├── Y_train.npy               # 训练集滑动窗口标签 (shape: n_train × 3)
│   ├── X_test.npy                # 测试集滑动窗口输入 (shape: n_test × 50 × 3)
│   ├── Y_test.npy                # 测试集滑动窗口标签 (shape: n_test × 3)
│   └── best_gru_model.pth        # 性能最优的模型权重文件
├── preprocess_uav.py             # [步骤1] 数据预处理与归一化
├── build_sequences.py            # [步骤2] 滑动窗口序列构建
├── gru_model.py                  # [核心] GRU 模型类定义 (UAVTrajectoryGRU)
├── train.py                      # [步骤3] 模型训练与早停优化
├── visualize.py                  # [步骤4] 测试集推理、多维可视化与耗时评估
├── .gitignore                    # Git 忽略规则
├── trajectory_3d_plot.png        # [输出] 3D 轨迹对比图
├── trajectory_2d_error.png       # [输出] 2D 综合误差折线图
└── inference_time_plot.png       # [输出] 单次预测耗时折线图
```

> **注意**：`Log Files/` 目录及 `OnboardGPS.csv` 需自行下载并放置；`processed_data/` 目录及所有 `.npy`、`.npz`、`.pth` 文件由脚本自动生成，已被 `.gitignore` 忽略。

---

## 🛠️ 环境准备

请确保 Python 版本 >= 3.8，并安装以下依赖：

```bash
pip install torch numpy pandas scikit-learn matplotlib
```

| 依赖库 | 用途 | 使用模块 |
|--------|------|----------|
| `torch` | GRU 模型构建、训练与推理 | `gru_model.py`, `train.py`, `visualize.py` |
| `numpy` | 数组运算与数据存储 | 全部模块 |
| `pandas` | CSV 读取与数据清洗 | `preprocess_uav.py` |
| `scikit-learn` | MinMaxScaler 归一化 | `preprocess_uav.py` |
| `matplotlib` | 3D/2D 可视化绘图 | `visualize.py` |

---

## 🏃 运行流程 (Quick Start)

为保证模型训练效果，请严格遵循以下流水线顺序执行：

### 1. 数据预处理
```bash
python preprocess_uav.py
```
- **输入**：`Log Files/OnboardGPS.csv`（原始飞行日志，~30Hz 采样）
- **处理逻辑**：
    1. 提取核心位置特征 `lat, lon, alt`（丢弃速度、航向、精度等非位置参数）
    2. 等间隔降采样：~30Hz → **10Hz**（每 3 个点保留 1 个，目标间隔 0.1s）
    3. Min-Max 归一化：将纬度/经度/海拔映射至 [0, 1] 区间
    4. 按时间顺序 80:20 切分训练集/测试集（**不打乱数据**）
- **输出**：`processed_data/train_data.npy`、`processed_data/test_data.npy`、`processed_data/scaler_params.npz`

### 2. 构造序列数据
```bash
python build_sequences.py
```
- **输入**：步骤 1 生成的 `train_data.npy` 和 `test_data.npy`
- **配置**：
    - 观测窗口 `Look_Back = 50`（即 5.0s 历史数据）
    - 预测步长 `Forward_Length = 0`（即当前时刻的目标位置）
- **滑动窗口公式**：对于时间点 t，输入 `data[t-49 : t+1]`，目标 `data[t+0]`（即 `data[t]`）
- **输出**：`processed_data/X_train.npy`、`Y_train.npy`、`X_test.npy`、`Y_test.npy`

### 3. 执行模型训练
```bash
python train.py
```
- **输入**：步骤 2 生成的滑动窗口序列数据
- **训练参数**：
    - Batch Size = 70
    - 学习率 = 1e-3（Adam 优化器）
    - 最大 Epochs = 500
    - 损失函数 = MSELoss
- **优化机制**：采用 **Early Stopping**（Patience=15），连续 15 个 Epoch 测试集损失无改善则终止训练，自动保存测试集表现最佳的模型权重
- **权重初始化**：Xavier Uniform (Glorot) 初始化，偏置项置零
- **设备支持**：自动检测 CUDA GPU，不可用时回退至 CPU
- **输出**：`processed_data/best_gru_model.pth`

### 4. 结果验证与可视化
```bash
python visualize.py
```
- **输入**：测试集数据 + 最佳模型权重 + 归一化参数
- **执行步骤**：
    1. 加载测试集与模型权重，执行推理
    2. 反归一化，还原为真实世界坐标，计算各项误差指标
    3. 绘制 3D 轨迹对比图 → `trajectory_3d_plot.png`
    4. 绘制 2D 综合误差折线图 → `trajectory_2d_error.png`
    5. 评估单次预测耗时并绘制折线图 → `inference_time_plot.png`
- **可视化参数**（可在 `visualize.py` 顶部修改）：
    - `PLOT_START = 0`：绘图起始索引
    - `PLOT_END = None`：绘图结束索引（`None` 表示绘制全部测试集）

---

## 🧠 模型细节

### 神经网络架构

```
输入 (batch_size, 50, 3)
    ↓
GRU × 2 层 (hidden_size=64, batch_first=True)
    ↓  ← 取最后一个时间步的隐藏状态 gru_out[:, -1, :]
Linear (64 → 3)
    ↓  ← 无激活函数，线性回归投影
输出 (batch_size, 3)  →  t 时刻的 (lat, lon, alt)
```

| 组件 | 参数 |
|------|------|
| 输入维度 | 3 (纬度, 经度, 海拔) |
| GRU 隐藏层维度 | 64 |
| GRU 堆叠层数 | 2 |
| 输出维度 | 3 (预测纬度, 经度, 海拔) |
| 权重初始化 | Xavier Uniform (Glorot) |
| 偏置初始化 | 全零 |
| 模型类 | `UAVTrajectoryGRU`（定义于 `gru_model.py`） |

### 误差评估指标

系统在推理阶段自动输出以下物理指标（均在反归一化后的真实坐标系下计算）：

| 指标 | 说明 |
|------|------|
| **MAE** (各维度) | 纬度、经度、高度各自的平均绝对误差 |
| **RMSE** (各维度) | 纬度、经度、高度各自的均方根误差 |
| **Average RMSE** | 三个维度 RMSE 的算术平均值 |
| **Average Euclidean Error** | 预测点与真实点在 3D 空间中的平均欧几里得距离 |

---

## ⚙️ 配置参数速查

各脚本的核心配置参数均定义在文件顶部，可根据需求修改：

| 脚本 | 参数 | 默认值 | 说明 |
|------|------|--------|------|
| `preprocess_uav.py` | `ORIGINAL_INTERVAL_S` | 0.033333 | 原始采样间隔 (秒) |
| | `TARGET_INTERVAL_S` | 0.1 | 目标采样间隔 (秒) |
| | `DOWNSAMPLE_FACTOR` | 3 | 降采样因子 (自动计算) |
| | `TRAIN_RATIO` | 0.8 | 训练集比例 |
| `build_sequences.py` | `LOOK_BACK` | 50 | 历史观测步长 |
| | `FORWARD_LENGTH` | 0 | 未来预测步长 |
| `train.py` | `BATCH_SIZE` | 70 | 批次大小 |
| | `LEARNING_RATE` | 1e-3 | Adam 学习率 |
| | `MAX_EPOCHS` | 500 | 最大训练轮数 |
| | `PATIENCE` | 15 | 早停耐心值 |
| | `HIDDEN_SIZE` | 64 | GRU 隐藏层维度 |
| | `NUM_LAYERS` | 2 | GRU 层数 |
| `visualize.py` | `PLOT_START` | 0 | 绘图起始索引 |
| | `PLOT_END` | None | 绘图结束索引 |

---

## 📊 可视化输出

项目运行 `visualize.py` 后将生成以下三张图表：

1. **3D 轨迹对比图** (`trajectory_3d_plot.png`)：展示无人机在三维空间中的实际机动路径（灰色虚线）与 GRU 预测路径（红色实线），标记起点（绿色圆点）与终点（蓝色三角）。
2. **2D 综合误差折线图** (`trajectory_2d_error.png`)：横轴为时间步，纵轴为 3D 欧氏距离综合真实误差，用于分析预测稳定性与误差波动。
3. **推理耗时折线图** (`inference_time_plot.png`)：横轴为时间步，纵轴为单次预测耗时（毫秒），红色虚线标注平均耗时，用于评估模型推理性能。

---

## ⚠️ 注意事项

- **数据集放置**：运行步骤 1 前，需将 `OnboardGPS.csv` 放置于 `Log Files/` 目录下。该文件来自 UMAV 数据集，需自行下载。
- **时序切分**：本项目为连续时间序列任务，训练集/测试集严格按时间顺序划分，**不可使用随机打乱切分**。
- **原始数据列名**：`OnboardGPS.csv` 中时间戳列名拼写为 `Timpstemp`（原始数据的拼写错误），代码已做兼容处理。
- **中文字体**：可视化脚本默认使用 `SimHei` 字体渲染中文标签。若系统未安装该字体，图表中的中文可能显示为方块，请安装对应字体或修改 `visualize.py` 中的 `plt.rcParams['font.sans-serif']` 配置。
- **GPU 加速**：训练与推理脚本自动检测 CUDA 设备。在 GPU 环境下推理耗时会显著降低。
- **执行顺序**：四个脚本存在严格的依赖关系，必须按 1→2→3→4 的顺序依次执行，不可跳步。

---

## 📄 数据集引用

Zurich Urban Micro Aerial Vehicle (UMAV/AGZ) Dataset.
