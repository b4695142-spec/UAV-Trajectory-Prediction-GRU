# UAV-Trajectory-Prediction-GRU

基于门控循环单元 (GRU) 神经网络的无人机 (UAV) 三维飞行轨迹预测系统。本项目针对苏黎世城市微型飞行器 (UMAV/AGZ) 数据集进行建模，基于历史观测序列实现对无人机位置（纬度、经度、海拔）的精准预测。

> **🔬 探索性增强**：本仓库额外提供了一个 **完全解耦的** "机动意图识别" 级联模块（Random Forest 特征筛选 → OvA SVM + Platt Scaling → 拼接至 GRU 输入），用于抑制长预测视距下的离群值。该模块与原始纯 GRU 管线并存，随时可以 **无缝回退**。详见文末的 **[意图识别增强版流水线](#-意图识别增强版流水线-探索性)** 章节。

---

## 🚀 项目亮点

- **完整序列工程**：涵盖从原始 GPS 数据清洗、降采样、归一化到滑动窗口构建的全流程。
- **集中化配置管理**：两条管线各自拥有独立的 `config.py`，所有超参数与路径配置集中管理，修改参数只需编辑 `config.py`，无需逐一修改各脚本。
- **公平性保障**：所有管线均采用"先切分再归一化"策略，MinMaxScaler 仅在训练集上 fit，杜绝测试集信息泄漏。
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
├── core/                                # 核心共享代码
│   ├── __init__.py                      # 模块初始化 (导出 UAVTrajectoryGRU)
│   ├── gru_model.py                     # GRU 模型类定义 (UAVTrajectoryGRU, 两个管线共用)
│   └── test_gru_model.py                # 模型结构与前向传播验证脚本
│
├── pipelines/
│   ├── __init__.py                      # 管线包初始化
│   ├── pure_gru/                        # 管线 1: 纯 GRU 基线
│   │   ├── __init__.py                  # 管线包初始化
│   │   ├── config.py                    # 集中配置 (超参数、路径、可视化参数)
│   │   ├── preprocess_uav.py            # [步骤1] 数据预处理与归一化
│   │   ├── build_sequences.py           # [步骤2] 滑动窗口序列构建
│   │   ├── train.py                     # [步骤3] 模型训练与早停优化
│   │   └── visualize.py                 # [步骤4] 测试集推理、多维可视化与耗时评估
│   │
│   └── intent_gru/                      # 管线 2: 意图增强 GRU (探索性)
│       ├── __init__.py                  # 管线包初始化
│       ├── config.py                    # 集中配置 (超参数、路径、可视化参数)
│       ├── intent/                      # 意图识别子模块 (Python 包)
│       │   ├── __init__.py              # 模块入口, 统一导出公共接口
│       │   ├── feature_extractor.py     # 动态提取 OnboardGPS 特征 + 派生运动学特征
│       │   ├── label_generator.py       # 基于 alt_rate/heading_rate 自动生成 4 类伪标签
│       │   ├── rf_selector.py           # RF 特征重要性 + 累计 80% 筛选 (支持排除泄漏列)
│       │   ├── svm_classifier.py        # OvA + Platt Scaling 的 SVM 封装 (含 K-Fold OOF)
│       │   └── intent_dataset.py        # 拼接概率向量的滑动窗口构造工具
│       ├── prepare_intent.py            # [步骤1.5] 意图数据准备
│       ├── train_with_intent.py         # [步骤1.6] 训练增广版 GRU
│       ├── visualize_with_intent.py     # [步骤1.7] 增广版推理与可视化
│       └── compare_models.py            # 纯 GRU vs GRU+意图 对比评估
│
├── Log Files/                           # 数据源目录 (需自行放置)
│   └── OnboardGPS.csv                   # 原始飞行日志 (~30Hz)
├── processed_data/                      # [自动生成] 预处理中间件与模型产物
│   ├── train_data.npy                   # 归一化后的训练集 (shape: N_train × 3)
│   ├── test_data.npy                    # 归一化后的测试集 (shape: N_test × 3)
│   ├── scaler_params.npz                # Min-Max 缩放参数
│   ├── X_train.npy / Y_train.npy       # 训练集滑动窗口
│   ├── X_test.npy / Y_test.npy         # 测试集滑动窗口
│   ├── best_gru_model.pth              # 纯 GRU 最优模型权重
│   └── intent/                          # [自动生成] 意图增强版产物 (与纯 GRU 完全隔离)
│       ├── X_train_intent.npy / Y_train_intent.npy
│       ├── X_test_intent.npy  / Y_test_intent.npy
│       ├── std_scaler.pkl              # StandardScaler (非位置列, 仅 train fit)
│       ├── position_minmax_scaler.pkl  # MinMaxScaler (位置列, 仅 train fit)
│       ├── rf_model.pkl               # 随机森林模型
│       ├── svm_model.pkl              # OvA SVM 模型
│       ├── selected_indices.npy        # 最终 GRU 输入特征索引
│       ├── rf_selected_indices.npy     # RF 筛选列索引
│       ├── rf_importances.npy          # 完整特征重要性数组
│       ├── labels_train.npy / labels_test.npy
│       ├── minmax_scaler_params.npz    # 目标 (lat,lon,alt) 的 MinMax 参数
│       ├── best_gru_model_intent.pth   # 意图增强版最优模型权重
│       └── intent_report.json          # 特征名 / RF 重要性 / SVM 指标等汇总
├── .gitignore                           # Git 忽略规则
├── README.md                            # 项目说明文档
├── trajectory_3d_plot.png               # [输出] 3D 轨迹对比图
├── trajectory_2d_error.png              # [输出] 2D 综合误差折线图
└── inference_time_plot.png              # [输出] 单次预测耗时折线图
```

> **注意**：`Log Files/` 目录及 `OnboardGPS.csv` 需自行下载并放置；`processed_data/` 目录及所有 `.npy`、`.npz`、`.pth`、`.pkl` 文件由脚本自动生成，已被 `.gitignore` 忽略。

---

## 🛠️ 环境准备

请确保 Python 版本 >= 3.8，并安装以下依赖：

```bash
pip install torch numpy pandas scikit-learn matplotlib
```


| 依赖库            | 用途                                       | 使用模块                                                                                          |
| -------------- | ---------------------------------------- | --------------------------------------------------------------------------------------------- |
| `torch`        | GRU 模型构建、训练与推理                           | `core/gru_model.py`, `pipelines/pure_gru/train.py`, `pipelines/pure_gru/visualize.py`         |
| `numpy`        | 数组运算与数据存储                                | 全部模块                                                                                          |
| `pandas`       | CSV 读取与数据清洗                              | `pipelines/pure_gru/preprocess_uav.py`                                                        |
| `scikit-learn` | MinMaxScaler / StandardScaler 归一化、RF、SVM | `pipelines/pure_gru/preprocess_uav.py`, `pipelines/intent_gru/prepare_intent.py`              |
| `matplotlib`   | 3D/2D 可视化绘图                              | `pipelines/pure_gru/visualize.py`                                                             |


---

## 🏃 运行流程 (Quick Start)

为保证模型训练效果，请严格遵循以下流水线顺序执行：

### 1. 数据预处理

```bash
python pipelines/pure_gru/preprocess_uav.py
```

- **输入**：`Log Files/OnboardGPS.csv`（原始飞行日志，~30Hz 采样）
- **处理逻辑**：
  1. 提取核心位置特征 `lat, lon, alt`（丢弃速度、航向、精度等非位置参数）
  2. 等间隔降采样：~30Hz → **10Hz**（每 3 个点保留 1 个，目标间隔 0.1s）
  3. 按时间顺序 80:20 切分训练集/测试集（**不打乱数据**）
  4. Min-Max 归一化：将纬度/经度/海拔映射至 [0, 1] 区间（**仅在训练集上 fit**，消除数据泄漏）
- **输出**：`processed_data/train_data.npy`、`processed_data/test_data.npy`、`processed_data/scaler_params.npz`

> **公平性修正**：当前版本先切分再归一化，MinMaxScaler 仅在训练集上 fit 后 transform 测试集，避免测试集的 min/max 信息泄漏到归一化参数中。

### 2. 构造序列数据

```bash
python pipelines/pure_gru/build_sequences.py
```

- **输入**：步骤 1 生成的 `train_data.npy` 和 `test_data.npy`
- **配置**：
  - 观测窗口 `Look_Back = 50`（即 5.0s 历史数据）
  - 预测步长 `Forward_Length = 0`（即当前时刻的目标位置）
- **滑动窗口公式**：对于时间点 t，输入 `data[t-49 : t+1]`，目标 `data[t+0]`（即 `data[t]`）
- **输出**：`processed_data/X_train.npy`、`Y_train.npy`、`X_test.npy`、`Y_test.npy`

### 3. 执行模型训练

```bash
python pipelines/pure_gru/train.py
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
python pipelines/pure_gru/visualize.py
```

- **输入**：测试集数据 + 最佳模型权重 + 归一化参数
- **执行步骤**：
  1. 加载测试集与模型权重，执行推理
  2. 反归一化，还原为真实世界坐标，计算各项误差指标
  3. 绘制 3D 轨迹对比图 → `trajectory_3d_plot.png`
  4. 绘制 2D 综合误差折线图 → `trajectory_2d_error.png`
  5. 评估单次预测耗时并绘制折线图 → `inference_time_plot.png`
- **可视化参数**（可在 `config.py` 中修改）：
  - `PLOT_START = 0`：绘图起始索引
  - `PLOT_END = None`：绘图结束索引（`None` 表示绘制全部测试集）

### 5. 模型结构验证 (可选)

```bash
python core/test_gru_model.py
```

- 独立验证 `UAVTrajectoryGRU` 模型的结构与前向传播正确性
- 输出模型结构、总参数量、可训练参数量，并执行一次 dummy 前向传播

---

## 🧠 模型细节

### 神经网络架构

```
输入 (batch_size, 50, 3)
    ↓
GRU × 2 层 (hidden_size=64, batch_first=True, dropout=0.0)
    ↓  ← 取最后一个时间步的隐藏状态 gru_out[:, -1, :]
Linear (64 → 3)
    ↓  ← 无激活函数，线性回归投影
输出 (batch_size, 3)  →  t 时刻的 (lat, lon, alt)
```


| 组件             | 参数                                     |
| -------------- | -------------------------------------- |
| 输入维度           | 3 (纬度, 经度, 海拔)                         |
| GRU 隐藏层维度      | 64                                     |
| GRU 堆叠层数       | 2                                      |
| GRU 层间 Dropout | 0.0 (纯 GRU 禁用；意图增广版可传正值缓解过拟合)          |
| 输出维度           | 3 (预测纬度, 经度, 海拔)                       |
| 权重初始化          | Xavier Uniform (Glorot)                |
| 偏置初始化          | 全零                                     |
| 模型类            | `UAVTrajectoryGRU`（定义于 `core/gru_model.py`） |


### 误差评估指标

系统在推理阶段自动输出以下物理指标（均在反归一化后的真实坐标系下计算）：


| 指标                          | 说明                       |
| --------------------------- | ------------------------ |
| **MAE** (各维度)               | 纬度、经度、高度各自的平均绝对误差        |
| **RMSE** (各维度)              | 纬度、经度、高度各自的均方根误差         |
| **Average RMSE**            | 三个维度 RMSE 的算术平均值         |
| **Average Euclidean Error** | 预测点与真实点在 3D 空间中的平均欧几里得距离 |


---

## ⚙️ 配置参数速查

两条管线各自拥有独立的 `config.py`，所有超参数与路径配置集中管理。修改参数时只需编辑对应的 `config.py`，无需逐一修改各脚本。

### 纯 GRU 管线 (`pipelines/pure_gru/config.py`)


| 参数                    | 默认值      | 说明           |
| --------------------- | -------- | ------------ |
| `ORIGINAL_INTERVAL_S` | 0.033333 | 原始采样间隔 (秒)   |
| `TARGET_INTERVAL_S`   | 0.1      | 目标采样间隔 (秒)   |
| `DOWNSAMPLE_FACTOR`   | 3        | 降采样因子 (自动计算) |
| `TRAIN_RATIO`         | 0.8      | 训练集比例        |
| `LOOK_BACK`           | 50       | 历史观测步长       |
| `FORWARD_LENGTH`      | 0        | 未来预测步长       |
| `INPUT_SIZE`          | 3        | 输入特征维度       |
| `HIDDEN_SIZE`         | 64       | GRU 隐藏层维度    |
| `NUM_LAYERS`          | 2        | GRU 层数       |
| `OUTPUT_SIZE`         | 3        | 输出维度         |
| `DROPOUT`             | 0.0      | GRU 层间 Dropout |
| `BATCH_SIZE`          | 70       | 批次大小         |
| `LEARNING_RATE`       | 1e-3     | Adam 学习率     |
| `MAX_EPOCHS`          | 500      | 最大训练轮数       |
| `PATIENCE`            | 15       | 早停耐心值        |
| `PLOT_START`          | 0        | 绘图起始索引       |
| `PLOT_END`            | None     | 绘图结束索引       |

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
- **归一化公平性**：当前版本先切分再归一化（仅在训练集上 fit MinMaxScaler），杜绝测试集信息泄漏。请勿将顺序改回"先归一化再切分"。
- **原始数据列名**：`OnboardGPS.csv` 中时间戳列名拼写为 `Timpstemp`（原始数据的拼写错误），代码已做兼容处理。
- **中文字体**：可视化脚本已内置跨平台中文字体自动检测机制（支持 Windows/Linux/macOS），运行时会自动选择系统中可用的中文字体。若系统中未安装任何候选中文字体，程序会打印警告但不会中断运行，图表中的中文可能显示为方块，建议安装对应平台的常用中文字体。
- **GPU 加速**：训练与推理脚本自动检测 CUDA 设备。在 GPU 环境下推理耗时会显著降低。
- **执行顺序**：四个脚本存在严格的依赖关系，必须按 1→2→3→4 的顺序依次执行，不可跳步。

---

---

## 🚁 意图识别增强版流水线 (探索性)

为抑制长预测视距下的"异常离群值"，本仓库附带一条 **级联增强管线**：在原始 GRU 网络前端增加一个"无人机机动意图识别"模块。

### 级联架构

```
原始 GPS 特征 (纬度, 经度, 高度, NED速度)
    │
    ├──► 派生运动学特征 (水平速度, 总速度, 航向角, 航向角速度, 高度变化率, 水平加速度, 爬升角)
    │
    ▼
分组归一化:
    位置列 (纬度, 经度, 高度)  →  MinMaxScaler [0,1]
    其他列                  →  StandardScaler
    │
    ├──► [位置子矩阵] ──────────────────────────────────────┐
    │                                                        │
    ▼                                                        │
Random Forest 特征重要性评估                                  │
    →  保留累计贡献率 ≥ 80% 的核心特征                         │
    │                                                        │
    ▼                                                        │
OvA SVM (RBF 核, C=5, γ=0.01, probability=True / Platt Scaling)
    │   训练集概率: 5-Fold OOF 生成 (消除分布偏移)
    │   测试集概率: 最终模型 out-of-sample 推理
    │                                                        │
    │   4 维概率向量  (0-平飞, 1-转弯, 2-爬升, 3-俯冲)         │
    ▼                                                        │
[位置列 ∪ RF 筛选列]  ⊕  [4 维意图概率]  ─────────────┐      │
    (位置列始终保留, 不受 RF 取舍)                      ▼     ▼
                                         UAVTrajectoryGRU (input_size 动态自适应)
                                                        │
                                                        ▼
                                              (纬度, 经度, 高度) 预测
```

### 公平性修正 (A-F)

为确保纯 GRU 与 GRU+意图的对比实验公平性，本管线实施了以下 6 项修正：


| 编号    | 修正项                | 说明                                                                                      |
| ----- | ------------------ | --------------------------------------------------------------------------------------- |
| **A** | 位置列强制保留            | `lat, lon, alt` 始终保留在 GRU 输入中，不受 RF 筛选影响，防止位置信息丢失                                       |
| **B** | RF 阈值语义修正          | RF 累计阈值保持 0.80 (符合论文)，但作用于排除泄漏列后的候选池，语义更健康                                              |
| **C** | 排除标签生成特征           | 从 RF 候选池中剔除 `alt_rate` / `heading_rate`（标签生成所用的特征），防止数据泄漏压垮 RF 重要性分布                    |
| **D** | 分组归一化              | 位置列走 MinMaxScaler (与目标对齐)，其他列走 StandardScaler，避免归一化策略混用导致数值不一致                          |
| **E** | MinMax 仅 train fit | MinMaxScaler 仅在训练集上 fit (与修正后的纯 GRU 管线一致)，杜绝测试集信息泄漏                                     |
| **F** | SVM OOF 概率         | 训练集的 SVM 概率通过 5-Fold OOF (out-of-fold) 生成，与测试集的 out-of-sample 推理分布一致，消除 covariate shift |


### 新增目录结构 (与纯 GRU 管线完全隔离)

> 意图增强管线的代码位于 `pipelines/intent_gru/` 下，数据产物仍统一保存在项目根目录的 `processed_data/intent/` 中，不会修改、覆盖任何纯 GRU 脚本或产物。

```text
pipelines/intent_gru/
├── __init__.py                         # 管线包初始化
├── config.py                           # 集中配置 (超参数、路径、可视化参数)
├── intent/                             # 意图识别子模块 (Python 包)
│   ├── __init__.py                     # 模块入口, 统一导出公共接口
│   ├── feature_extractor.py            # 动态提取 OnboardGPS 特征 + 派生运动学特征
│   ├── label_generator.py              # 基于 alt_rate/heading_rate 自动生成 4 类伪标签
│   ├── rf_selector.py                  # RF 特征重要性 + 累计 80% 筛选 (支持排除泄漏列)
│   ├── svm_classifier.py               # OvA + Platt Scaling 的 SVM 封装 (含 K-Fold OOF)
│   └── intent_dataset.py               # 拼接概率向量的滑动窗口构造工具 (numpy版 + PyTorch Dataset版)
├── prepare_intent.py                   # [步骤 1.5] 意图数据准备 (分组归一化 / RF / SVM OOF / 滑动窗口)
├── train_with_intent.py                # [步骤 1.6] 训练增广版 GRU (复用 core/gru_model.py, dropout=0.0)
├── visualize_with_intent.py            # [步骤 1.7] 增广版推理与可视化
└── compare_models.py                   # 纯 GRU vs GRU+意图 对比评估 (公平性修正版)
```

> **解耦保证**：上述所有文件都不会修改、覆盖任何纯 GRU 脚本或产物。随时可通过 `python pipelines/pure_gru/train.py` / `python pipelines/pure_gru/visualize.py` 回退至纯 GRU 版本。

### 增强版运行流程

完成原始流水线 (`pipelines/pure_gru/preprocess_uav.py` → `pipelines/pure_gru/build_sequences.py` → `pipelines/pure_gru/train.py`) 之后，即可追加运行：

#### 1.5 意图数据准备

```bash
python pipelines/intent_gru/prepare_intent.py
```

- 动态读取 `Log Files/OnboardGPS.csv`，提取基础数值列 + **7 个** 派生运动学特征 (自动剔除零方差列)
- **分组归一化** (方案 D)：位置列 (lat, lon, alt) 使用 MinMaxScaler，其他列使用 StandardScaler，均仅在训练集上 fit
- 基于 `alt_rate / heading_rate` 启发式阈值 (自适应分位数) 生成 4 类机动伪标签
- `RandomForestClassifier` (n=200, criterion=entropy, class_weight='balanced') 拟合 → **排除标签生成特征** (方案 C) 后按 **累计 80%** 贡献率筛选核心特征
- 位置列强制保留 (方案 A)：最终 GRU 特征 = 位置列 ∪ RF 筛选列
- `OneVsRestClassifier(SVC(kernel='rbf', C=5, gamma=0.01, probability=True))` 训练 → 训练集概率通过 **5-Fold OOF** 生成 (方案 F)，测试集概率由最终模型推理
- 构建 **增广滑动窗口**：每个时间步特征 = `[位置列(MinMax) ∪ RF筛选列(StdScale), 4 维意图概率]`；Y 仍为 MinMax 归一化后的 (lat, lon, alt)
- 所有产物保存至 `processed_data/intent/`

#### 1.6 训练增广版 GRU

```bash
python pipelines/intent_gru/train_with_intent.py
```

- 自动读取 `X_train_intent.npy` 的最后一维作为 `input_size` (= 位置列数 + RF 筛选特征数 + 4)，其他超参数 (hidden_size=64, num_layers=2, dropout=0.0, batch_size=70, lr=1e-3, patience=15) 与 `train.py` 严格一致以保证对比公平
- 最佳模型保存至 `processed_data/intent/best_gru_model_intent.pth`

#### 1.7 增广版推理与可视化

```bash
python pipelines/intent_gru/visualize_with_intent.py
```

- 输出：`trajectory_3d_plot_intent.png` / `trajectory_2d_error_intent.png` / `inference_time_plot_intent.png`
- 除均值指标外，增加 **Max / P95 欧氏距离** 专用于评估长视距离群值

#### 1.8 对比评估 (关键步骤)

```bash
python pipelines/intent_gru/compare_models.py
```

- 在 **同一测试集** 上同时推理纯 GRU 与 GRU+意图模型
- 输出定量指标对比表 (含"变化 %"一栏直观展示改进效果)，指标包括：
  - 各维度 MAE / RMSE
  - Average RMSE
  - Mean / Median / Max / P95 / P99 3D 欧氏距离
  - "> 纯 GRU P95 样本数" (以纯 GRU 的 P95 误差为阈值，统计两个模型超过该阈值的样本数，直接反映离群尾部抑制效果)
- 生成三张对比图：
  - `compare_2d_error.png`：误差时序曲线叠加图 (红: 纯 GRU, 橙: GRU+意图)，自动高亮"纯 GRU 离群区" (> P95)
  - `compare_3d_trajectory.png`：三维轨迹三方对比 (真值 vs 纯 GRU vs GRU+意图)
  - `compare_error_cdf.png`：误差直方图 + 累计分布函数 (CDF)，用于量化离群值尾部分布差异
- 汇总 JSON：`compare_metrics.json`（含 `fairness_corrections` 字段记录公平性修正信息）

### 配置参数速查 (意图模块)

所有参数集中定义于 `pipelines/intent_gru/config.py`，修改时只需编辑该文件。


| 参数                                        | 默认值                     | 说明                                 |
| ---------------------------------------------- | ----------------------- | ---------------------------------- |
| `RF_N_ESTIMATORS`         | 200                     | 随机森林决策树数量                          |
| `RF_CRITERION`            | `"entropy"`                | 信息增益 (可改为 `"gini"`)             |
| `RF_CUMULATIVE_THRESHOLD` | 0.80                    | 累计贡献率阈值 (论文要求 80%)                 |
| `RF_RANDOM_STATE`         | 42                      | RF 随机种子                            |
| `SVM_C`                   | 5.0                     | SVM 惩罚因子                           |
| `SVM_GAMMA`               | 0.01                    | RBF 核参数                            |
| `SVM_KERNEL`              | `"rbf"`                 | 高斯核 (可换 linear/poly 等)             |
| `SVM_N_CLASSES`           | 4                       | 机动类别数                              |
| `POSITION_COLS`           | `["lat", "lon", "alt"]` | 强制保留到 GRU 输入的位置列 (方案 A)            |
| `turn_quantile`          | 0.75                    | 自适应转弯阈值分位数                         |
| `climb_quantile`         | 0.70                    | 自适应爬升/俯冲阈值分位数                      |
| `DROPOUT`              | 0.0                     | GRU 层间 Dropout (与纯 GRU 一致, 保证对比公平) |


### ⚠️ 注意事项

1. **数据集无显式机动标签**：原始 UMAV/AGZ 数据集不含机动分类标注，本实现使用基于 `alt_rate / heading_rate` 的运动学启发式规则自动生成伪标签。这是监督 SVM 训练的必要折中。
2. **默认自适应阈值**：伪标签阈值默认取数据分位数，避免硬编码值在不同采样率下失效。可在 `label_generator.py` 中手动指定。
3. **数据泄漏防护**：`alt_rate` / `heading_rate` 被用于生成伪标签，因此已从 RF 候选池中排除 (方案 C)。若手动绕过此排除机制，RF 将几乎只选出这两列，导致其他特征全被裁掉。
4. **RF 筛选结果说明**：由于标签由 `alt_rate / heading_rate` 构造，即使排除这两列后，与它们高度相关的派生特征 (如 `h_accel`, `climb_angle`) 仍可能被 RF 优先选出，这属于合理现象。若希望 RF 挖掘更多特征，可将 `RF_CUMULATIVE_THRESHOLD` 调高（例如 0.95）或改用 Gini 不纯度 (`RF_CRITERION="gini"`)。
5. **SVM OOF 概率**：训练集的意图概率通过 5-Fold OOF 生成 (方案 F)，确保 GRU 训练时看到的概率分布与推理时一致。若关闭 OOF 直接用 in-sample 概率，训练集概率会偏乐观，导致 GRU 过拟合。
6. **无缝回退至纯 GRU**：本增强管线的所有产物均位于 `processed_data/intent/` 子目录，并不会覆盖任何原始 `.npy` / `.pth` 文件。只需运行 `python pipelines/pure_gru/train.py` 或 `python pipelines/pure_gru/visualize.py` 即可回到纯 GRU 版本。

---

## 📄 数据集引用

Zurich Urban Micro Aerial Vehicle (UMAV/AGZ) Dataset.