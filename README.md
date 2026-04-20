# UAV-Trajectory-Prediction-GRU

基于门控循环单元 (GRU) 神经网络的无人机 (UAV) 三维飞行轨迹预测系统。本项目针对苏黎世城市微型飞行器 (UMAV/AGZ) 数据集进行建模，基于历史观测序列实现对无人机位置（纬度、经度、海拔）的精准预测。

> **🔬 探索性增强**：本仓库额外提供了一个 **完全解耦的** "机动意图识别" 级联模块（Random Forest 特征筛选 → OvA SVM + Platt Scaling → 拼接至 GRU 输入），用于抑制长预测视距下的离群值。该模块与原始纯 GRU 管线并存，随时可以 **无缝回退**。详见文末的 **[意图识别增强版流水线](#-意图识别增强版流水线-探索性)** 章节。

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
- **中文字体**：可视化脚本已内置跨平台中文字体自动检测机制（支持 Windows/Linux/macOS），运行时会自动选择系统中可用的中文字体。若系统中未安装任何候选中文字体，程序会打印警告但不会中断运行，图表中的中文可能显示为方块，建议安装对应平台的常用中文字体。
- **GPU 加速**：训练与推理脚本自动检测 CUDA 设备。在 GPU 环境下推理耗时会显著降低。
- **执行顺序**：四个脚本存在严格的依赖关系，必须按 1→2→3→4 的顺序依次执行，不可跳步。

---

---

## 🚁 意图识别增强版流水线 (探索性)

为抑制长预测视距下的"异常离群值"，本仓库附带一条 **级联增强管线**：在原始 GRU 网络前端增加一个"无人机机动意图识别"模块。

### 级联架构

```
原始 GPS 特征 (lat, lon, alt, v_n/e/d, eph_m, …)
    │
    ├──► 派生运动学特征 (h_speed, heading_rate, alt_rate, h_accel, climb_angle, …)
    │
    ▼
StandardScaler (train fit → all transform)
    │
    ▼
Random Forest 特征重要性评估  →  保留累计贡献率 ≥ 80% 的核心特征
    │
    ▼
OvA SVM (RBF 核, C=5, γ=0.01, probability=True / Platt Scaling)
    │                                                        │
    │   4 维概率向量  (0-平飞, 1-转弯, 2-爬升, 3-俯冲)       │
    ▼                                                        │
[筛选后的标准化特征]  ⊕  [4 维意图概率]  ──────────────┐     │
                                                        ▼     ▼
                                         UAVTrajectoryGRU (input_size 动态自适应)
                                                        │
                                                        ▼
                                              (lat, lon, alt) 预测
```

### 新增目录结构 (与原管线完全隔离)

```text
UAV-Trajectory-Prediction-GRU/
├── intent/                             # [新增] 意图识别模块 (Python 包)
│   ├── feature_extractor.py            # 动态提取 OnboardGPS 特征 + 派生运动学特征
│   ├── label_generator.py              # 基于 alt_rate/heading_rate 自动生成 4 类伪标签
│   ├── rf_selector.py                  # RF 特征重要性 + 累计 80% 筛选
│   ├── svm_classifier.py               # OvA + Platt Scaling 的 SVM 封装
│   └── intent_dataset.py               # 拼接概率向量的滑动窗口构造工具
├── prepare_intent.py                   # [新增] 步骤 1.5: 意图数据准备 (StandardScaler / RF / SVM / 滑动窗口)
├── train_with_intent.py                # [新增] 训练增广版 GRU (复用 UAVTrajectoryGRU 类)
├── visualize_with_intent.py            # [新增] 增广版推理与可视化
├── compare_models.py                   # [新增] 纯 GRU vs GRU+意图 对比评估
├── processed_data/
│   └── intent/                         # [自动生成] 增广版产物目录 (与纯 GRU 完全隔离)
│       ├── X_train_intent.npy / Y_train_intent.npy
│       ├── X_test_intent.npy  / Y_test_intent.npy
│       ├── std_scaler.pkl / rf_model.pkl / svm_model.pkl
│       ├── selected_indices.npy / rf_importances.npy
│       ├── labels_train.npy / labels_test.npy
│       ├── minmax_scaler_params.npz
│       ├── best_gru_model_intent.pth
│       └── intent_report.json          # 特征名 / RF 重要性 / SVM 指标 / 标签分布等汇总
└── (compare_*.png / compare_metrics.json 对比评估输出)
```

> **解耦保证**：上述所有新增文件都不会修改、覆盖任何原始脚本或产物。随时可通过 `python train.py` / `python visualize.py` 回退至纯 GRU 版本。

### 增强版运行流程

完成原始流水线 (`preprocess_uav.py` → `build_sequences.py` → `train.py`) 之后，即可追加运行：

#### 1.5 意图数据准备
```bash
python prepare_intent.py
```
- 动态读取 `Log Files/OnboardGPS.csv`，提取 **10 个** 基础数值列 + **7 个** 派生运动学特征 (自动剔除零方差列)
- `StandardScaler` 标准化 (仅在训练集上 fit，测试集 transform)
- 基于 `alt_rate / heading_rate` 启发式阈值 (自适应分位数) 生成 4 类机动伪标签
- `RandomForestClassifier` (n=200, class_weight='balanced') 拟合 → 按 **累计 80%** 贡献率筛选核心特征
- `OneVsRestClassifier(SVC(kernel='rbf', C=5, gamma=0.01, probability=True))` 训练并做全量概率推理
- 构建 **增广滑动窗口**：每个时间步特征 = `[筛选后的标准化特征, 4 维意图概率]`
- 所有产物保存至 `processed_data/intent/`

#### 1.6 训练增广版 GRU
```bash
python train_with_intent.py
```
- 自动读取 `X_train_intent.npy` 的最后一维作为 `input_size` (= 筛选特征数 + 4)，其他超参数与 `train.py` 严格一致以保证对比公平
- 最佳模型保存至 `processed_data/intent/best_gru_model_intent.pth`

#### 1.7 增广版推理与可视化
```bash
python visualize_with_intent.py
```
- 输出：`trajectory_3d_plot_intent.png` / `trajectory_2d_error_intent.png` / `inference_time_plot_intent.png`
- 除均值指标外，增加 **Max / P95 / P99 欧氏距离** 专用于评估长视距离群值

#### 1.8 对比评估 (关键步骤)
```bash
python compare_models.py
```
- 在 **同一测试集** 上同时推理纯 GRU 与 GRU+意图模型
- 输出定量指标对比表 (含"变化 %"一栏直观展示改进效果)
- 生成三张对比图：
    - `compare_2d_error.png`：误差时序曲线叠加图 (红: 纯 GRU, 橙: GRU+意图)，自动高亮"纯 GRU 离群区"
    - `compare_3d_trajectory.png`：三维轨迹三方对比 (真值 vs 纯 GRU vs GRU+意图)
    - `compare_error_cdf.png`：误差直方图 + 累计分布函数 (CDF)，用于量化离群值尾部分布差异
- 汇总 JSON：`compare_metrics.json`

### 配置参数速查 (意图模块)

| 脚本 / 参数 | 默认值 | 说明 |
|------|------|------|
| `prepare_intent.py :: RF_N_ESTIMATORS` | 200 | 随机森林决策树数量 |
| `prepare_intent.py :: RF_CRITERION` | `"gini"` | 可改为 `"entropy"` (信息增益) |
| `prepare_intent.py :: RF_CUMULATIVE_THRESHOLD` | 0.80 | 累计贡献率阈值 (论文要求 80%) |
| `prepare_intent.py :: SVM_C` | 5.0 | SVM 惩罚因子 |
| `prepare_intent.py :: SVM_GAMMA` | 0.01 | RBF 核参数 |
| `prepare_intent.py :: SVM_KERNEL` | `"rbf"` | 高斯核 (可换 linear/poly 等) |
| `label_generator.py :: turn_quantile` | 0.75 | 自适应转弯阈值分位数 |
| `label_generator.py :: climb_quantile` | 0.70 | 自适应爬升/俯冲阈值分位数 |

### ⚠️ 注意事项

1. **数据集无显式机动标签**：原始 UMAV/AGZ 数据集不含机动分类标注，本实现使用基于 `alt_rate / heading_rate` 的运动学启发式规则自动生成伪标签。这是监督 SVM 训练的必要折中。
2. **默认自适应阈值**：伪标签阈值默认取数据分位数，避免硬编码值在不同采样率下失效。可在 `label_generator.py` 中手动指定。
3. **RF 筛选结果与标签生成强相关**：默认伪标签由 `alt_rate / heading_rate` 构造，因此 RF 通常会优先选出这两个特征（符合预期）。若希望 RF 挖掘更多特征，可将 `RF_CUMULATIVE_THRESHOLD` 调高（例如 0.95）或改用信息增益 (`RF_CRITERION="entropy"`)。
4. **无缝回退至纯 GRU**：本增强管线的所有产物均位于 `processed_data/intent/` 子目录，并不会覆盖任何原始 `.npy` / `.pth` 文件。只需运行 `python train.py` 或 `python visualize.py` 即可回到纯 GRU 版本。

---

## 📄 数据集引用

Zurich Urban Micro Aerial Vehicle (UMAV/AGZ) Dataset.
