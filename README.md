# UAV-Trajectory-Prediction-GRU

基于门控循环单元 (GRU) 神经网络的无人机 (UAV) 三维飞行轨迹预测系统。本项目针对苏黎世城市微型飞行器 (UMAV/AGZ) 数据集进行建模，基于历史观测序列实现对无人机位置（纬度、经度、海拔）的精准预测。

> **🔬 探索性增强**：本仓库额外提供了两条增强管线：
> - **意图识别增强版**（Random Forest 特征筛选 → OvA SVM + Platt Scaling → 拼接至 GRU 输入），用于抑制长预测视距下的离群值。
> - **Attention-Bi-GRU 增强版**（Bi-GRU 编/解码器 + 动态 Seq2Seq Attention + 每层 GeLU 意图融合），严格遵循论文 *"Research on trajectory prediction algorithm based on unmanned aerial vehicles behavioral intentions"* Section 4.2 的公式 (24)-(32) 与 Figure 9/10。
>
> 上述管线与原始纯 GRU 管线并存，随时可以 **无缝回退**。详见下文各管线章节。

---

## 🚀 项目亮点

- **完整序列工程**：涵盖从原始 GPS 数据清洗、降采样、归一化到滑动窗口构建的全流程。
- **集中化配置管理**：三条管线各自拥有独立的 `config.py`，所有超参数与路径配置集中管理，修改参数只需编辑 `config.py`，无需逐一修改各脚本。
- **公平性保障**：所有管线均采用"先切分再归一化"策略，Scaler 仅在训练集上 fit，杜绝测试集信息泄漏。
- **三种模型架构对比**：
  - **纯 GRU 基线**：双层单向 GRU + Linear，简洁高效。
  - **GRU + 意图识别**：在 GRU 输入端拼接 4 维 SVM 机动意图概率向量。
  - **Attention-Bi-GRU + 意图融合**：Bi-GRU 编码器/解码器 + 动态缩放点积 Attention + 每层每步 GeLU 意图条件注入，支持自回归多步解码。
- **精准评估体系**：除常规 MSE 损失外，还计算了物理含义明确的 MAE、RMSE、Average RMSE、3D 空间欧氏距离误差（Mean / Median / Max / P95 / P99）。
- **三维度可视化**：
  - **3D 轨迹图**：直观对比真实轨迹与预测轨迹在三维空间中的重合度。
  - **2D 误差图**：实时分析 3D 欧氏距离综合预测误差随时间步的变化趋势。
  - **推理耗时图**：评估模型单次预测延迟，分析推理性能随时间步的波动情况。
- **双管线对比评估**：提供 `compare_models.py`（纯 GRU vs GRU+意图）和 `compare_all.py`（纯 GRU vs Attention-Bi-GRU+意图）脚本，在统一测试集上进行定量/定性对比，包含误差 CDF、离群值统计、超参数差异声明等多维分析。

---

## 📂 目录结构

```text
UAV-Trajectory-Prediction-GRU/
├── core/                                # 核心共享代码
│   ├── __init__.py                      # 模块初始化 (导出 UAVTrajectoryGRU, AttentionBiGRU 等)
│   ├── gru_model.py                     # GRU 模型类定义 (UAVTrajectoryGRU, 纯 GRU / 意图 GRU 共用)
│   ├── attention_bigru_model.py          # Attention-Bi-GRU 模型类定义 (AttentionBiGRU + 子模块)
│   └── test_gru_model.py                # 模型结构与前向传播验证脚本
│
├── pipelines/
│   ├── __init__.py                      # 管线包初始化
│   │
│   ├── pure_gru/                        # 管线 1: 纯 GRU 基线
│   │   ├── __init__.py                  # 管线包初始化
│   │   ├── config.py                    # 集中配置 (超参数、路径、可视化参数)
│   │   ├── preprocess_uav.py            # [步骤1] 数据预处理与归一化
│   │   ├── build_sequences.py           # [步骤2] 滑动窗口序列构建
│   │   ├── train.py                     # [步骤3] 模型训练与早停优化
│   │   └── visualize.py                 # [步骤4] 测试集推理、多维可视化与耗时评估
│   │
│   ├── intent_gru/                      # 管线 2: 意图增强 GRU (探索性)
│   │   ├── __init__.py                  # 管线包初始化
│   │   ├── config.py                    # 集中配置 (超参数、路径、可视化参数)
│   │   ├── intent/                      # 意图识别子模块 (Python 包)
│   │   │   ├── __init__.py              # 模块入口, 统一导出公共接口
│   │   │   ├── feature_extractor.py     # 动态提取 OnboardGPS 特征 + 派生运动学特征
│   │   │   ├── label_generator.py       # 基于 alt_rate/heading_rate 自动生成 4 类伪标签
│   │   │   ├── rf_selector.py           # RF 特征重要性 + 累计 80% 筛选 (支持排除泄漏列)
│   │   │   ├── svm_classifier.py        # OvA + Platt Scaling 的 SVM 封装 (含 K-Fold OOF)
│   │   │   └── intent_dataset.py        # 拼接概率向量的滑动窗口构造工具
│   │   ├── prepare_intent.py            # [步骤1.5] 意图数据准备
│   │   ├── train_with_intent.py         # [步骤1.6] 训练增广版 GRU
│   │   ├── visualize_with_intent.py     # [步骤1.7] 增广版推理与可视化
│   │   └── compare_models.py            # 纯 GRU vs GRU+意图 对比评估
│   │
│   └── attention_bigru/                 # 管线 3: Attention-Bi-GRU + 意图融合 (论文完整实现)
│       ├── __init__.py                  # 管线包初始化
│       ├── config.py                    # 集中配置 (超参数、路径、可视化参数)
│       ├── prepare_data.py              # [步骤2.1] 数据准备 (StandardScaler + 意图识别)
│       ├── train.py                     # [步骤2.2] 训练 Attention-Bi-GRU
│       ├── visualize.py                 # [步骤2.3] 推理与可视化
│       └── compare_all.py               # 纯 GRU vs Attention-Bi-GRU 对比评估
│
├── references/                          # 参考论文
│   ├── Research on trajectory prediction algorithm based on UAVs behavioral intentions.pdf
│   ├── GRU-based deep learning framework for real-time accurate and scalable UAV trajectory prediction.pdf
│   └── The Zurich urban micro aerial vehicle dataset.pdf
│
├── Log Files/                           # 数据源目录 (需自行放置)
│   └── OnboardGPS.csv                   # 原始飞行日志 (~30Hz)
│
├── processed_data/                      # [自动生成] 预处理中间件与模型产物
│   ├── train_data.npy                   # 归一化后的训练集 (shape: N_train × 3)
│   ├── test_data.npy                    # 归一化后的测试集 (shape: N_test × 3)
│   ├── scaler_params.npz                # Min-Max 缩放参数
│   ├── X_train.npy / Y_train.npy       # 训练集滑动窗口
│   ├── X_test.npy / Y_test.npy         # 测试集滑动窗口
│   ├── best_gru_model.pth              # 纯 GRU 最优模型权重
│   │
│   ├── intent/                          # [自动生成] 意图增强版产物 (与纯 GRU 完全隔离)
│   │   ├── X_train_intent.npy / Y_train_intent.npy
│   │   ├── X_test_intent.npy  / Y_test_intent.npy
│   │   ├── std_scaler.pkl              # StandardScaler (非位置列, 仅 train fit)
│   │   ├── position_minmax_scaler.pkl  # MinMaxScaler (位置列, 仅 train fit)
│   │   ├── minmax_scaler_params.npz    # 目标 (lat,lon,alt) 的 MinMax 参数
│   │   ├── rf_model.pkl               # 随机森林模型
│   │   ├── svm_model.pkl              # OvA SVM 模型
│   │   ├── selected_indices.npy        # 最终 GRU 输入特征索引
│   │   ├── rf_selected_indices.npy     # RF 筛选列索引
│   │   ├── rf_importances.npy          # 完整特征重要性数组
│   │   ├── labels_train.npy / labels_test.npy
│   │   ├── best_gru_model_intent.pth   # 意图增强版最优模型权重
│   │   └── intent_report.json          # 特征名 / RF 重要性 / SVM 指标等汇总
│   │
│   └── attention_bigru/                 # [自动生成] Attention-Bi-GRU 管线产物 (独立隔离)
│       ├── X_train_intent.npy / Y_train_intent.npy
│       ├── X_test_intent.npy  / Y_test_intent.npy
│       ├── scaler_params.npz           # StandardScaler 参数 (目标 Y 的 mean/scale)
│       ├── target_std_scaler.pkl       # 目标 Y 的 StandardScaler
│       ├── feature_std_scaler.pkl      # 全部特征的 StandardScaler
│       ├── rf_model.pkl               # 随机森林模型
│       ├── svm_model.pkl              # OvA SVM 模型
│       ├── selected_indices.npy        # 最终模型输入特征索引
│       ├── rf_selected_indices.npy     # RF 筛选列索引
│       ├── rf_importances.npy          # 完整特征重要性数组
│       ├── labels_train.npy / labels_test.npy
│       ├── best_attention_bigru_model.pth  # Attention-Bi-GRU 最优模型权重
│       └── intent_report.json          # 特征名 / RF 重要性 / SVM 指标等汇总
│
├── .gitignore                           # Git 忽略规则
└── README.md                            # 项目说明文档
```

运行各管线后，项目根目录下还会生成以下可视化输出文件：

```text
UAV-Trajectory-Prediction-GRU/
├── trajectory_3d_plot.png               # [纯 GRU 输出] 3D 轨迹对比图
├── trajectory_2d_error.png              # [纯 GRU 输出] 2D 综合误差折线图
├── inference_time_plot.png              # [纯 GRU 输出] 单次预测耗时折线图
├── trajectory_3d_plot_intent.png        # [意图 GRU 输出] 3D 轨迹对比图
├── trajectory_2d_error_intent.png       # [意图 GRU 输出] 2D 综合误差折线图
├── inference_time_plot_intent.png       # [意图 GRU 输出] 单次预测耗时折线图
├── trajectory_3d_plot_attn_bigru.png    # [Attention-Bi-GRU 输出] 3D 轨迹对比图
├── trajectory_2d_error_attn_bigru.png   # [Attention-Bi-GRU 输出] 2D 综合误差折线图
├── inference_time_plot_attn_bigru.png   # [Attention-Bi-GRU 输出] 单次预测耗时折线图
├── compare_2d_error.png                 # [意图对比输出] 2D 误差曲线对比图
├── compare_3d_trajectory.png            # [意图对比输出] 3D 轨迹对比图
├── compare_error_cdf.png                # [意图对比输出] 误差直方图 + CDF 对比图
├── compare_metrics.json                 # [意图对比输出] 定量指标汇总 JSON
├── compare_2d_error_all.png             # [Attn 对比输出] 2D 误差曲线对比图
├── compare_3d_trajectory_all.png        # [Attn 对比输出] 3D 轨迹对比图
├── compare_error_cdf_all.png            # [Attn 对比输出] 误差直方图 + CDF 对比图
└── compare_metrics_all.json             # [Attn 对比输出] 定量指标汇总 JSON
```

> **注意**：`Log Files/` 目录及 `OnboardGPS.csv` 需自行下载并放置；`processed_data/` 目录及所有 `.npy`、`.npz`、`.pth`、`.pkl` 文件由脚本自动生成，已被 `.gitignore` 忽略。

---

## 🛠️ 环境准备

请确保 Python 版本 >= 3.10（使用了 `X | None` 类型注解语法），并安装以下依赖：

```bash
pip install torch numpy pandas scikit-learn matplotlib
```


| 依赖库            | 用途                                       | 使用模块                                                                                                                                                                                                                                                          |
| -------------- | ---------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `torch`        | GRU / Attention-Bi-GRU 模型构建、训练与推理        | `core/gru_model.py`, `core/attention_bigru_model.py`, `pipelines/pure_gru/train.py`, `pipelines/pure_gru/visualize.py`, `pipelines/intent_gru/train_with_intent.py`, `pipelines/intent_gru/visualize_with_intent.py`, `pipelines/intent_gru/compare_models.py`, `pipelines/attention_bigru/train.py`, `pipelines/attention_bigru/visualize.py`, `pipelines/attention_bigru/compare_all.py` |
| `numpy`        | 数组运算与数据存储                                | 全部模块                                                                                                                                                                                                                          |
| `pandas`       | CSV 读取与数据清洗                              | `pipelines/pure_gru/preprocess_uav.py`, `pipelines/intent_gru/intent/feature_extractor.py`                                                                                                                                    |
| `scikit-learn` | MinMaxScaler / StandardScaler 归一化、RF、SVM | `pipelines/pure_gru/preprocess_uav.py`, `pipelines/intent_gru/prepare_intent.py`, `pipelines/intent_gru/intent/rf_selector.py`, `pipelines/intent_gru/intent/svm_classifier.py`, `pipelines/attention_bigru/prepare_data.py` |
| `matplotlib`   | 3D/2D 可视化绘图                              | `pipelines/pure_gru/visualize.py`, `pipelines/intent_gru/visualize_with_intent.py`, `pipelines/intent_gru/compare_models.py`, `pipelines/attention_bigru/visualize.py`, `pipelines/attention_bigru/compare_all.py`           |


---

## 🏃 运行流程 — 纯 GRU 管线 (Quick Start)

为保证模型训练效果，请严格遵循以下流水线顺序执行：

### 1. 数据预处理

```bash
cd pipelines/pure_gru
python preprocess_uav.py
```

- **输入**：`Log Files/OnboardGPS.csv`（原始飞行日志，30Hz 采样）
- **处理逻辑**：
  1. 提取核心位置特征 `lat, lon, alt`（丢弃速度、航向、精度等非位置参数）
  2. 等间隔降采样：30Hz → **10Hz**（每 3 个点保留 1 个，目标间隔 0.1s）
  3. 按时间顺序 80:20 切分训练集/测试集（**不打乱数据**）
  4. Min-Max 归一化：将纬度/经度/海拔映射至 [0, 1] 区间（**仅在训练集上 fit**，消除数据泄漏）
- **输出**：`processed_data/train_data.npy`、`processed_data/test_data.npy`、`processed_data/scaler_params.npz`

> **公平性修正**：当前版本先切分再归一化，MinMaxScaler 仅在训练集上 fit 后 transform 测试集，避免测试集的 min/max 信息泄漏到归一化参数中。

### 2. 构造序列数据

```bash
python build_sequences.py
```

- **输入**：步骤 1 生成的 `train_data.npy` 和 `test_data.npy`
- **配置**：
  - 观测窗口 `Look_Back = 50`（即 5.0s 历史数据）
  - 预测步长 `Forward_Length = 0`（即当前时刻的目标位置）
- **滑动窗口公式**：对于时间点 t，输入 `data[t-Look_Back+1 : t+1]`，目标 `data[t+Forward_Length]`
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
- **可视化参数**（可在 `config.py` 中修改）：
  - `PLOT_START = 0`：绘图起始索引
  - `PLOT_END = None`：绘图结束索引（`None` 表示绘制全部测试集）

### 5. 模型结构验证 (可选)

```bash
cd ../../
python core/test_gru_model.py
```

- 独立验证 `UAVTrajectoryGRU` 模型的结构与前向传播正确性
- 输出模型结构、总参数量、可训练参数量，并执行一次 dummy 前向传播

---

## 🧠 模型细节

### 模型 1：UAVTrajectoryGRU（纯 GRU 基线 / 意图 GRU 共用）

```
输入 (batch_size, 50, input_size)
    ↓
GRU × 2 层 (hidden_size=64, batch_first=True, dropout=0.0)
    ↓  ← 取最后一个时间步的隐藏状态 gru_out[:, -1, :]
Linear (64 → 3)
    ↓  ← 无激活函数，线性回归投影
输出 (batch_size, 3)  →  t 时刻的 (lat, lon, alt)
```


| 组件             | 参数                                          |
| -------------- | ------------------------------------------- |
| 输入维度           | 3 (纯 GRU) / 动态 (意图 GRU: 位置 + RF 筛选列 + 4 维概率) |
| GRU 隐藏层维度      | 64                                          |
| GRU 堆叠层数       | 2                                           |
| GRU 层间 Dropout | 0.0 (纯 GRU 禁用；意图增广版可传正值缓解过拟合)               |
| 输出维度           | 3 (预测纬度, 经度, 海拔)                            |
| 权重初始化          | Xavier Uniform (Glorot)                     |
| 偏置初始化          | 全零                                          |
| 模型类            | `UAVTrajectoryGRU`（定义于 `core/gru_model.py`） |


### 模型 2：AttentionBiGRU（Attention-Bi-GRU + 意图融合）

严格遵循论文 *"Research on trajectory prediction algorithm based on unmanned aerial vehicles behavioral intentions"* Section 4.2 的公式 (24)-(32) 与 Figure 9/10：

```
输入 x ∈ R^(B × L × input_size),  P_S ∈ R^(B × L × n_intent)
    │
    ▼
┌──── 编码器 (N_enc=4 层, 每层双残差 Add & Norm) ────┐
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
```


| 组件                  | 参数                                                    |
| ------------------- | ----------------------------------------------------- |
| 输入维度 (input_size)  | 动态 (位置 + RF 筛选列，不含意图概率)                               |
| d_model             | 128 (= 2 × hidden_size)                               |
| Bi-GRU 隐藏层维度        | 64                                                    |
| 编码器层数 (N_enc)       | 4                                                     |
| 解码器层数 (N_dec)       | 4                                                     |
| FFN 中间维度 (d_ff)     | 128                                                   |
| 意图概率维度 (n_intent)  | 4                                                     |
| 解码步数 (T_dec)        | 1 (与纯 GRU 对齐；架构支持多步)                                  |
| Dropout             | 0.2 (论文 Table 3)                                     |
| 注意力机制               | 缩放点积 Attention (论文公式 30), Q 每步动态重新计算                   |
| 意图融合方式              | 每层每步独立 GeLU 变换 + Concat 融合 (论文公式 31-32)               |
| 权重初始化               | Xavier Uniform (Glorot)                               |
| 偏置初始化               | 全零 (LayerNorm weight 保持默认全 1)                         |
| 归一化方法               | StandardScaler (论文 Equation 8)                        |
| 模型类                 | `AttentionBiGRU`（定义于 `core/attention_bigru_model.py`） |


### 误差评估指标

系统在推理阶段自动输出以下物理指标（均在反归一化后的真实坐标系下计算）：


| 指标                          | 说明                       |
| --------------------------- | ------------------------ |
| **MAE** (各维度)               | 纬度、经度、高度各自的平均绝对误差        |
| **RMSE** (各维度)              | 纬度、经度、高度各自的均方根误差         |
| **Average RMSE**            | 三个维度 RMSE 的算术平均值         |
| **Mean Euclidean Error**    | 预测点与真实点在 3D 空间中的平均欧几里得距离 |
| **Median Euclidean Error**  | 3D 欧氏距离的中位数              |
| **Max Euclidean Error**     | 3D 欧氏距离的最大值 (离群值指标)      |
| **P95 Euclidean Error**     | 3D 欧氏距离的 95 百分位          |
| **P99 Euclidean Error**     | 3D 欧氏距离的 99 百分位          |


---

## ⚙️ 配置参数速查

三条管线各自拥有独立的 `config.py`，所有超参数与路径配置集中管理。修改参数时只需编辑对应的 `config.py`，无需逐一修改各脚本。

### 纯 GRU 管线 (`pipelines/pure_gru/config.py`)


| 参数                    | 默认值      | 说明             |
| --------------------- | -------- | -------------- |
| `ORIGINAL_INTERVAL_S` | 0.033333 | 原始采样间隔 (秒)     |
| `TARGET_INTERVAL_S`   | 0.1      | 目标采样间隔 (秒)     |
| `DOWNSAMPLE_FACTOR`   | 3        | 降采样因子 (自动计算)   |
| `TRAIN_RATIO`         | 0.8      | 训练集比例          |
| `LOOK_BACK`           | 50       | 历史观测步长         |
| `FORWARD_LENGTH`      | 0        | 未来预测步长         |
| `INPUT_SIZE`          | 3        | 输入特征维度         |
| `HIDDEN_SIZE`         | 64       | GRU 隐藏层维度      |
| `NUM_LAYERS`          | 2        | GRU 层数         |
| `OUTPUT_SIZE`         | 3        | 输出维度           |
| `DROPOUT`             | 0.0      | GRU 层间 Dropout |
| `BATCH_SIZE`          | 70       | 批次大小           |
| `LEARNING_RATE`       | 1e-3     | Adam 学习率       |
| `MAX_EPOCHS`          | 500      | 最大训练轮数         |
| `PATIENCE`            | 15       | 早停耐心值          |
| `PLOT_START`          | 0        | 绘图起始索引         |
| `PLOT_END`            | None     | 绘图结束索引         |


### 意图增强 GRU 管线 (`pipelines/intent_gru/config.py`)


| 参数                        | 默认值                  | 说明                                 |
| ------------------------- | -------------------- | ---------------------------------- |
| `ORIGINAL_INTERVAL_S`     | 0.033333             | 原始采样间隔 (秒)                         |
| `TARGET_INTERVAL_S`       | 0.1                  | 目标采样间隔 (秒)                         |
| `DOWNSAMPLE_FACTOR`       | 3                    | 降采样因子 (自动计算)                       |
| `TRAIN_RATIO`             | 0.8                  | 训练集比例                              |
| `LOOK_BACK`               | 50                   | 历史观测步长                             |
| `FORWARD_LENGTH`          | 0                    | 未来预测步长                             |
| `RF_N_ESTIMATORS`         | 200                  | 随机森林决策树数量                          |
| `RF_CRITERION`            | "entropy"            | 随机森林分裂判据                           |
| `RF_CUMULATIVE_THRESHOLD` | 0.80                 | RF 累计重要性保留阈值                       |
| `RF_RANDOM_STATE`         | 42                   | 随机森林随机种子                           |
| `SVM_C`                   | 5.0                  | SVM 正则化参数                          |
| `SVM_GAMMA`               | 0.01                 | SVM RBF 核宽度参数                      |
| `SVM_KERNEL`              | "rbf"                | SVM 核函数                            |
| `SVM_N_CLASSES`           | 4                    | 机动类别数 (平飞/转弯/爬升/俯冲)                |
| `POSITION_COLS`           | ["lat", "lon", "alt"] | 强制保留的位置列                           |
| `PURE_INPUT_SIZE`         | 3                    | 纯 GRU 输入维度 (对比评估用)                 |
| `HIDDEN_SIZE`             | 64                   | GRU 隐藏层维度                          |
| `NUM_LAYERS`              | 2                    | GRU 层数                             |
| `OUTPUT_SIZE`             | 3                    | 输出维度                               |
| `DROPOUT`                 | 0.0                  | GRU 层间 Dropout (与纯 GRU 一致, 保证对比公平) |
| `BATCH_SIZE`              | 70                   | 批次大小                               |
| `LEARNING_RATE`           | 1e-3                 | Adam 学习率                           |
| `MAX_EPOCHS`              | 500                  | 最大训练轮数                             |
| `PATIENCE`                | 15                   | 早停耐心值                              |
| `PLOT_START`              | 0                    | 绘图起始索引                             |
| `PLOT_END`                | None                 | 绘图结束索引                             |


### Attention-Bi-GRU 管线 (`pipelines/attention_bigru/config.py`)


| 参数                             | 默认值                  | 说明                                              |
| ------------------------------ | -------------------- | ----------------------------------------------- |
| `ORIGINAL_INTERVAL_S`          | 0.033333             | 原始采样间隔 (秒)                                      |
| `TARGET_INTERVAL_S`            | 0.1                  | 目标采样间隔 (秒)                                      |
| `DOWNSAMPLE_FACTOR`            | 3                    | 降采样因子 (自动计算)                                    |
| `TRAIN_RATIO`                  | 0.8                  | 训练集比例                                           |
| `LOOK_BACK`                    | 50                   | 历史观测步长                                          |
| `FORWARD_LENGTH`               | 0                    | 未来预测步长                                          |
| `RF_N_ESTIMATORS`              | 200                  | 随机森林决策树数量                                       |
| `RF_CRITERION`                 | "entropy"            | 随机森林分裂判据                                        |
| `RF_CUMULATIVE_THRESHOLD`      | 0.80                 | RF 累计重要性保留阈值                                    |
| `RF_RANDOM_STATE`              | 42                   | 随机森林随机种子                                        |
| `EXCLUDE_LABEL_LEAK_FEATURES`  | False                | 是否在 RF/SVM 候选池中排除标签生成特征 (本管线默认不排除)              |
| `SVM_C`                        | 5.0                  | SVM 正则化参数                                       |
| `SVM_GAMMA`                    | 0.01                 | SVM RBF 核宽度参数                                   |
| `SVM_KERNEL`                   | "rbf"                | SVM 核函数                                         |
| `SVM_N_CLASSES`                | 4                    | 机动类别数                                           |
| `SVM_OOF_SPLITS`               | 5                    | SVM OOF 折数                                      |
| `POSITION_COLS`                | ["lat", "lon", "alt"] | 强制保留的位置列                                        |
| `HIDDEN_SIZE`                  | 64                   | Bi-GRU 隐藏层维度 (单向)                               |
| `N_ENC_LAYERS`                 | 4                    | 编码器层数 (论文 Table 3)                              |
| `N_DEC_LAYERS`                 | 4                    | 解码器层数 (论文 Table 3)                              |
| `D_FF`                         | 128                  | FFN 中间层维度 (= 2 × hidden_size)                   |
| `DROPOUT`                      | 0.2                  | Dropout (论文 Table 3)                            |
| `OUTPUT_SIZE`                  | 3                    | 输出维度                                            |
| `N_INTENT`                     | 4                    | 意图概率维度                                          |
| `N_DECODE_STEPS`               | 1                    | 解码步数 (与纯 GRU 对齐; 架构支持多步)                       |
| `BATCH_SIZE`                   | 64                   | 批次大小 (论文 Table 3)                               |
| `LEARNING_RATE`                | 1e-3                 | Adam 学习率                                        |
| `MAX_EPOCHS`                   | 300                  | 最大训练轮数 (论文 Table 3)                             |
| `PATIENCE`                     | 15                   | 早停耐心值                                           |
| `PLOT_START`                   | 0                    | 绘图起始索引                                          |
| `PLOT_END`                     | None                 | 绘图结束索引                                          |


---

## 📊 可视化输出

### 纯 GRU 管线

项目运行 `visualize.py` 后将生成以下三张图表：

1. **3D 轨迹对比图** (`trajectory_3d_plot.png`)：展示无人机在三维空间中的实际机动路径（灰色虚线）与 GRU 预测路径（红色实线），标记起点（绿色圆点）与终点（蓝色三角）。
2. **2D 综合误差折线图** (`trajectory_2d_error.png`)：横轴为时间步，纵轴为 3D 欧氏距离综合真实误差，用于分析预测稳定性与误差波动。
3. **推理耗时折线图** (`inference_time_plot.png`)：横轴为时间步，纵轴为单次预测耗时（毫秒），红色虚线标注平均耗时，用于评估模型推理性能。

### Attention-Bi-GRU 管线

运行 `visualize.py` 后将生成以下三张图表：

1. **3D 轨迹对比图** (`trajectory_3d_plot_attn_bigru.png`)：实际轨迹（灰色虚线）vs Attention-Bi-GRU + 意图预测轨迹（蓝色实线）。
2. **2D 综合误差折线图** (`trajectory_2d_error_attn_bigru.png`)：3D 欧氏距离误差随时间步变化。
3. **推理耗时折线图** (`inference_time_plot_attn_bigru.png`)：单次预测耗时（毫秒），含平均耗时标注。

---

## ⚠️ 注意事项

- **数据集放置**：运行步骤 1 前，需将 `OnboardGPS.csv` 放置于 `Log Files/` 目录下。该文件来自 UMAV 数据集，需自行下载。
- **时序切分**：本项目为连续时间序列任务，训练集/测试集严格按时间顺序划分，**不可使用随机打乱切分**。
- **归一化公平性**：当前版本先切分再归一化（Scaler 仅在训练集上 fit），杜绝测试集信息泄漏。请勿将顺序改回"先归一化再切分"。
- **归一化方法差异**：纯 GRU 和意图 GRU 管线使用 MinMaxScaler [0,1]；Attention-Bi-GRU 管线使用 StandardScaler（论文 Equation 8 Z-score）。反归一化时需使用各自对应的逆变换。
- **原始数据列名**：`OnboardGPS.csv` 中时间戳列名拼写为 `Timpstemp`（原始数据的拼写错误），代码已做兼容处理。
- **中文字体**：可视化脚本已内置跨平台中文字体自动检测机制（支持 Windows/Linux/macOS），运行时会自动选择系统中可用的中文字体。若系统中未安装任何候选中文字体，程序会打印警告但不会中断运行，图表中的中文可能显示为方块，建议安装对应平台的常用中文字体。
- **GPU 加速**：训练与推理脚本自动检测 CUDA 设备。在 GPU 环境下推理耗时会显著降低。
- **执行顺序**：各管线的脚本存在严格的依赖关系，必须按标注的步骤顺序依次执行，不可跳步。
- **Python 版本**：Attention-Bi-GRU 模型代码使用了 `X | None` 类型注解语法，需要 Python >= 3.10。

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


### 运行流程

> **前置条件**：需先完成纯 GRU 管线的步骤 1-3（数据预处理、序列构建、模型训练），因为 `compare_models.py` 需要加载纯 GRU 的测试集数据和模型权重进行对比。

```bash
cd pipelines/intent_gru
```

#### 步骤 1.5：意图数据准备

```bash
python prepare_intent.py
```

- **输入**：`Log Files/OnboardGPS.csv`
- **处理逻辑**：
  1. 动态提取 OnboardGPS.csv 中的数值特征 + 派生运动学特征（水平速度、航向角、航向角速率、高度变化率等）
  2. 分组归一化：位置列 MinMaxScaler / 其他列 StandardScaler（均仅在训练集上 fit）
  3. 基于 `alt_rate` / `heading_rate` 自动生成 4 类机动伪标签（平飞/转弯/爬升/俯冲）
  4. Random Forest 特征重要性评估 → 保留累计贡献率 ≥ 80% 的核心特征（自动剔除标签生成特征防泄漏）
  5. OvA SVM + Platt Scaling 训练（训练集概率使用 5-Fold OOF 生成）
  6. 构建增广滑动窗口序列：`GRU 输入 = [位置列 ∪ RF 筛选列] ⊕ [4 维 SVM 概率]`
- **输出**：`processed_data/intent/` 下的全部产物

#### 步骤 1.6：训练增广版 GRU

```bash
python train_with_intent.py
```

- **输入**：步骤 1.5 生成的增广序列数据
- **说明**：与纯 GRU 的 `train.py` 功能等价，唯一区别在于 `input_size` 由数据动态决定（而非固定为 3），模型权重保存至 `processed_data/intent/best_gru_model_intent.pth`
- **输出**：`processed_data/intent/best_gru_model_intent.pth`

#### 步骤 1.7：增广版推理与可视化

```bash
python visualize_with_intent.py
```

- **输入**：意图增广测试集 + 增广模型权重 + 归一化参数
- **输出**：
  - `trajectory_3d_plot_intent.png`：3D 轨迹对比图
  - `trajectory_2d_error_intent.png`：2D 综合误差折线图
  - `inference_time_plot_intent.png`：单次预测耗时折线图

#### 步骤 1.8：双管线对比评估 (可选)

```bash
python compare_models.py
```

- **前置条件**：纯 GRU 管线（步骤 1-3）和意图增强管线（步骤 1.5-1.6）均已完成
- **功能**：在统一测试集上对纯 GRU 与 GRU+意图模型进行定量/定性对比
- **对比内容**：
  1. **定量指标表**：MAE、RMSE、Average RMSE、Mean/Median/Max/P95/P99 欧氏距离
  2. **2D 误差曲线对比**：红色为纯 GRU，橙色为 GRU+意图
  3. **3D 轨迹对比**：灰色虚线为真值，红色为纯 GRU 预测，橙色为 GRU+意图预测
  4. **误差直方图 + CDF 对比**：量化离群值尾部分布差异
- **输出**：
  - `compare_2d_error.png`：2D 误差曲线对比图
  - `compare_3d_trajectory.png`：3D 轨迹对比图
  - `compare_error_cdf.png`：误差直方图 + CDF 对比图
  - `compare_metrics.json`：定量指标汇总 JSON

---

## 🔬 Attention-Bi-GRU 增强版流水线 (论文完整实现)

严格遵循论文 *"Research on trajectory prediction algorithm based on unmanned aerial vehicles behavioral intentions"* (Drones 2025, 9, 640) Section 4.2 的完整 Seq2Seq 架构实现，包含 Bi-GRU 编码器/解码器、动态缩放点积 Attention、每层每步 GeLU 意图融合和自回归多步解码。

### 与其他管线的关键差异


| 维度         | 纯 GRU                     | 意图 GRU                          | Attention-Bi-GRU                                       |
| ---------- | -------------------------- | -------------------------------- | ------------------------------------------------------ |
| 模型架构       | 单向 GRU → Linear            | 单向 GRU → Linear (输入增广)           | Bi-GRU 编/解码器 + 动态 Attention + 每层意图融合                   |
| 意图融合方式     | 无                          | 输入端拼接 4 维概率向量                    | 每层每步 GeLU 变换 + Concat 条件注入 (论文公式 31-32)               |
| 注意力机制      | 无                          | 无                                | 动态 Seq2Seq 缩放点积 Attention (Q 每步重新计算, 论文公式 27/30)      |
| 归一化方法      | MinMaxScaler [0,1]         | 位置列 MinMax / 其他列 Standard        | **StandardScaler (论文 Equation 8)**                     |
| 编/解码层数     | 2 层 GRU                    | 2 层 GRU                          | 编码器 4 层 + 解码器 4 层                                      |
| Dropout    | 0.0                        | 0.0                              | 0.2 (论文 Table 3)                                      |
| 学习率        | 1e-3                       | 1e-3                             | 1e-3                                                   |
| Batch Size | 70                         | 70                               | 64 (论文 Table 3)                                        |
| Max Epochs | 500                        | 500                              | 300 (论文 Table 3)                                       |
| 解码步数       | 1                          | 1                                | 1 (架构支持多步)                                             |
| 数据产物目录     | `processed_data/`          | `processed_data/intent/`         | `processed_data/attention_bigru/`                      |

### 运行流程

> **前置条件**：需先完成纯 GRU 管线的步骤 1-3，因为 `compare_all.py` 需要加载纯 GRU 的测试集数据和模型权重进行对比。

```bash
cd pipelines/attention_bigru
```

#### 步骤 2.1：数据准备 (StandardScaler + 意图识别)

```bash
python prepare_data.py
```

- **输入**：`Log Files/OnboardGPS.csv`
- **处理逻辑**：
  1. 动态提取 OnboardGPS.csv 中的数值特征 + 派生运动学特征
  2. **全部特征使用 StandardScaler 归一化**（论文 Equation 8 Z-score），包括位置列——这是与 intent_gru 管线的核心差异
  3. 基于 `alt_rate` / `heading_rate` 自动生成 4 类机动伪标签
  4. Random Forest 特征重要性评估 → 保留累计贡献率 ≥ 80% 的核心特征
  5. OvA SVM + Platt Scaling 训练（训练集概率使用 5-Fold OOF 生成）
  6. 构建增广滑动窗口序列：`模型输入 = [StandardScaler(位置列 ∪ RF 筛选列)] ⊕ [4 维 SVM 概率]`
- **与 intent_gru/prepare_intent.py 的核心差异**：

  | 维度       | intent_gru/prepare_intent.py | attention_bigru/prepare_data.py    |
  | -------- | ---------------------------- | ---------------------------------- |
  | 位置列归一化   | MinMaxScaler [0, 1]          | StandardScaler (论文 Eq.8)           |
  | 其他列归一化   | StandardScaler               | StandardScaler (一致)                |
  | 目标 Y 归一化 | MinMaxScaler [0, 1]          | StandardScaler (论文 Eq.8)           |
  | 产物目录     | processed_data/intent/       | processed_data/attention_bigru/    |
  | 意图识别子模块  | 直接调用                        | import 复用 (pipelines.intent_gru.intent) |

- **输出**：`processed_data/attention_bigru/` 下的全部产物

#### 步骤 2.2：训练 Attention-Bi-GRU

```bash
python train.py
```

- **输入**：步骤 2.1 生成的 StandardScaler 归一化增广数据
- **训练参数**（论文 Table 3）：
  - Batch Size = 64
  - 学习率 = 1e-3（Adam 优化器）
  - 最大 Epochs = 300
  - 损失函数 = MSELoss
  - 早停 Patience = 15
  - Dropout = 0.2
- **数据拆分**：增广 X 的最后 4 列为 SVM 概率（`n_intent=4`），其余为结构性特征，分别传入模型的 `x` 和 `intent_probs` 参数
- **输出**：`processed_data/attention_bigru/best_attention_bigru_model.pth`

#### 步骤 2.3：推理与可视化

```bash
python visualize.py
```

- **输入**：StandardScaler 归一化测试集 + 训练好的模型权重
- **反归一化**：使用 StandardScaler 逆变换 `X_original = X_scaled * σ + μ`（论文 Equation 8 的逆运算）
- **输出**：
  - `trajectory_3d_plot_attn_bigru.png`：3D 轨迹对比图（实际轨迹 vs Attention-Bi-GRU + 意图预测轨迹）
  - `trajectory_2d_error_attn_bigru.png`：2D 综合误差折线图
  - `inference_time_plot_attn_bigru.png`：单次预测耗时折线图

#### 步骤 2.4：双管线对比评估 (可选)

```bash
python compare_all.py
```

- **前置条件**：纯 GRU 管线（步骤 1-3）和 Attention-Bi-GRU 管线（步骤 2.1-2.2）均已完成
- **功能**：在统一测试集上对纯 GRU 与 Attention-Bi-GRU + 意图模型进行定量/定性对比
- **对比内容**：
  1. **定量指标表**：MAE、RMSE、Average RMSE、Mean/Median/Max/P95/P99 欧氏距离，以及超 pure_gru P95 阈值的离群样本数
  2. **超参数对比表**：明确标注两条管线的归一化方法、架构、学习率等差异
  3. **2D 误差曲线对比**：红色为纯 GRU，蓝色为 Attention-Bi-GRU + 意图，高亮离群值区间
  4. **3D 轨迹对比**：灰色虚线为真值，红色为纯 GRU 预测，蓝色为 Attention-Bi-GRU + 意图预测
  5. **误差直方图 + CDF 对比**：量化离群值尾部分布差异
- **公平性声明**：
  - ✅ 同一数据源 (OnboardGPS.csv) 与同一 80:20 时序切分
  - ✅ 两条管线 Scaler 均仅在训练集上 fit
  - ⚠️ 归一化方法不同 (MinMax vs Standard)
  - ⚠️ 超参数差异 (lr / batch / dropout / epochs / 架构)，性能差异不能完全归因于架构改进
- **输出**：
  - `compare_2d_error_all.png`：2D 误差曲线对比图
  - `compare_3d_trajectory_all.png`：3D 轨迹对比图
  - `compare_error_cdf_all.png`：误差直方图 + CDF 对比图
  - `compare_metrics_all.json`：定量指标 + 超参数汇总 JSON

### 有意偏差与混淆风险说明

Attention-Bi-GRU 管线与纯 GRU 基线之间存在以下有意偏差，在解读对比结果时需注意：

| 偏差项       | 纯 GRU       | Attention-Bi-GRU    | 偏差理由                   | 混淆风险 |
| --------- | ------------ | ------------------- | ---------------------- | ---- |
| 归一化方法     | MinMaxScaler | StandardScaler (论文) | 论文 Equation 8 明确使用 Z-score | 高    |
| 学习率       | 1e-3         | 1e-3                | —                      | 无    |
| Batch Size | 70           | 64 (论文)             | 论文 Table 3 设定          | 中    |
| Dropout   | 0.0          | 0.2 (论文)            | 论文 Table 3 设定          | 高    |
| Max Epochs | 500          | 300 (论文)            | 论文 Table 3 设定          | 低    |
| Look Back | 50           | 50                  | 与纯 GRU 对齐              | 无    |

> **建议后续补充**：增加一组使用与纯 GRU 相同超参数 (lr=1e-3, batch=70, dropout=0.0) 的 Attention-Bi-GRU 作为对照，以隔离超参数影响。

---

## 📚 参考文献


| 编号  | 文献信息                                                                                                                                                              | 对应技术模块                                                                                                   |
| --- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------- |
| 1   | Yoon S, Jang D, Yoon H, et al. GRU-based deep learning framework for real-time, accurate, and scalable UAV trajectory prediction[J]. Drones, 2025, 9(2): 142-168.  | **纯 GRU 管线**（双层 GRU 架构、滑动窗口序列构建、训练与早停策略）的实现方法参考自该论文；**意图识别增强管线**（将机动意图概率向量拼接至 GRU 输入以抑制离群值）的核心思想同样源自该论文。 |
| 2   | Cao Y, Zhang J D, Shi G Q, et al. Research on trajectory prediction algorithm based on unmanned aerial vehicles behavioral intentions[J]. Drones, 2025, 9(9): 640. | 意图识别子模块中机动类别划分（平飞/转弯/爬升/俯冲）与 RF+SVM 级联分类方案的设计参考了该论文的行为意图建模思路；**Attention-Bi-GRU 管线**的完整 Seq2Seq 架构（编码器/解码器 + 动态 Attention + 每层 GeLU 意图融合）严格遵循该论文 Section 4.2 公式 (24)-(32) 与 Figure 9/10。 |
| 3   | Majdik A L, Till C, Scaramuzza D. The Zurich urban micro aerial vehicle dataset[J]. The International Journal of Robotics Research, 2017, 36(3): 269-273.          | 实验所用飞行数据集（`OnboardGPS.csv`）来源于该论文公开的 UMAV/AGZ 数据集。                                                       |
