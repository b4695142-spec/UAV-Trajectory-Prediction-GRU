"""
==============================================================================
意图识别 — 拼接 SVM 概率向量的滑动窗口数据集
==============================================================================
本模块提供两种接口:

1. `build_augmented_sequences()` — 纯 numpy 函数
   基于已有的 (逐时刻) 筛选特征矩阵 + SVM 概率矩阵构建滑动窗口，
   输出形状为 (n, look_back, n_selected + n_classes) 的增广输入张量。

2. `IntentAugmentedDataset` — PyTorch Dataset
   在线式用法 (不预先构造全部滑动窗口，而是在 __getitem__ 时动态切片)。
   适合内存受限或希望 SVM 推理延迟分摊到训练迭代中的场景。

两种接口实现了同一套"拼接意图概率"的语义，可按需选用。
==============================================================================
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


# =========================================================================
# 1) 纯 numpy 版本 (推荐，用于离线预构建 .npy 产物)
# =========================================================================
def build_augmented_sequences(
    feat_selected: np.ndarray,
    intent_probs: np.ndarray,
    target: np.ndarray,
    look_back: int,
    forward_length: int = 0,
    dataset_name: str = "dataset",
    verbose: bool = True,
    decode_steps: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    基于"筛选后的标准化特征 + SVM 概率"构建滑动窗口增广序列。

    数学定义:
        对有效时间点 t (look_back - 1 <= t <= N - forward_length - decode_steps):
            X[i][s] = concat(feat_selected[t-look_back+1+s], intent_probs[t-look_back+1+s])
            Y[i]    = target[t + forward_length : t + forward_length + decode_steps]

        当 decode_steps == 1 时, Y 自动 squeeze 为 (n_valid, 3), 与单步预测管线
        (pure_gru / intent_gru) 完全兼容; 当 decode_steps > 1 时, Y 形状为
        (n_valid, decode_steps, 3), 用于 Attention-Bi-GRU 的自回归多步训练。

    参数:
        feat_selected:     shape (N, n_selected), StandardScaler 标准化 + RF 筛选后的特征
        intent_probs:      shape (N, n_classes), 逐时刻 SVM 概率向量 (n_classes=4)
        target:            shape (N, 3), 归一化后的 (lat, lon, alt), 作为回归目标
        look_back:         历史观测步长 (默认用主管线的 50)
        forward_length:    未来预测起点偏移 (默认 0 = 当前时刻)
        dataset_name:      日志用名称
        verbose:           是否打印日志
        decode_steps:      解码步数 (默认 1 = 单步, 与 pure_gru / intent_gru 兼容)
                           > 1 时 Y 形状变为 (n_valid, decode_steps, 3)

    返回:
        X: np.ndarray, shape (n_valid, look_back, n_selected + n_classes)
        Y: np.ndarray, shape (n_valid, 3)              当 decode_steps == 1
                       shape (n_valid, decode_steps, 3) 当 decode_steps > 1
    """
    if decode_steps < 1:
        raise ValueError(f"decode_steps 必须 >= 1, 当前 = {decode_steps}")

    # --- 形状校验 -------------------------------------------------------
    N = feat_selected.shape[0]
    if intent_probs.shape[0] != N:
        raise ValueError(
            f"feat_selected ({N}) 与 intent_probs ({intent_probs.shape[0]}) 行数不一致"
        )
    if target.shape[0] != N:
        raise ValueError(
            f"feat_selected ({N}) 与 target ({target.shape[0]}) 行数不一致"
        )

    t_min = look_back - 1
    # 多步: 需要保留 decode_steps 个未来步, 因此上界整体左移 (decode_steps - 1)
    t_max = N - forward_length - decode_steps
    if t_min > t_max:
        raise ValueError(
            f"[{dataset_name}] 数据量不足 (N={N}, "
            f"need >= {look_back + forward_length + decode_steps - 1})"
        )

    # --- 预先拼接 (N, n_selected + n_classes) --------------------------
    augmented = np.concatenate([feat_selected, intent_probs], axis=1)
    n_feat = augmented.shape[1]
    n_valid = t_max - t_min + 1
    out_dim = target.shape[1]

    if verbose:
        print(f"  [{dataset_name}] 构建增广滑动窗口序列")
        print(f"    N = {N}, Look_Back = {look_back}, Forward_Length = {forward_length}, "
              f"Decode_Steps = {decode_steps}")
        print(f"    单步特征维度 = {n_feat} (= {feat_selected.shape[1]} 筛选特征"
              f" + {intent_probs.shape[1]} 意图概率)")
        print(f"    有效样本数量 = {n_valid}")

    # --- 生成 X, Y ------------------------------------------------------
    X = np.empty((n_valid, look_back, n_feat), dtype=np.float32)

    if decode_steps == 1:
        Y = np.empty((n_valid, out_dim), dtype=np.float32)
        for i, t in enumerate(range(t_min, t_max + 1)):
            X[i] = augmented[t - look_back + 1: t + 1]
            Y[i] = target[t + forward_length]
    else:
        Y = np.empty((n_valid, decode_steps, out_dim), dtype=np.float32)
        for i, t in enumerate(range(t_min, t_max + 1)):
            X[i] = augmented[t - look_back + 1: t + 1]
            start = t + forward_length
            Y[i] = target[start: start + decode_steps]

    if verbose:
        print(f"    ✅ X shape: {X.shape}  |  Y shape: {Y.shape}\n")

    return X, Y


# =========================================================================
# 2) 在线式 PyTorch Dataset (备用，适合边训边做 SVM 推理的场景)
# =========================================================================
class IntentAugmentedDataset(Dataset):
    """
    将"逐时刻的标准化特征 + SVM 概率 (可选延迟计算) + MinMax 目标"封装为
    PyTorch Dataset，每次 __getitem__ 动态切出一个滑动窗口。

    典型用法 (已预先计算好 SVM 概率):

        ds = IntentAugmentedDataset(
            feat_selected=feat_sel_np,    # (N, n_selected)
            target=target_np,             # (N, 3)
            intent_probs=probs_np,        # (N, 4), 已由 SVM 预先计算
            look_back=50,
        )
        loader = DataLoader(ds, batch_size=70, shuffle=True)

    若希望在 __getitem__ 里动态调用 SVM，可传入 svm_classifier 参数 (见构造)，
    但通常离线预计算更高效，因此默认关闭此路径。
    """

    def __init__(
        self,
        feat_selected: np.ndarray,
        target: np.ndarray,
        intent_probs: Optional[np.ndarray] = None,
        svm_classifier=None,              # 可选: IntentSVMClassifier 实例
        look_back: int = 50,
        forward_length: int = 0,
    ):
        if feat_selected.ndim != 2:
            raise ValueError("feat_selected 必须是 (N, n_selected) 二维数组")
        if target.ndim != 2:
            raise ValueError("target 必须是 (N, 3) 二维数组")
        if feat_selected.shape[0] != target.shape[0]:
            raise ValueError("feat_selected 与 target 行数需一致")

        self.feat_selected = feat_selected.astype(np.float32)
        self.target = target.astype(np.float32)
        self.look_back = int(look_back)
        self.forward_length = int(forward_length)

        # 概率矩阵 (可延迟计算)
        if intent_probs is None:
            if svm_classifier is None:
                raise ValueError("必须提供 intent_probs 或 svm_classifier")
            intent_probs = svm_classifier.predict_proba(feat_selected)
        self.intent_probs = intent_probs.astype(np.float32)

        if self.intent_probs.shape[0] != self.feat_selected.shape[0]:
            raise ValueError("intent_probs 与 feat_selected 行数不一致")

        # 预拼接 (内存成本较低，换来更快的采样)
        self._augmented = np.concatenate(
            [self.feat_selected, self.intent_probs], axis=1
        )

        # 计算有效索引范围
        self._t_min = self.look_back - 1
        self._t_max = self.target.shape[0] - self.forward_length - 1
        if self._t_min > self._t_max:
            raise ValueError(
                f"数据长度不足: N={self.target.shape[0]}, "
                f"需要至少 {self.look_back + self.forward_length}"
            )
        self._n_samples = self._t_max - self._t_min + 1

    # ---------------------------------------------------------------- #
    def __len__(self) -> int:
        return self._n_samples

    def __getitem__(self, idx: int):
        if idx < 0 or idx >= self._n_samples:
            raise IndexError(f"索引 {idx} 越界 [0, {self._n_samples})")

        t = self._t_min + idx
        x_seq = self._augmented[t - self.look_back + 1: t + 1]    # (L, n_feat)
        y = self.target[t + self.forward_length]                  # (3,)
        return (
            torch.from_numpy(x_seq).float(),
            torch.from_numpy(y).float(),
        )

    # ---------------------------------------------------------------- #
    @property
    def input_size(self) -> int:
        """供 UAVTrajectoryGRU 实例化时读取的单步输入维度。"""
        return self._augmented.shape[1]
