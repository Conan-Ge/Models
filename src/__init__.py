"""
Transformer 项目包初始化文件
作用：
1. 标识 src 为 Python 包，支持跨模块导入
2. 集中导入核心类、函数、配置，简化外部调用
3. 全局初始化（创建结果文件夹、固定随机种子、设置设备）
4. 规范包导出接口
"""

# -------------------------- 第一步：全局初始化（确保环境就绪）--------------------------
import os
import torch
import numpy as np

# 1. 创建结果文件夹（避免训练时因文件夹不存在报错）
os.makedirs("../results", exist_ok=True)  # 项目根目录的 results 文件夹
os.makedirs("../dataset", exist_ok=True)  # 数据集文件夹（若未手动创建）

# 2. 固定随机种子（全局统一，确保可复现性）
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)  # 多GPU场景
torch.backends.cudnn.deterministic = True  # 禁用非确定性算法
torch.backends.cudnn.benchmark = False     # 禁用自动优化（牺牲速度换可复现）

# 3. 全局设备配置（统一设备，避免模块间设备不一致）
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[初始化] 项目运行设备：{DEVICE}")
if torch.cuda.is_available():
    print(f"[初始化] GPU型号：{torch.cuda.get_device_name(0)}")
    print(f"[初始化] GPU显存：{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

# -------------------------- 第二步：导入核心模块（简化外部调用）--------------------------
# 1. 配置参数（从 config.py 导入所有超参数）
from .config import (
    MAX_LEN, BATCH_SIZE, EPOCHS,
    D_MODEL, N_HEADS, D_K, D_V, D_FF, N_LAYERS, DROPOUT,
    VOCAB_SIZE, LEARNING_RATE, WEIGHT_DECAY, CLIP_NORM,
    DATASET
)

# 2. 模型核心模块（从 model.py 导入关键类）
from .model import (
    ScaledDotProductAttention,
    MultiHeadAttention,
    PositionWiseFeedForward,
    EncoderLayer,
    DecoderLayer,
    Transformer
)

# 3. 工具函数（从 utils.py 导入常用工具）
from .utils import (
    PositionalEncoding,
    create_mask,
    calculate_perplexity
)

# 4. 数据加载（从 data.py 导入数据集和词汇表）
from .data import (
    vocab,
    train_loader,
    val_loader,
    test_loader
)

# 5. 训练逻辑（从 train.py 导入训练/验证函数）
from .train import (
    train_epoch,
    val_epoch
)

# -------------------------- 第三步：规范包导出接口（__all__）--------------------------
# 定义 "from src import *" 时会导出的内容，避免导出内部无关变量
__all__ = [
    # 全局配置
    "SEED", "DEVICE", "DATASET",
    # 超参数
    "MAX_LEN", "BATCH_SIZE", "EPOCHS",
    "D_MODEL", "N_HEADS", "D_K", "D_V", "D_FF", "N_LAYERS", "DROPOUT",
    "VOCAB_SIZE", "LEARNING_RATE", "WEIGHT_DECAY", "CLIP_NORM",
    # 模型类
    "ScaledDotProductAttention", "MultiHeadAttention",
    "PositionWiseFeedForward", "EncoderLayer", "DecoderLayer", "Transformer",
    # 工具函数
    "PositionalEncoding", "create_mask", "calculate_perplexity",
    # 数据相关
    "vocab", "train_loader", "val_loader", "test_loader",
    # 训练函数
    "train_epoch", "val_epoch"
]

# -------------------------- 第四步：初始化验证（可选，打印日志）--------------------------
print(f"[初始化] Transformer 包加载完成！")
print(f"[初始化] 当前数据集：{DATASET}")
print(f"[初始化] 模型配置：D_MODEL={D_MODEL}, N_HEADS={N_HEADS}, N_LAYERS={N_LAYERS}")
print(f"[初始化] 训练配置：EPOCHS={EPOCHS}, BATCH_SIZE={BATCH_SIZE}, LR={LEARNING_RATE}")