import torch

# 基础配置
SEED = 42  # 固定随机种子
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LEN = 512  # 序列最大长度
BATCH_SIZE = 4
EPOCHS = 20

# 模型配置
D_MODEL = 128  # 嵌入维度
N_HEADS = 4    # 多头注意力头数
D_K = D_MODEL // N_HEADS  # 每个头的维度
D_V = D_K
D_FF = 512     # 前馈网络隐藏层维度
N_LAYERS = 3   # Encoder/Decoder层数
DROPOUT = 0.1  #  dropout概率

# 数据集配置（可选：WikiText-2/AG News/IWSLT2017）
DATASET = "wikitext-2"  # 仅Encoder用；若用Encoder-Decoder，改为"iwslt2017"
VOCAB_SIZE = 30000  # 词汇表大小

# 训练配置
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
CLIP_NORM = 1.0  # 梯度裁剪阈值