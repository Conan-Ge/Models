import torch
import torch.nn as nn
import numpy as np
# 关键：导入 config.py 中的超参数（解决 DROPOUT 未定义问题）
from config import D_MODEL, MAX_LEN, DEVICE, DROPOUT


# 1. 位置编码（正弦位置编码，解决Transformer无位置信息问题）
class PositionalEncoding(nn.Module):
    def __init__(self, dropout=DROPOUT):  # 现在 DROPOUT 已导入，不会报错
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 生成位置编码矩阵：(MAX_LEN, D_MODEL)
        pe = torch.zeros(MAX_LEN, D_MODEL, device=DEVICE)
        position = torch.arange(0, MAX_LEN, dtype=torch.float, device=DEVICE).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, D_MODEL, 2).float() * (-np.log(10000.0) / D_MODEL))

        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数维度用正弦
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数维度用余弦
        pe = pe.unsqueeze(0)  # (1, MAX_LEN, D_MODEL)，适配batch维度
        self.register_buffer("pe", pe)  # 不参与梯度更新的缓冲区

    def forward(self, x):
        # x: (batch_size, seq_len, D_MODEL)
        x = x + self.pe[:, :x.size(1), :]  # 只取序列长度对应的位置编码
        return self.dropout(x)


# 2. 掩码函数（用于Decoder的掩码自注意力）
def create_mask(src, tgt=None):
    """
    src: (batch_size, src_seq_len)
    tgt: (batch_size, tgt_seq_len)（仅Encoder-Decoder时需要）
    返回：src_mask, tgt_mask
    """
    # 源序列掩码（屏蔽padding token，假设padding_idx=0）
    src_mask = (src != 0).unsqueeze(1).unsqueeze(2)  # (batch_size, 1, 1, src_seq_len)

    if tgt is None:
        return src_mask, None

    # 目标序列掩码：1. 屏蔽padding；2. 屏蔽未来token（下三角矩阵）
    tgt_pad_mask = (tgt != 0).unsqueeze(1).unsqueeze(3)  # (batch_size, 1, tgt_seq_len, 1)
    tgt_seq_len = tgt.size(1)
    tgt_subseq_mask = torch.tril(
        torch.ones(tgt_seq_len, tgt_seq_len, device=DEVICE)).bool()  # (tgt_seq_len, tgt_seq_len)
    tgt_mask = tgt_pad_mask & tgt_subseq_mask  # (batch_size, 1, tgt_seq_len, tgt_seq_len)

    return src_mask, tgt_mask


# 3. 评估指标（困惑度Perplexity，语言模型常用）
def calculate_perplexity(loss):
    return torch.exp(loss)