import torch
import torch.nn as nn
import torch.nn.functional as F
from config import D_MODEL, N_HEADS, D_K, D_V, D_FF, N_LAYERS, DROPOUT, VOCAB_SIZE
from utils import PositionalEncoding, create_mask  # 新增 create_mask 导入


# 1. 缩放点积注意力
class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k, v, mask=None):
        """
        q: (batch_size, N_HEADS, seq_len_q, D_K)
        k: (batch_size, N_HEADS, seq_len_k, D_K)
        v: (batch_size, N_HEADS, seq_len_v, D_V)
        mask: (batch_size, 1, seq_len_q, seq_len_k)
        返回：output, attn_weights
        """
        # 计算注意力分数：(batch_size, N_HEADS, seq_len_q, seq_len_k)
        scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(D_K, dtype=torch.float32))

        # 应用掩码（屏蔽无效位置，设为-1e9）
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # 注意力权重（softmax归一化）
        attn_weights = F.softmax(scores, dim=-1)  # (batch_size, N_HEADS, seq_len_q, seq_len_k)

        # 输出：注意力加权求和
        output = torch.matmul(attn_weights, v)  # (batch_size, N_HEADS, seq_len_q, D_V)
        return output, attn_weights


# 2. 多头注意力
class MultiHeadAttention(nn.Module):
    def __init__(self):
        super().__init__()
        # 线性投影层（q, k, v分别投影到D_MODEL维度）
        self.w_q = nn.Linear(D_MODEL, D_MODEL)
        self.w_k = nn.Linear(D_MODEL, D_MODEL)
        self.w_v = nn.Linear(D_MODEL, D_MODEL)
        # 输出投影层
        self.w_o = nn.Linear(D_MODEL, D_MODEL)
        self.dropout = nn.Dropout(DROPOUT)
        self.attn = ScaledDotProductAttention()

    def forward(self, q, k, v, mask=None):
        """
        q/k/v: (batch_size, seq_len, D_MODEL)
        返回：output (batch_size, seq_len_q, D_MODEL), attn_weights
        """
        batch_size = q.size(0)

        # 线性投影 + 拆分多头：(batch_size, seq_len, D_MODEL) → (batch_size, N_HEADS, seq_len, D_K)
        q = self.w_q(q).view(batch_size, -1, N_HEADS, D_K).transpose(1, 2)
        k = self.w_k(k).view(batch_size, -1, N_HEADS, D_K).transpose(1, 2)
        v = self.w_v(v).view(batch_size, -1, N_HEADS, D_V).transpose(1, 2)

        # 缩放点积注意力
        attn_output, attn_weights = self.attn(q, k, v, mask)  # (batch_size, N_HEADS, seq_len_q, D_K)

        # 多头拼接：(batch_size, N_HEADS, seq_len_q, D_K) → (batch_size, seq_len_q, D_MODEL)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, D_MODEL)

        # 输出投影 + dropout
        output = self.dropout(self.w_o(attn_output))
        return output, attn_weights


# 3. 位置-wise前馈网络
class PositionWiseFeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(D_MODEL, D_FF)
        self.fc2 = nn.Linear(D_FF, D_MODEL)
        self.dropout = nn.Dropout(DROPOUT)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x: (batch_size, seq_len, D_MODEL) → (batch_size, seq_len, D_MODEL)
        return self.fc2(self.dropout(self.relu(self.fc1(x))))


# 4. Encoder层（单层）
class EncoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = MultiHeadAttention()
        self.ffn = PositionWiseFeedForward()
        # 层归一化（Pre-LN结构，更稳定）
        self.norm1 = nn.LayerNorm(D_MODEL)
        self.norm2 = nn.LayerNorm(D_MODEL)
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x, src_mask):
        """
        x: (batch_size, src_seq_len, D_MODEL)
        src_mask: (batch_size, 1, 1, src_seq_len)
        返回：x (batch_size, src_seq_len, D_MODEL)
        """
        # 自注意力子层（残差连接+层归一化）
        attn_output, _ = self.self_attn(x, x, x, src_mask)
        x = self.norm1(x + self.dropout(attn_output))

        # 前馈子层（残差连接+层归一化）
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_output))

        return x


# 5. Decoder层（单层，可选实现）
class DecoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = MultiHeadAttention()  # 掩码自注意力
        self.cross_attn = MultiHeadAttention()  # 编码器-解码器交叉注意力
        self.ffn = PositionWiseFeedForward()
        self.norm1 = nn.LayerNorm(D_MODEL)
        self.norm2 = nn.LayerNorm(D_MODEL)
        self.norm3 = nn.LayerNorm(D_MODEL)
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x, enc_output, tgt_mask, src_mask):
        """
        x: (batch_size, tgt_seq_len, D_MODEL)
        enc_output: (batch_size, src_seq_len, D_MODEL)
        tgt_mask: (batch_size, 1, tgt_seq_len, tgt_seq_len)
        src_mask: (batch_size, 1, 1, src_seq_len)
        返回：x (batch_size, tgt_seq_len, D_MODEL)
        """
        # 掩码自注意力
        attn1_output, _ = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn1_output))

        # 交叉注意力（q来自Decoder，k/v来自Encoder）
        attn2_output, _ = self.cross_attn(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout(attn2_output))

        # 前馈子层
        ffn_output = self.ffn(x)
        x = self.norm3(x + self.dropout(ffn_output))

        return x


# 6. 完整Transformer模型
class Transformer(nn.Module):
    def __init__(self, has_decoder=False):
        super().__init__()
        self.has_decoder = has_decoder  # 是否包含Decoder（False=仅Encoder，True=Encoder-Decoder）

        # 嵌入层（词嵌入+位置编码）
        self.src_embedding = nn.Embedding(VOCAB_SIZE, D_MODEL)
        self.pos_encoding = PositionalEncoding()

        # Encoder堆叠
        self.encoder_layers = nn.ModuleList([EncoderLayer() for _ in range(N_LAYERS)])

        # Decoder堆叠（可选）
        if self.has_decoder:
            self.tgt_embedding = nn.Embedding(VOCAB_SIZE, D_MODEL)
            self.decoder_layers = nn.ModuleList([DecoderLayer() for _ in range(N_LAYERS)])

        # 输出层（语言模型/翻译任务）
        self.fc_out = nn.Linear(D_MODEL, VOCAB_SIZE)
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, src, tgt=None):
        """
        src: (batch_size, src_seq_len)
        tgt: (batch_size, tgt_seq_len)（仅Encoder-Decoder时传入）
        返回：logits (batch_size, seq_len, VOCAB_SIZE)
        """
        batch_size = src.size(0)
        src_seq_len = src.size(1)

        # 1. Encoder前向传播
        src_emb = self.dropout(self.pos_encoding(self.src_embedding(src)))  # (batch_size, src_seq_len, D_MODEL)
        src_mask, tgt_mask = create_mask(src, tgt)

        enc_output = src_emb
        for enc_layer in self.encoder_layers:
            enc_output = enc_layer(enc_output, src_mask)  # (batch_size, src_seq_len, D_MODEL)

        # 仅Encoder模型（如语言模型/分类）：直接输出
        if not self.has_decoder:
            logits = self.fc_out(enc_output)  # (batch_size, src_seq_len, VOCAB_SIZE)
            return logits

        # 2. Decoder前向传播（如翻译/摘要）
        tgt_emb = self.dropout(self.pos_encoding(self.tgt_embedding(tgt)))  # (batch_size, tgt_seq_len, D_MODEL)

        dec_output = tgt_emb
        for dec_layer in self.decoder_layers:
            dec_output = dec_layer(dec_output, enc_output, tgt_mask, src_mask)  # (batch_size, tgt_seq_len, D_MODEL)

        logits = self.fc_out(dec_output)  # (batch_size, tgt_seq_len, VOCAB_SIZE)
        return logits