import torch
import pandas as pd  # 导入 pandas（需安装：pip install pandas pyarrow）
from torchtext.data.utils import get_tokenizer
from torch.utils.data import DataLoader, TensorDataset
from config import BATCH_SIZE, MAX_LEN, VOCAB_SIZE, DEVICE
import os

# -------------------------- 1. 读取 Parquet 格式数据集 --------------------------
DATASET_DIR = "../dataset"
# 读取训练/验证/测试集（如果没有 train/test.parquet，只保留 valid 即可）
train_path = os.path.join(DATASET_DIR, "train.parquet")
valid_path = os.path.join(DATASET_DIR, "valid.parquet")
test_path = os.path.join(DATASET_DIR, "test.parquet")

# 读取 Parquet 文件（pyarrow 是解析引擎，必须安装）
def read_parquet(file_path):
    df = pd.read_parquet(file_path, engine="pyarrow")
    # 假设数据集有 "text" 列存储文本，过滤空值
    texts = df["text"].dropna().str.strip().tolist()
    return [text for text in texts if text]  # 过滤空字符串

# 加载数据（如果没有 train/test，可注释对应行，只用 valid 测试）
train_iter = read_parquet(train_path) if os.path.exists(train_path) else []
val_iter = read_parquet(valid_path)
test_iter = read_parquet(test_path) if os.path.exists(test_path) else []

print(f"Parquet 数据集加载成功！")
print(f"  - 训练集文本数：{len(train_iter)}")
print(f"  - 验证集文本数：{len(val_iter)}")
print(f"  - 测试集文本数：{len(test_iter)}")

# -------------------------- 2. 后续逻辑完全不变（词汇表、张量转换、DataLoader）--------------------------
# （以下代码和之前的纯本地版本完全一致，直接复制粘贴即可）
tokenizer = get_tokenizer("basic_english")

# 统计词频
word_freq = {}
for text in train_iter or val_iter:  # 若没有训练集，用验证集构建词汇表
    tokens = tokenizer(text)
    for token in tokens:
        word_freq[token] = word_freq.get(token, 0) + 1

# 构建词汇表
sorted_words = sorted(word_freq.keys(), key=lambda x: word_freq[x], reverse=True)
selected_words = sorted_words[:VOCAB_SIZE - 2]
vocab_list = ["<pad>", "<unk>"] + selected_words
vocab = {word: idx for idx, word in enumerate(vocab_list)}

def token_to_idx(token):
    return vocab.get(token, vocab["<unk>"])

# 文本转张量
def text_to_tensor(text):
    tokens = tokenizer(text.strip())
    if len(tokens) > MAX_LEN:
        tokens = tokens[:MAX_LEN]
    else:
        tokens = tokens + ["<pad>"] * (MAX_LEN - len(tokens))
    indices = [token_to_idx(token) for token in tokens]
    return torch.tensor(indices, dtype=torch.long, device=DEVICE)

# 构建数据集
def build_dataset(data_iter):
    inputs, labels = [], []
    for text in data_iter:
        tensor = text_to_tensor(text)
        if len(tensor) >= 2:
            inputs.append(tensor[:-1])
            labels.append(tensor[1:])
    return TensorDataset(torch.stack(inputs), torch.stack(labels)) if inputs else None

# 生成数据集（处理无训练集/测试集的情况）
train_dataset = build_dataset(train_iter) if train_iter else None
val_dataset = build_dataset(val_iter)
test_dataset = build_dataset(test_iter) if test_iter else None

# 构建 DataLoader
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False) if train_dataset else None
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False) if test_dataset else None

# 打印统计
print(f"\n预处理完成！")
if train_loader:
    print(f"  - 训练集：{len(train_dataset)} 样本 | {len(train_loader)} 批次")
print(f"  - 验证集：{len(val_dataset)} 样本 | {len(val_loader)} 批次")
if test_loader:
    print(f"  - 测试集：{len(test_dataset)} 样本 | {len(test_loader)} 批次")
print(f"  - 词汇表大小：{len(vocab)}")