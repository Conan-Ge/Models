import torch
from torchtext.data.utils import get_tokenizer
from torch.utils.data import DataLoader, TensorDataset
from config import BATCH_SIZE, MAX_LEN, VOCAB_SIZE, DEVICE
import os

# -------------------------- 1. 本地数据集路径（必须和你的文件位置一致！）--------------------------
DATASET_DIR = "../dataset"
TRAIN_PATH = os.path.join(DATASET_DIR, "wiki.train.tokens")
VAL_PATH = os.path.join(DATASET_DIR, "wiki.valid.tokens")
TEST_PATH = os.path.join(DATASET_DIR, "wiki.test.tokens")

# 验证文件是否存在
assert os.path.exists(TRAIN_PATH), f"❌ 找不到训练集文件！路径：{TRAIN_PATH}"
assert os.path.exists(VAL_PATH), f"❌ 找不到验证集文件！路径：{VAL_PATH}"
assert os.path.exists(TEST_PATH), f"❌ 找不到测试集文件！路径：{TEST_PATH}"

# -------------------------- 2. 读取本地文本文件（不联网）--------------------------
def read_text_file(file_path):
    texts = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:  # 过滤空行
                texts.append(line)
    return texts

# 加载本地文件到文本列表
train_iter = read_text_file(TRAIN_PATH)
val_iter = read_text_file(VAL_PATH)
test_iter = read_text_file(TEST_PATH)

# -------------------------- 3. 手动构建词汇表（适配所有torchtext版本）--------------------------
tokenizer = get_tokenizer("basic_english")  # 英文分词器

# 步骤1：统计训练集中所有词的频率（过滤低频词）
word_freq = {}
for text in train_iter:
    tokens = tokenizer(text)
    for token in tokens:
        if token in word_freq:
            word_freq[token] += 1
        else:
            word_freq[token] = 1

# 步骤2：按频率排序，取前 VOCAB_SIZE-2 个词（预留 <pad> 和 <unk>）
sorted_words = sorted(word_freq.keys(), key=lambda x: word_freq[x], reverse=True)
selected_words = sorted_words[:VOCAB_SIZE - 2]  # 减2是因为要加2个特殊标记

# 步骤3：构建词汇表（特殊标记在前，普通词在后）
vocab_list = ["<pad>", "<unk>"] + selected_words
vocab = {word: idx for idx, word in enumerate(vocab_list)}  # 字典格式：词→索引

# 步骤4：定义获取词索引的函数（替代 vocab() 方法）
def get_token_index(token):
    return vocab.get(token, vocab["<unk>"])  # 未知词返回 <unk> 的索引（1）

# -------------------------- 4. 文本转张量（固定长度）--------------------------
def text_to_tensor(text):
    tokens = tokenizer(text)
    # 截断过长序列
    if len(tokens) > MAX_LEN:
        tokens = tokens[:MAX_LEN]
    # 填充过短序列
    else:
        tokens = tokens + ["<pad>"] * (MAX_LEN - len(tokens))
    # 用手动定义的 get_token_index 转换索引
    token_indices = [get_token_index(token) for token in tokens]
    return torch.tensor(token_indices, dtype=torch.long, device=DEVICE)

# -------------------------- 5. 构建语言模型数据集 --------------------------
def build_lm_dataset(data_iter):
    inputs = []
    labels = []
    for text in data_iter:
        tensor = text_to_tensor(text)
        if len(tensor) < 2:
            continue
        inputs.append(tensor[:-1])  # 输入：前MAX_LEN-1个词
        labels.append(tensor[1:])   # 标签：后MAX_LEN-1个词
    return TensorDataset(torch.stack(inputs), torch.stack(labels))

# 生成数据集
train_dataset = build_lm_dataset(train_iter)
val_dataset = build_lm_dataset(val_iter)
test_dataset = build_lm_dataset(test_iter)

# -------------------------- 6. 构建DataLoader --------------------------
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=True
)
val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    drop_last=False
)
test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    drop_last=False
)

# -------------------------- 7. 打印加载成功信息 --------------------------
print(f" 本地数据集加载成功！")
print(f" 数据统计：")
print(f"  - 训练集：{len(train_dataset)} 样本 | {len(train_loader)} 批次")
print(f"  - 验证集：{len(val_dataset)} 样本 | {len(val_loader)} 批次")
print(f"  - 测试集：{len(test_dataset)} 样本 | {len(test_loader)} 批次")
print(f"  - 词汇表大小：{len(vocab)}")