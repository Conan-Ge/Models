import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os
from model import Transformer
from data import train_loader, val_loader, test_loader
from utils import calculate_perplexity
from config import EPOCHS, LEARNING_RATE, WEIGHT_DECAY, CLIP_NORM, DEVICE, SEED

# 固定随机种子（可复现性）
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

# 初始化模型、损失函数、优化器
model = Transformer(has_decoder=False).to(DEVICE)  # 仅Encoder；若需Decoder，设为True
criterion = nn.CrossEntropyLoss(ignore_index=0)  # 忽略padding token（index=0）
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

# 记录训练曲线
train_losses = []
val_losses = []
train_perplexities = []
val_perplexities = []


# 训练函数
def train_epoch():
    model.train()
    total_loss = 0.0
    for batch in train_loader:
        src, tgt = batch  # src: (batch_size, seq_len-1), tgt: (batch_size, seq_len-1)
        optimizer.zero_grad()

        logits = model(src)  # (batch_size, seq_len-1, VOCAB_SIZE)
        # CrossEntropyLoss要求输入形状：(batch_size*seq_len, VOCAB_SIZE)，标签：(batch_size*seq_len)
        loss = criterion(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1))

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)  # 梯度裁剪（稳定训练）
        optimizer.step()

        total_loss += loss.item() * src.size(0)  # 按batch大小加权

    avg_loss = total_loss / len(train_loader.dataset)
    avg_perplexity = calculate_perplexity(torch.tensor(avg_loss))
    return avg_loss, avg_perplexity.item()


# 验证函数
def val_epoch():
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in val_loader:
            src, tgt = batch
            logits = model(src)
            loss = criterion(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1))
            total_loss += loss.item() * src.size(0)

    avg_loss = total_loss / len(val_loader.dataset)
    avg_perplexity = calculate_perplexity(torch.tensor(avg_loss))
    return avg_loss, avg_perplexity.item()


# 主训练循环
print(f"Training on {DEVICE}...")
for epoch in range(EPOCHS):
    train_loss, train_perp = train_epoch()
    val_loss, val_perp = val_epoch()

    # 记录结果
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_perplexities.append(train_perp)
    val_perplexities.append(val_perp)

    # 打印日志
    print(
        f"Epoch {epoch + 1:2d} | Train Loss: {train_loss:.4f} | Train Perp: {train_perp:.2f} | Val Loss: {val_loss:.4f} | Val Perp: {val_perp:.2f}")

# 测试集评估
test_loss, test_perp = val_epoch()  # 复用val_epoch逻辑（仅前向传播）
print(f"\nTest Loss: {test_loss:.4f} | Test Perplexity: {test_perp:.2f}")

# 保存模型
os.makedirs("results", exist_ok=True)
torch.save(model.state_dict(), "results/transformer_encoder.pth")

# 绘制训练曲线（损失+困惑度）
plt.figure(figsize=(12, 4))

# 损失曲线
plt.subplot(1, 2, 1)
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("Training & Validation Loss")

# 困惑度曲线
plt.subplot(1, 2, 2)
plt.plot(train_perplexities, label="Train Perplexity")
plt.plot(val_perplexities, label="Val Perplexity")
plt.xlabel("Epoch")
plt.ylabel("Perplexity")
plt.legend()
plt.title("Training & Validation Perplexity")

plt.tight_layout()
plt.savefig("results/training_curves.png")
plt.show()

# 保存评估结果到表格
with open("results/evaluation_results.txt", "w") as f:
    f.write(f"Test Loss: {test_loss:.4f}\n")
    f.write(f"Test Perplexity: {test_perp:.2f}\n")
    f.write(f"\nEpoch-wise Results:\n")
    f.write(f"Epoch | Train Loss | Train Perp | Val Loss | Val Perp\n")
    for i in range(EPOCHS):
        f.write(
            f"{i + 1:5d} | {train_losses[i]:.4f} | {train_perplexities[i]:.2f} | {val_losses[i]:.4f} | {val_perplexities[i]:.2f}\n")