import os
import pickle

import matplotlib.pyplot as plt
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader

from dataset import CharDataset
from model import CharLSTM

# 参数设置
seq_len = 64
batch_size = 128
num_epochs = 15
learning_rate = 0.001
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
origin_version = "85"
checkpoint_path = "checkpoints/charlstm_epoch" + origin_version + ".pt"  # 用于继续训练的模型路径

# 加载数据
dataset = CharDataset("resources/poems.txt", seq_len=seq_len)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 保存词表
os.makedirs("checkpoints", exist_ok=True)
with open("checkpoints/char_vocab.pkl", "wb") as f:
    pickle.dump({
        "char2idx": dataset.char2idx,
        "idx2char": dataset.idx2char
    }, f)
print("词表已保存到 checkpoints/char_vocab.pkl")

# 初始化模型
model = CharLSTM(vocab_size=dataset.vocab_size).to(device)
criterion = CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=learning_rate)

new_epoch = 0
# 如有之前训练的模型可加载
if os.path.exists(checkpoint_path):
    print(f"加载已有模型参数：{checkpoint_path}")
    new_epoch = int(origin_version)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))

# 用于绘图
loss_list = []
ppl_list = []

# 训练过程
for epoch in range(new_epoch, num_epochs + new_epoch):
    model.train()
    total_loss = 0
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        output, _ = model(x)
        loss = criterion(output.view(-1, dataset.vocab_size), y.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    ppl = torch.exp(torch.tensor(avg_loss))  # Perplexity
    loss_list.append(avg_loss)
    ppl_list.append(ppl.item())

    print(f"Epoch {epoch+1}/{num_epochs + new_epoch}, Loss: {avg_loss:.4f}, PPL: {ppl:.2f}")

    # 保存模型
    if (epoch+1) % 5 == 0:
        torch.save(model.state_dict(), f"checkpoints/charlstm_epoch{epoch+1}.pt")

print("训练完成！")

# 绘制 PPL 和 Loss 曲线
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(loss_list, label="Loss")
plt.title("Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(ppl_list, label="Perplexity", color="orange")
plt.title("PPL Curve")
plt.xlabel("Epoch")
plt.ylabel("Perplexity")
plt.legend()

plt.tight_layout()
plt.savefig("resources/training_curves" + origin_version + ".png")  # 保存图片
plt.show()
