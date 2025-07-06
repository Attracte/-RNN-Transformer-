import os
import pickle

import torch
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import Dataset
from model import CharLSTM

class CharDataset(Dataset):
    def __init__(self, filepath, seq_len=64):
        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read().replace('\\n', '\n')
        # 保留换行符作为字符
        self.chars = sorted(list(set(text)))
        self.char2idx = {ch: i for i, ch in enumerate(self.chars)}
        self.idx2char = {i: ch for i, ch in enumerate(self.chars)}

        self.vocab_size = len(self.chars)
        self.seq_len = seq_len
        self.data = [self.char2idx[ch] for ch in text]

    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx: idx + self.seq_len], dtype=torch.long)
        y = torch.tensor(self.data[idx + 1: idx + 1 + self.seq_len], dtype=torch.long)
        return x, y


# 参数设置
seq_len = 64
batch_size = 128
num_epochs = 50
learning_rate = 0.001
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载数据集
dataset = CharDataset("resources/poems.txt", seq_len=seq_len)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 初始化模型VB3
model = CharLSTM(vocab_size=dataset.vocab_size).to(device)
criterion = CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=learning_rate)

# 检查是否存在已保存的模型
version = "50"
checkpoint_path = "checkpoints/charlstm_epoch"+version+".pt"
resume_epoch = 0


if os.path.exists(checkpoint_path):
    print(f"加载已有模型参数：{checkpoint_path}")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    # 如果你保存了优化器状态，也可以恢复（可选）
    # optimizer.load_state_dict(torch.load("checkpoints/optimizer_epoch20.pt"))
    resume_epoch = int(version)
    # 根据文件名或你记录的轮次

for epoch in range(resume_epoch, resume_epoch + num_epochs):
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
    print(f"Epoch {epoch + 1}/{num_epochs + resume_epoch} - Loss: {avg_loss:.4f}")

    # 保存模型
    if (epoch + 1) % 10 == 0:
        torch.save(model.state_dict(), f"checkpoints/charlstm_epoch{epoch+1}.pt")

print("训练完成！")

vocab_path = "checkpoints/char_vocab.pkl"
with open(vocab_path, "wb") as f:
    pickle.dump({
        "char2idx": dataset.char2idx,
        "idx2char": dataset.idx2char
    }, f)

print(f"✅ 词表已保存到 {vocab_path}，vocab_size = {dataset.vocab_size}")