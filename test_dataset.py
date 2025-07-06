from dataset import CharDataset
from torch.utils.data import DataLoader

dataset = CharDataset("resources/poems.txt", seq_len=64)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

print("总字符数：", len(dataset.text))
print("独立字符数（vocab size）：", dataset.vocab_size)
print("前10个字符：", dataset.text[:10])

# 取一个 batch 看看
for x, y in dataloader:
    print("输入 shape:", x.shape)  # [B, seq_len]
    print("输出 shape:", y.shape)
    break
