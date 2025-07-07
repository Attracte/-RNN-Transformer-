# 测试数据集的读取是否正确

from torch.utils.data import DataLoader

from dataset import CharDataset

dataset = CharDataset("resources/poems.txt", seq_len=64)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

print("总字符数：", len(dataset.all_text))
print("独立字符数（vocab size）：", dataset.vocab_size)
print("前10个字符：", dataset.all_text[:10])
print("诗歌数量：", dataset.poems.__len__())

# 取一个 batch 看看
for x, y in dataloader:
    print("输入 shape:", x.shape)  # [B, seq_len]
    print("输出 shape:", y.shape)
    break
