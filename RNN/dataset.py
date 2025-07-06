import torch
from torch.utils.data import Dataset

class CharDataset(Dataset):
    def __init__(self, text_path, seq_len=64):
        with open(text_path, 'r', encoding='utf-8') as f:
            text = f.read()

        self.seq_len = seq_len
        self.text = text
        self.chars = sorted(list(set(text)))
        self.vocab_size = len(self.chars)

        # 字符与索引映射
        self.char2idx = {ch: idx for idx, ch in enumerate(self.chars)}
        self.idx2char = {idx: ch for idx, ch in enumerate(self.chars)}

        # 编码整篇文本为索引列表
        self.data = [self.char2idx[ch] for ch in text if ch in self.char2idx]

    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.seq_len]
        y = self.data[idx + 1 : idx + self.seq_len + 1]
        return torch.tensor(x), torch.tensor(y)
