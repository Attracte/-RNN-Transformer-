import re

import torch
from torch.utils.data import Dataset

PAD_TOKEN = '<PAD>'
UNK_TOKEN = '<UNK>'

class CharDataset(Dataset):
    def __init__(self, file_path, seq_len=64, char2idx=None):
        self.seq_len = seq_len
        self.poems = self.load_and_process(file_path)

        self.PAD_TOKEN = '<PAD>'

        if char2idx is None:
            # 原始自动构造词表（没过滤）
            self.all_text = ''.join(self.poems)
            chars = sorted(list(set(self.all_text)))
            chars.append(self.PAD_TOKEN)
            self.chars = chars
            self.char2idx = {ch: i for i, ch in enumerate(self.chars)}
            self.idx2char = {i: ch for ch, i in self.char2idx.items()}
        else:
            # 使用传入的过滤词表
            self.char2idx = char2idx
            self.idx2char = {i: ch for ch, i in char2idx.items()}
            self.chars = list(char2idx.keys())

        self.vocab_size = len(self.chars)

        self.data = []
        pad_id = self.char2idx[self.PAD_TOKEN]
        for poem in self.poems:
            encoded = []
            for ch in poem:
                if ch in self.char2idx:
                    encoded.append(self.char2idx[ch])
                else:
                    encoded.append(self.char2idx.get('<UNK>', 1))  # <UNK>索引
            if len(encoded) < self.seq_len + 1:
                encoded += [pad_id] * (self.seq_len + 1 - len(encoded))
            for i in range(len(encoded) - self.seq_len):
                x = encoded[i:i + self.seq_len]
                y = encoded[i + 1:i + self.seq_len + 1]
                self.data.append((x, y))

    def load_and_process(self, file_path):
        poems = []
        with open(file_path, "r", encoding="utf-8") as f:
            current_poem = []
            for line in f:
                line = line.strip()
                if not line:
                    if current_poem:
                        poems.append(''.join(current_poem))
                        current_poem = []
                    continue
                if re.match(r'^\[.*?\]$', line):
                    continue
                current_poem.append(line)
            if current_poem:
                poems.append(''.join(current_poem))
        return poems

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x, y = self.data[idx]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)
