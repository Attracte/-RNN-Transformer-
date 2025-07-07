import re

import torch
from torch.utils.data import Dataset

PAD_TOKEN = '<PAD>'


class CharDataset(Dataset):
    def __init__(self, file_path, seq_len=64):
        self.seq_len = seq_len
        self.poems = self.load_and_process(file_path)

        # 构建字符集，加入 PAD_TOKEN
        self.all_text = ''.join(self.poems)
        chars = sorted(list(set(self.all_text)))
        chars.append(PAD_TOKEN)  # 加入补齐符号
        self.chars = chars

        self.char2idx = {ch: i for i, ch in enumerate(self.chars)}
        self.idx2char = {i: ch for ch, i in self.char2idx.items()}
        self.vocab_size = len(self.chars)

        self.data = []
        for poem in self.poems:
            encoded = [self.char2idx[ch] for ch in poem if ch in self.char2idx]
            # 补齐用PAD_TOKEN的索引
            pad_id = self.char2idx[PAD_TOKEN]
            if len(encoded) < self.seq_len + 1:
                pad_len = self.seq_len + 1 - len(encoded)
                encoded += [pad_id] * pad_len

            for i in range(0, len(encoded) - self.seq_len):
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
