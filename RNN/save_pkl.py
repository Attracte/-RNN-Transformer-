# 保存词表
import pickle

from dataset import CharDataset

dataset = CharDataset("resources/poems.txt", seq_len=64)

with open("checkpoints/char_vocab.pkl", "wb") as f:
    pickle.dump({
        "char2idx": dataset.char2idx,
        "idx2char": dataset.idx2char
    }, f)

print("词表重新保存，vocab_size =", len(dataset.char2idx))
