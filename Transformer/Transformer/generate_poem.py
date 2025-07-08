import os
import re
import torch
import argparse
from torch import nn


# Transformer 模型定义
class CharLevelTransformer(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_heads, num_layers, ff_dim, seq_length, dropout=0.1):
        super(CharLevelTransformer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.positional_encoding = torch.zeros(1, seq_length, embedding_dim)  # 改为普通 Tensor
        self.transformer = nn.Transformer(
            d_model=embedding_dim,
            nhead=num_heads,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True
        )
        self.fc = nn.Linear(embedding_dim, vocab_size)

    def forward(self, x, y):
        if x.size(1) > self.positional_encoding.size(1):
            extra_positional_encoding = torch.zeros(
                1, x.size(1) - self.positional_encoding.size(1), self.positional_encoding.size(2),
                device=x.device  # 确保扩展部分与输入张量在同一设备
            )
            self.positional_encoding = torch.cat([self.positional_encoding.to(x.device), extra_positional_encoding],
                                                 dim=1)

        x = self.embedding(x) + self.positional_encoding[:, :x.size(1), :].to(x.device)
        y = self.embedding(y) + self.positional_encoding[:, :y.size(1), :].to(x.device)

        tgt_mask = self.generate_square_subsequent_mask(y.size(1)).to(x.device)
        out = self.transformer(x, y, tgt_mask=tgt_mask)
        out = self.fc(out)
        return out

    def generate_square_subsequent_mask(self, size):
        mask = torch.triu(torch.ones(size, size), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask


# 生成文本函数
def generate_text(model, start_text, char2idx, idx2char, seq_length, length, poem_type, top_k=0, top_p=1.0,
                  temperature=0.8, device='cuda'):
    """
    使用训练好的模型生成文本，确保标点符号只出现在行尾。
    """
    model.eval()
    unk_idx = char2idx.get('<UNK>', 0)

    # 初始化输入序列
    input_seq = [char2idx.get(char, unk_idx) for char in start_text]
    input_seq = torch.tensor(input_seq, dtype=torch.long).unsqueeze(0).to(device)

    # 标题
    generated_text = start_text + "\n\n"  # 标题和正文之间添加换行

    # 控制行数和每行字数
    num_lines = 4 if "绝句" in poem_type else 8  # 绝句 4 行，律诗 8 行
    line_length = 5 if "五言" in poem_type else 7  # 五言 5 字，七言 7 字
    punctuation = ['，', '。']  # 可用的标点符号

    with torch.no_grad():
        for line_idx in range(1, num_lines + 1):  # 从 1 开始计数以便判断单双数行
            line = ""
            for char_idx in range(line_length):  # 控制每行的字数
                tgt_input = input_seq[:, -seq_length:]  # 限制 tgt_input 的长度不超过 seq_length
                output = model(input_seq, tgt_input)

                # 只取最后一个时间步的输出
                output = output[:, -1, :] / temperature
                probs = torch.softmax(output, dim=-1)

                # 禁止标点符号出现在诗句中间
                for p in punctuation:
                    if p in char2idx:
                        probs[0, char2idx[p]] = 0

                # Top-K 采样逻辑
                if top_k > 0:
                    top_k_probs, top_k_indices = torch.topk(probs, top_k)
                    probs = torch.zeros_like(probs).scatter_(1, top_k_indices, top_k_probs)

                # Top-P 采样逻辑
                if top_p < 1.0:
                    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_probs[sorted_indices_to_remove] = 0
                    probs = torch.zeros_like(probs).scatter_(1, sorted_indices, sorted_probs)

                # 重新归一化概率
                probs = probs / probs.sum()

                # 检查非法值
                if torch.isnan(probs).any() or torch.isinf(probs).any() or (probs < 0).any():
                    print("发现非法概率值，使用均匀分布作为默认值")
                    probs = torch.ones_like(probs) / probs.size(-1)

                # 如果概率总和为 0，使用均匀分布作为默认值
                if probs.sum() == 0:
                    print("概率总和为 0，使用均匀分布作为默认值")
                    probs = torch.ones_like(probs) / probs.size(-1)

                # 打印概率张量的统计信息（用于调试）
                print(f"概率张量统计信息：max={probs.max().item()}, min={probs.min().item()}, sum={probs.sum().item()}")

                # 从概率分布中采样
                next_char_idx = torch.multinomial(probs, num_samples=1).item()
                next_char = idx2char[next_char_idx]

                # 确保生成的字符是有效的汉字
                if not re.match(r'[\u4e00-\u9fa5]', next_char):
                    continue

                line += next_char
                input_seq = torch.cat([input_seq, torch.tensor([[next_char_idx]], dtype=torch.long).to(device)], dim=1)

            # 在行尾手动添加标点符号（不在模型生成范围内）
            if line_idx % 2 == 1:  # 单数行（1、3、5、7）
                line += "，"
            else:  # 双数行（2、4、6、8）
                line += "。"

            generated_text += line + "\n"

    return generated_text


# 主程序入口
if __name__ == "__main__":
    # 命令行参数解析
    parser = argparse.ArgumentParser(description="生成诗歌")
    parser.add_argument('--model_path', type=str, required=True, help="保存的模型路径")
    parser.add_argument('--start_text', type=str, required=True, help="生成诗歌的起始文本")
    parser.add_argument('--poem_type', type=str, choices=["五言绝句", "五言律诗", "七言绝句", "七言律诗"],
                        required=True, help="诗歌类型")
    parser.add_argument('--seq_length', type=int, default=20, help="训练时的序列长度")
    parser.add_argument('--length', type=int, default=50, help="生成的诗歌总长度")
    parser.add_argument('--top_k', type=int, default=0, help="Top-K 采样的值")
    parser.add_argument('--top_p', type=float, default=1.0, help="Top-P 采样的值")
    parser.add_argument('--temperature', type=float, default=1.0, help="温度参数")
    args = parser.parse_args()

    # 加载模型和词表
    vocab_path = args.model_path.replace('.pth', '_vocab.pth')
    vocab = torch.load(vocab_path)
    char2idx, idx2char = vocab['char2idx'], vocab['idx2char']

    seq_length = args.seq_length
    embedding_dim = 128
    num_heads = 4
    num_layers = 2
    ff_dim = 512
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = CharLevelTransformer(len(char2idx), embedding_dim, num_heads, num_layers, ff_dim, seq_length).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    # 生成诗歌
    generated_poem = generate_text(
        model=model,
        start_text=args.start_text,
        char2idx=char2idx,
        idx2char=idx2char,
        seq_length=args.seq_length,
        poem_type=args.poem_type,
        length=args.length,
        top_k=args.top_k,
        top_p=args.top_p,
        temperature=args.temperature,
        device=device
    )

    # 打印生成的诗歌
    print(f"\n生成的诗歌 ({args.poem_type}):")
    print(generated_poem)
