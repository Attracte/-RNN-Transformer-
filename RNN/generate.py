import pickle
import re

import torch
import torch.nn.functional as F

from RNN.model import CharLSTM

model_version = "30"

def sample_from_logits(logits, temperature=1.0, top_k=None, top_p=None):
    logits = logits / temperature
    probs = F.softmax(logits, dim=-1)

    if top_k:
        top_k = max(top_k, 1)
        indices_to_remove = probs < torch.topk(probs, top_k)[0][..., -1, None]
        probs = probs.masked_fill(indices_to_remove, 0)
        probs = probs / probs.sum()

    if top_p:
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        probs[sorted_indices[sorted_indices_to_remove]] = 0
        probs = probs / probs.sum()

    return torch.multinomial(probs, 1).item()


def generate_fixed_length_poem(model, start_text, total_chars,
                               temperature, top_k, top_p, device, char2idx, idx2char):
    model.eval()
    input_ids = [char2idx.get(ch, 0) for ch in start_text]
    input_seq = torch.tensor([input_ids], dtype=torch.long).to(device)
    hidden = None
    result = start_text

    with torch.no_grad():
        while count_chinese_chars(result) < total_chars:
            output, hidden = model(input_seq, hidden)
            logits = output[:, -1, :]
            next_id = sample_from_logits(logits.squeeze(0), temperature, top_k, top_p)
            next_char = idx2char[next_id]
            # 跳过补齐符号
            if next_char == '<PAD>':
                continue

            result += next_char
            input_seq = torch.tensor([[next_id]], dtype=torch.long).to(device)

    return result


def count_chinese_chars(text):
    return len(re.findall(r'[\u4e00-\u9fff]', text))


def format_poem_pair_lines(text, line_length, line_count):
    # 删除标签，中文句号、逗号，换行符，回车符
    text = re.sub(r'(\[.*?\]|。|\n|\r|，)', '', text)

    # 只统计汉字，但保留原始标点
    chinese_chars = []
    char_positions = []  # 对应原始文本中的索引位置

    for idx, ch in enumerate(text):
        if re.match(r'[\u4e00-\u9fff]', ch):
            chinese_chars.append(ch)
            char_positions.append(idx)

    lines = []
    for i in range(line_count):
        start = i * line_length
        end = start + line_length
        if end > len(chinese_chars):
            break  # 汉字不足
        start_pos = char_positions[start]
        end_pos = char_positions[end - 1] + 1  # 保留最后一个汉字
        line = text[start_pos:end_pos]

        # 检查该行最后是否已有标点
        if not re.search(r'[，,。.!？；]$', line):
            if (i + 1) % 2 == 1:
                line += "，"
            else:
                line += "。"

        lines.append(line)

    # 每两句之间空一行
    formatted = []
    for i in range(0, len(lines), 2):
        formatted.append(lines[i] + lines[i + 1])
    return "\n".join(formatted).strip()


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open("checkpoints/cleanModel/char_vocab.pkl", "rb") as f:
        vocab_data = pickle.load(f)
    char2idx = vocab_data["char2idx"]
    idx2char = vocab_data["idx2char"]

    model = CharLSTM(vocab_size=len(char2idx))
    model.load_state_dict(
        torch.load(f"checkpoints/cleanModel/charlstm_epoch" + model_version + ".pt", map_location=device))
    model.to(device)

    poem_type_map = {
        "1": ("五言绝句", 4, 5),
        "2": ("七言绝句", 4, 7),
        "3": ("五言律诗", 8, 5),
        "4": ("七言律诗", 8, 7),
    }

    print("请选择诗体：")
    print("1. 五言绝句（20字）")
    print("2. 七言绝句（28字）")
    print("3. 五言律诗（40字）")
    print("4. 七言律诗（56字）")
    choice = input("请输入数字 (1-4)：").strip()
    _, line_count, line_length = poem_type_map.get(choice, ("七言律诗", 8, 7))
    total_chars = line_count * line_length

    start_text = input("请输入起始句（可留空）：").strip()

    temperature = float(input("请输入温度（0.7~1.0）："))
    top_k = input("请输入Top-K（整数，留空表示不使用）：").strip()
    top_p = input("请输入Top-P（0~1，留空表示不使用）：").strip()

    top_k = int(top_k) if top_k else None
    top_p = float(top_p) if top_p else None

    raw_poem = generate_fixed_length_poem(
        model=model,
        start_text=start_text,
        total_chars=total_chars,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        device=device,
        char2idx=char2idx,
        idx2char=idx2char
    )

    # 正则去除所有[xxxx]形式的标签及多余空行
    clean_poem = re.sub(r'\[.*?\]', '', raw_poem)
    clean_poem = re.sub(r'\n+', '\n', clean_poem).strip()
    print("\n生成古诗：")
    print(format_poem_pair_lines(clean_poem, line_length=line_length, line_count=line_count))
