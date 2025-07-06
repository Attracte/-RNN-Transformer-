import pickle
import re

import torch
import torch.nn.functional as F

from model import CharLSTM

model_version = "100"


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

def format_poem_one_line_one_sentence(text, max_lines=8):
    # 用标点符号断句
    sentences = re.split(r'([。！？；])', text)
    lines = []
    for i in range(0, len(sentences) - 1, 2):
        line = sentences[i].strip() + sentences[i+1]
        if len(line.strip()) >= 4:
            lines.append(line)
        if len(lines) >= max_lines:
            break
    return "\n".join(lines)

def generate_poem(model, start_text, length, temperature, top_k, top_p, device, char2idx, idx2char):
    model.eval()
    input_ids = [char2idx.get(ch, 0) for ch in start_text]
    input_seq = torch.tensor([input_ids], dtype=torch.long).to(device)
    hidden = None
    result = start_text

    end_tokens = {"。", "！", "？"}
    sentence_count = 0
    max_sentences = length // 7  # 根据字数估算句数

    with torch.no_grad():
        for _ in range(length):
            output, hidden = model(input_seq, hidden)
            logits = output[:, -1, :]
            next_id = sample_from_logits(logits.squeeze(0), temperature, top_k, top_p)
            next_char = idx2char[next_id]
            result += next_char
            input_seq = torch.tensor([[next_id]], dtype=torch.long).to(device)

            if next_char in end_tokens:
                sentence_count += 1
                if sentence_count >= max_sentences:
                    break

    return result

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open("checkpoints/char_vocab.pkl", "rb") as f:
        vocab_data = pickle.load(f)
    char2idx = vocab_data["char2idx"]
    idx2char = vocab_data["idx2char"]

    model = CharLSTM(vocab_size=len(char2idx))
    model.load_state_dict(torch.load(f"checkpoints/charlstm_epoch{model_version}.pt", map_location=device))
    model.to(device)

    start_text = input("请输入起始句（建议为完整短句）：").strip()
    temperature = float(input("请输入温度（0.7~1.0）："))
    top_k = input("请输入Top-K（整数，留空表示不使用）：")
    top_p = input("请输入Top-P（0~1，留空表示不使用）：")
    top_k = int(top_k) if top_k.strip() else None
    top_p = float(top_p) if top_p.strip() else None

    raw_text = generate_poem(
        model=model,
        start_text=start_text,
        length=96,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        device=device,
        char2idx=char2idx,
        idx2char=idx2char
    )

    print("\n生成古诗：")
    print(format_poem_one_line_one_sentence(raw_text, max_lines=8))
