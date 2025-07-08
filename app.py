import pickle

import torch
from flask import Flask, render_template, request

from RNN.generate import generate_fixed_length_poem as rnn_generate
from RNN.model import CharLSTM

# from Transformer.generate import generate_fixed_length_poem as transformer_generate  # 假设路径
# from Transformer.model import TransformerModel  # 假设路径

app = Flask(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 路径映射
model_configs = {
    "rnn": {
        "model_class": CharLSTM,
        "model_path": "RNN/checkpoints/charlstm_epoch.pt",
        "vocab_path": "RNN/checkpoints/char_vocab.pkl",
        "generate_func": rnn_generate
    },
    # "transformer": {
    #     "model_class": TransformerModel,
    #     "model_path": "Transformer/checkpoints/transformer_epoch.pt",
    #     "vocab_path": "Transformer/checkpoints/char_vocab.pkl",
    #     "generate_func": transformer_generate
    # }
}

poem_type_map = {
    "1": ("五言绝句", 4, 5),
    "2": ("七言绝句", 4, 7),
    "3": ("五言律诗", 8, 5),
    "4": ("七言律诗", 8, 7),
}


@app.route("/", methods=["GET", "POST"])
def index():
    result = ""
    if request.method == "POST":
        model_type = request.form.get("model_type", "rnn")
        poem_type = request.form.get("poem_type", "4")
        start_text = request.form.get("start_text", "")
        temperature = float(request.form.get("temperature", 1.0))
        top_k = request.form.get("top_k", "")
        top_p = request.form.get("top_p", "")

        top_k = int(top_k) if top_k else None
        top_p = float(top_p) if top_p else None
        _, line_count, line_length = poem_type_map[poem_type]
        total_chars = line_count * line_length

        # 模型加载逻辑
        config = model_configs[model_type]
        with open(config["vocab_path"], "rb") as f:
            vocab = pickle.load(f)
        char2idx = vocab["char2idx"]
        idx2char = vocab["idx2char"]

        model = config["model_class"](vocab_size=len(char2idx))
        model.load_state_dict(torch.load(config["model_path"], map_location=device))
        model.to(device)
        model.eval()

        # 使用不同生成函数
        raw = config["generate_func"](
            model=model,
            start_text=start_text,
            total_chars=total_chars,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            device=device,
            char2idx=char2idx,
            idx2char=idx2char,
            line_length=line_length
        )

        # 统一格式化输出
        from RNN.generate import format_poem_pair_lines
        result = format_poem_pair_lines(raw, line_length, line_count)

    return render_template("index.html", result=result)


if __name__ == '__main__':
    app.run(debug=True)
