import pickle

import torch
from flask import Flask, request, render_template

from RNN.generate import generate_fixed_length_poem, format_poem_pair_lines
from RNN.model import CharLSTM

app = Flask(__name__)

# 加载词表
with open("RNN/checkpoints/cleanModel/char_vocab.pkl", "rb") as f:
    vocab_data = pickle.load(f)
char2idx = vocab_data["char2idx"]
idx2char = vocab_data["idx2char"]

# 加载模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CharLSTM(vocab_size=len(char2idx))
model.load_state_dict(torch.load("RNN/checkpoints/cleanModel/charlstm_epoch100.pt", map_location=device))
model.to(device)
model.eval()

poem_type_map = {
    "1": ("五言绝句", 4, 5),
    "2": ("七言绝句", 4, 7),
    "3": ("五言律诗", 8, 5),
    "4": ("七言律诗", 8, 7),
}


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    error = None

    # 默认参数值
    form_data = {
        "poem_type": "4",
        "start_text": "",
        "temperature": "1.0",
        "top_k": "",
        "top_p": "",
    }

    if request.method == "POST":
        form_data.update({
            "poem_type": request.form["poem_type"],
            "start_text": request.form["start_text"].strip(),
            "temperature": request.form["temperature"].strip(),
            "top_k": request.form["top_k"].strip(),
            "top_p": request.form["top_p"].strip(),
        })

        # 校验起始句和温度
        if not form_data["start_text"]:
            error = "起始句不能为空，请输入一句古诗开头！"
        elif not form_data["temperature"]:
            error = "温度不能为空，请输入一个 0.7~1.0 之间的数值！"
        else:
            try:
                temperature = float(form_data["temperature"])
                if temperature <= 0 or temperature > 2:
                    raise ValueError()
            except:
                error = "温度必须是一个合理的数字（如 0.8）！"

        if not error:
            top_k = int(form_data["top_k"]) if form_data["top_k"] else None
            top_p = float(form_data["top_p"]) if form_data["top_p"] else None
            _, line_count, line_length = poem_type_map.get(form_data["poem_type"], ("七言律诗", 8, 7))
            total_chars = line_count * line_length

            raw_poem = generate_fixed_length_poem(
                model=model,
                start_text=form_data["start_text"],
                total_chars=total_chars,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                device=device,
                char2idx=char2idx,
                idx2char=idx2char
            )
            result = format_poem_pair_lines(raw_poem, line_length, line_count)

    return render_template("index.html", result=result, error=error, form=form_data)


if __name__ == "__main__":
    app.run(debug=True)
