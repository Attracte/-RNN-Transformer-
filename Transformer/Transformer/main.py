import re
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt


# =====================
# 数据预处理函数
# =====================
def preprocess_data(file_path):
    """
    读取并预处理诗词数据。
    - 分词：按字符切分。
    - 去除特殊字符，确保格式统一。
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        data = file.read()
    # 去除空行和特殊字符
    data = re.sub(r'[^\u4e00-\u9fa5，。！？\n]', '', data)
    # 按行分割
    poems = data.split('\n')
    poems = [poem for poem in poems if poem.strip()]
    return poems


# =====================
# 数据集定义
# =====================
class PoetryDataset(Dataset):
    def __init__(self, poems, tokenizer, max_length=128):
        self.poems = poems
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.poems)

    def __getitem__(self, idx):
        poem = self.poems[idx]
        encoding = self.tokenizer(poem, truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt')
        return encoding.input_ids.squeeze(), encoding.attention_mask.squeeze()


# =====================
# 模型训练函数
# =====================
def train_model(data_path, model_save_path, epochs=10, batch_size=8, lr=5e-5):
    """
    训练Transformer模型。
    :param data_path: 数据文件路径
    :param model_save_path: 模型保存路径
    :param epochs: 训练轮数
    :param batch_size: 批量大小
    :param lr: 学习率
    """
    # 加载数据
    poems = preprocess_data(data_path)
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    dataset = PoetryDataset(poems, tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 模型
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # 训练
    loss_history = []
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for batch in dataloader:
            inputs, attention_mask = batch
            outputs = model(inputs, attention_mask=attention_mask, labels=inputs)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            epoch_loss += loss.item()
        avg_loss = epoch_loss / len(dataloader)
        loss_history.append(avg_loss)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss}")

    # 保存模型
    model.save_pretrained(model_save_path)
    tokenizer.save_pretrained(model_save_path)

    # 绘制损失曲线
    plt.plot(loss_history, label="Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(f"{model_save_path}/loss_curve.png")
    plt.close()


# =====================
# 文本生成函数
# =====================
def generate_poem(start_text, length, top_k, top_p, temperature, model_path):
    """
    生成古诗。
    :param start_text: 起始文本
    :param length: 生成的诗词总长度
    :param top_k: Top-K采样
    :param top_p: Top-P采样
    :param temperature: 温度参数
    :param model_path: 模型路径
    :return: 生成的诗词
    """
    # 加载模型和tokenizer
    model = GPT2LMHeadModel.from_pretrained(model_path)
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)

    # 输入文本
    input_ids = tokenizer.encode(start_text, return_tensors='pt')

    # 生成
    output = model.generate(
        input_ids,
        max_length=length,
        top_k=top_k,
        top_p=top_p,
        temperature=temperature,
        do_sample=True
    )

    # 解码
    return tokenizer.decode(output[0], skip_special_tokens=True)


# =====================
# 主函数
# =====================
if __name__ == "__main__":
    # 配置参数
    data_path = "data/poems.txt"  # 数据文件路径
    model_save_path = "output/models/"  # 模型保存路径
    epochs = 10  # 训练轮数
    batch_size = 8  # 批量大小
    lr = 5e-5  # 学习率

    # 训练模型
    print("开始训练模型...")
    train_model(data_path, model_save_path, epochs, batch_size, lr)
    print("模型训练完成！")

    # 生成古诗
    start_text = "海畔尖山"  # 起始文本
    length = 50             # 诗词总长度
    top_k = 50              # Top-K采样
    top_p = 0.95            # Top-P采样
    temperature = 1.0       # 温度参数

    print("开始生成古诗...")
    poem = generate_poem(start_text, length, top_k, top_p, temperature, model_save_path)
    print("生成的诗词：")
    print(poem)

    # 保存生成结果
    with open('output/results/generated_poem.txt', 'w', encoding='utf-8') as file:
        file.write(poem)
