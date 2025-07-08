#!/usr/bin/env python
# coding: utf-8

# In[11]:


import re
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
from tqdm import tqdm
import matplotlib.pyplot as plt
import time
import warnings
warnings.filterwarnings("ignore")

# 数据预处理函数
def preprocess_data_char_level(file_path, seq_length):
    """
    读取并预处理诗词数据，按字符切分，过滤掉过短的诗。
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        data = file.read()
    
    # 提取诗词内容（去除标题和空行）
    poems = []
    current_poem = []
    for line in data.split('\n'):
        line = line.strip()
        if line.startswith('[') and line.endswith(']'):  # 跳过标题行
            if current_poem:
                poems.append(''.join(current_poem))
                current_poem = []
        elif line:  # 只处理非空行
            # 去除标点符号外的特殊字符
            cleaned_line = re.sub(r'[^\u4e00-\u9fa5，。！？、；：]', '', line)
            current_poem.append(cleaned_line)
    
    if current_poem:  # 添加最后一首诗
        poems.append(''.join(current_poem))
    
    # 过滤掉过短的诗
    poems = [poem for poem in poems if len(poem) >= seq_length]
    print(f"预处理后的诗词数量: {len(poems)}")
    print(f"示例诗词: {poems[0]}")  # 打印第一首诗作为示例
    return poems



# 创建字符级别的词表

def create_vocab(poems):
    """
    创建字符级别的词表。
    """
    all_chars = ''.join(poems)
    unique_chars = sorted(list(set(all_chars)))  # 排序以保证一致性
    # 添加特殊标记
    unique_chars = ['<PAD>', '<UNK>'] + unique_chars
    char2idx = {char: idx for idx, char in enumerate(unique_chars)}
    idx2char = {idx: char for idx, char in enumerate(unique_chars)}
    print(f"词表大小: {len(char2idx)}")
    return char2idx, idx2char


# 数据集定义

class CharLevelDataset(Dataset):
    def __init__(self, poems, char2idx, seq_length):
        self.data = []
        self.char2idx = char2idx
        self.seq_length = seq_length
        self.unk_idx = char2idx.get('<UNK>', 0)

        for poem in poems:
            # 将字符转换为索引，未知字符用UNK代替
            encoded = [char2idx.get(char, self.unk_idx) for char in poem]
            # 生成训练样本
            for i in range(len(encoded) - seq_length):
                self.data.append((encoded[i:i + seq_length], encoded[i + 1:i + seq_length + 1]))
        
        print(f"生成的数据样本数量: {len(self.data)}")
        if len(self.data) > 0:
            print(f"样本示例 - 输入: {self.data[0][0]}, 输出: {self.data[0][1]}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x, y = self.data[idx]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)


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
        # 动态扩展位置编码以匹配输入序列长度
        if x.size(1) > self.positional_encoding.size(1):
            extra_positional_encoding = torch.zeros(
                1, x.size(1) - self.positional_encoding.size(1), self.positional_encoding.size(2),
                device=x.device  # 确保扩展部分与输入张量在同一设备
            )
            self.positional_encoding = torch.cat([self.positional_encoding.to(x.device), extra_positional_encoding], dim=1)

        # Embedding + Positional Encoding
        x = self.embedding(x) + self.positional_encoding[:, :x.size(1), :].to(x.device)
        y = self.embedding(y) + self.positional_encoding[:, :y.size(1), :].to(x.device)

        # Transformer
        tgt_mask = self.generate_square_subsequent_mask(y.size(1)).to(x.device)
        out = self.transformer(x, y, tgt_mask=tgt_mask)

        # 输出层
        out = self.fc(out)
        return out

    def generate_square_subsequent_mask(self, size):
        """
        生成用于 Transformer 的解码器掩码，防止看到未来的时间步。
        """
        mask = torch.triu(torch.ones(size, size), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask

def plot_metrics(metrics_path, losses, perplexities, accuracies, inference_times):
    """
    绘制训练过程中的指标曲线，包括 Loss、PPL、Accuracy 和推理时间。
    """
    # 绘制 Loss 曲线
    plt.figure()
    plt.plot(range(1, len(losses) + 1), losses, label='Loss', color='blue')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.legend()
    plt.savefig(os.path.join(metrics_path, 'loss_curve.png'))
    plt.close()

    # 绘制 PPL 曲线
    plt.figure()
    plt.plot(range(1, len(perplexities) + 1), perplexities, label='PPL', color='green')
    plt.xlabel('Epochs')
    plt.ylabel('Perplexity')
    plt.title('Training Perplexity Curve')
    plt.legend()
    plt.savefig(os.path.join(metrics_path, 'ppl_curve.png'))
    plt.close()

    # 绘制 Accuracy 曲线
    plt.figure()
    plt.plot(range(1, len(accuracies) + 1), accuracies, label='Accuracy', color='orange')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training Accuracy Curve')
    plt.legend()
    plt.savefig(os.path.join(metrics_path, 'accuracy_curve.png'))
    plt.close()

    # 绘制推理时间曲线
    plt.figure()
    plt.plot(range(1, len(inference_times) + 1), inference_times, label='Inference Time', color='red')
    plt.xlabel('Epochs')
    plt.ylabel('Time (s)')
    plt.title('Inference Time Curve')
    plt.legend()
    plt.savefig(os.path.join(metrics_path, 'inference_time_curve.png'))
    plt.close()

    print(f"所有指标曲线已保存到 {metrics_path}")
    
from sklearn.metrics import ConfusionMatrixDisplay, roc_curve, auc

def plot_confusion_matrix_and_roc(metrics_path, y_true, y_pred, num_classes, max_classes=10):
    """
    绘制混淆矩阵和 ROC 曲线，限制最多显示 max_classes 个类别。
    """
    # 限制类别数量
    if num_classes > max_classes:
        print(f"类别数量过多（{num_classes}），仅绘制前 {max_classes} 个类别。")
        y_true = np.clip(y_true, 0, max_classes - 1)
        y_pred = np.clip(y_pred, 0, max_classes - 1)
        num_classes = max_classes

    # 绘制混淆矩阵
    cm = confusion_matrix(y_true, y_pred, labels=range(num_classes))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=range(num_classes))
    disp.plot(cmap=plt.cm.Blues, xticks_rotation='vertical')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(metrics_path, 'confusion_matrix.png'))
    plt.close()

    # 绘制 ROC 曲线
    # ROC 曲线需要二分类问题，这里以每个类别计算一条 ROC 曲线
    fpr = {}
    tpr = {}
    roc_auc = {}
    y_true_one_hot = np.eye(num_classes)[y_true]  # 将 y_true 转换为 one-hot 编码
    y_pred_one_hot = np.eye(num_classes)[y_pred]  # 将 y_pred 转换为 one-hot 编码

    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_one_hot[:, i], y_pred_one_hot[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # 绘制所有类别的 ROC 曲线
    plt.figure()
    for i in range(num_classes):
        plt.plot(fpr[i], tpr[i], lw=2, label=f'Class {i} (AUC = {roc_auc[i]:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.savefig(os.path.join(metrics_path, 'roc_curve.png'))
    plt.close()

    print(f"混淆矩阵和 ROC 曲线已保存到 {metrics_path}")


def generate_text(model, start_text, char2idx, idx2char, seq_length, length, poem_type, temperature=0.8, device='cuda'):
    """
    使用训练好的模型生成文本，确保标点符号只出现在行尾
    """
    model.eval()
    unk_idx = char2idx.get('<UNK>', 0)
    
    # 初始化输入序列
    input_seq = [char2idx.get(char, unk_idx) for char in start_text]
    input_seq = torch.tensor(input_seq, dtype=torch.long).unsqueeze(0).to(device)
    
    # 标题
    generated_text = start_text + "\n\n"  # 标题和正文之间添加换行

    # 控制行数和每行字数
    num_lines = 4 if "绝句" in poem_type else 8  # 绝句4行，律诗8行
    line_length = 5 if "五言" in poem_type else 7  # 五言5字，七言7字
    punctuation = ['，', '。']  # 可用的标点符号

    with torch.no_grad():
        for line_idx in range(1, num_lines + 1):  # 从1开始计数以便判断单双数行
            line = ""
            for char_idx in range(line_length):  # 控制每行的字数
                tgt_input = input_seq[:, -seq_length:]  # 限制tgt_input的长度不超过seq_length
                output = model(input_seq, tgt_input)
                
                # 只取最后一个时间步的输出
                output = output[:, -1, :] / temperature
                probs = torch.softmax(output, dim=-1)
                
                # 在整个诗句生成过程中完全禁止生成标点
                for p in punctuation:
                    if p in char2idx:
                        probs[0, char2idx[p]] = 0
                # 重新归一化概率
                probs = probs / probs.sum()
                
                # 从概率分布中采样
                next_char_idx = torch.multinomial(probs, num_samples=1).item()
                next_char = idx2char[next_char_idx]
                
                line += next_char
                input_seq = torch.cat([input_seq, torch.tensor([[next_char_idx]], dtype=torch.long).to(device)], dim=1)
            
            # 在行尾手动添加标点符号（不在模型生成范围内）
            if line_idx % 2 == 1:  # 单数行（1、3、5、7）
                line += "，"
            else:  # 双数行（2、4、6、8）
                line += "。"
            
            generated_text += line + "\n"
    
    return generated_text

# 模型训练函数

import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc

def train_transformer(data_path, model_save_path, metrics_path, seq_length=20, embedding_dim=128, num_heads=4, 
                      num_layers=2, ff_dim=512, epochs=20, batch_size=64, lr=0.001, device='cuda'):
    # 数据预处理
    poems = preprocess_data_char_level(data_path, seq_length)
    char2idx, idx2char = create_vocab(poems)
    dataset = CharLevelDataset(poems, char2idx, seq_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 模型初始化
    model = CharLevelTransformer(len(char2idx), embedding_dim, num_heads, num_layers, ff_dim, seq_length).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # 训练指标
    losses = []
    perplexities = []
    accuracies = []
    inference_times = []

    # 用于记录所有真实标签和预测标签
    all_labels = []
    all_predictions = []

    # 训练
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        total_correct = 0
        total_samples = 0
        epoch_start_time = time.time()

        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for x, y in progress_bar:
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            output = model(x, y[:, :-1])
            
            # 计算损失
            loss = criterion(output.view(-1, output.size(-1)), y[:, 1:].reshape(-1))
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()

            # 计算准确率
            predictions = output.argmax(dim=-1)
            correct = (predictions == y[:, 1:]).sum().item()
            total_correct += correct
            total_samples += y[:, 1:].numel()

            # 记录真实标签和预测标签
            all_labels.extend(y[:, 1:].cpu().numpy().flatten())  # 展平真实标签
            all_predictions.extend(predictions.cpu().numpy().flatten())  # 展平预测标签

            progress_bar.set_postfix({"Loss": loss.item()})

        avg_loss = total_loss / len(dataloader)
        accuracy = total_correct / total_samples
        perplexity = np.exp(avg_loss)  # PPL 是 Loss 的指数形式
        epoch_end_time = time.time()
        inference_time = epoch_end_time - epoch_start_time

        losses.append(avg_loss)
        accuracies.append(accuracy)
        perplexities.append(perplexity)
        inference_times.append(inference_time)

        print(f"Epoch {epoch+1}/{epochs}, Avg Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}, PPL: {perplexity:.4f}, Inference Time: {inference_time:.2f}s")

    # 保存模型和词表
    torch.save(model.state_dict(), model_save_path)
    torch.save({'char2idx': char2idx, 'idx2char': idx2char}, 
               model_save_path.replace('.pth', '_vocab.pth'))
    print(f"模型和词表已保存到 {model_save_path}")

    # 绘制并保存曲线
    os.makedirs(metrics_path, exist_ok=True)
    plot_metrics(metrics_path, losses, perplexities, accuracies, inference_times)

    # 绘制混淆矩阵和 ROC 曲线
    plot_confusion_matrix_and_roc(metrics_path, all_labels, all_predictions, num_classes=len(char2idx))


# In[12]:



# 主程序入口
if __name__ == "__main__":
    # 配置参数
    data_path = r"F:\poem generation\data\poems_small.txt"
    model_save_path = r"F:\poem generation\output\models\transformer_poem_model2.pth"
    metrics_path = r"F:\poem generation\output\metrics"
    results_path = r"F:\poem generation\output\results"

    # 训练参数
    seq_length = 20
    embedding_dim = 128
    num_heads = 4
    num_layers = 2
    ff_dim = 512
    epochs = 5
    batch_size = 8
    lr = 0.001
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")

    # 训练模型
    train_transformer(data_path, model_save_path, metrics_path, seq_length, embedding_dim, num_heads, 
                      num_layers, ff_dim, epochs, batch_size, lr, device)

    # 加载模型和词表
    vocab = torch.load(model_save_path.replace('.pth', '_vocab.pth'))
    char2idx, idx2char = vocab['char2idx'], vocab['idx2char']
    
    model = CharLevelTransformer(len(char2idx), embedding_dim, num_heads, num_layers, ff_dim, seq_length).to(device)
    model.load_state_dict(torch.load(model_save_path, map_location=device))

    # 生成文本示例
    start_texts = ["海畔尖山", "赠君一法", "怒风狂雨"]
    poem_types = ["五言绝句", "七言绝句", "七言律诗"]
    for start, poem_type in zip(start_texts, poem_types):
        generated = generate_text(model, start, char2idx, idx2char, seq_length, 
                                  length=50, poem_type=poem_type, temperature=0.7, device=device)
        print(f"\n生成的诗句 ({poem_type}):")
        print(generated)

        # 保存生成结果
        os.makedirs(results_path, exist_ok=True)
        result_file = os.path.join(results_path, f"{start}_{poem_type}_generated.txt")
        with open(result_file, 'w', encoding='utf-8') as f:
            f.write(generated)
        print(f"生成的诗句已保存到 {result_file}")


# In[16]:


import torch

# 加载保存的模型和词表
model_save_path = r"F:\poem generation\output\models\transformer_poem_model2.pth"
vocab_path = model_save_path.replace('.pth', '_vocab.pth')

# 加载词表
vocab = torch.load(vocab_path)
char2idx, idx2char = vocab['char2idx'], vocab['idx2char']

# 加载模型
seq_length = 20  # 设置与训练时相同的序列长度
embedding_dim = 128
num_heads = 4
num_layers = 2
ff_dim = 512
device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = CharLevelTransformer(len(char2idx), embedding_dim, num_heads, num_layers, ff_dim, seq_length).to(device)
model.load_state_dict(torch.load(model_save_path, map_location=device))
model.eval()

    # 生成文本示例
start_texts = ["海畔尖山", "赠君一法", "怒风狂雨"]
poem_types = ["五言绝句", "七言绝句", "七言律诗"]
for start, poem_type in zip(start_texts, poem_types):
        generated = generate_text(model, start, char2idx, idx2char, seq_length, 
                                  length=50, poem_type=poem_type, temperature=0.7, device=device)
        print(f"\n生成的诗句 ({poem_type}):")
        print(generated)

        # 保存生成结果
        os.makedirs(results_path, exist_ok=True)
        result_file = os.path.join(results_path, f"{start}_{poem_type}_generated.txt")
        with open(result_file, 'w', encoding='utf-8') as f:
            f.write(generated)
        print(f"生成的诗句已保存到 {result_file}")


# In[ ]:




