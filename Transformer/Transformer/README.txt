主要库：
import re
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
from tqdm import tqdm
import matplotlib.pyplot as plt
import time
import warnings



用法：
1. main.py：模型训练、作图和生成古诗脚本
2. 在命令行使用单独的生成命令：
命令行命令：
python generate_poem.py     --model_path ".\output\models\transformer_poem_model2.pth"     --start_text "海畔尖山"     --poem_type "五言绝句"     --seq_length 20     --length 50     --top_k 5     --top_p 0.9     --temperature 0.7
