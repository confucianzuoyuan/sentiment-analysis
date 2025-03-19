from starlette.applications import Starlette
from starlette.responses import JSONResponse, FileResponse
from starlette.routing import Route
from starlette.middleware.cors import CORSMiddleware
import json
import os
import collections

import datasets
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
import transformers
import pandas as pd

seed = 1234

np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True

transformer_name = "bert-base-chinese"

tokenizer = transformers.AutoTokenizer.from_pretrained(transformer_name)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Transformer(nn.Module):
    def __init__(self, transformer, output_dim, freeze):
        super().__init__()
        self.transformer = transformer
        hidden_dim = transformer.config.hidden_size
        self.fc = nn.Linear(hidden_dim, output_dim)
        if freeze:
            for param in self.transformer.parameters():
                param.requires_grad = False

    def forward(self, ids):
        # ids = [batch size, seq len]
        output = self.transformer(ids, output_attentions=True)
        hidden = output.last_hidden_state
        # hidden = [batch size, seq len, hidden dim]
        attention = output.attentions[-1]
        # attention = [batch size, n heads, seq len, seq len]
        cls_hidden = hidden[:, 0, :]
        prediction = self.fc(torch.tanh(cls_hidden))
        # prediction = [batch size, output dim]
        return prediction

transformer = transformers.AutoModel.from_pretrained(transformer_name)

model = Transformer(transformer, 2, False)

model.load_state_dict(torch.load("./fine-tuning-Bert-for-sentiment-analysis-master/transformer.pt"))

model = model.to(device)

def predict_sentiment(text, model, tokenizer, device):
    ids = tokenizer(text)["input_ids"]
    tensor = torch.LongTensor(ids).unsqueeze(dim=0).to(device)
    prediction = model(tensor).squeeze(dim=0)
    probability = torch.softmax(prediction, dim=-1)
    predicted_class = prediction.argmax(dim=-1).item()
    predicted_probability = probability[predicted_class].item()
    return predicted_class, predicted_probability

# 定义处理请求的视图
async def submit(request):
    data = await request.json()
    res = predict_sentiment(data['content'], model, tokenizer, device)
    if res[0] > 0.5:
        return JSONResponse({'result': '好评！'})
    else:
        return JSONResponse({'result': '差评！'})

# 定义静态文件处理视图
async def serve_index(request):
    return FileResponse('index.html')

# 定义路由
routes = [
    Route('/submit', submit, methods=["POST"]),
    Route('/', serve_index),
]

# 创建 Starlette 应用
app = Starlette(debug=True, routes=routes)

# 添加 CORS 中间件以允许跨域请求（可选）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

