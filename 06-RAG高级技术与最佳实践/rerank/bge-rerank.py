# from modelscope import snapshot_download
#
# model_dir = snapshot_download('BAAI/bge-reranker-large', cache_dir='E:/projects/python/AI/models')

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model_dir = "E:/projects/python/AI/models/BAAI/bge-reranker-large"
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForSequenceClassification.from_pretrained(model_dir)

# 将模型移动到 GPU（如果可用）
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.eval()

pairs = [
    ['what is panda?', 'The giant panda is a bear species endemic to China.'],
    ['what is panda?', 'Pandas are cute.'],
    ['what is panda?', 'The Eiffel Tower is in Paris.']
]

# 编码，返回的张量默认在 CPU
inputs = tokenizer(pairs, padding=True, truncation=True, return_tensors='pt')

# 关键步骤：将所有输入张量移动到与模型相同的设备
inputs = {k: v.to(device) for k, v in inputs.items()}

with torch.no_grad():
    outputs = model(**inputs)
    scores = outputs.logits.view(-1).float()
    print(scores)