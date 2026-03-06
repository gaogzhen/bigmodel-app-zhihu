import torch

print("PyTorch 版本:", torch.__version__)
print("CUDA 是否可用:", torch.cuda.is_available())
print("GPU 数量:", torch.cuda.device_count())

if torch.cuda.is_available():
    print("GPU 型号:", torch.cuda.get_device_name(0))