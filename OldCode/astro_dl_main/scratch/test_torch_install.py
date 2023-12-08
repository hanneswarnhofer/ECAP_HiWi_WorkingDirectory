import torch
print(torch.cuda.is_available())
print("torch version", torch.__version__)
print(torch.backends.cudnn.version())
