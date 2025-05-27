import torch

print(torch.cuda.is_available())         # True se houver GPU
print(torch.cuda.device_count())         # Número de GPUs
print(torch.cuda.get_device_name(0))