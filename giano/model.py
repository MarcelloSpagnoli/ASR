import torch
from torch import nn

device = "gpu" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available()  else "cpu"
print(device)
