import torch
import torch.nn as nn

layer=nn.Linear(1024,4096)
print(list(layer.parameters()))
print(layer.weight)