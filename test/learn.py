import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1024, 4096),
            nn.ReLU(),
            nn.Linear(4096, 1024),
        )

    def forward(self, x):
        return self.net(x)

block = Block().cpu()

x = torch.randn(8, 1024, device="cpu", requires_grad=True)

# 普通前向：保存 block 内部所有激活值
y = block(x)
print(dir(y.grad_fn))
print(y.grad_fn._saved_mat1)
print(y.grad_fn._saved_mat2)
# print("y_grad_fn=",y.grad_fn)
# print("block.net[0].weight.grad =", block.net[0].weight.grad)
# print("block.net[0].bias.grad =", block.net[0].bias.grad)
# print("block.net[2].weight.grad =", block.net[2].weight.grad)
# print("block.net[2].bias.grad =", block.net[2].bias.grad)
# loss = y.mean()
# loss.backward()
# print("----------------------------------after backward-----------------------")
# print("y_grad_fn=",y.grad_fn)
# print("block.net[0].weight.grad =", block.net[0].weight.grad)
# print("block.net[0].bias.grad =", block.net[0].bias.grad)
# print("block.net[2].weight.grad =", block.net[2].weight.grad)
# print("block.net[2].bias.grad =", block.net[2].bias.grad)