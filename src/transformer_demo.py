import math
import torch
import torch.nn.functional as F
from transformer import TransformerBlock

block = TransformerBlock(dim=192, heads=4)
print("block parameters:", sum(p.numel() for p in block.parameters()))

tokens = torch.randn(2, 768, 192)
out = block(tokens)
print("tokens", tuple(tokens.shape), "->", tuple(out.shape))

# check: PyTorch's fast attention equals our Topic 5 recipe
q = torch.randn(2, 4, 10, 48)
k = torch.randn(2, 4, 10, 48)
v = torch.randn(2, 4, 10, 48)
fast = F.scaled_dot_product_attention(q, k, v)
manual = torch.softmax(q @ k.transpose(-2, -1) / math.sqrt(48), dim=-1) @ v
print("fast attention equals the manual recipe:", torch.allclose(fast, manual, atol=1e-5))
