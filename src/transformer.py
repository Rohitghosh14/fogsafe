import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(dim, dim * 3)     # makes queries, keys and values in one go
        self.proj = nn.Linear(dim, dim)        # mixes the heads back together

    def forward(self, x):                      # x: (B, N, dim)
        B, N, C = x.shape
        qkv = self.qkv(x)                                          # (B, N, 3*C)
        qkv = qkv.reshape(B, N, 3, self.heads, C // self.heads)    # split into q/k/v and into heads
        q, k, v = qkv.permute(2, 0, 3, 1, 4)                       # each: (B, heads, N, C/heads)
        out = F.scaled_dot_product_attention(q, k, v)              # the Topic 5 recipe, for every head
        out = out.transpose(1, 2).reshape(B, N, C)                 # glue the heads together
        return self.proj(out)

class TransformerBlock(nn.Module):
    def __init__(self, dim=192, heads=4, mlp_ratio=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * mlp_ratio), nn.GELU(), nn.Linear(dim * mlp_ratio, dim))

    def forward(self, x):
        x = x + self.attn(self.norm1(x))       # listen, with a shortcut
        x = x + self.mlp(self.norm2(x))        # think, with a shortcut
        return x
    