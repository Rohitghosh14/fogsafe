import torch
import torch.nn as nn
from unet import UNet
from transformer import TransformerBlock

class UNetT(UNet):
    """The plain U-Net, with transformer blocks at the bottom."""
    def __init__(self, base=24, n_blocks=2, heads=4):
        super().__init__(base)                       # everything the plain U-Net has
        dim = base * 8                               # 192 channels at the bottom
        self.pos = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.blocks = nn.ModuleList([TransformerBlock(dim, heads) for _ in range(n_blocks)])

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottom(self.pool(e3))                          # (B, 192, h, w)

        b = b + self.pos(b)                                     # tell every position where it is
        B, C, h, w = b.shape
        t = b.flatten(2).transpose(1, 2)                        # (B, h*w, 192): the tokens
        for blk in self.blocks:
            t = blk(t)                                          # every token listens to every token
        b = t.transpose(1, 2).reshape(B, C, h, w)               # back to a picture

        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return x + self.head(d1)