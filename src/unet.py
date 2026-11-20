import torch
import torch.nn as nn
from blocks import ConvBlock

class UNet(nn.Module):
    def __init__(self, base=24):
        super().__init__()
        c1, c2, c3, c4 = base, base * 2, base * 4, base * 8      # 24, 48, 96, 192
        # going down
        self.enc1 = ConvBlock(3, c1)
        self.enc2 = ConvBlock(c1, c2)
        self.enc3 = ConvBlock(c2, c3)
        self.pool = nn.MaxPool2d(2)
        # the bottom
        self.bottom = ConvBlock(c3, c4)
        # going up
        self.up3 = nn.ConvTranspose2d(c4, c3, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(c3 * 2, c3)
        self.up2 = nn.ConvTranspose2d(c3, c2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(c2 * 2, c2)
        self.up1 = nn.ConvTranspose2d(c2, c1, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(c1 * 2, c1)
        # the output: a correction image with 3 channels
        self.head = nn.Conv2d(c1, 3, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)                                      # (24, H,   W)
        e2 = self.enc2(self.pool(e1))                          # (48, H/2, W/2)
        e3 = self.enc3(self.pool(e2))                          # (96, H/4, W/4)
        b = self.bottom(self.pool(e3))                         # (192, H/8, W/8)
        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))    # glue in e3 (skip)
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))   # glue in e2 (skip)
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))   # glue in e1 (skip)
        return x + self.head(d1)                               # input + correction