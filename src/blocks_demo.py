import torch
from torch.utils.data import DataLoader
from blocks import ConvBlock
from dataset import KittiFog

block = ConvBlock(3, 24)
n_params = sum(p.numel() for p in block.parameters())
print("block parameters:", n_params)

x = torch.randn(1, 3, 192, 256)              # one random 'image'
y = block(x)
print("random input", tuple(x.shape), "->", tuple(y.shape))

loader = DataLoader(KittiFog("train", train=True), batch_size=8, shuffle=True)
img, disp, ids = next(iter(loader))
img = img.float() / 255.0                    # uint8 0..255 -> decimals 0..1
with torch.no_grad():
    out = block(img)
print("real batch", tuple(img.shape), "->", tuple(out.shape))
print("output min %.3f  max %.3f  (ReLU: nothing below 0)" % (out.min().item(), out.max().item()))