import torch
from torch.utils.data import DataLoader
from unet import UNet
from model import UNetT
from dataset import KittiFog

count = lambda m: sum(p.numel() for p in m.parameters())
plain, ours = UNet(), UNetT()
print("plain U-Net :", count(plain))
print("our UNetT   :", count(ours), "| extra:", count(ours) - count(plain))
print("  transformer blocks:", count(ours.blocks), "| position conv:", count(ours.pos))

x = torch.randn(2, 3, 192, 256)
print("output shape:", tuple(ours(x).shape))

with torch.no_grad():                                    # look inside the middle
    e1 = ours.enc1(x)
    e2 = ours.enc2(ours.pool(e1))
    e3 = ours.enc3(ours.pool(e2))
    b = ours.bottom(ours.pool(e3))
    t = (b + ours.pos(b)).flatten(2).transpose(1, 2)
    print("bottleneck:", tuple(b.shape), "-> tokens:", tuple(t.shape))
    for blk in ours.blocks:
        t = blk(t)
    print("after transformer, tokens:", tuple(t.shape))

# wiring check: does a learning signal reach the transformer?
loss = ((ours(x) - torch.rand_like(x)) ** 2).mean()
loss.backward()
g = ours.blocks[0].attn.qkv.weight.grad
print("gradient reaches the transformer:", g is not None and g.abs().sum().item() > 0)

# identity check: with the head set to zero, the output must equal the input
img, disp, ids = next(iter(DataLoader(KittiFog("train", train=True), batch_size=4, shuffle=True)))
img = img.float() / 255.0
with torch.no_grad():
    ours.head.weight.zero_()
    ours.head.bias.zero_()
    print("zero head -> max |out - in| =", (ours(img) - img).abs().max().item())