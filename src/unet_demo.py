import torch
from torch.utils.data import DataLoader
from unet import UNet
from dataset import KittiFog

model = UNet()
print("U-Net parameters:", sum(p.numel() for p in model.parameters()))

x = torch.randn(2, 3, 192, 256)
y = model(x)
print("input", tuple(x.shape), "-> output", tuple(y.shape))

with torch.no_grad():                                          # look inside the U
    e1 = model.enc1(x)
    e2 = model.enc2(model.pool(e1))
    e3 = model.enc3(model.pool(e2))
    b = model.bottom(model.pool(e3))
for name, t in [("e1", e1), ("e2", e2), ("e3", e3), ("bottom", b)]:
    print(f"  {name:7s}", tuple(t.shape))

# identity trick: if the correction head is all zeros, the output must equal the input
img, disp, ids = next(iter(DataLoader(KittiFog("train", train=True), batch_size=4, shuffle=True)))
img = img.float() / 255.0
with torch.no_grad():
    model.head.weight.zero_()
    model.head.bias.zero_()
    out = model(img)
print("with the head set to zero, max |output - input| =", (out - img).abs().max().item())