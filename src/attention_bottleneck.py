import torch
from unet import UNet

model = UNet()
x = torch.randn(2, 3, 192, 256)
with torch.no_grad():
    e1 = model.enc1(x)
    e2 = model.enc2(model.pool(e1))
    e3 = model.enc3(model.pool(e2))
    b = model.bottom(model.pool(e3))
    print("bottleneck picture:", tuple(b.shape))              # (2, 192, 24, 32)

    tokens = b.flatten(2).transpose(1, 2)                      # (2, 768, 192): 768 tokens of 192 numbers
    print("as tokens:", tuple(tokens.shape))

    scores = tokens @ tokens.transpose(1, 2) / (192 ** 0.5)    # (2, 768, 768)
    weights = torch.softmax(scores, dim=-1)
    print("attention table:", tuple(weights.shape))
    print("row 0 sums to:", weights[0, 0].sum().item())
    print("largest weight of token 0: %.4f | uniform would be %.4f" % (weights[0, 0].max().item(), 1 / 768))