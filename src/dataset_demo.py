import numpy as np
import cv2
from torch.utils.data import DataLoader
from dataset import KittiFog

train = KittiFog("train", train=True)
val = KittiFog("val", train=False)
print("train images:", len(train), "| val images:", len(val))

img, disp, i = train[0]
print("one sample:", img.shape, img.dtype, "|", disp.shape, disp.dtype, "| image number", i)

loader = DataLoader(train, batch_size=8, shuffle=True, num_workers=0)
img_b, disp_b, ids_b = next(iter(loader))
print("one batch:", img_b.shape, disp_b.shape, ids_b.tolist())

# save the 8 crops as one picture, to look at them
tiles = [img_b[j].permute(1, 2, 0).numpy() for j in range(8)]        # back to (H, W, 3)
grid = np.vstack([np.hstack(tiles[:4]), np.hstack(tiles[4:])])
cv2.imwrite("results/batch_demo.png", cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
print("saved results/batch_demo.png", grid.shape)