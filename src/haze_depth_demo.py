import json
import numpy as np
import cv2
from haze_depth import disp_to_depth, add_haze_depth

root = "data/processed/prepared/"
images = np.load(root + "images_192x640.npy", mmap_mode="r")
disps = np.load(root + "disp_96x320.npy", mmap_mode="r")
k = json.load(open(root + "meta.json"))["depth_k"]
print("k =", k)

i = 0
clear = np.array(images[i])                           # (192, 640, 3)
disp_small = np.array(disps[i]).astype(np.float32)    # (96, 320), half size
disp = cv2.resize(disp_small, (640, 192), interpolation=cv2.INTER_LINEAR)   # (192, 640)
depth = disp_to_depth(disp, k)
print("depth in metres: min %.1f  median %.1f  max %.1f" % (depth.min(), np.median(depth), depth.max()))

A = 0.9
levels = {"light": 0.02, "medium": 0.05, "dense": 0.08}
pieces = [clear] + [add_haze_depth(clear, depth, b, A) for b in levels.values()]
row = np.hstack(pieces)                               # clear | light | medium | dense
cv2.imwrite("results/haze_depth_demo.png", cv2.cvtColor(row, cv2.COLOR_RGB2BGR))
print("saved results/haze_depth_demo.png", row.shape)