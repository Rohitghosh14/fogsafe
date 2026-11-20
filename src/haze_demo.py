import numpy as np
import cv2
from haze_basic import add_haze

images = np.load("data/processed/prepared/images_192x640.npy", mmap_mode="r")
print("all images:", images.shape, images.dtype)

clear = np.array(images[0])        # one image, copied into memory
A = 0.9
pieces = [clear] + [add_haze(clear, t, A) for t in (0.8, 0.5, 0.2)]
stack = np.vstack(pieces)          # four pictures on top of each other

cv2.imwrite("results/haze_demo.png", cv2.cvtColor(stack, cv2.COLOR_RGB2BGR))
print("saved results/haze_demo.png", stack.shape)
