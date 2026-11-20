import json
import numpy as np
import torch
from torch.utils.data import Dataset

ROOT = "data/processed/prepared/"

class KittiFog(Dataset):
    def __init__(self, split, train=False, crop_w=256):
        ids = open(ROOT + f"splits/{split}.txt").read().split()
        self.ids = [int(x) for x in ids]          # image numbers belonging to this split
        self.train = train
        self.crop_w = crop_w
        self.k = json.load(open(ROOT + "meta.json"))["depth_k"]
        self.images = np.load(ROOT + "images_192x640.npy", mmap_mode="r")
        self.disps = np.load(ROOT + "disp_96x320.npy", mmap_mode="r")

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, n):
        i = self.ids[n]
        img = np.array(self.images[i])                        # (192, 640, 3) uint8
        disp = np.array(self.disps[i]).astype(np.float32)     # (96, 320)
        if self.train:
            x0 = np.random.randint(0, (640 - self.crop_w) // 2 + 1) * 2    # always even
            img = img[:, x0:x0 + self.crop_w]                  # keep full height, cut a strip
            disp = disp[:, x0 // 2:(x0 + self.crop_w) // 2]    # same strip in the half-size depth map
            if np.random.rand() < 0.5:                         # flip half of the time
                img = img[:, ::-1]
                disp = disp[:, ::-1]
        img = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).contiguous()   # (3, H, W)
        disp = torch.from_numpy(np.ascontiguousarray(disp))[None]                         # (1, h, w)
        return img, disp, i
    