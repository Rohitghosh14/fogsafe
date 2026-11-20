import numpy as np

def disp_to_depth(disp, k, lo=1.0, hi=120.0):
    """disp: relative inverse depth (bigger = nearer). Returns metres."""
    depth = k / (disp + 1e-3)
    return np.clip(depth, lo, hi)

def add_haze_depth(clear, depth_m, beta, A):
    """clear: (H, W, 3) numbers 0-255.  depth_m: (H, W) metres.  beta, A: single numbers."""
    J = clear.astype(np.float32) / 255.0
    t = np.exp(-beta * depth_m)          # shape (H, W): a different t for every pixel
    t = t[:, :, None]                    # shape (H, W, 1): so it multiplies all 3 color channels
    I = J * t + A * (1.0 - t)
    return (I * 255).astype(np.uint8)