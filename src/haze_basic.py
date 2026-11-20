import numpy as np

def add_haze(clear, t, A):
    """clear: image (H, W, 3) with numbers 0-255
    t: how much light survives, 0 to 1 (one number for the whole image, for now)
    A: brightness of the fog, 0 to 1"""
    J = clear.astype(np.float32) / 255.0   # the clear image as decimals 0..1
    I = J * t + A * (1.0 - t)              # the haze formula
    return (I * 255).astype(np.uint8)      # back to normal image numbers 0..255