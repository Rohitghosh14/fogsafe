import zipfile
import cv2
import numpy as np
from pathlib import Path

raw = Path("data/raw")
sample = "000000"

with zipfile.ZipFile(raw / "data_object_label_2.zip") as z:
    text = z.read(f"training/label_2/{sample}.txt").decode()
print("RAW LABEL FILE:")
print(text)

for line in text.strip().splitlines():
    p = line.split()
    kind = p[0]
    x1, y1, x2, y2 = map(float, p[4:8])
    X, Y, Z = map(float, p[11:14])
    print(f"{kind:12s} box=({x1:.0f},{y1:.0f})-({x2:.0f},{y2:.0f})  distance_forward={Z:.1f} m")

with zipfile.ZipFile(raw / "data_object_image_2.zip") as z:
    data = z.read(f"training/image_2/{sample}.png")
img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
print("image shape (height, width, channels):", img.shape)
cv2.imwrite("results/sample_000000.png", img)
print("saved results/sample_000000.png")
