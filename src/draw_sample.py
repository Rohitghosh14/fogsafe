import zipfile
import cv2
import numpy as np
from pathlib import Path

raw = Path("data/raw")
sample = "000000"

with zipfile.ZipFile(raw / "data_object_label_2.zip") as z:
    text = z.read(f"training/label_2/{sample}.txt").decode()
with zipfile.ZipFile(raw / "data_object_image_2.zip") as z:
    data = z.read(f"training/image_2/{sample}.png")

img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)

for line in text.strip().splitlines():
    p = line.split()
    if p[0] == "DontCare":
        continue
    x1, y1, x2, y2 = map(int, map(float, p[4:8]))
    z_dist = float(p[13])
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(img, f"{p[0]} {z_dist:.1f}m", (x1, max(y1 - 6, 12)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

cv2.imwrite("results/sample_000000_box.png", img)
print("saved results/sample_000000_box.png")
