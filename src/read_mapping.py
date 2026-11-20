import zipfile
import re
from collections import Counter

with zipfile.ZipFile("data/raw/devkit_object.zip") as z:
    rand_text = z.read("mapping/train_rand.txt").decode()
    map_text = z.read("mapping/train_mapping.txt").decode()

print("rand file starts with:", rand_text[:60].replace("\n", "\\n"))
print("mapping file starts with:")
for line in map_text.splitlines()[:3]:
    print("   ", line)

rand = [int(x) for x in re.split(r"[,\s]+", rand_text.strip()) if x]
mapping = [line.split() for line in map_text.strip().splitlines()]
print("rand entries:", len(rand), "| mapping lines:", len(mapping))

# image i -> rand[i] (1-based) -> mapping line rand[i] - 1 (0-based)
drive_of = {}
for i, idx in enumerate(rand):
    date, drive, frame = mapping[idx - 1]
    drive_of[i] = drive

print("image 0 ->", rand[0], mapping[rand[0] - 1])

counts = Counter(drive_of.values())
print("number of drives:", len(counts))
for d, n in sorted(counts.items()):
    print(f"   {d}: {n} images")
