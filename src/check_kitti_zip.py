import zipfile
from pathlib import Path

raw = Path("data/raw")

for name in ["data_object_label_2.zip", "data_object_image_2.zip"]:
    path = raw / name
    print(f"{name}: {path.stat().st_size / 1e9:.2f} GB")

    with zipfile.ZipFile(path) as z:
        names = z.namelist()
        print("  first 3 names:", names[:3])

        train = [n for n in names if n.startswith("training/") and not n.endswith("/")]
        test = [n for n in names if n.startswith("testing/") and not n.endswith("/")]
        print("  training files:", len(train))
        print("  testing files:", len(test))

        print("  checking for damage (can take a few minutes)...")
        bad = z.testzip()
        print("  damaged file:", bad if bad else "none")
