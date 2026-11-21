# FogSafe

**Compact, fog-robust driving perception: dehazing, object detection and distance estimation for low-compute devices.**

> **Status: work in progress.** The checklist at the bottom shows what is done and what is left. Nothing in this README claims more than has been measured.

---

## 1. What this project does

Fog hides road users from a camera, and the dangerous error is a *missed* pedestrian or vehicle. This project asks one question:

> Can a **small** model (under about 5M deployed parameters, 30+ FPS) keep detection reliable in fog, and how does it compare with well-known dehazing models when everyone is trained and tested under the *same* recipe?

**Main ideas**

1. **UNetT (our model):** a U-Net whose bottleneck is replaced by two transformer blocks. At the bottleneck the picture is tiny (768 positions for a 192x256 crop), so attention is cheap there, and it gives the decoder a direct view of global fog colour and thickness.
2. **Judge by safety, not only by picture quality:** PSNR/SSIM measure how nice the image looks; detection accuracy and **pedestrian/vehicle recall at each fog level** measure what matters on the road.
3. **Joint training (planned):** train a dehazing head on the detector's features and *drop it at inference*, so fog-robust features cost nothing at deployment.
4. **Fair comparison:** AOD-Net, FFA-Net, U-Net and ViT dehazers share one data split, one fog generator, one loss, one optimizer, one schedule and one training budget.

**Scope:** camera pipeline only. Radar and driver input (steering, brake) are documented *future work*.

---

## 2. How we do it

```
KITTI images + labels
   |  resize to 192x640, split BY DRIVE (141 drives)
   v
Depth Anything V2 (Small)  ->  relative inverse depth  ->  metres (one constant k = 44.57, fitted on KITTI distances)
   v
fog added on the fly:   I = J*t + A*(1 - t),   t = exp(-beta * depth)
   v
dehazer: AOD-Net | U-Net | ViT | FFA-Net | UNetT (ours)  -->  [planned] YOLOv8n detector  -->  boxes + distance
```

### Our model: UNetT

```
 input --ConvBlock--> e1 (24,192,256) ----------------------------------------+ skip
   | pool                                                                      |
 ConvBlock --> e2 (48,96,128) -----------------------------+ skip              |
   | pool                                                   |                  |
 ConvBlock --> e3 (96,48,64) -----------------+ skip        |                  |
   | pool                                      |             |                  |
 ConvBlock --> BOTTOM (192,24,32)              |             |                  |
   | + position conv (3x3, depth-wise)                                          
   | flatten -> 768 tokens x 192 numbers
   | 2 x TransformerBlock (4 heads, MLP x4)  <-- every position listens to every other
   | fold back to a picture
   v  decoder (up, glue the skips, ConvBlock) ... -> 1x1 conv -> correction
 output = input + correction
```

| Model | Parameters | Notes |
|---|---|---|
| Plain U-Net (`base=24`) | 1,084,731 | the baseline and our starting point |
| **UNetT (ours)** | **1,976,379** | +889,728 (two transformer blocks) +1,920 (position conv) |
| Plain U-Net (`base=32`) | 1,927,075 | **parameter-matched control**, so a gain can be credited to the transformer and not to extra size |

Calculated cost of the transformer: about **+1.13 GMACs** for a 192x256 crop (about 3.9 to 5.0 GMACs, +29%).

### Fair-comparison recipe (identical for every dehazer)

- Same data and drive-level split, same fog generator, same L1 loss
- AdamW (no weight decay), 200-step warm-up then cosine decay to 1%, gradient clip 1.0, mixed precision (fp16)
- Effective batch 8, **8,000 steps** each, 192-row x 256-column random crops for training
- Only the learning rate differs per model (AOD-Net 1e-3, U-Net 5e-4, ViT 2e-4, FFA-Net 5e-5 after instability at 1e-4)
- Evaluation on the held-out test drives at clear, light, medium and dense fog

---

## 3. Dataset

### Dataset used: KITTI 2D Object Detection

| | Link |
|---|---|
| **Official page** | https://www.cvlibs.net/datasets/kitti/eval_object.php (choose the *2D object* benchmark) |
| Left colour images (about 12 GB) | https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_image_2.zip |
| Training labels (5 MB) | https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_label_2.zip |
| Development kit (63 KB, drive mapping) | https://s3.eu-central-1.amazonaws.com/avg-kitti/devkit_object.zip |

**Citation:** A. Geiger, P. Lenz, R. Urtasun, *Are we ready for autonomous driving? The KITTI vision benchmark suite*, CVPR 2012.
KITTI has its own terms of use; please read them on the official page before redistributing anything derived from it. **No data is stored in this repository.**

### Other models and data involved

| Item | Role | Link |
|---|---|---|
| Depth Anything V2 (Small), pretrained | estimates depth to build the fog | https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf |
| YOLOv8n, COCO-pretrained (Ultralytics) | detector to be fine-tuned (planned) | https://github.com/ultralytics/ultralytics |
| RESIDE / RTTS | real foggy photos for a real-fog check (planned, not used yet) | https://sites.google.com/view/reside-dehaze-datasets |

### Modifications we made to the data

| # | Modification | Details |
|---|---|---|
| 1 | **Labelled images only** | The 7,481 labelled training images. The 7,518 KITTI test images have no public labels and are not used |
| 2 | **Split by drive, not by image** | Frames from one drive are near-copies, so a random split would leak. 141 drives are assigned whole to one split (shuffle seed 42, greedy balance to about 70/15/15): **5,384 train / 1,044 validation / 1,053 test** images |
| 3 | **Resize** | Every image to 192x640 (area interpolation), stored as RGB uint8. Box coordinates rescaled to match. `DontCare` regions ignored |
| 4 | **Depth maps** | Depth Anything V2 Small on a 518x1722 input, stored at 96x320 as float16 *relative inverse depth* (bigger = nearer) |
| 5 | **Depth to metres** | `depth = k / (disp + 0.001)`, clipped to 1-120 m. One global constant **k = 44.57** fitted as the median of `true distance x disp` over **37,758** labelled objects (3-60 m). Resulting median distance error **17.3%** (best quarter within 8.1%, worst quarter above 32.9%), similar at all distances |
| 6 | **Synthetic fog, generated on the fly** | `I = J*t + A*(1-t)`, `t = exp(-beta*depth)`. Training: beta uniform in [0.01, 0.10], A uniform in [0.8, 1.0], 10% of samples left clear. Evaluation: A = 0.9 and beta = 0.02 (light), 0.05 (medium), 0.08 (dense), i.e. visibility of about 196 m, 78 m and 49 m (`3.912 / beta`). No hazy images are stored |
| 7 | **Augmentation** | Random 256-pixel-wide crops (full height) and horizontal flips, applied to image and depth map together (crop start is even so the half-size depth map stays aligned) |
| 8 | **Detector labels (planned)** | KITTI classes to be grouped into road-user classes for YOLO; mapping to be finalised |

**Limitations of this data:** KITTI is clear-weather, mostly daytime driving around Karlsruhe, so the fog is synthetic and built from *estimated* depth. Dark, textureless areas (for example a shadowed wall) can receive too little fog. Real fog is therefore checked separately (planned, RTTS).

![Fog levels: clear, light, medium, dense](docs/figures/check_haze.png)
![Estimated depth](docs/figures/check_depth.png)

---

## 4. Results so far (baselines only)

Test split, 1,053 images, equal budget of 8,000 steps. PSNR in dB, higher is better.

| Model | Parameters | GMACs @256x256 | Clear | Light | Medium | Dense |
|---|---|---|---|---|---|---|
| Hazy input (no dehazing) | - | - | - | 12.4 | 8.4 | 6.8 |
| AOD-Net | 1.8K | 0.11 | 20.9 | 16.8 | 14.4 | 13.2 |
| ViT (plain) | 2.75M | 5.21 | 44.3 | 20.5 | 17.7 | 15.4 |
| FFA-Net | 4.46M | 287.3 | 61.5 | 23.1 | 19.8 | 17.3 |
| **U-Net** | **1.08M** | **5.20** | 38.4 | **27.0** | **22.0** | **18.8** |

SSIM (light / medium / dense): U-Net 0.941 / 0.879 / 0.822, FFA-Net 0.911 / 0.845 / 0.780, AOD-Net 0.819 / 0.704 / 0.632, ViT 0.786 / 0.671 / 0.598.

**Reading this honestly**

- At this *short, equal* budget the small U-Net beats FFA-Net by 3.9 / 2.2 / 1.5 dB with about 4x fewer parameters and 55x less compute. FFA-Net was still improving at 8,000 steps and is designed for much longer training, so the ranking holds for this budget only.
- Our FFA-Net re-implementation matches the published size (4,455,913 parameters) and compute (287.3 vs 287.8 GMACs).
- **No result for UNetT yet.** Whether the transformer helps is exactly what the next runs will test, against both the plain U-Net and the parameter-matched control.
- **FFA-Net incident:** the first run diverged to NaN near step 1,250 in fp16. It was diagnosed from the log and saved weights, the trainer was rebuilt (non-finite steps skipped, automatic rollback to the last healthy checkpoint, loss-scale logging), and the re-run finished with zero rollbacks. The failed run's numbers were discarded.
- The baseline implementations (`kaggle/fogsafe/models/`) were written for this project **following the original papers**; they are not the authors' code.

---

## 5. Repository structure

```
fogsafe/
|-- src/                  our own model, built step by step (one small demo script per topic)
|   |-- haze_basic.py, haze_depth.py      fog: constant t, then per-pixel t from depth
|   |-- dataset.py                        KITTI dataset (drive-level splits, random crops, flips)
|   |-- blocks.py, unet.py                ConvBlock and the U-Net
|   |-- transformer.py, model.py          transformer block and UNetT (ours)
|   |-- fog_gpu.py, metrics.py            fog on the GPU, PSNR
|   |-- train_ours.py                     training loop (AdamW, warm-up + cosine, fp16, validation)
|   `-- *_demo.py, attention_*.py, ...    small runnable checks for each concept
|-- kaggle/               baselines and data preparation (run on Kaggle)
|   |-- prepare_kitti.py                  zips -> arrays, depth maps, labels, split
|   |-- fogsafe/                          fog, data, metrics, trainer, evaluator, 4 baseline models
|   |-- notebooks/                        01_prepare_data, 02_train_eval
|   `-- tests/smoke_test.py               self-check
|-- docs/                 notes and figures
|-- pyproject.toml, uv.lock, .python-version
`-- README.md
```

---

## 6. Setup and how to run

Requires an NVIDIA GPU for training. Dependencies are managed with [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/Rohitghosh14/fogsafe.git
cd fogsafe
uv sync
```

(`pyproject.toml` points PyTorch to the CUDA 12.8 build on Windows; on other systems adjust the index.)

**Get the data.** Download the three KITTI files from Section 3 into `data/raw/`. Then run `kaggle/notebooks/01_prepare_data.ipynb` on Kaggle (GPU and internet on) and unzip its output into `data/processed/prepared/`. It should contain `images_192x640.npy`, `disp_96x320.npy`, `labels.json`, `meta.json`, `depth_report.json` and a `splits/` folder.

**Run the building blocks** (each prints what it checks):

```bash
uv run python src/haze_depth_demo.py      # fog from depth
uv run python src/dataset_demo.py         # batches, shapes, split sizes
uv run python src/unet_demo.py            # U-Net: 1,084,731 parameters
uv run python src/model_demo.py           # UNetT:  1,976,379 parameters
uv run python src/train_ours.py           # training loop (edit MODEL / STEPS at the top)
```

**Baselines on Kaggle:** upload `kaggle/` as a private dataset and follow `kaggle/README_KAGGLE.md`.

Hardware used: laptop RTX 4050 (6 GB) for our own model, Kaggle (2x Tesla T4) for baselines. Training on the laptop is kept under **77 C** and **5.5 GiB** of GPU memory by a guard (in progress).

---

## 7. Progress checklist

**Setup and data**
- [x] Environment: uv, Python 3.12, PyTorch with CUDA, GPU verified
- [x] YOLOv8n smoke test (3.15M parameters, 8.7 GFLOPs at 640x640)
- [x] KITTI downloaded and verified (7,481 labels, 7,481 train + 7,518 test images, no damaged files)
- [x] Drive-level split (5,384 / 1,044 / 1,053)
- [x] Depth maps and metre calibration (median error 17.3%)
- [x] Prepared dataset on Kaggle and on the laptop

**Fog**
- [x] Atmospheric scattering model (constant t), then per-pixel t from depth
- [x] Fog generated on the GPU, with the same numbers as the Kaggle code

**Baselines (Kaggle)**
- [x] AOD-Net, U-Net, ViT trained and evaluated (8,000 steps)
- [x] FFA-Net: failed run diagnosed, trainer rebuilt, valid re-run
- [ ] Optional: longer-budget re-runs of all baselines (they were still improving)
- [ ] Optional: official pretrained weights (trained on RESIDE) as a separate, clearly labelled row

**Our model (built step by step)**
- [x] Dataset and DataLoader
- [x] ConvBlock, U-Net (matches the baseline: 1,084,731 parameters)
- [x] Attention, transformer block, UNetT (1,976,379 parameters, gradients reach the transformer)
- [x] Training loop: overfit-one-batch test and 300-step test (stable, no skipped steps)
- [ ] GPU guard (pause at 75 C, resume at 70 C, memory cap, refuse to start if the GPU is busy)
- [ ] Train UNetT for 8,000 steps (learning rate 3e-4 vs 5e-4, chosen on validation only)
- [ ] Train the parameter-matched control (U-Net, base=32)
- [ ] Repeat with several seeds, to tell real gains from noise

**Detection and distance**
- [ ] Build the YOLO-format detection dataset (class mapping, fog-augmented training)
- [ ] Fine-tune YOLOv8n on KITTI
- [ ] Test the detector behind every dehazer, ours included (mAP@50 and pedestrian/vehicle recall at clear / light / medium / dense fog)
- [ ] Distance estimation and error against KITTI's true distances

**Joint training**
- [ ] Dehazing head on the detector's features, dropped at inference
- [ ] Compare: raw, dehaze-first, joint training

**Evaluation and write-up**
- [ ] Parameters, FLOPs and FPS for every model on one machine (GPU), plus a CPU-only speed test on an older laptop
- [ ] Real-fog check on RTTS
- [ ] Final comparison table and plots
- [ ] Fill the `docs/` notes and finalise this README (architecture diagram, results)

**Stretch (after the core work; the main goal does not change)**
- [ ] Demo video: boxes, distance and fog level on foggy driving video
- [ ] Rule-based scene relations (for example "pedestrian left of car, 8 m ahead")
- [ ] Radar fusion and driver input (future work)

---

## 8. Known limitations

- Synthetic fog from estimated depth is an approximation, not real fog.
- KITTI is clear-weather daytime driving from one region; generalisation is measured, not assumed.
- PSNR/SSIM are image-quality scores; the detection tests are the decisive measure and are still to come.
- Baselines were compared at one short budget; longer budgets may change the ranking.

---

## 9. References

- A. Geiger, P. Lenz, R. Urtasun. *Are we ready for autonomous driving? The KITTI vision benchmark suite.* CVPR 2012.
- L. Yang et al. *Depth Anything V2.* 2024.
- B. Li et al. *AOD-Net: All-in-One Dehazing Network.* ICCV 2017.
- X. Qin, Z. Wang, Y. Bai, X. Xie, H. Jia. *FFA-Net: Feature Fusion Attention Network for Single Image Dehazing.* AAAI 2020.
- O. Ronneberger, P. Fischer, T. Brox. *U-Net: Convolutional Networks for Biomedical Image Segmentation.* MICCAI 2015.
- A. Dosovitskiy et al. *An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale.* ICLR 2021.
- A. Vaswani et al. *Attention Is All You Need.* NeurIPS 2017.
- Y. Song, Z. He, H. Qian, X. Du. *Vision Transformers for Single Image Dehazing* (DehazeFormer). IEEE TIP 2023.
- Ultralytics YOLOv8. https://github.com/ultralytics/ultralytics

## 10. Author

Rohit (GitHub: [Rohitghosh14](https://github.com/Rohitghosh14)).
Code license: to be added. KITTI and the pretrained models follow their own licences.
