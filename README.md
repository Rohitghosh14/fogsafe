# FogSafe

A compact, fog-robust driving perception model: dehazing + object detection + distance, built for low compute.

## Goal
- Deployed model under about 5M parameters and 30+ FPS
- Compared fairly against AOD-Net, FFA-Net, U-Net and ViT dehazers
- Judged by detection accuracy and pedestrian/vehicle recall at each haze level, plus params, FLOPs and FPS

## Status (work in progress)
- [x] Environment (uv, PyTorch CUDA), KITTI data pipeline, drive-level split
- [x] Synthetic fog from estimated depth (Depth Anything V2 Small, calibrated on KITTI distances)
- [x] Baselines trained on Kaggle: AOD-Net, U-Net, ViT, FFA-Net
- [x] Our model: U-Net with a transformer at the bottleneck (UNetT)
- [ ] Training, detection tests, joint training, final comparison

## Setup
uv sync

Data is not included. The prepared dataset is built from KITTI with the scripts in this repo.
