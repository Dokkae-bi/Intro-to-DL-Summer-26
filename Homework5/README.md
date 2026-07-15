# Homework 5 — Vision Transformers for Image Classification (CIFAR-100)

ECGR 4106 — Intro to Deep Learning

## Overview

Two experiments on CIFAR-100 (100 classes, 32×32 RGB):

**Problem 1 — ViT from scratch vs. ResNet-18.** A Vision Transformer implemented
from scratch (Conv2d patch embedding, [CLS] token, learned positional embeddings,
pre-LayerNorm encoder blocks with manually implemented multi-head self-attention,
4× GELU MLP). Five configurations swept ablation-style (patch size, embedding dim,
depth, heads — one axis at a time) against an ImageNet-pretrained ResNet-18, all
with the mandated recipe (batch 64, 10 epochs, Adam lr=0.001).

**Problem 2 — Pretrained Swin fine-tuning vs. scratch Swin.** Swin-Tiny and
Swin-Small (Hugging Face checkpoints, frozen backbone, head-only fine-tuning,
5 epochs, batch 32, lr=2e-5) compared against a from-scratch Swin adapted for
32×32 inputs (3 stages, window 4, shifted-window attention with relative position
bias, patch merging) trained at the mandated lr=0.001 with gradient clipping.

## Methodology

- Class-balanced 45k/5k train/validation split (450/50 per class, seed 42),
  identical indices reused by every model in both problems
- Best-validation checkpointing; checkpoint restored before a single test
  evaluation per model — all headline numbers are restored-checkpoint test accuracy
- Training time measures the optimization loop only (CUDA-synchronized);
  validation/test run outside the timed window
- FLOPs via `torch.utils.flop_counter` (forward pass, batch 1, 1 MAC = 2 FLOPs)
- No augmentation; regularization by regime (dropout 0.1 on scratch transformers,
  gradient clipping on scratch Swin, none added to the ~77K-param linear probes)

## Files

| File | Description |
|---|---|
| `Homework_5_final.ipynb` | Executed notebook (all results, tables, and figures) |
| `vit_model.py` | Standalone ViT implementation (self-test: `python vit_model.py`) |
| `swin_model.py` | Standalone scratch Swin implementation (self-test included) |
| `results/*.json` | Per-run metrics: P1/P2 main results + supplementary runs |

## Headline results (single clean Restart-and-Run-All, Tesla T4)

| Model | Test acc |
|---|---|
| Best scratch ViT (patch 4 / embed 256 / 4 blocks / 8 heads) | 30.01% |
| ResNet-18 (ImageNet-pretrained, fine-tuned) | 49.83% |
| Swin-Tiny (pretrained, head-only) | 65.51% |
| Swin-Small (pretrained, head-only) | 69.78% |
| Scratch Swin (mandated lr=0.001) | 4.89% |
| Scratch Swin (supplementary lr=3e-4) | 28.67% |

Recurring finding: Adam at lr=0.001 without warmup destabilized every
sufficiently wide/deep scratch transformer (ViT embed-512 and the scratch Swin);
the identical architectures train normally at lr=3e-4.

## Running

Open the notebook in Google Colab with a GPU runtime and Restart-and-Run-All
(~2–3 h on a T4). CPU fallback is included but impractically slow for training.
Dependencies (`transformers`) install in-notebook; PyTorch/Torchvision use
Colab's preinstalled versions. Results were produced with PyTorch 2.11.0,
Torchvision 0.26.0, Transformers 5.12.1. The notebook caches the CIFAR-100
archive to Google Drive when available and falls back to `./results` for JSON
output when Drive is not mounted.
