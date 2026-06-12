# ECGR 4106 – Homework 1: CNN Architectures on CIFAR-10

**Name:** Anthony Kang  
**Student ID:** 801440598

## Overview
This repository implements and compares four CNN architectures on CIFAR-10:
- **Problem 1:** Modified AlexNet (adapted for 32×32 inputs) + dropout study
- **Problem 2:** Adapted VGG-11 (width-scaled to match AlexNet's parameter budget) + dropout study
- **Problem 3:** ResNet-11 vs. ResNet-18 (implemented from scratch) + dropout study

The full written analysis is in `Homework_1_Report.pdf`.

## How to Run
1. Open `Homework1.ipynb` in Google Colab.
2. Set the runtime to GPU: `Runtime → Change runtime type → GPU`.
3. Run all cells top to bottom: `Runtime → Run all`.

The notebook downloads CIFAR-10 automatically (via torchvision) on first run. 
No manual data setup is required.

## Notes
- All experiments use a fixed seed (42), an identical train/val/test split, and the 
  same data augmentation and normalization across all three problems, as required.
- ResNet models train for 50 epochs; AlexNet and VGG train for 30. Total runtime is 
  roughly [X] minutes on a GPU.
- Results (accuracy, confusion matrices, training curves) are produced inline as each 
  section runs.

## Files
- `Homework1.ipynb` — all code for Problems 1–3
- `Homework_1_Report.pdf` — written report with analysis and figures
