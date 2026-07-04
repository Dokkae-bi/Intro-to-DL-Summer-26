# Homework 4 — Transformers for Character Prediction and Machine Translation

## Overview

This homework replaces the RNN-based models from Homeworks 2 and 3 with
transformer architectures and compares them across four problems:

- **Problem 1** — Character-level transformer on the HW2 essay text (sequence
  lengths 10/20/30), compared against the HW2 RNN/LSTM/GRU results.
- **Problem 2** — Transformer on tiny Shakespeare: 2-block/2-head baseline at
  sequence lengths 20 and 30, an 8-configuration architecture sweep
  (1/2/4 blocks × 2/4/8 heads), a hidden-size (d_model) experiment,
  inference-time measurement, a sequence-length-50 run, and generation-quality
  comparison. Compared against the HW2 LSTM/GRU results.
- **Problem 3** — Transformer encoder-decoder for English→French translation
  on vast_english_french.txt, sweeping the same 8 configurations and reporting
  training loss, validation loss, exact-match accuracy, and BLEU-4, with
  qualitative examples. Compared against the HW3 GRU baseline and
  GRU + Luong attention models.
- **Problem 4** — Same as Problem 3 with the direction reversed
  (French→English), plus a concluding comparison of which direction the
  transformer optimized more easily.

## Files

- `Homework_4.ipynb` — the complete notebook, all four problems, with saved
  outputs from a clean top-to-bottom run (Restart and Run All).
- `Homework_4_Report.pdf` — the full report with results tables, loss curves,
  qualitative examples, and analysis.

## Setup / How to Run

The notebook was developed and run in Google Colab on a GPU runtime.

1. Problems 1–2 need no external files (the essay text is embedded and tiny
   Shakespeare is downloaded in-notebook).
2. Problems 3–4 require `vast_english_french.txt` in Google Drive. The
   notebook mounts Drive and reads from:
   `/content/drive/MyDrive/vast_english_french.txt`
   (adjust `DATA_PATH` in the P3 data cell if your copy lives elsewhere).
3. Run all cells top to bottom. Every cell is self-contained with respect to
   earlier cells only, so Restart and Run All executes cleanly.

## Key Settings

- Seed 42 everywhere, matching HW2/HW3. Problems 3–4 reproduce the exact
  444/111 train/validation split from HW3 (`random.seed(42)` before the
  shuffle).
- All transformers use d_model = 128, pre-norm blocks (`norm_first=True`),
  Adam at lr = 0.001, and gradient clipping at 1.0. Dropout is 0.0 for the
  Problem 1 memorization task and 0.1 for Problems 2–4.
- Problems 3–4 train with padded batches of 32, full teacher forcing via a
  causal decoder mask, and `ignore_index=PAD` in the loss. Evaluation is
  free-running greedy decoding with BLEU-4 (NLTK, method-1 smoothing),
  identical to HW3.

## Headline Results

| Problem | Best result |
|---|---|
| P1 (essay, memorization) | 0.935 full-text accuracy (seq 20) vs. 0.996 HW2 LSTM |
| P2 (tiny Shakespeare) | 0.5100 test acc / 5.092 perplexity (4 blocks × 8 heads) vs. 0.5750 HW2 wide LSTM |
| P3 (EN→FR) | BLEU-4 0.1513 (4 blocks × 2 heads) vs. 0.1479 HW3 attention GRU |
| P4 (FR→EN) | BLEU-4 0.1632 (4 blocks × 4 heads) vs. 0.1267 HW3 GRUs — best run in either homework |

The full analysis, comparison tables, and loss curves are in the report.
