# Homework 3 — Sequence-to-Sequence Machine Translation

**ECGR 4106 — Intro to Deep Learning**

GRU encoder–decoder models for English↔French translation, built from scratch in PyTorch. The project establishes a baseline seq2seq model, adds Luong dot-product attention, and evaluates both architectures in both translation directions.

---

## Overview

| Problem | Task | Architecture |
|---|---|---|
| 1 | English → French | Baseline GRU encoder–decoder |
| 2 | English → French | GRU encoder–decoder + Luong attention |
| 3 | French → English | Both architectures, reversed direction |

All four runs share a single fixed 80/20 train/validation split and are evaluated with two metrics: traditional sequence (exact-match) accuracy and validation BLEU-4.

## Dataset

`vast_english_french.txt` — 555 tab-separated English–French sentence pairs.

- Shuffled with a fixed seed (`42`) and split 80/20 into **444 training** and **111 validation** pairs. The same split is reused unchanged across all three problems.
- Two separate vocabularies (English: 881 words, French: 970 words) are built **from the training split only**, each reserving `<PAD>`, `<SOS>`, `<EOS>`, and `<UNK>` tokens.
- Because the vocabulary is large relative to the corpus, ~20% of validation word tokens are unseen in training (mapped to `<UNK>`), and ~68% of validation sentences contain at least one unknown word.

## Model

- **Encoder:** embedding (256) → single-layer GRU (256). The final hidden state is the context vector.
- **Baseline decoder:** embedding (256) → single-layer GRU (256) → linear projection to the target vocabulary, initialized from the encoder's final hidden state.
- **Attention decoder (Luong dot-product):** retains *all* encoder hidden states; at each step scores them against the decoder state via dot product, softmaxes into attention weights, forms a weighted context vector, and fuses it with the GRU output (`tanh` of a linear layer) before projecting to the target vocabulary. No fixed maximum source length.

## Training

| Setting | Value |
|---|---|
| Loss | Cross-entropy (per target token) |
| Optimizer | Adam, learning rate 0.001 |
| Teacher-forcing ratio | 0.5 |
| Epochs | 40 |
| Batch size | 1 (per-sentence) |
| Decoding (eval) | Greedy, stop at `<EOS>`, length cap 20 |

## Results

| # | Direction | Architecture | Exact-match | BLEU-4 |
|---|---|---|---:|---:|
| P1 | EN → FR | Baseline GRU | 0.00% | 0.1262 |
| P2 | EN → FR | + Attention | 0.90% | 0.1479 |
| P3a | FR → EN | Baseline GRU | 0.00% | 0.1267 |
| P3b | FR → EN | + Attention | 0.00% | 0.1267 |

Attention improved both metrics in the EN→FR direction. All four runs show clear overfitting (training loss approaches zero while validation loss minimizes near epoch 5 and rises after), driven by the small, lexically sparse dataset. See the report for full loss curves, attention heatmaps, qualitative samples, and the direction synthesis.

## Repository contents

- `Homework_3.ipynb` — full notebook: data pipeline, both architectures, training, evaluation, and figures.
- `vast_english_french.txt` — dataset.
- `Homework_3.pdf` — written report.

## How to run

The notebook is built for Google Colab.

1. Open `Homework_3.ipynb` in Colab.
2. Make `vast_english_french.txt` available and point `DATA_PATH` at it (upload to `/content/`, or mount Google Drive).
3. Runtime → Run all. A GPU runtime is recommended but not required.

**Dependencies:** `torch`, `nltk`, `matplotlib`, `numpy` (all preinstalled in Colab).
