# Homework 2 — Character-Level Language Modeling (RNN / LSTM / GRU)

**ECGR 4106 – Introduction to Deep Learning**
University of North Carolina at Charlotte, Department of Electrical and Computer Engineering

## Overview

This homework builds character-level language models (next-character prediction) using recurrent
neural networks. It starts from the professor's RNN example (originally a sine-wave regression
problem) and adapts it for text, which is a classification problem over a character vocabulary.

There are two problems:

- **Problem 1** trains and compares a plain RNN, an LSTM, and a GRU on a short essay, across
  sequence lengths of 10, 20, and 30. It compares training loss, validation accuracy, training
  time, and model size, and includes loss curves and generated text samples.
- **Problem 2** trains LSTM and GRU models on the tiny Shakespeare dataset (~1.1M characters)
  using the provided data loader. It compares the two models at sequence lengths 20 and 30, runs
  a hyperparameter sweep (hidden size, number of layers, and the fully-connected head), reports
  perplexity, tests sequence length 50, and generates Shakespeare-style text at several
  temperatures.

## Files

| File | Description |
|---|---|
| `Homework_2.ipynb` | The full Colab notebook with all code and saved outputs |

## How to Run

The notebook is built for Google Colab with a GPU runtime.

1. Open `Homework_2.ipynb` in Google Colab.
2. Set the runtime to GPU (`Runtime → Change runtime type → GPU`).
3. Run `Runtime → Restart and run all`.

The cells are ordered so the notebook runs cleanly top to bottom. Problem 2 downloads the tiny
Shakespeare dataset automatically, so an internet connection is required.

**Note on runtime:** Problem 2 trains on ~1.1M characters, so a full run takes roughly 30–40
minutes on a GPU. The notebook is committed with its outputs saved, so all results can be viewed
without re-running.

## Key Settings

- **Optimizer:** Adam (with gradient clipping at max norm 1.0)
- **Loss:** Cross-entropy
- **Input encoding:** one-hot in Problem 1, learned embedding in Problem 2
- **Random seed:** 42 (set for reproducibility, including the train/test split in Problem 2)

A note on reproducibility: a fixed seed is used throughout, but some GPU operations are not fully
deterministic, so accuracy values (especially at the longer sequence lengths) can vary slightly
between runs. The report discusses this where it matters.

## Dependencies

All libraries are pre-installed in Google Colab:

- PyTorch
- NumPy
- Matplotlib
- Requests (used to download the tiny Shakespeare dataset)
