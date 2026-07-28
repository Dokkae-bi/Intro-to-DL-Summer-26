# Image Captioning — Experiment Results (v2.1)
Compiled from Colab Drive exports (`image_captioning_project` downloads).
- Frozen at: `2026-07-28 03:34:30`
- Experiment version: `v2.1`
- Split source: `karpathy`
- Selection rule: max validation BLEU-4 (greedy decoding)
- BLEU impl: `nltk.corpus_bleu/BP-applied/no-smoothing`

## Validation BLEU-4 (final seeds, greedy)

| Architecture | LR | Params | Seed1 | Seed2 | Seed3 | Mean | Std |
|---|---:|---:|---:|---:|---:|---:|---:|
| gru_bahdanau | 0.0003 | 5,759,725 | 0.1958 | 0.1959 | 0.1976 | **0.1964** | 0.0010 |
| gru_baseline | 0.0003 | 4,185,837 | 0.1822 | 0.1952 | 0.1843 | **0.1872** | 0.0070 |
| gru_luong | 0.0003 | 5,234,925 | 0.1704 | 0.1716 | 0.1689 | **0.1703** | 0.0013 |
| gru_multihead | 0.0003 | 6,285,549 | 0.1917 | 0.1881 | 0.1970 | **0.1923** | 0.0045 |
| transformer | 0.0003 | 10,028,013 | 0.1887 | 0.1900 | 0.1904 | **0.1897** | 0.0009 |

## Test BLEU (manifest-gated, greedy)

| Architecture | Seed1 | Seed2 | Seed3 | Mean BLEU-4 | Std | Mean ms/image |
|---|---:|---:|---:|---:|---:|---:|
| gru_bahdanau | 0.1979 | 0.2093 | 0.2197 | **0.2089** | 0.0109 | 0.5 |
| gru_baseline | 0.1845 | 0.2008 | 0.1877 | **0.1910** | 0.0086 | 0.4 |
| gru_luong | 0.1747 | 0.1793 | 0.1739 | **0.1760** | 0.0029 | 0.5 |
| gru_multihead | 0.1917 | 0.1870 | 0.2052 | **0.1946** | 0.0095 | 0.6 |
| transformer | 0.1959 | 0.2017 | 0.2037 | **0.2004** | 0.0041 | 1.2 |

### Full test BLEU-1…4 (mean over seeds)

| Architecture | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 |
|---|---:|---:|---:|---:|
| gru_bahdanau | 0.6236 | 0.4484 | 0.3085 | 0.2089 |
| gru_baseline | 0.6016 | 0.4233 | 0.2858 | 0.1910 |
| gru_luong | 0.5822 | 0.4026 | 0.2678 | 0.1760 |
| gru_multihead | 0.6021 | 0.4256 | 0.2902 | 0.1946 |
| transformer | 0.6058 | 0.4296 | 0.2957 | 0.2004 |

## METEOR / CIDEr

CIDEr here is the notebook’s TF-IDF cosine variant (**not** official CIDEr-D). Report it as CIDEr-TFIDF.

### Validation (seed 1 checkpoints)

| Architecture | METEOR | CIDEr-TFIDF |
|---|---:|---:|
| gru_bahdanau | 0.4042 | 0.0399 |
| gru_baseline | 0.3811 | 0.0345 |
| gru_luong | 0.3771 | 0.0338 |
| gru_multihead | 0.3851 | 0.0376 |
| transformer | 0.3955 | 0.0371 |

### Test (mean over seeds)

| Architecture | METEOR | CIDEr-TFIDF |
|---|---:|---:|
| gru_bahdanau | 0.4128 | 0.0416 |
| gru_baseline | 0.4072 | 0.0386 |
| gru_luong | 0.3839 | 0.0343 |
| gru_multihead | 0.3998 | 0.0385 |
| transformer | 0.4065 | 0.0396 |

## Decode policies

| Architecture | Decode policy |
|---|---|
| gru_bahdanau | incremental (recurrent state reuse) |
| gru_baseline | incremental (recurrent state reuse) |
| gru_luong | incremental (recurrent state reuse) |
| gru_multihead | incremental (recurrent state reuse) |
| transformer | prefix-recompute (non-incremental) |

## Files in this folder

- `FROZEN_manifest.json` — locked experiment definition
- `FINAL_test_results.json` — per-seed test BLEU + latency
- `FINAL_val_meteor_cider.json` / `FINAL_test_meteor_cider.json`
- `results/*.json` — available per-run training logs (partial download; 7/15 finals present locally)
- `checkpoints_inventory/checkpoint_locations.json` — which `.pt` files existed across Drive downloads

## Not committed (by design)

- Model checkpoints (`*.pt`, ~50–120 MB each; Transformer exceeds GitHub’s 100 MB limit)
- `resnet50_feats_7x7_fp16.pt` (~1.6 GB)
- Flickr8k raw images
- Keep those on Google Drive: `/content/drive/MyDrive/image_captioning_project`

## Local download checkpoint coverage (manifest finals)

| Checkpoint | Status in local downloads |
|---|---|
| `v2.1_gru_bahdanau_lr0.0003_seed1_final_best.pt` | present_in_downloads |
| `v2.1_gru_bahdanau_lr0.0003_seed2_final_best.pt` | present_in_downloads |
| `v2.1_gru_bahdanau_lr0.0003_seed3_final_best.pt` | present_in_downloads |
| `v2.1_gru_baseline_lr0.0003_seed1_final_best.pt` | missing_from_all_local_downloads |
| `v2.1_gru_baseline_lr0.0003_seed2_final_best.pt` | present_in_downloads |
| `v2.1_gru_baseline_lr0.0003_seed3_final_best.pt` | present_in_downloads |
| `v2.1_gru_luong_lr0.0003_seed1_final_best.pt` | missing_from_all_local_downloads |
| `v2.1_gru_luong_lr0.0003_seed2_final_best.pt` | missing_from_all_local_downloads |
| `v2.1_gru_luong_lr0.0003_seed3_final_best.pt` | present_in_downloads |
| `v2.1_gru_multihead_lr0.0003_seed1_final_best.pt` | missing_from_all_local_downloads |
| `v2.1_gru_multihead_lr0.0003_seed2_final_best.pt` | present_in_downloads |
| `v2.1_gru_multihead_lr0.0003_seed3_final_best.pt` | missing_from_all_local_downloads |
| `v2.1_transformer_lr0.0003_seed1_final_best.pt` | missing_from_all_local_downloads |
| `v2.1_transformer_lr0.0003_seed2_final_best.pt` | present_in_downloads |
| `v2.1_transformer_lr0.0003_seed3_final_best.pt` | present_in_downloads |
