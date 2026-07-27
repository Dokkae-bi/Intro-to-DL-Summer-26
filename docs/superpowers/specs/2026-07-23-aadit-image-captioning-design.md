# Aadit Half — Image Captioning Harness v2.1 Design

**Date:** 2026-07-23  
**Course:** ECGR 4106 Final Project  
**Team:** Anthony & Aadit  
**Status:** Approved for planning (Approach 1 — full protocol surface, Colab training deferred)

## Goal

Complete Aadit’s half of the final project by extending Anthony’s shared Colab harness so all attention / Transformer decoders and analysis deliverables are implemented and runnable. GPU training remains on Colab (not executed in this coding session).

## Source of truth

- Harness: `Final Project/captioning_harness_v2_1 (2).ipynb`
- Proposal: `Final Project/Image_Captioning_Proposal.docx`
- Anthony’s baseline notebook (reference only): `Final Project/Final_Project_image_captioning.ipynb`

## Non-goals

- Running full Flickr8k training / LR search in this session
- Fine-tuning the ResNet encoder (first item in harness cut order)
- Freezing `FROZEN_manifest.json` or test evaluation before all five architectures finish
- Written report prose and Canvas submission (Aadit handles Canvas separately)
- Changing shared `CFG` hyperparameters that must stay identical across models

## Architecture contract (unchanged)

Every registered model must implement:

```
forward(feats, caps)                        -> logits [B, T-1, V]
init_state(feats)                           -> state             feats is [B, 49, 2048]
step(state, tokens, return_attention=False) -> (logits [B, V], state)
                                            -> (logits [B, V], state, attn [B, H, 49]) if True
```

Rules:

- Attention shape is always `[B, heads, 49]`. Single-head models return `[B, 1, 49]`.
- No-attention models return `None` for attention (baseline already does).
- Declare `MODEL_CONFIGS[arch]` and `DECODE_POLICY[arch]` for every architecture.
- Do not modify Anthony’s training loop, data definition, feature cache binding, or protocol driver semantics.
- Keep `EXPERIMENT_VERSION = "v2.1"` unless training or model serialization semantics change after existing valid runs; filling empty registry slots does not require a bump.

## Models to implement

| Architecture | Attention | Decode policy string |
|---|---|---|
| `gru_luong` | Luong dot-product over 49 spatial features | `incremental (recurrent state reuse)` |
| `gru_bahdanau` | Bahdanau additive attention | `incremental (recurrent state reuse)` |
| `gru_multihead` | Lightweight multi-head attention over spatial features | `incremental (recurrent state reuse)` |
| `transformer` | Cross-attention from causal decoder to image memory | `prefix-recompute (non-incremental)` |

### Shared decoder patterns (GRU attention family)

- Project spatial features to `hidden_dim` once and cache the projection in decode state.
- Initialize GRU hidden state from mean-pooled features (same spirit as baseline).
- **Pinned forward/step convention (Show, Attend and Tell style):** at each timestep, use the current GRU hidden state as the attention query over projected feats → context vector → concatenate `[embedding; context]` → GRU → logits from the new hidden state (optionally with a deep output layer). Teacher-forced `forward` and free-running `step` must use the same ordering.
- `init_state` retains projected feats (and initial `h`) so `step` can attend every decode step.
- `HAS_ATTENTION = True` for the three GRU attention models and Transformer.

### Transformer decoder

- Treat encoder features as memory (`[B, 49, D]` after projection).
- Causal self-attention over caption tokens + cross-attention to image memory.
- Depth/heads chosen to stay in the same parameter ballpark as GRU variants where practical; record exact `MODEL_CONFIGS`.
- `step` recomputes the growing prefix (sequences ≤ 30). Exact policy string: `prefix-recompute (non-incremental)`.

### Config defaults (aligned with baseline unless noted)

- `dropout`: 0.3
- `gru_luong` / `gru_bahdanau`: `heads: 1`
- `gru_multihead`: `heads: 4`
- Transformer: `layers: 2`, `heads: 4`, `ff_dim: 1024`, `dropout: 0.3` (adjust only if parameter count is wildly off baseline; document any change in `MODEL_CONFIGS`)

## Analysis & metrics

### METEOR and CIDEr

- Add `evaluate_extra_metrics(model, img_loader)` that returns METEOR and CIDEr on the same hyp/ref pairing style as BLEU evaluation.
- Prefer dependency-light implementations so Colab stays one-click:
  - METEOR: NLTK (download required corpora once in a setup cell).
  - CIDEr: self-contained TF-IDF n-gram scorer in the notebook (avoid hard dependency on `pycocoevalcap` unless install is trivial and pinned).
- Primary model selection remains validation BLEU-4 (Anthony’s protocol). Extra metrics are reported alongside, not used to pick checkpoints unless jointly agreed later.

### Beam search

- `beam_decode_batch(model, feats, beam_size=3)` built on `step`.
- Default beam size 3; beam 5 is optional polish per harness cut order.
- Development comparisons on **validation** only. Optional beam numbers on test only after manifest freeze.

### Attention heatmaps

- `show_attention_heatmaps(arch, lr, seed, ...)` using greedy decode with `collect_attention=True`.
- Reshape spatial weights to 7×7, upsample to image size, overlay per generated token.
- Skip models with `HAS_ATTENTION = False`.

### Failure analysis

- Scan validation predictions for:
  - repeated tokens / stuttering
  - extremely short or generic captions
  - high lexical novelty vs references (hallucination heuristic)
- Emit a small summary table plus a handful of annotated examples suitable for the report.

## Notebook layout

All changes live in `Final Project/captioning_harness_v2_1 (2).ipynb`:

1. Replace the Aadit template in the model registry cell with the four registered classes + configs + decode policies.
2. Add markdown + code cells for METEOR/CIDEr helpers.
3. Add markdown + code cells for beam search.
4. Add markdown + code cells for attention heatmaps.
5. Add markdown + code cells for failure analysis.
6. Add Aadit run cells calling `run_protocol` for each of the four architectures, with smoke-test guidance (`CFG["max_epochs"] = 1` in a scratch copy first).
7. Leave Anthony’s freeze / test / baseline protocol cells behaviorally intact.

## Verification (pre-GPU)

- Instantiation + one `forward` shape check per architecture.
- One `step(..., return_attention=True)` check: attention is `None` or `[B, H, 49]`.
- Documented 1-epoch Colab smoke test (not run in this coding session).
- Heatmap path exercised only on attention models after a checkpoint exists.

## Cut order (if Colab time is tight)

1. Skip fine-tuning ablation (already out of scope).
2. Drop beam size 5 (keep beam 3 or greedy-only if needed).
3. Narrow LR search only if necessary (prefer shared LR over dropping an architecture).
4. Reduce qualitative sample count before dropping an architecture.
5. Drop an architecture only as last resort and revise stated scope / `EXPECTED_ARCHS` accordingly.
6. Cut seeds last.

## Success criteria

- Four architectures registered and callable via `run_protocol`.
- Attention contract satisfied for heatmap code.
- METEOR, CIDEr, beam (size 3), heatmaps, and failure-analysis cells present and documented.
- Notebook remains Colab-runnable with only `PROJECT_ROOT` needing per-account edits.
- No changes to the shared data definition, feature-cache hash binding, or manifest/test gating logic.
