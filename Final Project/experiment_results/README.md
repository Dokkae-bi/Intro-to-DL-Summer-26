# Experiment results (ECGR 4106 — Image Captioning v2.1)

Organized export of the Colab / Drive experiment after freezing the manifest and running test evaluation.

## Quick view

See **[RESULTS_SUMMARY.md](./RESULTS_SUMMARY.md)** for validation/test tables (BLEU, METEOR, CIDEr-TFIDF).

## Source downloads merged

Three local Drive zip/folder exports were reconciled:

1. `image_captioning_project` — primary (manifest + FINAL_* JSONs + partial `results/` + some checkpoints)
2. `image_captioning_project 2` — duplicate of (1) for the key JSON artifacts
3. `image_captioning_project 3` — additional checkpoints only (no `results/` JSONs)

JSON deliverables were taken from download (1)/(2) (byte-identical). Checkpoint binaries were **not** copied into git; see `checkpoints_inventory/`.

## Reproducing figures / reload

Use the shared Drive project root in the harness notebook. Checkpoints remain there; this folder holds the numbers for the report and GitHub.
