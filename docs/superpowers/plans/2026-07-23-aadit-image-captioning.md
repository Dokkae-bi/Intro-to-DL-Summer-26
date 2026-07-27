# Aadit Image Captioning Harness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend `Final Project/captioning_harness_v2_1 (2).ipynb` with Luong, Bahdanau, multi-head, and Transformer decoders plus METEOR/CIDEr, beam search, heatmaps, failure analysis, and ready-to-run `run_protocol` cells.

**Architecture:** Keep Anthony’s harness contract (`forward` / `init_state` / `step` + attention `[B, H, 49]`). Register four new models beside `gru_baseline`, then append analysis and Aadit run cells without changing shared `CFG`, data hashes, or protocol drivers.

**Tech Stack:** PyTorch, torchvision, NLTK (BLEU/METEOR), NumPy, Matplotlib, Google Colab notebook

**Spec:** `docs/superpowers/specs/2026-07-23-aadit-image-captioning-design.md`

## Global Constraints

- Edit only `Final Project/captioning_harness_v2_1 (2).ipynb` for deliverable code (plus this plan/spec already written).
- Do not change shared `CFG` training hyperparameters (`emb_dim`, `hidden_dim`, `batch_size`, `patience`, `max_epochs`, `grad_clip`, etc.).
- Keep `EXPERIMENT_VERSION = "v2.1"` unless serialization/training semantics change.
- Attention return shape must be `[B, heads, 49]` or `None`.
- Primary selection metric remains validation BLEU-4; METEOR/CIDEr are report-only.
- Do not freeze the manifest or run full GPU training in this implementation session.
- Do not commit unless the user explicitly asks.
- Fine-tuning ablation stays unimplemented (section 13 stub remains).

## File Structure

| Path | Responsibility |
|---|---|
| `Final Project/captioning_harness_v2_1 (2).ipynb` cell 12 | Model registry: baseline + four new models, configs, decode policies |
| Same notebook — new cells after cell 14 (or before section 8) | Optional: keep metrics/beam near harness decode helpers; prefer inserting Aadit analysis after qualitative samples (after cell 24) to minimize churn to Anthony’s numbered sections |
| Same notebook — new cells before handoff (before cell 31) | Aadit `run_protocol` cells + smoke-test guidance |
| `docs/superpowers/specs/2026-07-23-aadit-image-captioning-design.md` | Approved design (read-only during implementation) |

**Insertion strategy (locked):**

1. Replace template in **cell 12** with full model implementations.
2. Insert new sections **after cell 24** (`show_samples`) and **before cell 25** (fine-tune ablation): METEOR/CIDEr, beam, heatmaps, failure analysis.
3. Insert Aadit run cells **after cell 30** (`summary_table`) and **before cell 31** (handoff), and update handoff markdown to mark models/analysis as implemented.

---

### Task 1: Register GRU Luong / Bahdanau / Multi-head models

**Files:**
- Modify: `Final Project/captioning_harness_v2_1 (2).ipynb` cell 12

**Interfaces:**
- Consumes: `CFG`, `VOCAB_SIZE`, `PAD_IDX`, `MODEL_CONFIGS`, `DECODE_POLICY`, `register_model`, `nn`, `torch`
- Produces: `MODEL_REGISTRY["gru_luong"|"gru_bahdanau"|"gru_multihead"]` classes with `HAS_ATTENTION=True`, `forward`, `init_state`, `step`

- [ ] **Step 1: Add MODEL_CONFIGS and DECODE_POLICY entries**

In cell 12 (and mirror any needed keys into the early `MODEL_CONFIGS` dict in cell 2 — cell 2 currently only has `gru_baseline`; either extend cell 2’s dict or have cell 12 assign into `MODEL_CONFIGS` before registration). Prefer assigning in cell 12 immediately before each `@register_model` so cell 2 stays Anthony-minimal:

```python
MODEL_CONFIGS["gru_luong"] = {"dropout": 0.3, "attention": "luong_dot", "heads": 1}
DECODE_POLICY["gru_luong"] = "incremental (recurrent state reuse)"

MODEL_CONFIGS["gru_bahdanau"] = {"dropout": 0.3, "attention": "bahdanau", "heads": 1}
DECODE_POLICY["gru_bahdanau"] = "incremental (recurrent state reuse)"

MODEL_CONFIGS["gru_multihead"] = {"dropout": 0.3, "attention": "multihead", "heads": 4}
DECODE_POLICY["gru_multihead"] = "incremental (recurrent state reuse)"
```

- [ ] **Step 2: Implement shared attention helpers + three GRU classes**

Replace the commented Aadit template with this implementation (Show-Attend-Tell ordering: attend with current `h`, concat emb+context into GRU):

```python
def _project_feats(feat_proj, feats):
    # feats: [B, 49, 2048] -> [B, 49, H]
    return feat_proj(feats)

class _GRUAttnBase(nn.Module):
    HAS_ATTENTION = True

    def _build_common(self, dropout):
        H, E, F = CFG["hidden_dim"], CFG["emb_dim"], CFG["feature_dim"]
        self.feat_proj = nn.Linear(F, H)
        self.init_h = nn.Linear(H, H)
        self.emb = nn.Embedding(VOCAB_SIZE, E, padding_idx=PAD_IDX)
        self.gru = nn.GRU(E + H, H, batch_first=True)
        self.fc = nn.Linear(H, VOCAB_SIZE)
        self.drop = nn.Dropout(dropout)

    def _h0(self, proj):
        return torch.tanh(self.init_h(proj.mean(dim=1))).unsqueeze(0)  # [1,B,H]

    def init_state(self, feats):
        proj = _project_feats(self.feat_proj, feats)
        return {"h": self._h0(proj), "proj": proj}

    def _attend(self, h, proj):
        raise NotImplementedError

    def forward(self, feats, caps):
        st = self.init_state(feats)
        emb = self.drop(self.emb(caps[:, :-1]))  # [B,T-1,E]
        outs = []
        h = st["h"]
        for t in range(emb.size(1)):
            ctx, _ = self._attend(h.squeeze(0), st["proj"])  # [B,H]
            inp = torch.cat([emb[:, t], ctx], dim=-1).unsqueeze(1)
            out, h = self.gru(inp, h)
            outs.append(self.fc(self.drop(out.squeeze(1))))
        return torch.stack(outs, dim=1)

    def step(self, state, tokens, return_attention=False):
        emb = self.emb(tokens)  # [B,E]
        ctx, attn = self._attend(state["h"].squeeze(0), state["proj"])
        inp = torch.cat([emb, ctx], dim=-1).unsqueeze(1)
        out, h = self.gru(inp, state["h"])
        state = {"h": h, "proj": state["proj"]}
        logits = self.fc(out.squeeze(1))
        return (logits, state, attn) if return_attention else (logits, state)

@register_model("gru_luong")
class GRULuong(_GRUAttnBase):
    def __init__(self):
        super().__init__()
        c = MODEL_CONFIGS["gru_luong"]
        self._build_common(c["dropout"])

    def _attend(self, h, proj):
        # Luong dot: score = h·proj
        # h [B,H], proj [B,49,H]
        scores = torch.einsum("bh,bnh->bn", h, proj)  # [B,49]
        attn = torch.softmax(scores, dim=-1)
        ctx = torch.einsum("bn,bnh->bh", attn, proj)
        return ctx, attn.unsqueeze(1)  # [B,1,49]

@register_model("gru_bahdanau")
class GRUBahdanau(_GRUAttnBase):
    def __init__(self):
        super().__init__()
        c = MODEL_CONFIGS["gru_bahdanau"]
        self._build_common(c["dropout"])
        H = CFG["hidden_dim"]
        self.attn_W = nn.Linear(H, H, bias=False)
        self.attn_U = nn.Linear(H, H, bias=False)
        self.attn_v = nn.Linear(H, 1, bias=False)

    def _attend(self, h, proj):
        # scores = v^T tanh(W h + U proj)
        e = torch.tanh(self.attn_W(h).unsqueeze(1) + self.attn_U(proj))  # [B,49,H]
        scores = self.attn_v(e).squeeze(-1)  # [B,49]
        attn = torch.softmax(scores, dim=-1)
        ctx = torch.einsum("bn,bnh->bh", attn, proj)
        return ctx, attn.unsqueeze(1)

@register_model("gru_multihead")
class GRUMultiHead(_GRUAttnBase):
    def __init__(self):
        super().__init__()
        c = MODEL_CONFIGS["gru_multihead"]
        self._build_common(c["dropout"])
        self.heads = c["heads"]
        H = CFG["hidden_dim"]
        assert H % self.heads == 0
        self.head_dim = H // self.heads
        self.q_proj = nn.Linear(H, H)
        self.k_proj = nn.Linear(H, H)
        self.v_proj = nn.Linear(H, H)
        self.out_proj = nn.Linear(H, H)

    def _attend(self, h, proj):
        B, N, _ = proj.shape
        Hh, Hd = self.heads, self.head_dim
        q = self.q_proj(h).view(B, Hh, Hd)                    # [B,H,Hd]
        k = self.k_proj(proj).view(B, N, Hh, Hd).transpose(1, 2)  # [B,H,N,Hd]
        v = self.v_proj(proj).view(B, N, Hh, Hd).transpose(1, 2)
        scores = torch.einsum("bhd,bhnd->bhn", q, k) / math.sqrt(Hd)
        attn = torch.softmax(scores, dim=-1)                  # [B,H,N]
        ctx = torch.einsum("bhn,bhnd->bhd", attn, v).reshape(B, Hh * Hd)
        ctx = self.out_proj(ctx)
        return ctx, attn  # [B, heads, 49]
```

- [ ] **Step 3: Verify registration string**

At end of cell 12, ensure:

```python
print("registered:", sorted(MODEL_REGISTRY))
```

Expected after Task 2 as well: all five names. After this task alone, at least `gru_baseline` + three GRU attention models.

- [ ] **Step 4: Local shape smoke (optional offline script)**

If GPU/Colab unavailable, run a tiny CPU check by pasting the classes into a throwaway script with fake `CFG`/`VOCAB_SIZE` — or defer shape checks to Task 3 cell in the notebook. Do not commit unless asked.

---

### Task 2: Register Transformer decoder

**Files:**
- Modify: `Final Project/captioning_harness_v2_1 (2).ipynb` cell 12

**Interfaces:**
- Consumes: same globals as Task 1
- Produces: `MODEL_REGISTRY["transformer"]` with `DECODE_POLICY["transformer"] = "prefix-recompute (non-incremental)"`

- [ ] **Step 1: Add config + policy**

```python
MODEL_CONFIGS["transformer"] = {
    "dropout": 0.3, "layers": 2, "heads": 4, "ff_dim": 1024, "attention": "transformer_cross"
}
DECODE_POLICY["transformer"] = "prefix-recompute (non-incremental)"
```

- [ ] **Step 2: Implement TransformerCaptioner**

```python
@register_model("transformer")
class TransformerCaptioner(nn.Module):
    HAS_ATTENTION = True

    def __init__(self):
        super().__init__()
        c = MODEL_CONFIGS["transformer"]
        H, E, F = CFG["hidden_dim"], CFG["emb_dim"], CFG["feature_dim"]
        self.d_model = H
        self.feat_proj = nn.Linear(F, H)
        self.emb = nn.Embedding(VOCAB_SIZE, H, padding_idx=PAD_IDX)
        self.pos = nn.Embedding(CFG["max_decode_len"] + 2, H)
        layer = nn.TransformerDecoderLayer(
            d_model=H, nhead=c["heads"], dim_feedforward=c["ff_dim"],
            dropout=c["dropout"], batch_first=True)
        self.decoder = nn.TransformerDecoder(layer, num_layers=c["layers"])
        self.fc = nn.Linear(H, VOCAB_SIZE)
        self.drop = nn.Dropout(c["dropout"])
        self.heads = c["heads"]

    def _causal_mask(self, T, device):
        return torch.triu(torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1)

    def _embed(self, tokens):
        # tokens [B,T]
        B, T = tokens.shape
        pos = torch.arange(T, device=tokens.device).unsqueeze(0).expand(B, T)
        return self.drop(self.emb(tokens) + self.pos(pos))

    def forward(self, feats, caps):
        mem = self.feat_proj(feats)                 # [B,49,H]
        tgt = self._embed(caps[:, :-1])
        T = tgt.size(1)
        out = self.decoder(tgt, mem, tgt_mask=self._causal_mask(T, feats.device))
        return self.fc(self.drop(out))

    def init_state(self, feats):
        return {"mem": self.feat_proj(feats), "tokens": None}

    def step(self, state, tokens, return_attention=False):
        # prefix-recompute: append token and run full decoder
        if state["tokens"] is None:
            seq = tokens.unsqueeze(1)
        else:
            seq = torch.cat([state["tokens"], tokens.unsqueeze(1)], dim=1)
        # truncate if somehow exceeds pos table
        if seq.size(1) > CFG["max_decode_len"] + 1:
            seq = seq[:, -(CFG["max_decode_len"] + 1):]
        tgt = self._embed(seq)
        T = tgt.size(1)
        # Capture cross-attention from last layer via a forward hook alternative:
        # Use manual multi-head attention probe on last hidden vs mem for heatmap contract.
        out = self.decoder(tgt, state["mem"], tgt_mask=self._causal_mask(T, tokens.device))
        logits = self.fc(out[:, -1])
        new_state = {"mem": state["mem"], "tokens": seq}
        if not return_attention:
            return logits, new_state
        # Probe cross-attn weights: scaled dot-product of last query vs mem keys
        q = out[:, -1]                              # [B,H]
        B, N, H = state["mem"].shape
        Hh = self.heads
        Hd = H // Hh
        qh = q.view(B, Hh, Hd)
        kh = state["mem"].view(B, N, Hh, Hd).transpose(1, 2)
        scores = torch.einsum("bhd,bhnd->bhn", qh, kh) / math.sqrt(Hd)
        attn = torch.softmax(scores, dim=-1)        # [B,H,49]
        return logits, new_state, attn
```

Note: Transformer heatmap uses a probe on decoder output vs memory (declared in report as approximate cross-attention visualization if hooks are not used). Prefer a cleaner approach if easy: register a forward hook on the last layer’s `multihead_attn` to capture `avg_weights`. If hook path is used, still return `[B, heads, 49]`.

- [ ] **Step 3: Confirm print lists all five architectures**

```python
assert set(MODEL_REGISTRY) == EXPECTED_ARCHS
print("registered:", sorted(MODEL_REGISTRY))
```

---

### Task 3: Shape-check cell

**Files:**
- Modify: `Final Project/captioning_harness_v2_1 (2).ipynb` — insert after cell 12 (new markdown + code) OR include at top of Aadit analysis section after cell 24

**Interfaces:**
- Consumes: `MODEL_REGISTRY`, `EXPECTED_ARCHS`, `device`, `CFG`
- Produces: printed OK / assertion failures

- [ ] **Step 1: Add markdown cell**

```markdown
## 6b. Aadit — model shape smoke test

Run after registering models. Uses random tensors; does not touch Flickr data. Confirms logits shapes and attention contract `[B, H, 49]`.
```

- [ ] **Step 2: Add code cell**

```python
def smoke_models(batch=2, T=8):
    feats = torch.randn(batch, CFG["num_pixels"], CFG["feature_dim"], device=device)
    caps = torch.randint(4, min(50, VOCAB_SIZE), (batch, T), device=device)
    caps[:, 0] = START_IDX
    for name, cls in MODEL_REGISTRY.items():
        m = cls().to(device).eval()
        with torch.no_grad():
            logits = m(feats, caps)
            assert logits.shape == (batch, T - 1, VOCAB_SIZE), (name, logits.shape)
            st = m.init_state(feats)
            tokens = torch.full((batch,), START_IDX, device=device)
            out = m.step(st, tokens, return_attention=True)
            assert len(out) == 3
            logits1, st2, attn = out
            assert logits1.shape == (batch, VOCAB_SIZE)
            if getattr(m, "HAS_ATTENTION", False):
                assert attn is not None and attn.shape[-1] == CFG["num_pixels"]
                assert attn.shape[0] == batch
            else:
                assert attn is None
        print(f"OK {name}: params={sum(p.numel() for p in m.parameters() if p.requires_grad):,}")
        del m
    print("all models passed shape smoke")

# smoke_models()
```

- [ ] **Step 3: Leave call commented** (Colab runs it); if a local torch env exists, uncomment temporarily to verify then re-comment.

---

### Task 4: METEOR and CIDEr helpers

**Files:**
- Modify: `Final Project/captioning_harness_v2_1 (2).ipynb` — insert after cell 24

**Interfaces:**
- Consumes: `greedy_decode_batch`, `decode_ids`, loaders
- Produces: `corpus_meteor`, `corpus_cider`, `evaluate_extra_metrics(model, img_loader)`

- [ ] **Step 1: Markdown cell**

```markdown
## 12b. Aadit — METEOR and CIDEr

Report-only extras. Checkpoint selection remains validation BLEU-4.
METEOR via NLTK; CIDEr via a self-contained TF-IDF n-gram scorer (no pycocoevalcap required).
```

- [ ] **Step 2: Code cell**

```python
import nltk
from nltk.translate.meteor_score import meteor_score

for pkg in ("wordnet", "omw-1.4", "punkt"):
    try:
        nltk.data.find(f"corpora/{pkg}" if pkg != "punkt" else "tokenizers/punkt")
    except LookupError:
        nltk.download(pkg, quiet=True)

def corpus_meteor(refs, hyps):
    # refs: List[List[List[str]]], hyps: List[List[str]]
    scores = []
    for r, h in zip(refs, hyps):
        scores.append(meteor_score(r, h))
    return float(sum(scores) / max(len(scores), 1))

def _ngram_counts(tokens, n):
    return Counter(tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1))

def corpus_cider(refs, hyps, n_max=4):
    # CIDEr-D style without length Gaussian for simplicity; document as CIDEr-TFIDF.
    docs = []
    for rlist in refs:
        for r in rlist:
            docs.append(r)
    df = Counter()
    for doc in docs:
        seen = set()
        for n in range(1, n_max + 1):
            for ng in _ngram_counts(doc, n):
                if ng not in seen:
                    df[ng] += 1
                    seen.add(ng)
    N = max(len(docs), 1)
    def tfidf_vec(tokens):
        vec = Counter()
        for n in range(1, n_max + 1):
            counts = _ngram_counts(tokens, n)
            tot = sum(counts.values()) or 1
            for ng, c in counts.items():
                idf = math.log(N / (df[ng] or 1)) + 1.0  # avoid div0; rare ngrams ok
                vec[ng] += (c / tot) * idf
        return vec
    def cos(a, b):
        if not a or not b:
            return 0.0
        keys = set(a) | set(b)
        dot = sum(a[k] * b[k] for k in keys)
        na = math.sqrt(sum(v*v for v in a.values()))
        nb = math.sqrt(sum(v*v for v in b.values()))
        return dot / (na * nb + 1e-12)
    scores = []
    for rlist, h in zip(refs, hyps):
        hv = tfidf_vec(h)
        scores.append(sum(cos(hv, tfidf_vec(r)) for r in rlist) / max(len(rlist), 1))
    return float(sum(scores) / max(len(scores), 1))

@torch.no_grad()
def evaluate_extra_metrics(model, img_loader):
    hyps, refs = [], []
    for feats, ref_batch in img_loader:
        ids = greedy_decode_batch(model, feats.to(device))
        for row, r in zip(ids.cpu().tolist(), ref_batch):
            hyps.append(decode_ids(row)); refs.append(r)
    return {"METEOR": corpus_meteor(refs, hyps), "CIDEr": corpus_cider(refs, hyps)}
```

- [ ] **Step 3: Sanity-check helpers with toy strings**

```python
_refs = [[["a", "dog", "runs"], ["a", "dog", "is", "running"]]]
_hyps = [["a", "dog", "runs"]]
print("toy METEOR", corpus_meteor(_refs, _hyps), "CIDEr", corpus_cider(_refs, _hyps))
```

Expected: finite floats in `[0, 1+]` range (CIDEr can exceed 1).

---

### Task 5: Beam search

**Files:**
- Modify: same notebook, after Task 4 cells

**Interfaces:**
- Consumes: `model.init_state`, `model.step`, `START_IDX`, `END_IDX`, `PAD_IDX`, `CFG`
- Produces: `beam_decode_batch(model, feats, beam_size=3)` → `LongTensor [B, L]`

- [ ] **Step 1: Markdown**

```markdown
## 12c. Aadit — beam search

Default `beam_size=3`. Compare to greedy on validation only during development.
```

- [ ] **Step 2: Implementation**

```python
@torch.no_grad()
def beam_decode_batch(model, feats, beam_size=3):
    """Per-image beam search (loop over batch). Returns token ids [B, max_len]."""
    model.eval()
    B = feats.size(0)
    outs = []
    for b in range(B):
        f = feats[b:b+1]
        state0 = model.init_state(f)
        # beam entries: (logprob, token_list, state, done)
        beams = [(0.0, [START_IDX], state0, False)]
        for _ in range(CFG["max_decode_len"]):
            nxt = []
            for lp, toks, st, done in beams:
                if done:
                    nxt.append((lp, toks, st, True)); continue
                logits, st2 = model.step(st, torch.tensor([toks[-1]], device=feats.device))
                logp = torch.log_softmax(logits[0], dim=-1)
                topv, topi = logp.topk(beam_size)
                for v, i in zip(topv.tolist(), topi.tolist()):
                    nt = toks + [i]
                    nxt.append((lp + v, nt, st2, i == END_IDX))
            nxt.sort(key=lambda x: x[0], reverse=True)
            beams = nxt[:beam_size]
            if all(x[3] for x in beams):
                break
        best = max(beams, key=lambda x: x[0])[1][1:]  # drop START
        # pad/truncate
        row = best[:CFG["max_decode_len"]] + [PAD_IDX] * max(0, CFG["max_decode_len"] - len(best))
        outs.append(row[:CFG["max_decode_len"]])
    return torch.tensor(outs, dtype=torch.long, device=feats.device)
```

- [ ] **Step 3: Note in markdown that beam-vs-greedy BLEU helper can wrap `evaluate_bleu` pattern by swapping decode function if needed later.**

---

### Task 6: Attention heatmaps

**Files:**
- Modify: same notebook after Task 5

**Interfaces:**
- Consumes: `greedy_decode_batch(..., collect_attention=True)`, checkpoints, `IMG_DIR`, `splits`
- Produces: `show_attention_heatmaps(arch, lr, seed, phase="final", n_words=5, sample_idx=0)`

- [ ] **Step 1: Markdown**

```markdown
## 12d. Aadit — attention heatmaps

Validation images only. Requires a trained checkpoint on Drive.
```

- [ ] **Step 2: Code**

```python
def show_attention_heatmaps(arch, lr, seed, phase="final", n_words=5, sample_idx=0):
    assert MODEL_REGISTRY[arch]().HAS_ATTENTION, f"{arch} has no attention"
    rid = run_id(arch, lr, seed, phase)
    best_path, _ = ckpt_paths(rid)
    ck = torch.load(best_path, map_location=device, weights_only=False)
    model = MODEL_REGISTRY[arch]().to(device)
    model.load_state_dict(ck["model_state_dict"]); model.eval()

    fn, refs = splits["val"][sample_idx]
    img = Image.open(os.path.join(IMG_DIR, fn)).convert("RGB")
    f = feats_all[file_to_idx[fn]].float().unsqueeze(0).to(device)
    ids, attns = greedy_decode_batch(model, f, collect_attention=True)
    tokens = decode_ids(ids[0].cpu().tolist())
    # attns: list of [1,H,49] or None
    n = min(n_words, len(tokens), len(attns))
    fig, axes = plt.subplots(1, n + 1, figsize=(3.2 * (n + 1), 3.2))
    axes[0].imshow(img); axes[0].set_title("image"); axes[0].axis("off")
    for i in range(n):
        a = attns[i]
        if a is None:
            axes[i+1].axis("off"); continue
        heat = a[0].mean(0).detach().cpu().numpy().reshape(7, 7)
        heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-8)
        heat_img = np.array(Image.fromarray(np.uint8(heat * 255)).resize(img.size, resample=Image.BILINEAR))
        axes[i+1].imshow(img)
        axes[i+1].imshow(heat_img, alpha=0.45, cmap="jet")
        axes[i+1].set_title(tokens[i]); axes[i+1].axis("off")
    fig.suptitle(rid)
    plt.tight_layout(); plt.show()
    del model; torch.cuda.empty_cache()

# show_attention_heatmaps("gru_luong", 3e-4, 1)
```

---

### Task 7: Failure analysis

**Files:**
- Modify: same notebook after Task 6

**Interfaces:**
- Consumes: model + val image loader / splits
- Produces: `failure_analysis(arch, lr, seed, phase="final", max_examples=8)` printing counts + examples

- [ ] **Step 1: Markdown**

```markdown
## 12e. Aadit — failure analysis (validation)

Flags repetitions, very short captions, and high novelty vs references (hallucination heuristic).
```

- [ ] **Step 2: Code**

```python
GENERIC = {"a", "the", "in", "on", "of", "and", "with", "is", "are", "to", "at"}

def _is_repeat(toks, min_run=3):
    if len(toks) < min_run:
        return False
    for i in range(len(toks) - min_run + 1):
        if len(set(toks[i:i+min_run])) == 1:
            return True
    return False

@torch.no_grad()
def failure_analysis(arch, lr, seed, phase="final", max_examples=8):
    rid = run_id(arch, lr, seed, phase)
    best_path, _ = ckpt_paths(rid)
    ck = torch.load(best_path, map_location=device, weights_only=False)
    model = MODEL_REGISTRY[arch]().to(device)
    model.load_state_dict(ck["model_state_dict"]); model.eval()
    loader = DataLoader(ImageRefDataset("val"), batch_size=CFG["batch_size"],
                        shuffle=False, collate_fn=ref_collate)
    rows = []
    for feats, ref_batch in loader:
        ids = greedy_decode_batch(model, feats.to(device))
        for row, refs in zip(ids.cpu().tolist(), ref_batch):
            hyp = decode_ids(row)
            ref_sets = [set(r) for r in refs]
            union = set().union(*ref_sets) if ref_sets else set()
            novel = [t for t in hyp if t not in union and t not in GENERIC]
            rows.append({
                "hyp": hyp,
                "refs": refs,
                "short": len(hyp) <= 3,
                "repeat": _is_repeat(hyp),
                "novel_ratio": len(novel) / max(len(hyp), 1),
                "novel": novel,
            })
    n = len(rows)
    n_short = sum(r["short"] for r in rows)
    n_rep = sum(r["repeat"] for r in rows)
    n_hall = sum(r["novel_ratio"] >= 0.4 for r in rows)
    print(f"{rid}: n={n} short={n_short} repeat={n_rep} high_novelty={n_hall}")
    shown = 0
    for r in rows:
        if not (r["short"] or r["repeat"] or r["novel_ratio"] >= 0.4):
            continue
        print("---")
        print("hyp:", " ".join(r["hyp"]))
        print("ref:", " ".join(r["refs"][0]))
        print("flags:",
              "short" if r["short"] else "",
              "repeat" if r["repeat"] else "",
              f"novel={r['novel_ratio']:.2f}")
        shown += 1
        if shown >= max_examples:
            break
    del model; torch.cuda.empty_cache()
    return {"n": n, "short": n_short, "repeat": n_rep, "high_novelty": n_hall}

# failure_analysis("gru_luong", 3e-4, 1)
```

---

### Task 8: Aadit run cells + handoff update

**Files:**
- Modify: insert before handoff cell; update handoff markdown

**Interfaces:**
- Consumes: `run_protocol`, `benchmark_run`, `summary_table`
- Produces: ready-to-run cells for four architectures

- [ ] **Step 1: Insert markdown**

```markdown
## 14b. Aadit — train attention / Transformer models

Smoke first in a scratch copy: set `CFG["max_epochs"] = 1`, run one `train_run(...)`, confirm JSON + checkpoint, restore `max_epochs`.
Then uncomment one architecture at a time.
```

- [ ] **Step 2: Insert code cell**

```python
# benchmark_run("gru_luong")
# luong_results = run_protocol("gru_luong", verbose=True)

# benchmark_run("gru_bahdanau")
# bahdanau_results = run_protocol("gru_bahdanau", verbose=True)

# benchmark_run("gru_multihead")
# multihead_results = run_protocol("gru_multihead", verbose=True)

# benchmark_run("transformer")
# transformer_results = run_protocol("transformer", verbose=True)

# summary_table()
```

- [ ] **Step 3: Update handoff cell (final markdown)** to state that model registry + analysis cells are implemented; remaining work is Colab execution, joint freeze/test, report, Canvas.

- [ ] **Step 4: Manual notebook review**

Open the notebook JSON / UI and confirm:

1. `EXPECTED_ARCHS` still lists five names.
2. Cell 12 registers all five.
3. Analysis helpers exist and calls are commented.
4. Anthony baseline `run_protocol("gru_baseline")` cell still present.
5. No `CFG` hyperparameter changes.

---

## Spec Coverage Check

| Spec requirement | Task |
|---|---|
| Luong / Bahdanau / multi-head | Task 1 |
| Transformer + prefix-recompute policy | Task 2 |
| Attention `[B,H,49]` contract | Tasks 1–3 |
| METEOR + CIDEr | Task 4 |
| Beam search (size 3) | Task 5 |
| Heatmaps | Task 6 |
| Failure analysis | Task 7 |
| `run_protocol` cells + smoke guidance | Task 8 |
| No fine-tune ablation / no CFG churn / no GPU training here | Global constraints |

## Placeholder / consistency self-review

- Exact decode policy strings match the spec.
- `HAS_ATTENTION` used consistently for heatmap gating.
- Beam default `3` matches spec.
- Transformer heatmap probe documented as acceptable fallback if MHA hooks are awkward in notebook code; implementer should prefer real cross-attn weights via hook when straightforward.

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-07-23-aadit-image-captioning.md`.

**Two execution options:**

1. **Subagent-Driven (recommended)** — fresh subagent per task, review between tasks  
2. **Inline Execution** — execute tasks in this session with checkpoints  

Which approach?
