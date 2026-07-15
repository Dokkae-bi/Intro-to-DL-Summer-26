"""
ECGR 4106 - Homework 5, Problem 2
Swin Transformer from scratch, adapted for 32x32 CIFAR-100.

Architecture follows Liu et al. 2021 (Swin Transformer), adapted for small
inputs: patch 4 -> 8x8 feature map, 3 stages (Swin-Tiny's first three),
window size 4, shift 2. Stages whose feature map is <= window size fall
back to global attention with no shift (same behavior as timm at small
resolutions).

Usage in Colab (after uploading this file to the session or mounting Drive):
    from swin_model import SwinScratch
    model = SwinScratch()
"""

import torch
import torch.nn as nn


def window_partition(x, ws):
    """(B, H, W, C) -> (num_windows*B, ws, ws, C)"""
    B, H, W, C = x.shape
    x = x.view(B, H // ws, ws, W // ws, ws, C)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, ws, ws, C)


def window_reverse(windows, ws, H, W):
    """(num_windows*B, ws, ws, C) -> (B, H, W, C)"""
    B = int(windows.shape[0] / (H * W / ws / ws))
    x = windows.view(B, H // ws, W // ws, ws, ws, -1)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)


class WindowAttention(nn.Module):
    """Multi-head self-attention within a window, with learned relative
    position bias (the Swin replacement for absolute positional embeddings)."""

    def __init__(self, dim, ws, num_heads):
        super().__init__()
        self.ws = ws
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5

        # Relative position bias table: (2*ws-1)^2 relative offsets per head
        self.rpb_table = nn.Parameter(
            torch.zeros((2 * ws - 1) ** 2, num_heads))

        # Precompute pairwise relative position index for all token pairs
        coords = torch.stack(torch.meshgrid(
            torch.arange(ws), torch.arange(ws), indexing="ij"))  # (2, ws, ws)
        coords = coords.flatten(1)                               # (2, ws*ws)
        rel = coords[:, :, None] - coords[:, None, :]            # (2, N, N)
        rel = rel.permute(1, 2, 0).contiguous()                  # (N, N, 2)
        rel[:, :, 0] += ws - 1
        rel[:, :, 1] += ws - 1
        rel[:, :, 0] *= 2 * ws - 1
        self.register_buffer("rpb_index", rel.sum(-1))           # (N, N)

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        nn.init.trunc_normal_(self.rpb_table, std=0.02)

    def forward(self, x, mask=None):
        # x: (num_windows*B, N, C) where N = ws*ws
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale            # (B_, H, N, N)

        bias = self.rpb_table[self.rpb_index.view(-1)]           # (N*N, heads)
        bias = bias.view(N, N, -1).permute(2, 0, 1)              # (heads, N, N)
        attn = attn + bias.unsqueeze(0)

        if mask is not None:                                     # shifted windows
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) \
                   + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        return self.proj(out)


class SwinBlock(nn.Module):
    """One Swin block: (optionally shifted) window attention + MLP,
    pre-LN residuals. shift_size=0 -> W-MSA, shift_size=ws//2 -> SW-MSA."""

    def __init__(self, dim, input_res, num_heads, ws=4, shift_size=0,
                 mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.input_res = input_res
        # If the window covers the whole feature map: global attention, no shift
        if min(input_res) <= ws:
            ws = min(input_res)
            shift_size = 0
        self.ws = ws
        self.shift_size = shift_size

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, ws, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, dim), nn.Dropout(dropout))
        self.drop = nn.Dropout(dropout)

        # Attention mask for shifted windows: tokens wrapped around by the
        # cyclic shift must not attend to non-adjacent content
        if self.shift_size > 0:
            H, W = input_res
            img_mask = torch.zeros(1, H, W, 1)
            slices = (slice(0, -ws), slice(-ws, -self.shift_size),
                      slice(-self.shift_size, None))
            cnt = 0
            for h in slices:
                for w in slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1
            mask_windows = window_partition(img_mask, ws).view(-1, ws * ws)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)) \
                                 .masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None
        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):
        # x: (B, H*W, C)
        H, W = self.input_res
        B, L, C = x.shape
        shortcut = x

        x = self.norm1(x).view(B, H, W, C)
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size),
                           dims=(1, 2))
        x = window_partition(x, self.ws).view(-1, self.ws * self.ws, C)
        x = self.attn(x, mask=self.attn_mask)
        x = window_reverse(x.view(-1, self.ws, self.ws, C), self.ws, H, W)
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(self.shift_size, self.shift_size),
                           dims=(1, 2))
        x = x.view(B, H * W, C)

        x = shortcut + self.drop(x)
        x = x + self.mlp(self.norm2(x))
        return x


class PatchMerging(nn.Module):
    """Downsample 2x: concatenate each 2x2 neighborhood (4C) -> linear to 2C."""

    def __init__(self, input_res, dim):
        super().__init__()
        self.input_res = input_res
        self.norm = nn.LayerNorm(4 * dim)
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)

    def forward(self, x):
        H, W = self.input_res
        B, L, C = x.shape
        x = x.view(B, H, W, C)
        x = torch.cat([x[:, 0::2, 0::2], x[:, 1::2, 0::2],
                       x[:, 0::2, 1::2], x[:, 1::2, 1::2]], dim=-1)
        x = x.view(B, -1, 4 * C)
        return self.reduction(self.norm(x))


class SwinScratch(nn.Module):
    """Swin for 32x32: patch 4 -> 8x8 map, 3 stages (Swin-Tiny's first three),
    window 4, mean-pool head (Swin uses no CLS token).

    Regularization: dropout 0.1 throughout; pair with gradient clipping
    (max norm 1.0) in the training loop for scratch training at lr=1e-3."""

    def __init__(self, img_size=32, patch_size=4, in_chans=3, num_classes=100,
                 embed_dim=96, depths=(2, 2, 6), num_heads=(3, 6, 12),
                 ws=4, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.patch_embed = nn.Conv2d(in_chans, embed_dim,
                                     kernel_size=patch_size, stride=patch_size)
        self.pos_drop = nn.Dropout(dropout)

        res = img_size // patch_size          # 8
        dim = embed_dim
        self.layers = nn.ModuleList()
        for i, (depth, heads) in enumerate(zip(depths, num_heads)):
            for d in range(depth):
                self.layers.append(SwinBlock(
                    dim, (res, res), heads, ws=ws,
                    shift_size=0 if d % 2 == 0 else ws // 2,
                    mlp_ratio=mlp_ratio, dropout=dropout))
            if i < len(depths) - 1:           # merge between stages
                self.layers.append(PatchMerging((res, res), dim))
                res //= 2
                dim *= 2

        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.patch_embed(x)               # (B, C, 8, 8)
        x = x.flatten(2).transpose(1, 2)      # (B, 64, C)
        x = self.pos_drop(x)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)                      # (B, 4, 384)
        return self.head(x.mean(dim=1))       # global average pool


if __name__ == "__main__":
    # Quick self-test (CPU fallback works: no device assumptions here)
    m = SwinScratch()
    out = m(torch.randn(2, 3, 32, 32))
    print(f"Output shape: {out.shape}")   # expect (2, 100)
    n = sum(p.numel() for p in m.parameters() if p.requires_grad)
    print(f"Params: {n:,}")               # expect ~11-12M
