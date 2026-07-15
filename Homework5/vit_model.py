"""
ECGR 4106 - Homework 5, Problem 1
Vision Transformer from scratch for CIFAR-100 (32x32 inputs).

Usage in Colab (after uploading this file to the session or mounting Drive):
    from vit_model import ViT, count_params
    model = ViT(patch_size=4, embed_dim=256, depth=4, num_heads=4)
"""

import torch
import torch.nn as nn


class PatchEmbed(nn.Module):
    """Splits image into non-overlapping patches and linearly projects each
    to the embedding dimension. Implemented as a Conv2d with kernel = stride
    = patch size, which is mathematically identical to unfold + linear."""

    def __init__(self, img_size=32, patch_size=4, in_chans=3, embed_dim=256):
        super().__init__()
        assert img_size % patch_size == 0, "image size must be divisible by patch size"
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim,
                              kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)                      # (B, D, H/P, W/P)
        x = x.flatten(2).transpose(1, 2)      # (B, N, D)
        return x


class MultiHeadSelfAttention(nn.Module):
    """Standard scaled dot-product multi-head self-attention,
    written out explicitly rather than using nn.MultiheadAttention."""

    def __init__(self, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(embed_dim, embed_dim * 3)   # fused Q,K,V projection
        self.proj = nn.Linear(embed_dim, embed_dim)      # output projection

    def forward(self, x):
        B, N, D = x.shape
        qkv = self.qkv(x)                                        # (B, N, 3D)
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)                         # (3, B, H, N, d)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale            # (B, H, N, N)
        attn = attn.softmax(dim=-1)
        out = attn @ v                                           # (B, H, N, d)
        out = out.transpose(1, 2).reshape(B, N, D)               # concat heads
        return self.proj(out)


class TransformerBlock(nn.Module):
    """Pre-LayerNorm encoder block:
    x = x + MSA(LN(x));  x = x + MLP(LN(x))"""

    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads)
        self.norm2 = nn.LayerNorm(embed_dim)
        hidden = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, embed_dim),
            nn.Dropout(dropout),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.drop(self.attn(self.norm1(x)))
        x = x + self.mlp(self.norm2(x))
        return x


class ViT(nn.Module):
    """Vision Transformer for CIFAR-100.
    patch embed -> [CLS] + learned positional embeddings -> L encoder blocks
    -> LayerNorm -> linear head on the CLS token.

    Regularization: dropout 0.1 applied to positional embeddings,
    attention output, and MLP layers."""

    def __init__(self, img_size=32, patch_size=4, in_chans=3, num_classes=100,
                 embed_dim=256, depth=4, num_heads=4, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        # Initialization (truncated normal, standard ViT practice)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)                            # (B, N, D)
        cls = self.cls_token.expand(B, -1, -1)             # (B, 1, D)
        x = torch.cat([cls, x], dim=1)                     # (B, N+1, D)
        x = self.pos_drop(x + self.pos_embed)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return self.head(x[:, 0])                          # classify on CLS


def count_params(model):
    """Number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Quick self-test (CPU fallback works: no device assumptions here)
    m = ViT(patch_size=4, embed_dim=256, depth=4, num_heads=4)
    out = m(torch.randn(2, 3, 32, 32))
    print(f"Output shape: {out.shape}")       # expect (2, 100)
    print(f"Params: {count_params(m):,}")     # expect ~3.2M
