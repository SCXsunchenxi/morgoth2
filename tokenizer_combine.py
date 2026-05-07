"""
tokenizer_combine.py
====================
Multi-tokenizer fusion classifier.

Six fusion strategies (--fusion_type):

  concat       — all modality tokens concatenated; one full self-attention
                 transformer.  Simple baseline.

  hierarchical — local per-modality transformer (shared weights) → global
                 transformer over K per-modality CLS tokens.

  cross_attn   — anchor modality (index 0) as Q; all others as KV.
                 Cross-attention blocks update the anchor with context.

  cls_cross    — independent per-modality transformers (unshared weights)
                 → fuse K CLS tokens with a small global transformer.

  gated        — independent per-modality transformers; a learnable gating
                 network weights each modality's CLS before fusion; optional
                 second-stage transformer on the fused feature.

  perceiver    — fixed-size latent array (learnable queries) alternates
                 between cross-attending to all modality tokens and
                 self-attending within the latent space.  Scales well to
                 many/long modality sequences.

Three head types (--head_type):

  linear       — Linear(D, C)
  mlp          — Linear → GELU → Dropout → Linear   [default]
  mlp_norm     — LayerNorm → Linear → GELU → Dropout → Linear

JSON config (tok_cfg.json)
──────────────────────────
{
  "tokenizers": [
    { "name": "EEG", "checkpoint": "...",
      "model": "vqnsp_encoder_base_decoder_3x200x12",
      "n_embed": 8192, "code_dim": 32, "eeg_size": 1600, "patch_size": 200,
      "channel_indices": [0,...,18], "freeze": true },
    { "name": "ECG", ..., "channel_indices": [19] },
    ...
  ]
}
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import sys
import time
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.layers import trunc_normal_
from timm.models import create_model
from torch.utils.data import DataLoader, Dataset

import backbone as _bb   # noqa — registers backbone models
import tokenizer as _tk  # noqa — registers tokenizer models
from backbone import Block, Mlp, DropPath
from tokenizer import VQNSP


# ============================================================================
# Tokenizer specification
# ============================================================================

@dataclass
class TokenizerSpec:
    checkpoint: str
    model: str
    channel_indices: List[int]
    n_embed: int = 8192
    code_dim: int = 32
    eeg_size: int = 1600
    patch_size: int = 200
    name: str = ""
    freeze: bool = True
    n_time_patches: int = field(init=False)
    n_tokens: int = field(init=False)

    def __post_init__(self):
        self.n_time_patches = self.eeg_size // self.patch_size
        self.n_tokens = len(self.channel_indices) * self.n_time_patches

    @classmethod
    def from_dict(cls, d: dict) -> "TokenizerSpec":
        known = {f for f in cls.__dataclass_fields__
                 if f not in ("n_time_patches", "n_tokens")}
        return cls(**{k: v for k, v in d.items() if k in known})


def load_specs(cfg_path: str) -> List[TokenizerSpec]:
    with open(cfg_path) as f:
        cfg = json.load(f)
    return [TokenizerSpec.from_dict(t) for t in cfg["tokenizers"]]


def _load_vqnsp(spec: TokenizerSpec, device: str) -> VQNSP:
    model = create_model(
        spec.model, n_code=spec.n_embed,
        code_dim=spec.code_dim, EEG_size=spec.eeg_size,
    )
    raw = torch.load(spec.checkpoint, map_location="cpu")
    state = raw.get("model", raw.get("state_dict", raw))
    state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state, strict=False)
    model.to(device)
    if spec.freeze:
        for p in model.parameters():
            p.requires_grad_(False)
        model.eval()
    return model


# ============================================================================
# Shared building blocks
# ============================================================================

def _make_blocks(embed_dim, depth, num_heads, mlp_ratio,
                 drop_rate, attn_drop_rate, drop_path_rate) -> nn.ModuleList:
    stoch = [x.item() for x in torch.linspace(0, drop_path_rate, max(depth, 1))]
    return nn.ModuleList([
        Block(
            dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
            qkv_bias=True, qk_norm=partial(nn.LayerNorm, eps=1e-6),
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=stoch[i],
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
        )
        for i in range(depth)
    ])


class CrossAttentionBlock(nn.Module):
    """Single cross-attention block: Q from one stream, KV from another."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0,
                 drop: float = 0.0, attn_drop: float = 0.0,
                 drop_path: float = 0.0):
        super().__init__()
        self.norm_q  = nn.LayerNorm(dim, eps=1e-6)
        self.norm_kv = nn.LayerNorm(dim, eps=1e-6)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads,
            dropout=attn_drop, batch_first=True,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = Mlp(in_features=dim,
                       hidden_features=int(dim * mlp_ratio), drop=drop)

    def forward(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(self.norm_q(q), self.norm_kv(kv), self.norm_kv(kv))
        q = q + self.drop_path(attn_out)
        q = q + self.drop_path(self.mlp(self.norm2(q)))
        return q


def _init_weights(m):
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)


# ============================================================================
# Classification head factory
# ============================================================================

def build_head(in_dim: int, out_dim: int,
               head_type: str = "mlp",
               hidden_dim: Optional[int] = None,
               dropout: float = 0.0) -> nn.Module:
    """
    head_type
    ─────────
    linear    : Linear(in, out)
    mlp       : Linear(in, H) → GELU → Dropout → Linear(H, out)
    mlp_norm  : LayerNorm(in) → Linear(in, H) → GELU → Dropout → Linear(H, out)
    """
    H = hidden_dim or in_dim
    if head_type == "linear":
        return nn.Linear(in_dim, out_dim)
    if head_type == "mlp":
        return nn.Sequential(
            nn.Linear(in_dim, H),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(H, out_dim),
        )
    if head_type == "mlp_norm":
        return nn.Sequential(
            nn.LayerNorm(in_dim, eps=1e-6),
            nn.Linear(in_dim, H),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(H, out_dim),
        )
    raise ValueError(f"Unknown head_type '{head_type}'. "
                     "Choose: linear | mlp | mlp_norm")


# ============================================================================
# Strategy 1 — ConcatFusion
# ============================================================================

class ConcatFusion(nn.Module):
    """
    All modality tokens concatenated into one sequence.
    Standard full self-attention across all tokens.

        [CLS] + [tok_0 ‥ tok_K] → Transformer(depth) → CLS → head
    """

    def __init__(self, specs, fusion_dim, depth, num_heads, mlp_ratio,
                 nb_classes, drop_rate, attn_drop_rate, drop_path_rate,
                 use_segment_embed, pool, is_binary,
                 head_type="mlp", head_hidden_dim=None, head_dropout=0.0):
        super().__init__()
        self.specs = specs
        self.pool  = pool
        self.is_binary = is_binary

        self.token_projs = nn.ModuleList([
            nn.Linear(s.code_dim, fusion_dim) for s in specs
        ])
        self.use_segment_embed = use_segment_embed
        if use_segment_embed:
            self.segment_embeds = nn.ParameterList([
                nn.Parameter(torch.zeros(1, 1, fusion_dim)) for _ in specs
            ])

        max_seq = sum(s.n_tokens for s in specs)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, fusion_dim))
        self.pos_embed  = nn.Parameter(torch.zeros(1, max_seq + 1, fusion_dim))
        self.pos_drop   = nn.Dropout(p=drop_rate)

        self.blocks = _make_blocks(fusion_dim, depth, num_heads, mlp_ratio,
                                   drop_rate, attn_drop_rate, drop_path_rate)
        self.norm = nn.LayerNorm(fusion_dim)
        self.head = build_head(fusion_dim, 1 if is_binary else nb_classes,
                               head_type, head_hidden_dim, head_dropout)

        trunc_normal_(self.cls_token, std=0.02)
        trunc_normal_(self.pos_embed, std=0.02)
        self.apply(_init_weights)

    def forward(self, token_seqs: List[torch.Tensor]) -> torch.Tensor:
        B = token_seqs[0].shape[0]
        parts = []
        for i, q in enumerate(token_seqs):
            t = self.token_projs[i](q)
            if self.use_segment_embed:
                t = t + self.segment_embeds[i]
            parts.append(t)
        tokens = torch.cat(parts, dim=1)
        cls    = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        tokens = tokens + self.pos_embed[:, :tokens.shape[1]]
        tokens = self.pos_drop(tokens)
        for blk in self.blocks:
            tokens = blk(tokens)
        tokens = self.norm(tokens)
        feat = tokens[:, 0] if self.pool == "cls" else tokens[:, 1:].mean(1)
        return self.head(feat)


# ============================================================================
# Strategy 2 — HierarchicalFusion
# ============================================================================

class HierarchicalFusion(nn.Module):
    """
    Stage 1 (local, weight-shared): per-modality transformer → CLS_i
    Stage 2 (global):               transformer over [CLS_g, CLS_0 ‥ CLS_K]
    """

    def __init__(self, specs, fusion_dim, depth_local, depth_global,
                 num_heads, mlp_ratio, nb_classes,
                 drop_rate, attn_drop_rate, drop_path_rate, is_binary,
                 head_type="mlp", head_hidden_dim=None, head_dropout=0.0):
        super().__init__()
        self.specs = specs
        self.is_binary = is_binary
        K = len(specs)

        self.token_projs = nn.ModuleList([
            nn.Linear(s.code_dim, fusion_dim) for s in specs
        ])
        self.local_cls  = nn.ParameterList([
            nn.Parameter(torch.zeros(1, 1, fusion_dim)) for _ in specs
        ])
        self.local_pos  = nn.ParameterList([
            nn.Parameter(torch.zeros(1, s.n_tokens + 1, fusion_dim)) for s in specs
        ])
        self.local_drop = nn.Dropout(p=drop_rate)
        self.local_blocks = _make_blocks(
            fusion_dim, depth_local, num_heads, mlp_ratio,
            drop_rate, attn_drop_rate, drop_path_rate)
        self.local_norm = nn.LayerNorm(fusion_dim)

        self.global_cls    = nn.Parameter(torch.zeros(1, 1, fusion_dim))
        self.global_pos    = nn.Parameter(torch.zeros(1, K + 1, fusion_dim))
        self.global_blocks = _make_blocks(
            fusion_dim, depth_global, num_heads, mlp_ratio,
            drop_rate, attn_drop_rate, drop_path_rate)
        self.global_norm   = nn.LayerNorm(fusion_dim)
        self.head = build_head(fusion_dim, 1 if is_binary else nb_classes,
                               head_type, head_hidden_dim, head_dropout)

        for p in self.local_cls:  trunc_normal_(p, std=0.02)
        for p in self.local_pos:  trunc_normal_(p, std=0.02)
        trunc_normal_(self.global_cls, std=0.02)
        trunc_normal_(self.global_pos, std=0.02)
        self.apply(_init_weights)

    def forward(self, token_seqs: List[torch.Tensor]) -> torch.Tensor:
        B = token_seqs[0].shape[0]
        local_cls_list = []
        for i, q in enumerate(token_seqs):
            t = self.token_projs[i](q)
            t = torch.cat([self.local_cls[i].expand(B, -1, -1), t], dim=1)
            t = t + self.local_pos[i][:, :t.shape[1]]
            t = self.local_drop(t)
            for blk in self.local_blocks:
                t = blk(t)
            local_cls_list.append(self.local_norm(t)[:, 0:1])

        ctx = torch.cat([self.global_cls.expand(B, -1, -1)] + local_cls_list, dim=1)
        ctx = ctx + self.global_pos[:, :ctx.shape[1]]
        for blk in self.global_blocks:
            ctx = blk(ctx)
        return self.head(self.global_norm(ctx)[:, 0])


# ============================================================================
# Strategy 3 — CrossAttentionFusion
# ============================================================================

class CrossAttentionFusion(nn.Module):
    """
    Anchor modality (index 0) as Q; all other modalities concatenated as KV.
    Cross-attention blocks update the anchor using context from all others.
    """

    def __init__(self, specs, fusion_dim, depth, num_heads, mlp_ratio,
                 nb_classes, drop_rate, attn_drop_rate, drop_path_rate,
                 pool, is_binary,
                 head_type="mlp", head_hidden_dim=None, head_dropout=0.0):
        super().__init__()
        self.specs = specs
        self.pool  = pool
        self.is_binary = is_binary

        self.token_projs = nn.ModuleList([
            nn.Linear(s.code_dim, fusion_dim) for s in specs
        ])
        self.cls_token  = nn.Parameter(torch.zeros(1, 1, fusion_dim))
        self.anchor_pos = nn.Parameter(
            torch.zeros(1, specs[0].n_tokens + 1, fusion_dim))
        self.pos_drop   = nn.Dropout(p=drop_rate)

        stoch = [x.item() for x in torch.linspace(0, drop_path_rate, max(depth, 1))]
        self.cross_blocks = nn.ModuleList([
            CrossAttentionBlock(fusion_dim, num_heads, mlp_ratio,
                                drop_rate, attn_drop_rate, stoch[i])
            for i in range(depth)
        ])
        self.norm = nn.LayerNorm(fusion_dim)
        self.head = build_head(fusion_dim, 1 if is_binary else nb_classes,
                               head_type, head_hidden_dim, head_dropout)

        trunc_normal_(self.cls_token, std=0.02)
        trunc_normal_(self.anchor_pos, std=0.02)
        self.apply(_init_weights)

    def forward(self, token_seqs: List[torch.Tensor]) -> torch.Tensor:
        B = token_seqs[0].shape[0]
        anchor = self.token_projs[0](token_seqs[0])
        cls    = self.cls_token.expand(B, -1, -1)
        anchor = torch.cat([cls, anchor], dim=1)
        anchor = anchor + self.anchor_pos[:, :anchor.shape[1]]
        anchor = self.pos_drop(anchor)

        if len(token_seqs) > 1:
            context = torch.cat(
                [self.token_projs[i](token_seqs[i]) for i in range(1, len(token_seqs))],
                dim=1)
        else:
            context = anchor

        for blk in self.cross_blocks:
            anchor = blk(anchor, context)
        anchor = self.norm(anchor)
        feat = anchor[:, 0] if self.pool == "cls" else anchor[:, 1:].mean(1)
        return self.head(feat)


# ============================================================================
# Strategy 4 — CLSCrossFusion
# ============================================================================

class CLSCrossFusion(nn.Module):
    """
    Each modality has its own independent transformer (unshared weights).
    The K CLS tokens are fused by a global self-attention transformer.
    """

    def __init__(self, specs, fusion_dim, depth_local, depth_global,
                 num_heads, mlp_ratio, nb_classes,
                 drop_rate, attn_drop_rate, drop_path_rate, is_binary,
                 head_type="mlp", head_hidden_dim=None, head_dropout=0.0):
        super().__init__()
        self.specs = specs
        self.is_binary = is_binary
        K = len(specs)

        self.token_projs = nn.ModuleList([
            nn.Linear(s.code_dim, fusion_dim) for s in specs
        ])
        self.local_cls   = nn.ParameterList([
            nn.Parameter(torch.zeros(1, 1, fusion_dim)) for _ in specs
        ])
        self.local_pos   = nn.ParameterList([
            nn.Parameter(torch.zeros(1, s.n_tokens + 1, fusion_dim)) for s in specs
        ])
        self.local_drop  = nn.Dropout(p=drop_rate)
        self.local_blocks = nn.ModuleList([
            _make_blocks(fusion_dim, depth_local, num_heads, mlp_ratio,
                         drop_rate, attn_drop_rate, drop_path_rate)
            for _ in specs
        ])
        self.local_norms = nn.ModuleList([nn.LayerNorm(fusion_dim) for _ in specs])

        self.global_pos    = nn.Parameter(torch.zeros(1, K, fusion_dim))
        self.global_blocks = _make_blocks(
            fusion_dim, depth_global, num_heads, mlp_ratio,
            drop_rate, attn_drop_rate, drop_path_rate)
        self.global_norm   = nn.LayerNorm(fusion_dim)
        self.head = build_head(fusion_dim, 1 if is_binary else nb_classes,
                               head_type, head_hidden_dim, head_dropout)

        for p in self.local_cls:  trunc_normal_(p, std=0.02)
        for p in self.local_pos:  trunc_normal_(p, std=0.02)
        trunc_normal_(self.global_pos, std=0.02)
        self.apply(_init_weights)

    def forward(self, token_seqs: List[torch.Tensor]) -> torch.Tensor:
        B = token_seqs[0].shape[0]
        cls_tokens = []
        for i, q in enumerate(token_seqs):
            t = self.token_projs[i](q)
            t = torch.cat([self.local_cls[i].expand(B, -1, -1), t], dim=1)
            t = t + self.local_pos[i][:, :t.shape[1]]
            t = self.local_drop(t)
            for blk in self.local_blocks[i]:
                t = blk(t)
            cls_tokens.append(self.local_norms[i](t)[:, 0:1])

        fused = torch.cat(cls_tokens, dim=1) + self.global_pos
        for blk in self.global_blocks:
            fused = blk(fused)
        return self.head(self.global_norm(fused).mean(1))


# ============================================================================
# Strategy 5 — GatedFusion
# ============================================================================

class GatedFusion(nn.Module):
    """
    Independent per-modality transformers → K CLS tokens.
    A gating network (softmax over a linear projection of the concatenated
    CLS tokens) learns a scalar weight per modality.
    The gated weighted sum of CLS tokens is then refined by a second-stage
    transformer and passed to the head.

    Why it's more powerful
    ──────────────────────
    • Modality weights are input-dependent (adaptive, not fixed).
    • If one modality is noisy or missing, the gate can suppress it.
    • The second-stage transformer can model interactions between the
      gated representation and any residual modality information.

        modality i  →  LocalTransformer_i  →  CLS_i
        gate weight: g = softmax(Linear(cat(CLS_0,...,CLS_K)))  ∈ ℝ^K
        fused feat : f = Σ_i  g_i · CLS_i
        [f] + pos  →  SecondTransformer  →  head
    """

    def __init__(self, specs, fusion_dim, depth_local, depth_global,
                 num_heads, mlp_ratio, nb_classes,
                 drop_rate, attn_drop_rate, drop_path_rate, is_binary,
                 head_type="mlp", head_hidden_dim=None, head_dropout=0.0):
        super().__init__()
        self.specs = specs
        self.is_binary = is_binary
        K = len(specs)

        self.token_projs = nn.ModuleList([
            nn.Linear(s.code_dim, fusion_dim) for s in specs
        ])
        self.local_cls   = nn.ParameterList([
            nn.Parameter(torch.zeros(1, 1, fusion_dim)) for _ in specs
        ])
        self.local_pos   = nn.ParameterList([
            nn.Parameter(torch.zeros(1, s.n_tokens + 1, fusion_dim)) for s in specs
        ])
        self.local_drop  = nn.Dropout(p=drop_rate)
        # independent local transformers per modality
        self.local_blocks = nn.ModuleList([
            _make_blocks(fusion_dim, depth_local, num_heads, mlp_ratio,
                         drop_rate, attn_drop_rate, drop_path_rate)
            for _ in specs
        ])
        self.local_norms = nn.ModuleList([nn.LayerNorm(fusion_dim) for _ in specs])

        # gating network: input = cat(CLS_0,...,CLS_K) → K scalar weights
        self.gate = nn.Sequential(
            nn.Linear(K * fusion_dim, K * fusion_dim // 2),
            nn.GELU(),
            nn.Dropout(drop_rate),
            nn.Linear(K * fusion_dim // 2, K),
        )

        # second-stage transformer refines the fused representation
        # input is treated as a single token so we use a small MLP-mixer instead
        # of a full self-attention; for depth_global > 0 we use real blocks on
        # a sequence of 1 token (which reduces to a residual MLP effectively)
        self.post_norm = nn.LayerNorm(fusion_dim)
        self.post_blocks = _make_blocks(
            fusion_dim, depth_global, num_heads, mlp_ratio,
            drop_rate, attn_drop_rate, drop_path_rate)
        self.post_final_norm = nn.LayerNorm(fusion_dim)

        self.head = build_head(fusion_dim, 1 if is_binary else nb_classes,
                               head_type, head_hidden_dim, head_dropout)

        for p in self.local_cls:  trunc_normal_(p, std=0.02)
        for p in self.local_pos:  trunc_normal_(p, std=0.02)
        self.apply(_init_weights)

    def forward(self, token_seqs: List[torch.Tensor]) -> torch.Tensor:
        B = token_seqs[0].shape[0]
        K = len(token_seqs)
        cls_list = []

        # ── local encoding ──────────────────────────────────────────────────
        for i, q in enumerate(token_seqs):
            t = self.token_projs[i](q)
            t = torch.cat([self.local_cls[i].expand(B, -1, -1), t], dim=1)
            t = t + self.local_pos[i][:, :t.shape[1]]
            t = self.local_drop(t)
            for blk in self.local_blocks[i]:
                t = blk(t)
            cls_list.append(self.local_norms[i](t)[:, 0])        # (B, D)

        # ── adaptive gating ─────────────────────────────────────────────────
        stacked = torch.stack(cls_list, dim=1)                    # (B, K, D)
        gate_in = stacked.flatten(1)                              # (B, K*D)
        weights = self.gate(gate_in).softmax(dim=-1)              # (B, K)
        fused = (weights.unsqueeze(-1) * stacked).sum(dim=1)      # (B, D)

        # ── second-stage transformer (on the single fused token) ───────────
        fused = self.post_norm(fused).unsqueeze(1)                # (B, 1, D)
        for blk in self.post_blocks:
            fused = blk(fused)
        feat = self.post_final_norm(fused).squeeze(1)             # (B, D)

        return self.head(feat)


# ============================================================================
# Strategy 6 — PerceiverFusion
# ============================================================================

class PerceiverFusion(nn.Module):
    """
    Perceiver-IO style fusion.  A fixed set of n_latents learnable query
    vectors alternates between:
      (a) cross-attending to ALL modality tokens (reads from the data)
      (b) self-attending within the latent space (refines representations)
    This is repeated n_layers times.  The final latents are mean-pooled and
    passed to the head.

    Why it's more powerful
    ──────────────────────
    • The latent array has constant size regardless of how many modalities
      or channels there are — scales well.
    • Each latent can specialise on different cross-modal patterns.
    • Multiple rounds of (cross + self) attention allow deep interaction
      between modalities without paying the O(N²) cost of full self-attention
      over the concatenated long sequence.

        latents (n_latents, D)
        data    = cat(proj(tok_0), ..., proj(tok_K))   (B, N_total, D)
        for r in range(n_layers):
            latents = CrossAttn(Q=latents, KV=data)    ← read from data
            latents = SelfAttn(latents)                ← refine
        feat = latents.mean(1) → head
    """

    def __init__(self, specs, fusion_dim, n_latents, n_layers,
                 num_heads, mlp_ratio, nb_classes,
                 drop_rate, attn_drop_rate, drop_path_rate, is_binary,
                 head_type="mlp", head_hidden_dim=None, head_dropout=0.0):
        super().__init__()
        self.specs = specs
        self.is_binary = is_binary

        # project each modality to fusion_dim
        self.token_projs = nn.ModuleList([
            nn.Linear(s.code_dim, fusion_dim) for s in specs
        ])

        # learnable latent queries
        self.latents = nn.Parameter(torch.zeros(1, n_latents, fusion_dim))
        trunc_normal_(self.latents, std=0.02)

        # per-round cross-attention and self-attention blocks
        stoch = [x.item() for x in torch.linspace(0, drop_path_rate,
                                                   max(2 * n_layers, 1))]
        self.cross_blocks = nn.ModuleList([
            CrossAttentionBlock(fusion_dim, num_heads, mlp_ratio,
                                drop_rate, attn_drop_rate, stoch[2 * r])
            for r in range(n_layers)
        ])
        self.self_blocks = nn.ModuleList([
            Block(dim=fusion_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                  qkv_bias=True, qk_norm=partial(nn.LayerNorm, eps=1e-6),
                  drop=drop_rate, attn_drop=attn_drop_rate,
                  drop_path=stoch[2 * r + 1],
                  norm_layer=partial(nn.LayerNorm, eps=1e-6))
            for r in range(n_layers)
        ])
        self.norm = nn.LayerNorm(fusion_dim)
        self.head = build_head(fusion_dim, 1 if is_binary else nb_classes,
                               head_type, head_hidden_dim, head_dropout)
        self.apply(_init_weights)

    def forward(self, token_seqs: List[torch.Tensor]) -> torch.Tensor:
        B = token_seqs[0].shape[0]

        # build full data sequence from all modalities
        data = torch.cat(
            [self.token_projs[i](q) for i, q in enumerate(token_seqs)],
            dim=1)                                                # (B, N_total, D)

        latents = self.latents.expand(B, -1, -1)                # (B, n_latents, D)

        for cross_blk, self_blk in zip(self.cross_blocks, self.self_blocks):
            latents = cross_blk(latents, data)                  # read from data
            latents = self_blk(latents)                         # refine latents

        latents = self.norm(latents)
        feat = latents.mean(dim=1)                              # (B, D)
        return self.head(feat)


# ============================================================================
# Factory
# ============================================================================

FUSION_TYPES = ("concat", "hierarchical", "cross_attn",
                "cls_cross", "gated", "perceiver")
HEAD_TYPES   = ("linear", "mlp", "mlp_norm")


def build_classifier(args, specs: List[TokenizerSpec]) -> nn.Module:
    """Instantiate the selected fusion classifier."""
    head_kw = dict(
        head_type=args.head_type,
        head_hidden_dim=args.head_hidden_dim,
        head_dropout=args.head_dropout,
    )
    common = dict(
        specs=specs,
        fusion_dim=args.fusion_dim,
        num_heads=args.num_heads,
        mlp_ratio=args.mlp_ratio,
        nb_classes=args.nb_classes,
        drop_rate=args.drop,
        attn_drop_rate=args.attn_drop,
        drop_path_rate=args.drop_path,
        is_binary=args.is_binary,
        **head_kw,
    )

    if args.fusion_type == "concat":
        return ConcatFusion(depth=args.depth,
                            use_segment_embed=not args.no_segment_embed,
                            pool=args.pool, **common)

    if args.fusion_type == "hierarchical":
        return HierarchicalFusion(depth_local=args.depth_local,
                                  depth_global=args.depth_global, **common)

    if args.fusion_type == "cross_attn":
        return CrossAttentionFusion(depth=args.depth, pool=args.pool, **common)

    if args.fusion_type == "cls_cross":
        return CLSCrossFusion(depth_local=args.depth_local,
                              depth_global=args.depth_global, **common)

    if args.fusion_type == "gated":
        return GatedFusion(depth_local=args.depth_local,
                           depth_global=args.depth_global, **common)

    if args.fusion_type == "perceiver":
        return PerceiverFusion(n_latents=args.n_latents,
                               n_layers=args.n_layers, **common)

    raise ValueError(f"Unknown fusion_type: {args.fusion_type}")


# ============================================================================
# Top-level system (tokenizers + classifier)
# ============================================================================

class MultiTokenizerSystem(nn.Module):
    """Owns both the K VQNSP tokenizers and the fusion classifier."""

    def __init__(self, specs: List[TokenizerSpec],
                 classifier: nn.Module, device: str):
        super().__init__()
        self.specs = specs
        self.classifier = classifier
        self.tokenizers = nn.ModuleList([_load_vqnsp(s, device) for s in specs])

    def encode_all(self, x: torch.Tensor) -> List[torch.Tensor]:
        """x : (B, total_C, n_win, T) → list of (B, n_i*n_win, code_dim_i)"""
        out = []
        for i, spec in enumerate(self.specs):
            x_i = x[:, spec.channel_indices, :, :]
            ctx = torch.no_grad() if spec.freeze else contextlib.nullcontext()
            with ctx:
                q, _, _ = self.tokenizers[i].encode(x_i)
            out.append(rearrange(q, 'b d h w -> b (h w) d'))
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.encode_all(x))

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.forward(x)
        return (torch.sigmoid(logits) if self.classifier.is_binary
                else logits.softmax(-1))


# ============================================================================
# Dataset
# ============================================================================

class MultiChannelHDF5Dataset(Dataset):
    """
    Load multi-channel data from one or more HDF5 files.
    Multiple files are concatenated along the channel axis.
    Supported shapes: (N,C,T_total) or (N,C,n_win,T).
    """

    _DATA_KEYS  = ("eeg", "data", "x", "signal")
    _LABEL_KEYS = ("label", "labels", "y", "target")

    def __init__(self, data_paths, data_keys=None, label_key="label",
                 eeg_size=1600, patch_size=200):
        super().__init__()
        self.n_win = eeg_size // patch_size
        self.patch_size = patch_size
        key_list = data_keys or [None] * len(data_paths)
        arrays, labels = [], None

        for path, hint in zip(data_paths, key_list):
            with h5py.File(path, "r") as f:
                if hint and hint in f:
                    arr = f[hint][:]
                else:
                    for k in self._DATA_KEYS:
                        if k in f:
                            arr = f[k][:]; break
                    else:
                        raise KeyError(
                            f"No data key in '{path}'. Keys: {list(f.keys())}")
                arr = arr.astype(np.float32)
                if arr.ndim == 3:
                    N, C, T = arr.shape
                    arr = arr.reshape(N, C, self.n_win, patch_size)
                arrays.append(arr)
                if labels is None:
                    for lk in (label_key,) + self._LABEL_KEYS:
                        if lk in f:
                            labels = f[lk][:].astype(np.int64); break

        self.data = (np.concatenate(arrays, axis=1)
                     if len(arrays) > 1 else arrays[0])
        self.labels = labels

    def __len__(self): return len(self.data)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.data[idx])
        y = (torch.tensor(self.labels[idx], dtype=torch.long)
             if self.labels is not None
             else torch.tensor(-1, dtype=torch.long))
        return x, y


# ============================================================================
# Training utilities
# ============================================================================

def build_optimizer(system, lr, weight_decay, unfreeze_tokenizers, tok_lr_scale):
    groups = [{"params": list(system.classifier.parameters()),
               "lr": lr, "weight_decay": weight_decay}]
    if unfreeze_tokenizers:
        tok_params = [p for p in system.tokenizers.parameters()
                      if p.requires_grad]
        if tok_params:
            groups.append({"params": tok_params,
                           "lr": lr * tok_lr_scale,
                           "weight_decay": weight_decay})
    return torch.optim.AdamW(groups)


def cosine_schedule(epoch, warmup, total, min_ratio=0.01):
    if epoch < warmup:
        return (epoch + 1) / max(1, warmup)
    t = (epoch - warmup) / max(1, total - warmup)
    return min_ratio + 0.5 * (1 - min_ratio) * (1 + math.cos(math.pi * t))


def _acc(logits, targets, is_binary):
    preds = ((logits.squeeze(-1).sigmoid() > 0.5).long()
             if is_binary else logits.argmax(-1))
    return (preds == targets).float().mean().item()


def train_one_epoch(system, loader, optimizer, scaler, device, label_smoothing):
    system.train()
    for i, spec in enumerate(system.specs):
        if spec.freeze:
            system.tokenizers[i].eval()
    total_loss = total_acc = n = 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=(device != "cpu")):
            logits = system(x)
            loss = (F.binary_cross_entropy_with_logits(logits.squeeze(-1), y.float())
                    if system.classifier.is_binary
                    else F.cross_entropy(logits, y, label_smoothing=label_smoothing))
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(system.parameters(), 1.0)
        scaler.step(optimizer); scaler.update()
        total_loss += loss.item()
        total_acc  += _acc(logits.detach(), y, system.classifier.is_binary)
        n += 1
    return {"loss": total_loss / max(n, 1), "acc": total_acc / max(n, 1)}


@torch.no_grad()
def evaluate(system, loader, device):
    system.eval()
    total_loss = total_acc = n = 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        logits = system(x)
        loss = (F.binary_cross_entropy_with_logits(logits.squeeze(-1), y.float())
                if system.classifier.is_binary
                else F.cross_entropy(logits, y))
        total_loss += loss.item()
        total_acc  += _acc(logits, y, system.classifier.is_binary)
        n += 1
    return {"loss": total_loss / max(n, 1), "acc": total_acc / max(n, 1)}


# ============================================================================
# CLI
# ============================================================================

def get_args():
    p = argparse.ArgumentParser(
        description="Multi-tokenizer fusion classifier.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--eval", action="store_true")

    # tokenizer config
    p.add_argument("--tokenizer_cfg", required=True)

    # data
    p.add_argument("--data_train", nargs="+", default=[])
    p.add_argument("--data_val",   nargs="+", default=[])
    p.add_argument("--data_keys",  nargs="+", default=None)
    p.add_argument("--label_key",  default="label")
    p.add_argument("--eeg_size",   type=int, default=1600)
    p.add_argument("--patch_size", type=int, default=200)

    # fusion strategy
    p.add_argument("--fusion_type", default="concat", choices=FUSION_TYPES,
                   help=(
                       "concat: full self-attn over all tokens | "
                       "hierarchical: local→global CLS | "
                       "cross_attn: EEG(Q) × others(KV) | "
                       "cls_cross: independent encoders, fuse CLS | "
                       "gated: adaptive modality weighting | "
                       "perceiver: latent array cross-attends to all tokens"
                   ))

    # shared transformer hyper-params
    p.add_argument("--fusion_dim",  type=int,   default=256)
    p.add_argument("--num_heads",   type=int,   default=8)
    p.add_argument("--mlp_ratio",   type=float, default=4.0)
    p.add_argument("--drop",        type=float, default=0.0)
    p.add_argument("--attn_drop",   type=float, default=0.0)
    p.add_argument("--drop_path",   type=float, default=0.1)

    # depth (strategy-dependent)
    p.add_argument("--depth",        type=int, default=6,
                   help="Total depth (concat, cross_attn).")
    p.add_argument("--depth_local",  type=int, default=3,
                   help="Per-modality local depth (hierarchical, cls_cross, gated).")
    p.add_argument("--depth_global", type=int, default=3,
                   help="Global fusion depth (hierarchical, cls_cross, gated).")

    # perceiver-specific
    p.add_argument("--n_latents", type=int, default=64,
                   help="[perceiver] Number of latent query vectors.")
    p.add_argument("--n_layers",  type=int, default=6,
                   help="[perceiver] Number of (cross+self) attention rounds.")

    # concat / cross_attn only
    p.add_argument("--pool",             default="cls", choices=["cls", "mean"])
    p.add_argument("--no_segment_embed", action="store_true")

    # head
    p.add_argument("--head_type",       default="mlp", choices=HEAD_TYPES,
                   help="Classification head type.")
    p.add_argument("--head_hidden_dim", type=int, default=None,
                   help="Hidden dim for mlp/mlp_norm head (default: fusion_dim).")
    p.add_argument("--head_dropout",    type=float, default=0.0,
                   help="Dropout inside the classification head.")

    # classification
    p.add_argument("--nb_classes", type=int, default=6)
    p.add_argument("--is_binary",  action="store_true")

    # tokenizer fine-tuning
    p.add_argument("--unfreeze_tokenizers", action="store_true")
    p.add_argument("--tok_lr_scale",        type=float, default=0.1)

    # training
    p.add_argument("--epochs",          type=int,   default=50)
    p.add_argument("--batch_size",      type=int,   default=32)
    p.add_argument("--lr",              type=float, default=1e-3)
    p.add_argument("--weight_decay",    type=float, default=0.05)
    p.add_argument("--warmup_epochs",   type=int,   default=5)
    p.add_argument("--label_smoothing", type=float, default=0.1)
    p.add_argument("--num_workers",     type=int,   default=4)

    # I/O
    p.add_argument("--output_dir",  default="multi_tok_out")
    p.add_argument("--checkpoint",  default="")
    p.add_argument("--device",      default="cuda")
    p.add_argument("--seed",        type=int, default=42)

    return p.parse_args()


# ============================================================================
# main
# ============================================================================

def main():
    args = get_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        args.device = "cpu"
    device = args.device

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    specs = load_specs(args.tokenizer_cfg)
    if args.unfreeze_tokenizers:
        for s in specs:
            s.freeze = False

    print(f"\n[MultiTok] fusion={args.fusion_type}  head={args.head_type}  "
          f"| {len(specs)} tokenizer(s):")
    for s in specs:
        ch = (f"ch{s.channel_indices[0]}‥{s.channel_indices[-1]}"
              if len(s.channel_indices) > 1 else f"ch{s.channel_indices[0]}")
        print(f"  [{s.name or s.model}]  {ch}  "
              f"n_tokens={s.n_tokens}  code_dim={s.code_dim}  freeze={s.freeze}")

    classifier = build_classifier(args, specs)
    system     = MultiTokenizerSystem(specs, classifier, device).to(device)

    n_cls = sum(p.numel() for p in classifier.parameters())
    print(f"[MultiTok] Fusion classifier: {n_cls/1e6:.2f}M params")

    best_acc, start_epoch = 0.0, 0
    if args.checkpoint:
        raw = torch.load(args.checkpoint, map_location=device)
        system.load_state_dict(raw.get("model", raw), strict=False)
        best_acc    = raw.get("best_acc", 0.0)
        start_epoch = raw.get("epoch", -1) + 1
        print(f"[MultiTok] Resumed epoch {start_epoch}, best_acc={best_acc:.4f}")

    if args.eval:
        if not args.data_val:
            sys.exit("[error] --data_val required for --eval")
        val_ds = MultiChannelHDF5Dataset(
            args.data_val, args.data_keys, args.label_key,
            args.eeg_size, args.patch_size)
        val_loader = DataLoader(val_ds, args.batch_size, shuffle=False,
                                num_workers=args.num_workers,
                                pin_memory=(device != "cpu"))
        m = evaluate(system, val_loader, device)
        print(f"\n[eval]  loss={m['loss']:.4f}  acc={m['acc']:.4f}")
        return

    if not args.data_train:
        sys.exit("[error] --data_train required")

    train_ds = MultiChannelHDF5Dataset(
        args.data_train, args.data_keys, args.label_key,
        args.eeg_size, args.patch_size)
    val_ds = (MultiChannelHDF5Dataset(
        args.data_val, args.data_keys, args.label_key,
        args.eeg_size, args.patch_size) if args.data_val else None)

    train_loader = DataLoader(train_ds, args.batch_size, shuffle=True,
                              num_workers=args.num_workers,
                              pin_memory=(device != "cpu"), drop_last=True)
    val_loader = (DataLoader(val_ds, args.batch_size, shuffle=False,
                             num_workers=args.num_workers,
                             pin_memory=(device != "cpu")) if val_ds else None)

    print(f"[MultiTok] Train {len(train_ds)} | Val {len(val_ds) if val_ds else 0}\n")

    optimizer = build_optimizer(system, args.lr, args.weight_decay,
                                args.unfreeze_tokenizers, args.tok_lr_scale)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda ep: cosine_schedule(ep, args.warmup_epochs, args.epochs))
    scaler = torch.cuda.amp.GradScaler(enabled=(device != "cpu"))
    log_path = output_dir / "train_log.jsonl"

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        tr = train_one_epoch(system, train_loader, optimizer, scaler,
                             device, args.label_smoothing)
        scheduler.step()

        log = {"epoch": epoch, "fusion_type": args.fusion_type,
               "head_type": args.head_type,
               "train_loss": round(tr["loss"], 6),
               "train_acc":  round(tr["acc"],  6),
               "lr": optimizer.param_groups[0]["lr"]}
        val_str = ""

        if val_loader:
            val = evaluate(system, val_loader, device)
            log["val_loss"] = round(val["loss"], 6)
            log["val_acc"]  = round(val["acc"],  6)
            if val["acc"] > best_acc:
                best_acc = val["acc"]
                torch.save({"model": system.state_dict(), "epoch": epoch,
                            "best_acc": best_acc, "args": vars(args)},
                           output_dir / "checkpoint_best.pth")
            val_str = (f"  val_loss={val['loss']:.4f}"
                       f"  val_acc={val['acc']:.4f}"
                       f"  ★best={best_acc:.4f}")

        print(f"Epoch [{epoch:3d}/{args.epochs}]"
              f"  train_loss={tr['loss']:.4f}"
              f"  train_acc={tr['acc']:.4f}"
              f"{val_str}  ({time.time()-t0:.1f}s)")

        with open(log_path, "a") as fh:
            fh.write(json.dumps(log) + "\n")

        torch.save({"model": system.state_dict(), "epoch": epoch,
                    "best_acc": best_acc, "args": vars(args)},
                   output_dir / "checkpoint_last.pth")

    print(f"\n[MultiTok] Done. Best val acc: {best_acc:.4f}")


if __name__ == "__main__":
    main()
