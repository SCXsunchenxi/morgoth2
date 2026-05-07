"""
model_scaling.py
================
Expand a trained EEG model to a larger scale without starting from scratch.

Three methods are implemented for each of the three model types
(tokenizer, transformer, classifier):

  Method A — Depth expansion
      Insert identity-like transformer blocks between (or after) existing blocks.
      New blocks have gamma_1 / gamma_2 set to 1e-6 so they are near-identity at
      first forward pass.  When gamma is unavailable (init_values=0) the attn.proj
      and mlp.fc2 weights are zeroed instead.

  Method B — Width expansion (Net2WiderNet-style)
      Double embed_dim by duplicating attention heads (H → 2H, head_dim unchanged).
      This is function-preserving at expansion time: the duplicated head pairs
      produce the same activations, and each duplicate is scaled by 0.5 so the
      projection sum is identical to the original.
      FFN hidden dim, pos/time embeddings, norm layers, and all heads are also
      expanded to match the new embed_dim.

  Method C — Weight inheritance
      Instantiate the target architecture (e.g. large or huge) from scratch and
      copy every state-dict key that matches by name and shape.
      Remaining (new) parameters keep their random initialisation.
      This is the least principled but works for any source → target pair.

Usage
-----
  python model_scaling.py \\
      --model_type  classifier \\
      --method      A \\
      --checkpoint  checkpoints/finetune/checkpoint_best.pth \\
      --model       base_patch200_200 \\
      --nb_classes  6 \\
      --new_depth   24 \\
      --output      checkpoints/finetune/expanded_depthA.pth

  python model_scaling.py \\
      --model_type  transformer \\
      --method      C \\
      --checkpoint  checkpoints/eegfounder/checkpoint.pth \\
      --model       base_patch200_1600_8k_vocab \\
      --target_model large_patch200_1600_8k_vocab \\
      --vocab_size  8192 \\
      --output      checkpoints/eegfounder/inherited_large.pth

See run_model_scaling.sh for ready-to-use shell commands.
"""

from __future__ import annotations

import argparse
import copy
import math
import sys
from functools import partial
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from timm.models import create_model

# ---------------------------------------------------------------------------
# Local imports
# ---------------------------------------------------------------------------
import backbone  # noqa: F401 — registers backbone models
import tokenizer as tok_module  # noqa: F401 — registers tokenizer models
from backbone import (
    NeuralTransformer,
    NeuralTransformerForMEM,
    NeuralTransformerForMaskedEEGModeling,
    Block,
    Attention,
    Mlp,
)
from tokenizer import VQNSP


# ===========================================================================
# Utilities
# ===========================================================================

def _load_checkpoint(path: str, device: str = "cpu") -> dict:
    """Load a checkpoint and return the raw state dict (strips 'model' wrapper)."""
    ckpt = torch.load(path, map_location=device)
    state = ckpt.get("model", ckpt.get("state_dict", ckpt))
    # strip DDP / DataParallel prefix
    state = {k.replace("module.", ""): v for k, v in state.items()}
    return state


def _save_checkpoint(state_dict: dict, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": state_dict}, path)
    print(f"[model_scaling] Saved expanded checkpoint → {path}")


def _count_params(model: nn.Module) -> str:
    n = sum(p.numel() for p in model.parameters())
    if n >= 1e9:
        return f"{n/1e9:.2f}B"
    return f"{n/1e6:.1f}M"


def _verify_output(src: nn.Module, dst: nn.Module, dummy: torch.Tensor,
                   input_chans=None, rtol: float = 1e-3):
    """Check that src and dst produce identical (or near-identical) outputs."""
    src.eval(); dst.eval()
    with torch.no_grad():
        if input_chans is not None:
            out_s = src(dummy, input_chans=input_chans)
            out_d = dst(dummy, input_chans=input_chans)
        else:
            out_s = src(dummy)
            out_d = dst(dummy)
    if isinstance(out_s, (tuple, list)):
        out_s, out_d = out_s[0], out_d[0]
    max_err = (out_s - out_d).abs().max().item()
    ok = max_err < rtol
    status = "PASS" if ok else "WARN"
    print(f"[verify] max |src−dst| = {max_err:.2e}  [{status}]")
    return ok


# ===========================================================================
# Method A — Depth expansion
# ===========================================================================

def _make_identity_block(ref_block: Block) -> Block:
    """
    Deep-copy a Block and set it up as a near-identity residual unit.

    If the block uses LayerScale (gamma_1 / gamma_2), those scalars are set
    to 1e-6, making the residual contribution negligible.  Otherwise the
    projection and MLP output weights are zeroed.
    """
    new_blk = copy.deepcopy(ref_block)
    if new_blk.gamma_1 is not None:
        with torch.no_grad():
            new_blk.gamma_1.fill_(1e-6)
            new_blk.gamma_2.fill_(1e-6)
    else:
        with torch.no_grad():
            new_blk.attn.proj.weight.zero_()
            if new_blk.attn.proj.bias is not None:
                new_blk.attn.proj.bias.zero_()
            new_blk.mlp.fc2.weight.zero_()
            if new_blk.mlp.fc2.bias is not None:
                new_blk.mlp.fc2.bias.zero_()
    return new_blk


def expand_depth(transformer: NeuralTransformer, new_depth: int) -> NeuralTransformer:
    """
    Expand a NeuralTransformer from depth D to new_depth by inserting identity
    blocks.  The original blocks are distributed evenly; new blocks fill the gaps.

    Returns a new NeuralTransformer object (the original is unchanged).
    """
    old_depth = len(transformer.blocks)
    if new_depth <= old_depth:
        raise ValueError(f"new_depth={new_depth} must be > current depth={old_depth}")

    # Build a mapping: which positions in [0, new_depth) get an original block
    # Strategy: copy block i to position round(i * new_depth / old_depth)
    positions = [round(i * new_depth / old_depth) for i in range(old_depth)]
    # resolve collisions by shifting forward
    for i in range(1, len(positions)):
        if positions[i] <= positions[i - 1]:
            positions[i] = positions[i - 1] + 1

    new_blocks = [None] * new_depth
    for orig_idx, pos in enumerate(positions):
        new_blocks[pos] = copy.deepcopy(transformer.blocks[orig_idx])

    # Fill gaps with identity copies of the nearest preceding block
    ref = transformer.blocks[-1]
    for i in range(new_depth):
        if new_blocks[i] is None:
            # find nearest original block before position i
            for j in range(i - 1, -1, -1):
                if new_blocks[j] is not None:
                    ref = new_blocks[j]
                    break
            new_blocks[i] = _make_identity_block(ref)

    # Create new model with the same config but new depth
    new_tf = copy.deepcopy(transformer)
    new_tf.blocks = nn.ModuleList(new_blocks)
    return new_tf


# ===========================================================================
# Method B — Width expansion (Net2WiderNet)
# ===========================================================================

def _expand_linear(
    old: nn.Linear,
    new_in: Optional[int],
    new_out: Optional[int],
    out_scale: float = 1.0,
) -> nn.Linear:
    """
    Expand a Linear layer from (old_in, old_out) to (new_in, new_out).

    If new_in > old_in, the extra input weights are copied from the first
    (new_in - old_in) columns (allows averaging later).
    If new_out > old_out, duplicate the first (new_out - old_out) rows
    and scale them by out_scale (used for function-preserving head duplication).
    """
    old_out, old_in = old.weight.shape
    ni = new_in if new_in is not None else old_in
    no = new_out if new_out is not None else old_out

    new_lin = nn.Linear(ni, no, bias=old.bias is not None)
    with torch.no_grad():
        # ── output dimension expansion ───────────────────────────────────────
        if no > old_out:
            extra = no - old_out
            W = torch.cat([old.weight.data,
                           old.weight.data[:extra] * out_scale], dim=0)
            new_lin.weight.data[:, :old_in] = W
            if old.bias is not None:
                b = torch.cat([old.bias.data,
                               old.bias.data[:extra] * out_scale])
                new_lin.bias.data = b
        else:
            new_lin.weight.data[:, :old_in] = old.weight.data
            if old.bias is not None:
                new_lin.bias.data = old.bias.data.clone()

        # ── input dimension expansion ────────────────────────────────────────
        if ni > old_in:
            extra = ni - old_in
            new_lin.weight.data[:, old_in:] = new_lin.weight.data[:, :extra]

    return new_lin


def _expand_layernorm(old: nn.LayerNorm, new_dim: int) -> nn.LayerNorm:
    new_ln = nn.LayerNorm(new_dim, eps=old.eps,
                          elementwise_affine=old.elementwise_affine)
    if old.elementwise_affine:
        with torch.no_grad():
            new_ln.weight.data[:old.normalized_shape[0]] = old.weight.data
            new_ln.bias.data[:old.normalized_shape[0]] = old.bias.data
    return new_ln


def _expand_attention(old_attn: Attention, new_dim: int) -> Attention:
    """
    Expand an Attention module from embed_dim D to new_dim 2D by duplicating
    each attention head.  head_dim is kept constant; num_heads doubles.
    The duplicate proj weights are scaled by 0.5 so the projection is
    function-preserving at expansion time.
    """
    old_dim = old_attn.proj.in_features  # = num_heads * head_dim
    head_dim = old_dim // old_attn.num_heads
    new_heads = (new_dim // head_dim)

    from backbone import Attention as _Attn
    new_attn = _Attn(
        dim=new_dim,
        num_heads=new_heads,
        qkv_bias=(old_attn.q_bias is not None),
        qk_scale=old_attn.scale * math.sqrt(head_dim),   # reconstruct original
        attn_drop=old_attn.attn_drop.p,
        proj_drop=old_attn.proj_drop.p,
        attn_head_dim=head_dim,
    )

    old_H = old_attn.num_heads
    with torch.no_grad():
        # qkv weight: shape (3*H*head_dim, D)  →  (3*2H*head_dim, 2D)
        # Duplicate each head block along output and input dims
        old_W_qkv = old_attn.qkv.weight.data   # (3*old_H*hd, D)
        # reshape to (3, H, hd, D)
        W = old_W_qkv.view(3, old_H, head_dim, old_dim)
        # duplicate heads → (3, 2H, hd, 2D_input?)
        # input dim: we duplicate full input block by stacking [W, W] along input dim
        # then duplicate heads
        W_dup = torch.cat([W, W], dim=1)  # (3, 2H, hd, D)
        # For input: the new input is [x_orig | x_orig_dup]
        # We set weights so output head i uses first D dims, head i+H uses last D dims
        # but since both halves are identical copies of input, we simply set:
        # W_new[h, :, :D] = W[h,:,:]   W_new[h+H,:, D:] = W[h,:,:]  rest zero
        new_H = new_heads
        new_W_qkv = torch.zeros(3, new_H, head_dim, new_dim)
        new_W_qkv[:, :old_H, :, :old_dim] = W
        new_W_qkv[:, old_H:, :, old_dim:] = W
        new_attn.qkv.weight.data = new_W_qkv.view(3 * new_H * head_dim, new_dim)

        # qkv bias
        if old_attn.q_bias is not None:
            q_b = old_attn.q_bias.data.view(old_H, head_dim)
            v_b = old_attn.v_bias.data.view(old_H, head_dim)
            new_attn.q_bias.data = torch.cat([q_b, q_b], dim=0).view(-1)
            new_attn.v_bias.data = torch.cat([v_b, v_b], dim=0).view(-1)

        # proj weight: shape (D, D)  →  (2D, 2D)
        # old proj maps (H * hd) → D
        # new proj maps (2H * hd) → 2D
        # Each pair of heads maps to the same output, so we split output into halves
        old_W_proj = old_attn.proj.weight.data  # (D, D)
        # reshape: (D, H, hd)
        Wp = old_W_proj.view(old_dim, old_H, head_dim)
        # Scale by 0.5 because two heads now contribute identically
        Wp_s = Wp * 0.5
        new_W_proj = torch.zeros(new_dim, new_dim)
        # first half of output (first D rows) ← first H heads (first D cols) + second H heads (last D cols)
        new_W_proj[:old_dim, :old_dim] = Wp_s.view(old_dim, old_dim)
        new_W_proj[:old_dim, old_dim:] = Wp_s.view(old_dim, old_dim)
        # second half: copy
        new_W_proj[old_dim:, :old_dim] = Wp_s.view(old_dim, old_dim)
        new_W_proj[old_dim:, old_dim:] = Wp_s.view(old_dim, old_dim)
        new_attn.proj.weight.data = new_W_proj

        if old_attn.proj.bias is not None:
            b = old_attn.proj.bias.data
            new_attn.proj.bias.data = torch.cat([b, b])

    return new_attn


def _expand_mlp(old_mlp: Mlp, new_dim: int) -> Mlp:
    """Expand MLP from D to 2D.  Hidden dim is kept at 4x the new dim."""
    old_dim = old_mlp.fc1.in_features
    old_hid = old_mlp.fc1.out_features
    new_hid = int(new_dim * old_hid / old_dim)

    new_mlp = Mlp(in_features=new_dim, hidden_features=new_hid)
    with torch.no_grad():
        # fc1: (old_hid, old_dim) → (new_hid, new_dim)
        # Duplicate rows and cols independently
        old_W1 = old_mlp.fc1.weight.data   # (old_hid, old_dim)
        # expand cols (input): stack [W | W]
        W1_wide = torch.cat([old_W1, old_W1], dim=1)  # (old_hid, new_dim)
        W1_wide = W1_wide * 0.5  # compensate for doubled input
        # expand rows (hidden): duplicate
        W1_tall = torch.cat([W1_wide, W1_wide], dim=0)  # (new_hid, new_dim)
        new_mlp.fc1.weight.data = W1_tall
        b1 = old_mlp.fc1.bias.data
        new_mlp.fc1.bias.data = torch.cat([b1, b1])

        # fc2: (old_dim, old_hid) → (new_dim, new_hid)
        old_W2 = old_mlp.fc2.weight.data   # (old_dim, old_hid)
        # expand cols (hidden input): duplicate
        W2_wide = torch.cat([old_W2, old_W2], dim=1)  # (old_dim, new_hid)
        W2_wide = W2_wide * 0.5  # compensate
        # expand rows (output): duplicate
        W2_tall = torch.cat([W2_wide, W2_wide], dim=0)  # (new_dim, new_hid)
        new_mlp.fc2.weight.data = W2_tall
        b2 = old_mlp.fc2.bias.data
        new_mlp.fc2.bias.data = torch.cat([b2, b2])

    return new_mlp


def _expand_block(old_blk: Block, new_dim: int) -> Block:
    """Expand a single Block from D to new_dim."""
    new_blk = copy.deepcopy(old_blk)
    new_blk.norm1 = _expand_layernorm(old_blk.norm1, new_dim)
    new_blk.norm2 = _expand_layernorm(old_blk.norm2, new_dim)
    new_blk.attn = _expand_attention(old_blk.attn, new_dim)
    new_blk.mlp = _expand_mlp(old_blk.mlp, new_dim)
    if old_blk.gamma_1 is not None:
        new_blk.gamma_1 = nn.Parameter(
            torch.cat([old_blk.gamma_1.data, old_blk.gamma_1.data]))
        new_blk.gamma_2 = nn.Parameter(
            torch.cat([old_blk.gamma_2.data, old_blk.gamma_2.data]))
    return new_blk


def expand_width(transformer: NeuralTransformer, new_embed_dim: int) -> NeuralTransformer:
    """
    Expand a NeuralTransformer to new_embed_dim via Net2WiderNet-style head
    duplication.  Only doubling (new_embed_dim == 2 * old) is supported.

    Returns a new NeuralTransformer; the original is unchanged.
    """
    old_dim = transformer.embed_dim
    if new_embed_dim != 2 * old_dim:
        raise ValueError(
            f"Width expansion currently only supports doubling "
            f"({old_dim} → {new_embed_dim} requested, expected {2*old_dim})"
        )

    new_tf = copy.deepcopy(transformer)
    new_tf.embed_dim = new_embed_dim
    new_tf.num_features = new_embed_dim

    # ── cls_token, pos_embed, time_embed ────────────────────────────────────
    with torch.no_grad():
        new_tf.cls_token = nn.Parameter(
            torch.cat([transformer.cls_token.data,
                       transformer.cls_token.data], dim=-1))
        if transformer.pos_embed is not None:
            new_tf.pos_embed = nn.Parameter(
                torch.cat([transformer.pos_embed.data,
                           transformer.pos_embed.data], dim=-1))
        if transformer.time_embed is not None:
            new_tf.time_embed = nn.Parameter(
                torch.cat([transformer.time_embed.data,
                           transformer.time_embed.data], dim=-1))

    # ── blocks ───────────────────────────────────────────────────────────────
    new_tf.blocks = nn.ModuleList([
        _expand_block(blk, new_embed_dim) for blk in transformer.blocks
    ])

    # ── norm layers ──────────────────────────────────────────────────────────
    if isinstance(transformer.norm, nn.LayerNorm):
        new_tf.norm = _expand_layernorm(transformer.norm, new_embed_dim)
    if transformer.fc_norm is not None:
        new_tf.fc_norm = _expand_layernorm(transformer.fc_norm, new_embed_dim)

    # ── classification head ──────────────────────────────────────────────────
    if isinstance(transformer.head, nn.Linear):
        new_tf.head = _expand_linear(transformer.head, new_in=new_embed_dim, new_out=None)

    return new_tf


# ===========================================================================
# Method C — Weight inheritance
# ===========================================================================

def weight_inherit(source: nn.Module, target: nn.Module) -> nn.Module:
    """
    Copy every state-dict key from source to target where names AND shapes match.
    All unmatched target parameters keep their (random) initialisation.
    """
    src_sd = source.state_dict()
    tgt_sd = target.state_dict()

    copied, skipped = 0, 0
    for k, v in src_sd.items():
        if k in tgt_sd and tgt_sd[k].shape == v.shape:
            tgt_sd[k] = v.clone()
            copied += 1
        else:
            skipped += 1

    target.load_state_dict(tgt_sd, strict=False)
    print(f"[weight_inherit] copied {copied} tensors, skipped {skipped} (shape mismatch / not in target)")
    return target


# ===========================================================================
# Model-type loaders
# ===========================================================================

def _load_tokenizer(args) -> VQNSP:
    """Create and load a VQNSP tokenizer."""
    model = create_model(
        args.model,
        n_code=args.n_embed,
        code_dim=args.embed_dim,
        EEG_size=args.eeg_size,
    )
    state = _load_checkpoint(args.checkpoint)
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


def _load_transformer(args) -> NeuralTransformerForMEM:
    """Create and load a pre-trained NeuralTransformerForMEM."""
    model = create_model(
        args.model,
        vocab_size=args.vocab_size,
    )
    state = _load_checkpoint(args.checkpoint)
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


def _load_classifier(args) -> NeuralTransformer:
    """Create and load a fine-tuned NeuralTransformer classifier."""
    model = create_model(
        args.model,
        num_classes=args.nb_classes,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
    )
    state = _load_checkpoint(args.checkpoint)
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


# ===========================================================================
# Per-model-type scaling routines
# ===========================================================================

# ── Tokenizer ────────────────────────────────────────────────────────────────

def scale_tokenizer_A(args):
    """Depth-expand the VQNSP encoder."""
    model = _load_tokenizer(args)
    print(f"[tokenizer A] source encoder: {_count_params(model.encoder)} params, "
          f"depth={len(model.encoder.blocks)}")

    model.encoder = expand_depth(model.encoder, args.new_depth)
    print(f"[tokenizer A] expanded encoder: {_count_params(model.encoder)} params, "
          f"depth={len(model.encoder.blocks)}")
    return model


def scale_tokenizer_B(args):
    """Width-expand the VQNSP encoder (doubles embed_dim)."""
    model = _load_tokenizer(args)
    old_dim = model.encoder.embed_dim
    new_dim = old_dim * 2
    print(f"[tokenizer B] source encoder: {_count_params(model.encoder)} params, dim={old_dim}")

    new_encoder = expand_width(model.encoder, new_dim)
    # Re-wire encode_task_layer input dim
    enc_dim = new_dim
    with torch.no_grad():
        old_etl = model.encode_task_layer
        # Sequential: Linear(old_dim, old_dim) → Tanh → Linear(old_dim, embed_dim)
        new_fc1 = _expand_linear(old_etl[0], new_in=enc_dim, new_out=enc_dim)
        new_fc2 = _expand_linear(old_etl[2], new_in=enc_dim, new_out=None)
        model.encode_task_layer = nn.Sequential(new_fc1, nn.Tanh(), new_fc2)

    model.encoder = new_encoder
    print(f"[tokenizer B] expanded encoder: {_count_params(model.encoder)} params, dim={new_dim}")
    return model


def scale_tokenizer_C(args):
    """Inherit tokenizer encoder weights into a larger architecture."""
    model_src = _load_tokenizer(args)

    # Build target tokenizer (same codebook config, larger encoder)
    model_dst = create_model(
        args.target_model,
        n_code=args.n_embed,
        code_dim=args.embed_dim,
        EEG_size=args.eeg_size,
    )
    print(f"[tokenizer C] source: {_count_params(model_src.encoder)} → "
          f"target: {_count_params(model_dst.encoder)}")
    weight_inherit(model_src, model_dst)
    return model_dst


# ── Transformer (pre-trained backbone) ───────────────────────────────────────

def scale_transformer_A(args):
    """Depth-expand the NeuralTransformerForMEM student backbone."""
    model = _load_transformer(args)
    student = model.student
    print(f"[transformer A] source: {_count_params(student)} params, "
          f"depth={len(student.blocks)}")

    model.student = expand_depth(student, args.new_depth)
    print(f"[transformer A] expanded: {_count_params(model.student)} params, "
          f"depth={len(model.student.blocks)}")
    return model


def scale_transformer_B(args):
    """Width-expand the NeuralTransformerForMEM student backbone."""
    model = _load_transformer(args)
    student = model.student
    old_dim = student.embed_dim
    new_dim = old_dim * 2
    print(f"[transformer B] source: {_count_params(student)} params, dim={old_dim}")

    new_student = expand_width(student, new_dim)
    # norm in student is a plain LayerNorm
    # lm_head: (embed_dim, vocab_size) → expand input
    new_lm = _expand_linear(model.student.lm_head, new_in=new_dim, new_out=None)
    new_student.lm_head = new_lm

    # projection_head in NeuralTransformerForMEM (only on outer model, not student)
    old_ph = model.projection_head
    new_ph_fc = _expand_linear(old_ph[0], new_in=new_dim, new_out=new_dim)
    model.projection_head = nn.Sequential(new_ph_fc, nn.ReLU())

    # outer lm_head
    new_outer_lm = _expand_linear(model.lm_head, new_in=new_dim, new_out=None)
    model.lm_head = new_outer_lm
    model.student = new_student

    print(f"[transformer B] expanded: {_count_params(model.student)} params, dim={new_dim}")
    return model


def scale_transformer_C(args):
    """Inherit pre-trained transformer weights into a larger architecture."""
    model_src = _load_transformer(args)
    model_dst = create_model(args.target_model, vocab_size=args.vocab_size)
    print(f"[transformer C] source: {_count_params(model_src.student)} → "
          f"target: {_count_params(model_dst.student)}")
    weight_inherit(model_src, model_dst)
    return model_dst


# ── Classifier ───────────────────────────────────────────────────────────────

def scale_classifier_A(args):
    """Depth-expand the NeuralTransformer classifier backbone."""
    model = _load_classifier(args)
    print(f"[classifier A] source: {_count_params(model)} params, "
          f"depth={len(model.blocks)}")

    expanded = expand_depth(model, args.new_depth)
    print(f"[classifier A] expanded: {_count_params(expanded)} params, "
          f"depth={len(expanded.blocks)}")
    return expanded


def scale_classifier_B(args):
    """Width-expand the NeuralTransformer classifier backbone."""
    model = _load_classifier(args)
    old_dim = model.embed_dim
    new_dim = old_dim * 2
    print(f"[classifier B] source: {_count_params(model)} params, dim={old_dim}")

    expanded = expand_width(model, new_dim)
    print(f"[classifier B] expanded: {_count_params(expanded)} params, dim={new_dim}")
    return expanded


def scale_classifier_C(args):
    """Inherit classifier weights into a larger architecture."""
    model_src = _load_classifier(args)
    model_dst = create_model(
        args.target_model,
        num_classes=args.nb_classes,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
    )
    print(f"[classifier C] source: {_count_params(model_src)} → "
          f"target: {_count_params(model_dst)}")
    weight_inherit(model_src, model_dst)
    return model_dst


# ===========================================================================
# Dispatch table
# ===========================================================================

_DISPATCH = {
    ("tokenizer",    "A"): scale_tokenizer_A,
    ("tokenizer",    "B"): scale_tokenizer_B,
    ("tokenizer",    "C"): scale_tokenizer_C,
    ("transformer",  "A"): scale_transformer_A,
    ("transformer",  "B"): scale_transformer_B,
    ("transformer",  "C"): scale_transformer_C,
    ("classifier",   "A"): scale_classifier_A,
    ("classifier",   "B"): scale_classifier_B,
    ("classifier",   "C"): scale_classifier_C,
}


# ===========================================================================
# CLI
# ===========================================================================

def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Expand a trained EEG model to a larger scale.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required
    p.add_argument("--model_type", required=True,
                   choices=["tokenizer", "transformer", "classifier"],
                   help="Which model to expand.")
    p.add_argument("--method", required=True, choices=["A", "B", "C"],
                   help="A=depth, B=width, C=weight-inherit.")
    p.add_argument("--checkpoint", required=True,
                   help="Path to source checkpoint (.pth).")
    p.add_argument("--model", required=True,
                   help="timm model name for the source architecture.")
    p.add_argument("--output", required=True,
                   help="Output path for the scaled checkpoint (.pth).")

    # Architecture — tokenizer
    p.add_argument("--n_embed", type=int, default=8192, help="Codebook size.")
    p.add_argument("--embed_dim", type=int, default=32,  help="Code dimensionality.")
    p.add_argument("--eeg_size", type=int, default=1600, help="EEG input length.")

    # Architecture — transformer
    p.add_argument("--vocab_size", type=int, default=8192)

    # Architecture — classifier
    p.add_argument("--nb_classes", type=int, default=6)
    p.add_argument("--drop",      type=float, default=0.0)
    p.add_argument("--drop_path", type=float, default=0.0)

    # Method A
    p.add_argument("--new_depth", type=int, default=24,
                   help="[Method A] Target depth after expansion.")

    # Method C
    p.add_argument("--target_model", default="",
                   help="[Method C] timm model name for the target (larger) architecture.")

    # Optional verification
    p.add_argument("--verify", action="store_true",
                   help="Run a quick forward-pass comparison (Methods A and B).")
    p.add_argument("--device", default="cpu")

    return p.parse_args()


def main():
    args = get_args()
    key = (args.model_type, args.method)

    if key not in _DISPATCH:
        print(f"[error] Unknown combination: {key}", file=sys.stderr)
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  model_scaling.py  —  {args.model_type}  /  Method {args.method}")
    print(f"{'='*60}")
    print(f"  source checkpoint : {args.checkpoint}")
    print(f"  source model      : {args.model}")
    if args.method == "C":
        print(f"  target model      : {args.target_model}")
    elif args.method == "A":
        print(f"  new depth         : {args.new_depth}")
    elif args.method == "B":
        print(f"  new embed_dim     : (auto: 2× current)")
    print(f"  output            : {args.output}")
    print(f"{'-'*60}")

    scaled = _DISPATCH[key](args)
    _save_checkpoint(scaled.state_dict(), args.output)

    print(f"\n  total parameters  : {_count_params(scaled)}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
