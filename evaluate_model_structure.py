"""
evaluate_model_structure.py
============================
Internal-structure evaluation for three model types:

  --model_type  tokenizer     VQNSP tokenizer  (analyses run on encoder)
  --model_type  transformer   Pre-trained NeuralTransformerForMEM backbone
  --model_type  classifier    Fine-tuned NeuralTransformer with class head

Five analyses (all saved as PNG + JSON metrics):

  [1] Representation Geometry
        t-SNE of CLS / patch-mean embeddings from the last layer.
        For classifiers: coloured by class label + Silhouette score.
        For tokenizer / transformer: coloured by k-means cluster.
        Also checks isotropy (singular-value concentration).

  [2] Attention Head Analysis
        Per-head attention entropy (focused vs diffuse).
        Head diversity matrix (correlation between heads).
        Dead-head detection (heads with near-uniform distribution).

  [3] Layer-wise CKA
        Centered Kernel Alignment between every pair of layers.
        Reveals which layers share information and where the model
        makes its biggest representational leap.

  [5] Calibration  (classifier only)
        Reliability diagram + Expected Calibration Error (ECE).
        Per-class confidence distribution.

  [6] Input Attribution
        Gradient × Input saliency: shows which EEG channels and
        time-points drive the model's output.
        Attention rollout: propagates attention across all layers
        back to the input patches.

Usage
-----
  # minimal (no EEG data — analyses 2 & 3 only, rest skipped)
  python evaluate_model_structure.py \\
      --model_type classifier \\
      --checkpoint path/to/ckpt.pth \\
      --output_dir eval_out

  # full evaluation
  python evaluate_model_structure.py \\
      --model_type classifier \\
      --checkpoint path/to/ckpt.pth \\
      --data_path  path/to/data.hdf5 \\
      --nb_classes 6 \\
      --dataset IIIC \\
      --output_dir eval_out
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from einops import rearrange

from timm.models import create_model

# registers @register_model entries
import tokenizer  as _tok_module   # noqa: F401
import task_model as _task_module  # noqa: F401

sys.path.insert(0, str(Path(__file__).parent))
from vis_codebook import _load_eeg_hdf5

import utils


# ═════════════════════════════════════════════════════════════════════════════
# Low-level: layer extraction with proper input_chans support
# ═════════════════════════════════════════════════════════════════════════════

def _extract_all_layers(
    transformer: nn.Module,
    x: torch.Tensor,
    input_chans=None,
    requires_grad: bool = False,
) -> List[torch.Tensor]:
    """
    Run x through a NeuralTransformer and return hidden-state tensors
    after every block — shape (B, 1 + N_patches, D) per layer.

    Mirrors forward_features() exactly (including input_chans for pos_embed)
    so spatial positions are correct for all channel subsets.

    Parameters
    ----------
    requires_grad : if True, tensors remain in the compute graph (for saliency).
    """
    ctx = torch.no_grad() if not requires_grad else torch.enable_grad()
    with ctx:
        B, n_ch, n_win, t = x.shape
        time_steps = n_win if t == transformer.patch_size else t

        tokens = transformer.patch_embed(x)
        cls_tok = transformer.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls_tok, tokens], dim=1)

        if transformer.pos_embed is not None:
            pos = (transformer.pos_embed[:, input_chans]
                   if input_chans is not None
                   else transformer.pos_embed)
            sp = pos[:, 1:].unsqueeze(2).expand(B, -1, time_steps, -1).flatten(1, 2)
            full_pos = torch.cat([pos[:, :1].expand(B, -1, -1), sp], dim=1)
            tokens = tokens + full_pos

        if transformer.time_embed is not None:
            n_sp = n_ch if t == transformer.patch_size else n_win
            t_emb = (transformer.time_embed[:, :time_steps]
                     .unsqueeze(1).expand(B, n_sp, -1, -1).flatten(1, 2))
            tokens[:, 1:] = tokens[:, 1:] + t_emb

        tokens = transformer.pos_drop(tokens)
        rpb = transformer.rel_pos_bias() if transformer.rel_pos_bias is not None else None

        layer_outs = []
        for blk in transformer.blocks:
            tokens = blk(tokens, rel_pos_bias=rpb)
            layer_outs.append(tokens)
        return layer_outs


# ═════════════════════════════════════════════════════════════════════════════
# Attention grabber (hook-based, no model modification)
# ═════════════════════════════════════════════════════════════════════════════

class AttentionGrabber:
    """
    Context manager that registers forward hooks on every Block to capture
    the attention weight matrix (B, H, N, N) after softmax.

    Calls block.attn(block.norm1(input), return_attention=True) inside the
    hook — this re-uses Block's own Attention module, so it respects all
    head configurations (q/k bias, relative pos bias, etc.).
    """

    def __init__(self, blocks: List[nn.Module]):
        self._blocks = blocks
        self._hooks: list = []
        # {layer_idx: list[(B, H, N, N)]} accumulated across batches
        self.maps: dict[int, list] = {}

    def __enter__(self):
        for i, blk in enumerate(self._blocks):
            def _make(idx, block):
                def hook(module, inp, output):
                    x_in = inp[0]
                    with torch.no_grad():
                        normed = block.norm1(x_in)
                        # Block.attn.forward with return_attention=True returns
                        # the (B, H, N, N) weight tensor before the value projection
                        attn = block.attn(normed, return_attention=True)
                    if idx not in self.maps:
                        self.maps[idx] = []
                    self.maps[idx].append(attn.detach().float().cpu())
                return hook
            self._hooks.append(blk.register_forward_hook(_make(i, blk)))
        return self

    def __exit__(self, *args):
        for h in self._hooks:
            h.remove()

    def mean_attn(self, layer_idx: int) -> torch.Tensor:
        """Return mean attention over the dataset: (H, N, N)."""
        return torch.cat(self.maps[layer_idx], dim=0).mean(0)

    def all_mean(self) -> dict[int, torch.Tensor]:
        return {i: self.mean_attn(i) for i in self.maps}


# ═════════════════════════════════════════════════════════════════════════════
# ModelAdapter — uniform interface for all three model types
# ═════════════════════════════════════════════════════════════════════════════

class ModelAdapter:
    """
    Wraps a model and exposes a consistent API regardless of type.

    Attributes
    ----------
    transformer  : the NeuralTransformer that produces token sequences
    blocks       : transformer.blocks  (for AttentionGrabber)
    has_labels   : True for classifiers (enables calibration + coloured t-SNE)
    model_type   : 'tokenizer' | 'transformer' | 'classifier'
    """

    def __init__(
        self,
        model: nn.Module,
        model_type: str,
        input_chans=None,
        nb_classes: int = 2,
        is_binary: bool = False,
    ):
        self.model      = model
        self.model_type = model_type
        self.input_chans = input_chans
        self.nb_classes = nb_classes
        self.is_binary  = is_binary

        if model_type == 'tokenizer':
            self.transformer = model.encoder
        elif model_type == 'transformer':
            self.transformer = model.encoder if hasattr(model, 'encoder') else model
        else:
            self.transformer = model

        self.blocks     = self.transformer.blocks
        self.n_layers   = len(self.blocks)
        self.has_labels = (model_type == 'classifier')

    # ── token representations ─────────────────────────────────────────────

    @torch.no_grad()
    def get_hidden_states(self, x: torch.Tensor) -> List[torch.Tensor]:
        return _extract_all_layers(self.transformer, x, self.input_chans)

    @torch.no_grad()
    def get_cls_last(self, x: torch.Tensor) -> torch.Tensor:
        """(B, D) CLS embedding from the final block, post-norm."""
        last = _extract_all_layers(self.transformer, x, self.input_chans)[-1]
        cls  = last[:, 0]
        if self.transformer.norm is not None:
            cls = self.transformer.norm(last)[:, 0]
        return cls

    # ── forward for gradient computation ─────────────────────────────────

    def forward_score(self, x: torch.Tensor, target: Optional[torch.Tensor] = None):
        """
        Returns a (B,) score tensor for backpropagation.
        - classifier  : logit of target class (or max logit if target is None)
        - others      : L2 norm of final CLS token (differentiable proxy)
        """
        if self.model_type == 'classifier':
            logits = self.model(x, self.input_chans)
            if self.is_binary:
                return logits.squeeze(-1)
            if target is not None:
                return logits.gather(1, target.view(-1, 1)).squeeze(1)
            return logits.max(dim=1).values
        else:
            # enable gradients inside _extract_all_layers
            last = _extract_all_layers(
                self.transformer, x, self.input_chans, requires_grad=True)[-1]
            cls = last[:, 0]
            if self.transformer.norm is not None:
                cls = self.transformer.norm(last)[:, 0]
            return cls.norm(dim=-1)

    # ── class logits / probabilities (classifier only) ────────────────────

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """(B, C) probability tensor. Raises if not a classifier."""
        assert self.has_labels, "predict_proba only available for classifiers"
        logits = self.model(x, self.input_chans)
        if self.is_binary:
            p = torch.sigmoid(logits).squeeze(-1)
            return torch.stack([1 - p, p], dim=1)
        return logits.softmax(dim=-1)


# ═════════════════════════════════════════════════════════════════════════════
# Data loading
# ═════════════════════════════════════════════════════════════════════════════

def load_eval_data(
    data_path: str,
    n_samples: int,
    eeg_size: int = 1600,
    t_patch: int = 200,
    device: torch.device = torch.device('cpu'),
) -> tuple[torch.Tensor, Optional[np.ndarray]]:
    """
    Load EEG data and optional labels from an HDF5 file.

    Returns
    -------
    x      : (N, C, A, T_patch) float32 tensor on `device`, divided by 100
    labels : (N,) int numpy array or None if file has no labels key
    """
    import h5py
    x_np = _load_eeg_hdf5(data_path, n_samples)     # (N, C, T)
    N, C, T = x_np.shape
    A       = eeg_size // t_patch
    T_need  = A * t_patch
    if T < T_need:
        x_np = np.pad(x_np, ((0,0),(0,0),(0, T_need - T)))
    else:
        x_np = x_np[:, :, :T_need]
    x = torch.tensor(x_np, dtype=torch.float32).to(device) / 100.0
    x = x.view(N, C, A, t_patch)

    labels = None
    with h5py.File(data_path, 'r') as f:
        for key in ('label', 'labels', 'y', 'target', 'targets'):
            if key in f:
                labels = f[key][:N][()].astype(np.int64)
                if labels.ndim > 1:
                    labels = labels.argmax(-1)
                break
    return x, labels


# ═════════════════════════════════════════════════════════════════════════════
# Linear CKA
# ═════════════════════════════════════════════════════════════════════════════

def _linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """
    Compute linear Centered Kernel Alignment between representation matrices.
    X : (N, D1), Y : (N, D2) — already centred.
    """
    XtY  = X.T @ Y
    XtX  = X.T @ X
    YtY  = Y.T @ Y
    num  = (XtY ** 2).sum()
    denom = np.sqrt((XtX ** 2).sum()) * np.sqrt((YtY ** 2).sum()) + 1e-12
    return float(num / denom)


def _centre(M: np.ndarray) -> np.ndarray:
    return M - M.mean(axis=0, keepdims=True)


# ═════════════════════════════════════════════════════════════════════════════
# Analysis 1 — Representation Geometry
# ═════════════════════════════════════════════════════════════════════════════

def analysis_representation_geometry(
    adapter: ModelAdapter,
    x: torch.Tensor,
    labels: Optional[np.ndarray],
    out_dir: str,
    batch_size: int = 64,
    n_tsne_max: int = 2000,
) -> dict:
    print("  computing CLS embeddings …", flush=True)
    cls_list = []
    for start in range(0, len(x), batch_size):
        cls_list.append(adapter.get_cls_last(x[start:start + batch_size]).cpu().float())
    cls = torch.cat(cls_list).numpy()     # (N, D)
    N, D = cls.shape

    # ── isotropy (singular value concentration) ───────────────────────────
    centred = _centre(cls)
    _, sv, _ = np.linalg.svd(centred, full_matrices=False)
    sv_norm  = sv / (sv.sum() + 1e-12)
    top1_frac  = float(sv_norm[0])
    top10_frac = float(sv_norm[:10].sum())
    # effective rank (exponential of entropy of sv distribution)
    eff_rank = float(np.exp(-(sv_norm * np.log(sv_norm + 1e-12)).sum()))

    fig_sv, ax_sv = plt.subplots(figsize=(8, 4))
    ax_sv.plot(np.cumsum(sv_norm) * 100, linewidth=1.5)
    ax_sv.axvline(10, color='grey', linestyle='--', linewidth=0.8)
    ax_sv.set_xlabel('Singular value rank'); ax_sv.set_ylabel('Cumulative variance (%)')
    ax_sv.set_title(f'CLS embedding isotropy  |  eff_rank={eff_rank:.1f}  '
                    f'top-1={top1_frac*100:.1f}%  top-10={top10_frac*100:.1f}%')
    ax_sv.grid(True, alpha=0.3)
    fig_sv.savefig(os.path.join(out_dir, 'geom_isotropy.png'), dpi=150, bbox_inches='tight')
    plt.close(fig_sv)

    # ── t-SNE ─────────────────────────────────────────────────────────────
    idx = np.random.choice(N, min(n_tsne_max, N), replace=False)
    sub = centred[idx]
    perp = min(30, max(5, len(idx) // 50))
    from sklearn.manifold import TSNE
    xy = TSNE(n_components=2, random_state=42, perplexity=perp).fit_transform(sub)

    fig_ts, ax_ts = plt.subplots(figsize=(8, 7))
    if labels is not None and adapter.has_labels:
        sub_labels = labels[idx]
        n_cls = int(sub_labels.max()) + 1
        cmap  = plt.get_cmap('tab10', n_cls)
        for c in range(n_cls):
            mask = sub_labels == c
            ax_ts.scatter(xy[mask, 0], xy[mask, 1], s=5, alpha=0.6,
                          color=cmap(c), label=f'class {c}')
        ax_ts.legend(fontsize=8, markerscale=3)

        # silhouette score
        from sklearn.metrics import silhouette_score
        if len(np.unique(sub_labels)) > 1:
            sil = silhouette_score(sub, sub_labels, metric='cosine',
                                   sample_size=min(5000, len(sub)))
        else:
            sil = float('nan')
        ax_ts.set_title(f't-SNE of CLS embeddings  |  Silhouette={sil:.3f}', fontsize=12)
    else:
        from sklearn.cluster import KMeans
        k = min(10, N // 20)
        km = KMeans(n_clusters=k, n_init=5, random_state=42).fit(sub)
        ax_ts.scatter(xy[:, 0], xy[:, 1], c=km.labels_, s=5, alpha=0.6,
                      cmap='tab10')
        sil = float('nan')
        ax_ts.set_title(f't-SNE (k-means, k={k}) — {adapter.model_type}', fontsize=12)

    ax_ts.set_xlabel('t-SNE 1'); ax_ts.set_ylabel('t-SNE 2')
    fig_ts.savefig(os.path.join(out_dir, 'geom_tsne.png'), dpi=150, bbox_inches='tight')
    plt.close(fig_ts)
    print(f"  → geom_isotropy.png  geom_tsne.png")

    return {
        'eff_rank':       round(eff_rank, 2),
        'top1_sv_frac':   round(top1_frac, 4),
        'top10_sv_frac':  round(top10_frac, 4),
        'silhouette':     round(sil, 4) if not np.isnan(sil) else None,
        'embed_dim':      D,
        'n_samples':      N,
    }


# ═════════════════════════════════════════════════════════════════════════════
# Analysis 2 — Attention Head Analysis
# ═════════════════════════════════════════════════════════════════════════════

def analysis_attention_heads(
    adapter: ModelAdapter,
    x: torch.Tensor,
    out_dir: str,
    batch_size: int = 32,
    dead_entropy_threshold: float = 0.95,
) -> dict:
    """
    Run model forward on data while collecting attention maps per block.
    Computes per-head entropy and head-diversity matrix.
    """
    print("  collecting attention maps …", flush=True)
    grabber = AttentionGrabber(adapter.blocks)

    with grabber:
        adapter.model.eval()
        with torch.no_grad():
            for start in range(0, len(x), batch_size):
                chunk = x[start:start + batch_size]
                if adapter.model_type == 'classifier':
                    adapter.model(chunk, adapter.input_chans)
                elif adapter.model_type == 'tokenizer':
                    adapter.model.encoder(chunk, adapter.input_chans,
                                          return_patch_tokens=True)
                else:
                    adapter.transformer(chunk, adapter.input_chans,
                                        return_patch_tokens=True)

    mean_attns = grabber.all_mean()   # {layer_idx: (H, N, N)}
    L = len(mean_attns)
    H = next(iter(mean_attns.values())).shape[0]

    # ── per-head entropy (H, L) ───────────────────────────────────────────
    entropy = np.zeros((L, H))
    max_entropy = float(np.log(next(iter(mean_attns.values())).shape[-1]))
    for li, attn in mean_attns.items():
        # attn: (H, N, N) — attention from each query position
        p = attn.numpy() + 1e-9
        ent = -(p * np.log(p)).sum(-1).mean(-1)   # (H,)
        entropy[li] = ent / max_entropy            # normalise to [0, 1]

    n_dead = int((entropy > dead_entropy_threshold).sum())
    dead_pct = 100.0 * n_dead / (L * H)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # left: heatmap layers × heads
    im = axes[0].imshow(entropy.T, cmap='RdYlGn_r', vmin=0, vmax=1,
                        aspect='auto', interpolation='nearest')
    axes[0].set_xlabel('Layer', fontsize=11)
    axes[0].set_ylabel('Head', fontsize=11)
    axes[0].set_title(
        f'Attention entropy (norm.)  |  dead heads (>{dead_entropy_threshold:.0%}): '
        f'{n_dead}/{L*H} ({dead_pct:.1f}%)',
        fontsize=10)
    plt.colorbar(im, ax=axes[0])
    axes[0].set_xticks(np.arange(L)); axes[0].set_xticklabels(np.arange(L), fontsize=7)
    axes[0].set_yticks(np.arange(H)); axes[0].set_yticklabels(np.arange(H), fontsize=7)

    # right: head-diversity — flatten all L×H heads to vectors, then pairwise correlation
    # Each head vector = its attention entropy profile across layers
    head_entropy_vecs = entropy.T    # (H, L)
    corr = np.corrcoef(head_entropy_vecs)   # (H, H)
    im2 = axes[1].imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1,
                         aspect='auto', interpolation='nearest')
    axes[1].set_title('Head diversity (correlation of entropy profiles)', fontsize=10)
    axes[1].set_xlabel('Head'); axes[1].set_ylabel('Head')
    plt.colorbar(im2, ax=axes[1])
    mean_off_diag_corr = float(
        corr[np.triu_indices(H, k=1)].mean()) if H > 1 else float('nan')
    axes[1].text(0.02, 0.97,
                 f'mean off-diag corr = {mean_off_diag_corr:.3f}',
                 transform=axes[1].transAxes, va='top', fontsize=9,
                 bbox=dict(boxstyle='round', fc='white', alpha=0.8))

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'attn_head_analysis.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

    # ── per-layer mean attention map (averaged over heads and samples) ────
    fig2, axes2 = plt.subplots(2, (L + 1) // 2, figsize=(L * 1.5, 6))
    axes2 = np.array(axes2).ravel()
    for li in range(L):
        attn_li = mean_attns[li].mean(0).numpy()   # (N, N) mean over heads
        ax = axes2[li]
        ax.imshow(attn_li, cmap='Blues', aspect='auto', interpolation='nearest')
        ax.set_title(f'L{li}', fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
    for li in range(L, len(axes2)):
        axes2[li].set_visible(False)
    fig2.suptitle('Mean attention map per layer (averaged over heads and samples)', fontsize=10)
    fig2.tight_layout()
    fig2.savefig(os.path.join(out_dir, 'attn_maps_per_layer.png'), dpi=150, bbox_inches='tight')
    plt.close(fig2)

    print(f"  → attn_head_analysis.png  attn_maps_per_layer.png")
    return {
        'n_layers':              L,
        'n_heads':               H,
        'dead_heads':            n_dead,
        'dead_pct':              round(dead_pct, 2),
        'mean_entropy':          round(float(entropy.mean()), 4),
        'mean_head_diversity':   round(float(mean_off_diag_corr), 4),
        'most_focused_layer':    int(entropy.mean(-1).argmin()),
        'most_uniform_layer':    int(entropy.mean(-1).argmax()),
    }


# ═════════════════════════════════════════════════════════════════════════════
# Analysis 3 — Layer-wise CKA
# ═════════════════════════════════════════════════════════════════════════════

def analysis_layer_cka(
    adapter: ModelAdapter,
    x: torch.Tensor,
    out_dir: str,
    batch_size: int = 32,
    n_max: int = 512,
) -> dict:
    """
    Collect CLS tokens from every layer, then compute pairwise linear CKA.
    """
    print("  collecting layer representations …", flush=True)
    x_sub = x[:min(n_max, len(x))]

    # accumulate CLS token per layer: list[L] of (N, D)
    layer_cls: list[list] = [[] for _ in range(adapter.n_layers)]
    with torch.no_grad():
        for start in range(0, len(x_sub), batch_size):
            chunk = x_sub[start:start + batch_size]
            hidden = adapter.get_hidden_states(chunk)
            for li, h in enumerate(hidden):
                layer_cls[li].append(h[:, 0].float().cpu())   # CLS token

    # stack: list[L] of (N, D)
    reps = [torch.cat(layer_cls[li]).numpy() for li in range(adapter.n_layers)]
    L = len(reps)

    print("  computing CKA matrix …", flush=True)
    centred = [_centre(r) for r in reps]
    cka_matrix = np.zeros((L, L))
    for i in range(L):
        for j in range(i, L):
            v = _linear_cka(centred[i], centred[j])
            cka_matrix[i, j] = v
            cka_matrix[j, i] = v

    # ── plot ──────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    im = axes[0].imshow(cka_matrix, cmap='Blues', vmin=0, vmax=1,
                        aspect='auto', interpolation='nearest')
    axes[0].set_title('Pairwise layer CKA', fontsize=12)
    axes[0].set_xlabel('Layer'); axes[0].set_ylabel('Layer')
    plt.colorbar(im, ax=axes[0])

    # adjacent-layer CKA profile
    adj = [cka_matrix[i, i+1] for i in range(L - 1)]
    axes[1].bar(np.arange(L - 1), adj, color='steelblue', alpha=0.8)
    axes[1].set_xlabel('Layer transition (i → i+1)', fontsize=11)
    axes[1].set_ylabel('CKA', fontsize=11)
    axes[1].set_title('Adjacent-layer CKA\n(low = big representational change)', fontsize=11)
    axes[1].set_ylim(0, 1); axes[1].grid(True, alpha=0.3)
    # annotate the layer with the biggest change
    if adj:
        pivot = int(np.argmin(adj))
        axes[1].bar(pivot, adj[pivot], color='tomato', alpha=0.9,
                    label=f'biggest jump: L{pivot}→L{pivot+1}')
        axes[1].legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'cka_layers.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  → cka_layers.png")

    # CKA between first and last layer (overall representation change)
    first_last_cka = float(cka_matrix[0, -1])
    return {
        'first_last_cka':          round(first_last_cka, 4),
        'mean_adjacent_cka':       round(float(np.mean(adj)), 4),
        'min_adjacent_cka':        round(float(np.min(adj)), 4),
        'biggest_change_layer':    int(np.argmin(adj)) if adj else None,
    }


# ═════════════════════════════════════════════════════════════════════════════
# Analysis 5 — Calibration  (classifier only)
# ═════════════════════════════════════════════════════════════════════════════

def analysis_calibration(
    adapter: ModelAdapter,
    x: torch.Tensor,
    labels: np.ndarray,
    out_dir: str,
    batch_size: int = 64,
    n_bins: int = 15,
) -> dict:
    """
    Expected Calibration Error + reliability diagram.
    """
    assert adapter.has_labels, "calibration requires a classifier"
    print("  computing predicted probabilities …", flush=True)
    probs_list, preds_list = [], []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            chunk = x[start:start + batch_size]
            p = adapter.predict_proba(chunk).cpu().float()
            probs_list.append(p)
            preds_list.append(p.argmax(dim=1))

    probs  = torch.cat(probs_list).numpy()     # (N, C)
    preds  = torch.cat(preds_list).numpy()     # (N,)
    confs  = probs.max(axis=1)                 # (N,) max confidence
    correct = (preds == labels).astype(float)  # (N,)
    N = len(labels)

    # ── ECE ───────────────────────────────────────────────────────────────
    bins   = np.linspace(0, 1, n_bins + 1)
    bin_lo = bins[:-1]
    bin_hi = bins[1:]
    bin_acc  = np.zeros(n_bins)
    bin_conf = np.zeros(n_bins)
    bin_count = np.zeros(n_bins, dtype=int)

    for bi in range(n_bins):
        mask = (confs >= bin_lo[bi]) & (confs < bin_hi[bi])
        if mask.sum() > 0:
            bin_acc[bi]   = correct[mask].mean()
            bin_conf[bi]  = confs[mask].mean()
            bin_count[bi] = int(mask.sum())

    ece = float((np.abs(bin_acc - bin_conf) * bin_count / N).sum())
    overconfident_pct = float(100 * (confs[correct == 0] > 0.8).mean())

    # ── reliability diagram ───────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    ax = axes[0]
    bin_centres = (bin_lo + bin_hi) / 2
    ax.bar(bin_centres, bin_acc, width=1/n_bins * 0.9,
           alpha=0.7, color='steelblue', label='accuracy')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='perfect calibration')
    # gap fill
    ax.bar(bin_centres, bin_conf - bin_acc,
           bottom=np.minimum(bin_acc, bin_conf),
           width=1/n_bins * 0.9, alpha=0.4, color='tomato',
           label='calibration gap')
    ax.set_xlabel('Confidence', fontsize=11)
    ax.set_ylabel('Accuracy', fontsize=11)
    ax.set_title(f'Reliability diagram  |  ECE={ece:.4f}', fontsize=11)
    ax.legend(fontsize=8); ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    # sample count per bin
    ax2 = axes[1]
    ax2.bar(bin_centres, bin_count, width=1/n_bins * 0.9, color='steelblue', alpha=0.7)
    ax2.set_xlabel('Confidence bin', fontsize=11)
    ax2.set_ylabel('Sample count', fontsize=11)
    ax2.set_title('Confidence distribution', fontsize=11)

    # per-class confidence distribution
    ax3 = axes[2]
    n_cls = probs.shape[1]
    cmap  = plt.get_cmap('tab10', n_cls)
    for c in range(n_cls):
        mask = labels == c
        if mask.sum() > 0:
            ax3.hist(probs[mask, c], bins=20, alpha=0.5,
                     color=cmap(c), label=f'class {c}', density=True)
    ax3.set_xlabel('Predicted probability for true class', fontsize=11)
    ax3.set_ylabel('Density', fontsize=11)
    ax3.set_title('Per-class confidence (true-class probability)', fontsize=11)
    ax3.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'calibration.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  → calibration.png")

    acc = float(correct.mean())
    return {
        'ece':                   round(ece, 5),
        'accuracy':              round(acc, 4),
        'mean_confidence':       round(float(confs.mean()), 4),
        'overconfident_pct':     round(overconfident_pct, 2),
        'calibration_grade': (
            'EXCELLENT' if ece < 0.02 else
            'GOOD'      if ece < 0.05 else
            'FAIR'      if ece < 0.10 else 'POOR'
        ),
    }


# ═════════════════════════════════════════════════════════════════════════════
# Analysis 6 — Input Attribution
# ═════════════════════════════════════════════════════════════════════════════

def analysis_attribution(
    adapter: ModelAdapter,
    x: torch.Tensor,
    labels: Optional[np.ndarray],
    out_dir: str,
    n_samples: int = 32,
    ch_names: Optional[List[str]] = None,
) -> dict:
    """
    Two attribution methods:

    A) Gradient × Input
       Saliency = |∇_x score| ⊙ |x|  aggregated over the batch.
       Reveals which EEG channels and time-points matter most.

    B) Attention rollout
       Propagates attention through all layers (with 0.5·I residual),
       producing a single (N_tokens,) importance vector per sample.
       Aggregated over the batch to show which patches are attended to.
    """
    n_use = min(n_samples, len(x))
    x_sub = x[:n_use]
    lab_sub = labels[:n_use] if labels is not None else None
    B, C, A, T = x_sub.shape

    # ─── A: Gradient × Input ─────────────────────────────────────────────
    print("  computing Gradient × Input saliency …", flush=True)
    adapter.model.eval()
    x_in = x_sub.clone().requires_grad_(True)
    target = torch.tensor(lab_sub, dtype=torch.long, device=x_sub.device) \
             if lab_sub is not None else None

    score = adapter.forward_score(x_in, target)
    score.sum().backward()

    with torch.no_grad():
        grad_input = (x_in.grad.abs() * x_in.abs()).detach().cpu().float()

    # reshape to (B, C, T_total) then mean over batch
    saliency = grad_input.view(B, C, -1).mean(0).numpy()    # (C, T_total)
    saliency /= saliency.max() + 1e-9

    fig_sal, ax_sal = plt.subplots(figsize=(14, 5))
    im_sal = ax_sal.imshow(saliency, aspect='auto', cmap='hot',
                           interpolation='nearest')
    ax_sal.set_xlabel('Time (samples)', fontsize=11)
    ax_sal.set_ylabel('Channel', fontsize=11)
    ax_sal.set_title(
        f'Gradient × Input saliency  |  {adapter.model_type}  '
        f'(mean over {n_use} samples)',
        fontsize=11)
    plt.colorbar(im_sal, ax=ax_sal)
    if ch_names and len(ch_names) >= C:
        ax_sal.set_yticks(np.arange(C))
        ax_sal.set_yticklabels(ch_names[:C], fontsize=6)
    fig_sal.tight_layout()
    fig_sal.savefig(os.path.join(out_dir, 'attr_grad_input.png'), dpi=150, bbox_inches='tight')
    plt.close(fig_sal)

    # top-5 most salient channels
    ch_importance = saliency.mean(-1)   # (C,)
    top5_idx  = np.argsort(ch_importance)[::-1][:5].tolist()
    top5_names = [ch_names[i] if ch_names and i < len(ch_names)
                  else str(i) for i in top5_idx]

    # ─── B: Attention rollout ─────────────────────────────────────────────
    print("  computing attention rollout …", flush=True)
    grabber = AttentionGrabber(adapter.blocks)
    with grabber:
        adapter.model.eval()
        with torch.no_grad():
            if adapter.model_type == 'classifier':
                adapter.model(x_sub, adapter.input_chans)
            elif adapter.model_type == 'tokenizer':
                adapter.model.encoder(x_sub, adapter.input_chans,
                                       return_patch_tokens=True)
            else:
                adapter.transformer(x_sub, adapter.input_chans,
                                    return_patch_tokens=True)

    # rollout: R_0 = I, R_l = R_{l-1} @ (0.5*A_l + 0.5*I)
    mean_attns = grabber.all_mean()   # {l: (H, N_tok, N_tok)}
    N_tok = next(iter(mean_attns.values())).shape[-1]
    rollout = np.eye(N_tok)

    for li in range(adapter.n_layers):
        attn_l = mean_attns[li].mean(0).numpy()   # (N_tok, N_tok) mean over heads
        a = 0.5 * attn_l + 0.5 * np.eye(N_tok)
        a /= a.sum(-1, keepdims=True) + 1e-12
        rollout = rollout @ a

    # CLS → patch importance: rollout[0, 1:] shape (N_patches,)
    cls_to_patches = rollout[0, 1:]
    cls_to_patches /= cls_to_patches.max() + 1e-9

    # reshape to spatial grid (C, A) then upsample to (C, T_total)
    try:
        importance_2d = cls_to_patches.reshape(C, A)
        importance_full = np.repeat(importance_2d, T, axis=1)   # (C, T_total)
    except ValueError:
        # token count might not equal C*A (e.g. different input_chans)
        importance_full = None

    fig_ro, axes_ro = plt.subplots(1, 2, figsize=(14, 4))

    ax = axes_ro[0]
    ax.bar(np.arange(len(cls_to_patches)), cls_to_patches,
           color='teal', alpha=0.8, width=1.0, linewidth=0)
    ax.set_xlabel('Patch index (spatial × temporal)', fontsize=10)
    ax.set_ylabel('Rollout importance', fontsize=10)
    ax.set_title(f'Attention rollout: CLS → patches  ({adapter.n_layers} layers)', fontsize=10)

    if importance_full is not None:
        ax2 = axes_ro[1]
        im2 = ax2.imshow(importance_full, aspect='auto', cmap='YlOrRd',
                         interpolation='nearest')
        ax2.set_xlabel('Time (samples)', fontsize=10)
        ax2.set_ylabel('Channel', fontsize=10)
        ax2.set_title('Rollout importance by channel and time', fontsize=10)
        plt.colorbar(im2, ax=ax2)
        if ch_names and len(ch_names) >= C:
            ax2.set_yticks(np.arange(C))
            ax2.set_yticklabels(ch_names[:C], fontsize=6)
    else:
        axes_ro[1].set_visible(False)

    fig_ro.tight_layout()
    fig_ro.savefig(os.path.join(out_dir, 'attr_attention_rollout.png'),
                  dpi=150, bbox_inches='tight')
    plt.close(fig_ro)
    print("  → attr_grad_input.png  attr_attention_rollout.png")

    return {
        'top5_salient_channels': top5_names,
        'top5_salient_ch_idx':   top5_idx,
        'mean_saliency_per_ch':  ch_importance.tolist(),
        'rollout_n_patches':     int(len(cls_to_patches)),
    }


# ═════════════════════════════════════════════════════════════════════════════
# Model loading
# ═════════════════════════════════════════════════════════════════════════════

def _strip_ddp(state: dict) -> dict:
    return {k.removeprefix('module.'): v for k, v in state.items()}


def load_model(args, device: torch.device) -> nn.Module:
    raw = torch.load(args.checkpoint, map_location=device)
    state = _strip_ddp(raw.get('model', raw.get('state_dict', raw)))

    if args.model_type == 'tokenizer':
        model = create_model(
            args.model,
            pretrained=False, as_tokenzer=False,
            n_code=args.n_embed, code_dim=args.embed_dim, EEG_size=args.eeg_size,
        )
    elif args.model_type == 'transformer':
        model = create_model(
            args.model,
            pretrained=False,
            vocab_size=args.vocab_size,
        )
    else:   # classifier
        model = create_model(
            args.model,
            pretrained=False,
            num_classes=args.nb_classes,
            drop_rate=args.drop,
            drop_path_rate=args.drop_path,
            use_mean_pooling=True,
            init_scale=0.001,
            use_rel_pos_bias=False,
            use_abs_pos_emb=True,
            init_values=0.1,
            qkv_bias=True,
        )

    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"  [warn] {len(missing)} missing keys")
    model.to(device).eval()
    return model


# ═════════════════════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════════════════════

def get_args():
    p = argparse.ArgumentParser('evaluate_model_structure', add_help=True)

    # ── required ──────────────────────────────────────────────────────────
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--model_type', required=True,
                   choices=['tokenizer', 'transformer', 'classifier'])

    # ── model architecture ────────────────────────────────────────────────
    p.add_argument('--model',      default='base_patch200_200')
    # tokenizer-specific
    p.add_argument('--n_embed',    default=8192, type=int)
    p.add_argument('--embed_dim',  default=32,   type=int)
    p.add_argument('--eeg_size',   default=1600, type=int)
    # transformer-specific
    p.add_argument('--vocab_size', default=8192, type=int)
    # classifier-specific
    p.add_argument('--nb_classes', default=2,    type=int)
    p.add_argument('--is_binary',  action='store_true', default=False)
    p.add_argument('--drop',       default=0.0,  type=float)
    p.add_argument('--drop_path',  default=0.0,  type=float)

    # ── dataset ───────────────────────────────────────────────────────────
    p.add_argument('--dataset', default=None,
                   help='IIIC | TUAB | TUEV | SLEEP  — infers ch_names automatically')
    p.add_argument('--ch_names', default=None, nargs='+',
                   help='Explicit channel names (overrides --dataset)')
    p.add_argument('--data_path',      default=None)
    p.add_argument('--n_samples',      default=1000, type=int)
    p.add_argument('--n_saliency',     default=32,   type=int,
                   help='Number of samples for gradient/rollout computation')

    # ── output ────────────────────────────────────────────────────────────
    p.add_argument('--output_dir', default='eval_structure_out')
    p.add_argument('--device',     default='cpu')

    # ── skip flags ────────────────────────────────────────────────────────
    p.add_argument('--skip_geom',   action='store_true')
    p.add_argument('--skip_attn',   action='store_true')
    p.add_argument('--skip_cka',    action='store_true')
    p.add_argument('--skip_calib',  action='store_true')
    p.add_argument('--skip_attr',   action='store_true')

    return p.parse_args()


# ═════════════════════════════════════════════════════════════════════════════
# Entry point
# ═════════════════════════════════════════════════════════════════════════════

_DATASET_CH = {
    'TUAB': ['FP1','FP2','F3','F4','C3','C4','P3','P4',
             'O1','O2','F7','F8','T3','T4','T5','T6',
             'A1','A2','FZ','CZ','PZ','T1','T2'],
    'TUEV': ['FP1','FP2','F3','F4','C3','C4','P3','P4',
             'O1','O2','F7','F8','T3','T4','T5','T6',
             'A1','A2','FZ','CZ','PZ','T1','T2'],
    'IIIC': ['FP1','F3','C3','P3','F7','T3','T5','O1','FZ','CZ','PZ',
             'FP2','F4','C4','P4','F8','T4','T6','O2'],
    'SLEEP':['FP1','F3','C3','P3','F7','T3','T5','O1','FZ','CZ','PZ',
             'FP2','F4','C4','P4','F8','T4','T6','O2'],
}


def main():
    args   = get_args()
    device = torch.device(args.device)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # ── channel names → input_chans ───────────────────────────────────────
    ch_names = args.ch_names
    if ch_names is None and args.dataset in _DATASET_CH:
        ch_names = _DATASET_CH[args.dataset]
    input_chans = utils.get_input_chans(ch_names) if ch_names else None

    # ── load model ────────────────────────────────────────────────────────
    print(f"\nLoading {args.model_type} from {args.checkpoint} …")
    model   = load_model(args, device)
    adapter = ModelAdapter(
        model, args.model_type,
        input_chans=input_chans,
        nb_classes=args.nb_classes,
        is_binary=args.is_binary,
    )
    print(f"  transformer: {adapter.n_layers} layers, "
          f"dim={adapter.transformer.embed_dim}")

    # ── load data ─────────────────────────────────────────────────────────
    x, labels = None, None
    need_data = not (args.skip_geom and args.skip_attn and args.skip_cka
                     and args.skip_calib and args.skip_attr)
    if args.data_path and need_data:
        print(f"\nLoading data from {args.data_path} …")
        x, labels = load_eval_data(
            args.data_path, args.n_samples,
            eeg_size=args.eeg_size, device=device,
        )
        # trim channels to what the model expects (from input_chans)
        if ch_names:
            n_expected = len(ch_names)
            if x.shape[1] > n_expected:
                x = x[:, :n_expected]
            elif x.shape[1] < n_expected:
                pad = torch.zeros(x.shape[0], n_expected - x.shape[1],
                                  *x.shape[2:], device=device)
                x = torch.cat([x, pad], dim=1)
        print(f"  x: {tuple(x.shape)}  labels: "
              f"{tuple(labels.shape) if labels is not None else None}")

    all_metrics: dict = {'model_type': args.model_type, 'checkpoint': args.checkpoint}

    # ── Analysis 1 ────────────────────────────────────────────────────────
    if not args.skip_geom:
        if x is not None:
            print("\n[1] Representation geometry …")
            all_metrics['geometry'] = analysis_representation_geometry(
                adapter, x, labels, args.output_dir)
        else:
            print("\n[1] Representation geometry — skipped (no --data_path)")

    # ── Analysis 2 ────────────────────────────────────────────────────────
    if not args.skip_attn:
        if x is not None:
            print("\n[2] Attention head analysis …")
            all_metrics['attention'] = analysis_attention_heads(
                adapter, x, args.output_dir)
        else:
            print("\n[2] Attention head analysis — skipped (no --data_path)")

    # ── Analysis 3 ────────────────────────────────────────────────────────
    if not args.skip_cka:
        if x is not None:
            print("\n[3] Layer-wise CKA …")
            all_metrics['cka'] = analysis_layer_cka(adapter, x, args.output_dir)
        else:
            print("\n[3] Layer-wise CKA — skipped (no --data_path)")

    # ── Analysis 5 ────────────────────────────────────────────────────────
    if not args.skip_calib:
        if adapter.has_labels and x is not None and labels is not None:
            print("\n[5] Calibration …")
            all_metrics['calibration'] = analysis_calibration(
                adapter, x, labels, args.output_dir)
        elif not adapter.has_labels:
            print("\n[5] Calibration — skipped (not a classifier)")
        else:
            print("\n[5] Calibration — skipped (no data / labels)")

    # ── Analysis 6 ────────────────────────────────────────────────────────
    if not args.skip_attr:
        if x is not None:
            print("\n[6] Input attribution …")
            all_metrics['attribution'] = analysis_attribution(
                adapter, x, labels, args.output_dir,
                n_samples=args.n_saliency, ch_names=ch_names)
        else:
            print("\n[6] Attribution — skipped (no --data_path)")

    # ── summary ───────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(" STRUCTURE EVALUATION SUMMARY")
    print("=" * 60)
    _print_section(all_metrics.get('geometry'),   '[1] Representation geometry')
    _print_section(all_metrics.get('attention'),  '[2] Attention heads')
    _print_section(all_metrics.get('cka'),        '[3] Layer CKA')
    _print_section(all_metrics.get('calibration'),'[5] Calibration')
    _print_section(all_metrics.get('attribution'),'[6] Attribution')
    print("=" * 60)

    json_path = os.path.join(args.output_dir, 'structure_metrics.json')
    with open(json_path, 'w') as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nAll metrics → {json_path}")
    print(f"All plots   → {args.output_dir}/")


def _print_section(d: Optional[dict], title: str):
    if d is None:
        return
    print(f"\n{title}")
    print("-" * 40)
    for k, v in d.items():
        if isinstance(v, float):
            print(f"  {k:<30}: {v:.4f}")
        elif isinstance(v, list) and len(v) <= 10:
            print(f"  {k:<30}: {v}")
        elif not isinstance(v, list):
            print(f"  {k:<30}: {v}")


if __name__ == '__main__':
    main()
