"""
vis_codebook.py
===============
Visualise the learned VQ-NSP codebook from a trained tokenizer checkpoint.

Five panels (each saved as a separate PNG):
  1. codebook_embedding.png   — t-SNE / UMAP projection coloured by usage frequency
  2. codebook_usage.png       — usage distribution + cumulative coverage (collapse diagnosis)
  3. codebook_similarity.png  — pairwise cosine-similarity heat-map for top codes
  4. codebook_spectra.png     — decoded spectral signature of top-k most-used codes
  5. codebook_topo.png        — dominant token per channel + per-channel token entropy
                                (requires --data_path)

Usage:
    # panels 1-4 only (no EEG data needed):
    python vis_codebook.py --checkpoint path/to/ckpt.pth --output_dir ./vis_out

    # all five panels:
    python vis_codebook.py --checkpoint path/to/ckpt.pth \\
        --data_path path/to/data.hdf5 \\
        --output_dir ./vis_out

Optional flags:
    --umap               use UMAP instead of t-SNE for panel 1
    --sim_max_show 256   max codes shown in similarity heatmap (default 256)
    --spectra_top_k 32   how many top-used codes to decode for panel 4
    --n_data_samples 2048
    --skip_spectra       skip panel 4 (faster, no decoder forward)
    --skip_topo          skip panel 5
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')   # non-interactive backend — safe for servers
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from timm.models import create_model

import tokenizer as _tok_module  # noqa: F401 — registers @register_model entries


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint loading
# ─────────────────────────────────────────────────────────────────────────────

def _load_state_dict(path: str, device: torch.device) -> dict:
    """Load a state dict from a raw model file or a training checkpoint."""
    raw = torch.load(path, map_location=device)
    state = raw.get('model', raw.get('state_dict', raw))
    # strip DDP prefix if present
    cleaned = {}
    for k, v in state.items():
        cleaned[k.removeprefix('module.')] = v
    return cleaned


def build_model(args: argparse.Namespace, device: torch.device):
    model = create_model(
        args.model,
        pretrained=False,
        as_tokenzer=False,
        n_code=args.n_embed,
        code_dim=args.embed_dim,
        EEG_size=args.eeg_size,
    )
    state = _load_state_dict(args.checkpoint, device)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"  [warn] {len(missing)} missing keys (e.g. {missing[:3]})")
    if unexpected:
        print(f"  [warn] {len(unexpected)} unexpected keys")
    model.to(device).eval()
    return model


def extract_codebook(model) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns
    -------
    codes : (K, D) float32 ndarray — codebook embedding matrix
    usage : (K,)  float32 ndarray — EMA cluster-size counts (zeros if unavailable)
    """
    codes = model.quantize.embedding.weight.data.cpu().float().numpy()
    if hasattr(model.quantize, 'cluster_size'):
        usage = model.quantize.cluster_size.data.cpu().float().numpy()
    else:
        usage = np.zeros(len(codes), dtype=np.float32)
    return codes, usage


# ─────────────────────────────────────────────────────────────────────────────
# Panel 1 — embedding space (t-SNE / UMAP)
# ─────────────────────────────────────────────────────────────────────────────

def panel_embedding(
    codes: np.ndarray,
    usage: np.ndarray,
    out_dir: str,
    use_umap: bool = False,
) -> None:
    """2-D projection of all K codebook vectors, coloured by log usage."""
    K = codes.shape[0]
    print(f"  projecting {K} vectors …", flush=True)

    method = 't-SNE'
    if use_umap:
        try:
            import umap as _umap
            reducer = _umap.UMAP(n_components=2, random_state=42, verbose=False)
            xy = reducer.fit_transform(codes)
            method = 'UMAP'
        except ImportError:
            print("  [warn] umap-learn not installed, falling back to t-SNE")

    if method == 't-SNE':
        from sklearn.manifold import TSNE
        perp = min(30, max(5, K // 100))
        xy = TSNE(n_components=2, random_state=42, perplexity=perp,
                  n_iter=1000).fit_transform(codes)

    log_usage = np.log1p(usage)

    fig, ax = plt.subplots(figsize=(8, 7))
    sc = ax.scatter(
        xy[:, 0], xy[:, 1],
        c=log_usage, cmap='viridis', s=4, alpha=0.6, linewidths=0,
    )
    cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('log(1 + usage count)', fontsize=10)
    ax.set_title(f'Codebook {method} — K={K}', fontsize=13)
    ax.set_xlabel(f'{method}-1'); ax.set_ylabel(f'{method}-2')
    ax.set_aspect('equal')

    dead = np.where(usage == 0)[0]
    if len(dead):
        ax.scatter(xy[dead, 0], xy[dead, 1], c='red', s=8, alpha=0.9,
                   zorder=5, label=f'unused ({len(dead)})')
        ax.legend(fontsize=9)

    path = os.path.join(out_dir, 'codebook_embedding.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Panel 2 — usage distribution + cumulative coverage
# ─────────────────────────────────────────────────────────────────────────────

def panel_usage(usage: np.ndarray, out_dir: str) -> None:
    K = len(usage)
    sorted_u = np.sort(usage)[::-1]
    total = sorted_u.sum() + 1e-9
    cumsum = np.cumsum(sorted_u) / total
    n_active = int((usage > 0).sum())
    n_dead = K - n_active

    fig, (ax_bar, ax_cum) = plt.subplots(1, 2, figsize=(14, 5))

    # ── left: sorted usage bar ──────────────────────────────────────────────
    colors = ['#d62728' if v == 0 else '#1f77b4' for v in sorted_u]
    ax_bar.bar(np.arange(K), sorted_u, color=colors, width=1.0, linewidth=0)
    ax_bar.set_yscale('symlog', linthresh=1)
    ax_bar.set_xlabel('Code rank (sorted by frequency)', fontsize=11)
    ax_bar.set_ylabel('Usage count (symlog)', fontsize=11)
    ax_bar.set_title(
        f'Usage per code  |  active: {n_active}/{K} ({100*n_active/K:.1f}%)'
        f'  |  dead (red): {n_dead}',
        fontsize=10,
    )
    # annotate dead fraction with a text box
    ax_bar.text(
        0.98, 0.97,
        f'codebook utilisation\n{100*n_active/K:.1f}%',
        transform=ax_bar.transAxes, ha='right', va='top', fontsize=10,
        bbox=dict(boxstyle='round', fc='white', alpha=0.8),
    )

    # ── right: cumulative coverage ──────────────────────────────────────────
    rank_pct = np.arange(1, K + 1) / K * 100
    ax_cum.plot(rank_pct, cumsum * 100, linewidth=1.8, color='steelblue')
    for p in (0.5, 0.8, 0.95):
        idx = int(np.searchsorted(cumsum, p))
        ax_cum.axhline(p * 100, color='grey', linestyle='--', linewidth=0.7)
        ax_cum.axvline(idx / K * 100, color='grey', linestyle='--', linewidth=0.7)
        ax_cum.annotate(
            f'{p*100:.0f}% @ top {idx/K*100:.1f}%',
            xy=(idx / K * 100, p * 100),
            xytext=(min(idx / K * 100 + 3, 70), p * 100 - 6),
            fontsize=8, color='dimgrey',
        )
    ax_cum.set_xlabel('Fraction of code vocabulary used (%)', fontsize=11)
    ax_cum.set_ylabel('Cumulative usage (%)', fontsize=11)
    ax_cum.set_title('Cumulative coverage', fontsize=11)
    ax_cum.set_xlim(0, 100); ax_cum.set_ylim(0, 101)

    fig.tight_layout()
    path = os.path.join(out_dir, 'codebook_usage.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Panel 3 — pairwise cosine similarity heat-map
# ─────────────────────────────────────────────────────────────────────────────

def panel_similarity(
    codes: np.ndarray,
    usage: np.ndarray,
    out_dir: str,
    max_show: int = 256,
) -> None:
    K = codes.shape[0]
    suffix = ''
    if K > max_show:
        top_idx = np.sort(np.argsort(usage)[::-1][:max_show])
        subset = codes[top_idx]
        suffix = f' (top-{max_show} by usage)'
    else:
        subset = codes

    # cosine similarity via L2-normalised dot product
    norms = np.linalg.norm(subset, axis=1, keepdims=True) + 1e-9
    normed = subset / norms
    sim = normed @ normed.T          # (M, M), values in [-1, 1]

    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(sim, cmap='RdBu_r', vmin=-1, vmax=1,
                   aspect='auto', interpolation='nearest')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label(
        'Cosine similarity', fontsize=10)
    ax.set_title(f'Pairwise cosine similarity{suffix}', fontsize=12)
    ax.set_xlabel('Code index'); ax.set_ylabel('Code index')

    upper = sim[np.triu_indices(len(subset), k=1)]
    ax.text(
        0.02, 0.97,
        f'mean |sim| = {np.abs(upper).mean():.3f}\n'
        f'max off-diag = {upper.max():.3f}',
        transform=ax.transAxes, fontsize=9, va='top',
        bbox=dict(boxstyle='round', fc='white', alpha=0.8),
    )

    path = os.path.join(out_dir, 'codebook_similarity.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Panel 4 — decoded spectral signatures
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def panel_spectra(
    model,
    codes_np: np.ndarray,
    usage_np: np.ndarray,
    out_dir: str,
    device: torch.device,
    top_k: int = 32,
) -> None:
    """
    For the top-k most-used codebook entries:
      1. Tile the single code vector across the full spatial layout (H channels × W patches).
      2. Pass through the decoder to get predicted amplitude spectra.
      3. Average over spatial positions → one spectrum per code.
    This reveals what frequency content each code "represents".
    """
    K, D = codes_np.shape
    top_k = min(top_k, K)
    top_idx = np.argsort(usage_np)[::-1][:top_k]

    H, W = model.token_shape   # (n_channels, n_time_patches) e.g. (62, 8)
    codes_t = torch.tensor(codes_np, dtype=torch.float32, device=device)

    spectra = []
    for idx in top_idx:
        # tile code vector across the full (H, W) spatial grid
        c = codes_t[idx]                         # (D,)
        z_q = c.view(1, D, 1, 1).expand(1, D, H, W).contiguous()   # (1, D, H, W)
        amp_pred, _, _ = model.decode(z_q)       # (1, H*W, decoder_out_dim)
        # mean over spatial positions
        spectrum = amp_pred.squeeze(0).mean(0).cpu().float().numpy()  # (decoder_out_dim,)
        spectra.append(spectrum)

    spectra = np.array(spectra)   # (top_k, decoder_out_dim)
    freq_bins = np.arange(spectra.shape[1])
    cmap = plt.get_cmap('tab20', top_k)

    fig, axes = plt.subplots(2, 1, figsize=(13, 8),
                             gridspec_kw={'height_ratios': [3, 1]})

    # ── top: individual spectra ──────────────────────────────────────────────
    ax = axes[0]
    for rank, (idx, spec) in enumerate(zip(top_idx, spectra)):
        label = f'#{idx} (n={int(usage_np[idx])})' if rank < 12 else None
        ax.plot(freq_bins, spec, color=cmap(rank), alpha=0.7,
                linewidth=1.0, label=label)
    ax.axhline(0, color='k', linewidth=0.5, linestyle='--')
    ax.set_xlabel('Frequency bin index', fontsize=11)
    ax.set_ylabel('Decoded amplitude (model units)', fontsize=11)
    ax.set_title(
        f'Spectral signatures decoded from top-{top_k} codebook entries\n'
        f'(each line = one code tiled across all {H}×{W} spatial positions)',
        fontsize=11,
    )
    ax.legend(fontsize=7, ncol=3, loc='upper right',
              title='code (EMA count)', title_fontsize=7)

    # ── bottom: heat-map of all top-k spectra ───────────────────────────────
    ax2 = axes[1]
    im = ax2.imshow(
        spectra, aspect='auto', cmap='RdBu_r',
        vmin=-np.abs(spectra).max(), vmax=np.abs(spectra).max(),
        interpolation='nearest',
    )
    ax2.set_xlabel('Frequency bin index', fontsize=10)
    ax2.set_ylabel('Code rank', fontsize=10)
    ax2.set_yticks(np.arange(top_k))
    ax2.set_yticklabels([str(i) for i in top_idx], fontsize=5)
    fig.colorbar(im, ax=ax2, orientation='vertical', fraction=0.02, pad=0.01)

    fig.tight_layout()
    path = os.path.join(out_dir, 'codebook_spectra.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Panel 5 — token spatial assignment on real EEG
# ─────────────────────────────────────────────────────────────────────────────

def _load_eeg_hdf5(path: str, n_samples: int) -> np.ndarray:
    """
    Flexible HDF5 reader.  Tries common dataset key names and returns an
    ndarray of shape (B, C, T).
    """
    import h5py
    with h5py.File(path, 'r') as f:
        keys = list(f.keys())
        found = None
        for candidate in ('eeg', 'data', 'X', 'signal', keys[0]):
            if candidate in f:
                found = candidate
                break
        if found is None:
            raise KeyError(f"Cannot find EEG data in {path}. Keys: {keys}")
        arr = f[found]
        n = min(n_samples, arr.shape[0])
        data = arr[:n][()]   # load into RAM
    # shape normalisation: accept (B, C, T), (B, T, C), (B, T)
    if data.ndim == 2:
        data = data[:, np.newaxis, :]        # add channel dim
    if data.shape[1] > data.shape[2]:       # likely (B, T, C) → transpose
        data = data.transpose(0, 2, 1)
    return data.astype(np.float32)


@torch.no_grad()
def panel_topo(
    model,
    data_path: str,
    out_dir: str,
    device: torch.device,
    n_samples: int = 2048,
    ch_names: list[str] | None = None,
) -> None:
    """
    Encode a batch of real EEG segments and show:
      (a) dominant code (mode) per channel as a colour-coded grid
      (b) normalised token entropy per channel (diversity of assignments)
    """
    try:
        import h5py  # noqa: F401
    except ImportError:
        print("  [warn] h5py not installed — skipping panel 5")
        return

    print(f"  loading ≤{n_samples} samples from {data_path} …", flush=True)
    raw = _load_eeg_hdf5(data_path, n_samples)   # (B, C, T)
    B, C, T = raw.shape

    # model expects input shape (B, N_ch, A, T_patch) where T_patch = 200
    T_patch = 200
    EEG_size = model.encoder.patch_embed.num_patches * model.patch_size
    A = EEG_size // T_patch          # number of amplitude segments per channel

    # trim or pad to exactly EEG_size time points
    T_needed = A * T_patch
    if T < T_needed:
        raw = np.pad(raw, ((0, 0), (0, 0), (0, T_needed - T)))
    else:
        raw = raw[:, :, :T_needed]

    data_t = torch.tensor(raw, dtype=torch.float32)   # (B, C, T_needed)
    n_ch_model = model.token_shape[0]                  # expected number of channels (62)

    if C != n_ch_model:
        print(f"  [warn] data has {C} channels but model expects {n_ch_model}; "
              f"padding/trimming channel axis")
        if C < n_ch_model:
            pad = torch.zeros(B, n_ch_model - C, T_needed)
            data_t = torch.cat([data_t, pad], dim=1)
        else:
            data_t = data_t[:, :n_ch_model, :]

    H, W = model.token_shape   # (n_ch, n_time_patches)
    K_vocab = model.quantize.num_tokens
    batch_size = 32
    all_ids = []   # list of (batch, H, W) ndarrays

    for start in range(0, B, batch_size):
        chunk = data_t[start:start + batch_size].to(device) / 100.0   # (b, C, T)
        # reshape to (b, C, A, T_patch) for encode
        chunk_4d = chunk.view(chunk.shape[0], n_ch_model, A, T_patch)
        _, ids, _ = model.encode(chunk_4d)       # ids: (b * H * W,)
        ids_3d = ids.view(-1, H, W).cpu().numpy()   # (b, H, W)
        all_ids.append(ids_3d)

    all_ids = np.concatenate(all_ids, axis=0)   # (N, H, W)
    N = all_ids.shape[0]

    # ── per-channel statistics ───────────────────────────────────────────────
    # flatten time & sample axes: for channel ch we get N*W token assignments
    flat = all_ids.reshape(N * W, H)   # (N*W, H)

    mode_code   = np.zeros(H, dtype=int)
    mode_frac   = np.zeros(H, dtype=float)   # fraction of assignments to mode code
    code_entropy = np.zeros(H, dtype=float)  # normalised Shannon entropy

    for ch in range(H):
        counts = np.bincount(flat[:, ch].astype(int), minlength=K_vocab)
        mode_code[ch]    = counts.argmax()
        mode_frac[ch]    = counts.max() / (counts.sum() + 1e-9)
        p = counts / (counts.sum() + 1e-9)
        ent = -np.sum(p * np.log(p + 1e-9))
        code_entropy[ch] = ent / np.log(K_vocab)  # normalised ∈ [0, 1]

    # ── figure A: channel grid coloured by dominant code ────────────────────
    ncols = 11
    nrows = int(np.ceil(H / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 1.5, nrows * 1.5))
    axes_flat = np.array(axes).ravel()

    cmap20 = plt.get_cmap('tab20', 20)
    for ch in range(H):
        ax = axes_flat[ch]
        face_color = cmap20(mode_code[ch] % 20)
        ax.set_facecolor(face_color)
        label = ch_names[ch] if ch_names and ch < len(ch_names) else str(ch)
        ax.text(0.5, 0.65, label, ha='center', va='center',
                fontsize=6.5, transform=ax.transAxes, fontweight='bold')
        ax.text(0.5, 0.25,
                f'c={mode_code[ch]}\n({mode_frac[ch]*100:.0f}%)',
                ha='center', va='center', fontsize=5.5,
                transform=ax.transAxes, color='white')
        ax.set_xticks([]); ax.set_yticks([])

    for ch in range(H, len(axes_flat)):
        axes_flat[ch].set_visible(False)

    fig.suptitle(
        f'Dominant code per channel  |  {N} samples × {W} patches\n'
        'cell colour = code family (mod 20),  c = mode code index,  % = mode fraction',
        fontsize=10,
    )
    fig.tight_layout()
    path_a = os.path.join(out_dir, 'codebook_topo.png')
    fig.savefig(path_a, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  → {path_a}")

    # ── figure B: per-channel token entropy bar chart ───────────────────────
    fig2, ax2 = plt.subplots(figsize=(max(10, H // 4), 4))
    bar_colors = plt.get_cmap('coolwarm_r')(code_entropy)
    bars = ax2.bar(np.arange(H), code_entropy, color=bar_colors, width=0.8)

    ax2.axhline(1.0, color='grey', linestyle='--', linewidth=0.8,
                label='max entropy (uniform)')
    ax2.set_xlabel('Channel index', fontsize=11)
    ax2.set_ylabel('Normalised token entropy', fontsize=11)
    ax2.set_title(
        'Token assignment entropy per channel\n'
        '(1 = perfectly uniform  |  0 = always the same code)',
        fontsize=11,
    )
    ax2.set_xlim(-0.5, H - 0.5); ax2.set_ylim(0, 1.05)
    ax2.legend(fontsize=9)

    if ch_names:
        ax2.set_xticks(np.arange(H))
        ax2.set_xticklabels(ch_names[:H], rotation=45, ha='right', fontsize=7)

    # annotate mean
    ax2.text(0.99, 0.96,
             f'mean = {code_entropy.mean():.3f}',
             transform=ax2.transAxes, ha='right', va='top', fontsize=10,
             bbox=dict(boxstyle='round', fc='white', alpha=0.8))

    fig2.tight_layout()
    path_b = os.path.join(out_dir, 'codebook_channel_entropy.png')
    fig2.savefig(path_b, dpi=150, bbox_inches='tight')
    plt.close(fig2)
    print(f"  → {path_b}")

    # ── figure C: token co-occurrence heat-map (which channels share codes) ──
    # Build per-channel code histograms, then compute cosine similarity between
    # channel histograms to see which channels "speak the same token language"
    hist = np.zeros((H, K_vocab), dtype=np.float32)
    for ch in range(H):
        counts = np.bincount(flat[:, ch].astype(int), minlength=K_vocab)
        hist[ch] = counts.astype(np.float32)

    norms = np.linalg.norm(hist, axis=1, keepdims=True) + 1e-9
    hist_normed = hist / norms
    ch_sim = hist_normed @ hist_normed.T   # (H, H)

    fig3, ax3 = plt.subplots(figsize=(9, 8))
    im3 = ax3.imshow(ch_sim, cmap='YlOrRd', vmin=0, vmax=1,
                     aspect='auto', interpolation='nearest')
    fig3.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04).set_label(
        'Token histogram cosine similarity', fontsize=10)
    ax3.set_title('Channel-wise token vocabulary similarity', fontsize=12)
    ax3.set_xlabel('Channel'); ax3.set_ylabel('Channel')
    if ch_names:
        ticks = np.arange(H)
        ax3.set_xticks(ticks); ax3.set_xticklabels(ch_names[:H], rotation=90, fontsize=6)
        ax3.set_yticks(ticks); ax3.set_yticklabels(ch_names[:H], fontsize=6)

    fig3.tight_layout()
    path_c = os.path.join(out_dir, 'codebook_channel_vocab_sim.png')
    fig3.savefig(path_c, dpi=150, bbox_inches='tight')
    plt.close(fig3)
    print(f"  → {path_c}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser('vis_codebook', add_help=True)

    # ── required ──────────────────────────────────────────────────────────────
    p.add_argument('--checkpoint', required=True,
                   help='Path to trained tokenizer .pth checkpoint')

    # ── model architecture ────────────────────────────────────────────────────
    p.add_argument('--model',    default='vqnsp_encoder_base_decoder_3x200x12')
    p.add_argument('--n_embed',  default=8192, type=int,
                   help='Codebook size K (must match checkpoint)')
    p.add_argument('--embed_dim', default=32,  type=int,
                   help='Code dimensionality D (must match checkpoint)')
    p.add_argument('--eeg_size', default=1600, type=int,
                   help='Input EEG length in samples (must match checkpoint)')

    # ── I/O ───────────────────────────────────────────────────────────────────
    p.add_argument('--output_dir', default='vis_codebook_out',
                   help='Directory to write PNG files to')
    p.add_argument('--device',     default='cpu')

    # ── panel 1 ───────────────────────────────────────────────────────────────
    p.add_argument('--umap', action='store_true',
                   help='Use UMAP instead of t-SNE for the embedding panel')

    # ── panel 3 ───────────────────────────────────────────────────────────────
    p.add_argument('--sim_max_show', default=256, type=int,
                   help='Max codes shown in the similarity heatmap')

    # ── panel 4 ───────────────────────────────────────────────────────────────
    p.add_argument('--spectra_top_k', default=32, type=int,
                   help='Number of top-used codes to decode for spectral panel')
    p.add_argument('--skip_spectra', action='store_true',
                   help='Skip panel 4 (faster; avoids decoder forward pass)')

    # ── panel 5 ───────────────────────────────────────────────────────────────
    p.add_argument('--data_path', default=None,
                   help='HDF5 EEG file for panel 5 (token spatial assignment)')
    p.add_argument('--n_data_samples', default=2048, type=int,
                   help='Number of EEG segments to encode for panel 5')
    p.add_argument('--ch_names', default=None, nargs='+',
                   help='Optional list of channel names for axis labels in panel 5')
    p.add_argument('--skip_topo', action='store_true',
                   help='Skip panel 5 even if --data_path is given')

    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = get_args()
    device = torch.device(args.device)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    print(f"Loading model from {args.checkpoint} …")
    model = build_model(args, device)
    codes, usage = extract_codebook(model)
    K, D = codes.shape
    n_active = int((usage > 0).sum())
    print(
        f"Codebook: K={K}, D={D}\n"
        f"  active entries : {n_active}/{K} ({100*n_active/K:.1f}%)\n"
        f"  dead entries   : {K - n_active}"
    )

    print("\n[Panel 1] Codebook embedding …")
    panel_embedding(codes, usage, args.output_dir, use_umap=args.umap)

    print("[Panel 2] Usage distribution …")
    panel_usage(usage, args.output_dir)

    print("[Panel 3] Pairwise cosine similarity …")
    panel_similarity(codes, usage, args.output_dir, max_show=args.sim_max_show)

    if not args.skip_spectra:
        print(f"[Panel 4] Decoded spectral signatures (top-{args.spectra_top_k}) …")
        panel_spectra(model, codes, usage, args.output_dir, device,
                      top_k=args.spectra_top_k)
    else:
        print("[Panel 4] Skipped (--skip_spectra)")

    if not args.skip_topo:
        if args.data_path:
            print("[Panel 5] Token spatial assignment …")
            panel_topo(model, args.data_path, args.output_dir, device,
                       n_samples=args.n_data_samples, ch_names=args.ch_names)
        else:
            print("[Panel 5] Skipped — provide --data_path to enable")
    else:
        print("[Panel 5] Skipped (--skip_topo)")

    print(f"\nDone. All plots saved under {args.output_dir}/")


if __name__ == '__main__':
    main()
