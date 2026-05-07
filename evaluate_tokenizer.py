"""
evaluate_tokenizer.py
=====================
Comprehensive evaluation of a trained VQ-NSP tokenizer.

Covers four evaluation dimensions:

  [1] Training curves    — parse log.txt, plot loss trajectories and
                           codebook utilisation across epochs
  [2] Codebook health    — utilisation rate, usage distribution, pairwise
                           similarity (reuses vis_codebook helpers)
  [3] Reconstruction     — encode→decode on real EEG; compute MSE, cosine
                           similarity, and SNR between true and predicted
                           amplitude / phase spectra; plot examples
  [4] Summary report     — print + save a human-readable score-card

Usage (no EEG data needed for dims 1-2):
    python evaluate_tokenizer.py \\
        --checkpoint EEGfounder/checkpoints/tokenizer/checkpoint.pth \\
        --log_txt    EEGfounder/checkpoints/tokenizer/log.txt \\
        --output_dir eval_out

Full evaluation (all four dimensions):
    python evaluate_tokenizer.py \\
        --checkpoint EEGfounder/checkpoints/tokenizer/checkpoint.pth \\
        --log_txt    EEGfounder/checkpoints/tokenizer/log.txt \\
        --data_path  /data/eeg/dataset.hdf5 \\
        --output_dir eval_out
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from einops import rearrange

# ── reuse helpers from vis_codebook ──────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from vis_codebook import (
    build_model,
    extract_codebook,
    panel_usage,
    panel_similarity,
    _load_eeg_hdf5,
)

import tokenizer as _tok_module  # noqa: F401 — registers @register_model entries


# ═════════════════════════════════════════════════════════════════════════════
# Dim 1 — Training curves
# ═════════════════════════════════════════════════════════════════════════════

def parse_log(log_path: str) -> dict[str, list]:
    """
    Parse a JSONL training log (one JSON dict per line).
    Returns a dict mapping metric name → list of values (one per epoch).
    """
    records = []
    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    if not records:
        return {}

    keys = set()
    for r in records:
        keys.update(r.keys())

    curves: dict[str, list] = {k: [] for k in sorted(keys)}
    for r in records:
        for k in keys:
            curves[k].append(r.get(k, float('nan')))
    return curves


def plot_training_curves(curves: dict[str, list], out_dir: str) -> dict[str, float]:
    """
    Plot training/validation loss curves and codebook utilisation.
    Returns a summary dict with final-epoch values.
    """
    epochs = curves.get('epoch', list(range(len(next(iter(curves.values()))))))

    # ── group keys ────────────────────────────────────────────────────────────
    train_loss_keys = sorted(k for k in curves if k.startswith('train_') and 'loss' in k)
    val_loss_keys   = sorted(k for k in curves if k.startswith('val_')   and 'loss' in k)
    util_key        = 'train_unused_codes' if 'train_unused_codes' in curves else None

    n_rows = 2 + (1 if util_key else 0)
    fig, axes = plt.subplots(n_rows, 1, figsize=(10, 4 * n_rows), sharex=True)
    axes = np.array(axes).ravel()

    # ── train losses ──────────────────────────────────────────────────────────
    ax = axes[0]
    colors = plt.get_cmap('tab10', max(len(train_loss_keys), 1))
    for i, k in enumerate(train_loss_keys):
        vals = np.array(curves[k], dtype=float)
        label = k.replace('train_', '')
        ax.plot(epochs, vals, label=label, color=colors(i), linewidth=1.5)
    ax.set_ylabel('Loss', fontsize=11)
    ax.set_title('Training losses per epoch', fontsize=12)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

    # ── val losses ────────────────────────────────────────────────────────────
    ax2 = axes[1]
    colors2 = plt.get_cmap('tab10', max(len(val_loss_keys), 1))
    paired = {}   # match train key to val key for gap annotation
    for i, k in enumerate(val_loss_keys):
        vals = np.array(curves[k], dtype=float)
        label = k.replace('val_', '')
        ax2.plot(epochs, vals, label=label, color=colors2(i), linewidth=1.5)
        train_k = k.replace('val_', 'train_')
        if train_k in curves:
            paired[label] = {
                'train_final': float(np.array(curves[train_k])[-1]),
                'val_final':   float(vals[-1]),
            }
    ax2.set_ylabel('Loss', fontsize=11)
    ax2.set_title('Validation losses per epoch', fontsize=12)
    ax2.legend(fontsize=8, ncol=2)
    ax2.grid(True, alpha=0.3)

    # ── codebook utilisation ──────────────────────────────────────────────────
    summary: dict[str, float] = {}
    if util_key:
        ax3 = axes[2]
        unused = np.array(curves[util_key], dtype=float)
        # infer K from the first record that has it, or use default 8192
        K = 8192
        utilisation = 100.0 * (K - unused) / K
        ax3.plot(epochs, utilisation, color='green', linewidth=2)
        ax3.axhline(80, color='orange', linestyle='--', linewidth=1,
                    label='80% threshold')
        ax3.axhline(50, color='red', linestyle='--', linewidth=1,
                    label='collapse warning (50%)')
        ax3.set_xlabel('Epoch', fontsize=11)
        ax3.set_ylabel('Active codes (%)', fontsize=11)
        ax3.set_title('Codebook utilisation over training', fontsize=12)
        ax3.set_ylim(0, 105)
        ax3.legend(fontsize=9)
        ax3.grid(True, alpha=0.3)
        summary['final_utilisation_pct'] = float(utilisation[-1])
        summary['max_utilisation_pct']   = float(utilisation.max())
    else:
        axes[-1].set_xlabel('Epoch', fontsize=11)

    fig.tight_layout()
    path = os.path.join(out_dir, 'eval_training_curves.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  → {path}")

    # collect final-epoch values for summary
    for k, v in curves.items():
        arr = np.array(v, dtype=float)
        finite = arr[np.isfinite(arr)]
        if len(finite):
            summary[f'final_{k}'] = float(finite[-1])

    return summary


# ═════════════════════════════════════════════════════════════════════════════
# Dim 2 — Codebook health (wraps vis_codebook helpers)
# ═════════════════════════════════════════════════════════════════════════════

def evaluate_codebook_health(
    model,
    out_dir: str,
    sim_max_show: int = 256,
) -> dict[str, float]:
    """
    Compute codebook health metrics and save visualisations.
    Returns a summary dict.
    """
    codes, usage = extract_codebook(model)
    K, D = codes.shape
    n_active  = int((usage > 0).sum())
    util_pct  = 100.0 * n_active / K

    # ── pairwise cosine similarity stats (no full plot needed here) ───────────
    max_show = min(sim_max_show, K)
    top_idx  = np.sort(np.argsort(usage)[::-1][:max_show])
    subset   = codes[top_idx]
    norms    = np.linalg.norm(subset, axis=1, keepdims=True) + 1e-9
    normed   = subset / norms
    sim      = normed @ normed.T
    upper    = sim[np.triu_indices(len(subset), k=1)]
    mean_abs_sim = float(np.abs(upper).mean())
    max_off_diag = float(upper.max())

    # usage statistics
    sorted_u = np.sort(usage)[::-1]
    total    = sorted_u.sum() + 1e-9
    cumsum   = np.cumsum(sorted_u) / total
    top5_coverage  = float(cumsum[max(0, int(K * 0.05) - 1)] * 100)
    top20_coverage = float(cumsum[max(0, int(K * 0.20) - 1)] * 100)

    # save panel_usage and panel_similarity to eval_out
    panel_usage(usage, out_dir)
    panel_similarity(codes, usage, out_dir, max_show=sim_max_show)
    # rename outputs to eval_* prefix
    for src, dst in [
        ('codebook_usage.png',      'eval_codebook_usage.png'),
        ('codebook_similarity.png', 'eval_codebook_similarity.png'),
    ]:
        s = os.path.join(out_dir, src)
        d = os.path.join(out_dir, dst)
        if os.path.exists(s):
            os.replace(s, d)
            print(f"  → {d}")

    return {
        'K':                    K,
        'D':                    D,
        'n_active':             n_active,
        'utilisation_pct':      round(util_pct, 2),
        'mean_abs_cosine_sim':  round(mean_abs_sim, 4),
        'max_off_diag_sim':     round(max_off_diag, 4),
        'top5pct_code_coverage':  round(top5_coverage, 1),
        'top20pct_code_coverage': round(top20_coverage, 1),
    }


# ═════════════════════════════════════════════════════════════════════════════
# Dim 3 — Reconstruction quality
# ═════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate_reconstruction(
    model,
    data_path: str,
    out_dir: str,
    device: torch.device,
    n_samples: int = 512,
    n_plot_examples: int = 4,
) -> dict[str, float]:
    """
    Encode then decode a set of EEG segments; compare the predicted amplitude
    and phase spectra against the ground-truth FFT targets.

    Metrics returned:
      amp_mse, amp_cosine_sim, amp_snr_db
      phase_mse, phase_cosine_sim
      commit_loss_mean
    """
    print(f"  loading ≤{n_samples} samples from {data_path} …", flush=True)
    raw = _load_eeg_hdf5(data_path, n_samples)        # (B, C, T)
    B, C, T = raw.shape

    T_patch  = 200
    EEG_size = model.encoder.patch_embed.num_patches * model.patch_size
    A        = EEG_size // T_patch
    T_needed = A * T_patch

    if T < T_needed:
        raw = np.pad(raw, ((0, 0), (0, 0), (0, T_needed - T)))
    else:
        raw = raw[:, :, :T_needed]

    n_ch = model.token_shape[0]
    data_t = torch.tensor(raw, dtype=torch.float32)
    if C != n_ch:
        if C < n_ch:
            pad = torch.zeros(B, n_ch - C, T_needed)
            data_t = torch.cat([data_t, pad], dim=1)
        else:
            data_t = data_t[:, :n_ch, :]

    # ── accumulate metrics over batches ───────────────────────────────────────
    batch_size  = 16
    all_amp_mse, all_phase_mse = [], []
    all_amp_cos, all_phase_cos = [], []
    all_amp_snr                = []
    all_commit                 = []

    # store a few examples for plotting
    examples: list[dict] = []

    for start in range(0, B, batch_size):
        chunk = data_t[start:start + batch_size].to(device) / 100.0    # (b, C, T)
        b     = chunk.shape[0]

        # reshape to (b, C, A, T_patch)
        x4d = chunk.view(b, n_ch, A, T_patch)

        # ── compute ground-truth FFT targets (same as VQNSP.forward) ──────────
        fft     = torch.fft.fft(x4d, dim=-1)
        amp_gt  = model._patch_normalize(torch.abs(fft))    # (b, C, A, T_patch)
        phase_gt = model._patch_normalize(torch.angle(fft))

        # ── encode → decode ────────────────────────────────────────────────────
        z_q, ids, vq_loss = model.encode(x4d)
        amp_pred, phase_pred, _ = model.decode(z_q)
        # amp_pred shape: (b, C*A, T_patch) — flatten spatial for comparison
        amp_gt_flat   = rearrange(amp_gt,   'b c a t -> b (c a) t')
        phase_gt_flat = rearrange(phase_gt, 'b c a t -> b (c a) t')

        # ── MSE ───────────────────────────────────────────────────────────────
        amp_mse   = F.mse_loss(amp_pred,   amp_gt_flat,   reduction='none').mean(dim=(1, 2))
        phase_mse = F.mse_loss(phase_pred, phase_gt_flat, reduction='none').mean(dim=(1, 2))
        all_amp_mse.extend(amp_mse.cpu().tolist())
        all_phase_mse.extend(phase_mse.cpu().tolist())

        # ── cosine similarity (averaged over patches) ─────────────────────────
        ap_norm = F.normalize(amp_pred,       dim=-1)
        ag_norm = F.normalize(amp_gt_flat,    dim=-1)
        pp_norm = F.normalize(phase_pred,     dim=-1)
        pg_norm = F.normalize(phase_gt_flat,  dim=-1)
        amp_cos   = (ap_norm * ag_norm).sum(-1).mean(-1)    # (b,)
        phase_cos = (pp_norm * pg_norm).sum(-1).mean(-1)
        all_amp_cos.extend(amp_cos.cpu().tolist())
        all_phase_cos.extend(phase_cos.cpu().tolist())

        # ── SNR (signal power / residual power) ───────────────────────────────
        residual_power = ((amp_pred - amp_gt_flat) ** 2).mean(dim=(1, 2))
        signal_power   = (amp_gt_flat ** 2).mean(dim=(1, 2)).clamp(min=1e-12)
        snr_db = 10.0 * torch.log10(signal_power / residual_power.clamp(min=1e-12))
        all_amp_snr.extend(snr_db.cpu().tolist())

        # ── commit loss ───────────────────────────────────────────────────────
        all_commit.append(vq_loss.item())

        # ── store first few examples ──────────────────────────────────────────
        if len(examples) < n_plot_examples:
            n_take = min(n_plot_examples - len(examples), b)
            for i in range(n_take):
                examples.append({
                    'amp_gt':    amp_gt_flat[i, 0].cpu().numpy(),   # first patch
                    'amp_pred':  amp_pred[i,    0].cpu().numpy(),
                    'phase_gt':  phase_gt_flat[i, 0].cpu().numpy(),
                    'phase_pred':phase_pred[i,  0].cpu().numpy(),
                    'snr_db':    float(snr_db[i].item()),
                })

    metrics = {
        'amp_mse':         round(float(np.mean(all_amp_mse)),  5),
        'amp_cosine_sim':  round(float(np.mean(all_amp_cos)),  4),
        'amp_snr_db':      round(float(np.mean(all_amp_snr)),  2),
        'phase_mse':       round(float(np.mean(all_phase_mse)),5),
        'phase_cosine_sim':round(float(np.mean(all_phase_cos)),4),
        'commit_loss_mean':round(float(np.mean(all_commit)),   5),
    }

    # ── reconstruction example plots ──────────────────────────────────────────
    _plot_reconstruction_examples(examples, out_dir)

    # ── metric distribution plots ─────────────────────────────────────────────
    _plot_metric_distributions(
        all_amp_mse, all_amp_cos, all_amp_snr,
        all_phase_mse, all_phase_cos,
        out_dir,
    )

    return metrics


def _plot_reconstruction_examples(examples: list[dict], out_dir: str) -> None:
    n = len(examples)
    if n == 0:
        return
    fig, axes = plt.subplots(n, 2, figsize=(13, 3 * n))
    if n == 1:
        axes = axes[np.newaxis, :]
    freq = np.arange(examples[0]['amp_gt'].shape[0])

    for i, ex in enumerate(examples):
        # amplitude
        ax = axes[i, 0]
        ax.plot(freq, ex['amp_gt'],   label='target',    linewidth=1.2, color='steelblue')
        ax.plot(freq, ex['amp_pred'], label='predicted', linewidth=1.2, color='tomato',
                linestyle='--')
        ax.set_title(f'Sample {i+1} — amplitude spectrum  (SNR={ex["snr_db"]:.1f} dB)',
                     fontsize=10)
        ax.set_xlabel('Frequency bin'); ax.set_ylabel('Normalised amplitude')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # phase
        ax2 = axes[i, 1]
        ax2.plot(freq, ex['phase_gt'],   label='target',    linewidth=1.2, color='steelblue')
        ax2.plot(freq, ex['phase_pred'], label='predicted', linewidth=1.2, color='tomato',
                 linestyle='--')
        ax2.set_title(f'Sample {i+1} — phase spectrum', fontsize=10)
        ax2.set_xlabel('Frequency bin'); ax2.set_ylabel('Normalised phase')
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)

    fig.suptitle('Encode → decode reconstruction examples (first patch of each sample)',
                 fontsize=11)
    fig.tight_layout()
    path = os.path.join(out_dir, 'eval_reconstruction_examples.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  → {path}")


def _plot_metric_distributions(
    amp_mse, amp_cos, amp_snr,
    phase_mse, phase_cos,
    out_dir: str,
) -> None:
    fig, axes = plt.subplots(1, 5, figsize=(18, 4))
    pairs = [
        (amp_mse,   'Amplitude MSE',         'steelblue'),
        (amp_cos,   'Amplitude cosine sim',  'seagreen'),
        (amp_snr,   'Amplitude SNR (dB)',    'darkorange'),
        (phase_mse, 'Phase MSE',             'mediumpurple'),
        (phase_cos, 'Phase cosine sim',      'crimson'),
    ]
    for ax, (data, title, color) in zip(axes, pairs):
        ax.hist(data, bins=40, color=color, alpha=0.75, edgecolor='white')
        ax.axvline(np.mean(data), color='k', linewidth=1.5, linestyle='--',
                   label=f'mean={np.mean(data):.3f}')
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle('Reconstruction metric distributions across EEG segments', fontsize=11)
    fig.tight_layout()
    path = os.path.join(out_dir, 'eval_reconstruction_distributions.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  → {path}")


# ═════════════════════════════════════════════════════════════════════════════
# Dim 4 — Summary report
# ═════════════════════════════════════════════════════════════════════════════

def _grade(value: float, thresholds: tuple, labels: tuple) -> str:
    """Map a scalar value to a grade label given threshold breakpoints."""
    for thresh, label in zip(thresholds, labels):
        if value >= thresh:
            return label
    return labels[-1]


def print_and_save_report(
    codebook_metrics: dict,
    curve_metrics: dict,
    recon_metrics: Optional[dict],
    out_dir: str,
    checkpoint: str,
) -> None:
    lines = []
    sep   = '=' * 60

    def add(s=''):
        lines.append(s)
        print(s)

    add(sep)
    add(' TOKENIZER EVALUATION REPORT')
    add(f' checkpoint: {checkpoint}')
    add(sep)

    # ── codebook health ───────────────────────────────────────────────────────
    add('\n[1] CODEBOOK HEALTH')
    add('-' * 40)
    K     = codebook_metrics['K']
    D     = codebook_metrics['D']
    util  = codebook_metrics['utilisation_pct']
    n_act = codebook_metrics['n_active']

    util_grade = _grade(util,
        (90, 80, 60, 0),
        ('EXCELLENT', 'GOOD', 'WARNING — collapse risk', 'CRITICAL — collapsed'))
    add(f'  Codebook size      : K={K}, D={D}')
    add(f'  Active codes       : {n_act}/{K}  ({util:.1f}%)  [{util_grade}]')
    add(f'  Top-5%  coverage   : {codebook_metrics["top5pct_code_coverage"]:.1f}%  '
        f'(ideal ≈ 5%, high = unbalanced)')
    add(f'  Top-20% coverage   : {codebook_metrics["top20pct_code_coverage"]:.1f}%')
    add(f'  Mean |cosine sim|  : {codebook_metrics["mean_abs_cosine_sim"]:.4f}  '
        f'(ideal < 0.2 — codes should be diverse)')
    add(f'  Max off-diag sim   : {codebook_metrics["max_off_diag_sim"]:.4f}  '
        f'(ideal < 0.5 — no redundant pairs)')

    # ── training curves ───────────────────────────────────────────────────────
    add('\n[2] TRAINING CURVES (final epoch)')
    add('-' * 40)
    if curve_metrics:
        interesting = [
            ('final_train_quant_loss',    'Train quant loss '),
            ('final_train_rec_loss',      'Train amp rec loss'),
            ('final_train_rec_angle_loss','Train phase loss  '),
            ('final_val_quant_loss',      'Val   quant loss  '),
            ('final_val_rec_loss',        'Val   amp rec loss'),
            ('final_val_rec_angle_loss',  'Val   phase loss  '),
            ('final_utilisation_pct',     'Codebook util (%) '),
        ]
        for key, label in interesting:
            if key in curve_metrics:
                add(f'  {label}: {curve_metrics[key]:.5g}')

        # train/val gap check
        for base in ('quant_loss', 'rec_loss'):
            tk = f'final_train_{base}'
            vk = f'final_val_{base}'
            if tk in curve_metrics and vk in curve_metrics:
                gap = curve_metrics[vk] - curve_metrics[tk]
                gap_pct = 100 * gap / (curve_metrics[tk] + 1e-9)
                flag = '  ← overfit?' if gap_pct > 20 else ''
                add(f'  train/val gap ({base}): {gap_pct:+.1f}%{flag}')
    else:
        add('  (no log.txt provided)')

    # ── reconstruction ────────────────────────────────────────────────────────
    add('\n[3] RECONSTRUCTION QUALITY')
    add('-' * 40)
    if recon_metrics:
        amp_cos   = recon_metrics['amp_cosine_sim']
        amp_snr   = recon_metrics['amp_snr_db']
        phase_cos = recon_metrics['phase_cosine_sim']

        cos_grade = _grade(amp_cos,
            (0.90, 0.75, 0.50, 0),
            ('EXCELLENT', 'GOOD', 'FAIR', 'POOR'))
        snr_grade = _grade(amp_snr,
            (15, 8, 3, -999),
            ('EXCELLENT', 'GOOD', 'FAIR', 'POOR'))

        add(f'  Amplitude MSE           : {recon_metrics["amp_mse"]:.5g}')
        add(f'  Amplitude cosine sim    : {amp_cos:.4f}  [{cos_grade}]')
        add(f'  Amplitude SNR           : {amp_snr:.2f} dB  [{snr_grade}]')
        add(f'  Phase MSE               : {recon_metrics["phase_mse"]:.5g}')
        add(f'  Phase cosine sim        : {phase_cos:.4f}')
        add(f'  VQ commit loss (mean)   : {recon_metrics["commit_loss_mean"]:.5g}')
    else:
        add('  (no --data_path provided; skipped)')

    # ── overall verdict ───────────────────────────────────────────────────────
    add(f'\n{"─"*40}')
    add(' OVERALL VERDICT')
    issues = []
    if codebook_metrics['utilisation_pct'] < 60:
        issues.append('codebook collapse (utilisation < 60%)')
    if codebook_metrics['mean_abs_cosine_sim'] > 0.4:
        issues.append('redundant codes (high mean cosine similarity)')
    if recon_metrics:
        if recon_metrics['amp_cosine_sim'] < 0.5:
            issues.append('poor amplitude reconstruction (cosine sim < 0.5)')
        if recon_metrics['amp_snr_db'] < 3:
            issues.append('very low SNR (< 3 dB)')

    if not issues:
        add(' No major issues detected. Tokenizer looks healthy.')
    else:
        add(' Issues detected:')
        for iss in issues:
            add(f'   ✗ {iss}')
    add(sep)

    # save to file
    path = os.path.join(out_dir, 'eval_report.txt')
    with open(path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'\n  Report saved → {path}')

    # also save metrics as JSON for downstream scripting
    all_metrics = {
        'codebook':      codebook_metrics,
        'curves':        curve_metrics,
        'reconstruction': recon_metrics or {},
    }
    json_path = os.path.join(out_dir, 'eval_metrics.json')
    with open(json_path, 'w') as f:
        json.dump(all_metrics, f, indent=2)
    print(f'  Metrics JSON  → {json_path}')


# ═════════════════════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════════════════════

def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser('evaluate_tokenizer', add_help=True)

    # ── required ──────────────────────────────────────────────────────────────
    p.add_argument('--checkpoint', required=True,
                   help='Path to trained tokenizer .pth checkpoint')

    # ── model architecture ────────────────────────────────────────────────────
    p.add_argument('--model',     default='vqnsp_encoder_base_decoder_3x200x12')
    p.add_argument('--n_embed',   default=8192, type=int)
    p.add_argument('--embed_dim', default=32,   type=int)
    p.add_argument('--eeg_size',  default=1600, type=int)

    # ── I/O ───────────────────────────────────────────────────────────────────
    p.add_argument('--output_dir', default='eval_tokenizer_out')
    p.add_argument('--device',     default='cpu')
    p.add_argument('--log_txt',    default=None,
                   help='Path to training log.txt for loss curve analysis')

    # ── reconstruction (dim 3) ────────────────────────────────────────────────
    p.add_argument('--data_path',      default=None,
                   help='HDF5 EEG file for reconstruction evaluation')
    p.add_argument('--n_data_samples', default=512, type=int,
                   help='Number of EEG segments to evaluate reconstruction on')
    p.add_argument('--n_plot_examples',default=4, type=int,
                   help='Number of example reconstructions to plot')

    # ── codebook vis options ──────────────────────────────────────────────────
    p.add_argument('--sim_max_show', default=256, type=int)

    # ── skip flags ────────────────────────────────────────────────────────────
    p.add_argument('--skip_curves', action='store_true',
                   help='Skip training curve analysis (dim 1)')
    p.add_argument('--skip_recon',  action='store_true',
                   help='Skip reconstruction evaluation (dim 3)')

    return p.parse_args()


# ═════════════════════════════════════════════════════════════════════════════
# Entry point
# ═════════════════════════════════════════════════════════════════════════════

def main() -> None:
    args = get_args()
    device = torch.device(args.device)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    print(f"Loading model from {args.checkpoint} …")
    model = build_model(args, device)

    curve_metrics: dict = {}
    recon_metrics: Optional[dict] = None

    # ── Dim 1: training curves ─────────────────────────────────────────────
    print('\n[Dim 1] Training curves …')
    if not args.skip_curves and args.log_txt and os.path.exists(args.log_txt):
        curves = parse_log(args.log_txt)
        if curves:
            curve_metrics = plot_training_curves(curves, args.output_dir)
        else:
            print('  log.txt is empty or unparseable — skipping')
    elif not args.log_txt:
        print('  (no --log_txt provided — skipping)')
    else:
        print(f'  log.txt not found at {args.log_txt} — skipping')

    # ── Dim 2: codebook health ─────────────────────────────────────────────
    print('\n[Dim 2] Codebook health …')
    codebook_metrics = evaluate_codebook_health(
        model, args.output_dir, sim_max_show=args.sim_max_show,
    )

    # ── Dim 3: reconstruction quality ─────────────────────────────────────
    print('\n[Dim 3] Reconstruction quality …')
    if not args.skip_recon and args.data_path:
        recon_metrics = evaluate_reconstruction(
            model, args.data_path, args.output_dir, device,
            n_samples=args.n_data_samples,
            n_plot_examples=args.n_plot_examples,
        )
    elif not args.data_path:
        print('  (no --data_path provided — skipping)')

    # ── Dim 4: summary report ──────────────────────────────────────────────
    print('\n[Dim 4] Summary report …')
    print_and_save_report(
        codebook_metrics=codebook_metrics,
        curve_metrics=curve_metrics,
        recon_metrics=recon_metrics,
        out_dir=args.output_dir,
        checkpoint=args.checkpoint,
    )


if __name__ == '__main__':
    main()
