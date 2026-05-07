"""
transfer_learning.py
====================
Transfer a fine-tuned EEG classification model from one domain to another
(e.g. adult EEG → pediatric EEG) using state-of-the-art transfer learning
strategies.

Strategies (--strategy, may combine with '+'):
  progressive     Gradual layer unfreezing + layer-wise LR decay (ULMFiT-style).
                  Stage 1: head only → Stage 2: head + last N blocks →
                  Stage 3: full model.  Best baseline; always recommended.

  dann            Domain-Adversarial Neural Networks.  A gradient-reversal layer
                  forces the backbone to produce domain-invariant features while
                  a domain classifier tries to separate source from target.
                  Requires unlabelled (or labelled) source data.

  mmd             Maximum Mean Discrepancy.  Matches feature distributions with a
                  multi-kernel RBF loss.  Simpler than DANN, no adversarial
                  instability.  Requires source data.

  coral           CORrelation ALignment.  Matches first- and second-order
                  statistics of source/target features.  Lightest domain-
                  alignment method; often competitive with MMD.
                  Requires source data.

  pseudo_label    Self-training.  Runs the source model on unlabelled target data,
                  keeps high-confidence predictions as pseudo-labels, then trains
                  jointly on labelled + pseudo-labelled target data.
                  Works with or without source data.

Strategies can be combined, e.g. --strategy progressive+mmd.
All domain-alignment methods (dann/mmd/coral) operate on the CLS/mean-pool
feature vector produced by forward_features().
"""

import argparse
import copy
import itertools
import json
import math
import os
import sys
import time
import datetime
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, ConcatDataset, Subset, TensorDataset
from einops import rearrange
from timm.models import create_model
from timm.layers import trunc_normal_

import utils
from utils import NativeScalerWithGradNormCount as NativeScaler
import task_model


# ============================================================================
# Gradient Reversal Layer  (DANN)
# ============================================================================

class _GRL(torch.autograd.Function):
    """Gradient Reversal Layer.  Forward = identity, backward = -alpha * grad."""
    @staticmethod
    def forward(ctx, x: torch.Tensor, alpha: float) -> torch.Tensor:
        ctx.alpha = alpha
        return x.clone()

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        return -ctx.alpha * grad, None


def grad_reverse(x: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    return _GRL.apply(x, alpha)


def dann_lambda(step: int, total_steps: int, gamma: float = 10.0) -> float:
    """
    DANN lambda schedule from Ganin & Lempitsky 2015:
        λ = 2 / (1 + exp(-γ·p)) - 1,  p = step / total_steps
    Starts near 0, saturates near 1.
    """
    p = step / max(total_steps, 1)
    return 2.0 / (1.0 + math.exp(-gamma * p)) - 1.0


class DomainClassifier(nn.Module):
    """
    Binary classifier: source (0) vs target (1).
    Receives features *after* gradient reversal, so the backbone is trained
    to produce domain-invariant representations.
    """
    def __init__(self, embed_dim: int, hidden_dim: int = 256, dropout: float = 0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.02)
                nn.init.zeros_(m.bias)

    def forward(self, feat: torch.Tensor, alpha: float) -> torch.Tensor:
        return self.net(grad_reverse(feat, alpha)).squeeze(-1)


# ============================================================================
# MMD loss  (multi-kernel RBF)
# ============================================================================

def _rbf_kernel(x: torch.Tensor, y: torch.Tensor, sigma: float) -> torch.Tensor:
    """RBF kernel k(x, y) = exp(-||x-y||²  /  2σ²)."""
    diff = x.unsqueeze(1) - y.unsqueeze(0)          # [N, M, D]
    sq   = diff.pow(2).sum(-1)                       # [N, M]
    return torch.exp(-sq / (2.0 * sigma * sigma))


def mmd_loss(src: torch.Tensor, tgt: torch.Tensor,
             bandwidths: Tuple[float, ...] = (0.5, 1.0, 2.0, 4.0, 8.0)) -> torch.Tensor:
    """
    Multi-kernel MMD²:
        MMD² = E[k(xs,xs)] + E[k(xt,xt)] - 2 E[k(xs,xt)]
    Averaged over all kernels.
    """
    loss = torch.zeros(1, device=src.device)
    for bw in bandwidths:
        kss = _rbf_kernel(src, src, bw).mean()
        ktt = _rbf_kernel(tgt, tgt, bw).mean()
        kst = _rbf_kernel(src, tgt, bw).mean()
        loss = loss + kss + ktt - 2.0 * kst
    return loss / len(bandwidths)


# ============================================================================
# CORAL loss  (correlation alignment)
# ============================================================================

def coral_loss(src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
    """
    CORAL loss:  (1 / 4d²) · ||C_src - C_tgt||²_F
    where C = feature covariance matrix.
    """
    d = src.shape[1]

    def _cov(x: torch.Tensor) -> torch.Tensor:
        n = x.shape[0]
        x = x - x.mean(0, keepdim=True)
        return (x.t() @ x) / (n - 1)

    cs = _cov(src)
    ct = _cov(tgt)
    return (cs - ct).pow(2).sum() / (4.0 * d * d)


# ============================================================================
# Pseudo-label generation
# ============================================================================

@torch.no_grad()
def generate_pseudo_labels(
    model: nn.Module,
    dataset,
    device: torch.device,
    input_chans,
    is_binary: bool,
    confidence_threshold: float = 0.9,
    batch_size: int = 64,
    num_workers: int = 4,
) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
    """
    Run the source model over `dataset` and return high-confidence pseudo-labels.

    Returns
    -------
    pseudo_data   : [N_pseudo, ...]  raw EEG tensors (unnormalised, as stored)
    pseudo_labels : [N_pseudo]       predicted class indices / probabilities
    accepted_idx  : list of original dataset indices that passed the threshold
    """
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True, drop_last=False)
    model.eval()

    all_data, all_probs, all_indices = [], [], []
    offset = 0
    for samples, _ in loader:
        raw = samples.clone()                            # keep unnormalised copy
        x   = samples.float().to(device) / 100
        x   = rearrange(x, 'B N (A T) -> B N A T', T=200)

        with torch.autocast(device_type=device.type if device.type != 'mps' else 'cpu'):
            logits = model(x, input_chans)

        if is_binary:
            probs = torch.sigmoid(logits).squeeze(-1).cpu()   # [B]
            conf  = torch.max(probs, 1 - probs)               # confidence = max(p, 1-p)
            keep  = conf >= confidence_threshold
        else:
            probs = F.softmax(logits, dim=-1).cpu()           # [B, C]
            conf  = probs.max(dim=-1).values                  # [B]
            keep  = conf >= confidence_threshold

        kept_idx = keep.nonzero(as_tuple=True)[0]
        for i in kept_idx.tolist():
            all_data.append(raw[i])
            all_probs.append(probs[i])
            all_indices.append(offset + i)
        offset += samples.shape[0]

    if not all_data:
        return torch.empty(0), torch.empty(0), []

    pseudo_data   = torch.stack(all_data)
    pseudo_labels = (
        torch.stack(all_probs).float()
        if is_binary
        else torch.stack(all_probs).argmax(dim=-1).long()
    )
    print(f"[Pseudo-label] {len(all_indices)}/{len(dataset)} samples accepted "
          f"(threshold={confidence_threshold})")
    return pseudo_data, pseudo_labels, all_indices


# ============================================================================
# Progressive fine-tuning helpers
# ============================================================================

def _freeze_all(model: nn.Module):
    for p in model.parameters():
        p.requires_grad_(False)


def _unfreeze(module: Optional[nn.Module]):
    if module is not None:
        for p in module.parameters():
            p.requires_grad_(True)


def set_progressive_stage(model: nn.Module, stage: int, last_n_blocks: int):
    """
    Stage 1: head only
    Stage 2: head + last `last_n_blocks` transformer blocks + norms
    Stage 3: everything
    """
    _freeze_all(model)
    if stage >= 1:
        _unfreeze(model.head)
        _unfreeze(model.fc_norm)
        _unfreeze(model.norm if not isinstance(model.norm, nn.Identity) else None)
    if stage >= 2:
        n = len(model.blocks)
        for i, blk in enumerate(model.blocks):
            if i >= n - last_n_blocks:
                _unfreeze(blk)
    if stage >= 3:
        for p in model.parameters():
            p.requires_grad_(True)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"  [Stage {stage}] trainable {trainable/1e6:.2f}M / {total/1e6:.2f}M "
          f"({100*trainable/total:.1f}%)")


# ============================================================================
# Feature extraction helper
# ============================================================================

def extract_features(model: nn.Module, x: torch.Tensor, input_chans) -> torch.Tensor:
    """Return the pooled feature vector (before head) from a NeuralTransformer."""
    return model.forward_features(x, input_chans=input_chans)


# ============================================================================
# Training — one epoch
# ============================================================================

def train_one_epoch(
    # core
    model: nn.Module,
    target_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    loss_scaler,
    # task
    input_chans,
    is_binary: bool,
    task_criterion: nn.Module,
    clip_grad: float = 3.0,
    # strategies
    use_dann: bool = False,
    use_mmd:  bool = False,
    use_coral: bool = False,
    source_loader: Optional[DataLoader] = None,
    domain_classifier: Optional[nn.Module] = None,
    dann_optimizer: Optional[torch.optim.Optimizer] = None,
    dann_total_steps: int = 1,
    dann_step_offset: int = 0,
    dann_weight: float = 1.0,
    mmd_weight:  float = 1.0,
    coral_weight: float = 1.0,
    mmd_bandwidths: Tuple[float, ...] = (0.5, 1.0, 2.0, 4.0, 8.0),
    # schedules
    lr_schedule_values=None,
    wd_schedule_values=None,
    start_steps: int = 0,
    log_writer=None,
) -> dict:

    model.train()
    if domain_classifier is not None:
        domain_classifier.train()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = f'Epoch [{epoch}]'

    # cycle source loader if needed for domain alignment
    src_iter: Iterator = iter([])
    if (use_dann or use_mmd or use_coral) and source_loader is not None:
        src_iter = itertools.cycle(source_loader)

    global_step = start_steps

    for step, (tgt_samples, tgt_targets) in enumerate(
            metric_logger.log_every(target_loader, 10, header)):

        it = start_steps + step
        if lr_schedule_values is not None:
            for pg in optimizer.param_groups:
                pg["lr"] = lr_schedule_values[it] * pg.get("lr_scale", 1.0)
        if wd_schedule_values is not None:
            for pg in optimizer.param_groups:
                if pg["weight_decay"] > 0:
                    pg["weight_decay"] = wd_schedule_values[it]

        # ---- target batch ----
        tgt_samples = tgt_samples.float().to(device, non_blocking=True) / 100
        tgt_samples = rearrange(tgt_samples, 'B N (A T) -> B N A T', T=200)
        tgt_targets = tgt_targets.to(device, non_blocking=True)
        if is_binary:
            tgt_targets = tgt_targets.float().unsqueeze(-1)

        # ---- source batch (for domain alignment) ----
        src_feat = None
        if (use_dann or use_mmd or use_coral) and source_loader is not None:
            try:
                src_batch = next(src_iter)
            except StopIteration:
                src_iter = itertools.cycle(source_loader)
                src_batch = next(src_iter)
            src_x = src_batch[0].float().to(device, non_blocking=True) / 100
            src_x = rearrange(src_x, 'B N (A T) -> B N A T', T=200)

        autocast_device = device.type if device.type != 'mps' else 'cpu'

        with torch.autocast(device_type=autocast_device):
            # --- task loss ---
            tgt_logits = model(tgt_samples, input_chans)
            task_loss  = task_criterion(tgt_logits, tgt_targets)
            total_loss = task_loss

            # --- domain alignment ---
            if (use_dann or use_mmd or use_coral) and source_loader is not None:
                tgt_feat = extract_features(model, tgt_samples, input_chans)
                with torch.no_grad() if not use_dann else torch.enable_grad():
                    src_feat = extract_features(model, src_x, input_chans)

                # DANN
                if use_dann and domain_classifier is not None:
                    lam = dann_lambda(dann_step_offset + step,
                                      dann_total_steps)
                    # domain labels: source=0, target=1
                    src_dom = torch.zeros(src_feat.shape[0], device=device)
                    tgt_dom = torch.ones( tgt_feat.shape[0], device=device)
                    all_feat = torch.cat([src_feat, tgt_feat], dim=0)
                    all_dom  = torch.cat([src_dom,  tgt_dom],  dim=0)
                    dom_logits = domain_classifier(all_feat, lam)
                    dann_loss  = F.binary_cross_entropy_with_logits(dom_logits, all_dom)
                    total_loss = total_loss + dann_weight * dann_loss
                    metric_logger.update(dann_loss=dann_loss.item())
                    metric_logger.update(dann_lambda=lam)

                # MMD
                if use_mmd:
                    mmd = mmd_loss(src_feat, tgt_feat, bandwidths=mmd_bandwidths)
                    total_loss = total_loss + mmd_weight * mmd
                    metric_logger.update(mmd_loss=mmd.item())

                # CORAL
                if use_coral:
                    cr = coral_loss(src_feat, tgt_feat)
                    total_loss = total_loss + coral_weight * cr
                    metric_logger.update(coral_loss=cr.item())

        loss_value = total_loss.item()
        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training.", flush=True)
            sys.exit(1)

        # ---- backward & optimise ----
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order

        # backbone + task head
        params_to_scale = list(model.parameters())
        if domain_classifier is not None:
            params_to_scale += list(domain_classifier.parameters())

        grad_norm = loss_scaler(total_loss, optimizer, clip_grad=clip_grad,
                                parameters=params_to_scale,
                                create_graph=is_second_order)
        optimizer.zero_grad()
        if dann_optimizer is not None:
            dann_optimizer.zero_grad()

        if device.type == 'cuda':
            torch.cuda.synchronize()

        metric_logger.update(loss=loss_value,
                             task_loss=task_loss.item(),
                             grad_norm=grad_norm if grad_norm is not None else 0.0)
        metric_logger.update(lr=max(pg["lr"] for pg in optimizer.param_groups))

        if log_writer is not None:
            log_writer.update(loss=loss_value, head="loss")
            log_writer.update(lr=metric_logger.lr.value, head="opt")
            log_writer.set_step()

        global_step += 1

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


# ============================================================================
# Evaluation
# ============================================================================

@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device,
             input_chans, is_binary: bool, metrics: list) -> dict:
    model.eval()
    criterion = nn.BCEWithLogitsLoss() if is_binary else nn.CrossEntropyLoss()
    all_preds, all_targets = [], []
    total_loss, n_batches = 0.0, 0

    autocast_device = device.type if device.type != 'mps' else 'cpu'

    for samples, targets in loader:
        x = samples.float().to(device) / 100
        x = rearrange(x, 'B N (A T) -> B N A T', T=200)
        t = targets.to(device)
        if is_binary:
            t = t.float().unsqueeze(-1)

        with torch.autocast(device_type=autocast_device):
            logits = model(x, input_chans)
            total_loss += criterion(logits, t).item()
        n_batches += 1

        if is_binary:
            all_preds.append(torch.sigmoid(logits).cpu())
            all_targets.append((t >= 0.5).int().cpu())
        else:
            all_preds.append(logits.cpu())
            all_targets.append(targets.cpu())

    preds   = torch.cat(all_preds).numpy()
    targets_ = torch.cat(all_targets).numpy()
    ret = utils.get_metrics(preds, targets_, metrics, is_binary)
    ret['loss'] = total_loss / max(n_batches, 1)
    print("  " + "  ".join(f"{k}: {v:.4f}" for k, v in ret.items()))
    return ret


# ============================================================================
# Dataset & model helpers
# ============================================================================

def get_ch_names_and_metrics(args):
    ds = args.dataset.upper()
    is_binary = args.is_binary
    if ds in ('TUAB', 'TUEP'):
        ch_names = ['FP1','FP2','F3','F4','C3','C4','P3','P4',
                    'O1','O2','F7','F8','T3','T4','T5','T6',
                    'A1','A2','FZ','CZ','PZ','T1','T2']
        metrics  = ['pr_auc','roc_auc','accuracy','balanced_accuracy']
        is_binary = True
    elif ds == 'TUEV':
        ch_names = ['FP1','FP2','F3','F4','C3','C4','P3','P4',
                    'O1','O2','F7','F8','T3','T4','T5','T6',
                    'A1','A2','FZ','CZ','PZ','T1','T2']
        metrics  = ['accuracy','balanced_accuracy','cohen_kappa','f1_weighted']
    elif ds in ('IIIC','IIIC_HM','IIIC_CHEWING'):
        if getattr(args,'train_eeg_montage','average') == 'bipolar':
            ch_names = ['FP1-F7','F7-T3','T3-T5','T5-O1','FP2-F8','F8-T4','T4-T6',
                        'T6-O2','FP1-F3','F3-C3','C3-P3','P3-O1','FP2-F4','F4-C4',
                        'C4-P4','P4-O2','FZ-CZ','CZ-PZ']
        else:
            ch_names = ['FP1','F3','C3','P3','F7','T3','T5','O1','FZ','CZ','PZ',
                        'FP2','F4','C4','P4','F8','T4','T6','O2']
        metrics  = ['accuracy','balanced_accuracy','cohen_kappa','f1_weighted']
    elif ds == 'SLEEP':
        ch_names = ['FP1','F3','C3','P3','F7','T3','T5','O1','FZ','CZ','PZ',
                    'FP2','F4','C4','P4','F8','T4','T6','O2']
        metrics  = ['accuracy','balanced_accuracy','cohen_kappa','f1_weighted']
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    return ch_names, metrics, is_binary


def build_model(args) -> nn.Module:
    return create_model(
        args.model,
        pretrained=False,
        num_classes=args.nb_classes,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        attn_drop_rate=args.attn_drop_rate,
        drop_block_rate=None,
        use_mean_pooling=args.use_mean_pooling,
        init_scale=args.init_scale,
        use_rel_pos_bias=args.rel_pos_bias,
        use_abs_pos_emb=args.abs_pos_emb,
        init_values=args.layer_scale_init_value,
        qkv_bias=args.qkv_bias,
    )


def load_checkpoint(model: nn.Module, ckpt_path: str,
                    source_nb_classes: Optional[int] = None):
    """
    Load a fine-tuned checkpoint.  If source_nb_classes != model.num_classes,
    the head is replaced after loading (transfer to different class count).
    """
    ckpt  = torch.load(ckpt_path, map_location='cpu')
    state = (ckpt.get('model') or ckpt.get('state_dict') or ckpt)
    state = {k.replace('module.', ''): v for k, v in state.items()}

    if source_nb_classes is not None and source_nb_classes != model.num_classes:
        # temporarily set the correct head size to load weights, then replace
        model.reset_classifier(source_nb_classes)
        model.load_state_dict(state, strict=False)
        print(f"[Checkpoint] Head replaced: {source_nb_classes} → {model.num_classes} classes")
        model.reset_classifier(model.num_classes)
        # re-init the new head
        nn.init.zeros_(model.head.bias)
        trunc_normal_(model.head.weight, std=0.02)
    else:
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            print(f"[Checkpoint] Missing   : {missing[:8]}")
        if unexpected:
            print(f"[Checkpoint] Unexpected: {unexpected[:8]}")

    print(f"[Checkpoint] Loaded from {ckpt_path}")


class _FolderDataset(torch.utils.data.Dataset):
    def __init__(self, root: str):
        self.files = sorted(Path(root).glob('*.pt'))
        if not self.files:
            raise FileNotFoundError(f"No .pt files in {root}")
    def __len__(self): return len(self.files)
    def __getitem__(self, i): return torch.load(self.files[i])


def _try_build_dataset(root: str, ch_names, args):
    try:
        from utils import EEGDataset
        return EEGDataset(root, ch_names=ch_names,
                          sample_length=getattr(args, 'train_eeg_length', False))
    except Exception:
        return _FolderDataset(root)


# ============================================================================
# Argument parser
# ============================================================================

def get_args():
    p = argparse.ArgumentParser('EEG Transfer Learning', add_help=False)

    # checkpoints / paths
    p.add_argument('--finetune', required=True,
                   help='Source-domain fine-tuned checkpoint')
    p.add_argument('--output_dir', default='./transfer_output')
    p.add_argument('--log_dir', default=None)

    # model architecture
    p.add_argument('--model', default='base_patch200_200')
    p.add_argument('--source_nb_classes', type=int, default=None,
                   help='Number of classes in the source checkpoint head. '
                        'Required when source and target class counts differ.')
    p.add_argument('--nb_classes', type=int, required=True,
                   help='Number of classes in the TARGET domain')
    p.add_argument('--drop',                  type=float, default=0.0)
    p.add_argument('--drop_path',             type=float, default=0.1)
    p.add_argument('--attn_drop_rate',        type=float, default=0.0)
    p.add_argument('--use_mean_pooling',      action='store_true', default=True)
    p.add_argument('--init_scale',            type=float, default=0.001)
    p.add_argument('--rel_pos_bias',          action='store_true', default=False)
    p.add_argument('--abs_pos_emb',           action='store_true', default=False)
    p.add_argument('--layer_scale_init_value',type=float, default=0.1)
    p.add_argument('--qkv_bias',              action='store_true', default=True)
    p.add_argument('--is_binary',             action='store_true', default=False)

    # data
    p.add_argument('--target_train_dir', required=True,
                   help='Training data in TARGET domain')
    p.add_argument('--target_val_dir',   default='',
                   help='Validation data in TARGET domain (optional)')
    p.add_argument('--source_train_dir', default='',
                   help='Training data in SOURCE domain (needed for dann/mmd/coral)')
    p.add_argument('--dataset',          default='IIIC',
                   help='Dataset identifier (same as finetune_classification.py)')
    p.add_argument('--train_eeg_montage',default='average')
    p.add_argument('--train_eeg_length', default=False, type=int)

    # strategy
    p.add_argument('--strategy', default='progressive',
                   help='Transfer learning strategy.  Combine with "+": '
                        'progressive | dann | mmd | coral | pseudo_label  '
                        'e.g. --strategy progressive+mmd')

    # progressive fine-tuning
    p.add_argument('--prog_stage1_epochs', type=int, default=2,
                   help='Epochs training head only (stage 1)')
    p.add_argument('--prog_stage2_epochs', type=int, default=3,
                   help='Epochs training head + last N blocks (stage 2)')
    p.add_argument('--last_n_blocks',      type=int, default=4,
                   help='Blocks to unfreeze in stage 2 / non-progressive partial training')

    # DANN
    p.add_argument('--dann_weight',  type=float, default=1.0)
    p.add_argument('--dann_hidden',  type=int,   default=256)
    p.add_argument('--dann_dropout', type=float, default=0.5)
    p.add_argument('--dann_gamma',   type=float, default=10.0,
                   help='Lambda schedule steepness (original DANN paper default=10)')

    # MMD
    p.add_argument('--mmd_weight',      type=float, default=1.0)
    p.add_argument('--mmd_bandwidths',  type=float, nargs='+',
                   default=[0.5, 1.0, 2.0, 4.0, 8.0])

    # CORAL
    p.add_argument('--coral_weight', type=float, default=1.0)

    # Pseudo-label
    p.add_argument('--pseudo_confidence', type=float, default=0.9,
                   help='Confidence threshold for pseudo-label acceptance')
    p.add_argument('--pseudo_weight',     type=float, default=0.5,
                   help='Loss weight for pseudo-labelled samples')
    p.add_argument('--pseudo_update_every', type=int, default=5,
                   help='Re-generate pseudo-labels every N epochs (0 = once)')

    # training
    p.add_argument('--epochs',          type=int,   default=20)
    p.add_argument('--batch_size',      type=int,   default=64)
    p.add_argument('--lr',              type=float, default=5e-5)
    p.add_argument('--min_lr',          type=float, default=1e-6)
    p.add_argument('--warmup_lr',       type=float, default=1e-6)
    p.add_argument('--warmup_epochs',   type=int,   default=2)
    p.add_argument('--warmup_steps',    type=int,   default=-1)
    p.add_argument('--weight_decay',    type=float, default=0.05)
    p.add_argument('--weight_decay_end',type=float, default=None)
    p.add_argument('--layer_decay',     type=float, default=0.9,
                   help='Layer-wise LR decay coefficient (1.0 = disabled)')
    p.add_argument('--clip_grad',       type=float, default=3.0)
    p.add_argument('--opt',             default='adamw')
    p.add_argument('--opt_eps',         type=float, default=1e-8)
    p.add_argument('--opt_betas',       type=float, nargs='+', default=None)
    p.add_argument('--smoothing',       type=float, default=0.1)
    p.add_argument('--save_ckpt_freq',  type=int,   default=1)

    # misc
    p.add_argument('--device',       default='cuda')
    p.add_argument('--seed',         type=int, default=0)
    p.add_argument('--num_workers',  type=int, default=8)
    p.add_argument('--pin_mem',      action='store_true', default=True)
    p.add_argument('--eeg_montage',  default='average')

    return p.parse_args()


# ============================================================================
# Main
# ============================================================================

def main():
    args    = get_args()
    strategies = set(s.strip().lower() for s in args.strategy.split('+'))
    use_progressive  = 'progressive'  in strategies
    use_dann         = 'dann'         in strategies
    use_mmd          = 'mmd'          in strategies
    use_coral        = 'coral'        in strategies
    use_pseudo_label = 'pseudo_label' in strategies

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cudnn.benchmark = True

    print("=" * 60)
    print(f" EEG Transfer Learning")
    print(f" Strategies : {args.strategy}")
    print(f" Source ckpt: {args.finetune}")
    print(f" Target data: {args.target_train_dir}")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Build model and load source weights
    # ------------------------------------------------------------------
    print("\n[1] Building model and loading source checkpoint ...")
    model = build_model(args)
    load_checkpoint(model, args.finetune,
                    source_nb_classes=args.source_nb_classes)
    model.to(device)

    embed_dim = model.embed_dim

    # ------------------------------------------------------------------
    # 2. Datasets
    # ------------------------------------------------------------------
    print("\n[2] Building datasets ...")
    ch_names, metrics, is_binary = get_ch_names_and_metrics(args)
    is_binary   = is_binary or args.is_binary
    input_chans = utils.get_input_chans(ch_names)

    target_train = _try_build_dataset(args.target_train_dir, ch_names, args)
    target_val   = (_try_build_dataset(args.target_val_dir, ch_names, args)
                    if args.target_val_dir else None)
    source_train = (_try_build_dataset(args.source_train_dir, ch_names, args)
                    if args.source_train_dir else None)

    print(f"  Target train : {len(target_train)} samples")
    if target_val:    print(f"  Target val   : {len(target_val)} samples")
    if source_train:  print(f"  Source train : {len(source_train)} samples")

    # source loader (domain alignment)
    source_loader = None
    if source_train and (use_dann or use_mmd or use_coral):
        source_loader = DataLoader(
            source_train, batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers, pin_memory=args.pin_mem, drop_last=True)

    val_loader = None
    if target_val:
        val_loader = DataLoader(
            target_val, batch_size=args.batch_size * 2, shuffle=False,
            num_workers=args.num_workers, pin_memory=args.pin_mem, drop_last=False)

    # ------------------------------------------------------------------
    # 3. Pseudo-label: generate initial pseudo-labels from source model
    # ------------------------------------------------------------------
    pseudo_dataset = None
    if use_pseudo_label:
        print("\n[3] Generating pseudo-labels with source model ...")
        pseudo_x, pseudo_y, _ = generate_pseudo_labels(
            model, target_train, device, input_chans, is_binary,
            confidence_threshold=args.pseudo_confidence,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        if len(pseudo_x) > 0:
            pseudo_dataset = TensorDataset(pseudo_x, pseudo_y)

    # ------------------------------------------------------------------
    # 4. DANN domain classifier
    # ------------------------------------------------------------------
    domain_classifier = None
    dann_optimizer    = None
    if use_dann:
        print("\n[4] Building domain classifier (DANN) ...")
        domain_classifier = DomainClassifier(
            embed_dim, hidden_dim=args.dann_hidden, dropout=args.dann_dropout
        ).to(device)
        dann_optimizer = torch.optim.Adam(
            domain_classifier.parameters(), lr=args.lr * 10)

    # ------------------------------------------------------------------
    # 5. Loss function
    # ------------------------------------------------------------------
    if args.smoothing > 0 and not is_binary:
        from timm.loss import LabelSmoothingCrossEntropy
        task_criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    elif is_binary:
        task_criterion = nn.BCEWithLogitsLoss()
    else:
        task_criterion = nn.CrossEntropyLoss()

    # ------------------------------------------------------------------
    # 6. Determine total epochs per stage (progressive)
    # ------------------------------------------------------------------
    if use_progressive:
        stage1_end = args.prog_stage1_epochs
        stage2_end = stage1_end + args.prog_stage2_epochs
        stage3_end = args.epochs
        print(f"\n[5] Progressive stages: "
              f"Stage 1 [0,{stage1_end}), "
              f"Stage 2 [{stage1_end},{stage2_end}), "
              f"Stage 3 [{stage2_end},{stage3_end})")
    else:
        # non-progressive: unfreeze last N blocks from the start
        set_progressive_stage(model, stage=2, last_n_blocks=args.last_n_blocks)
        stage1_end = stage2_end = -1  # won't trigger

    # ------------------------------------------------------------------
    # 7. Optimizer and schedules
    # ------------------------------------------------------------------
    def _build_optimizer(model):
        if args.layer_decay < 1.0:
            n_layers = model.get_num_layers()
            assigner = utils.LayerDecayValueAssigner(
                [args.layer_decay ** (n_layers + 1 - i)
                 for i in range(n_layers + 2)])
            return utils.create_optimizer(
                args, model,
                get_num_layer=assigner.get_layer_id,
                get_layer_scale=assigner.get_scale)
        return utils.create_optimizer(args, model)

    optimizer    = _build_optimizer(model)
    loss_scaler  = NativeScaler()

    # n_steps_per_epoch based on combined dataset size
    n_steps = (len(target_train) +
               (len(pseudo_dataset) if pseudo_dataset else 0)) // args.batch_size

    lr_schedule_values = utils.cosine_scheduler(
        args.lr, args.min_lr, args.epochs, n_steps,
        warmup_epochs=args.warmup_epochs, warmup_steps=args.warmup_steps)
    wd_end = args.weight_decay_end or args.weight_decay
    wd_schedule_values = utils.cosine_scheduler(
        args.weight_decay, wd_end, args.epochs, n_steps)

    log_writer = None
    if args.log_dir:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = utils.TensorboardLogger(log_dir=args.log_dir)

    # ------------------------------------------------------------------
    # 8. Training loop
    # ------------------------------------------------------------------
    best_metric = -1.0
    start_time  = time.time()
    dann_step_offset = 0

    for epoch in range(args.epochs):

        # ---- progressive stage transition ----
        if use_progressive:
            if epoch == 0:
                print(f"\n[Stage 1] Head only")
                set_progressive_stage(model, stage=1, last_n_blocks=args.last_n_blocks)
                optimizer = _build_optimizer(model)
            elif epoch == stage1_end:
                print(f"\n[Stage 2] Head + last {args.last_n_blocks} blocks")
                set_progressive_stage(model, stage=2, last_n_blocks=args.last_n_blocks)
                optimizer = _build_optimizer(model)
            elif epoch == stage2_end:
                print(f"\n[Stage 3] Full model")
                set_progressive_stage(model, stage=3, last_n_blocks=args.last_n_blocks)
                optimizer = _build_optimizer(model)

        # ---- refresh pseudo-labels periodically ----
        if use_pseudo_label and args.pseudo_update_every > 0:
            if epoch > 0 and epoch % args.pseudo_update_every == 0:
                print(f"\n[Epoch {epoch}] Refreshing pseudo-labels ...")
                pseudo_x, pseudo_y, _ = generate_pseudo_labels(
                    model, target_train, device, input_chans, is_binary,
                    confidence_threshold=args.pseudo_confidence,
                    batch_size=args.batch_size,
                    num_workers=args.num_workers,
                )
                pseudo_dataset = TensorDataset(pseudo_x, pseudo_y) if len(pseudo_x) > 0 else None

        # ---- build this epoch's training dataset ----
        if use_pseudo_label and pseudo_dataset is not None:
            # weighted combination: full target labelled + weighted pseudo-labelled
            # We duplicate pseudo samples to achieve the desired weight ratio
            n_pseudo = int(len(target_train) * args.pseudo_weight)
            if n_pseudo > 0 and len(pseudo_dataset) > 0:
                pseudo_indices = torch.randint(len(pseudo_dataset), (n_pseudo,)).tolist()
                pseudo_subset  = Subset(pseudo_dataset, pseudo_indices)
                epoch_dataset  = ConcatDataset([target_train, pseudo_subset])
            else:
                epoch_dataset = target_train
        else:
            epoch_dataset = target_train

        epoch_loader = DataLoader(
            epoch_dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers, pin_memory=args.pin_mem, drop_last=True)

        # ---- train ----
        train_stats = train_one_epoch(
            model=model,
            target_loader=epoch_loader,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            loss_scaler=loss_scaler,
            input_chans=input_chans,
            is_binary=is_binary,
            task_criterion=task_criterion,
            clip_grad=args.clip_grad,
            use_dann=use_dann,
            use_mmd=use_mmd,
            use_coral=use_coral,
            source_loader=source_loader,
            domain_classifier=domain_classifier,
            dann_optimizer=dann_optimizer,
            dann_total_steps=args.epochs * len(epoch_loader),
            dann_step_offset=dann_step_offset,
            dann_weight=args.dann_weight,
            mmd_weight=args.mmd_weight,
            coral_weight=args.coral_weight,
            mmd_bandwidths=tuple(args.mmd_bandwidths),
            lr_schedule_values=lr_schedule_values,
            wd_schedule_values=wd_schedule_values,
            start_steps=epoch * len(epoch_loader),
            log_writer=log_writer,
        )
        dann_step_offset += len(epoch_loader)

        # ---- save checkpoint ----
        if args.output_dir and (epoch + 1) % args.save_ckpt_freq == 0:
            ckpt_path = os.path.join(args.output_dir, f'checkpoint_epoch{epoch:03d}.pth')
            save_dict = {'model': model.state_dict(),
                         'optimizer': optimizer.state_dict(),
                         'epoch': epoch, 'args': vars(args)}
            if domain_classifier is not None:
                save_dict['domain_classifier'] = domain_classifier.state_dict()
            torch.save(save_dict, ckpt_path)

        # ---- validation ----
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()}, 'epoch': epoch}
        if val_loader is not None:
            print(f"  [Val epoch {epoch}]")
            val_stats = evaluate(model, val_loader, device, input_chans,
                                 is_binary, metrics)
            log_stats.update({f'val_{k}': v for k, v in val_stats.items()})

            key = metrics[0]
            if val_stats.get(key, -1) > best_metric:
                best_metric = val_stats[key]
                best_path   = os.path.join(args.output_dir, 'checkpoint_best.pth')
                torch.save({'model': model.state_dict(), 'epoch': epoch,
                            f'best_{key}': best_metric, 'args': vars(args)}, best_path)
                print(f"  [Best] {key}={best_metric:.4f} → {best_path}")

        if args.output_dir:
            with open(os.path.join(args.output_dir, 'log.txt'), 'a') as f:
                f.write(json.dumps(log_stats) + '\n')

    # ---- final save ----
    final_path = os.path.join(args.output_dir, 'checkpoint_final.pth')
    torch.save({'model': model.state_dict(), 'args': vars(args)}, final_path)
    elapsed = str(datetime.timedelta(seconds=int(time.time() - start_time)))
    print(f"\n[Done] Final checkpoint: {final_path}")
    print(f"[Done] Total time: {elapsed}")


if __name__ == '__main__':
    main()
