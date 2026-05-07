"""
hard_mining.py
==============
Continue training a fine-tuned EEG classification model with hard example mining.

Two update modes
----------------
  --mode A  Keep model structure, change parameters only.
            --finetune_layers  last_n  (recommended: last 4 blocks + norm + head)
                               all     (full fine-tuning)
            --last_n_blocks    N       (default 4, used only with last_n)

  --mode B  Insert a residual adapter before the classification head.
            Adapter: Linear(D→D//4) → GELU → Linear(D//4→D), initialised as identity.
            Only the adapter + head are trained by default; set --also_finetune_backbone
            to additionally unfreeze the last N transformer blocks.

Two data strategies
-------------------
  --data_strategy  full      Use hard examples ∪ all original training data.
  --data_strategy  hard_only Use hard examples only; apply a continual-learning
                             regulariser to prevent catastrophic forgetting.
                             --cl_method  ewc   Elastic Weight Consolidation
                             --cl_method  kd    Knowledge Distillation from frozen
                                                original model (default)

Hard example selection
----------------------
  --hard_ratio      Top fraction of training samples by per-sample loss (default 0.3).
  --hard_min_loss   Alternative threshold: keep samples with loss > this value.
                    If both are given, their union is used.
"""

import argparse
import copy
import json
import math
import os
import sys
import time
import datetime
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, Subset, ConcatDataset
from einops import rearrange
from timm.models import create_model
from timm.layers import trunc_normal_

import utils
from utils import NativeScalerWithGradNormCount as NativeScaler
import task_model


# ---------------------------------------------------------------------------
# Adapter (Mode B)
# ---------------------------------------------------------------------------

class ResidualAdapter(nn.Module):
    """
    Bottleneck residual adapter inserted between features and the head.
    Initialised as (near) identity so the model behaves like the pre-trained
    model at the start of hard mining.

    down: D → D//reduction  (no bias, matched to LoRA convention)
    up:   D//reduction → D  (initialised to zero so residual starts at 0)
    """
    def __init__(self, embed_dim: int, reduction: int = 4, dropout: float = 0.0):
        super().__init__()
        bottleneck = max(embed_dim // reduction, 16)
        self.down = nn.Linear(embed_dim, bottleneck, bias=True)
        self.act  = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.up   = nn.Linear(bottleneck, embed_dim, bias=True)
        # initialise up-projection to zero → residual starts at 0
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)
        trunc_normal_(self.down.weight, std=0.02)
        nn.init.zeros_(self.down.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.drop(self.up(self.act(self.down(x))))


class ModelWithAdapter(nn.Module):
    """
    Wraps a NeuralTransformer, inserting a ResidualAdapter between
    forward_features and head.  The original model's state dict can be loaded
    without strict=False (adapter keys are new).
    """
    def __init__(self, base_model: nn.Module, reduction: int = 4, adapter_dropout: float = 0.0):
        super().__init__()
        self.base = base_model
        embed_dim = base_model.embed_dim
        self.adapter = ResidualAdapter(embed_dim, reduction=reduction, dropout=adapter_dropout)

    @property
    def embed_dim(self):
        return self.base.embed_dim

    @property
    def num_classes(self):
        return self.base.num_classes

    @torch.jit.ignore
    def no_weight_decay(self):
        nwd = set()
        for k in self.base.no_weight_decay():
            nwd.add(f'base.{k}')
        return nwd

    def forward(self, x, input_chans=None, **kwargs):
        feat = self.base.forward_features(x, input_chans=input_chans)
        feat = self.adapter(feat)
        return self.base.head(feat)


# ---------------------------------------------------------------------------
# EWC (Elastic Weight Consolidation)
# ---------------------------------------------------------------------------

class EWC:
    """
    Compute and store the Fisher Information Matrix diagonal and the reference
    parameters.  Used to add a regularisation term to the loss:
        ewc_loss = λ/2 * Σ_i F_i * (θ_i - θ_i*)²
    """
    def __init__(self, model: nn.Module, data_loader: DataLoader,
                 device: torch.device, input_chans, is_binary: bool,
                 n_samples: int = 2000):
        self.params  = {}   # θ* : reference parameter values
        self.fisher  = {}   # F  : Fisher Information diagonal

        self._compute(model, data_loader, device, input_chans, is_binary, n_samples)

    @torch.no_grad()
    def _copy_params(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.params[name] = param.detach().clone()

    def _compute(self, model, data_loader, device, input_chans, is_binary, n_samples):
        print(f"[EWC] Computing Fisher Information on up to {n_samples} samples ...")
        model.eval()
        self._copy_params(model)

        fisher_acc = {n: torch.zeros_like(p)
                      for n, p in model.named_parameters() if p.requires_grad}

        if is_binary:
            criterion = nn.BCEWithLogitsLoss()
        else:
            criterion = nn.CrossEntropyLoss()

        seen = 0
        for samples, targets in data_loader:
            if seen >= n_samples:
                break
            samples = samples.float().to(device) / 100
            samples = rearrange(samples, 'B N (A T) -> B N A T', T=200)
            targets = targets.to(device)
            if is_binary:
                targets = targets.float().unsqueeze(-1)

            model.zero_grad()
            output = model(samples, input_chans)
            loss = criterion(output, targets)
            loss.backward()

            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    fisher_acc[name] += param.grad.detach().pow(2)

            seen += samples.shape[0]

        scale = 1.0 / max(seen, 1)
        for name in fisher_acc:
            self.fisher[name] = fisher_acc[name] * scale

        print(f"[EWC] Fisher computed over {seen} samples.")

    def penalty(self, model: nn.Module) -> torch.Tensor:
        loss = torch.tensor(0.0, device=next(model.parameters()).device)
        for name, param in model.named_parameters():
            if name in self.fisher:
                loss = loss + (self.fisher[name] * (param - self.params[name]).pow(2)).sum()
        return 0.5 * loss


# ---------------------------------------------------------------------------
# Knowledge Distillation loss
# ---------------------------------------------------------------------------

def kd_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor,
            temperature: float = 2.0) -> torch.Tensor:
    """
    Symmetric KL distillation.
    Returns T² * KL(soft_student || soft_teacher).
    """
    T = temperature
    p_s = F.log_softmax(student_logits / T, dim=-1)
    p_t = F.softmax(teacher_logits  / T, dim=-1)
    return F.kl_div(p_s, p_t, reduction='batchmean') * (T * T)


# ---------------------------------------------------------------------------
# Per-sample loss scoring
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_sample_losses(
    model: nn.Module,
    dataset,
    device: torch.device,
    input_chans,
    is_binary: bool,
    batch_size: int = 64,
    num_workers: int = 4,
) -> torch.Tensor:
    """
    Pass the whole dataset through the model and return a 1-D tensor of
    per-sample losses (length = len(dataset)), in original index order.
    """
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True, drop_last=False)
    criterion = (nn.BCEWithLogitsLoss(reduction='none')
                 if is_binary else nn.CrossEntropyLoss(reduction='none'))
    model.eval()
    all_losses = []
    for samples, targets in loader:
        samples = samples.float().to(device) / 100
        samples = rearrange(samples, 'B N (A T) -> B N A T', T=200)
        targets = targets.to(device)
        if is_binary:
            targets = targets.float().unsqueeze(-1)
        loss = criterion(model(samples, input_chans), targets)
        if loss.ndim > 1:
            loss = loss.mean(dim=1)
        all_losses.append(loss.cpu())
    return torch.cat(all_losses)   # (N,)


# ---------------------------------------------------------------------------
# Mine hard examples
# ---------------------------------------------------------------------------

@torch.no_grad()
def mine_hard_samples(model: nn.Module, dataset, device: torch.device,
                      input_chans, is_binary: bool,
                      hard_ratio: float = 0.3,
                      hard_min_loss: float = -1.0,
                      batch_size: int = 64,
                      num_workers: int = 4) -> tuple[List[int], torch.Tensor]:
    """
    Score every sample, return (hard_indices, all_losses).

    Selection criterion (union if both given):
      - top `hard_ratio` fraction by loss
      - loss > `hard_min_loss`
    """
    all_losses = compute_sample_losses(
        model, dataset, device, input_chans, is_binary, batch_size, num_workers)
    N = len(all_losses)
    hard_idx: set = set()

    if hard_ratio > 0:
        k = max(1, int(N * hard_ratio))
        hard_idx.update(torch.topk(all_losses, k).indices.tolist())

    if hard_min_loss > 0:
        hard_idx.update(
            (all_losses > hard_min_loss).nonzero(as_tuple=True)[0].tolist())

    hard_idx = sorted(hard_idx)
    print(f"[Mining] {len(hard_idx)}/{N} hard examples selected "
          f"(mean loss {all_losses.mean():.4f}, "
          f"hard mean {all_losses[list(hard_idx)].mean():.4f})")
    return hard_idx, all_losses


# ---------------------------------------------------------------------------
# Curriculum learning
# ---------------------------------------------------------------------------

class CurriculumScheduler:
    """
    Controls sample selection and sampling weights at each training epoch.

    Two strategies
    --------------
    'easy_to_hard'  (classical Bengio curriculum)
        Start by including only the easiest samples (lowest loss); expand the
        eligible set over time until all samples are covered.  Protects early
        training from noisy gradients caused by examples the model cannot yet
        handle.

    'hard_focus'  (progressive hard-example weighting)
        Always draw from the full selected set, but start with uniform weights
        and progressively up-weight the harder (higher-loss) examples.
        Avoids the abrupt switch to hard-only that standard hard mining uses.

    Pacing functions
    ----------------
    'linear'  c = t / T
    'root'    c = sqrt(t / T)   — faster warm-up, recommended default
    'step'    c ∈ {0, 0.5, 1}  — three discrete phases

    Parameters
    ----------
    sample_losses : np.ndarray  shape (N,)
        Per-sample loss values at the time of (re-)mining.
    strategy : str
        'easy_to_hard' | 'hard_focus'
    total_epochs : int
    pacing : str
        'linear' | 'root' | 'step'
    max_alpha : float
        (hard_focus) final exponent in w_i ∝ loss_i^α.
        α=0 → uniform; α=max_alpha → strongly focused on hardest.
    warmup_epochs : int
        Epochs at the very start where uniform/full-set sampling is used
        regardless of strategy.  Gives the model a stable starting point.
    min_fraction : float
        (easy_to_hard) minimum fraction of the dataset included even at
        competence=0.  Prevents the training set from shrinking to nothing.
    """

    def __init__(
        self,
        sample_losses: np.ndarray,
        strategy: str,
        total_epochs: int,
        pacing: str = 'root',
        max_alpha: float = 3.0,
        warmup_epochs: int = 1,
        min_fraction: float = 0.1,
    ):
        self.strategy       = strategy
        self.total_epochs   = total_epochs
        self.pacing         = pacing
        self.max_alpha      = max_alpha
        self.warmup_epochs  = warmup_epochs
        self.min_fraction   = min_fraction
        self._set_losses(sample_losses)

    def _set_losses(self, losses: np.ndarray):
        self._raw = losses.copy()
        lo, hi = losses.min(), losses.max()
        self._normed = (losses - lo) / (hi - lo + 1e-9)   # ∈ [0, 1]

    def update_losses(self, new_losses: np.ndarray) -> None:
        """Refresh difficulty scores after a re-mining pass."""
        self._set_losses(new_losses)

    # ── competence ────────────────────────────────────────────────────────────

    def competence(self, epoch: int) -> float:
        """Scalar progress value c ∈ [0, 1] for the given epoch."""
        if epoch < self.warmup_epochs:
            return 0.0
        span = max(self.total_epochs - self.warmup_epochs - 1, 1)
        t = np.clip((epoch - self.warmup_epochs) / span, 0.0, 1.0)
        if self.pacing == 'linear':
            return float(t)
        elif self.pacing == 'root':
            return float(np.sqrt(t))
        else:   # step: three equal phases
            return float(0.0 if t < 1/3 else (0.5 if t < 2/3 else 1.0))

    # ── sampler construction ──────────────────────────────────────────────────

    def get_sampler(
        self,
        epoch: int,
        indices: Optional[List[int]] = None,
    ) -> torch.utils.data.WeightedRandomSampler:
        """
        Build a WeightedRandomSampler for one epoch.

        Parameters
        ----------
        epoch : int
        indices : list of int, optional
            Restrict to a subset of the scored population (e.g. hard examples).
            If None, use all scored samples.
        """
        c      = self.competence(epoch)
        losses = self._normed if indices is None else self._normed[indices]
        N      = len(losses)

        if self.strategy == 'easy_to_hard':
            # threshold grows from min_fraction to 1.0
            threshold = self.min_fraction + (1.0 - self.min_fraction) * c
            # "easiness" = 1 − normalised_loss
            easiness = 1.0 - losses
            # include the `threshold` fraction of easiest samples
            cutoff = np.quantile(easiness, 1.0 - threshold)
            mask    = (easiness >= cutoff).astype(np.float32)
            weights = mask
        else:   # hard_focus
            alpha   = c * self.max_alpha
            weights = np.power(losses + 1e-9, alpha).astype(np.float32)

        weights_t = torch.tensor(weights, dtype=torch.float64)
        return torch.utils.data.WeightedRandomSampler(
            weights=weights_t,
            num_samples=N,
            replacement=True,
        )

    def describe(self, epoch: int) -> str:
        c = self.competence(epoch)
        if self.strategy == 'easy_to_hard':
            frac = self.min_fraction + (1.0 - self.min_fraction) * c
            return (f"curriculum easy_to_hard | pacing={self.pacing} "
                    f"| c={c:.2f} → include easiest {frac*100:.0f}%")
        else:
            alpha = c * self.max_alpha
            return (f"curriculum hard_focus | pacing={self.pacing} "
                    f"| c={c:.2f} → loss^{alpha:.2f} sampling")


# ---------------------------------------------------------------------------
# Freeze helpers (Mode A)
# ---------------------------------------------------------------------------

def set_trainable_last_n(model: nn.Module, last_n: int):
    """
    Freeze everything, then unfreeze the last `last_n` transformer blocks,
    the layer norms (norm / fc_norm), and the classification head.
    Works for both raw NeuralTransformer and ModelWithAdapter.
    """
    base = model.base if isinstance(model, ModelWithAdapter) else model

    # freeze all first
    for p in base.parameters():
        p.requires_grad_(False)

    # unfreeze last N blocks
    n_blocks = len(base.blocks)
    for i, blk in enumerate(base.blocks):
        if i >= n_blocks - last_n:
            for p in blk.parameters():
                p.requires_grad_(True)

    # unfreeze norms and head
    for module in [base.norm, base.fc_norm, base.head]:
        if module is not None:
            for p in module.parameters():
                p.requires_grad_(True)

    # always unfreeze adapter if present
    if isinstance(model, ModelWithAdapter):
        for p in model.adapter.parameters():
            p.requires_grad_(True)

    _print_trainable(model)


def set_all_trainable(model: nn.Module):
    for p in model.parameters():
        p.requires_grad_(True)
    _print_trainable(model)


def _print_trainable(model: nn.Module):
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Params] trainable {trainable/1e6:.2f}M / total {total/1e6:.2f}M "
          f"({100*trainable/total:.1f}%)")


# ---------------------------------------------------------------------------
# Training one epoch
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    data_loader,
    optimizer,
    device: torch.device,
    epoch: int,
    loss_scaler,
    clip_grad: float = 0.0,
    input_chans=None,
    is_binary: bool = True,
    criterion: nn.Module = None,
    # continual learning
    ewc: Optional[EWC] = None,
    ewc_lambda: float = 1000.0,
    teacher_model: Optional[nn.Module] = None,
    kd_weight: float = 1.0,
    kd_temperature: float = 2.0,
    # schedules
    lr_schedule_values=None,
    wd_schedule_values=None,
    start_steps: int = 0,
    log_writer=None,
):
    model.train()
    if teacher_model is not None:
        teacher_model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = f'Epoch [{epoch}]'

    for step, (samples, targets) in enumerate(
            metric_logger.log_every(data_loader, 10, header)):
        it = start_steps + step
        if lr_schedule_values is not None:
            for pg in optimizer.param_groups:
                pg["lr"] = lr_schedule_values[it] * pg.get("lr_scale", 1.0)
        if wd_schedule_values is not None:
            for pg in optimizer.param_groups:
                if pg["weight_decay"] > 0:
                    pg["weight_decay"] = wd_schedule_values[it]

        samples = samples.float().to(device, non_blocking=True) / 100
        samples = rearrange(samples, 'B N (A T) -> B N A T', T=200)
        targets = targets.to(device, non_blocking=True)
        if is_binary:
            targets = targets.float().unsqueeze(-1)

        with torch.autocast(device_type=device.type if device.type != 'mps' else 'cpu'):
            logits = model(samples, input_chans)
            task_loss = criterion(logits, targets)

            total_loss = task_loss

            # EWC regularisation
            if ewc is not None:
                base_m = model.base if isinstance(model, ModelWithAdapter) else model
                ewc_reg = ewc.penalty(base_m)
                total_loss = total_loss + ewc_lambda * ewc_reg
                metric_logger.update(ewc_loss=ewc_reg.item())

            # Knowledge Distillation
            if teacher_model is not None:
                with torch.no_grad():
                    teacher_logits = teacher_model(samples, input_chans)
                kd = kd_loss(logits, teacher_logits, temperature=kd_temperature)
                total_loss = total_loss + kd_weight * kd
                metric_logger.update(kd_loss=kd.item())

        loss_value = total_loss.item()
        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training.", flush=True)
            sys.exit(1)

        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        grad_norm = loss_scaler(total_loss, optimizer, clip_grad=clip_grad,
                                parameters=model.parameters(),
                                create_graph=is_second_order)
        optimizer.zero_grad()

        torch.cuda.synchronize() if device.type == 'cuda' else None

        metric_logger.update(loss=loss_value)
        metric_logger.update(task_loss=task_loss.item())

        max_lr = max(pg["lr"] for pg in optimizer.param_groups)
        metric_logger.update(lr=max_lr)

        if log_writer is not None:
            log_writer.update(loss=loss_value, head="loss")
            log_writer.update(lr=max_lr, head="opt")
            log_writer.set_step()

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model: nn.Module, data_loader, device: torch.device,
             input_chans, is_binary: bool, metrics: list):
    model.eval()
    if is_binary:
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss()

    all_preds, all_targets = [], []
    total_loss = 0.0
    n_batches  = 0

    for samples, targets in data_loader:
        samples = samples.float().to(device) / 100
        samples = rearrange(samples, 'B N (A T) -> B N A T', T=200)
        targets = targets.to(device)
        if is_binary:
            targets = targets.float().unsqueeze(-1)

        with torch.autocast(device_type=device.type if device.type != 'mps' else 'cpu'):
            logits = model(samples, input_chans)
            total_loss += criterion(logits, targets).item()
        n_batches += 1

        if is_binary:
            preds = torch.sigmoid(logits).cpu()
            gt    = (targets >= 0.5).int().cpu()
        else:
            preds = logits.cpu()
            gt    = targets.cpu()

        all_preds.append(preds)
        all_targets.append(gt)

    all_preds   = torch.cat(all_preds).numpy()
    all_targets = torch.cat(all_targets).numpy()
    ret = utils.get_metrics(all_preds, all_targets, metrics, is_binary)
    ret['loss'] = total_loss / max(n_batches, 1)
    print("  " + "  ".join(f"{k}: {v:.4f}" for k, v in ret.items()))
    return ret


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def get_args():
    parser = argparse.ArgumentParser('Hard-mining fine-tuning', add_help=False)

    # --- checkpoint ---
    parser.add_argument('--finetune', required=True,
                        help='Path to the fine-tuned model checkpoint (.pth)')
    parser.add_argument('--output_dir', default='./hard_mining_output',
                        help='Directory to save checkpoints and logs')

    # --- model ---
    parser.add_argument('--model', default='base_patch200_200', type=str)
    parser.add_argument('--nb_classes', default=2, type=int)
    parser.add_argument('--drop', type=float, default=0.0)
    parser.add_argument('--drop_path', type=float, default=0.0)
    parser.add_argument('--attn_drop_rate', type=float, default=0.0)
    parser.add_argument('--use_mean_pooling', action='store_true', default=True)
    parser.add_argument('--init_scale', type=float, default=0.001)
    parser.add_argument('--rel_pos_bias', action='store_true', default=False)
    parser.add_argument('--abs_pos_emb', action='store_true', default=False)
    parser.add_argument('--layer_scale_init_value', type=float, default=0.1)
    parser.add_argument('--qkv_bias', action='store_true', default=True)
    parser.add_argument('--is_binary', action='store_true', default=False,
                        help='Binary classification task')

    # --- update mode ---
    parser.add_argument('--mode', choices=['A', 'B'], default='A',
                        help='A: same structure, change params  |  B: add adapter layer')
    # Mode A options
    parser.add_argument('--finetune_layers', choices=['last_n', 'all'], default='last_n',
                        help='(Mode A) which layers to unfreeze')
    parser.add_argument('--last_n_blocks', type=int, default=4,
                        help='(Mode A, last_n) number of transformer blocks to unfreeze '
                             '(from the end). Recommended: 4 for base-12, 6 for large-24.')
    # Mode B options
    parser.add_argument('--adapter_reduction', type=int, default=4,
                        help='(Mode B) bottleneck reduction factor for adapter')
    parser.add_argument('--adapter_dropout', type=float, default=0.0)
    parser.add_argument('--also_finetune_backbone', action='store_true', default=False,
                        help='(Mode B) additionally unfreeze the last N backbone blocks '
                             'alongside the adapter')

    # --- data strategy ---
    parser.add_argument('--data_strategy', choices=['full', 'hard_only'], default='full',
                        help='full: hard + all original data  |  '
                             'hard_only: hard examples with continual learning')
    parser.add_argument('--train_data_dir', required=True,
                        help='Path to original training data (same format as finetune_classification.py)')
    parser.add_argument('--dataset', default='IIIC', type=str,
                        help='Dataset identifier (same as finetune_classification.py)')
    parser.add_argument('--train_eeg_montage', default='average', type=str)
    parser.add_argument('--train_eeg_length', default=False, type=int)
    parser.add_argument('--training_data_dir', default='', type=str)
    # val data
    parser.add_argument('--val_data_dir', default='', type=str,
                        help='Optional validation data directory')

    # --- curriculum learning ---
    parser.add_argument('--curriculum',
                        choices=['none', 'easy_to_hard', 'hard_focus'],
                        default='none',
                        help=(
                            'none        — standard hard mining (default)\n'
                            'easy_to_hard — start with easy examples, '
                            'progressively include harder ones (classic curriculum)\n'
                            'hard_focus  — start with uniform sampling over selected '
                            'examples, progressively up-weight the hardest ones'
                        ))
    parser.add_argument('--curriculum_pacing',
                        choices=['linear', 'root', 'step'], default='root',
                        help='Pacing function: linear | root (faster warm-up) | step (3 phases)')
    parser.add_argument('--curriculum_warmup_epochs', type=int, default=1,
                        help='Epochs at start where uniform/full-set sampling is used '
                             'regardless of curriculum strategy')
    parser.add_argument('--curriculum_max_alpha', type=float, default=3.0,
                        help='(hard_focus) final exponent α in w∝loss^α. '
                             'Higher = more focused on hardest examples at end of training')
    parser.add_argument('--curriculum_min_fraction', type=float, default=0.1,
                        help='(easy_to_hard) minimum fraction of dataset included '
                             'at competence=0 (default 0.1)')

    # --- hard mining ---
    parser.add_argument('--hard_ratio', type=float, default=0.3,
                        help='Top fraction of training samples by loss to treat as hard')
    parser.add_argument('--hard_min_loss', type=float, default=-1.0,
                        help='Keep samples with loss > this threshold (disabled if <= 0)')
    parser.add_argument('--remine_every', type=int, default=5,
                        help='Re-mine hard examples every N epochs (0 = mine once)')

    # --- continual learning (hard_only mode) ---
    parser.add_argument('--cl_method', choices=['ewc', 'kd'], default='kd',
                        help='(hard_only) continual learning method: '
                             'ewc = Elastic Weight Consolidation, '
                             'kd  = Knowledge Distillation from original model')
    parser.add_argument('--ewc_lambda', type=float, default=1000.0,
                        help='(EWC) regularisation strength')
    parser.add_argument('--ewc_samples', type=int, default=2000,
                        help='(EWC) number of samples to estimate Fisher Information')
    parser.add_argument('--kd_weight', type=float, default=1.0,
                        help='(KD) weight of distillation loss')
    parser.add_argument('--kd_temperature', type=float, default=2.0,
                        help='(KD) softmax temperature for distillation')

    # --- training ---
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--min_lr', type=float, default=1e-6)
    parser.add_argument('--warmup_lr', type=float, default=1e-6)
    parser.add_argument('--warmup_epochs', type=int, default=2)
    parser.add_argument('--warmup_steps', type=int, default=-1)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--weight_decay_end', type=float, default=None)
    parser.add_argument('--clip_grad', type=float, default=3.0)
    parser.add_argument('--opt', default='adamw', type=str)
    parser.add_argument('--opt_eps', type=float, default=1e-8)
    parser.add_argument('--opt_betas', type=float, nargs='+', default=None)
    parser.add_argument('--layer_decay', type=float, default=1.0,
                        help='Layer-wise LR decay. Set 1.0 to disable.')
    parser.add_argument('--save_ckpt_freq', type=int, default=1)
    parser.add_argument('--smoothing', type=float, default=0.0)

    # --- misc ---
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--pin_mem', action='store_true', default=True)
    parser.add_argument('--log_dir', default=None)
    parser.add_argument('--eeg_montage', default='average')

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def load_checkpoint(model: nn.Module, ckpt_path: str):
    """Load a fine-tuned checkpoint saved by finetune_classification.py."""
    ckpt = torch.load(ckpt_path, map_location='cpu')
    # support various checkpoint formats
    state = (ckpt.get('model')
             or ckpt.get('state_dict')
             or ckpt.get('module')
             or ckpt)
    # strip 'module.' prefix produced by DDP
    state = {k.replace('module.', ''): v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[Checkpoint] Missing keys  : {missing[:10]}{'...' if len(missing)>10 else ''}")
    if unexpected:
        print(f"[Checkpoint] Unexpected keys: {unexpected[:10]}{'...' if len(unexpected)>10 else ''}")
    print(f"[Checkpoint] Loaded from {ckpt_path}")


def build_base_model(args) -> nn.Module:
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


# ---------------------------------------------------------------------------
# Dataset helpers — mirrors finetune_classification.get_dataset
# ---------------------------------------------------------------------------

def get_ch_names_and_metrics(args):
    """Return (ch_names, metrics, is_binary) for the chosen dataset."""
    ds = args.dataset
    is_binary = args.is_binary

    if ds in ('TUAB', 'TUEP'):
        ch_names = ['FP1', 'FP2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4',
                    'O1', 'O2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6',
                    'A1', 'A2', 'FZ', 'CZ', 'PZ', 'T1', 'T2']
        metrics   = ['pr_auc', 'roc_auc', 'accuracy', 'balanced_accuracy']
        is_binary = True
    elif ds == 'TUEV':
        ch_names = ['FP1', 'FP2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4',
                    'O1', 'O2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6',
                    'A1', 'A2', 'FZ', 'CZ', 'PZ', 'T1', 'T2']
        metrics   = ['accuracy', 'balanced_accuracy', 'cohen_kappa', 'f1_weighted']
    elif ds in ('IIIC', 'IIIC_hm', 'IIIC_chewing'):
        if getattr(args, 'train_eeg_montage', 'average') == 'bipolar':
            ch_names = ['FP1-F7', 'F7-T3', 'T3-T5', 'T5-O1', 'FP2-F8', 'F8-T4', 'T4-T6',
                        'T6-O2', 'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1', 'FP2-F4', 'F4-C4',
                        'C4-P4', 'P4-O2', 'FZ-CZ', 'CZ-PZ']
        else:
            ch_names = ['FP1', 'F3', 'C3', 'P3', 'F7', 'T3', 'T5', 'O1', 'FZ', 'CZ', 'PZ',
                        'FP2', 'F4', 'C4', 'P4', 'F8', 'T4', 'T6', 'O2']
        metrics   = ['accuracy', 'balanced_accuracy', 'cohen_kappa', 'f1_weighted']
    elif ds == 'SLEEP':
        ch_names = ['FP1', 'F3', 'C3', 'P3', 'F7', 'T3', 'T5', 'O1', 'FZ', 'CZ', 'PZ',
                    'FP2', 'F4', 'C4', 'P4', 'F8', 'T4', 'T6', 'O2']
        metrics   = ['accuracy', 'balanced_accuracy', 'cohen_kappa', 'f1_weighted']
    else:
        raise ValueError(f"Unknown dataset: {ds}")

    return ch_names, metrics, is_binary


def build_datasets(args, ch_names):
    """Build train (and optionally val) datasets from args.train_data_dir."""
    # Support either a single directory or a list file
    # We reuse utils.build_pretraining_dataset when available; fall back to
    # a generic folder-based dataset otherwise.
    train_dir = args.train_data_dir or args.training_data_dir
    if not train_dir:
        raise ValueError("--train_data_dir must be specified")

    # Try to use the same dataset loader as finetune_classification
    try:
        from utils import EEGDataset
        train_dataset = EEGDataset(train_dir, ch_names=ch_names,
                                   sample_length=getattr(args, 'train_eeg_length', False))
    except Exception:
        # Fall back: assume the directory has .pt / .pkl files loadable as (data, label) tuples
        train_dataset = _FolderDataset(train_dir)

    val_dataset = None
    if args.val_data_dir:
        try:
            from utils import EEGDataset
            val_dataset = EEGDataset(args.val_data_dir, ch_names=ch_names,
                                     sample_length=getattr(args, 'train_eeg_length', False))
        except Exception:
            val_dataset = _FolderDataset(args.val_data_dir)

    return train_dataset, val_dataset


class _FolderDataset(torch.utils.data.Dataset):
    """
    Minimal fallback dataset: expects a directory of .pt files each containing
    a (data_tensor, label_tensor) tuple.
    """
    def __init__(self, root: str):
        self.files = sorted(Path(root).glob('*.pt'))
        if not self.files:
            raise FileNotFoundError(f"No .pt files found in {root}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        return torch.load(self.files[idx])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = get_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    # reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cudnn.benchmark = True

    # ------------------------------------------------------------------ #
    # 1. Build base model and load fine-tuned weights
    # ------------------------------------------------------------------ #
    print("\n[Step 1] Loading fine-tuned model ...")
    base_model = build_base_model(args)
    load_checkpoint(base_model, args.finetune)
    base_model.to(device)

    # ------------------------------------------------------------------ #
    # 2. Dataset and channel names
    # ------------------------------------------------------------------ #
    print("\n[Step 2] Building datasets ...")
    ch_names, metrics, is_binary = get_ch_names_and_metrics(args)
    is_binary = is_binary or args.is_binary
    input_chans = utils.get_input_chans(ch_names)

    train_dataset, val_dataset = build_datasets(args, ch_names)
    print(f"  Train dataset: {len(train_dataset)} samples")
    if val_dataset:
        print(f"  Val   dataset: {len(val_dataset)} samples")

    # ------------------------------------------------------------------ #
    # 3. Mine hard examples from the full training set
    # ------------------------------------------------------------------ #
    print(f"\n[Step 3] Mining hard examples (hard_ratio={args.hard_ratio}) ...")
    hard_indices, all_losses = mine_hard_samples(
        base_model, train_dataset, device, input_chans, is_binary,
        hard_ratio=args.hard_ratio,
        hard_min_loss=args.hard_min_loss,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    hard_dataset = Subset(train_dataset, hard_indices)

    # ------------------------------------------------------------------ #
    # 4. Continual learning setup (hard_only strategy)
    # ------------------------------------------------------------------ #
    ewc_obj      = None
    teacher_model = None

    if args.data_strategy == 'hard_only':
        full_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=args.pin_mem, drop_last=False)

        if args.cl_method == 'ewc':
            print("\n[Step 4] Computing Fisher Information Matrix for EWC ...")
            ewc_obj = EWC(base_model, full_loader, device, input_chans, is_binary,
                          n_samples=args.ewc_samples)
        else:  # kd
            print("\n[Step 4] Creating frozen teacher model for Knowledge Distillation ...")
            teacher_model = copy.deepcopy(base_model)
            teacher_model.eval()
            for p in teacher_model.parameters():
                p.requires_grad_(False)

    # ------------------------------------------------------------------ #
    # 4b. Curriculum scheduler (optional)
    # ------------------------------------------------------------------ #
    curriculum: Optional[CurriculumScheduler] = None
    if args.curriculum != 'none':
        print(f"\n[Step 4b] Setting up curriculum: strategy={args.curriculum}, "
              f"pacing={args.curriculum_pacing}, "
              f"warmup_epochs={args.curriculum_warmup_epochs}")
        # losses used by the curriculum:
        #   hard_only → score only the mined hard examples (focus within hard set)
        #   full      → score the full dataset (span easy-to-hard across everything)
        if args.data_strategy == 'hard_only':
            curriculum_losses = all_losses[hard_indices].numpy()
        else:
            curriculum_losses = all_losses.numpy()
        curriculum = CurriculumScheduler(
            sample_losses=curriculum_losses,
            strategy=args.curriculum,
            total_epochs=args.epochs,
            pacing=args.curriculum_pacing,
            max_alpha=args.curriculum_max_alpha,
            warmup_epochs=args.curriculum_warmup_epochs,
            min_fraction=args.curriculum_min_fraction,
        )
        print(f"  Epoch 0 state: {curriculum.describe(0)}")

    # ------------------------------------------------------------------ #
    # 5. Build training model (Mode A or B)
    # ------------------------------------------------------------------ #
    print(f"\n[Step 5] Configuring model for Mode {args.mode} ...")

    if args.mode == 'A':
        model = base_model
        if args.finetune_layers == 'last_n':
            print(f"  Unfreezing last {args.last_n_blocks} blocks + norm + head")
            set_trainable_last_n(model, args.last_n_blocks)
        else:
            print("  Unfreezing ALL layers")
            set_all_trainable(model)
    else:  # Mode B
        print(f"  Adding residual adapter (reduction={args.adapter_reduction})")
        model = ModelWithAdapter(
            base_model,
            reduction=args.adapter_reduction,
            adapter_dropout=args.adapter_dropout,
        )
        # by default freeze backbone, train only adapter + head
        for p in model.base.parameters():
            p.requires_grad_(False)
        for p in model.adapter.parameters():
            p.requires_grad_(True)
        for p in model.base.head.parameters():
            p.requires_grad_(True)

        if args.also_finetune_backbone:
            print(f"  Also unfreezing last {args.last_n_blocks} backbone blocks")
            set_trainable_last_n(model, args.last_n_blocks)
        else:
            _print_trainable(model)

    model.to(device)

    # ------------------------------------------------------------------ #
    # 6. Build training data loader
    # ------------------------------------------------------------------ #
    def _make_train_loader(epoch: int = 0) -> DataLoader:
        """
        Build a DataLoader for one epoch, respecting the curriculum schedule
        (if active).  When no curriculum is used, falls back to plain shuffle.
        """
        if args.data_strategy == 'full':
            dataset_for_epoch = ConcatDataset([train_dataset, hard_dataset])
        else:
            dataset_for_epoch = hard_dataset

        if curriculum is not None:
            # When curriculum is active, the WeightedRandomSampler already
            # controls the difficulty distribution, so we don't need the
            # hard_dataset duplication (it would cause a sampler/dataset
            # length mismatch for the 'full' strategy).
            # Use train_dataset directly (full) or hard_dataset (hard_only).
            base_dataset = (hard_dataset if args.data_strategy == 'hard_only'
                            else train_dataset)
            sampler = curriculum.get_sampler(epoch)
            return DataLoader(
                base_dataset,
                batch_size=args.batch_size,
                sampler=sampler,
                num_workers=args.num_workers,
                pin_memory=args.pin_mem,
                drop_last=True,
            )
        else:
            return DataLoader(
                dataset_for_epoch,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=args.num_workers,
                pin_memory=args.pin_mem,
                drop_last=True,
            )

    if args.data_strategy == 'full':
        print(f"\n[Step 6] Training on ALL data + hard examples "
              f"({len(train_dataset) + len(hard_dataset)} samples, "
              f"hard examples duplicated)")
    else:
        print(f"\n[Step 6] Training on hard examples only: {len(hard_dataset)} samples")
    if curriculum is not None:
        print(f"  Curriculum active: {args.curriculum} / {args.curriculum_pacing}")

    train_loader = _make_train_loader(epoch=0)

    val_loader = None
    if val_dataset:
        val_loader = DataLoader(
            val_dataset, batch_size=args.batch_size * 2, shuffle=False,
            num_workers=args.num_workers, pin_memory=args.pin_mem, drop_last=False)

    # ------------------------------------------------------------------ #
    # 7. Optimizer, loss, and LR schedule
    # ------------------------------------------------------------------ #
    n_steps_per_epoch = len(train_loader)

    if args.layer_decay < 1.0:
        # layer-wise LR decay (mirrors finetune_classification.py)
        n_layers = (model.base if isinstance(model, ModelWithAdapter) else model).get_num_layers()
        assigner = utils.LayerDecayValueAssigner(
            [args.layer_decay ** (n_layers + 1 - i) for i in range(n_layers + 2)])
        param_groups = utils.get_parameter_groups(
            model, args.weight_decay,
            assigner.get_scale, assigner.get_layer_id)
        optimizer = utils.create_optimizer(args, model, skip_list=None,
                                           get_num_layer=assigner.get_layer_id,
                                           get_layer_scale=assigner.get_scale)
    else:
        optimizer = utils.create_optimizer(args, model)

    loss_scaler = NativeScaler()

    if args.smoothing > 0:
        from timm.loss import LabelSmoothingCrossEntropy
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    elif is_binary:
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss()

    lr_schedule_values = utils.cosine_scheduler(
        args.lr, args.min_lr, args.epochs, n_steps_per_epoch,
        warmup_epochs=args.warmup_epochs, warmup_steps=args.warmup_steps)

    wd_end = args.weight_decay_end if args.weight_decay_end is not None else args.weight_decay
    wd_schedule_values = utils.cosine_scheduler(
        args.weight_decay, wd_end, args.epochs, n_steps_per_epoch)

    log_writer = None
    if args.log_dir and utils.is_main_process():
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = utils.TensorboardLogger(log_dir=args.log_dir)

    # ------------------------------------------------------------------ #
    # 8. Training loop
    # ------------------------------------------------------------------ #
    print(f"\n[Step 8] Starting hard-mining training for {args.epochs} epochs ...")
    print(f"  Mode           : {args.mode}")
    print(f"  Data strategy  : {args.data_strategy}"
          + (f" ({args.cl_method})" if args.data_strategy == 'hard_only' else ''))
    print(f"  Steps/epoch    : {n_steps_per_epoch}")
    print()

    best_metric = -1.0
    start_time  = time.time()
    log_stats   = {}

    for epoch in range(args.epochs):
        # re-mine hard examples periodically
        if args.remine_every > 0 and epoch > 0 and epoch % args.remine_every == 0:
            print(f"\n[Epoch {epoch}] Re-mining hard examples ...")
            hard_indices, all_losses = mine_hard_samples(
                model, train_dataset, device, input_chans, is_binary,
                hard_ratio=args.hard_ratio,
                hard_min_loss=args.hard_min_loss,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
            )
            hard_dataset = Subset(train_dataset, hard_indices)
            # refresh curriculum difficulty scores with updated losses
            if curriculum is not None:
                new_curriculum_losses = (
                    all_losses[hard_indices].numpy()
                    if args.data_strategy == 'hard_only'
                    else all_losses.numpy()
                )
                curriculum.update_losses(new_curriculum_losses)

        # rebuild loader each epoch when curriculum is active (weights change)
        # or after re-mining (dataset changed)
        if curriculum is not None or (
            args.remine_every > 0 and epoch > 0 and epoch % args.remine_every == 0
        ):
            train_loader = _make_train_loader(epoch=epoch)
            n_steps_per_epoch = len(train_loader)
            if curriculum is not None:
                print(f"  {curriculum.describe(epoch)}")

        train_stats = train_one_epoch(
            model, train_loader, optimizer, device, epoch, loss_scaler,
            clip_grad=args.clip_grad,
            input_chans=input_chans,
            is_binary=is_binary,
            criterion=criterion,
            ewc=ewc_obj,
            ewc_lambda=args.ewc_lambda,
            teacher_model=teacher_model,
            kd_weight=args.kd_weight,
            kd_temperature=args.kd_temperature,
            lr_schedule_values=lr_schedule_values,
            wd_schedule_values=wd_schedule_values,
            start_steps=epoch * n_steps_per_epoch,
            log_writer=log_writer,
        )

        # save checkpoint
        if args.output_dir and (epoch + 1) % args.save_ckpt_freq == 0:
            ckpt_path = os.path.join(args.output_dir, f'checkpoint_epoch{epoch:03d}.pth')
            save_obj = {'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'epoch': epoch,
                        'args': vars(args)}
            torch.save(save_obj, ckpt_path)

        log_stats = {f'train_{k}': v for k, v in train_stats.items()}
        log_stats['epoch'] = epoch

        # validation
        if val_loader is not None:
            print(f"  [Validation epoch {epoch}]")
            val_stats = evaluate(model, val_loader, device, input_chans,
                                 is_binary, metrics)
            log_stats.update({f'val_{k}': v for k, v in val_stats.items()})

            # save best model
            key = metrics[0]
            if val_stats.get(key, -1) > best_metric:
                best_metric = val_stats[key]
                best_path = os.path.join(args.output_dir, 'checkpoint_best.pth')
                torch.save({'model': model.state_dict(), 'epoch': epoch,
                            'best_metric': best_metric, 'args': vars(args)},
                           best_path)
                print(f"  [Best] {key}={best_metric:.4f} saved to {best_path}")

        # write log
        if args.output_dir:
            with open(os.path.join(args.output_dir, 'log.txt'), 'a') as f:
                f.write(json.dumps(log_stats) + '\n')

    # final checkpoint
    final_path = os.path.join(args.output_dir, 'checkpoint_final.pth')
    torch.save({'model': model.state_dict(), 'args': vars(args)}, final_path)
    print(f"\n[Done] Final model saved to {final_path}")
    print(f"[Done] Total time: {str(datetime.timedelta(seconds=int(time.time()-start_time)))}")


if __name__ == '__main__':
    main()
