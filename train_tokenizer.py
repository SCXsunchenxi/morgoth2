"""
train_tokenizer.py
==================
Training script for the VQ-NSP EEG tokenizer.

Objectives (all optional via CLI weights):
  - FFT amplitude reconstruction      (always on, weight = 1)
  - FFT phase reconstruction           (always on, weight = 1)
  - Raw signal reconstruction          (--signal_rec_weight)
  - Contrastive / InfoNCE              (--contrastive_weight)

Usage example:
    torchrun --nproc_per_node=4 train_tokenizer.py \
        --output_dir ./ckpt/tokenizer \
        --contrastive_weight 0.1 \
        --signal_rec_weight 1.0 \
        --epochs 100
"""

import argparse
import datetime
import json
import math
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from einops import rearrange
from timm.models import create_model

import tokenizer as tok_module
import utils
from utils import NativeScalerWithGradNormCount as GradScaler
from utils import create_optimizer


# ──────────────────────────────────────────────────────────────────────────────
# EEG augmentation
# ──────────────────────────────────────────────────────────────────────────────

def augment_eeg(
    x: torch.Tensor,
    noise_std: float = 0.05,
    shift_max: int = 50,
    scale_lo: float = 0.8,
    scale_hi: float = 1.2,
) -> torch.Tensor:
    """
    Stochastic augmentation for contrastive learning.
    x : [B, N_ch, T]  — signal already divided by 100

    Three independent perturbations applied in sequence:
      1. Additive Gaussian noise scaled to per-sample std
      2. Uniform amplitude scaling
      3. Circular time-domain shift
    """
    B, N, T = x.shape
    # 1. signal-adaptive noise
    sigma = x.std(dim=-1, keepdim=True).clamp(min=1e-6)
    x = x + torch.randn_like(x) * (noise_std * sigma)
    # 2. amplitude jitter
    gain = x.new_empty(B, 1, 1).uniform_(scale_lo, scale_hi)
    x = x * gain
    # 3. temporal roll
    delta = int(torch.randint(-shift_max, shift_max + 1, (1,)).item())
    x = torch.roll(x, delta, dims=-1)
    return x


# ──────────────────────────────────────────────────────────────────────────────
# Trainer
# ──────────────────────────────────────────────────────────────────────────────

class VQNSPTrainer:
    """
    Encapsulates the per-epoch training and validation logic for the VQ-NSP
    tokenizer.  Keeps mutable training state (model, optimiser, scaler) inside
    the object so the global ``main()`` function remains a thin orchestrator.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scaler: GradScaler,
        device: torch.device,
        args: argparse.Namespace,
    ) -> None:
        self.model = model
        self.opt = optimizer
        self.scaler = scaler
        self.device = device
        self.args = args

    # ------------------------------------------------------------------ public

    def run_epoch(
        self,
        loader_ch_pairs: List[Tuple],
        epoch: int,
        global_step_offset: int,
        lr_schedule: Optional[np.ndarray] = None,
        log_writer=None,
    ) -> Dict[str, float]:
        """
        Execute one training epoch over all data-loader / channel-name pairs.
        Returns a dict of averaged metrics.
        """
        self._reset_codebook_stats()

        self.model.train()
        tracker = utils.MetricLogger(delimiter="  ")
        tracker.add_meter("lr",     utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
        tracker.add_meter("min_lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
        header = f"Epoch [{epoch}]"

        cumulative_steps = 0
        for loader, ch_names in loader_ch_pairs:
            input_chans = utils.get_input_chans(ch_names)
            for local_step, batch in enumerate(
                tracker.log_every(loader, 10, header)
            ):
                global_step = global_step_offset + local_step + cumulative_steps

                # ---- learning-rate schedule ----
                if lr_schedule is not None:
                    for pg in self.opt.param_groups:
                        pg["lr"] = lr_schedule[global_step] * pg.get("lr_scale", 1.0)

                # ---- forward ----
                eeg = batch.float().to(self.device, non_blocking=True) / 100.0
                eeg_aug = augment_eeg(eeg) if self.args.contrastive_weight > 0 else None

                with torch.cuda.amp.autocast():
                    loss, loss_breakdown = self.model(
                        eeg,
                        x_aug=eeg_aug,
                        input_chans=input_chans,
                        contrastive_weight=self.args.contrastive_weight,
                        contrastive_temperature=self.args.contrastive_temperature,
                        signal_rec_weight=self.args.signal_rec_weight,
                    )

                loss_val = loss.item()
                if not math.isfinite(loss_val):
                    print(f"[WARN] non-finite loss={loss_val}, aborting.", flush=True)
                    utils.save_nan_model(self.args, self.model)
                    sys.exit(1)

                # ---- backward ----
                self.opt.zero_grad()
                is_second_order = getattr(self.opt, "is_second_order", False)
                grad_norm = self.scaler(
                    loss, self.opt,
                    clip_grad=self.args.clip_grad,
                    parameters=self.model.parameters(),
                    create_graph=is_second_order,
                )
                torch.cuda.synchronize()

                # ---- logging ----
                sub_losses = {
                    k.split("/")[-1]: v
                    for k, v in loss_breakdown.items()
                    if "total_loss" not in k
                }
                tracker.update(loss=loss_val, **sub_losses)
                lr_now, lr_min = self._lr_range()
                tracker.update(lr=lr_now, min_lr=lr_min)
                wd = self._current_wd()
                if wd is not None:
                    tracker.update(weight_decay=wd)
                tracker.update(grad_norm=grad_norm)

                if log_writer is not None:
                    log_writer.update(**sub_losses, head="train/loss")
                    log_writer.update(lr=lr_now, min_lr=lr_min, head="opt")
                    if wd is not None:
                        log_writer.update(weight_decay=wd, head="opt")
                    log_writer.update(
                        grad_norm=grad_norm,
                        loss_scale=self.scaler.state_dict()["scale"],
                        head="opt",
                    )
                    log_writer.set_step()

            cumulative_steps += local_step + 1

        tracker.synchronize_between_processes()
        print("Epoch stats:", tracker)
        return self._epoch_stats(tracker)

    @torch.no_grad()
    def validate(
        self,
        loader_ch_pairs: List[Tuple],
        log_writer=None,
        epoch: int = 0,
    ) -> Dict[str, float]:
        """Validation pass — no gradient computation."""
        self._reset_codebook_stats()
        self.model.eval()

        tracker = utils.MetricLogger(delimiter="  ")
        for loader, ch_names in loader_ch_pairs:
            input_chans = utils.get_input_chans(ch_names)
            sub_losses: Dict[str, float] = {}
            for batch in tracker.log_every(loader, 10, "Val"):
                eeg = batch.float().to(self.device, non_blocking=True) / 100.0
                loss, loss_breakdown = self.model(eeg, input_chans=input_chans)
                tracker.update(loss=loss.item())
                sub_losses = {
                    k.split("/")[-1]: v
                    for k, v in loss_breakdown.items()
                    if "total_loss" not in k
                }
            tracker.update(**sub_losses)

        tracker.synchronize_between_processes()
        print("Val stats:", tracker)
        return self._epoch_stats(tracker)

    @torch.no_grad()
    def codebook_utilization(self, loader: torch.utils.data.DataLoader) -> None:
        """Print fraction of unused codebook entries."""
        self.model.eval()
        n_codes = self.args.codebook_n_emd
        counts = torch.zeros(n_codes, dtype=torch.float64, device=self.device)

        for batch in loader:
            eeg = batch.float().to(self.device, non_blocking=True) / 100.0
            tokens = utils.get_model(self.model).get_tokens(eeg)["token"].view(-1)
            gathered = [torch.zeros_like(tokens) for _ in range(utils.get_world_size())]
            torch.distributed.all_gather(gathered, tokens)
            counts += torch.bincount(
                torch.cat(gathered).view(-1), minlength=n_codes
            )

        unused = (counts == 0).sum().item()
        print(
            f"[Codebook] {unused}/{n_codes} entries unused "
            f"({100.0 * unused / n_codes:.1f}%)"
        )

    # ------------------------------------------------------------------ private

    def _reset_codebook_stats(self) -> None:
        raw = utils.get_model(self.model)
        if hasattr(raw, "quantize"):
            try:
                raw.quantize.reset_cluster_size(self.device)
            except Exception:
                pass

    def _lr_range(self) -> Tuple[float, float]:
        lo, hi = 10.0, 0.0
        for pg in self.opt.param_groups:
            lo = min(lo, pg["lr"])
            hi = max(hi, pg["lr"])
        return hi, lo

    def _current_wd(self) -> Optional[float]:
        for pg in self.opt.param_groups:
            if pg.get("weight_decay", 0) > 0:
                return pg["weight_decay"]
        return None

    def _epoch_stats(self, tracker: utils.MetricLogger) -> Dict[str, float]:
        stats = {k: m.global_avg for k, m in tracker.meters.items()}
        # report unused codebook codes if available
        raw = utils.get_model(self.model)
        if hasattr(raw, "quantize"):
            try:
                cs = raw.quantize._codebook.cluster_size
            except AttributeError:
                cs = raw.quantize.cluster_size
            unused = (cs == 0).sum().item()
            stats["unused_codes"] = unused
            print(f"[Codebook] unused entries this epoch: {unused}")
        return stats


# ──────────────────────────────────────────────────────────────────────────────
# Dataset / dataloader helpers
# ──────────────────────────────────────────────────────────────────────────────

def _make_loader(
    dataset: torch.utils.data.Dataset,
    sampler: torch.utils.data.Sampler,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    drop_last: bool = True,
) -> torch.utils.data.DataLoader:
    return torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )


def build_loaders(
    args: argparse.Namespace,
) -> Tuple[List, Optional[List], List, Optional[List]]:
    """
    Build (train_loaders, val_loaders, train_ch_names, val_ch_names).
    Edit the dataset paths inside this function for your data.
    """
    datasets_train = [
        ["/data/routine_30_minutes_eeg_edf_hdf5/dataset.hdf5"]
    ]
    time_window_train = [10]

    datasets_val = [
        ["/data/routine_30_minutes_eeg_edf_hdf5/dataset.hdf5"]
    ]

    train_sets, train_ch = utils.build_pretraining_dataset(
        datasets_train, time_window_train, stride_size=200
    )

    if args.disable_eval:
        val_sets, val_ch = None, None
    else:
        val_sets, val_ch = utils.build_pretraining_dataset(datasets_val, [4])

    world  = utils.get_world_size()
    rank   = utils.get_rank()

    train_loaders, val_loaders = [], []

    for ds in train_sets:
        samp = torch.utils.data.DistributedSampler(
            ds, num_replicas=world, rank=rank, shuffle=True
        )
        train_loaders.append(
            _make_loader(ds, samp, args.batch_size, args.num_workers, args.pin_mem)
        )

    if val_sets is not None:
        for ds in val_sets:
            if args.dist_eval:
                samp = torch.utils.data.DistributedSampler(
                    ds, num_replicas=world, rank=rank, shuffle=False
                )
            else:
                samp = torch.utils.data.SequentialSampler(ds)
            val_loaders.append(
                _make_loader(
                    ds, samp,
                    int(1.5 * args.batch_size),
                    args.num_workers,
                    args.pin_mem,
                    drop_last=False,
                )
            )

    n_train = sum(len(ds) for ds in train_sets)
    steps_per_epoch = n_train // args.batch_size // world
    return train_loaders, val_loaders or None, train_ch, val_ch, steps_per_epoch


# ──────────────────────────────────────────────────────────────────────────────
# Model factory
# ──────────────────────────────────────────────────────────────────────────────

def build_vqnsp_model(args: argparse.Namespace) -> torch.nn.Module:
    return create_model(
        args.model,
        pretrained=False,
        as_tokenzer=False,
        n_code=args.codebook_n_emd,
        code_dim=args.codebook_emd_dim,
        EEG_size=args.input_size,
        decay=args.ema_decay,
        quantize_kmeans_init=args.quantize_kmeans_init,
    )


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("VQ-NSP tokenizer training", add_help=False)

    # ---- schedule ----
    p.add_argument("--batch_size",      default=128, type=int)
    p.add_argument("--epochs",          default=2,   type=int)
    p.add_argument("--save_ckpt_freq",  default=1,   type=int)
    p.add_argument("--start_epoch",     default=0,   type=int, metavar="N")

    # ---- model ----
    p.add_argument("--model", default="vqnsp_encoder_base_decoder_3x200x12",
                   type=str, metavar="MODEL")
    p.add_argument("--codebook_n_emd",       default=8192, type=int)
    p.add_argument("--codebook_emd_dim",     default=64,   type=int)
    p.add_argument("--ema_decay",            default=0.99, type=float)
    p.add_argument("--quantize_kmeans_init", action="store_true")
    p.add_argument("--input_size",           default=1600, type=int)

    # ---- objectives ----
    p.add_argument("--contrastive_weight",      default=0.1,  type=float,
                   help="InfoNCE loss weight (0 = disabled)")
    p.add_argument("--contrastive_temperature", default=0.1,  type=float,
                   help="temperature for InfoNCE")
    p.add_argument("--signal_rec_weight",       default=1.0,  type=float,
                   help="time-domain signal reconstruction weight (0 = disabled)")

    # ---- optimiser ----
    p.add_argument("--opt",             default="adamw", type=str, metavar="OPTIMIZER")
    p.add_argument("--opt_eps",         default=1e-8,    type=float, metavar="EPSILON")
    p.add_argument("--opt_betas",       default=None,    type=float, nargs="+")
    p.add_argument("--clip_grad",       type=float,      default=None, metavar="NORM")
    p.add_argument("--weight_decay",    type=float,      default=1e-4)
    p.add_argument("--weight_decay_end",type=float,      default=None)
    p.add_argument("--lr",              type=float,      default=5e-5, metavar="LR")
    p.add_argument("--warmup_lr",       type=float,      default=1e-6, metavar="LR")
    p.add_argument("--min_lr",          type=float,      default=1e-5, metavar="LR")
    p.add_argument("--warmup_epochs",   type=int,        default=1,  metavar="N")
    p.add_argument("--warmup_steps",    type=int,        default=-1, metavar="N")

    # ---- I/O ----
    p.add_argument("--output_dir",  default="")
    p.add_argument("--log_dir",     default=None)
    p.add_argument("--device",      default="cuda")
    p.add_argument("--seed",        default=0, type=int)
    p.add_argument("--resume",      default="")
    p.add_argument("--auto_resume", action="store_true")
    p.add_argument("--no_auto_resume", action="store_false", dest="auto_resume")
    p.set_defaults(auto_resume=True)

    p.add_argument("--dist_eval",              action="store_true", default=True)
    p.add_argument("--disable_eval",           action="store_true", default=False)
    p.add_argument("--eval",                   action="store_true", default=False)
    p.add_argument("--calculate_codebook_usage", action="store_true", default=False)

    p.add_argument("--num_workers", default=10, type=int)
    p.add_argument("--pin_mem",     action="store_true")
    p.add_argument("--no_pin_mem",  action="store_false", dest="pin_mem")
    p.set_defaults(pin_mem=True)

    # ---- distributed ----
    p.add_argument("--world_size",   default=1,       type=int)
    p.add_argument("--local_rank",   default=-1,      type=int)
    p.add_argument("--dist_on_itp",  action="store_true")
    p.add_argument("--dist_url",     default="env://")

    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main(args: argparse.Namespace) -> None:
    utils.init_distributed_mode(args)
    print(args)

    device = torch.device(args.device)
    torch.manual_seed(args.seed + utils.get_rank())
    np.random.seed(args.seed + utils.get_rank())
    cudnn.benchmark = True

    # ---- model ----
    model = build_vqnsp_model(args).to(device)

    # ---- summary ----
    for part_name in ("encoder", "decoder"):
        part = getattr(model, part_name)
        n_learn = sum(p.numel() for p in part.parameters() if p.requires_grad)
        n_fixed = sum(p.numel() for p in part.parameters() if not p.requires_grad)
        print(f"  {part_name}: {n_learn/1e6:.2f}M learnable, {n_fixed/1e6:.2f}M frozen")
    n_total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total learnable params: {n_total/1e6:.2f}M")

    # ---- data ----
    (
        train_loaders, val_loaders,
        train_ch_names, val_ch_names,
        steps_per_epoch,
    ) = build_loaders(args)

    # ---- LR scaling ----
    total_batch = args.batch_size * utils.get_world_size()
    args.lr = total_batch / 128 * args.lr
    print(f"  Effective LR: {args.lr:.2e}  |  Batch: {total_batch}  |  Steps/epoch: {steps_per_epoch}")

    # ---- optimiser ----
    optimizer = create_optimizer(args, model)
    scaler    = GradScaler()

    # ---- DDP ----
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], find_unused_parameters=True
        )
        model_without_ddp = model.module

    # ---- LR schedule ----
    lr_schedule = utils.cosine_scheduler(
        args.lr, args.min_lr, args.epochs, steps_per_epoch,
        warmup_epochs=args.warmup_epochs, warmup_steps=args.warmup_steps,
    )

    utils.auto_load_model(
        args=args, model=model, model_without_ddp=model_without_ddp,
        optimizer=optimizer, loss_scaler=scaler,
    )

    # ---- tensorboard ----
    log_writer = None
    if utils.get_rank() == 0 and args.log_dir:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = utils.TensorboardLogger(log_dir=args.log_dir)

    trainer = VQNSPTrainer(model, optimizer, scaler, device, args)
    train_pairs = list(zip(train_loaders, train_ch_names))
    val_pairs   = list(zip(val_loaders,   val_ch_names)) if val_loaders else None

    # ---- eval-only modes ----
    if args.eval:
        if val_pairs:
            trainer.validate(val_pairs, epoch=0)
        return

    if args.calculate_codebook_usage:
        if train_loaders:
            trainer.codebook_utilization(train_loaders[0])
        return

    # ---- training loop ----
    print(f"Starting training for {args.epochs} epochs")
    t0 = time.time()

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            for dl in train_loaders:
                dl.sampler.set_epoch(epoch)

        if log_writer is not None:
            log_writer.set_step(epoch * steps_per_epoch)

        train_stats = trainer.run_epoch(
            train_pairs, epoch,
            global_step_offset=epoch * steps_per_epoch,
            lr_schedule=lr_schedule,
            log_writer=log_writer,
        )

        if args.output_dir:
            utils.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp,
                optimizer=optimizer, loss_scaler=scaler,
                epoch=epoch, save_ckpt_freq=args.save_ckpt_freq,
            )

        log_row: Dict = {f"train_{k}": v for k, v in train_stats.items()}
        if val_pairs:
            val_stats = trainer.validate(val_pairs, log_writer=log_writer, epoch=epoch)
            n_val = sum(len(dl.dataset) for dl in val_loaders)
            print(f"Val loss ({n_val} samples): {val_stats.get('loss', float('nan')):.4f}")
            if log_writer is not None:
                log_writer.update(**val_stats, head="val/loss")
            log_row.update({f"val_{k}": v for k, v in val_stats.items()})
        log_row["epoch"] = epoch

        if args.output_dir and utils.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), "a", encoding="utf-8") as fh:
                fh.write(json.dumps(log_row) + "\n")

    elapsed = str(datetime.timedelta(seconds=int(time.time() - t0)))
    print(f"Training finished in {elapsed}")


if __name__ == "__main__":
    opts = get_args()
    if opts.output_dir:
        Path(opts.output_dir).mkdir(parents=True, exist_ok=True)
    main(opts)
