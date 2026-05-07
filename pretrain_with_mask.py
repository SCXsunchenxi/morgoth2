"""
pretrain_with_mask.py
=====================
Pre-training script for the EEG transformer backbone via:

  1. Bidirectional masked-token prediction (MLM-style, symmetric)
  2. Next-segment prediction / causal future masking  (--future_pred_weight)

Model structure is unchanged from the fine-tuned checkpoints — the two
objectives differ only in *which* tokens are masked, not in any parameter.

Usage example:
    torchrun --nproc_per_node=4 pretrain_with_mask.py \
        --tokenizer_weight ./ckpt/tokenizer/checkpoint.pth \
        --output_dir ./ckpt/pretrain \
        --future_pred_weight 1.0 \
        --epochs 200
"""

import argparse
import datetime
import json
import math
import os
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from einops import rearrange
from timm.models import create_model

import backbone
import tokenizer as tok_module
import utils
from utils import NativeScalerWithGradNormCount as GradScaler
from utils import create_optimizer


# ──────────────────────────────────────────────────────────────────────────────
# Masking strategies
# ──────────────────────────────────────────────────────────────────────────────

class MaskingPolicy:
    """
    Factory for boolean mask tensors consumed by NeuralTransformerForMEM.
    All methods return ``BoolTensor[B, N_ch*N_win]`` where ``True`` marks
    positions that the model must predict (i.e. are hidden from context).
    """

    @staticmethod
    def random_uniform(flat_tokens: torch.Tensor, mask_ratio: float) -> torch.BoolTensor:
        """
        Per-sample random masking via noise-argsort shuffling.
        flat_tokens : [B, L, D]  — only shape is used (values ignored)
        """
        B, L, _ = flat_tokens.shape
        n_keep   = int(L * (1.0 - mask_ratio))
        noise    = torch.rand(B, L, device=flat_tokens.device)
        order    = torch.argsort(noise, dim=1)           # ascending → small = keep
        restore  = torch.argsort(order,  dim=1)
        # build binary mask in shuffled space, then unshuffle
        mask = torch.ones(B, L, device=flat_tokens.device)
        mask[:, :n_keep] = 0
        mask = torch.gather(mask, 1, restore)
        return mask.bool()

    @staticmethod
    def causal_future(
        eeg_4d: torch.Tensor,
        future_frac: float = 0.5,
    ) -> torch.BoolTensor:
        """
        Causal / next-segment mask for temporal prediction.

        The last ``future_frac`` fraction of time windows is marked as future
        for every channel simultaneously.  The model receives the earlier
        context windows as input and must predict the VQ token ids of the
        masked future windows.

        eeg_4d : [B, N_ch, N_win, T_patch]   (already rearranged from raw EEG)
        Returns : [B, N_ch * N_win]
        """
        B, n_ch, n_win, _ = eeg_4d.shape
        context_len   = int(n_win * (1.0 - future_frac))
        # flattened position p → time index = p % n_win
        time_positions = torch.arange(n_ch * n_win, device=eeg_4d.device) % n_win
        future_mask    = (time_positions >= context_len)         # [n_ch * n_win]
        return future_mask.unsqueeze(0).expand(B, -1).contiguous().bool()


# ──────────────────────────────────────────────────────────────────────────────
# Pre-training engine
# ──────────────────────────────────────────────────────────────────────────────

class MaskedEEGPretrainer:
    """
    Runs one epoch of masked-token prediction pre-training, combining:
      - Symmetric MLM loss  (random masking, forward + symmetric pass)
      - Optional NSP loss   (causal future masking)

    The transformer model ``NeuralTransformerForMEM`` is called with only
    different ``bool_masked_pos`` values — no structural changes are needed.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scaler: GradScaler,
        device: torch.device,
        args: argparse.Namespace,
    ) -> None:
        self.model     = model
        self.tokenizer = tokenizer
        self.opt       = optimizer
        self.scaler    = scaler
        self.device    = device
        self.args      = args
        self._ce       = nn.CrossEntropyLoss()

    # ------------------------------------------------------------------ public

    def run_epoch(
        self,
        loader_ch_pairs: List[Tuple],
        epoch: int,
        global_step_offset: int,
        lr_schedule: Optional[np.ndarray]  = None,
        wd_schedule:  Optional[np.ndarray] = None,
        log_writer=None,
    ) -> Dict[str, float]:
        """Execute one pre-training epoch. Returns averaged metric dict."""
        self.model.train()

        tracker = utils.MetricLogger(delimiter="  ")
        tracker.add_meter("lr",     utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
        tracker.add_meter("min_lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
        header  = f"Epoch [{epoch}]"
        accum   = self.args.gradient_accumulation_steps

        cumulative = 0
        for loader, ch_names in loader_ch_pairs:
            if len(loader) == 0:
                continue
            input_chans = utils.get_input_chans(ch_names)

            for local_step, batch in enumerate(
                tracker.log_every(loader, 10 * accum, header)
            ):
                global_step = global_step_offset + local_step + cumulative

                # ---- per-step schedule update ----
                self._apply_schedules(global_step, lr_schedule, wd_schedule)

                # ---- prepare input ----
                raw = batch.float().to(self.device, non_blocking=True) / 100.0
                eeg = rearrange(raw, "B N (A T) -> B N A T", T=200)

                # ---- tokenise (no grad) ----
                with torch.no_grad():
                    with torch.cuda.amp.autocast():
                        token_ids = self.tokenizer.get_codebook_indices(eeg, input_chans)

                    # random mask labels
                    mlm_mask  = MaskingPolicy.random_uniform(
                        eeg.flatten(1, 2).unsqueeze(-1), mask_ratio=0.5
                    ).to(self.device)
                    labels_mlm     = token_ids[mlm_mask]
                    labels_mlm_sym = token_ids[~mlm_mask]

                    # future mask labels (computed once, used inside sync ctx)
                    nsp_mask, labels_nsp = None, None
                    if self.args.future_pred_weight > 0:
                        nsp_mask   = MaskingPolicy.causal_future(
                            eeg, future_frac=self.args.future_ratio
                        )
                        labels_nsp = token_ids[nsp_mask]

                # ---- forward ----
                sync_ctx = (
                    self.model.no_sync
                    if self.args.distributed and (local_step + 1) % accum != 0
                    else nullcontext
                )
                with sync_ctx():
                    with torch.cuda.amp.autocast():
                        pred_mlm, pred_mlm_sym = self.model(
                            eeg, input_chans, bool_masked_pos=mlm_mask
                        )
                        loss_mlm     = self._ce(pred_mlm,     labels_mlm)
                        loss_mlm_sym = self._ce(pred_mlm_sym, labels_mlm_sym)
                        total_loss   = loss_mlm + loss_mlm_sym

                        # NSP pass (same model, different mask)
                        pred_nsp = None
                        if nsp_mask is not None:
                            pred_nsp, _ = self.model(
                                eeg, input_chans, bool_masked_pos=nsp_mask
                            )
                            loss_nsp = self._ce(pred_nsp, labels_nsp)
                            total_loss = total_loss + self.args.future_pred_weight * loss_nsp

                loss_val = total_loss.item()
                if not math.isfinite(loss_val):
                    print(
                        f"[WARN] non-finite loss={loss_val} at rank "
                        f"{utils.get_rank()}, aborting.",
                        flush=True,
                    )
                    sys.exit(1)

                # ---- backward ----
                is_second_order = getattr(self.opt, "is_second_order", False)
                total_loss = total_loss / accum
                grad_norm  = self.scaler(
                    total_loss, self.opt,
                    clip_grad=self.args.clip_grad,
                    parameters=self.model.parameters(),
                    create_graph=is_second_order,
                    update_grad=(local_step + 1) % accum == 0,
                )
                scale_val = self.scaler.state_dict()["scale"]
                if (local_step + 1) % accum == 0:
                    self.opt.zero_grad()

                torch.cuda.synchronize()

                # ---- per-step metrics ----
                mlm_acc     = (pred_mlm.argmax(-1) == labels_mlm).float().mean().item()
                mlm_acc_sym = (pred_mlm_sym.argmax(-1) == labels_mlm_sym).float().mean().item()
                tracker.update(
                    loss=loss_val,
                    loss_mlm=loss_mlm.item() / 2,
                    mlm_acc=mlm_acc,
                    mlm_acc_sym=mlm_acc_sym,
                    loss_scale=scale_val,
                )
                if pred_nsp is not None:
                    nsp_acc = (pred_nsp.argmax(-1) == labels_nsp).float().mean().item()
                    tracker.update(nsp_acc=nsp_acc, loss_nsp=loss_nsp.item())

                lr_now, lr_min = self._lr_range()
                tracker.update(lr=lr_now, min_lr=lr_min)
                wd = self._current_wd()
                if wd is not None:
                    tracker.update(weight_decay=wd)
                tracker.update(grad_norm=grad_norm)

                if log_writer is not None:
                    log_writer.update(
                        mlm_acc=mlm_acc, mlm_acc_sym=mlm_acc_sym,
                        loss_mlm=loss_mlm.item() / 2,
                        head="loss",
                    )
                    if pred_nsp is not None:
                        log_writer.update(nsp_acc=nsp_acc, loss_nsp=loss_nsp.item(), head="loss")
                    log_writer.update(loss=loss_val, head="loss")
                    log_writer.update(
                        loss_scale=scale_val, lr=lr_now, min_lr=lr_min,
                        weight_decay=wd, grad_norm=grad_norm,
                        head="opt",
                    )
                    log_writer.set_step()

            cumulative += local_step + 1

        tracker.synchronize_between_processes()
        print("Epoch stats:", tracker)
        return {k: m.global_avg for k, m in tracker.meters.items()}

    # ------------------------------------------------------------------ private

    def _apply_schedules(
        self,
        step: int,
        lr_sched: Optional[np.ndarray],
        wd_sched: Optional[np.ndarray],
    ) -> None:
        for pg in self.opt.param_groups:
            if lr_sched is not None:
                pg["lr"] = lr_sched[step] * pg.get("lr_scale", 1.0)
            if wd_sched is not None and pg.get("weight_decay", 0) > 0:
                pg["weight_decay"] = wd_sched[step]

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


# ──────────────────────────────────────────────────────────────────────────────
# Model & tokenizer factories
# ──────────────────────────────────────────────────────────────────────────────

def build_pretrain_model(args: argparse.Namespace) -> torch.nn.Module:
    print(f"[Model] {args.model}")
    return create_model(
        args.model,
        pretrained=False,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
        use_shared_rel_pos_bias=args.rel_pos_bias,
        use_abs_pos_emb=args.abs_pos_emb,
        init_values=args.layer_scale_init_value,
        vocab_size=args.codebook_size,
    )


def build_eeg_tokenizer(args: argparse.Namespace) -> torch.nn.Module:
    print(f"[Tokenizer] {args.tokenizer_model}")
    return create_model(
        args.tokenizer_model,
        pretrained=True,
        pretrained_weight=args.tokenizer_weight,
        as_tokenzer=True,
        n_code=args.codebook_size,
        code_dim=args.codebook_dim,
    ).eval()


def build_loaders(args: argparse.Namespace) -> Tuple[List, List, int]:
    """
    Returns (train_loader_ch_pairs, train_steps_per_epoch).
    Edit the dataset paths here.
    """
    datasets_train = [
        ["/data/routine_30_minutes_eeg_edf_hdf5/dataset.hdf5"]
    ]
    time_window = [10]

    train_sets, train_ch = utils.build_pretraining_dataset(
        datasets_train, time_window,
        stride_size=200 * 10,
        start_percentage=0, end_percentage=1,
    )

    world = utils.get_world_size()
    rank  = utils.get_rank()

    loaders = []
    for ds in train_sets:
        samp = torch.utils.data.DistributedSampler(
            ds, num_replicas=world, rank=rank, shuffle=True
        )
        loaders.append(
            torch.utils.data.DataLoader(
                ds, sampler=samp,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                pin_memory=args.pin_mem,
                drop_last=True,
            )
        )

    n_train = sum(len(ds) for ds in train_sets)
    steps   = n_train // args.batch_size // world
    return loaders, train_ch, steps


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("EEG transformer pre-training", add_help=False)

    p.add_argument("--batch_size",     default=128, type=int)
    p.add_argument("--epochs",         default=2,   type=int)
    p.add_argument("--save_ckpt_freq", default=1,   type=int)
    p.add_argument("--start_epoch",    default=0,   type=int)

    # tokenizer
    p.add_argument("--tokenizer_weight", type=str)
    p.add_argument("--tokenizer_model",  type=str,
                   default="vqnsp_encoder_base_decoder_3x200x12")

    # backbone model
    p.add_argument("--model", default="base_patch200_1600_8k_vocab",
                   type=str, metavar="MODEL")
    p.add_argument("--rel_pos_bias",    action="store_true")
    p.add_argument("--disable_rel_pos_bias", action="store_true", dest="rel_pos_bias")
    p.set_defaults(rel_pos_bias=False)
    p.add_argument("--abs_pos_emb", action="store_true")
    p.set_defaults(abs_pos_emb=True)
    p.add_argument("--layer_scale_init_value", default=0.1, type=float)
    p.add_argument("--input_size",             default=1600, type=int)
    p.add_argument("--drop_path",              default=0.0,  type=float, metavar="PCT")

    # codebook
    p.add_argument("--codebook_size", default=8192, type=int)
    p.add_argument("--codebook_dim",  default=64,   type=int)

    # objectives
    p.add_argument("--future_pred_weight", default=1.0, type=float,
                   help="next-segment prediction loss weight (0 = disabled)")
    p.add_argument("--future_ratio",       default=0.5, type=float,
                   help="fraction of time windows treated as future targets")

    # optimiser
    p.add_argument("--opt",              default="adamw", type=str, metavar="OPTIMIZER")
    p.add_argument("--opt_eps",          default=1e-8,    type=float)
    p.add_argument("--opt_betas",        default=None,    type=float, nargs="+")
    p.add_argument("--clip_grad",        type=float,      default=3.0, metavar="NORM")
    p.add_argument("--momentum",         type=float,      default=0.9)
    p.add_argument("--weight_decay",     type=float,      default=0.05)
    p.add_argument("--weight_decay_end", type=float,      default=None)
    p.add_argument("--lr",               type=float,      default=5e-4, metavar="LR")
    p.add_argument("--warmup_lr",        type=float,      default=1e-6, metavar="LR")
    p.add_argument("--min_lr",           type=float,      default=1e-5, metavar="LR")
    p.add_argument("--warmup_epochs",    type=int,        default=1,  metavar="N")
    p.add_argument("--warmup_steps",     type=int,        default=-1, metavar="N")

    # I/O
    p.add_argument("--output_dir", default="")
    p.add_argument("--log_dir",    default=None)
    p.add_argument("--device",     default="cuda")
    p.add_argument("--seed",       default=0,  type=int)
    p.add_argument("--resume",     default="")
    p.add_argument("--auto_resume",    action="store_true")
    p.add_argument("--no_auto_resume", action="store_false", dest="auto_resume")
    p.set_defaults(auto_resume=True)

    p.add_argument("--num_workers", default=10, type=int)
    p.add_argument("--pin_mem",     action="store_true")
    p.add_argument("--no_pin_mem",  action="store_false", dest="pin_mem")
    p.set_defaults(pin_mem=True)

    # distributed
    p.add_argument("--world_size",  default=1,       type=int)
    p.add_argument("--local_rank",  default=-1,      type=int)
    p.add_argument("--dist_on_itp", action="store_true")
    p.add_argument("--dist_url",    default="env://")
    p.add_argument("--gradient_accumulation_steps", default=1, type=int)

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

    # ---- models ----
    model     = build_pretrain_model(args)
    args.patch_size   = model.patch_size
    args.window_size  = (1, args.input_size // model.patch_size)
    print(f"  Patch size: {model.patch_size}")

    eeg_tokenizer = build_eeg_tokenizer(args).to(device)

    # ---- data ----
    train_loaders, train_ch_names, steps_per_epoch = build_loaders(args)
    loader_ch_pairs = list(zip(train_loaders, train_ch_names))

    # ---- optimiser ----
    total_batch = args.batch_size * utils.get_world_size() * args.gradient_accumulation_steps
    print(f"  Effective batch size: {total_batch}  |  Steps/epoch: {steps_per_epoch}")
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable params: {n_params / 1e6:.2f}M")
    print(f"  Tokenizer: {sum(p.numel() for p in eeg_tokenizer.parameters())/1e6:.2f}M params")

    model.to(device)
    model_without_ddp = model

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], find_unused_parameters=True
        )
        model_without_ddp = model.module

    optimizer = create_optimizer(args, model_without_ddp)
    scaler    = GradScaler()

    lr_schedule = utils.cosine_scheduler(
        args.lr, args.min_lr, args.epochs, steps_per_epoch,
        warmup_epochs=args.warmup_epochs, warmup_steps=args.warmup_steps,
    )
    if args.weight_decay_end is None:
        args.weight_decay_end = args.weight_decay
    wd_schedule = utils.cosine_scheduler(
        args.weight_decay, args.weight_decay_end,
        args.epochs, steps_per_epoch,
    )
    print(f"  WD range: [{min(wd_schedule):.4e}, {max(wd_schedule):.4e}]")

    utils.auto_load_model(
        args=args, model=model, model_without_ddp=model_without_ddp,
        optimizer=optimizer, loss_scaler=scaler,
    )

    log_writer = None
    if utils.get_rank() == 0 and args.log_dir:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = utils.TensorboardLogger(log_dir=args.log_dir)

    engine = MaskedEEGPretrainer(
        model, eeg_tokenizer, optimizer, scaler, device, args
    )

    print(f"Starting pre-training for {args.epochs} epochs")
    t0 = time.time()

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            for dl in train_loaders:
                dl.sampler.set_epoch(epoch)

        if log_writer is not None:
            log_writer.set_step(epoch * steps_per_epoch)

        train_stats = engine.run_epoch(
            loader_ch_pairs, epoch,
            global_step_offset=epoch * steps_per_epoch,
            lr_schedule=lr_schedule,
            wd_schedule=wd_schedule,
            log_writer=log_writer,
        )

        if args.output_dir:
            utils.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp,
                optimizer=optimizer, loss_scaler=scaler,
                epoch=epoch, save_ckpt_freq=args.save_ckpt_freq,
            )

        log_row = {f"train_{k}": v for k, v in train_stats.items()}
        log_row["epoch"] = epoch

        if args.output_dir and utils.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), "a", encoding="utf-8") as fh:
                fh.write(json.dumps(log_row) + "\n")

    elapsed = str(datetime.timedelta(seconds=int(time.time() - t0)))
    print(f"Pre-training finished in {elapsed}")


if __name__ == "__main__":
    opts = get_args()
    if opts.output_dir:
        Path(opts.output_dir).mkdir(parents=True, exist_ok=True)
    main(opts)
