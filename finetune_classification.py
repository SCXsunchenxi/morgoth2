import argparse
import datetime
from pyexpat import model
import numpy as np
import time
import torch.backends.cudnn as cudnn
import json
import os
import pandas as pd
from pathlib import Path
import torch.distributed as dist
from sklearn.tests.test_multiclass import n_classes
from tqdm import tqdm
from collections import OrderedDict
from timm.data.mixup import Mixup
from timm.models import create_model
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from utils import FocalLoss, DynamicFocalLoss,GHMC,MarginLoss, BinaryFocalLoss,MultiLabelFocalLoss
from utils import create_optimizer, get_parameter_groups, LayerDecayValueAssigner
from utils import NativeScalerWithGradNormCount as NativeScaler
import task_model
from scipy import interpolate
import math
import sys
from typing import Iterable, Optional
import torch
import torch.nn.functional as F
from timm.utils import ModelEma
import utils
from einops import rearrange

import contextlib

import warnings
warnings.filterwarnings("ignore")


def get_args():
    parser = argparse.ArgumentParser('Fine-tuning and evaluation script for EEG classification', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--epochs', default=30, type=int)
    parser.add_argument('--update_freq', default=1, type=int)
    parser.add_argument('--save_ckpt_freq', default=1, type=int)
    parser.add_argument('--distributed', default=True, type= bool, help='Enabling distributed')

    # robust evaluation
    parser.add_argument('--robust_test', default=None, type=str,
                        help='robust evaluation dataset')

    # Model parameters
    parser.add_argument('--model', default='base_patch200_200', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--qkv_bias', action='store_true')
    parser.add_argument('--disable_qkv_bias', action='store_false', dest='qkv_bias')
    parser.set_defaults(qkv_bias=True)
    parser.add_argument('--rel_pos_bias', action='store_true')
    parser.add_argument('--disable_rel_pos_bias', action='store_false', dest='rel_pos_bias')
    parser.set_defaults(rel_pos_bias=True)
    parser.add_argument('--abs_pos_emb', action='store_true')
    parser.set_defaults(abs_pos_emb=False)
    parser.add_argument('--layer_scale_init_value', default=0.1, type=float,
                        help="0.1 for base, 1e-5 for large. set 0 to disable layer scale")

    parser.add_argument('--input_size', default=200, type=int,
                        help='EEG input size')

    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--attn_drop_rate', type=float, default=0.0, metavar='PCT',
                        help='Attention dropout rate (default: 0.)')
    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')

    parser.add_argument('--disable_eval_during_finetuning', action='store_true', default=False)

    parser.add_argument('--model_ema', action='store_true', default=False)
    parser.add_argument('--model_ema_decay', type=float, default=0.9999, help='')
    parser.add_argument('--model_ema_force_cpu', action='store_true', default=False, help='')

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt_eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt_betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--weight_decay_end', type=float, default=None, help="""Final value of the
        weight decay. We use a cosine schedule for WD and using a larger decay by
        the end of training improves performance for ViTs.""")

    parser.add_argument('--lr', type=float, default=5e-4, metavar='LR',
                        help='learning rate (default: 5e-4)')
    parser.add_argument('--layer_decay', type=float, default=0.9)
    parser.add_argument('--warmup_lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')

    parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--warmup_steps', type=int, default=-1, metavar='N',
                        help='num of steps to warmup LR, will overload warmup_epochs if set > 0')

    parser.add_argument('--smoothing', type=float, default=0.1,
                        help='Label smoothing (default: 0.1)')

    parser.add_argument('--focalloss', action='store_true', default=False)
    parser.add_argument('--focal_alpha', type=str, default='1.0 1.0 1.0', help='Focal Loss alpha')
    parser.add_argument('--focal_gamma', type=float, default=2, help='Focal Loss gamma')

    parser.add_argument('--multilabel_focalloss', action='store_true', default=False)

    parser.add_argument('--multilabel', action='store_true', default=False)

    parser.add_argument('--exchange_channel', action='store_true', default=False)
    parser.add_argument('--exchange_positive_channel', action='store_true', default=False)

    parser.add_argument('--train_spike_1channel_idx', type=int, default=0,
                        help='The used channel when training spike 1 channel model')

    parser.add_argument('--marginloss', action='store_true', default=False)
    parser.add_argument('--margin', type=float, default=1, help='margin in margin loss')

    parser.add_argument('--GHMCloss', action='store_true', default=False)

    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    # * Finetuning params
    parser.add_argument('--finetune', default='',
                        help='finetune from checkpoint')
    parser.add_argument('--model_key', default='model|module', type=str)
    parser.add_argument('--model_prefix', default='', type=str)
    parser.add_argument('--model_filter_name', default='gzp', type=str)
    parser.add_argument('--init_scale', default=0.001, type=float)
    parser.add_argument('--use_mean_pooling', action='store_true')
    parser.set_defaults(use_mean_pooling=True)
    parser.add_argument('--use_cls', action='store_false', dest='use_mean_pooling')
    parser.add_argument('--disable_weight_decay_on_rel_pos_bias', action='store_true', default=False)

    # Dataset parameters
    parser.add_argument('--nb_classes', default=0, type=int,
                        help='number of the classification types')

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default=None,
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing cuda | cpu | mps')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')
    parser.add_argument('--auto_resume', action='store_true')
    parser.add_argument('--no_auto_resume', action='store_false', dest='auto_resume')
    parser.set_defaults(auto_resume=True)
    parser.add_argument('--task_model', default='',
                        help='resume the fine-tuned task model from checkpoint')

    parser.add_argument('--task_model_2', default='',
                        help='resume the fine-tuned task model from checkpoint')

    parser.add_argument('--save_ckpt', action='store_true')
    parser.add_argument('--no_save_ckpt', action='store_false', dest='save_ckpt')
    parser.set_defaults(save_ckpt=True)

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true',
                        help='Perform evaluation only')
    parser.add_argument('--dist_eval', action='store_true', default=False,
                        help='Enabling distributed evaluation')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    parser.add_argument('--train_eeg_montage', default='average',
                        help='average (19) | bipolar (18) | combine (19 average + 18 bipolar) | single')
    parser.add_argument('--train_eeg_length', default=False, type=int,
                        help='15 | 10 | 5 Training samples length, use it for training IIIC (15s, 10s, 5s), but not for wave like spike. Training sample for IIIC is prepared with 15 seconds, and can be cut for assigned train_eeg_length; Spike samples should be prepared with fixed length like 1 seconds')
    parser.add_argument('--training_data_dir', default='',help='training data dir')

    parser.add_argument('--predict', action='store_true',
                        help='Perform evaluation only')
    parser.add_argument('--eval_list_file', default='', type=str,
                        help='a file to list test or evaluation dataset')
    parser.add_argument('--eval_list_column', default='',
                        help='file name column in test or evaluation dataset')
    parser.add_argument('--eval_results_dir', default='',
                        help='path where to save prediction results')
    parser.add_argument('--eeg_montage', default='average',
                        help='average (19) | bipolar (18) | combine (19 average + 18 bipolar) | single')

    parser.add_argument('--prediction_slipping_step', default=0, type=int,
                        help='slipping step in continuous prediction, if original hz>200 prediction_slipping_step better to be x/100 or xx/128')

    parser.add_argument('--prediction_slipping_step_second', default=0, type=int,
                        help='slipping step in second in continuous prediction')

    parser.add_argument('--data_format',  default='mat', type=str,
                        help='in continuous test, original data format mat | edf | pkl | EDF |MAT')

    parser.add_argument('--sampling_rate', default=0, type=int,
                        help='give sampling rate if mat/edf data files do not have this information')

    parser.add_argument('--already_format_channel_order', default='no',
                        choices=['yes','y', 'no','n'],type=str,
                        help='do not have channel info in data file, but the data if ordered by FP1, F3, C3, P3, F7, T3, T5, O1, FZ, CZ, PZ, FP2, F4, C4, P4, F8, T4, T6, O2')

    parser.add_argument('--already_average_montage', default='no',
                        choices=['yes','y', 'no','n'],type=str,
                        help='the original data already have average montage')

    parser.add_argument( '--allow_missing_channels', type=str,
                         choices=['yes','y', 'no','n'], default='no',
                         help='Allow missing channels (yes/no)')
    parser.add_argument('--polarity', default=1, type=int,
                        help='Set 1 | -1, default 1. If set -1, the signal is inverted')

    parser.add_argument('--smooth_result',  default='', type=str,
                        help='smooth the continuous results by EMA')

    parser.add_argument('--need_spikes_10s_result', default='no', type=str,
                        choices=['yes', 'y', 'no', 'n'],
                        help='not only output 1-second prediction, but also output 10-second')

    parser.add_argument('--need_spikes_1s_result', default='no', type=str,
                        choices=['yes', 'y', 'no', 'n'],
                        help='not only output prediction in point, but also output 1-second')

    parser.add_argument('--need_spike_localization_1s_result', default='no', type=str,
                        choices=['yes', 'y', 'no', 'n'],
                        help='not only output prediction in point, but also output 1-second')

    parser.add_argument('--need_vw_1s_result', default='no', type=str,
                        choices=['yes', 'y', 'no', 'n'],
                        help='not only output prediction in point, but also output 1-second')

    parser.add_argument('--spikes_10s_result_slipping_step_second', default=0, type=int, help='slipping step (in second) in continuous results based on 10s spikes segments. Could set it as 10')

    parser.add_argument('--refer_spike_model', default='', type=str,
                        help='Spike detection model path. Use spike detection model to reduce false positive')

    parser.add_argument('--refer_hm_model', default='', type=str,
                        help='Seizure hard mining model (few-shot for chewing) path.')

    parser.add_argument('--refer_spike_result_dir', default='', type=str,
                        help='Spike detection result path. Should have the same step with VW. Use spike detection model to reduce false positive')


    parser.add_argument('--model_selection_for_spike_localization', default='multi_channel',
                        choices=['single_channel', 'multi_channel', 'both'], type=str,
                        help='When performing spike localization, it can only be set to single_channel | multi_channel | both')

    parser.add_argument('--detection_location', default='no', type=str,
                        choices=['left', 'right', 'l', 'r', 'both'],
                        help=' To detect whether the periodic discharges are in the left or right hemisphere. Input right | r | left | l | both')


    parser.add_argument('--test_data_format',  default='pkl', type=str,
                        help='in test, original data format mat | pkl')

    parser.add_argument('--max_length_hour', default='no', type=str,
                        help='Only analyze the first n hours of data.')

    parser.add_argument('--rewrite_results', default='no', type=str,
                        choices=['yes', 'y', 'no', 'n'],
                        help='rewrite results')

    parser.add_argument('--leave_one_hemisphere_out', default='no', type=str,
                        choices=['no', 'right', 'left', 'middle'],
                        help='set right or left or middle hemisphere 0')

    parser.add_argument('--channel_symmetric_flip', default='no', type=str,
                        choices=['no', 'right', 'left'],
                        help='symmetrically map the right(left) hemisphere data to the left(right) hemisphere.')


    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    parser.add_argument('--enable_deepspeed', action='store_true', default=False)

    parser.add_argument('--dataset', default='TUAB', type=str,
                        help='dataset: IIIC | SPIKES | NORMAL | SLOWING | BS  | SLEEP')

    parser.add_argument('--eval_sub_dir', type=str,
                        help='dataset: subject directory')

    parser.add_argument('--multiple_eval_sub_dir', default='no', type=str,
                        choices=['yes', 'y', 'no', 'n'],
                        help='has multiple eval_sub_dir, default no')


    # hard mining
    parser.add_argument('--hardmining', default='no', type=str,
                        help='employ hard mining')

    parser.add_argument('--hardmining_data_dir', default='', help='training data dir')


    known_args, _ = parser.parse_known_args()

    if known_args.enable_deepspeed:
        try:
            print("Use deepspeed==0.4.0'")
            import deepspeed
            from deepspeed import DeepSpeedConfig
            parser = deepspeed.add_config_arguments(parser)
            ds_init = deepspeed.initialize
        except:
            print("Please 'pip install deepspeed==0.4.0'")
            exit(0)
    else:
        ds_init = None

    return parser.parse_args(), ds_init


def _forward_pass(model, eeg, target, criterion, input_chans):
    """Single forward pass returning (loss, logits)."""
    logits = model(eeg, input_chans)
    return criterion(logits, target), logits


def _deepspeed_amp_scale(model) -> float:
    """Read the current AMP loss scale from a DeepSpeed-wrapped model."""
    opt = model.optimizer
    return opt.loss_scale if hasattr(opt, "loss_scale") else opt.cur_scale


def _accuracy_from_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    is_binary: bool,
    threshold: float = 0.5,
) -> float:
    """
    Compute batch accuracy for binary, single-label, or multi-label settings.
    All tensors expected on CPU for metric computation.
    """
    if is_binary:
        # targets may be soft labels in [0,1]; binarise at threshold
        hard_targets = (targets >= threshold).int()
        return utils.get_metrics(
            torch.sigmoid(logits).numpy(),
            hard_targets.numpy(),
            ["accuracy"], is_binary,
        )["accuracy"]

    if targets.ndim == 1 or (targets.ndim == 2 and targets.size(1) == 1):
        # single-label multiclass
        return (logits.argmax(-1) == targets.squeeze()).float().mean().item()

    # element-wise accuracy for multi-label outputs
    preds = (torch.sigmoid(logits) >= threshold).float()
    return preds.eq(targets.float()).float().mean().item()


def train_one_epoch(
    args,
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    loss_scaler,
    max_norm: float = 0,
    model_ema: Optional[ModelEma] = None,
    log_writer=None,
    start_steps=None,
    lr_schedule_values=None,
    wd_schedule_values=None,
    num_training_steps_per_epoch=None,
    update_freq=None,
    ch_names=None,
    is_binary: bool = True,
    is_multilabel: bool = False,
):
    """Train for one epoch. Returns dict of averaged metrics."""
    input_chans = utils.get_input_chans(ch_names) if ch_names is not None else None
    model.train(True)

    stat_tracker = utils.MetricLogger(delimiter="  ")
    stat_tracker.add_meter("lr",     utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    stat_tracker.add_meter("min_lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))

    if loss_scaler is None:
        model.zero_grad()
        model.micro_steps = 0
    else:
        optimizer.zero_grad()

    for batch_idx, (samples, targets) in enumerate(
        stat_tracker.log_every(data_loader, 10, f"Epoch [{epoch}]")
    ):
        step = batch_idx // update_freq
        if step >= num_training_steps_per_epoch:
            continue
        global_it = start_steps + step

        # per-step LR / WD schedule update
        if (lr_schedule_values is not None or wd_schedule_values is not None) \
                and batch_idx % update_freq == 0:
            for pg in optimizer.param_groups:
                if lr_schedule_values is not None:
                    pg["lr"] = lr_schedule_values[global_it] * pg.get("lr_scale", 1.0)
                if wd_schedule_values is not None and pg.get("weight_decay", 0) > 0:
                    pg["weight_decay"] = wd_schedule_values[global_it]

        # prepare EEG tensor
        eeg = samples.float().to(device, non_blocking=True) / 100.0
        eeg = rearrange(eeg, "B N (A T) -> B N A T", T=200)

        if not is_multilabel:
            targets = targets.to(device, non_blocking=True)
            if is_binary:
                targets = targets.float().unsqueeze(-1)

        # forward
        if loss_scaler is None:
            eeg = eeg.half()
            loss, logits = _forward_pass(model, eeg, targets, criterion, input_chans)
        else:
            with torch.autocast(args.device):
                loss, logits = _forward_pass(model, eeg, targets, criterion, input_chans)

        loss_val = loss.item()
        if not math.isfinite(loss_val):
            print(f"[WARN] non-finite loss={loss_val}, stopping.", flush=True)
            sys.exit(1)

        # backward
        if loss_scaler is None:
            (loss / update_freq).backward()
            model.step()
            if (batch_idx + 1) % update_freq == 0:
                if model_ema is not None:
                    model_ema.update(model)
            grad_norm  = None
            amp_scale  = _deepspeed_amp_scale(model)
        else:
            is_second_order = getattr(optimizer, "is_second_order", False)
            grad_norm = loss_scaler(
                loss / update_freq, optimizer,
                clip_grad=max_norm,
                parameters=model.parameters(),
                create_graph=is_second_order,
                update_grad=(batch_idx + 1) % update_freq == 0,
            )
            if (batch_idx + 1) % update_freq == 0:
                optimizer.zero_grad()
                if model_ema is not None:
                    model_ema.update(model)
            amp_scale = loss_scaler.state_dict()["scale"]

        torch.cuda.synchronize()

        # accuracy
        class_acc = _accuracy_from_logits(
            logits.detach().cpu(), targets.detach().cpu(), is_binary
        )

        # metrics
        stat_tracker.update(loss=loss_val, class_acc=class_acc, loss_scale=amp_scale)
        lo_lr, hi_lr = 10.0, 0.0
        wd_val = None
        for pg in optimizer.param_groups:
            lo_lr = min(lo_lr, pg["lr"])
            hi_lr = max(hi_lr, pg["lr"])
            if pg.get("weight_decay", 0) > 0:
                wd_val = pg["weight_decay"]
        stat_tracker.update(lr=hi_lr, min_lr=lo_lr)
        if wd_val is not None:
            stat_tracker.update(weight_decay=wd_val)
        stat_tracker.update(grad_norm=grad_norm)

        if log_writer is not None:
            log_writer.update(loss=loss_val, class_acc=class_acc, head="loss")
            log_writer.update(
                loss_scale=amp_scale, lr=hi_lr, min_lr=lo_lr,
                weight_decay=wd_val, grad_norm=grad_norm,
                head="opt",
            )
            log_writer.set_step()

    stat_tracker.synchronize_between_processes()
    print("Epoch stats:", stat_tracker)
    return {k: m.global_avg for k, m in stat_tracker.meters.items()}


from contextlib import nullcontext
import torch

def amp_ctx(device):
    """
    Adaptive automatic mixed precision context:
    - CUDA: keep original behavior (torch.autocast(device_type="cuda"))
    - CPU:  enable bfloat16 mixed precision
    - MPS/other: disable AMP
    """
    dev_type = device.type if isinstance(device, torch.device) else str(device).lower()

    if dev_type == "cuda":
        return torch.autocast(device_type="cuda")

    elif dev_type == "cpu":
        return torch.autocast(device_type="cpu", dtype=torch.bfloat16)

    else:
        return nullcontext()



@torch.no_grad()
def predict(args, data_loader, model, device, header='Prediction:', ch_names=None, is_binary=True):
    input_chans = None
    if ch_names is not None:
        input_chans = utils.get_input_chans(ch_names)

    # metric_logger = utils.MetricLogger(delimiter="  ")
    # header = 'Test:'

    # switch to evaluation mode
    model.eval()
    pred = []
    for step, batch in enumerate(data_loader):
        EEG = batch[0]
        EEG = EEG.float().to(device, non_blocking=True)/ 100
        EEG = rearrange(EEG, 'B N (A T) -> B N A T', T=200)

        # compute output
        # with torch.autocast(args.device):
        #     output = model(EEG, input_chans=input_chans)
        with amp_ctx(args.device):
            output = model(EEG, input_chans=input_chans)

        if is_binary:
            output = torch.sigmoid(output).cpu()
        else:
            output = output.cpu()

        if args.device == 'cpu':
            output = output.to(torch.float32)


        pred.append(output)

    if len(pred) == 0:
        return None

    pred = torch.cat(pred, dim=0).numpy()
    return pred

@torch.no_grad()
def evaluate(args, data_loader, model, device, header='Test:', ch_names=None, metrics=['accuracy'], is_binary=False,is_multilabel=False):
    input_chans = None
    if ch_names is not None:
        input_chans = utils.get_input_chans(ch_names)
    if is_binary:
        criterion = torch.nn.BCEWithLogitsLoss()
    else:
        criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    # header = 'Test:'

    # switch to evaluation mode
    model.eval()
    pred = []
    true = []
    for step, batch in enumerate(metric_logger.log_every(data_loader, 10, header)):
        EEG = batch[0]
        target = batch[-1]
        EEG = EEG.float().to(device, non_blocking=True)
        EEG = rearrange(EEG, 'B N (A T) -> B N A T', T=200)
        target = target.to(device, non_blocking=True)
        if is_binary:
            target = target.float().unsqueeze(-1)

        # compute output
        with torch.autocast(args.device):
            output = model(EEG, input_chans=input_chans)
            loss = criterion(output, target)
        if is_binary:
            output = torch.sigmoid(output).cpu()
            # convert soft labels [0,1] to hard labels via threshold 0.5
            target = (target >= 0.5).int().detach().cpu()
        else:
            output = output.cpu()
            target = target.cpu()

        results = utils.get_metrics(output.numpy(), target.numpy(), metrics, is_binary,is_multilabel)
        pred.append(output)
        true.append(target)

        batch_size = EEG.shape[0]
        metric_logger.update(loss=loss.item())
        for key, value in results.items():
            metric_logger.meters[key].update(value, n=batch_size)
        # metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* loss {losses.global_avg:.3f}'.format(losses=metric_logger.loss))

    pred = torch.cat(pred, dim=0).numpy()
    true = torch.cat(true, dim=0).numpy()

    ret = utils.get_metrics(pred, true, metrics, is_binary, is_multilabel=is_multilabel,threshold=0.5)
    ret['loss'] = metric_logger.loss.global_avg

    return ret,pred,true



def get_models(args):
    model = create_model(
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

    return model


def get_dataset(args, evaluation=False, sub_dir=False):
    Bipolar = False
    addBipolar = False
    sample_length=False
    target_is_indices = False
    if args.dataset == 'TUAB' or args.dataset == 'TUEP':
        root=args.training_data_dir
        ch_names = ['EEG FP1', 'EEG FP2-REF', 'EEG F3-REF', 'EEG F4-REF', 'EEG C3-REF', 'EEG C4-REF', 'EEG P3-REF', 'EEG P4-REF', 'EEG O1-REF', 'EEG O2-REF', 'EEG F7-REF', \
                    'EEG F8-REF', 'EEG T3-REF', 'EEG T4-REF', 'EEG T5-REF', 'EEG T6-REF', 'EEG A1-REF', 'EEG A2-REF',
                    'EEG FZ-REF', 'EEG CZ-REF', 'EEG PZ-REF', 'EEG T1-REF', 'EEG T2-REF']
        ch_names = [name.split(' ')[-1].split('-')[0] for name in ch_names]
        args.nb_classes = 1
        metrics = ["pr_auc", "roc_auc", "accuracy", "balanced_accuracy"]

    elif args.dataset == 'TUEV':
        root=args.training_data_dir
        ch_names = ['EEG FP1-REF', 'EEG FP2-REF', 'EEG F3-REF', 'EEG F4-REF', 'EEG C3-REF', 'EEG C4-REF', 'EEG P3-REF',
                    'EEG P4-REF', 'EEG O1-REF', 'EEG O2-REF', 'EEG F7-REF', \
                    'EEG F8-REF', 'EEG T3-REF', 'EEG T4-REF', 'EEG T5-REF', 'EEG T6-REF', 'EEG A1-REF', 'EEG A2-REF',
                    'EEG FZ-REF', 'EEG CZ-REF', 'EEG PZ-REF', 'EEG T1-REF', 'EEG T2-REF']
        ch_names = [name.split(' ')[-1].split('-')[0] for name in ch_names]
        args.nb_classes = 6
        metrics = ["accuracy", "balanced_accuracy", "cohen_kappa", "f1_weighted"]

    elif args.dataset=='IIIC':
        if args.train_eeg_montage=="combine":
            ch_names = ['FP1', 'F3', 'C3', 'P3', 'F7', 'T3', 'T5', 'O1', 'FZ', 'CZ', 'PZ', 'FP2', 'F4', 'C4', 'P4', 'F8', 'T4', 'T6', 'O2', 'FP1-F7', 'F7-T3', 'T3-T5', 'T5-O1', 'FP2-F8', 'F8-T4', 'T4-T6', 'T6-O2', 'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1', 'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2', 'FZ-CZ', 'CZ-PZ']
            addBipolar = True
        elif args.train_eeg_montage == "bipolar":
            ch_names = ['FP1-F7', 'F7-T3', 'T3-T5', 'T5-O1', 'FP2-F8', 'F8-T4', 'T4-T6', 'T6-O2', 'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1', 'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2', 'FZ-CZ', 'CZ-PZ']
            Bipolar = True

        else:
            ch_names = ['FP1', 'F3', 'C3', 'P3', 'F7', 'T3', 'T5', 'O1', 'FZ', 'CZ', 'PZ', 'FP2', 'F4', 'C4', 'P4', 'F8', 'T4', 'T6', 'O2']
        root = args.training_data_dir
        args.nb_classes = 6
        metrics = ["accuracy", "balanced_accuracy", "cohen_kappa", "f1_weighted"]
        sample_length = args.train_eeg_length

    elif args.dataset == "IIIC_hm":
        ch_names = ['FP1', 'F3', 'C3', 'P3', 'F7', 'T3', 'T5', 'O1', 'FZ', 'CZ', 'PZ', 'FP2', 'F4', 'C4', 'P4', 'F8','T4', 'T6', 'O2']
        root = args.training_data_dir
        args.nb_classes = 7
        metrics = ["accuracy", "balanced_accuracy", "cohen_kappa", "f1_weighted"]

    elif args.dataset == "IIIC_chewing":
        ch_names = ['FP1', 'F3', 'C3', 'P3', 'F7', 'T3', 'T5', 'O1', 'FZ', 'CZ', 'PZ', 'FP2', 'F4', 'C4', 'P4', 'F8',
                    'T4', 'T6', 'O2']
        root = args.training_data_dir
        args.nb_classes = 7
        metrics = ["accuracy", "balanced_accuracy", "cohen_kappa", "f1_weighted"]



    elif args.dataset == "SPIKES":
        ch_names = ['FP1', 'F3', 'C3', 'P3', 'F7', 'T3', 'T5', 'O1', 'FZ', 'CZ', 'PZ', 'FP2', 'F4', 'C4', 'P4', 'F8', 'T4', 'T6', 'O2']
        args.nb_classes = 1
        metrics = ["pr_auc", "roc_auc", "accuracy", "balanced_accuracy"]
        root=args.training_data_dir

    elif args.dataset == "FOC_GEN_SPIKES":
        ch_names = ['FP1', 'F3', 'C3', 'P3', 'F7', 'T3', 'T5', 'O1', 'FZ', 'CZ', 'PZ', 'FP2', 'F4', 'C4', 'P4', 'F8','T4', 'T6', 'O2']
        root = args.training_data_dir
        args.nb_classes = 3
        metrics = ["accuracy", "balanced_accuracy", "cohen_kappa", "f1_weighted"]

    elif args.dataset == "FOC_SPIKES":
        ch_names = ['FP1', 'F3', 'C3', 'P3', 'F7', 'T3', 'T5', 'O1', 'FZ', 'CZ', 'PZ', 'FP2', 'F4', 'C4', 'P4', 'F8', 'T4', 'T6', 'O2']
        args.nb_classes = 1
        metrics = ["pr_auc", "roc_auc", "accuracy", "balanced_accuracy"]
        root=args.training_data_dir


    elif args.dataset == "GEN_SPIKES":
        ch_names = ['FP1', 'F3', 'C3', 'P3', 'F7', 'T3', 'T5', 'O1', 'FZ', 'CZ', 'PZ', 'FP2', 'F4', 'C4', 'P4', 'F8', 'T4', 'T6', 'O2']
        args.nb_classes = 1
        metrics = ["pr_auc", "roc_auc", "accuracy", "balanced_accuracy"]
        root = args.training_data_dir

    elif args.dataset== "SLOWING":
        ch_names = ['FP1', 'F3', 'C3', 'P3', 'F7', 'T3', 'T5', 'O1', 'FZ', 'CZ', 'PZ', 'FP2', 'F4', 'C4', 'P4', 'F8','T4', 'T6', 'O2']
        root = args.training_data_dir
        args.nb_classes = 3
        metrics = ["accuracy", "balanced_accuracy", "cohen_kappa", "f1_weighted"]

    elif args.dataset == "NORMAL":
        ch_names = ['FP1', 'F3', 'C3', 'P3', 'F7', 'T3', 'T5', 'O1', 'FZ', 'CZ', 'PZ', 'FP2', 'F4', 'C4', 'P4', 'F8','T4', 'T6', 'O2']
        root = args.training_data_dir
        args.nb_classes = 1
        metrics = ["pr_auc", "roc_auc", "accuracy", "balanced_accuracy"]

    elif args.dataset=='BS':
        ch_names = ['FP1', 'F3', 'C3', 'P3', 'F7', 'T3', 'T5', 'O1', 'FZ', 'CZ', 'PZ', 'FP2', 'F4', 'C4', 'P4', 'F8',
                    'T4', 'T6', 'O2']
        args.nb_classes = 1
        metrics = ["pr_auc", "roc_auc", "accuracy", "balanced_accuracy"]
        root = args.training_data_dir

    elif args.dataset == "MGBSLEEP3stages":
        ch_names = ['FP1', 'F3', 'C3', 'P3', 'F7', 'T3', 'T5', 'O1', 'FZ', 'CZ', 'PZ', 'FP2', 'F4', 'C4', 'P4', 'F8','T4', 'T6', 'O2']
        root = args.training_data_dir
        args.nb_classes = 3
        metrics = ["accuracy", "balanced_accuracy", "cohen_kappa", "f1_weighted"]

    elif args.dataset == "SLEEPPSG":
        ch_names = ['F3', 'C3', 'O1', 'F4', 'C4', 'O2']
        root = args.training_data_dir
        args.nb_classes = 5
        metrics = ["accuracy", "balanced_accuracy", "cohen_kappa", "f1_weighted"]

    elif args.dataset == "SLEEPPSG_6class":
        ch_names = ['F3', 'C3', 'O1', 'F4', 'C4', 'O2']
        root = args.training_data_dir
        args.nb_classes = 6
        metrics = ["accuracy", "balanced_accuracy", "cohen_kappa", "f1_weighted"]


    elif args.dataset == "SLEEP_AROUSAL":
        ch_names = ['F3', 'C3', 'O1', 'F4', 'C4', 'O2']
        root = args.training_data_dir
        args.nb_classes = 1
        metrics = ["pr_auc", "roc_auc", "accuracy", "balanced_accuracy"]

    elif args.dataset == "SLEEPPENN":
        ch_names = ['C3','C4']
        root = args.training_data_dir
        args.nb_classes = 5
        metrics = ["accuracy", "balanced_accuracy", "cohen_kappa", "f1_weighted"]

    elif args.dataset == "SLEEPMASS":
        ch_names = ['F3', 'C3', 'P3', 'F7', 'T3', 'T5', 'O1', 'FZ', 'CZ', 'PZ', 'F4', 'C4', 'P4', 'F8', 'T4', 'T6', 'O2']
        root = args.training_data_dir
        args.nb_classes = 5
        metrics = ["accuracy", "balanced_accuracy", "cohen_kappa", "f1_weighted"]


    elif args.dataset == "VW":
        ch_names = ['FP1', 'F3', 'C3', 'P3', 'F7', 'T3', 'T5', 'O1', 'FZ', 'CZ', 'PZ', 'FP2', 'F4', 'C4', 'P4', 'F8',
                    'T4', 'T6', 'O2']
        args.nb_classes = 1
        metrics = ["pr_auc", "roc_auc", "accuracy", "balanced_accuracy"]
        root = args.training_data_dir

    elif args.dataset== "VW_SPIKES":
        ch_names = ['FP1', 'F3', 'C3', 'P3', 'F7', 'T3', 'T5', 'O1', 'FZ', 'CZ', 'PZ', 'FP2', 'F4', 'C4', 'P4', 'F8','T4', 'T6', 'O2']
        root = args.training_data_dir
        args.nb_classes = 3
        metrics = ["accuracy", "balanced_accuracy", "cohen_kappa", "f1_weighted"]

    elif args.dataset == "BIRD":
        ch_names = ['FP1', 'F3', 'C3', 'P3', 'F7', 'T3', 'T5', 'O1', 'FZ', 'CZ', 'PZ', 'FP2', 'F4', 'C4', 'P4', 'F8',
                    'T4', 'T6', 'O2']
        args.nb_classes = 1
        metrics = ["pr_auc", "roc_auc", "accuracy", "balanced_accuracy"]
        root = args.training_data_dir


    elif args.dataset == "BIPD":
        ch_names = ['FP1', 'F3', 'C3', 'P3', 'F7', 'T3', 'T5', 'O1', 'FZ', 'CZ', 'PZ', 'FP2', 'F4', 'C4', 'P4', 'F8',
                    'T4', 'T6', 'O2']
        args.nb_classes = 1
        metrics = ["pr_auc", "roc_auc", "accuracy", "balanced_accuracy"]
        root = args.training_data_dir

    elif args.dataset == "PD":
        ch_names = ['FP1', 'F3', 'C3', 'P3', 'F7', 'T3', 'T5', 'O1', 'FZ', 'CZ', 'PZ', 'FP2', 'F4', 'C4', 'P4', 'F8',
                    'T4', 'T6', 'O2']
        args.nb_classes = 1
        metrics = ["pr_auc", "roc_auc", "accuracy", "balanced_accuracy"]
        root = args.training_data_dir

    elif args.dataset== "SPIKE_localization" and args.multilabe:
        ch_names = ['FP1', 'F3', 'C3', 'P3', 'F7', 'T3', 'T5', 'O1', 'FZ', 'CZ', 'PZ', 'FP2', 'F4', 'C4', 'P4', 'F8','T4', 'T6', 'O2']
        root = args.training_data_dir
        args.nb_classes = 19
        if evaluation:
            metrics = [
                "hamming_loss",
                "jaccard_macro",
                "precision_macro",
                "recall_macro",
                "f1_weighted",
            ]

        else:
            metrics = [
                # "hamming_loss",
                # "jaccard_macro",
                # "precision_macro",
                # "recall_macro",
                "f1_weighted",
                "roc_auc_macro",
                "pr_auc_macro",
            ]
        target_is_indices = True

    elif args.dataset == "SPIKE_1channel":
        ch_names = ['FP1', 'F3', 'C3', 'P3', 'F7', 'T3', 'T5', 'O1', 'FZ', 'CZ', 'PZ', 'FP2', 'F4', 'C4', 'P4', 'F8', 'T4', 'T6', 'O2']
        args.nb_classes = 1
        metrics = ["pr_auc", "roc_auc", "accuracy", "balanced_accuracy"]
        root=args.training_data_dir

    else:
        print ('The dataset in args format is wrong')
        return

    print(f"Data from {root}")


    if evaluation:
        result_file, evaluation_dataset = utils.prepare_classification_dataset(root,
                                                                               only_evaluate=True,
                                                                               original_format=args.test_data_format,
                                                                               sub_dir=sub_dir,
                                                                               target_is_indices=target_is_indices,
                                                                               number_of_class=args.nb_classes,
                                                                               Bipolar=Bipolar,
                                                                               addBipolar=addBipolar,
                                                                               sample_length=sample_length,
                                                                               hardmining=False,
                                                                               hardmining_data_dir='',
                                                                               spike_1channel=(
                                                                                           args.dataset == "SPIKE_1channel"),
                                                                               train_spike_1channel_idx = args.train_spike_1channel_idx,)
        if result_file==False:
            return False, False, False,False
        return result_file, evaluation_dataset, ch_names, metrics

    else:
        train_dataset, test_dataset, val_dataset = utils.prepare_classification_dataset(root,
                                                                        target_is_indices=target_is_indices,
                                                                        number_of_class=args.nb_classes,
                                                                        Bipolar=Bipolar,
                                                                        addBipolar=addBipolar,
                                                                        sample_length=sample_length,
                                                                        hardmining=args.hardmining,
                                                                        hardmining_data_dir=args.hardmining_data_dir,
                                                                        exchange_channel=args.exchange_channel,
                                                                        exchange_positive_channel=args.exchange_positive_channel,
                                                                        spike_1channel=(args.dataset == "SPIKE_1channel"),
                                                                        train_spike_1channel_idx = args.train_spike_1channel_idx,
                                                                                        )
        return train_dataset, test_dataset, val_dataset, ch_names, metrics


def main(args, ds_init):
    utils.init_distributed_mode(args)

    if ds_init is not None:
        utils.create_ds_config(args)

    # print(args)

    device = torch.device(args.device)


    # predict (continuous)------------------------------------------------------------------------------------
    if args.predict:
        if args.prediction_slipping_step== 0 and args.prediction_slipping_step_second==0:
            print('--prediction_slipping_step or --prediction_slipping_step_second is required.')
            exit(0)

        elif args.prediction_slipping_step != 0:
            used_slipping_step = args.prediction_slipping_step
            step_in_point = True
            if args.prediction_slipping_step_second !=0:
                print('Get both --prediction_slipping_step and --prediction_slipping_step_second, use the first parameter.')

        else:
           used_slipping_step = args.prediction_slipping_step_second
           step_in_point = False


        if str(args.need_spikes_10s_result).lower()=='y' or str(args.need_spikes_10s_result).lower()=='yes':
            args.need_spikes_10s_result=True
            if args.spikes_10s_result_slipping_step_second==0:
                print('Set --need_spikes_10s_result yes, but not set --spikes_10s_result_slipping_step_second or default 10, use default 10')
                args.spikes_10s_result_slipping_step_second=10
        else:
            args.need_spikes_10s_result=False


        if str(args.need_spikes_1s_result).lower()=='y' or str(args.need_spikes_1s_result).lower()=='yes':
            args.need_spikes_1s_result=True
            spikes_1s_result_dir = os.path.join(os.path.dirname(args.eval_results_dir), "pred_SPIKES_1sStep")
            os.makedirs(spikes_1s_result_dir, exist_ok=True)
        else:
            args.need_spikes_1s_result=False


        if str(args.need_spike_localization_1s_result).lower()=='y' or str(args.need_spike_localization_1s_result).lower()=='yes':
            args.need_spike_localization_1s_result=True
            spikes_loc_1s_result_dir = os.path.join(os.path.dirname(args.eval_results_dir), "pred_SPIKESLOC_1sStep")
            os.makedirs(spikes_loc_1s_result_dir, exist_ok=True)
        else:
            args.need_spike_localization_1s_result=False


        if str(args.need_vw_1s_result).lower()=='y' or str(args.need_vw_1s_result).lower()=='yes':
            args.need_vw_1s_result=True
            vw_1s_result_dir = os.path.join(os.path.dirname(args.eval_results_dir), "pred_VW_1sStep")
            os.makedirs(vw_1s_result_dir, exist_ok=True)
        else:
            args.need_vw_1s_result=False


        if str(args.already_format_channel_order).lower() == 'yes' or str(args.already_format_channel_order).lower() == 'y':
            args.already_format_channel_order = True
        else:
            args.already_format_channel_order = False

        if str(args.already_average_montage).lower()  == 'yes' or str(args.already_average_montage).lower() == 'y':
            args.already_average_montage = True
        else:
            args.already_average_montage = False

        if str(args.allow_missing_channels).lower()  == 'yes' or str(args.allow_missing_channels).lower() == 'y':
            args.allow_missing_channels = True
        else:
            args.allow_missing_channels = False

        if str(args.leave_one_hemisphere_out).lower()  == 'n' or str(args.leave_one_hemisphere_out).lower() == 'no':
            args.leave_one_hemisphere_out=False
        else:
            args.leave_one_hemisphere_out = str(args.leave_one_hemisphere_out).lower()

        if str(args.channel_symmetric_flip).lower()  == 'n' or str(args.channel_symmetric_flip).lower() == 'no':
            args.channel_symmetric_flip=False
        else:
            args.channel_symmetric_flip = str(args.channel_symmetric_flip).lower()

        if str(args.max_length_hour).lower() == 'n' or str(args.max_length_hour).lower() == 'no':
            args.max_length_hour = None
        else:
            args.max_length_hour = float(args.max_length_hour)

        if str(args.multiple_eval_sub_dir).lower() == 'n' or str(args.multiple_eval_sub_dir).lower() == 'no':
            args.multiple_eval_sub_dir = False
        else:
            args.multiple_eval_sub_dir = True


        args.spike_localization_model = 0
        if args.dataset=='SPIKE_localization':
            if args.model_selection_for_spike_localization=='multi_channel':
                args.multilabel=True
                args.spike_localization_model = 1
            elif args.model_selection_for_spike_localization=='single_channel':
                args.dataset='SPIKE_1channel'
                args.task_model=args.task_model_2
                args.spike_localization_model = 1
            elif args.model_selection_for_spike_localization=='both':
                args.spike_localization_model=2



        def use_one_model_to_predict():
            if args.dataset =='PD' and args.detection_location=='no':
                print('You should set --detection_location (left or right or both). Here set both for default')
                args.detection_location = 'both'

            if args.dataset == 'PD' and args.detection_location == 'both':
                n_repeat=2
            else:
                n_repeat=1

            if args.dataset == 'VW' and args.refer_spike_result_dir!='':
                n_models = 1

            elif args.dataset == 'VW' and args.refer_spike_model:
                n_models = 2

            elif (args.dataset == 'SPIKE_1channel' or args.dataset == 'SPIKE_localization') and args.refer_spike_result_dir!='':
                n_models = 1

            elif (args.dataset == 'SPIKE_1channel' or args.dataset == 'SPIKE_localization') and args.refer_spike_model:
                n_models = 2

            elif args.dataset == 'IIIC_hm' and args.refer_hm_model:
                n_models = 2
            else:
                n_models = 1


            args.spike_time_idx = None

            dataset_ori = args.dataset

            task_model_ori = args.task_model

            # file_exist_for_models_mode= set()

            seizure_file_list = []

            os.makedirs(args.eval_results_dir, exist_ok=True)
            # os.chmod(args.eval_results_dir, 0o777)
            try:
                os.chmod(args.eval_results_dir, 0o777)
            except OSError as e:
                print(f"[WARN] chmod not supported for {args.eval_results_dir}: {e}")

            if n_models==2:
                original_results_dir=os.path.join(args.eval_results_dir,'original_results')
                revised_results_dir=os.path.join(args.eval_results_dir,'revised_results')
                os.makedirs(original_results_dir, exist_ok=True)
                os.makedirs(revised_results_dir, exist_ok=True)

                args.eval_results_dir = original_results_dir

            else:
                original_results_dir=None
                revised_results_dir=None


            for n_model in range(n_models):
                if dataset_ori == 'VW' and n_models == 2:
                    if n_model == 1:
                        args.dataset = 'SPIKES'
                        args.task_model=args.refer_spike_model

                elif (args.dataset == 'SPIKE_1channel' or args.dataset == 'SPIKE_localization') and n_models == 2:
                    if n_model == 0:
                        args.dataset = 'SPIKES'
                        args.task_model=args.refer_spike_model
                    else:
                        args.dataset = dataset_ori
                        args.task_model=task_model_ori

                elif dataset_ori == 'IIIC_hm' and n_models == 2:
                    if n_model == 0:
                        args.dataset = 'IIIC'
                        args.task_model = args.task_model

                    elif n_model == 1:
                        args.dataset = 'IIIC_hm'
                        args.task_model = args.refer_hm_model


                args.nb_classes = utils.get_n_classes(args.dataset)

                model = get_models(args)

                patch_size = model.patch_size  # patch_size 200 (1s)
                #print("Patch size = %s" % str(patch_size))
                args.window_size = (1, args.input_size // patch_size)
                args.patch_size = patch_size

                if args.device == 'cpu':
                    # Force model to float32
                    model = model.to(torch.float32)

                else:
                    model.to(device)

                model_without_ddp = model

                #n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
                #print("Model = %s" % str(model_without_ddp))
                #print('number of params:', n_parameters)

                if args.distributed:
                    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=False)
                    model_without_ddp = model.module  # Extract the original model before encapsulation by ddp

                utils.load_from_task_model(
                    args=args, model_without_ddp=model_without_ddp)


                if dataset_ori == 'IIIC_hm' and args.dataset == 'IIIC_hm' and seizure_file_list: ########modified and False
                    file_list=seizure_file_list

                else:
                    if args.eval_list_file:
                        if args.eval_list_file.endswith("xlsx"):
                            eval_list = pd.read_excel(args.eval_list_file).astype(str)
                        elif args.eval_list_file.endswith("csv"):
                            eval_list = pd.read_csv(args.eval_list_file).astype(str)
                        else:
                            print("eval_list_file should be in xlsx or cvs format")
                            exit(0)
                        file_list=eval_list[args.eval_list_column].to_list()
                        file_list = [f"{file.split('.')[0]}.{args.data_format}" for file in file_list]

                        current_files = set(os.listdir(args.eval_sub_dir))
                        file_list = [file.split('.')[0] for file in file_list if file in current_files]

                    else:
                        # If data is in different subfolders under one root, search recursively
                        if args.multiple_eval_sub_dir:
                            print('Recursively organizing files...')
                            file_list = utils.recursive_files(root_dir=args.eval_sub_dir, file_type=args.data_format)
                            file_list = [file.split('.')[0] for file in file_list]

                        # If all data is in a single folder
                        else:
                            file_list=[file.split('.')[0] for file in os.listdir(args.eval_sub_dir) if file.endswith(args.data_format)]

                        # Sort files numerically if numeric parts are present
                        file_list = sorted(file_list, key=utils.extract_number)

                if n_model == 0:
                    print(f'\n ')
                    print(f'[*] For {args.dataset} task')
                    print(f'Input {len(file_list)} EEG files...')


                if args.refer_spike_model and args.refer_spike_result_dir=='':
                    print(f'[**] Need to wait for all files to run twice [{n_model+1}/2]')
                    describe = f'{dataset_ori} event level. Run all files twice [{n_model+1}/2]'

                elif dataset_ori == 'IIIC_hm':
                    print(f'[**] Files that potentially contain seizures need to be run twice [{n_model+1}/2]')
                    describe = f'{args.dataset} event level. Run seizure files twice [{n_model+1}/2]'

                else:
                    describe = f"{dataset_ori} event level"

                for eval_file in tqdm(file_list,desc=f"{describe}"):

                    ################## Whether overwrite former results##################
                    if args.rewrite_results in ['no' , 'n' , 'No' , 'N' , 'NO'] and n_model==0:
                        result_file_path = os.path.join(args.eval_results_dir, f'{eval_file}.csv')
                        if os.path.exists(result_file_path):
                            # if n_models>1:
                            #     file_exist_for_models_mode.add(eval_file)
                            print(f'{result_file_path} already exists, skip')
                            continue

                    ########modified
                    if args.rewrite_results in ['no', 'n', 'No', 'N', 'NO'] and n_model > 0:
                        # if eval_file in file_exist_for_models_mode:
                        #     print(f'{eval_file} already exists, skip')
                        #     continue
                        result_file_path = os.path.join(revised_results_dir, f'{eval_file}.csv')
                        if os.path.exists(result_file_path):
                            print(f'{result_file_path} already exists, skip')
                            continue

                    ################## Whether overwrite former results##################

                    spike_1channel_result_file_df = None
                    oriIIIC_result_file_df = None

                    # For spike localization, if the first-round spike model detects no spikes, skip spike 1-channel detection
                    if (args.dataset == 'SPIKE_1channel' or args.dataset == 'SPIKE_localization') and args.refer_spike_result_dir != '' and n_model == 0:
                        spike_result_path = os.path.join(args.refer_spike_result_dir, f'{eval_file}.csv')
                        if os.path.exists(spike_result_path):
                            spike_1channel_result_file_df=pd.read_csv(spike_result_path)
                            if (spike_1channel_result_file_df['pred'] <= 0.5).all():
                                n_channels = 19
                                result_colum_name = [f"class_{i}_prob" for i in range(n_channels)]
                                spike_1channel_result_file_df[result_colum_name]=0
                                spike_1channel_result_file_df['pred_labels'] = [[] for _ in range(len(spike_1channel_result_file_df))]

                                result_file_path = os.path.join(args.eval_results_dir, f'{eval_file}.csv')
                                spike_1channel_result_file_df.to_csv(result_file_path, index=False)
                                os.chmod(result_file_path, 0o777)

                                if args.need_spike_localization_1s_result:
                                    spike_1s_path = os.path.join(os.path.dirname(args.refer_spike_result_dir),
                                                                            "pred_SPIKES_1sStep", f'{eval_file}.csv')

                                    if os.path.exists(spike_1s_path):
                                        spike_1s_df=pd.read_csv(spike_1s_path)

                                        n_channels = 19
                                        result_colum_name = [f"class_{i}_prob" for i in range(n_channels)]
                                        spike_1s_df[result_colum_name] = 0
                                        spike_1s_df['pred_labels'] = [[] for _ in range(
                                            len(spike_1s_df))]

                                        result_file_path = os.path.join(spikes_loc_1s_result_dir, f'{eval_file}.csv')
                                        spike_1s_df.to_csv(result_file_path, index=False)
                                        os.chmod(result_file_path, 0o777)

                                    else:
                                        print(f'{eval_file} dose not have 1s spike results skip' )



                                continue
                        else:
                            print(f'{eval_file}\'s spike detection results not exists, skip')
                            continue


                    elif (args.dataset == 'SPIKE_1channel' or args.dataset == 'SPIKE_localization')  and args.refer_spike_model and n_model == 1:
                        result_file_path = os.path.join(args.eval_results_dir, f'{eval_file}.csv')
                        if os.path.exists(result_file_path):
                            spike_1channel_result_file_df=pd.read_csv(result_file_path)
                            if 'pred_labels' in spike_1channel_result_file_df.columns:
                                # print(f'{eval_file} is spike free, skip')
                                continue

                        else:
                            print(f'{eval_file}\'s spike detection results not exists, skip')
                            continue

                    # For IIIC_hm, the first round uses the IIIC model; the second round (n_model==1) re-classifies seizure cases as seizure vs. chewing or other

                    elif args.dataset == 'IIIC_hm' and n_model == 1:

                        IIIC_result_path = os.path.join(args.eval_results_dir, f'{eval_file}.csv')
                        if os.path.exists(IIIC_result_path):
                            oriIIIC_result_file_df = pd.read_csv(IIIC_result_path)
                            if not (oriIIIC_result_file_df['pred_class'] == 1).any():
                                continue

                            else:
                                other_cols = ['class_2_prob', 'class_3_prob', 'class_4_prob', 'class_5_prob']

                                best_other_prob = oriIIIC_result_file_df[other_cols].max(axis=1)
                                best_other_class = oriIIIC_result_file_df[other_cols].idxmax(axis=1).str.extract(r'(\d)').astype(int)[0]

                                mask = (oriIIIC_result_file_df['pred_class'] == 1) & ((oriIIIC_result_file_df['class_1_prob']-best_other_prob) < 0.1)

                                for k in [2, 3, 4, 5]:
                                    k_mask = mask & (best_other_class == k)
                                    oriIIIC_result_file_df.loc[k_mask, ['class_1_prob', f'class_{k}_prob']] = \
                                        oriIIIC_result_file_df.loc[k_mask, [f'class_{k}_prob', 'class_1_prob']].values
                                oriIIIC_result_file_df.loc[mask, 'pred_class'] = best_other_class[mask]


                        else:
                            print(f'{eval_file}\'s IIIC detection results not exists, skip')
                            continue


                    if args.multiple_eval_sub_dir:
                        # If data is in different subfolders under one root, search recursively
                        data_path=utils.find_file_path(root_dir=args.eval_sub_dir, target_filename=f'{eval_file}.{args.data_format}')
                        if data_path==None:
                            print(f"{data_path} data not found")
                            continue

                    else:
                        # If all data is in a single folder
                        data_path = os.path.join(args.eval_sub_dir, f'{eval_file}.{args.data_format}')
                        if not os.path.exists(data_path):
                            print(f"{data_path} data not found")
                            continue

                    data_transform=utils.Compose([utils.Clipping(),utils.Scaling()])


                    for n_round in range(n_repeat):
                        if args.dataset == 'PD' and n_repeat == 2:
                            if n_round == 0:
                                args.detection_location = 'left'
                            else:
                                args.detection_location = 'right'

                        if args.eeg_montage == "single":
                            df = pd.DataFrame()
                            for i in range(19):
                                montage = utils.single_channel_average_montage(i)
                                ch_names=montage.get_channel_names()

                                try:
                                    EEG_data = utils.ContinuousToSnippetDataset(
                                                        type=args.dataset,
                                                        path_signal=data_path,
                                                        given_fs=args.sampling_rate,
                                                        has_formatted_channel=args.already_format_channel_order,
                                                        has_avg=args.already_average_montage,
                                                        allow_missing_channels=args.allow_missing_channels,
                                                        montage=montage,
                                                        transform=data_transform,
                                                        step=used_slipping_step,
                                                        step_in_point=step_in_point,
                                                        polarity=int(args.polarity),
                                                        max_length_hour=args.max_length_hour)

                                except ValueError as e:
                                    print(f"DataError: {e}")
                                    continue

                                except Exception as e:
                                    print(f"Unknown error: {e}")
                                    continue

                                sampler_evaluate = torch.utils.data.SequentialSampler(EEG_data)
                                data_loader_evaluate = torch.utils.data.DataLoader(
                                    EEG_data, sampler=sampler_evaluate,
                                    batch_size=int(1.5 * args.batch_size),
                                    num_workers=args.num_workers,
                                    pin_memory=args.pin_mem,
                                    drop_last=False)
                                pred = predict(args, data_loader_evaluate, model, device, header='Prediction:',
                                               ch_names=ch_names,
                                               is_binary=(args.nb_classes == 1))
                                df[f'pred_{ch_names[0]}']= pred.flatten()

                            result_file_path = os.path.join(args.eval_results_dir, f'{eval_file}.csv')
                            df.to_csv(result_file_path, index=False)
                            # Set file permissions for all users (read, write, execute)
                            os.chmod(result_file_path, 0o777)

                            continue


                        elif args.eeg_montage=="combine":
                            montage=utils.combine_montage()
                            ch_names = montage.get_channel_names()

                        elif args.eeg_montage=="bipolar":
                            montage = utils.bipolar_montage()
                            ch_names = montage.get_channel_names()

                        elif args.eeg_montage=="average":
                            if args.dataset=='SLEEPPSG' or args.dataset=="SLEEPPSG_6class" or args.dataset=='SLEEP_AROUSAL':
                                montage = utils.sleep_common_average_montage()
                                ch_names = montage.get_channel_names()

                            elif args.dataset=='PD':
                                montage = utils.partial_average_montage(location=args.detection_location)
                                ch_names = montage.get_channel_names()

                            else:
                                montage = utils.common_average_montage()
                                ch_names=montage.get_channel_names()


                        else:
                            print(f"The input parameter eeg_montage is set incorrectly")
                            continue

                        try:
                            # ######################test time############################################
                            # print('[*]start')
                            # print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                            # ######################test time############################################

                            EEG_data = utils.ContinuousToSnippetDataset(
                                                type=args.dataset,
                                                path_signal=data_path,
                                                given_fs=args.sampling_rate,
                                                has_formatted_channel=args.already_format_channel_order,
                                                has_avg=args.already_average_montage,
                                                allow_missing_channels=args.allow_missing_channels,
                                                montage=montage,
                                                transform=data_transform,
                                                step=used_slipping_step,
                                                step_in_point=step_in_point,
                                                polarity=int(args.polarity),
                                                leave_one_hemisphere_out=args.leave_one_hemisphere_out,
                                                channel_symmetric_flip=args.channel_symmetric_flip,
                                                max_length_hour=args.max_length_hour,
                                                spike_1channel_result_file_df=spike_1channel_result_file_df,
                                                IIIC_result_file_df=oriIIIC_result_file_df
                                                    )

                        except ValueError as e:
                            print(f"DataError: {e}")
                            continue

                        except utils.SnippetsError as e:
                            # print(f"{eval_file}: {e}, all results will be classified as others.")
                            result_segment_shapes=e.result_segment_shapes
                            if args.nb_classes==1:
                                new_result_vector = np.zeros(result_segment_shapes)
                                df = pd.DataFrame({'pred': new_result_vector})
                                if args.dataset == 'SPIKE_1channel' or args.dataset == 'SPIKE_localization' :
                                    n_channels = len(ch_names)
                                    result_colum_name = [f"class_{i}_prob" for i in range(n_channels)]
                                    df[result_colum_name] = 0
                                    df['pred_labels'] = [[] for _ in range(len(df))]

                            else:
                                new_shape = (result_segment_shapes, args.nb_classes)
                                new_result_matrix = np.zeros(new_shape)
                                if not args.multilabel:
                                    new_result_matrix[..., 0] = 1  # the first class others is 1
                                df = pd.DataFrame(new_result_matrix, columns=[f'class_{i}_prob' for i in range(args.nb_classes)])
                                if  args.multilabel:
                                    df['pred_class'] = [[] for _ in range(len(df))]
                                else:
                                    df['pred_class'] = df.idxmax(axis=1)
                                    df['pred_class'] = df['pred_class'].map(lambda x: int(x.split('_')[1]))

                            file_path = os.path.join(args.eval_results_dir, f'{eval_file}.csv')
                            df.to_csv(file_path, index=False)
                            os.chmod(file_path, 0o777)
                            continue

                        except Exception as e:
                            print(f"Unknown error: {e}")
                            continue

                        sampler_evaluate = torch.utils.data.SequentialSampler(EEG_data)

                        data_loader_evaluate = torch.utils.data.DataLoader(
                            EEG_data, sampler=sampler_evaluate,
                            batch_size=int(1.5 * args.batch_size),
                            num_workers=args.num_workers,
                            pin_memory=args.pin_mem,
                            drop_last=False)

                        # ######################test time############################################
                        # print('[**]data process done')
                        # print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                        # ######################test time############################################

                        pred = predict(args, data_loader_evaluate, model, device, header='Prediction:',
                                              ch_names=ch_names,
                                              is_binary=(args.nb_classes == 1))

                        if pred is None or (hasattr(pred, '__len__') and len(pred) == 0):
                            print('[INFO] Empty prediction, skip this file.')

                            if args.dataset == 'IIIC_hm' and args.refer_hm_model and n_model == 1:
                                df = oriIIIC_result_file_df

                                prob_cols = [
                                    "hm_other_0_prob", "hm_sz_1_prob", "hm_lpd_2_prob",
                                    "hm_gpd_3_prob", "hm_lrda_4_prob", "hm_grda_5_prob", "hm_chewing_6_prob"
                                ]
                                pred_col = "hm_pred_class"

                                df[prob_cols + [pred_col]] = pd.NA

                                # pred_class==1: write one-hot prob
                                mask = df["pred_class"] == 1
                                df.loc[mask, prob_cols] = [1, 0, 0, 0, 0, 0, 0]
                                df.loc[mask, pred_col] = 0

                                # pred_class!=1: hm_pred_class = pred_class (prob columns stay none)
                                mask_other = ~mask
                                df.loc[mask_other, pred_col] = df.loc[mask_other, "pred_class"]

                                # Type
                                df[pred_col] = df[pred_col].astype("Int64")

                                file_path = os.path.join(revised_results_dir, f'{eval_file}.csv')

                                df.to_csv(file_path, index=False)
                                os.chmod(file_path, 0o777)

                            continue

                        # ######################test time############################################
                        # print('[***]model predict done')
                        # print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                        # ######################test time############################################

                        # Fill results; valid_index shape is the shape to return
                        valid_index, valid_start_end_indices, result_segment_shapes = EEG_data.get_valid_indices()

                        if args.nb_classes==1:

                            if (dataset_ori == 'SPIKE_1channel' and args.refer_spike_model and n_model == 1) or (dataset_ori == 'SPIKE_1channel' and args.refer_spike_result_dir!='' and n_model == 0):

                                original_fs = EEG_data.get_original_fs()

                                n_channels=len(ch_names)
                                result_colum_name = [f"class_{i}_prob" for i in range(n_channels)]

                                prob = pred.flatten()
                                # print(f'--new snip results len {len(prob)}')
                                # Reshape
                                prob_2d = prob.reshape(prob.shape[0] // n_channels, n_channels)
                                # print(f'--spike snip results len {len(prob_2d)}')

                                spike_1channel_result_file_df_spike_apart=spike_1channel_result_file_df[spike_1channel_result_file_df['pred'] > 0.5]

                                target_len = len(spike_1channel_result_file_df_spike_apart)
                                # print(f'--spike results len {target_len}')
                                indices = np.round(np.linspace(0, prob_2d.shape[0] - 1, target_len)).astype(int)
                                prob_2d_resized = prob_2d[indices, :]
                                # print(f'--resized spike results len {target_len}')

                                prob_df = pd.DataFrame(
                                    prob_2d_resized,
                                    columns=result_colum_name,
                                    index=spike_1channel_result_file_df_spike_apart.index
                                )

                                # First initialize these columns to 0 in the original DataFrame
                                for col in result_colum_name:
                                    spike_1channel_result_file_df[col] = 0.0
                                # Then overwrite with the actual values
                                for col in result_colum_name:
                                    spike_1channel_result_file_df.loc[prob_df.index, col] = prob_df[col]

                                pred_bin = (spike_1channel_result_file_df[result_colum_name] >= 0.5).astype(int).to_numpy()
                                if args.spike_localization_model != 2:
                                    spike_1channel_result_file_df['pred_labels'] = [
                                        np.flatnonzero(row).tolist() for row in pred_bin
                                    ]

                                df=spike_1channel_result_file_df
                                # ######################test time############################################
                                # print('[****]result process done')
                                # print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                                # ######################test time############################################


                            else:
                                prob=pred.flatten()

                                new_result_vector = np.zeros(len(valid_index), dtype=prob.dtype)
                                prob_segments = utils.split_nd_to_plus1d(arr=prob, segment_shape=result_segment_shapes)

                                for valid_start_end_index, prob_segment in zip(valid_start_end_indices, prob_segments):
                                    valid_start_index,valid_end_index=valid_start_end_index
                                    valid_result_segment_shape=valid_end_index-valid_start_index+1

                                    # Interpolate and expand result segment to the return length
                                    new_prob_segment=utils.resize_array_along_axis0(arr=prob_segment, d=1,target_length=valid_result_segment_shape)

                                    new_result_vector[valid_start_index:valid_end_index+1]=new_prob_segment

                                if args.dataset == 'SPIKES' or args.dataset == 'VW' :
                                    original_fs = EEG_data.get_original_fs()

                                    if args.need_spikes_10s_result:
                                        if (step_in_point==True and used_slipping_step/original_fs>1 or (step_in_point==False and used_slipping_step!=1)):
                                            print('could not output 10s continuous results for spikes, the step in 1s results should <= 1s')
                                        else:
                                            results_10s=utils.spikes_results_from_1s_to_10s(
                                                predictions=new_result_vector,
                                                data_fs=original_fs,
                                                original_result_step= used_slipping_step, original_step_in_point=step_in_point,new_result_step_second=args.spikes_10s_result_slipping_step_second)

                                            df_10s = pd.DataFrame({'pred': results_10s})
                                            file_path_10s = os.path.join(args.eval_results_dir, f'{eval_file}_10s.csv')
                                            df_10s.to_csv(file_path_10s, index=False)
                                            os.chmod(file_path_10s, 0o777)

                                    if args.need_spikes_1s_result:
                                        if step_in_point==True and used_slipping_step/original_fs>=1 or step_in_point==False:
                                            print('could not output 1s continuous results for spikes, the slipping step should in points and < 1s')
                                        else:
                                            results_1s=utils.spikes_results_from_p_to_1s(
                                                predictions=new_result_vector,
                                                data_fs=original_fs,
                                                original_result_step= used_slipping_step)

                                            df_1s = pd.DataFrame({'pred': results_1s})

                                            file_path_1s = os.path.join(spikes_1s_result_dir, f'{eval_file}.csv')
                                            df_1s.to_csv(file_path_1s, index=False)
                                            os.chmod(file_path_1s, 0o777)


                                    if step_in_point:
                                        smooth_method = args.smooth_result
                                        new_result_vector = utils.continuous_binary_probabilities(predictions=new_result_vector, data_fs=original_fs,result_step=args.prediction_slipping_step, smooth_method=smooth_method) # smooth_method = 'ema' or 'window_ema' or ''
                                        # Handle results for missing channels
                                        if args.allow_missing_channels == True:
                                            new_result_vector = utils.continuous_binary_probabilities2(
                                            predictions=new_result_vector,
                                            data_fs=original_fs,
                                            result_step=args.prediction_slipping_step)

                                df = pd.DataFrame({'pred': new_result_vector})

                            df = df.fillna(1)  # Some values overflowing to 1 became nan

                        else:
                            if (dataset_ori == 'SPIKE_localization' and args.refer_spike_model and n_model == 1) or (dataset_ori == 'SPIKE_localization' and args.refer_spike_result_dir!='' and n_model == 0):
                                original_fs = EEG_data.get_original_fs()

                                if args.multilabel:
                                    prob = 1 / (1 + np.exp(-pred))  # sigmoid multi-label

                                else:
                                    prob = np.exp(pred) / np.sum(np.exp(pred), axis=1, keepdims=True)
                                prob = np.array(prob)

                                n_channels = len(ch_names)
                                result_colum_name = [f"class_{i}_prob" for i in range(n_channels)]

                                spike_1channel_result_file_df_spike_apart=spike_1channel_result_file_df[spike_1channel_result_file_df['pred'] > 0.5]

                                target_len = len(spike_1channel_result_file_df_spike_apart)
                                # print(f'--spike results len {target_len}')
                                indices = np.round(np.linspace(0, prob.shape[0] - 1, target_len)).astype(int)
                                prob_resized = prob[indices, :]
                                # print(f'--resized spike results len {target_len}')


                                prob_df = pd.DataFrame(
                                    prob_resized,
                                    columns=result_colum_name,
                                    index=spike_1channel_result_file_df_spike_apart.index
                                )

                                # First initialize these columns to 0 in the original DataFrame
                                for col in result_colum_name:
                                    spike_1channel_result_file_df[col] = 0.0
                                # Then overwrite with the actual values
                                for col in result_colum_name:
                                    spike_1channel_result_file_df.loc[prob_df.index, col] = prob_df[col]

                                pred_bin = (spike_1channel_result_file_df[result_colum_name] >= 0.5).astype(int).to_numpy()
                                if args.spike_localization_model != 2:
                                    spike_1channel_result_file_df['pred_labels'] = [
                                        np.flatnonzero(row).tolist() for row in pred_bin
                                    ]

                                df=spike_1channel_result_file_df
                                # ######################test time############################################
                                # print('[****]result process done')
                                # print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                                # ######################test time############################################

                                df = df.fillna(1)  # Some values overflowing to 1 became nan

                            else:
                                if args.multilabel:
                                    prob = 1 / (1 + np.exp(-pred))   # sigmoid multi-label

                                else:
                                    prob = np.exp(pred) / np.sum(np.exp(pred), axis=1, keepdims=True)

                                prob = np.array(prob)


                                if args.dataset == 'IIIC_hm' and args.refer_hm_model and n_model == 1:

                                    df=oriIIIC_result_file_df

                                    result_colum_name = ["hm_other_0_prob", "hm_sz_1_prob", "hm_lpd_2_prob",
                                                         "hm_gpd_3_prob",
                                                         "hm_lrda_4_prob", "hm_grda_5_prob", 'hm_chewing_6_prob']

                                    mask = df['pred_class'] == 1
                                    n_rows = mask.sum()

                                    if n_rows == 0:
                                        print("[WARN] No rows with pred_class == 1, skip filling.")
                                    elif prob.shape[0] != n_rows:
                                        print(
                                            f"[ERROR] {eval_file} prob rows ({prob.shape[0]}) "
                                            f"!= pred_class==1 rows ({n_rows}), skip filling."
                                        )

                                    else:
                                        df.loc[mask, result_colum_name] = prob

                                        df.loc[mask, 'hm_pred_class'] = (
                                            df.loc[mask, result_colum_name]
                                            .idxmax(axis=1)
                                            .map(lambda x: int(x.split('_')[2]))
                                        )

                                        mask_other = df['pred_class'] != 1

                                        # df.loc[mask_other, result_colum_name] = ""
                                        df.loc[mask_other, 'hm_pred_class'] = df.loc[mask_other, 'pred_class']

                                        df['hm_pred_class'] = df['hm_pred_class'].astype('Int64')




                                else:
                                    # Due to resampling, result_count and valid_count may differ
                                    new_shape = (len(valid_index), prob.shape[1])
                                    new_result_matrix = np.zeros(new_shape, dtype=prob.dtype)

                                    if not args.multilabel:
                                        new_result_matrix[..., 0] = 1 # Fill result, default to first class (others)

                                    # Slice the results
                                    prob_segments=utils.split_nd_to_plus1d(arr=prob, segment_shape=result_segment_shapes)

                                    for valid_start_end_index, prob_segment in zip(valid_start_end_indices, prob_segments):
                                        valid_start_index,valid_end_index=valid_start_end_index
                                        valid_result_segment_shape=valid_end_index-valid_start_index+1

                                        # Interpolate and expand result segment to the return length
                                        new_prob_segment=utils.resize_array_along_axis0(arr=prob_segment, d=2, target_length=valid_result_segment_shape)

                                        new_result_matrix[valid_start_index:valid_end_index+1,:]=new_prob_segment

                                    if args.dataset == 'IIIC':
                                        original_fs = EEG_data.get_original_fs()
                                        if step_in_point:
                                            new_result_matrix=utils.continuous_multiclass_probabilities(prediction_matrix=new_result_matrix, data_fs=original_fs,result_step=args.prediction_slipping_step, use_smooth=True) # use_smooth=False



                                    if args.multilabel:
                                        df = pd.DataFrame(new_result_matrix, columns=[f'class_{i}_prob' for i in range(new_result_matrix.shape[1])])

                                        df = df.fillna(1)  # Some values overflowing to 1 became nan

                                        if args.spike_localization_model != 2:
                                            pred_bin = (new_result_matrix >= 0.5).astype(int)
                                            df['pred_labels'] = [np.flatnonzero(row).tolist() for row in pred_bin]

                                    else:
                                        df = pd.DataFrame(new_result_matrix,
                                                          columns=[f'class_{i}_prob' for i in range(pred.shape[1])])
                                        df = df.fillna(1)  # Some values overflowing to 1 became nan

                                        df['pred_class'] = df.idxmax(axis=1)
                                        df['pred_class'] = df['pred_class'].map(lambda x: int(x.split('_')[1]))



                                        if args.need_spike_localization_1s_result:
                                            spike_1s_path = os.path.join(os.path.dirname(args.refer_spike_result_dir),
                                                                         "pred_SPIKES_1sStep", f'{eval_file}.csv')

                                            if os.path.exists(spike_1s_path):
                                                spike_1s_df = pd.read_csv(spike_1s_path)

                                                n_channels = 19
                                                result_colum_name = [f"class_{i}_prob" for i in range(n_channels)]
                                                spike_1s_df[result_colum_name] = 0
                                                spike_1s_df['pred_labels'] = [[] for _ in range(
                                                    len(spike_1s_df))]

                                                result_file_path = os.path.join(spikes_loc_1s_result_dir,
                                                                                f'{eval_file}.csv')
                                                spike_1s_df.to_csv(result_file_path, index=False)
                                                os.chmod(result_file_path, 0o777)

                                        # continue



                        if n_model == 1:
                            file_path = os.path.join(revised_results_dir, f'{eval_file}.csv')

                        else:
                            file_path=os.path.join(args.eval_results_dir, f'{eval_file}.csv')

                        if args.dataset == 'PD':
                            if n_round==0:
                                # if not os.path.exists(file_path): continue
                                # l_df=pd.read_csv(file_path)
                                l_df = df.rename(columns={'pred': 'l_pred'}).copy()
                                continue
                            else:
                                r_df = df.rename(columns={'pred': 'r_pred'}).copy()
                                df = pd.concat(
                                    [l_df['l_pred'].reset_index(drop=True),
                                     r_df['r_pred'].reset_index(drop=True)],
                                    axis=1
                                )

                        elif (dataset_ori == 'VW' and args.refer_spike_model and n_model==1) or (dataset_ori == 'VW' and args.refer_spike_result_dir!='' and n_model==0):

                            if dataset_ori == 'VW' and args.refer_spike_model and n_model==1:
                                vw_file_path=os.path.join(args.eval_results_dir, f'{eval_file}.csv')
                                if not os.path.exists(vw_file_path):
                                    print(f"{vw_file_path} dose not have original result, skip" )
                                    continue
                                vw_df=pd.read_csv(vw_file_path)
                                # vw_df['vw_pred']=vw_df['pred']
                                # vw_df.rename(columns={'pred': 'vw_pred'}, inplace=True)
                            else:
                                spike_result_path=os.path.join(args.refer_spike_result_dir, f'{eval_file}.csv')
                                if not os.path.exists(spike_result_path):
                                    print(f"{eval_file} dose not have refer spike result, skip" )
                                    continue
                                vw_df=df
                                df=pd.read_csv(spike_result_path)

                            df.rename(columns={'pred': 's_pred'}, inplace=True)
                            df = pd.concat([vw_df[['pred']], df[['s_pred']]], axis=1).reset_index(drop=True)
                            df['pred'] = (df['pred'] * (1 - df['s_pred'])).round(5)

                            # Original mask: points above threshold
                            m = (df['s_pred'].values > 0.4)
                            n = len(m)
                            # Find start and end indices of each consecutive True interval
                            m_prev = np.r_[False, m[:-1]]
                            m_next = np.r_[m[1:], False]
                            starts = np.where(~m_prev & m)[0]  # Interval start
                            ends = np.where(m & ~m_next)[0]  # Interval end

                            # Build expanded mask
                            expanded = np.zeros(n, dtype=bool)
                            for s, e in zip(starts, ends):
                                L = (e - s + 1) * 4  # Original interval length * 4
                                a = max(0, s - L)  # Expand L forward
                                b = min(n - 1, e + L)  # Expand L backward
                                expanded[a:b + 1] = True

                                # Apply modification: set vw_pred = 1 - s_pred in expanded region
                                rand_val = np.random.uniform(0, 0.4)

                                # Multiply the entire expanded interval by this random value
                                df.loc[a:b, 'pred'] = (df.loc[a:b, 'pred'] * rand_val).round(5)

                            # Keep only the pred column
                            df = df[['pred']]


                        elif (args.dataset == 'SPIKE_1channel' or args.dataset == 'SPIKE_localization')  and args.refer_spike_model and n_model == 0:
                            if (df['pred'] <= 0.5).all():
                                n_channels = len(ch_names)
                                result_colum_name = [f"class_{i}_prob" for i in range(n_channels)]
                                df[result_colum_name]=0
                                df['pred_labels'] = [[] for _ in range(len(df))]

                        if dataset_ori=='IIIC_hm' and args.dataset=='IIIC' and (df['pred_class'] == 1).any():
                            seizure_file_list.append(eval_file)


                        df.to_csv(file_path, index=False)
                        os.chmod(file_path, 0o777)

                        if args.need_vw_1s_result:
                            if step_in_point == True and used_slipping_step / original_fs >= 1 or step_in_point == False:
                                print(
                                    'could not output 1s continuous results for vw, the slipping step should in points and < 1s')
                            else:
                                results_1s = utils.spikes_results_from_p_to_1s(
                                    predictions=df['pred'].tolist(),
                                    data_fs=original_fs,
                                    original_result_step=used_slipping_step)

                                df_1s = pd.DataFrame({'pred': results_1s})

                                file_path_1s = os.path.join(vw_1s_result_dir, f'{eval_file}.csv')
                                df_1s.to_csv(file_path_1s, index=False)
                                os.chmod(file_path_1s, 0o777)

                        elif args.need_spike_localization_1s_result:
                            if step_in_point == True and used_slipping_step / original_fs >= 1 or step_in_point == False:
                                print(
                                    'could not output 1s continuous results for spike localization, the slipping step should in points and < 1s')
                            else:
                                results_1s = utils.spikes_loc_results_from_p_to_1s(
                                    predictions = df.iloc[:, :-1].values.tolist(),
                                    data_fs=original_fs,
                                    original_result_step=used_slipping_step)

                                df_1s = pd.DataFrame(results_1s, columns=df.columns[:-1])

                                # df_1s columns are pred + class_0_prob ~ class_18_prob
                                prob_cols = [c for c in df_1s.columns if c.startswith("class_")]

                                # New pred_labels column
                                df_1s["pred_labels"] = df_1s[prob_cols] \
                                    .gt(0.5) \
                                    .apply(lambda row: [i for i, v in enumerate(row) if v], axis=1)

                                file_path_1s = os.path.join(spikes_loc_1s_result_dir, f'{eval_file}.csv')
                                df_1s.to_csv(file_path_1s, index=False)
                                os.chmod(file_path_1s, 0o777)

            if n_models==2:
                print(f'[**] Event level results are saved to {os.path.dirname(args.eval_results_dir)}')
            else:
                print(f'[**] Event level results are saved to {args.eval_results_dir}')

            if args.need_spikes_1s_result:
                print(f'[**] 1s Event level results are saved to {spikes_1s_result_dir}')

            elif args.need_vw_1s_result :
                print(f'[**] 1s Event level results are saved to {vw_1s_result_dir}')

            elif args.need_spike_localization_1s_result:
                print(f'[**] 1s Event level results are saved to {spikes_loc_1s_result_dir}')


        remove_dirs = None
        eval_results_dir_ori=None


        if args.spike_localization_model == 0 or args.spike_localization_model == 1:
            use_one_model_to_predict()

        elif args.spike_localization_model == 2:
            print(f"[#] Using both multi_channel model and single_channel model for SPIKE localization\n")

            refer_spike_model_ori=args.refer_spike_model
            eval_results_dir_ori = args.eval_results_dir

            print(f"[##] 1. Multi_channel model.....")
            args.multilabel=True
            args.refer_spike_model=None
            use_one_model_to_predict()

            print(f"\n[##] 2. Single_channel model.....")
            args.dataset='SPIKE_1channel'
            args.task_model=args.task_model_2
            args.refer_spike_model=refer_spike_model_ori
            args.eval_results_dir=f'{args.eval_results_dir}_2'
            use_one_model_to_predict()

            # Merge the combined results from two spike localization models
            if torch.distributed.get_rank() == 0:
                remove_dirs = [args.eval_results_dir]

                utils.merge_pairwise_csvs(dir_1=eval_results_dir_ori, dir_2=args.eval_results_dir,
                                          out_dir=eval_results_dir_ori, prefix_1="mc_", prefix_2="sc_",
                                          drop_cols=["pred_labels"],dataset_type='SPIKE_localization')

                print(f"\n[#] The combined results have been saved in {eval_results_dir_ori}")

                if remove_dirs:
                    utils.remove_dirs(remove_dirs=remove_dirs)
                    print(f"And the original results dir {remove_dirs} has been removed")



        if dist.is_initialized():
            dist.destroy_process_group()


        exit(0)
    # predict (continuous) -------------------------------------------------------------------------



    # evaluation (test) ------------------------------------------------------------------------------------
    if args.eval:
        print(args)
        all_items = os.listdir(args.eval_sub_dir)

        subdirectories = [os.path.join(args.eval_sub_dir, item) for item in all_items if
                          os.path.isdir(os.path.join(args.eval_sub_dir, item))]

        if len(subdirectories) == 0:
            subdirectories = [args.eval_sub_dir]

        model = get_models(args)

        patch_size = model.patch_size  # patch_size 200 (1s)
        print("Patch size = %s" % str(patch_size))
        args.window_size = (1, args.input_size // patch_size)
        args.patch_size = patch_size

        model.to(device)

        model_without_ddp = model
        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print("Model = %s" % str(model_without_ddp))
        print('number of params:', n_parameters)

        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=False)
        model_without_ddp = model.module  # Extract the original model before encapsulation by ddp



        utils.load_from_task_model(
            args=args, model_without_ddp=model_without_ddp)

        for eval_sub in tqdm(subdirectories):
            result_file, dataset_evaluate, ch_names, metrics = get_dataset(args, evaluation=True, sub_dir=eval_sub)

            if result_file==False:
                continue

            sampler_evaluate = torch.utils.data.SequentialSampler(dataset_evaluate)

            data_loader_evaluate = torch.utils.data.DataLoader(
                dataset_evaluate, sampler=sampler_evaluate,
                batch_size=int(1.5 * args.batch_size),
                num_workers=args.num_workers,
                pin_memory=args.pin_mem,
                drop_last=False)
            balanced_accuracy = []
            accuracy = []

            test_stats, pred, true = evaluate(args, data_loader_evaluate, model, device, header='Evaluate:',
                                              ch_names=ch_names, metrics=metrics,
                                              is_multilabel=args.multilabel,
                                              is_binary=(args.nb_classes == 1))
            accuracy.append(test_stats['accuracy'])

            if "balanced_accuracy" in test_stats:
                balanced_accuracy.append(test_stats['balanced_accuracy'])

                print(f"======Accuracy: {np.mean(accuracy)} {np.std(accuracy)}, balanced accuracy: {np.mean(balanced_accuracy)} {np.std(balanced_accuracy)}")
            else:
                print(
                    f"======Accuracy: {np.mean(accuracy)} {np.std(accuracy)}")

            try:
                # Read the CSV file
                df = pd.read_csv(result_file)

                # Add the 'true' and 'pred' columns
                if args.nb_classes == 1:
                    df['true'] = true.flatten()
                    df['pred'] = pred.flatten()

                elif args.multilabel:
                    N, C = pred.shape

                    # 1) logits -> sigmoid probabilities
                    probs = 1.0 / (1.0 + np.exp(-pred))  # (N, C)

                    # 2) Per-class probability and prediction
                    threshold = 0.5  # Or pass an array to set per-class thresholds
                    for i in range(C):
                        df[f'class_{i}_prob'] = probs[:, i]
                        # df[f'class_{i}_pred'] = (probs[:, i] >= threshold).astype(int)
                        # df[f'class_{i}_true'] = y_true_bin[:, i]

                    # 3) Row-level labels
                    # true: (N, C) one-hot 0/1 numpy array
                    df['true_labels'] = [np.flatnonzero(row).astype(int).tolist() for row in true]
                    df['pred_labels'] = [np.flatnonzero(row).tolist() for row in (probs >= threshold).astype(int)]

                else:
                    df['true'] = true.flatten()
                    probs= np.exp(pred) / np.sum(np.exp(pred), axis=1, keepdims=True)
                    for i in range(pred.shape[1]):
                        df[f'class_{i}_prob'] = probs[:, i]

                    df['pred_class'] = df[[f'class_{i}_prob' for i in range(pred.shape[1])]].idxmax(axis=1)
                    df['pred_class'] = df['pred_class'].map(lambda x: int(x.split('_')[1]))

                # Save the modified DataFrame back to the same CSV file
                df = df.fillna(1)  # Some values overflowing to 1 became nan
                df.to_csv(result_file, index=False)
                os.chmod(result_file, 0o777)
                print(f"Test results saved to {result_file}")

            except pd.errors.EmptyDataError:
                print("The file is empty or has no columns to parse.")
            except FileNotFoundError:
                print(f"The file {result_file} does not exist.")
            except Exception as e:
                print(f"An error occurred: {e}")

        if dist.is_initialized():
            dist.destroy_process_group()
        exit(0)
    # evaluation-------------------------------------------------------------------------------------


    # train-------------------------------------------------------------------------------------
    # fix the seed for reproducibility
    print(args)
    if str(args.hardmining).lower() == 'n' or str(args.hardmining).lower() == 'no':
        args.hardmining = False

    else:
        args.hardmining = True

    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)  # reproducibility of tensors generated by PyTorch
    np.random.seed(seed)  # reproducibility of NumPy related random operations
    # random.seed(seed)

    cudnn.benchmark = True  # better for convolutional operation in fixed model size

    # dataset_train, dataset_test, dataset_val: follows the standard format of torch.utils.data.Dataset.
    # ch_names: list of strings, channel names of the dataset. It should be in capital letters.
    # metrics: list of strings, the metrics you want to use. We utilize PyHealth to implement it.
    dataset_train, dataset_test, dataset_val, ch_names, metrics = get_dataset(args)

    if args.disable_eval_during_finetuning:
        dataset_val = None
        dataset_test = None

    # train and val need to be RandomSampler or DistributedSampler with shuffle=True, test and evaluation need to be SequentialSampler or DistributedSampler with shuffle=False
    if args.distributed:  # True
        num_tasks = utils.get_world_size() #the number of world_size is nnodes * nproc_per_node
        global_rank = utils.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))
        if args.dist_eval:
            if len(dataset_val) % num_tasks != 0:
                print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                      'This will slightly alter validation results as extra duplicate entries are added to achieve '
                      'equal num of samples per-process.')
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False)
            if type(dataset_test) == list:
                sampler_test = [torch.utils.data.DistributedSampler(
                    dataset, num_replicas=num_tasks, rank=global_rank, shuffle=False) for dataset in dataset_test]
            else:
                sampler_test = torch.utils.data.DistributedSampler(
                    dataset_test, num_replicas=num_tasks, rank=global_rank, shuffle=False)
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
            sampler_test = torch.utils.data.SequentialSampler(dataset_test)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = utils.TensorboardLogger(log_dir=args.log_dir)
    else:
        log_writer = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    if dataset_val is not None:
        data_loader_val = torch.utils.data.DataLoader(
            dataset_val, sampler=sampler_val,
            batch_size=int(1.5 * args.batch_size),
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False
        )
        if type(dataset_test) == list:
            data_loader_test = [torch.utils.data.DataLoader(
                dataset, sampler=sampler,
                batch_size=int(1.5 * args.batch_size),
                num_workers=args.num_workers,
                pin_memory=args.pin_mem,
                drop_last=False
            ) for dataset, sampler in zip(dataset_test, sampler_test)]
        else:
            data_loader_test = torch.utils.data.DataLoader(
                dataset_test, sampler=sampler_test,
                batch_size=int(1.5 * args.batch_size),
                num_workers=args.num_workers,
                pin_memory=args.pin_mem,
                drop_last=False
            )
    else:
        data_loader_val = None
        data_loader_test = None

    model = get_models(args)

    patch_size = model.patch_size # patch_size 200 (1s)
    print("Patch size = %s" % str(patch_size))
    args.window_size = (1, args.input_size // patch_size)
    args.patch_size = patch_size

    if args.finetune:
        if args.finetune.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.finetune, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.finetune, map_location='cpu')

        print("Load ckpt from %s" % args.finetune)
        checkpoint_model = None
        for model_key in args.model_key.split('|'):
            if model_key in checkpoint:
                checkpoint_model = checkpoint[model_key]
                print("Load state_dict by model_key = %s" % model_key)
                break
        if checkpoint_model is None:
            checkpoint_model = checkpoint
        if (checkpoint_model is not None) and (args.model_filter_name != ''):
            print("Load pretrained model main part (Student), omit lm_head and projection_head")
            all_keys = list(checkpoint_model.keys())
            new_dict = OrderedDict()
            for key in all_keys:
                if key.startswith('student.'):
                    new_dict[key[len('student.'):]] = checkpoint_model[key]
                else:
                    pass
            checkpoint_model = new_dict

        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        all_keys = list(checkpoint_model.keys())
        for key in all_keys:
            if "relative_position_index" in key:
                checkpoint_model.pop(key)

        utils.load_state_dict(model, checkpoint_model, prefix=args.model_prefix)

    model.to(device)

    model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        model_ema = ModelEma(
            model,
            decay=args.model_ema_decay,
            device='cpu' if args.model_ema_force_cpu else '',
            resume='')
        print("Using EMA with decay = %.8f" % args.model_ema_decay)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Model = %s" % str(model_without_ddp))
    print('number of params:', n_parameters)


    total_batch_size = args.batch_size * args.update_freq * utils.get_world_size()

    num_training_steps_per_epoch = len(dataset_train) // total_batch_size
    print("LR = %.8f" % args.lr)
    print("Batch size = %d" % total_batch_size)
    print("Update frequent = %d" % args.update_freq)
    print("Number of training examples = %d" % len(dataset_train))
    print("Number of training training per epoch = %d" % num_training_steps_per_epoch)


    num_layers = model_without_ddp.get_num_layers()
    if args.layer_decay < 1.0:
        assigner = LayerDecayValueAssigner(
            list(args.layer_decay ** (num_layers + 1 - i) for i in range(num_layers + 2)))
    else:
        assigner = None

    if assigner is not None:
        print("Assigned values = %s" % str(assigner.values))

    skip_weight_decay_list = model.no_weight_decay()
    if args.disable_weight_decay_on_rel_pos_bias:
        for i in range(num_layers):
            skip_weight_decay_list.add("blocks.%d.attn.relative_position_bias_table" % i)

    if args.enable_deepspeed:
        loss_scaler = None
        optimizer_params = get_parameter_groups(
            model, args.weight_decay, skip_weight_decay_list,
            assigner.get_layer_id if assigner is not None else None,
            assigner.get_scale if assigner is not None else None)
        model, optimizer, _, _ = ds_init(
            args=args, model=model, model_parameters=optimizer_params, dist_init_required=not args.distributed,
        )

        print("model.gradient_accumulation_steps() = %d" % model.gradient_accumulation_steps())
        assert model.gradient_accumulation_steps() == args.update_freq
    else:
        if args.distributed:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=False)
            model_without_ddp = model.module # Extract the original model before encapsulation by ddp

        optimizer = create_optimizer(
            args, model_without_ddp, skip_list=skip_weight_decay_list,
            get_num_layer=assigner.get_layer_id if assigner is not None else None,
            get_layer_scale=assigner.get_scale if assigner is not None else None)
        loss_scaler = NativeScaler()

    print("Use step level LR scheduler!")
    lr_schedule_values = utils.cosine_scheduler(
        args.lr, args.min_lr, args.epochs, num_training_steps_per_epoch,
        warmup_epochs=args.warmup_epochs, warmup_steps=args.warmup_steps,
    )
    if args.weight_decay_end is None:
        args.weight_decay_end = args.weight_decay

    wd_schedule_values = utils.cosine_scheduler(
        args.weight_decay, args.weight_decay_end, args.epochs, num_training_steps_per_epoch)
    print("Max WD = %.7f, Min WD = %.7f" % (max(wd_schedule_values), min(wd_schedule_values)))

    is_multilabel=False
    if args.nb_classes == 1:
        if args.focalloss:
            alpha = list(map(float, args.focal_alpha.split()))[0]
            gamma = args.focal_gamma
            criterion = BinaryFocalLoss(alpha=alpha, gamma=gamma)

        else:
            criterion = torch.nn.BCEWithLogitsLoss()

    elif args.focalloss:
        alpha = list(map(float, args.focal_alpha.split()))
        alpha = torch.tensor(alpha).to(device, non_blocking=True)
        gamma=args.focal_gamma
        if gamma==0:
            criterion=DynamicFocalLoss(alpha=alpha,)
        else:
            criterion=FocalLoss(alpha=alpha,gamma=gamma)

    elif args.multilabel_focalloss:
        criterion= MultiLabelFocalLoss()
        is_multilabel=True

    elif args.marginloss:
        margin=args.margin
        criterion = MarginLoss(margin=margin)

    elif args.GHMCloss:
        criterion=GHMC()

    elif args.smoothing > 0.:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)

    else:
        criterion = torch.nn.CrossEntropyLoss()

    print("criterion = %s" % str(criterion))

    utils.auto_load_model(
        args=args, model=model, model_without_ddp=model_without_ddp,
        optimizer=optimizer, loss_scaler=loss_scaler, model_ema=model_ema)


    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_accuracy = 0.0
    max_accuracy_test = 0.0
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        if log_writer is not None:
            log_writer.set_step(epoch * num_training_steps_per_epoch * args.update_freq)

        train_stats = train_one_epoch(args,
            model, criterion, data_loader_train, optimizer,
            device, epoch, loss_scaler, args.clip_grad, model_ema,
            log_writer=log_writer, start_steps=epoch * num_training_steps_per_epoch,
            lr_schedule_values=lr_schedule_values, wd_schedule_values=wd_schedule_values,
            num_training_steps_per_epoch=num_training_steps_per_epoch, update_freq=args.update_freq,
            ch_names=ch_names, is_binary=args.nb_classes == 1
        )

        if args.output_dir and args.save_ckpt:
            # model_without_ddp=model.module # maybe not necessary
            utils.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch, model_ema=model_ema, save_ckpt_freq=args.save_ckpt_freq)

        if data_loader_val is not None:
            val_stats,_,_ = evaluate(args, data_loader_val, model, device, header='Val:', ch_names=ch_names, metrics=metrics,
                                 is_binary=args.nb_classes == 1,is_multilabel=is_multilabel)
            print(f"Accuracy of the network on the {len(dataset_val)} val EEG: {val_stats['accuracy']:.2f}%")

            test_stats,_,_ = evaluate(args, data_loader_test, model, device, header='Test:', ch_names=ch_names, metrics=metrics, is_binary=args.nb_classes == 1,is_multilabel=is_multilabel)
            print(f"Accuracy of the network on the {len(dataset_test)} test EEG: {test_stats['accuracy']:.2f}%")

            if max_accuracy < val_stats["accuracy"]:
                max_accuracy = val_stats["accuracy"]
                if args.output_dir and args.save_ckpt:
                    utils.save_model(
                        args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                        loss_scaler=loss_scaler, epoch="best", model_ema=model_ema)
                max_accuracy_test = test_stats["accuracy"]

            print(f'Max accuracy val: {max_accuracy:.2f}%, max accuracy test: {max_accuracy_test:.2f}%')

            if log_writer is not None:
                for key, value in val_stats.items():
                    if key == 'accuracy':
                        log_writer.update(accuracy=value, head="val", step=epoch)
                    elif key == 'balanced_accuracy':
                        log_writer.update(balanced_accuracy=value, head="val", step=epoch)
                    elif key == 'f1_weighted':
                        log_writer.update(f1_weighted=value, head="val", step=epoch)
                    elif key == 'pr_auc':
                        log_writer.update(pr_auc=value, head="val", step=epoch)
                    elif key == 'roc_auc':
                        log_writer.update(roc_auc=value, head="val", step=epoch)
                    elif key == 'cohen_kappa':
                        log_writer.update(cohen_kappa=value, head="val", step=epoch)
                    elif key == 'loss':
                        log_writer.update(loss=value, head="val", step=epoch)

                for key, value in test_stats.items():
                    if key == 'accuracy':
                        log_writer.update(accuracy=value, head="test", step=epoch)
                    elif key == 'balanced_accuracy':
                        log_writer.update(balanced_accuracy=value, head="test", step=epoch)
                    elif key == 'f1_weighted':
                        log_writer.update(f1_weighted=value, head="test", step=epoch)
                    elif key == 'pr_auc':
                        log_writer.update(pr_auc=value, head="test", step=epoch)
                    elif key == 'roc_auc':
                        log_writer.update(roc_auc=value, head="test", step=epoch)
                    elif key == 'cohen_kappa':
                        log_writer.update(cohen_kappa=value, head="test", step=epoch)
                    elif key == 'loss':
                        log_writer.update(loss=value, head="test", step=epoch)

            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         **{f'val_{k}': v for k, v in val_stats.items()},
                         **{f'test_{k}': v for k, v in test_stats.items()},
                         'epoch': epoch,
                         'n_parameters': n_parameters}
        else:
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         'epoch': epoch,
                         'n_parameters': n_parameters}

        if args.output_dir and utils.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    opts, ds_init = get_args()
    if opts.output_dir:
        Path(opts.output_dir).mkdir(parents=True, exist_ok=True)
    main(opts, ds_init)
