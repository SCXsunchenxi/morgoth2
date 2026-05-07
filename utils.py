import io
import os
import math
import time
import json
import glob
from collections import defaultdict, deque
import datetime
import numpy as np
from pathlib import Path
import argparse
import torch
import mne
# from mne import create_info
mne.set_log_level("ERROR")
import re
import torch.distributed as dist
from torch import inf
from torch import optim as optim
from tensorboardX import SummaryWriter
import bisect
import pickle
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import resample
from pyhealth.metrics import binary_metrics_fn, multiclass_metrics_fn, multilabel_metrics_fn
from timm.utils import get_state_dict
from timm.optim.adafactor import Adafactor
from timm.optim.adahessian import Adahessian
from timm.optim.adamp import AdamP
from timm.optim.lookahead import Lookahead
from timm.optim.nadam import Nadam
from timm.optim.nvnovograd import NvNovoGrad
from timm.optim.radam import RAdam
from timm.optim.rmsprop_tf import RMSpropTF
from timm.optim.sgdp import SGDP
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler
import scipy.io
import mat73
import h5py
import hdf5storage
import logging
logging.basicConfig(level=logging.CRITICAL)
from scipy.signal import butter, filtfilt, iirnotch
import sys
from typing import List
from collections import Counter
import random
import shutil
from pandas.errors import EmptyDataError, ParserError
import torch.distributed as dist

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)
list_path = List[Path]


standard_1020 = [
    'FP1', 'FPZ', 'FP2',
    'AF9', 'AF7', 'AF5', 'AF3', 'AF1', 'AFZ', 'AF2', 'AF4', 'AF6', 'AF8', 'AF10', \
    'F9', 'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8', 'F10', \
    'FT9', 'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8', 'FT10', \
    'T9', 'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8', 'T10', \
    'TP9', 'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8', 'TP10', \
    'P9', 'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8', 'P10', \
    'PO9', 'PO7', 'PO5', 'PO3', 'PO1', 'POZ', 'PO2', 'PO4', 'PO6', 'PO8', 'PO10', \
    'O1', 'OZ', 'O2', \
    'O9', 'CB1', 'CB2', \
    'IZ', 'O10', 'T3', 'T5', 'T4', 'T6', 'M1', 'M2', 'A1', 'A2', \
    'CFC1', 'CFC2', 'CFC3', 'CFC4', 'CFC5', 'CFC6', 'CFC7', 'CFC8', \
    'CCP1', 'CCP2', 'CCP3', 'CCP4', 'CCP5', 'CCP6', 'CCP7', 'CCP8', \
    'T1', 'T2', 'FTT9h', 'TTP7h', 'TPP9h', 'FTT10h', 'TPP8h', 'TPP10h', \
]



standard_1020_19 = ['FP1', 'F3', 'C3', 'P3', 'F7', 'T3', 'T5', 'O1', 'FZ', 'CZ', 'PZ', 'FP2', 'F4', 'C4', 'P4', 'F8','T4', 'T6', 'O2']

eeg_channels1  = ['FP1', 'F3', 'C3', 'P3', 'F7', 'T3', 'T5', 'O1', 'FZ', 'CZ', 'PZ', 'FP2', 'F4', 'C4', 'P4', 'F8', 'T4', 'T6', 'O2']

# T3=T7  T5=P7 T4=T8  T6=P8
eeg_channels2  = ['FP1', 'F3', 'C3', 'P3', 'F7', 'T7', 'P7', 'O1', 'FZ', 'CZ', 'PZ', 'FP2', 'F4', 'C4', 'P4', 'F8','T8', 'P8', 'O2']

sleep_channels1=['F3', 'C3',  'O1',  'F4', 'C4', 'O2']

sleep_channels2=  ['F3-M2', 'C3-M2',  'O1-M2',  'F4-M1',  'C4-M1',  'O2-M1']
sleep_channels2_1=['F3-AVG','C3-AVG', 'O1-AVG', 'F4-AVG', 'C4-AVG', 'O2-AVG']
sleep_channels2_2=['F3-M1', 'C3-M1',  'O1-M1',  'F4-M2',  'C4-M2',  'O2-M2']

class DynamicFocalLoss(nn.Module):
    def __init__(self, alpha, gamma_base=2, reduction='mean'):
        """
        Dynamic Focal Loss
        Args:
            alpha (list or tensor): Class-wise weights.
            gamma_base (float): Base gamma value for dynamic adjustment.
            reduction (str): Reduction method: 'mean', 'sum', or 'none'.
        """
        super(DynamicFocalLoss, self).__init__()
        self.alpha = torch.tensor(alpha)
        self.gamma_base = gamma_base
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Args:
            inputs (tensor): Model outputs (logits), shape (N, C).
            targets (tensor): Ground truth labels, shape (N,).
        Returns:
            Tensor: Focal Loss value.
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        gamma = self.gamma_base * (1 - pt)
        focal_loss = self.alpha[targets] * (1 - pt) ** gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class FocalLoss(nn.Module):
    def __init__(self, alpha, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        alpha = torch.as_tensor(alpha)
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha[targets] * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class BinaryFocalLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=2, reduction='mean'):
        super(BinaryFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets.float(), reduction='none')
        pt = torch.exp(-ce_loss)

        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_loss = alpha_t * (1 - pt) ** self.gamma * ce_loss

        # focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class MultiLabelFocalLoss(nn.Module):
    """
    Multi-Label Focal Loss (supports targets as index lists or multi-hot vectors)
    - inputs: logits, shape [B, C]
    - targets:
        1) [B, C] 0/1 vector, or
        2) list/tuple of lists, each sample stores positive class indices (e.g. [[0,2],[1,3],[4]])
    Args:
      alpha: None | float | Tensor([C]), positive sample weight
      gamma: focal parameter, default 2.0
      reduction: 'none' | 'mean' | 'sum'
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean', eps=1e-7):
        super().__init__()
        if alpha is not None and not torch.is_tensor(alpha):
            alpha = torch.tensor(alpha, dtype=torch.float32)
        self.register_buffer('alpha', alpha if isinstance(alpha, torch.Tensor) else None, persistent=False)
        self.gamma = gamma
        self.reduction = reduction
        self.eps = eps

    def forward(self, inputs, targets):
        B, C = inputs.shape

        # --- auto-handle targets ---
        if isinstance(targets, (list, tuple)):
            # index list -> multi-hot
            y = torch.zeros(B, C, dtype=torch.float32, device=inputs.device)
            for i, idxs in enumerate(targets):
                if len(idxs) > 0:
                    y[i, torch.as_tensor(idxs, dtype=torch.long, device=inputs.device)] = 1.0
            targets = y
        else:
            # already a tensor
            targets = targets.float().to(inputs.device)

        # basic BCE (element-wise)
        ce = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')  # [B, C]

        # p_t
        p = torch.sigmoid(inputs)
        p_t = p * targets + (1 - p) * (1 - targets)
        mod = (1.0 - p_t).clamp(min=0.0) ** self.gamma

        # alpha_t
        if self.alpha is not None:
            alpha = self.alpha
            if alpha.ndim == 0:   # scalar
                alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
            else:                 # [C]
                alpha_t = alpha.view(1, -1) * targets + (1 - alpha.view(1, -1)) * (1 - targets)
            loss = alpha_t * mod * ce
        else:
            loss = mod * ce

        # reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss  # [B, C]


class MarginLoss(nn.Module):
    def __init__(self, margin=1.0, reduction='mean'):
        """
        Multi-class Margin Loss
        Args:
            margin (float): Margin parameter.
            reduction (str): Reduction method: 'mean', 'sum', or 'none'.
        """
        super(MarginLoss, self).__init__()
        self.margin = margin
        self.reduction = reduction

    def forward(self, logits, targets):
        """
        Args:
            logits (tensor): Model outputs (logits), shape (N, C).
            targets (tensor): Ground truth labels, shape (N,).
        Returns:
            Tensor: Margin Loss value.
        """
        true_logits = logits[torch.arange(len(targets)), targets].unsqueeze(1)  # Shape: (N, 1)

        margin_loss = torch.clamp(self.margin - (true_logits - logits), min=0)  # Shape: (N, C)
        margin_loss[torch.arange(len(targets)), targets] = 0  # correct class does not contribute to loss

        if self.reduction == 'mean':
            return margin_loss.mean()
        elif self.reduction == 'sum':
            return margin_loss.sum()
        else:
            return margin_loss


class GHMC(nn.Module):
    def __init__(self, bins=10, momentum=0.0):
        super(GHMC, self).__init__()
        self.bins = bins
        self.momentum = momentum
        self.edges = torch.arange(bins + 1).float() / bins
        self.edges[-1] += 1e-6
        if momentum > 0:
            self.acc_sum = torch.zeros(bins)

    def forward(self, pred, target):
        # pred: [N, C], target: [N]
        N, C = pred.size()
        device = pred.device
        edges = self.edges.to(device)
        mmt = self.momentum

        # Compute the gradient magnitude
        loss = F.cross_entropy(pred, target, reduction='none')
        g = torch.abs(pred.detach().softmax(dim=1).view(-1) - F.one_hot(target, C).float().view(-1))

        # Compute the bin index for each gradient magnitude
        g_bin = torch.bucketize(g, edges)

        # Count the number of samples in each bin
        one_hot = F.one_hot(g_bin, self.bins).float()
        if mmt > 0:
            acc_sum = self.acc_sum.to(device)
            acc_sum = acc_sum * mmt + one_hot.sum(dim=0) * (1 - mmt)
            self.acc_sum = acc_sum.cpu()
        else:
            acc_sum = one_hot.sum(dim=0)

        # Compute the density of each bin
        num_valid = (acc_sum > 0).sum()
        acc_sum = torch.clamp(acc_sum, min=1)
        density = acc_sum / num_valid

        # Compute the GHM Loss
        ghm_loss = loss / density[g_bin]
        ghm_loss = ghm_loss.sum() / N

        return ghm_loss


def bool_flag(s):
    """
    Parse boolean arguments from the command line.
    """
    FALSY_STRINGS = {"off", "false", "0"}
    TRUTHY_STRINGS = {"on", "true", "1"}
    if s.lower() in FALSY_STRINGS:
        return False
    elif s.lower() in TRUTHY_STRINGS:
        return True
    else:
        raise argparse.ArgumentTypeError("invalid value for a boolean flag")


def get_model(model):
    if isinstance(model, torch.nn.DataParallel) \
            or isinstance(model, torch.nn.parallel.DistributedDataParallel):
        return model.module
    else:
        return model


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = [
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}'
        ]
        if torch.cuda.is_available():
            log_msg.append('max mem: {memory:.0f}')
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        if len(iterable)!=0:
            average_time = total_time / len(iterable)
        else: average_time=total_time
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, average_time))


class TensorboardLogger(object):
    def __init__(self, log_dir):
        self.writer = SummaryWriter(logdir=log_dir)
        self.step = 0

    def set_step(self, step=None):
        if step is not None:
            self.step = step
        else:
            self.step += 1

    def update(self, head='scalar', step=None, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.writer.add_scalar(head + "/" + k, v, self.step if step is None else step)

    def update_image(self, head='images', step=None, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue
            self.writer.add_image(head + "/" + k, v, self.step if step is None else step)

    def flush(self):
        self.writer.flush()


def _load_checkpoint_for_ema(model_ema, checkpoint):
    """
    Workaround for ModelEma._load_checkpoint to accept an already-loaded object
    """
    mem_file = io.BytesIO()
    torch.save(checkpoint, mem_file)
    mem_file.seek(0)
    model_ema._load_checkpoint(mem_file)


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def all_reduce(tensor, op=dist.ReduceOp.SUM, async_op=False):
    world_size = get_world_size()

    if world_size == 1:
        return tensor
    dist.all_reduce(tensor, op=op, async_op=async_op)

    return tensor


def all_gather_batch(tensors):
    """
    Performs all_gather operation on the provided tensors.
    """
    # Queue the gathered tensors
    world_size = get_world_size()
    # There is no need for reduction in the single-proc case
    if world_size == 1:
        return tensors
    tensor_list = []
    output_tensor = []
    for tensor in tensors:
        tensor_all = [torch.ones_like(tensor) for _ in range(world_size)]
        dist.all_gather(
            tensor_all,
            tensor,
            async_op=False  # performance opt
        )

        tensor_list.append(tensor_all)

    for tensor_all in tensor_list:
        output_tensor.append(torch.cat(tensor_all, dim=0))
    return output_tensor


class GatherLayer(torch.autograd.Function):
    """
    Gather tensors from all workers with support for backward propagation:
    This implementation does not cut the gradients as torch.distributed.all_gather does.
    """

    @staticmethod
    def forward(ctx, x):
        output = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
        dist.all_gather(output, x)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        all_gradients = torch.stack(grads)
        dist.all_reduce(all_gradients)
        return all_gradients[dist.get_rank()]


def all_gather_batch_with_grad(tensors):
    """
    Performs all_gather operation on the provided tensors.
    Graph remains connected for backward grad computation.
    """
    # Queue the gathered tensors
    world_size = get_world_size()
    # There is no need for reduction in the single-proc case
    if world_size == 1:
        return tensors
    tensor_list = []
    output_tensor = []

    for tensor in tensors:
        tensor_all = GatherLayer.apply(tensor)
        tensor_list.append(tensor_all)

    for tensor_all in tensor_list:
        output_tensor.append(torch.cat(tensor_all, dim=0))
    return output_tensor


def _get_rank_env():
    if "RANK" in os.environ:
        return int(os.environ["RANK"])
    else:
        return int(os.environ['OMPI_COMM_WORLD_RANK'])


def _get_local_rank_env():
    if "LOCAL_RANK" in os.environ:
        return int(os.environ["LOCAL_RANK"])
    else:
        return int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])


def _get_world_size_env():
    if "WORLD_SIZE" in os.environ:
        return int(os.environ["WORLD_SIZE"])
    else:
        return int(os.environ['OMPI_COMM_WORLD_SIZE'])


def init_distributed_mode(args):
    if args.dist_on_itp:
        args.rank = _get_rank_env()
        args.world_size = _get_world_size_env()  # int(os.environ['OMPI_COMM_WORLD_SIZE'])
        args.gpu = _get_local_rank_env()
        args.dist_url = "tcp://%s:%s" % (os.environ['MASTER_ADDR'], os.environ['MASTER_PORT'])
        os.environ['LOCAL_RANK'] = str(args.gpu)
        os.environ['RANK'] = str(args.rank)
        os.environ['WORLD_SIZE'] = str(args.world_size)
        # ["RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT", "LOCAL_RANK"]
    elif 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    # print('| distributed init (rank {}): {}, gpu {}'.format(args.rank, args.dist_url, args.gpu), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)


def load_state_dict(model, state_dict, prefix='', ignore_missing="relative_position_index"):
    missing_keys = []
    unexpected_keys = []
    error_msgs = []
    # copy state_dict so _load_from_state_dict can modify it
    metadata = getattr(state_dict, '_metadata', None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    def load(module, prefix=''):
        local_metadata = {} if metadata is None else metadata.get(
            prefix[:-1], {})
        module._load_from_state_dict(
            state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + '.')

    load(model, prefix=prefix)

    warn_missing_keys = []
    ignore_missing_keys = []
    for key in missing_keys:
        keep_flag = True
        for ignore_key in ignore_missing.split('|'):
            if ignore_key in key:
                keep_flag = False
                break
        if keep_flag:
            warn_missing_keys.append(key)
        else:
            ignore_missing_keys.append(key)

    missing_keys = warn_missing_keys

    if len(missing_keys) > 0:
        print("Weights of {} not initialized from pretrained model: {}".format(
            model.__class__.__name__, missing_keys))
    if len(unexpected_keys) > 0:
        print("Weights from pretrained model not used in {}: {}".format(
            model.__class__.__name__, unexpected_keys))
    if len(ignore_missing_keys) > 0:
        print("Ignored weights of {} not initialized from pretrained model: {}".format(
            model.__class__.__name__, ignore_missing_keys))
    if len(error_msgs) > 0:
        print('\n'.join(error_msgs))


def get_grad_norm(parameters, norm_type=2):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    total_norm = 0
    for p in parameters:
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm.item() ** norm_type
    total_norm = total_norm ** (1. / norm_type)
    return total_norm


class NativeScalerWithGradNormCount:
    state_dict_key = "amp_scaler"

    def __init__(self):
        self._scaler =  torch.amp.GradScaler('cuda')

    def __call__(self, loss, optimizer, clip_grad=None, parameters=None, create_graph=False, update_grad=True,
                 layer_names=None):
        self._scaler.scale(loss).backward(create_graph=create_graph)
        if update_grad:
            if clip_grad is not None:
                assert parameters is not None
                self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            else:
                self._scaler.unscale_(optimizer)
                norm = get_grad_norm_(parameters, layer_names=layer_names)
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            norm = None
        return norm

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)


def get_grad_norm_(parameters, norm_type: float = 2.0, layer_names=None) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]

    parameters = [p for p in parameters if p.grad is not None]

    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device

    if norm_type == inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        # total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), norm_type)
        layer_norm = torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters])
        total_norm = torch.norm(layer_norm, norm_type)
        # print(layer_norm.max(dim=0))

        if layer_names is not None:
            if torch.isnan(total_norm) or torch.isinf(total_norm) or total_norm > 1.0:
                value_top, name_top = torch.topk(layer_norm, k=5)
                print(f"Top norm value: {value_top}")
                print(f"Top norm name: {[layer_names[i][7:] for i in name_top.tolist()]}")

    return total_norm


def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0,
                     start_warmup_value=0, warmup_steps=-1):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_steps > 0:
        warmup_iters = warmup_steps
    print("Set warmup steps = %d" % warmup_iters)
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = np.array(
        [final_value + 0.5 * (base_value - final_value) * (1 + math.cos(math.pi * i / (len(iters)))) for i in iters])

    schedule = np.concatenate((warmup_schedule, schedule))

    assert len(schedule) == epochs * niter_per_ep
    return schedule


def save_model(args, epoch, model, model_without_ddp, optimizer, loss_scaler, model_ema=None, optimizer_disc=None,
               save_ckpt_freq=1):
    output_dir = Path(args.output_dir)
    epoch_name = str(epoch)

    if not getattr(args, 'enable_deepspeed', False):
        print(f'[*] Saving model {epoch_name}')
        checkpoint_paths = [output_dir / 'checkpoint.pth']
        if epoch == 'best':
            checkpoint_paths = [output_dir / ('checkpoint-%s.pth' % epoch_name), ]
        elif (epoch + 1) % save_ckpt_freq == 0:
            checkpoint_paths.append(output_dir / ('checkpoint-%s.pth' % epoch_name))

        for checkpoint_path in checkpoint_paths:
            to_save = {
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                # 'scaler': loss_scaler.state_dict(),
                'args': args,
            }
            if loss_scaler is not None:
                to_save['scaler'] = loss_scaler.state_dict()

            if model_ema is not None:
                to_save['model_ema'] = get_state_dict(model_ema)

            if optimizer_disc is not None:
                to_save['optimizer_disc'] = optimizer_disc.state_dict()

            save_on_master(to_save, checkpoint_path)
    else:
        print('Using deepseek to save model...')
        client_state = {'epoch': epoch}
        if model_ema is not None:
            client_state['model_ema'] = get_state_dict(model_ema)
        model.save_checkpoint(save_dir=args.output_dir, tag="checkpoint-%s" % epoch_name, client_state=client_state)


def auto_load_model(args, model, model_without_ddp, optimizer, loss_scaler, model_ema=None, optimizer_disc=None):
    output_dir = Path(args.output_dir)

    if not getattr(args, 'enable_deepspeed', False):
        # torch.amp
        if args.auto_resume and len(args.resume) == 0:
            all_checkpoints = glob.glob(os.path.join(output_dir, 'checkpoint.pth'))
            if len(all_checkpoints) > 0:
                args.resume = os.path.join(output_dir, 'checkpoint.pth')
            else:
                all_checkpoints = glob.glob(os.path.join(output_dir, 'checkpoint-*.pth'))
                latest_ckpt = -1
                for ckpt in all_checkpoints:
                    t = ckpt.split('-')[-1].split('.')[0]
                    if t.isdigit():
                        latest_ckpt = max(int(t), latest_ckpt)
                if latest_ckpt >= 0:
                    args.resume = os.path.join(output_dir, 'checkpoint-%d.pth' % latest_ckpt)
            print("Auto resume checkpoint: %s" % args.resume)

        if args.resume:
            if args.resume.startswith('https'):
                checkpoint = torch.hub.load_state_dict_from_url(
                    args.resume, map_location='cpu', check_hash=True)
            else:
                checkpoint = torch.load(args.resume, map_location='cpu',weights_only=False)
            model_without_ddp.load_state_dict(checkpoint['model'],strict=False)  # strict: bool=True, , strict=False
            print("Resume checkpoint %s" % args.resume)
            if 'optimizer' in checkpoint and 'epoch' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer'])
                print(f"Resume checkpoint at epoch {checkpoint['epoch']}")
                args.start_epoch = 1  # checkpoint['epoch'] + 1
                if hasattr(args, 'model_ema') and args.model_ema:
                    _load_checkpoint_for_ema(model_ema, checkpoint['model_ema'])
                if 'scaler' in checkpoint:
                    loss_scaler.load_state_dict(checkpoint['scaler'])
                print("With optim & sched!")
            if 'optimizer_disc' in checkpoint:
                optimizer_disc.load_state_dict(checkpoint['optimizer_disc'])
    else:
        # deepspeed, only support '--auto_resume'.
        if args.auto_resume:
            all_checkpoints = glob.glob(os.path.join(output_dir, 'checkpoint-*'))
            latest_ckpt = -1
            for ckpt in all_checkpoints:
                t = ckpt.split('-')[-1].split('.')[0]
                if t.isdigit():
                    latest_ckpt = max(int(t), latest_ckpt)
            if latest_ckpt >= 0:
                args.resume = os.path.join(output_dir, 'checkpoint-%d' % latest_ckpt)
                print("Auto resume checkpoint: %d" % latest_ckpt)
                _, client_states = model.load_checkpoint(args.output_dir, tag='checkpoint-%d' % latest_ckpt)
                args.start_epoch = client_states['epoch'] + 1
                if model_ema is not None:
                    if args.model_ema:
                        _load_checkpoint_for_ema(model_ema, client_states['model_ema'])


def load_from_task_model(args, model_without_ddp,):
    checkpoint = torch.load(args.task_model, map_location='cpu',weights_only=False)
    model_without_ddp.load_state_dict(checkpoint['model'],strict=False)  # strict: bool=True, , strict=False
    #print("Resume checkpoint %s" % args.resume)


def create_ds_config(args):
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    with open(os.path.join(args.output_dir, "latest"), mode="w") as f:
        pass

    args.deepspeed_config = os.path.join(args.output_dir, "deepspeed_config.json")
    with open(args.deepspeed_config, mode="w") as writer:
        ds_config = {
            "train_batch_size": args.batch_size * args.update_freq * get_world_size(),
            "train_micro_batch_size_per_gpu": args.batch_size,
            "steps_per_print": 1000,
            "optimizer": {
                "type": "Adam",
                "adam_w_mode": True,
                "params": {
                    "lr": args.lr,
                    "weight_decay": args.weight_decay,
                    "bias_correction": True,
                    "betas": [
                        0.9,
                        0.999
                    ],
                    "eps": 1e-8
                }
            },
            "fp16": {
                "enabled": True,
                "loss_scale": 0,
                "initial_scale_power": 7,
                "loss_scale_window": 128
            }
        }

        writer.write(json.dumps(ds_config, indent=2))


def build_pretraining_dataset(datasets: list, time_window: list, stride_size=200, start_percentage=0, end_percentage=1):
    shock_dataset_list = []
    ch_names_list = []
    for dataset_list, window_size in zip(datasets, time_window):
        dataset = ShockDataset([Path(file_path) for file_path in dataset_list], window_size * 200, stride_size, start_percentage, end_percentage)
        shock_dataset_list.append(dataset)
        ch_names_list.append(dataset.get_ch_names())
    return shock_dataset_list, ch_names_list

def get_input_chans(ch_names):
    input_chans = [0]  # for cls token
    for ch_name in ch_names:
        ch_upper = ch_name.upper()
        if ch_upper not in standard_1020:
            raise ValueError(f"Channel '{ch_name}' not found in standard_1020 montage")
        input_chans.append(standard_1020.index(ch_upper) + 1)
    return input_chans



def get_metrics(output, target, metrics, is_binary, is_multilabel=False, threshold=0.5):

    def exact_match_fn(y_true, y_prob=None, y_pred=None, threshold=0.5):
        """subset accuracy / exact match"""
        if y_pred is None:
            if y_prob is None:
                raise ValueError("Need y_prob or y_pred")
            y_pred = (y_prob >= threshold).astype(np.int32)
        y_true = np.rint(np.clip(y_true, 0, 1)).astype(np.int32)  # normalize to {0,1}
        return (y_pred == y_true).all(axis=1).mean().item()

    def _sigmoid(x):
        # numerically stable sigmoid
        x = np.clip(x, -30, 30)
        return 1.0 / (1.0 + np.exp(-x))

    if is_binary:
        if 'roc_auc' not in metrics or sum(target) * (
                len(target) - sum(target)) != 0:  # to prevent all 0 or all 1 and raise the AUROC error
            results = binary_metrics_fn(
                target,
                output,
                metrics=metrics,
                threshold=threshold,
            )
        else:
            results = {
                "accuracy": 0.0,
                "balanced_accuracy": 0.0,
                "pr_auc": 0.0,
                "roc_auc": 0.0,
            }
    elif is_multilabel:
        # probabilities
        y_prob = _sigmoid(output).astype(np.float32)  # [N, C]

        # discrete predictions
        y_pred = (y_prob >= threshold).astype(np.int32)  # [N, C] in {0,1}

        # normalize y_true to {0,1} int32
        y_true = np.asarray(target)
        y_true = np.rint(np.clip(y_true, 0, 1)).astype(np.int32)

        # filter to metrics supported by pyhealth
        supported = {
            "hamming_loss",
            "jaccard_macro", "jaccard_micro",
            "precision_micro", "recall_micro", "f1_micro",
            "precision_macro", "recall_macro", "f1_macro",
            "precision_weighted", "recall_weighted", "f1_weighted",
            "roc_auc_macro", "roc_auc_micro","roc_auc_weighted",
            "pr_auc_macro", "pr_auc_micro", "pr_auc_weighted"
        }
        use_metrics = list(set(metrics) & supported)
        # print(use_metrics)

        results = {}
        if use_metrics:
            results = multilabel_metrics_fn(
                y_true=y_true,
                y_prob=y_prob,  # pass probability if available; will use threshold to generate y_pred
                metrics=use_metrics,
                threshold=0.5  # threshold to binarize probabilities
            )
        results["accuracy"] = exact_match_fn(y_true, y_pred=y_pred, threshold=0.5)

        if "pr_auc_macro" in results:
            results["pr_auc"] = results["pr_auc_macro"]

        if "roc_auc_macro" in results:
            results["roc_auc"] = results["roc_auc_macro"]


    else:
        results = multiclass_metrics_fn(
            target, output, metrics=metrics
        )
    return results


try:
    from apex.optimizers import FusedNovoGrad, FusedAdam, FusedLAMB, FusedSGD

    has_apex = True
except ImportError:
    has_apex = False


def get_num_layer_for_vit(var_name, num_max_layer):
    if var_name in ("cls_token", "mask_token", "pos_embed"):
        return 0
    elif var_name.startswith("patch_embed"):
        return 0
    elif var_name.startswith("rel_pos_bias"):
        return num_max_layer - 1
    elif var_name.startswith("blocks"):
        layer_id = int(var_name.split('.')[1])
        return layer_id + 1
    else:
        return num_max_layer - 1


class LayerDecayValueAssigner(object):
    def __init__(self, values):
        self.values = values

    def get_scale(self, layer_id):
        return self.values[layer_id]

    def get_layer_id(self, var_name):
        return get_num_layer_for_vit(var_name, len(self.values))


def get_parameter_groups(model, weight_decay=1e-5, skip_list=(), get_num_layer=None, get_layer_scale=None, **kwargs):
    parameter_group_names = {}
    parameter_group_vars = {}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(kwargs.get('filter_name', [])) > 0:
            flag = False
            for filter_n in kwargs.get('filter_name', []):
                if filter_n in name:
                    print(f"filter {name} because of the pattern {filter_n}")
                    flag = True
            if flag:
                continue
        if param.ndim <= 1 or name.endswith(".bias") or name in skip_list:  # param.ndim <= 1 len(param.shape) == 1
            group_name = "no_decay"
            this_weight_decay = 0.
        else:
            group_name = "decay"
            this_weight_decay = weight_decay
        if get_num_layer is not None:
            layer_id = get_num_layer(name)
            group_name = "layer_%d_%s" % (layer_id, group_name)
        else:
            layer_id = None

        if group_name not in parameter_group_names:
            if get_layer_scale is not None:
                scale = get_layer_scale(layer_id)
            else:
                scale = 1.

            parameter_group_names[group_name] = {
                "weight_decay": this_weight_decay,
                "params": [],
                "lr_scale": scale
            }
            parameter_group_vars[group_name] = {
                "weight_decay": this_weight_decay,
                "params": [],
                "lr_scale": scale
            }

        parameter_group_vars[group_name]["params"].append(param)
        parameter_group_names[group_name]["params"].append(name)
    print("Param groups = %s" % json.dumps(parameter_group_names, indent=2))
    return list(parameter_group_vars.values())


def create_optimizer(args, model, get_num_layer=None, get_layer_scale=None, filter_bias_and_bn=True, skip_list=None,
                     **kwargs):
    opt_lower = args.opt.lower()
    weight_decay = args.weight_decay
    if weight_decay and filter_bias_and_bn:
        skip = {}
        if skip_list is not None:
            skip = skip_list
        elif hasattr(model, 'no_weight_decay'):
            skip = model.no_weight_decay()
        print(f"Skip weight decay name marked in model: {skip}")
        parameters = get_parameter_groups(model, weight_decay, skip, get_num_layer, get_layer_scale, **kwargs)
        weight_decay = 0.
    else:
        parameters = model.parameters()

    if 'fused' in opt_lower:
        assert has_apex and torch.cuda.is_available(), 'APEX and CUDA required for fused optimizers'

    opt_args = dict(lr=args.lr, weight_decay=weight_decay)
    if hasattr(args, 'opt_eps') and args.opt_eps is not None:
        opt_args['eps'] = args.opt_eps
    if hasattr(args, 'opt_betas') and args.opt_betas is not None:
        opt_args['betas'] = args.opt_betas

    print('Optimizer config:', opt_args)
    opt_split = opt_lower.split('_')
    opt_lower = opt_split[-1]
    if opt_lower == 'sgd' or opt_lower == 'nesterov':
        opt_args.pop('eps', None)
        optimizer = optim.SGD(parameters, momentum=args.momentum, nesterov=True, **opt_args)
    elif opt_lower == 'momentum':
        opt_args.pop('eps', None)
        optimizer = optim.SGD(parameters, momentum=args.momentum, nesterov=False, **opt_args)
    elif opt_lower == 'adam':
        optimizer = optim.Adam(parameters, **opt_args)
        optimizer = optim.Adam(parameters, **opt_args)
    elif opt_lower == 'adamw':
        optimizer = optim.AdamW(parameters, **opt_args)
    elif opt_lower == 'nadam':
        optimizer = Nadam(parameters, **opt_args)
    elif opt_lower == 'radam':
        optimizer = RAdam(parameters, **opt_args)
    elif opt_lower == 'adamp':
        optimizer = AdamP(parameters, wd_ratio=0.01, nesterov=True, **opt_args)
    elif opt_lower == 'sgdp':
        optimizer = SGDP(parameters, momentum=args.momentum, nesterov=True, **opt_args)
    elif opt_lower == 'adadelta':
        optimizer = optim.Adadelta(parameters, **opt_args)
    elif opt_lower == 'adafactor':
        if not args.lr:
            opt_args['lr'] = None
        optimizer = Adafactor(parameters, **opt_args)
    elif opt_lower == 'adahessian':
        optimizer = Adahessian(parameters, **opt_args)
    elif opt_lower == 'rmsprop':
        optimizer = optim.RMSprop(parameters, alpha=0.9, momentum=args.momentum, **opt_args)
    elif opt_lower == 'rmsproptf':
        optimizer = RMSpropTF(parameters, alpha=0.9, momentum=args.momentum, **opt_args)
    elif opt_lower == 'nvnovograd':
        optimizer = NvNovoGrad(parameters, **opt_args)
    elif opt_lower == 'fusedsgd':
        opt_args.pop('eps', None)
        optimizer = FusedSGD(parameters, momentum=args.momentum, nesterov=True, **opt_args)
    elif opt_lower == 'fusedmomentum':
        opt_args.pop('eps', None)
        optimizer = FusedSGD(parameters, momentum=args.momentum, nesterov=False, **opt_args)
    elif opt_lower == 'fusedadam':
        optimizer = FusedAdam(parameters, adam_w_mode=False, **opt_args)
    elif opt_lower == 'fusedadamw':
        optimizer = FusedAdam(parameters, adam_w_mode=True, **opt_args)
    elif opt_lower == 'fusedlamb':
        optimizer = FusedLAMB(parameters, **opt_args)
    elif opt_lower == 'fusednovograd':
        opt_args.setdefault('betas', (0.95, 0.98))
        optimizer = FusedNovoGrad(parameters, **opt_args)
    else:
        assert False and "Invalid optimizer"
        raise ValueError

    if len(opt_split) > 1:
        if opt_split[0] == 'lookahead':
            optimizer = Lookahead(optimizer)

    return optimizer



def spikes_results_from_p_to_1s(
    predictions,
    data_fs: float,
    original_result_step: int,
    mode: str = "center",              # default: center mode
    start_offset_samples: int = 0      # sample offset of the first window start relative to 0s; use this to calibrate if first window does not start at 0
):
    ''''
    From sliding window predictions, select one representative value per second (no aggregation).

    Args:
    - predictions: 1D array or list (prediction score/probability per window)
    - data_fs: sampling rate (Hz)
    - original_result_step: sliding step (samples)
    - mode:
        'center' -> select the prediction closest to the midpoint of that second (s+0.5s) (recommended; constant at second 0 when win_len=fs)
        'last'   -> select the last prediction within that second
        'max'    -> select the prediction with the highest value within that second
    - win_len: window length (samples). If provided, timestamps are based on window center; otherwise based on window start
    - start_offset_samples: sample offset of the first window start (default 0). Set this value if the first window does not start at 0s.
    '''


    preds = np.asarray(predictions)
    n_preds = len(preds)

    win_len = data_fs

    # === compute timestamp (seconds) for each prediction ===
    # start time = k*step + start_offset_samples
    starts = np.arange(n_preds) * original_result_step + start_offset_samples
    times = (starts + win_len / 2.0) / data_fs

    # === assign each prediction to second s by floor(time) ===
    sec_idx = np.floor(times).astype(int)
    n_secs = sec_idx.max() + 1

    out = []

    for s in range(n_secs):
        mask = (sec_idx == s)
        if not np.any(mask):
            out.append(np.nan)
            continue

        idxs = np.where(mask)[0]

        if mode == "center":
            # select the prediction closest to the midpoint s+0.5 of that second
            target = s + 0.5
            chosen = preds[idxs[np.argmin(np.abs(times[idxs] - target))]]
        elif mode == "last":
            chosen = preds[idxs[-1]]
        elif mode == "max":
            chosen = preds[idxs[np.argmax(preds[idxs])]]
        else:
            raise ValueError(f"Unsupported mode: {mode}")

        out.append(chosen)

    return out

def spikes_loc_results_from_p_to_1s(
    predictions,
    data_fs: float,
    original_result_step: int,
    start_offset_samples: int = 0
):
    """
    Use the first N-1 columns of predictions (not the last column), performing center sampling per second.
    """
    arr = np.asarray(predictions, dtype=object)

    # pred + class_0_prob ~ class_18_prob
    preds = arr.astype(float)     # (n_preds, n_classes)
    n_preds, n_classes = preds.shape

    win_len = data_fs

    # timestamps
    starts = np.arange(n_preds) * original_result_step + start_offset_samples
    times = (starts + win_len / 2.0) / data_fs

    sec_idx = np.floor(times).astype(int)
    n_secs = sec_idx.max() + 1

    out = []

    for s in range(n_secs):
        idxs = np.where(sec_idx == s)[0]
        if len(idxs) == 0:
            out.append(np.full(n_classes, np.nan))
            continue

        # center
        target = s + 0.5
        chosen_idx = idxs[np.argmin(np.abs(times[idxs] - target))]
        out.append(preds[chosen_idx])

    return np.vstack(out)

def spikes_results_from_1s_to_10s(predictions,data_fs, original_result_step,original_step_in_point,new_result_step_second):
    if original_step_in_point:
        new_result_step = new_result_step_second*data_fs
        window_size = int(10 * data_fs / original_result_step)
    else:
        new_result_step=new_result_step_second
        window_size = int(10 / original_result_step)

    results_10s=[]
    for i in range(0,len(predictions)-1,new_result_step//original_result_step):

        results_1s=predictions[i:i+window_size]

        filtered_values = results_1s[results_1s >= 0.5]

        if len(filtered_values) > data_fs/4/original_result_step: # 0.25s
            avg_value = filtered_values.mean()  # compute mean of qualifying values
        else:
            avg_value = results_1s.mean()

        results_10s.append(avg_value)

    return results_10s


def exponential_moving_average(data, alpha=0.2):
    smoothed = np.zeros_like(data)
    smoothed[0] = data[0]
    for i in range(1, len(data)):
        smoothed[i] = alpha * data[i] + (1 - alpha) * smoothed[i - 1]
    return smoothed


def continuous_binary_probabilities(predictions, data_fs, result_step,smooth_method = 'ema'):
    if result_step > int(data_fs / 8):
        return predictions

    if len(predictions)*result_step < 2 * data_fs:
        return predictions

    window_size = int(data_fs/5/result_step) # window 0.2 second, 0.5 second in last version
    step = window_size

    if window_size>=len(predictions):
        return predictions

    smoothed_data = np.copy(predictions)

    if smooth_method == 'ema':
        smoothed_data=exponential_moving_average(smoothed_data,alpha=0.2)

    elif smooth_method == 'window_ema':

        for start in range(0, len(predictions)-window_size, step):
            end = min(len(predictions), start + window_size + 1)
            window = np.array(predictions[start:end])

            if np.sum(window < 0.5) > window_size: #* 3 / 4
                window = np.zeros_like(window)
            else:
                window = exponential_moving_average(window, alpha=0.1)
            smoothed_data[start:end] = window
    else:
        return smoothed_data

    return smoothed_data
#
# def continuous_binary_probabilities2(predictions, data_fs, result_step):
#     i = 0
#     n=len(predictions)
#     continuous_size=int(data_fs/result_step/4) #1/4 second
#     while i < n:
#         if predictions[i] < 0.5:
#             predictions[i]=0
#             i += 1
#             continue
#         # find intervals where values are continuously >= 0.5
#         start = i
#         while i < n and predictions[i] >= 0.5:
#             i += 1
#         end = i
#
#         if end - start <=continuous_size:
#             for j in range(start, end):
#                 predictions[j] = max(predictions[j] - 0.5, 0)
#     return predictions

def continuous_binary_probabilities2(predictions, data_fs, result_step):

    n = len(predictions)
    continuous_size = int(data_fs / result_step / 4)  # 1/4 second

    # initialize flag array
    keep_indices = [False] * n  # mark indices to keep

    # first pass: mark indices to keep
    i = 0
    while i < n:
        if predictions[i] < 0.5:
            i += 1
            continue
        start = i
        while i < n and predictions[i] >= 0.5:
            i += 1
        end = i
        # keep 0.5s before and after
        if end - start > continuous_size:
            keep_start = max(start - continuous_size*2, 0)
            keep_end = min(end + continuous_size*2, n)

            for j in range(keep_start, keep_end):
                keep_indices[j] = True

    # second pass: process predictions based on flag results
    for i in range(n):
        if not keep_indices[i]:
            predictions[i] = max(predictions[i] - 0.5, 0)  # subtract 0.5, ensure not less than 0

    return predictions

def continuous_multiclass_probabilities(prediction_matrix, data_fs,result_step, use_smooth=True):
    if not use_smooth:
        return prediction_matrix

    if result_step > int(data_fs * 10) or prediction_matrix.shape[0] <3:
        return prediction_matrix


    predictions = np.argmax(prediction_matrix, axis=1)  # get predicted class for each row

    for i in range(1, len(predictions) - 1):
        if predictions[i] != predictions[i - 1] and predictions[i-1] == predictions[i + 1]:
            prediction_matrix[i] = (prediction_matrix[i - 1] + prediction_matrix[i + 1]) / 2

    return prediction_matrix


def prepare_classification_dataset(root,original_format='pkl', target_is_indices=False,number_of_class=0,Bipolar=False, addBipolar=False, only_evaluate=False, sub_dir=False, sample_length=False, hardmining=False, hardmining_data_dir='',exchange_channel=False,exchange_positive_channel=False,spike_1channel=False,train_spike_1channel_idx=0):
    if only_evaluate:
        print(f"[*] For {sub_dir}.......")
        result_file = os.path.join(sub_dir, 'pred.csv')
        evaluation_files = sorted([f for f in os.listdir(sub_dir) if f.endswith(original_format)])

        if len(evaluation_files)==0:
            print(f'{sub_dir} dose not have {original_format} data')
            return False,False

        df = pd.DataFrame({
            'data': evaluation_files
        })
        df.to_csv(result_file, index=False)

        if 'TUEV' in result_file:
            evaluation_dataset=TUEVLoader(sub_dir, evaluation_files)

        else:
            evaluation_dataset = MGBClassLoader(sub_dir,
                                                evaluation_files,
                                                original_format=original_format,
                                                target_is_indices=target_is_indices,
                                                number_of_class=number_of_class,
                                                Bipolar=Bipolar,
                                                addBipolar=addBipolar,
                                                sample_length=sample_length,
                                                spike_1channel=spike_1channel,
                                                train_spike_1channel_idx=train_spike_1channel_idx
                                                )
        print(f'[*] Evaluation sizes: {len(evaluation_dataset)}')
        return result_file, evaluation_dataset

    else:
        # hardmining is actually unused here, but too lazy to remove it
        if not hardmining:
            train_files_path = glob.glob(os.path.join(root, "train", "*.pkl"))
        else:
            train_files_path = glob.glob(os.path.join(hardmining_data_dir, "*.pkl"))

        val_files_path = glob.glob(os.path.join(root, "val", "*.pkl"))
        test_files_path = glob.glob(os.path.join(root, "test", "*.pkl"))

        train_files = [os.path.basename(file) for file in train_files_path]
        val_files = [os.path.basename(file) for file in val_files_path]
        test_files = [os.path.basename(file) for file in test_files_path]

        # prepare training and test data loader
        if not hardmining:
            if exchange_channel:
                print("[*] Channels in training dataset will be exchanged randomly")
            elif exchange_positive_channel:
                print("[*] Positive channels in training dataset will be exchanged")
            train_dataset = MGBClassLoader(os.path.join(root, "train"),
                                           train_files,
                                           target_is_indices=target_is_indices,
                                           number_of_class=number_of_class,
                                           Bipolar=Bipolar,
                                           addBipolar=addBipolar,
                                           sample_length=sample_length,
                                           exchange_channel=exchange_channel,
                                           exchange_positive_channel=exchange_positive_channel,
                                           spike_1channel=spike_1channel,
                                           train_spike_1channel_idx=train_spike_1channel_idx
                                           )
        else:
            if exchange_channel:
                print("[*] Channels in training dataset will be exchanged randomly")
            elif exchange_positive_channel:
                print("[*] Positive channels in training dataset will be exchanged")
            train_dataset = MGBClassLoader(hardmining_data_dir,
                                           train_files,
                                           target_is_indices=target_is_indices,
                                           number_of_class=number_of_class,
                                           Bipolar=Bipolar,
                                           addBipolar=addBipolar,
                                           sample_length=sample_length,
                                           exchange_channel=exchange_channel,
                                           exchange_positive_channel=exchange_positive_channel,
                                           spike_1channel=spike_1channel,
                                           train_spike_1channel_idx=train_spike_1channel_idx
                                           )

        val_dataset = MGBClassLoader(os.path.join(root, "val"),
                                     val_files,
                                     target_is_indices=target_is_indices,
                                     number_of_class=number_of_class,
                                     Bipolar=Bipolar,
                                     addBipolar=addBipolar,
                                     sample_length=sample_length,
                                     spike_1channel=spike_1channel,
                                     train_spike_1channel_idx=train_spike_1channel_idx)
        test_dataset = MGBClassLoader(os.path.join(root, "test"),
                                      test_files,
                                      target_is_indices=target_is_indices,
                                      number_of_class=number_of_class,
                                      Bipolar=Bipolar,
                                      addBipolar=addBipolar,
                                      sample_length=sample_length,
                                      spike_1channel=spike_1channel,
                                      train_spike_1channel_idx=train_spike_1channel_idx)

        print(f'Train, Val, Test sizes: {len(train_files)}, {len(val_files)}, {len(test_files)}')
        return train_dataset, test_dataset, val_dataset






class SingleShockDataset(Dataset):
    """Read single hdf5 file regardless of label, subject, and paradigm."""

    def __init__(self, file_path: Path, window_size: int = 200, stride_size: int = 1, start_percentage: float = 0,
                 end_percentage: float = 1):
        '''
        Extract datasets from file_path.

        param Path file_path: the path of target data
        param int window_size: the length of a single sample
        param int stride_size: the interval between two adjacent samples
        param float start_percentage: Index of percentage of the first sample of the dataset in the data file (inclusive)
        param float end_percentage: Index of percentage of end of dataset sample in data file (not included)
        '''
        self.__file_path = file_path
        self.__window_size = window_size
        self.__stride_size = stride_size
        self.__start_percentage = start_percentage
        self.__end_percentage = end_percentage

        self.__file = None
        self.__length = None
        self.__feature_size = None

        self.__subjects = []
        self.__global_idxes = []
        self.__local_idxes = []

        self.__init_dataset()

    def __init_dataset(self) -> None:
        self.__file = h5py.File(str(self.__file_path), 'r')
        self.__subjects = [i for i in self.__file]

        global_idx = 0
        for subject in self.__subjects:
            self.__global_idxes.append(global_idx)  # the start index of the subject's sample in the dataset
            subject_len = self.__file[subject]['eeg'].shape[1]
            # total number of samples
            total_sample_num = (subject_len - self.__window_size) // self.__stride_size + 1
            # cut out part of samples
            start_idx = int(total_sample_num * self.__start_percentage) * self.__stride_size
            end_idx = int(total_sample_num * self.__end_percentage - 1) * self.__stride_size

            self.__local_idxes.append(start_idx)
            global_idx += (end_idx - start_idx) // self.__stride_size + 1
        self.__length = global_idx

        self.__feature_size = [i for i in self.__file[self.__subjects[0]]['eeg'].shape]
        self.__feature_size[1] = self.__window_size

    @property
    def feature_size(self):
        return self.__feature_size

    def __len__(self):
        return self.__length

    def __getitem__(self, idx: int):
        subject_idx = bisect.bisect(self.__global_idxes, idx) - 1
        item_start_idx = (idx - self.__global_idxes[subject_idx]) * self.__stride_size + self.__local_idxes[subject_idx]
        return self.__file[self.__subjects[subject_idx]]['eeg'][:, item_start_idx:item_start_idx + self.__window_size]

    def free(self) -> None:
        if self.__file:
            self.__file.close()
            self.__file = None

    def get_ch_names(self):
        return self.__file[self.__subjects[0]]['eeg'].attrs['chOrder']


class ShockDataset(Dataset):
    """integrate multiple hdf5 files"""

    def __init__(self, file_paths: list_path, window_size: int = 200, stride_size: int = 1, start_percentage: float = 0,
                 end_percentage: float = 1):
        '''
        Arguments will be passed to SingleShockDataset. Refer to SingleShockDataset.
        '''
        self.__file_paths = file_paths
        self.__window_size = window_size
        self.__stride_size = stride_size
        self.__start_percentage = start_percentage
        self.__end_percentage = end_percentage

        self.__datasets = []
        self.__length = None
        self.__feature_size = None

        self.__dataset_idxes = []

        self.__init_dataset()

    def __init_dataset(self) -> None:
        self.__datasets = [
            SingleShockDataset(file_path, self.__window_size, self.__stride_size, self.__start_percentage,
                               self.__end_percentage) for file_path in self.__file_paths]

        # calculate the number of samples for each subdataset to form the integral indexes
        dataset_idx = 0
        for dataset in self.__datasets:
            self.__dataset_idxes.append(dataset_idx)
            dataset_idx += len(dataset)
        self.__length = dataset_idx

        self.__feature_size = self.__datasets[0].feature_size

    @property
    def feature_size(self):
        return self.__feature_size

    def __len__(self):
        return self.__length

    def __getitem__(self, idx: int):
        dataset_idx = bisect.bisect(self.__dataset_idxes, idx) - 1
        item_idx = (idx - self.__dataset_idxes[dataset_idx])
        return self.__datasets[dataset_idx][item_idx]

    def free(self) -> None:
        for dataset in self.__datasets:
            dataset.free()

    def get_ch_names(self):
        return self.__datasets[0].get_ch_names()


class TUEVLoader(torch.utils.data.Dataset):
    def __init__(self, root, files, sampling_rate=200):
        self.root = root
        self.files = files
        self.default_rate = 200
        self.sampling_rate = sampling_rate

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        sample = pickle.load(open(os.path.join(self.root, self.files[index]), "rb"))
        X = sample["signal"]
        if self.sampling_rate != self.default_rate:
            X = resample(X, 5 * self.sampling_rate, axis=-1)
        Y = int(sample["label"][0] - 1)
        X = torch.FloatTensor(X)
        return X, Y



class MGBClassLoader(torch.utils.data.Dataset):
    def __init__(self, root, files, original_format='pkl',target_is_indices=False, number_of_class=0, Bipolar=False, addBipolar=False, sample_length=False, exchange_channel=False,exchange_positive_channel=False, spike_1channel=False, train_spike_1channel_idx=0):
        self.root = root
        self.files = files
        # self.default_rate = 200
        # self.sampling_rate = sampling_rate # no longer needed; by default during evaluation, segment only does normalization, everything else is already processed
        self.data_format = original_format
        self.Bipolar = Bipolar
        self.addBipolar = addBipolar
        self.sample_length = sample_length
        self.target_is_indices = target_is_indices
        self.number_of_class = number_of_class

        self.exchange_channel=exchange_channel
        self.exchange_positive_channel = exchange_positive_channel
        self.flipping=False
        self.exchange_channel_id=[]
        self.spike_1channel=spike_1channel
        self.train_spike_1channel_idx=int(train_spike_1channel_idx)


    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        path_signal=os.path.join(self.root, self.files[index])


        if self.data_format  == 'mat':
            try:
                sample = mat73.loadmat(path_signal)
                X = sample['data']
                # X = X[:19, :]
                # X = EEG_avg(X)
                # X = EEG_clip(X) # avg and clip for mat data are already done in data_provider
                X = EEG_normalize(X)
                try:
                    Y = sample['y'].item()  # try to get 'y' and convert to scalar
                except KeyError:
                    Y = 0  # if key 'y' does not exist, set default value

            except TypeError:
                try:
                    sample = scipy.io.loadmat(path_signal)
                    X = sample['data']
            ############  for data that has only been filtered (when running all morgoth test sets)  ############
                    # X= X[:19,:]
                    # # print(X.shape)
                    # X = EEG_avg(X)
                    # X = EEG_clip(X)
            ############  for data that has only been filtered (when running all morgoth test sets)  ############
                    X = EEG_normalize(X)
                    try:
                        Y = sample['y'].item()  # try to get 'y' and convert to scalar
                    except KeyError:
                        if self.target_is_indices:
                            Y=[]
                        else:
                            Y = 0  # if key 'y' does not exist, set default value

                except Exception as e:
                    raise ValueError(f'Failed to load {path_signal}. Mat type error : {e}')
        else:
            # sample = pickle.load(open(path_signal, "rb"))
            # X = sample["X"]
            # # X = EEG_avg(X)
            # # X = EEG_clip(X)  # avg and clip for pkl data are already done in data_provider
            # X = EEG_normalize(X) # use norm on segment
            # Y = sample["y"]
            # if Y==-1: # MoE has some labels that are -1
            #     Y = 0

            try:
                with open(path_signal, "rb") as f:
                    sample = pickle.load(f)
            except Exception as e:
                print(f"[SKIP BAD FILE] {path_signal}: {type(e).__name__} - {e}")
                # randomly or sequentially take the next sample
                new_index = (index + 1) % len(self.files)
                return self.__getitem__(new_index)

            try:
                X = sample["X"]
                Y = sample["y"]
            except KeyError as e:
                print(f"[SKIP MISSING KEY] {path_signal}: {e}")
                new_index = (index + 1) % len(self.files)
                return self.__getitem__(new_index)

                # check if X has NaN/Inf
            if not isinstance(X, np.ndarray) or not np.isfinite(X).all():
                print(f"[SKIP INVALID X] {path_signal}")
                new_index = (index + 1) % len(self.files)
                return self.__getitem__(new_index)

                # normal processing
            X = EEG_normalize(X)
            if Y == -1:  # MoE has some labels that are -1
                Y = 0

        if self.exchange_channel:
            self.flipping = random.random() < 0.5
            if  self.flipping:
                self.exchange_channel_id = list(range(0, 8))
            else:
                n = random.randint(1, 8)  # randomly generate an integer in [1, 8]
                self.exchange_channel_id = sorted(random.sample(range(0, 8), n - 1))

            for ch in self.exchange_channel_id:
                tmp = X[ch].copy()
                X[ch] = X[ch + 11]
                X[ch + 11] = tmp

        if self.target_is_indices or self.spike_1channel:
            if self.spike_1channel:
                one_hot_len=19

            else:
                one_hot_len = self.number_of_class

            y = np.zeros((one_hot_len,), dtype=np.float32)

            # multiple indices
            if isinstance(Y, (list, np.ndarray)):
                for k in Y:
                    try:
                        k = int(k)  # force convert to Python int
                    except Exception:
                        print('Wrong multiple labels when converting to one-hot vector')
                        continue
                    if 0 <= k < one_hot_len:
                        y[k] = 1.0
            else: # single index
                try:
                    k = int(Y)
                    if 0 <= k < one_hot_len:
                        y[k] = 1.0
                except Exception:
                    print('Wrong labels when converting to one-hot vector')
                    pass

            if self.exchange_channel:
                # swap labels according to self.exchange_channel_id
                for ch in self.exchange_channel_id:
                    y[ch], y[ch + 11] = y[ch + 11], y[ch]

            elif self.exchange_positive_channel:
                k = random.randint(1, len(Y))  # randomly decide k
                self.exchange_channel_id = random.sample(list(Y), k)

                for ch in self.exchange_channel_id:
                    y[ch], y[ch + 11] = y[ch + 11], y[ch]

                    tmp = X[ch].copy()
                    X[ch] = X[ch + 11]
                    X[ch + 11] = tmp

            if self.spike_1channel:
                if self.train_spike_1channel_idx==-1:
                    train_spike_1channel_idx_random = random.randint(0, 18)  # randomly select an integer in [0, 18]
                    y = int(y[train_spike_1channel_idx_random])
                    # copy that channel's data to all channels
                    X[:] = X[train_spike_1channel_idx_random]
                else:
                    y=int(y[self.train_spike_1channel_idx])
                    # copy that channel's data to all channels
                    X[:] = X[self.train_spike_1channel_idx]

            Y = y

        if self.sample_length:
            if self.sample_length != 15:
                middle_start = (15 / 2 - self.sample_length / 2) * 200
                middle_end = (15 / 2 + self.sample_length / 2) * 200
                X = X[:, int(middle_start):int(middle_end)]
                X = X[:, :int(self.sample_length) * 200]

        if self.Bipolar:
            X, _ = bipolar(X)
            X = X[:, :2800]  # 200*int(256/18)=1200  about 14 seconds
        elif self.addBipolar:
            X_bi, _ = bipolar(X)
            X = np.vstack((X, X_bi))
            X = X[:, :1200]  # 200*int(256/37)=1200  about 6 seconds
        else:
            X = X[:, :2600]  # 200*int(256/19)=2600  about 13 seconds

        X = torch.FloatTensor(X)
        return X, Y




class ContinuousToSnippetDataset(torch.utils.data.Dataset):
    # Dataset that takes a continous signal and returns snippets of a given length
    # input shape: (n_channels,n_timepoints), output shape: (n_snippets,n_channels,ts)
    def __init__(self,type,
                 path_signal,
                 given_fs=0,
                 has_avg=False,
                 has_formatted_channel=False,
                 allow_missing_channels=False,
                 montage = None,
                 transform = None,
                 step = 2000,
                 step_in_point = True,
                 polarity=1,
                 leave_one_hemisphere_out=False,
                 channel_symmetric_flip=False,
                 max_length_hour=None,
                 spike_1channel_result_file_df=None,
                 IIIC_result_file_df=None,

                ):

        self.type = type
        self.fs = 200
        self.original_avg=False
        self.missing_channels=None
        self.mono_channels=None
        self.leave_one_hemisphere_out=leave_one_hemisphere_out
        self.channel_symmetric_flip=channel_symmetric_flip

        file_extension = os.path.splitext(path_signal)[1].lower()
        if file_extension == '.mat':
            try:
                raw= mat73.loadmat(path_signal)
                signal=raw['data']
                self_fs=get_frequency_from_mat( raw_mat=raw)

            except TypeError:
                try:
                    raw = hdf5storage.loadmat(path_signal)
                    signal = raw['data']
                    self_fs = get_frequency_from_mat(raw_mat=raw)

                except Exception as e:
                    try:
                        raw = scipy.io.loadmat(path_signal)
                        signal = raw['data']
                        self_fs = get_frequency_from_mat(raw_mat=raw)

                    except Exception as e:
                        raise ValueError(f'Failed to load {path_signal}. Mat type error : {e}')

            if has_formatted_channel:
                if self.type == 'SLEEPPSG' or self.type=="SLEEPPSG_6class" or self.type == 'SLEEP_AROUSAL':
                    if signal.shape[0]==19:
                        signal = signal[[ 1, 2, 7, 12, 13, 18], :]
                    else:
                        signal =signal[:6,:]
                    original_avg = False
                else:
                    signal=signal[:19,:]
                    original_avg=False

            else:
                # order channels; if original data is not sorted, channel names must be provided in the mat file
                channels, original_avg = get_channel_names_from_mat(raw)

                data_dic = dict(zip(channels, signal))

                if self.type == 'SLEEPPSG' or self.type=="SLEEPPSG_6class" or self.type == 'SLEEP_AROUSAL':
                    if set(channels).issuperset(set(sleep_channels1)):
                        data_dic = sort_dict_by_keys(input_dict=data_dic, key_order=sleep_channels1,
                                                     default_value=np.array([0] * signal.shape[1]))
                    elif set(channels).issuperset(set(sleep_channels2)):
                        data_dic = sort_dict_by_keys(input_dict=data_dic, key_order=sleep_channels2,
                                                     default_value=np.array([0] * signal.shape[1]))

                    elif set(channels).issuperset(set(sleep_channels2_1)):
                        data_dic = sort_dict_by_keys(input_dict=data_dic, key_order=sleep_channels2_1,
                                                     default_value=np.array([0] * signal.shape[1]))

                    elif set(channels).issuperset(set(sleep_channels2_2)):
                        data_dic = sort_dict_by_keys(input_dict=data_dic, key_order=sleep_channels2_2,
                                                     default_value=np.array([0] * signal.shape[1]))

                    else:
                        raise ValueError(
                            "EDF file does not contain all channels from either sleep_channels1 or sleep_channels2.")
                else:
                    if set(channels).issuperset(set(eeg_channels1)):
                        data_dic = sort_dict_by_keys(input_dict=data_dic, key_order=eeg_channels1,
                                                     default_value=np.array([0] * signal.shape[1]))
                    elif set(channels).issuperset(set(eeg_channels2)):
                        data_dic = sort_dict_by_keys(input_dict=data_dic, key_order=eeg_channels2,
                                                     default_value=np.array([0] * signal.shape[1]))
                    else:
                        if allow_missing_channels:
                            missing_channels1 = set(eeg_channels1) - set(channels)
                            missing_channels2 = set(eeg_channels2) - set(channels)

                            if len(missing_channels1) <= len(missing_channels2):
                                self.mono_channels = eeg_channels1
                                self.missing_channels = [ch for ch in eeg_channels1 if ch not in channels]
                                selected_channels = [ch for ch in eeg_channels1 if ch in channels]
                            else:
                                self.mono_channels = eeg_channels2
                                self.missing_channels = [ch for ch in eeg_channels2 if ch not in channels]
                                selected_channels = [ch for ch in eeg_channels2 if ch in channels]

                            data_dic = sort_dict_by_keys(input_dict=data_dic, key_order=selected_channels,
                                                         default_value=np.array([0] * signal.shape[1]))

                        else:
                            raise ValueError("Mat file does not contain all EEG channels")

                signal = np.array(list(data_dic.values()))

            if max_length_hour is not None and signal.shape[1] > int(max_length_hour * 3600 * self_fs):
                signal = signal[:, :int(max_length_hour * 3600 * self_fs)]

            if polarity == -1:
                signal = signal * -1

            if original_avg or has_avg:
                self.original_avg = True
            else:
                self.original_avg = False

        elif file_extension == '.edf' or file_extension == '.EDF':
            try:
                raw = mne.io.read_raw_edf(path_signal, preload=True)
            except Exception as e:
                raise ValueError(f"Edf file {path_signal} corrupted: {e}")

            # may have EEG in name initially
            raw.rename_channels(
                {name: name.replace('EEG', '').replace('eeg', '').replace('POL', '').replace('pol', '').strip() for name in raw.info['ch_names']})

            new_channel_names = {ch_name: ch_name.upper() for ch_name in raw.ch_names}
            raw.rename_channels(new_channel_names)
            channels = raw.ch_names

            if '-AVG' in channels[0]:
                self.original_avg = True
                # channels = [ch.replace('-AVG', '') for ch in channels]
            elif has_avg:
                self.original_avg = True
            else:
                self.original_avg = False

            # Remove the reference name in the channel names
            # new_channel_names={ch_name: ch_name.split('-')[0] for ch_name in channels}
            # raw.rename_channels(new_channel_names)

            # Remove the reference name in the channel names; if duplicates arise after removal, keep the original
            # Create a mapping of new channel names

            # new_channel_names = {ch_name: ch_name.split('-')[0] for ch_name in channels}
            # new_channel_names = {ch_name: ch_name.split('(')[0] for ch_name in new_channel_names}

            new_channel_names = {
                ch_name: re.sub(r"\(.*?\)", "", ch_name).split('-')[0].strip()
                for ch_name in channels
            }

            # Count occurrences of each new channel name
            counter = Counter(new_channel_names.values())
            # Create final mapping: only rename unique channels
            final_channel_names = {}
            for old_name, new_name in new_channel_names.items():
                if counter[new_name] == 1:  # Only rename if unique
                    final_channel_names[old_name] = new_name
                else:
                    final_channel_names[old_name] = old_name  # Keep original name if duplicate
            # Apply the updated names
            raw.rename_channels(final_channel_names)


            channels=raw.ch_names

            # sleep must have 6 channels
            if self.type=='SLEEPPSG' or self.type=="SLEEPPSG_6class" or self.type == 'SLEEP_AROUSAL':
                if set(channels).issuperset(set(sleep_channels1)):
                    selected_channels = sleep_channels1
                elif set(channels).issuperset(set(sleep_channels2)):
                    selected_channels = sleep_channels2
                elif set(channels).issuperset(set(sleep_channels2_1)):
                    selected_channels = sleep_channels2_1
                elif set(channels).issuperset(set(sleep_channels2_2)):
                    selected_channels = sleep_channels2_2

                else:
                    missing = set(sleep_channels1) - set(channels)
                    if not missing:
                        missing = set(sleep_channels2) - set(channels)
                        if not missing:
                            missing = set(sleep_channels2_1) - set(channels)
                            if not missing:
                                missing = set(sleep_channels2_2) - set(channels)
                    raise ValueError(
                        f"{path_signal} EDF file does not contain all channels from either sleep_channels1 or sleep_channels. Missing {missing}")

            else:
                # can have fewer channels, fill missing with 0
                if allow_missing_channels:
                    missing_channels1 = set(eeg_channels1) - set(channels)
                    missing_channels2 = set(eeg_channels2) - set(channels)
                    if len(missing_channels1) <= len(missing_channels2):
                        self.mono_channels=eeg_channels1
                        self.missing_channels = [ch for ch in eeg_channels1 if ch not in channels]
                        selected_channels = [ch for ch in eeg_channels1 if ch in channels]
                    else:
                        self.mono_channels = eeg_channels2
                        self.missing_channels = [ch for ch in eeg_channels2 if ch not in channels]
                        selected_channels = [ch for ch in eeg_channels2 if ch in channels]
                else:
                    if set(channels).issuperset(set(eeg_channels1)):
                        selected_channels = eeg_channels1
                    elif set(channels).issuperset(set(eeg_channels2)):
                        selected_channels = eeg_channels2
                    else:
                        missing = set(eeg_channels1) - set(channels)
                        if not missing:
                            missing = set(eeg_channels2) - set(channels)
                        raise ValueError(f"{path_signal} EDF file does not contain all channels from either eeg_channels1 or eeg_channels2. Missing {missing}")

            raw_selected = raw.copy().pick(selected_channels)
            if max_length_hour is not None and raw_selected.times[-1] >  max_length_hour * 3600:
                raw_selected.crop(tmin=0, tmax=int(max_length_hour * 3600))
            signal=raw_selected.get_data(units='uV')

            ############### for HEP data, should *10 before input the model ################
            # signal=signal*10
            # print('hep\'s signal * 10')
            ############### for HEP data, should *10 before input the model ################
            if polarity == -1:
                signal = signal * -1

            self_fs = int(raw.info['sfreq'])

        elif file_extension == '.pkl':
            with open(path_signal, 'rb') as f:
                eeg_data = pickle.load(f)
            signal=eeg_data['X'][:19,:]
            self_fs = given_fs
            self.original_avg=False

        else:
            raise ValueError("Should be mat or edf or pkl.")


        # if the data file contains fs, check if it matches the given fs; if not, use the one from the data
        if self_fs == 0:
            if given_fs != 0:
                self_fs = given_fs
            else:
                raise ValueError(f'{path_signal} has no sampling rate in data file or input parameter')

        elif given_fs != 0 and self_fs != given_fs:
            print('Input sampling rate dose not match recorded sampling rate, use recorded')

        self.self_fs = self_fs

        ######## Only test on center n s(compare with non-continuous baselines) ##################
        # signal=same_segment_with_kaggle(signal=signal,fs=self.self_fs,seq_length=30)
        ######## Only test on center n s(compare with non-continuous baselines) ##################

        # mark indices for flat/extreme values
        def is_constant(tensor):
            max_values = torch.max(tensor, dim=1).values
            min_values = torch.min(tensor, dim=1).values
            diff = max_values - min_values
            return torch.all(diff < 1).item()

        # invalid segments are usually not multiples of 10, so set neighbors of invalid indices to invalid as well
        def expand_zeros(lst):
            lst = np.array(lst)  # convert to NumPy array
            zero_mask = (lst == 0)

            # create offset arrays
            left_shift = np.roll(zero_mask, shift=1)
            right_shift = np.roll(zero_mask, shift=-1)

            # avoid boundary effects
            left_shift[0] = False
            right_shift[-1] = False

            # only affect positions that were originally 1
            lst[(left_shift | right_shift) & (lst == 1)] = 0
            return lst.tolist()

        window_size=self._get_window_size()

        if step_in_point:
            original_step = step
            new_step = max(int(original_step / self.self_fs * self.fs), 1)  # minimum step 1 point
        else:
            original_step = int(step * self.self_fs)
            new_step = int(step * self.fs)

        original_window_size=window_size * self.self_fs # generate snippets of shape (n_snippets,n_channels,ts)
        new_window_size = window_size * self.fs

        original_signal = torch.FloatTensor(signal.astype(np.float32))
        original_snippets = original_signal.unfold(dimension=1, size=original_window_size, step=original_step).permute(1, 0, 2)

        num_snippets, num_channels, time_steps = original_snippets.shape
        self.valid_data_index = [0]*num_snippets

        cut_min = 20 if ("SPIKE" in self.type or "VW" ==self.type) else 2

        for snippet_idx in range(num_snippets):
            snippet_data = original_snippets[snippet_idx, :, :]
            is_valid = 1 # assume valid

            # check for NaN
            if torch.any(torch.all(torch.isnan(snippet_data), dim=0)).item():
                is_valid = 0

            # check for extreme values
            if not self.original_avg and (torch.all(torch.abs(snippet_data) < cut_min) or torch.all(torch.abs(snippet_data) >3000)):
                is_valid = 0

            # check for flat signal
            # subtract the mean at each time point; some synchronously changing signals are also flat
            snippet_data -= torch.mean(snippet_data, dim=0, keepdim=True)
            if is_constant(snippet_data):
                is_valid=0

            self.valid_data_index[snippet_idx]=is_valid

        self.valid_data_index=expand_zeros(self.valid_data_index)
        if all(x == 0 for x in self.valid_data_index):
            raise SnippetsError(result_segment_shapes=int(num_snippets) ,message="EEG has no valid snippets")

        # find start and end indices of valid segments
        def find_continuous_valid_indices(arr):
            arr = np.array(arr)
            # find transition points between 1 and 0
            diff = np.diff(arr, prepend=0, append=0)
            # find start and end positions of consecutive 1s
            starts = np.where(diff == 1)[0]
            ends = np.where(diff == -1)[0] - 1
            return list(zip(starts, ends))

        # start and end indices of valid segments
        self.valid_start_end_indices = find_continuous_valid_indices(self.valid_data_index)

        self.result_segment_shape=[]# length of each result segment
        all_snippets = []
        for start_idx, end_idx in self.valid_start_end_indices:
            start_point= original_step*start_idx
            end_point= original_step*end_idx+original_window_size
            valid_signal=signal[:,start_point:end_point]

            if (("SPIKE" in self.type) or ("VW" == self.type))and self.self_fs>128:
                valid_signal = resample_signal(valid_signal, original_rate=self.self_fs,target_rate=128)
                valid_signal = EEG_bandfilter(valid_signal, fs=128)
                valid_signal = EEG_notchfilter(valid_signal, fs=128)
                self_fs=128

            valid_signal = resample_signal(valid_signal, original_rate=self_fs, target_rate=self.fs)
            valid_signal = EEG_bandfilter(valid_signal, fs=self.fs)
            valid_signal = EEG_notchfilter(valid_signal, fs=self.fs)

            # move signal to torch
            valid_signal = torch.FloatTensor(valid_signal.astype(np.float32))
            # generate snippets of shape (n_snippets,n_channels,ts)
            snippet=valid_signal.unfold(dimension=1, size=new_window_size, step=new_step).permute(1, 0, 2)
            all_snippets.append(snippet)

            self.result_segment_shape.append(snippet.shape[0])

        self.snippets = torch.cat(all_snippets, dim=0)

        if spike_1channel_result_file_df is not None:

            snip = self.snippets # self.snippets: [N, C, T]
            N, C, T = snip.shape
            # print(f'--original snip len {N}')

            # print(f'--original result df len {len(spike_1channel_result_file_df)}')

            filtered_df = spike_1channel_result_file_df[spike_1channel_result_file_df['pred'] > 0].reset_index(drop=True)
            # print(f'-->0  result df len {len(filtered_df)}')

            resized_filtered_df = resize_df_along_axis0(filtered_df, target_length=N)
            # print(f'--resized result df len {len(resized_filtered_df)}')

            self.spike_time_idx = resized_filtered_df.index[resized_filtered_df['pred'] > 0.5].tolist()

            # after selecting subset, sample count becomes M
            self.snippets = snip[self.spike_time_idx]

            # perform average montage before channel copying; if done later like others, it becomes all zeros
            channel_means = self.snippets.mean(dim=1, keepdim=True)
            self.snippets = self.snippets - channel_means
            # print(f'--channel_means shape {channel_means.shape}')

            # perform clip before channel copying; faster than doing it after copying
            self.snippets = torch.clamp(self.snippets, -500, 500)

            # perform scaling before channel copying; faster than doing it after copying
            # target interval [-100, 100]
            a, b = -100.0, 100.0
            x = self.snippets  # [M, C, T], torch.Tensor
            # min-max along time dimension per sample × channel
            x_min = x.amin(dim=2, keepdim=True)  # [M, C, 1]
            x_max = x.amax(dim=2, keepdim=True)  # [M, C, 1]
            denom = (x_max - x_min).clamp_min(1e-6)  # avoid division by zero (all-constant channel)
            # normalize to [0,1]
            x_norm = (x - x_min) / denom
            # linear map to [a, b] = [-S, S]
            self.snippets = x_norm * (b - a) + a  # still [M, C, T] torch.Tensor (preserves gradient/device)

            M = self.snippets.shape[0]
            # print(f'--spike snip len {M}')

            # copy: for each channel of each sample, create a copy where all channels have that channel's waveform
            rep = self.snippets[:, :, None, :].repeat(1, 1, C, 1)  # [M, C, C, T]

            # merge “source channel” into batch dimension: get [M*C, C, T]
            self.snippets = rep.reshape(M * C, C, T)
            # print(f'--new snip len {self.snippets.shape[0]}')


        elif IIIC_result_file_df is not None:
            snip = self.snippets  # self.snippets: [N, C, T]

            resized_filtered_df = IIIC_result_file_df
            # print(f'--resized result df len {len(resized_filtered_df)}')

            #  df contains class_0_prob ~ class_6_prob
            cols_1to6 = [f'class_{i}_prob' for i in range(0, 6)]
            # “100000” mode: class_1_prob=1, others are 0
            pattern = np.array([1, 0, 0, 0, 0, 0], dtype=float)
            mask_100000 = np.isclose(
                IIIC_result_file_df[cols_1to6].to_numpy(dtype=float),
                pattern
            ).all(axis=1)
            # 1) remove these rows
            resized_filtered_df = resized_filtered_df.loc[~mask_100000].copy()
            # 2) reindex
            resized_filtered_df.reset_index(drop=True, inplace=True)
            # 3) then get indices where pred_class == 1
            self.seizure_time_idx = resized_filtered_df.index[
                resized_filtered_df['pred_class'] == 1
                ].tolist()
            # select subset
            self.snippets = snip[self.seizure_time_idx]
            # print(f'--seizure_time_idx len {len(self.seizure_time_idx)}')
            # print(f'--new snip len {self.snippets.shape[0]}')


        # set montage
        self.montage = montage
        # set transform
        self.transform = transform


    def __len__(self):
        # get item zero of self. snippets, which has shape (n_snippets,n_channels,ts)
        return self.snippets.shape[0]

    def _preprocess(self, signal):
        '''preprocess signal and apply montage, transform and normalization'''

        # apply montage: avg
        if self.montage is not None and self.type != 'SPIKE_1channel': #SPIKE_1channel performs montage in init

            signal = self.montage(signal,mono_channels=self.mono_channels,missing_channels=self.missing_channels)

        # apply transformations: clip and scaling (normalize)
        if self.transform is not None:
            signal = self.transform(signal)

        if self.leave_one_hemisphere_out is not False:
            signal=leave_one_hemisphere_out_func(data=signal,side=self.leave_one_hemisphere_out)
        if self.channel_symmetric_flip is not False:
            signal = channel_symmetric_flip_func(data=signal,side=self.channel_symmetric_flip)

        # transfer to torch
        if isinstance(signal, np.ndarray):
            signal = torch.FloatTensor(signal.copy())

        return signal

    def __getitem__(self, idx):
        # get the snippet
        # print(self.snippets)
        signal = self.snippets[idx, :, :]
        # preprocess signal
        signal = self._preprocess(signal)

        # return signal and dummy label, the latter to prevent lightning dataloader from complaining
        return signal,0

    def _get_window_size(self):
        if self.type=='SPIKES' or self.type=='VW' or self.type=='VW_SPIKES' or self.type=='SPIKE_localization' or self.type=='SPIKE_1channel':
            return 1
        else: return 10

    def get_valid_indices(self):
        return self.valid_data_index, self.valid_start_end_indices, self.result_segment_shape

    def get_original_fs(self):
        return self.self_fs

    def get_spike_time_idx(self):
        return self.spike_time_idx

def get_n_classes(dataset):
    n_class_map={
        'NORMAL': 1,
        'BS':1,
        'SPIKES':1,
        'FOC_GEN_SPIKES':3,
        'SLOWING':3,
        'IIIC':6,
        'IIIC_hm':7,
        'IIIC_chewing': 7,
        'MGBSLEEP3stages': 3,
        'SLEEPPSG':5,
        "SLEEPPSG_6class":6,
        'SLEEP_AROUSAL':1,
        'VW': 1,
        'VW_SPIKES': 3,
        'BIRD':1,
        'BIPD': 1,
        'PD': 1,
        'SPIKE_localization': 19,
        'SPIKE_1channel':1,
    }
    return n_class_map[dataset]


class common_average_montage():
    def __init__(self):
        self.mono_channels = eeg_channels1
        # self.channel_average = ['FP1-avg', 'F3-avg', 'C3-avg', 'P3-avg', 'F7-avg', 'T3-avg', 'T5-avg', 'O1-avg', 'FZ-avg', 'CZ-avg', 'PZ-avg', 'FP2-avg', 'F4-avg', 'C4-avg', 'P4-avg', 'F8-avg', 'T4-avg', 'T6-avg', 'O2-avg']  # 19
        #self.average_ids = [self.mono_channels.index(ch.split('-')[0]) for ch in self.channel_average]

    def __call__(self, signal, mono_channels=None, missing_channels=None):
        # Common Average Montage
        # signal = signal[self.average_ids]
        column_means = torch.mean(signal, dim=0, keepdim=True)
        data_centered = signal - column_means

        if missing_channels is not None:
            self.mono_channels = mono_channels
            valid_channels = [ch for ch in self.mono_channels if ch not in missing_channels]
            valid_indices = [self.mono_channels.index(ch) for ch in valid_channels]

            # create an all-zero tensor to store the full signal data
            num_channels = len(self.mono_channels)
            num_samples = data_centered.size(1)
            full_signal = torch.zeros((num_channels, num_samples))

            # fill the original signal into the corresponding positions in the full signal
            for idx, valid_idx in enumerate(valid_indices):
                full_signal[valid_idx, :] = data_centered[idx, :]

            data_centered=full_signal
        return data_centered

    def get_channel_names(self):
        return self.mono_channels


class partial_average_montage():
    def __init__(self, location='left'):
        self.mono_channels = eeg_channels1
        self.location=location
        # self.channel_average = ['FP1-avg', 'F3-avg', 'C3-avg', 'P3-avg', 'F7-avg', 'T3-avg', 'T5-avg', 'O1-avg', 'FZ-avg', 'CZ-avg', 'PZ-avg', 'FP2-avg', 'F4-avg', 'C4-avg', 'P4-avg', 'F8-avg', 'T4-avg', 'T6-avg', 'O2-avg']  # 19
        #self.average_ids = [self.mono_channels.index(ch.split('-')[0]) for ch in self.channel_average]

    def __call__(self,
                 signal,
                 mono_channels=None,
                 missing_channels=None
            ):

        column_means = torch.mean(signal, dim=0, keepdim=True)
        data_centered = signal - column_means

        if self.location in ['left', 'l']:
            # replace the last 8 channels with copies of the first 8 channels
            data_centered[-8:, :] = data_centered[:8, :]

        elif self.location in ['right', 'r']:
            # replace the first 8 channels with copies of the last 8 channels
            data_centered[:8, :] = data_centered[-8:, :]

        else:
            raise ValueError("location must be 'left | l' or 'right | r'")

        return data_centered

    def get_channel_names(self):
        return self.mono_channels



class sleep_common_average_montage():
    def __init__(self):
        self.mono_channels =['F3', 'C3',  'O1',  'F4', 'C4', 'O2']

        # self.channel_average =  ['F3-avg', 'C3-avg', 'O1-avg', 'F4-avg', 'C4-avg', 'O2-avg']
        # self.average_ids = [self.mono_channels.index(ch.split('-')[0]) for ch in self.channel_average]

    def __call__(self, signal,mono_channels=None,missing_channels=None):
        # Common Average Montage
        # signal=signal[self.average_ids]
        column_means = torch.mean(signal, dim=0, keepdim=True)
        data_centered = signal - column_means

        return data_centered

    def get_channel_names(self):
        return self.mono_channels

class bipolar_montage():
    def __init__(self):
        self.mono_channels = ['FP1', 'F3', 'C3', 'P3', 'F7', 'T3', 'T5', 'O1', 'FZ', 'CZ', 'PZ', 'FP2', 'F4', 'C4', 'P4', 'F8', 'T4', 'T6', 'O2']
        self.bipolar_channels = ['FP1-F7', 'F7-T3', 'T3-T5', 'T5-O1', 'FP2-F8', 'F8-T4', 'T4-T6', 'T6-O2', 'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1', 'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2', 'FZ-CZ', 'CZ-PZ'] # 18


        self.bipolar_ids = np.array(
            [[self.mono_channels.index(bc.split('-')[0]), self.mono_channels.index(bc.split('-')[1])] for bc in self.bipolar_channels])

    def __call__(self, signal):
        # Bipolar Montage
        bipolar_signal = signal[self.bipolar_ids[:, 0]] - signal[self.bipolar_ids[:, 1]]
        return bipolar_signal

    def get_channel_names(self):
        return self.bipolar_channels


def bipolar(data_monopolar):
    # Initialize the bipolar data array
    data_bipolar = np.zeros((18, data_monopolar.shape[1]))
    # Group LL
    data_bipolar[0, :] = data_monopolar[0, :] - data_monopolar[4, :]  # Fp1-F7
    data_bipolar[1, :] = data_monopolar[4, :] - data_monopolar[5, :]  # F7-T3
    data_bipolar[2, :] = data_monopolar[5, :] - data_monopolar[6, :]  # T3-T5
    data_bipolar[3, :] = data_monopolar[6, :] - data_monopolar[7, :]  # T5-O1

    # Group RL
    data_bipolar[4, :] = data_monopolar[11, :] - data_monopolar[15, :]  # Fp2-F8
    data_bipolar[5, :] = data_monopolar[15, :] - data_monopolar[16, :]  # F8-T4
    data_bipolar[6, :] = data_monopolar[16, :] - data_monopolar[17, :]  # T4-T6
    data_bipolar[7, :] = data_monopolar[17, :] - data_monopolar[18, :]  # T6-O2

    # Group LP
    data_bipolar[8, :] = data_monopolar[0, :] - data_monopolar[1, :]  # Fp1-F3
    data_bipolar[9, :] = data_monopolar[1, :] - data_monopolar[2, :]  # F3-C3
    data_bipolar[10, :] = data_monopolar[2, :] - data_monopolar[3, :]  # C3-P3
    data_bipolar[11, :] = data_monopolar[3, :] - data_monopolar[7, :]  # P3-O1

    # Group RP
    data_bipolar[12, :] = data_monopolar[11, :] - data_monopolar[12, :]  # Fp2-F4
    data_bipolar[13, :] = data_monopolar[12, :] - data_monopolar[13, :]  # F4-C4
    data_bipolar[14, :] = data_monopolar[13, :] - data_monopolar[14, :]  # C4-P4
    data_bipolar[15, :] = data_monopolar[14, :] - data_monopolar[18, :]  # P4-O2

    # Group midline
    data_bipolar[16, :] = data_monopolar[8, :] - data_monopolar[9, :]  # Fz-Cz
    data_bipolar[17, :] = data_monopolar[9, :] - data_monopolar[10, :]  # Cz-Pz

    # 10-20 system
    channel_names = ["Fp1-F7","F7-T3","T3-T5","T5-O1","Fp2-F8","F8-T4","T4-T6","T6-O2","Fp1-F3","F3-C3","C3-P3","P3-O1","Fp2-F4","F4-C4","C4-P4","P4-O2","Fz-Cz","Cz-Pz"]
    # MCN system, coresponding to the above

    # channel_names=["FP1-F7", "F7-T7", "T7-P7", "P7-O1", "FP2-F8", "F8-T8", "T8-P8", "P8-O2", "FP1-F3", "F3-C3", "C3-P3", "P3-O1","FP2-F4", "F4-C4", "C4-P4", "P4-O2"]

    return data_bipolar,channel_names

class combine_montage():
    def __init__(self):
        self.mono_channels = ['FP1', 'F3', 'C3', 'P3', 'F7', 'T3', 'T5', 'O1', 'FZ', 'CZ', 'PZ', 'FP2', 'F4', 'C4', 'P4',
                         'F8', 'T4', 'T6', 'O2']
        self.bipolar_channels = ['FP1-F7', 'F7-T3', 'T3-T5', 'T5-O1', 'FP2-F8', 'F8-T4', 'T4-T6', 'T6-O2', 'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1', 'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2', 'FZ-CZ', 'CZ-PZ']  # 18
        self.channel_average = ['FP1-avg', 'F3-avg', 'C3-avg', 'P3-avg', 'F7-avg', 'T3-avg', 'T5-avg', 'O1-avg', 'FZ-avg', 'CZ-avg', 'PZ-avg', 'FP2-avg', 'F4-avg', 'C4-avg', 'P4-avg', 'F8-avg', 'T4-avg', 'T6-avg',
                           'O2-avg']  # 19

        self.bipolar_ids = np.array(
            [[self.mono_channels.index(bc.split('-')[0]), self.mono_channels.index(bc.split('-')[1])] for bc in self.bipolar_channels])

        self.average_ids = [self.mono_channels.index(ch.split('-')[0]) for ch in self.channel_average]


    def __call__(self, signal):
        common_average_signal =  signal[self.average_ids] - torch.mean(signal[self.average_ids], dim=0, keepdim=True)
        bipolar_signal = signal[self.bipolar_ids[:, 0]] - signal[self.bipolar_ids[:, 1]]

        combined_signal = np.vstack([common_average_signal,bipolar_signal])

        return combined_signal

    def get_channel_names(self):
        return  self.mono_channels+self.bipolar_channels


class single_channel_average_montage():
    def __init__(self,channel_idx):
        self.channel_idx = [channel_idx]
        self.mono_channels = ['FP1', 'F3', 'C3', 'P3', 'F7', 'T3', 'T5', 'O1', 'FZ', 'CZ', 'PZ', 'FP2', 'F4', 'C4', 'P4', 'F8', 'T4', 'T6', 'O2']

    def __call__(self, signal):
        # Common Average Montage
        signal=signal[self.channel_idx]
        column_means = torch.mean(signal, dim=0, keepdim=True)
        data_centered = signal - column_means

        return data_centered

    def get_channel_names(self):
        return [self.mono_channels[self.channel_idx[0]]]


class Compose:
    """Composes several transforms together. This transform does not support torchscript.
    """
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += f"    {t}"
        format_string += "\n)"
        return format_string

class Clipping:
    def __init__(self,clip_at = 500):
        self.clip_at = clip_at #microV

    def __call__(self,signal):
        out_data = np.clip(signal, -self.clip_at, self.clip_at)
        return out_data

class Scaling:
    def __init__(self, scaling_at=100):
        self.scaling_at = scaling_at  # microV

    def __call__(self, signal):
        normalized_eeg_data = np.zeros_like(signal)
        for i in range(signal.shape[0]):
            channel_data = signal[i, :].reshape(-1, 1)
            scaler = MinMaxScaler(feature_range=(-self.scaling_at, self.scaling_at))
            normalized_channel_data = scaler.fit_transform(channel_data)
            normalized_eeg_data[i, :] = normalized_channel_data.flatten()
        return normalized_eeg_data


class Normalize: # not use
    def __init__(self, q=0.95):
        self.q = q  # microV

    def __call__(self, signal):
        normalized_eeg_data = signal / (np.quantile(np.abs(signal), q=self.q, method="linear", axis=-1, keepdims=True) + 1e-8)
        return normalized_eeg_data


def EEG_bandfilter(data, fs, order=4, low=0.5, high=70):
    nyquist = 0.5 * fs
    low = low / nyquist
    high = high / nyquist
    if high>1:
        b, a = butter(order, low, btype='high') # only highpass
    else:
        b, a = butter(order, [low, high], btype='band')
    filtered_data = np.zeros_like(data)
    for i in range(data.shape[0]):
        filtered_data[i, :] = filtfilt(b, a, data[i, :])
    return filtered_data


def EEG_notchfilter(data, fs=200, notch_width=1.0):
    Q_50 = 50 / notch_width
    b_50, a_50 = iirnotch(50, Q_50, fs)

    Q_60 = 60 / notch_width
    b_60, a_60 = iirnotch(60, Q_60, fs)

    filtered_data = np.zeros_like(data)
    for i in range(data.shape[0]):
        filtered_data[i, :] = filtfilt(b_50, a_50, data[i, :])
        filtered_data[i, :] = filtfilt(b_60, a_60, filtered_data[i, :])
    return filtered_data


def replace_nan_with_channel_mean(data):
    for channel in range(data.shape[0]):
        channel_data = data[channel, :]
        channel_mean = np.nanmean(channel_data)
        channel_data[np.isnan(channel_data)] = channel_mean
        data[channel, :] = channel_data
    return data

def replace_nan_with_zero(data):
    data[np.isnan(data)] = 0
    return data

def interpolate_nan(signal):
    """Fill NaN values in a 1D signal using linear interpolation"""
    nans = np.isnan(signal)
    if np.any(nans):  # only process if the data contains NaN
        x = np.arange(len(signal))
        signal[nans] = np.interp(x[nans], x[~nans], signal[~nans])
    return signal

def EEG_avg(eeg_data):
    avg = np.mean(eeg_data, axis=0)
    out_data = eeg_data - avg[np.newaxis, :]
    return out_data

def EEG_clip(eeg_data):
    out_data = np.clip(eeg_data, -500, 500)
    return out_data

def EEG_normalize(eeg_data):
    out_data = np.zeros_like(eeg_data)
    for i in range(eeg_data.shape[0]):
        channel_data = eeg_data[i, :].reshape(-1, 1)
        scaler = MinMaxScaler(feature_range=(-100, 100))
        normalized_channel_data = scaler.fit_transform(channel_data)
        out_data[i, :] = normalized_channel_data.flatten()
    return out_data

def resample_signal(signal, original_rate, target_rate, n_jobs=5):

    if original_rate == target_rate:
        return signal
    # num_samples = int(signal.shape[1] * (target_rate / original_rate))
    # resampled_signal = np.zeros((signal.shape[0], num_samples))
    # for i in range(signal.shape[0]):
    #     resampled_signal[i, :] = resample(signal[i, :], num_samples)

    resampled_signal = mne.filter.resample(signal, down=original_rate, up=target_rate, n_jobs=n_jobs)
    return resampled_signal


def get_frequency_from_mat(raw_mat):
    try:
        fs_value = raw_mat['Fs']
    except KeyError:
        try:
            fs_value = raw_mat['fs']
        except KeyError:
            try:
                fs_value = raw_mat['sampling_rate']
            except KeyError:
                    return 0

    if isinstance(fs_value, np.ndarray):
        if fs_value.shape == (1, 1, 1):
            fs_value = fs_value[0, 0, 0]
        elif fs_value.shape == (1, 1):
            fs_value = fs_value[0, 0]
        elif fs_value.shape == (1,):
            fs_value = fs_value[0]
        elif fs_value.shape == ():
            fs_value=fs_value.item()
        else:
            print('Unexpected array shape for fs value in mat')
            return 0

    if isinstance(fs_value, np.ndarray):
        fs_value = fs_value.item()

    return int(fs_value)


def get_channel_names_from_mat(raw_mat):
    """
    Extract channel names from the channels array and remove leading/trailing spaces.
    :param channels: Channels array
    :return: List of channel names
    """
    try:
        channels = raw_mat['channels']
    except KeyError:
        try:
            channels = raw_mat['channel_locations']
        except KeyError:
            raise ValueError(f'No channel names found in mat')

    channel_names = []
    # Iterate through the channels array
    for channel in channels:
        # Handle different cases
        if isinstance(channel, np.ndarray):
            # If channel is a nested array
            if channel.size == 1:
                channel_name = channel.item()
            else:
                channel_name = channel[0]

            # Further unwrap if necessary
            if isinstance(channel_name, np.ndarray):
                if channel_name.size == 1:
                    channel_name = channel_name.item()
                else:
                    channel_name = channel_name[0]

                # Further unwrap if necessary
                if isinstance(channel_name, np.ndarray):
                    if channel_name.size == 1:
                        channel_name = channel_name.item()
                    else:
                        channel_name = channel_name[0]
        elif isinstance(channel, list):
            # If channel is a list
            channel_name = channel[0]
        else:
            # If channel is a single element
            channel_name = channel

        # Ensure the channel name is a string
        if isinstance(channel_name, np.ndarray):
            channel_name = channel_name.item()
        channel_names.append(channel_name.strip())

    channel_names=[ch.upper() for ch in channel_names]
    if '-AVG' in channel_names[0]:
        hasAVG=True
        # channel_names=[ch.replace('-AVG', '') for ch in channel_names]
    else:
        hasAVG=False

    channel_names = [ch.split('-')[0] for ch in channel_names]

    return channel_names, hasAVG


def sort_dict_by_keys(input_dict, key_order, default_value=None, remaining_keys=False):
    """
    Sorts the keys of a dictionary according to a specified order, where the keys in the order appear in the strings of the dictionary keys.

    :param input_dict: Input dictionary
    :param key_order: List of keys specifying the order
    :param default_value: Default value for keys in `key_order` that are not found in `input_dict`
    :return: Sorted dictionary
    """
    sorted_dict = {}

    for key in key_order:
        # Find matching keys
        matched_keys = [k for k in input_dict.keys() if re.search(key.upper(), k.upper())]

        if matched_keys:
            # If there are matching keys, select the first one
            matched_key = matched_keys[0]
            sorted_dict[matched_key] = input_dict[matched_key]
        else:
            # If there are no matching keys, use the default value
            sorted_dict[key] = default_value

    # Add remaining keys
    if remaining_keys:
        for k in input_dict.keys():
            if k not in sorted_dict:
                sorted_dict[k] = input_dict[k]

    return sorted_dict



def remove_nan_columns(data):
    """
    Remove all-NaN columns and return the cleaned data, column count, and number of columns removed from front and back.

    Args:
        data (np.ndarray): 2D array of shape (19, n).

    Returns:
        cleaned_data (np.ndarray): data with all-NaN columns removed.
        n_removed_front_points (int): number of columns removed from the front.
        n_removed_back_points (int): number of columns removed from the back.
    """
    # check if each column is all NaN
    nan_columns = np.all(np.isnan(data), axis=0)

    # count columns removed from the front
    n_removed_front_points = 0
    for i in range(data.shape[1]):
        if nan_columns[i]:
            n_removed_front_points += 1
        else:
            break

    # count columns removed from the back
    n_removed_back_points = 0
    for i in range(data.shape[1] - 1, -1, -1):
        if nan_columns[i]:
            n_removed_back_points += 1
        else:
            break

    # remove all-NaN columns
    cleaned_data = data[:, ~nan_columns]

    # return cleaned data and removed column counts
    return cleaned_data, n_removed_front_points, n_removed_back_points


def find_valid_segment(data, n, threshold=5000):
    """
    Find a data segment containing column n where all values have absolute value less than the threshold.

    Args:
        data (np.ndarray): 2D array of shape (19, m).
        n (int): index of the target column.
        threshold (float): threshold value, default 5000.

    Returns:
        new_data (np.ndarray): new data segment, shape (19, new_m).
        new_n (int): index of the original column n in the new data.
    """
    num_channels, num_columns = data.shape

    # ensure n is within valid range
    if n < 0 or n >= num_columns:
        print(f"index {n} not in [0, {num_columns - 1}]")
        return np.array([]), -1

    # initialize start and end column indices of the segment
    start = n
    end = n

    # expand forward to find the starting column satisfying the condition
    while start > 0:
        if np.all(np.abs(data[:, start - 1]) < threshold):
            start -= 1
        else:
            break

    # expand backward to find the ending column satisfying the condition
    while end < num_columns - 1:
        if np.all(np.abs(data[:, end + 1]) < threshold):
            end += 1
        else:
            break

    # extract the new data segment
    new_data = data[:, start:end + 1]

    # compute index of the original column n in the new data
    new_n = n - start

    return new_data, new_n



def leave_one_hemisphere_out_func(data, side='right'):
    if side == 'right' or side == 'r':
        data[-8:, :] = 0
    elif  side == 'left' or side == 'l':
        data[:8, :] = 0
    elif side == 'middle' or side == 'm':
        data[8:11, :] = 0
    else:
        raise ValueError(f'Hemisphere side should be right or left or middle')
    return data

def channel_symmetric_flip_func(data,side='right'):
    if side == 'right' or side == 'r':
        data[:8, :] =  data[-8:, :]
    elif side == 'left' or side == 'l':
        data[-8:, :] = data[:8, :]
    else:
        raise ValueError(f'Hemisphere side should be right or left')
    return data


def resize_array_along_axis0(arr, d, target_length):
    """
    Uniformly resize the first dimension of a 1D or 2D NumPy array (increase or decrease channels/samples).

    Args:
    - arr: original 1D or 2D NumPy array (original_length,) or (original_length, n_samples)
    - target_length: target length (can be larger or smaller than original)

    Returns:
    - new_arr: resized 1D or 2D NumPy array (target_length,) or (target_length, n_samples)
    """
    original_length = arr.shape[0]  # original first dimension (channels/samples)

    # if target length equals original length, return directly
    if target_length == original_length:
        return arr

    # compute new uniform indices
    indices = np.round(np.linspace(0, original_length - 1, target_length)).astype(int)

    if d==1:
        return arr[indices]
    else:
        # return arr[indices:,]
        return arr[indices,: ]


def resize_df_along_axis0(df: pd.DataFrame, target_length: int) -> pd.DataFrame:
    """
    Uniformly resize the number of rows in a DataFrame to target_length
    (can be more or fewer than original, by uniformly sampling row indices)

    Args:
    - df: input DataFrame
    - target_length: target number of rows

    Returns:
    - new_df: resized DataFrame
    """
    original_length = len(df)

    if target_length == original_length:
        return df.copy()

    # compute uniform row indices
    indices = np.round(
        np.linspace(0, original_length - 1, target_length)
    ).astype(int)

    return df.iloc[indices].reset_index(drop=True)

def split_nd_to_plus1d(arr, segment_shape):
    """
    Efficiently split an n-dimensional array by `segment_shape` using NumPy.

    Args:
    - arr: original n-dimensional NumPy array e.g. (total_rows, n_samples)
    - segment_shape: list of row counts for each sub-array to split into

    Returns:
    - segments: list of split (n+1)D NumPy arrays
    """
    indices = np.add.accumulate(segment_shape)[:-1]  # compute split indices (excluding the last)
    return np.split(arr, indices, axis=0)

class SnippetsError(Exception):
    def __init__(self, result_segment_shapes, message="No valid snippets"):
        self.result_segment_shapes = result_segment_shapes
        self.message = message
        super().__init__(self.message)



def same_segment_with_kaggle(signal,fs,seq_length=50):
    # middle 50s
    if signal.shape[1] / fs < seq_length:
        return None

    elif signal.shape[1] / fs > seq_length:
        seq_samples = seq_length * fs  # number of sample points corresponding to SEQ_LENGTH

        start_idx = (signal.shape[1] - seq_samples) // 2
        end_idx = start_idx + seq_samples

        return specify_segment_for_continuous_test(signal,start_idx,end_idx)
    else:
        return signal


def specify_segment_for_continuous_test(signal, start, end):
    return  signal[:, int(start):int(end)]



def recursive_files(root_dir,file_type):
    root_path = Path(root_dir)
    return [file.name for file in root_path.rglob(f'*{file_type}')]  # extract file names only

def find_file_path(root_dir, target_filename):
    root_path = Path(root_dir)
    for file in root_path.rglob(target_filename):
        return str(file)  # return full path
    return None  # if file not found, return None


def extract_number(filename):
    numbers = re.findall(r'\d+', filename)  # find all numbers
    return int(numbers[0]) if numbers else float('inf')  # if numbers found, convert to int; otherwise sort to end


def _coerce_indices_to_int_list(labels, C):
    """Normalize multi-label indices to a valid int list, filtering out-of-range/invalid values."""
    # None/NaN: return empty directly
    if labels is None or (isinstance(labels, float) and np.isnan(labels)):
        return []

    # single scalar -> [int]
    if isinstance(labels, (int, np.integer)):
        return [int(labels)] if 0 <= int(labels) < C else []

    # try to treat as iterable
    try:
        arr = np.array(list(labels), dtype=object)
    except TypeError:
        # not iterable, treat as single value
        try:
            v = int(labels)
            return [v] if 0 <= v < C else []
        except Exception:
            return []

    out = []
    for v in arr:
        # skip None/NaN
        if v is None:
            continue
        if isinstance(v, float) and np.isnan(v):
            continue
        try:
            iv = int(v)  # allow convertible types like '3', 3.0
            if 0 <= iv < C:
                out.append(iv)
        except Exception:
            # invalid values (e.g. 'abc') are skipped
            continue
    # deduplicate and sort (optional)
    return sorted(set(out))

def make_one_hot_from_indices(y_true, C):
    """y_true: list-like, each element is the label index for that sample (can be scalar/list/array/Series)"""
    N = len(y_true)
    y_bin = np.zeros((N, C), dtype=int)
    for i, labels in enumerate(y_true):
        idxs = _coerce_indices_to_int_list(labels, C)  # key: ensure int indices
        if idxs:  # may be empty
            y_bin[i, idxs] = 1
    return y_bin




def merge_pairwise_csvs(
    dir_1: str,
    dir_2: str,
    out_dir: str,
    prefix_1: str = "d1_",
    prefix_2: str = "d2_",
    drop_cols : list= None,
    dataset_type: str='',
):
    '''''
    Column-wise merge of same-named CSVs from two directories:
      1) Remove pred_labels column from both sides (if exists)
      2) Add prefix only to columns that appear in both tables:
            dir_1 -> prefix_1
            dir_2 -> prefix_2
      3) Outer join by index (fill with NaN if row counts differ)
      4) Save to out_dir with the same filename as output csv
      5) Optional: whether to delete the second directory (dangerous operation)

    Args:
      dir_1:        first directory
      dir_2:        second directory
      out_dir:      output directory
      prefix_1:     prefix to add to columns in dir_1 that share names with dir_2
      prefix_2:     prefix to add to columns in dir_2 that share names with dir_1
      remove_dir_2: whether to delete the second directory
    
    '''



    os.makedirs(out_dir, exist_ok=True)

    files_1 = {os.path.basename(p): p for p in glob.glob(os.path.join(dir_1, "*.csv"))}
    files_2 = {os.path.basename(p): p for p in glob.glob(os.path.join(dir_2, "*.csv"))}

    common_names = sorted(set(files_1.keys()) & set(files_2.keys()))
    if not common_names:
        print("No same files")
        return

    for name in common_names:
        path_1 = files_1[name]
        path_2 = files_2[name]

        df1 = pd.read_csv(path_1)
        df2 = pd.read_csv(path_2)


        # delete columns
        if drop_cols:
            for df in (df1, df2):
                df.drop(columns=drop_cols, inplace=True, errors="ignore")

        # find duplicate column names
        overlapping_cols = set(df1.columns) & set(df2.columns)

        def sort_key(col):
            parts = col.split('_')
            if len(parts) > 1 and parts[1].isdigit():
                return int(parts[1])
            return float('inf')  # columns without numbers go last

        overlapping_cols = sorted(overlapping_cols, key=sort_key)

        # add prefix to duplicate column names
        if overlapping_cols:
            df1 = df1.rename(columns={c: f"{prefix_1}{c}" for c in overlapping_cols})
            df2 = df2.rename(columns={c: f"{prefix_2}{c}" for c in overlapping_cols})

        # align by row (align to maximum length)
        max_len = max(len(df1), len(df2))
        df1_aligned = df1.reindex(range(max_len)).reset_index(drop=True)
        df2_aligned = df2.reindex(range(max_len)).reset_index(drop=True)

        merged = pd.concat([df1_aligned, df2_aligned], axis=1)

        if dataset_type == 'SPIKE_localization' and overlapping_cols:
            for col in overlapping_cols:
                c1 = f"{prefix_1}{col}"
                c2 = f"{prefix_2}{col}"
                if c1 in merged.columns and c2 in merged.columns:
                    # convert to numeric, non-numeric set to NaN
                    v1 = pd.to_numeric(merged[c1], errors="coerce")
                    v2 = pd.to_numeric(merged[c2], errors="coerce")

                    # conditional selection
                    cond = (v1 >= 0.9) | (v1 <= 0.1)
                    combined = v2.copy()
                    combined.loc[cond] = v1.loc[cond]

                    merged[col] = combined
        cols_for_vote = [c for c in overlapping_cols if c in merged.columns]

        def row_to_labels(row):
            labels = []
            for c in cols_for_vote:
                val = row[c]
                try:
                    v = float(val)
                except (TypeError, ValueError):
                    continue
                if v > 0.5:
                    parts = c.split('_')
                    # take the second part after splitting by '_' (fall back to original name if not found)
                    label = parts[1] if len(parts) > 1 else c
                    labels.append(label)
            return labels

        merged["pred_labels"] = merged.apply(row_to_labels, axis=1)

        out_path = os.path.join(out_dir, name)
        merged.to_csv(out_path, index=False)




def remove_dirs(
    remove_dirs: list = None
):
    if remove_dirs:
        for remove_dir in remove_dirs:
            shutil.rmtree(remove_dir, ignore_errors=True)



