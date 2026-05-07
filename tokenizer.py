import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from timm.layers import trunc_normal_
from timm.models import register_model
import torch.distributed as dist
from einops import rearrange, repeat
from backbone import NeuralTransformer


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def _l2_normalize(t: torch.Tensor) -> torch.Tensor:
    """Unit-normalise along the last dimension."""
    return F.normalize(t, p=2, dim=-1)


def _ema_step(buf: torch.Tensor, new_val: torch.Tensor, decay: float) -> None:
    """In-place exponential moving average update."""
    buf.data.mul_(decay).add_(new_val, alpha=1.0 - decay)


def _ema_step_normed(buf: torch.Tensor, new_val: torch.Tensor, decay: float) -> None:
    """EMA update followed by L2 renormalisation — used for the codebook."""
    buf.data.mul_(decay).add_(new_val, alpha=1.0 - decay)
    buf.data.copy_(_l2_normalize(buf.data))


def _reservoir_sample(data: torch.Tensor, n: int) -> torch.Tensor:
    """Draw n rows from data without replacement (or with replacement if N < n)."""
    N = data.shape[0]
    idx = (
        torch.randperm(N, device=data.device)[:n]
        if N >= n
        else torch.randint(0, N, (n,), device=data.device)
    )
    return data[idx]


def _codebook_kmeans(
    samples: torch.Tensor,
    k: int,
    n_iters: int = 10,
    cosine: bool = False,
) -> tuple:
    """
    Simple k-means / spherical k-means on a flat sample matrix.
    Returns (cluster_centres, cluster_counts).
    """
    D, dtype, device = samples.shape[-1], samples.dtype, samples.device
    centres = _reservoir_sample(samples, k)

    for _ in range(n_iters):
        if cosine:
            affinities = samples @ centres.t()
        else:
            affinities = -(
                (rearrange(samples, 'n d -> n () d') - rearrange(centres, 'c d -> () c d')) ** 2
            ).sum(-1)
        nearest = affinities.max(-1).indices
        bin_counts = torch.bincount(nearest, minlength=k)
        safe_counts = bin_counts.masked_fill(bin_counts == 0, 1)

        new_centres = centres.new_zeros(k, D, dtype=dtype)
        new_centres.scatter_add_(0, repeat(nearest, 'n -> n d', d=D), samples)
        new_centres = new_centres / safe_counts.unsqueeze(-1)
        if cosine:
            new_centres = _l2_normalize(new_centres)
        centres = torch.where((bin_counts == 0)[..., None], centres, new_centres)

    return centres, bin_counts


# ---------------------------------------------------------------------------
# Codebook with EMA updates
# ---------------------------------------------------------------------------

class EmbeddingEMA(nn.Module):
    def __init__(
        self,
        num_tokens: int,
        codebook_dim: int,
        decay: float = 0.99,
        eps: float = 1e-5,
        kmeans_init: bool = True,
        codebook_init_path: str = '',
    ):
        super().__init__()
        self.num_tokens = num_tokens
        self.codebook_dim = codebook_dim
        self.decay = decay
        self.eps = eps

        if codebook_init_path:
            print(f"load init codebook weight from {codebook_init_path}")
            init_w = torch.load(codebook_init_path, map_location='cpu').clone()
            self.register_buffer('initted', torch.ones(1))
        else:
            init_w = (
                torch.zeros(num_tokens, codebook_dim)
                if kmeans_init
                else _l2_normalize(torch.randn(num_tokens, codebook_dim))
            )
            self.register_buffer('initted', torch.Tensor([not kmeans_init]))

        self.weight = nn.Parameter(init_w, requires_grad=False)
        self.cluster_size = nn.Parameter(torch.zeros(num_tokens), requires_grad=False)
        self.embed_avg = nn.Parameter(init_w.clone(), requires_grad=False)
        self.update = True

    @torch.jit.ignore
    def init_embed_(self, data: torch.Tensor):
        if self.initted:
            return
        print("Performing K-means init for codebook")
        centers, counts = _codebook_kmeans(data, self.num_tokens, 10, cosine=True)
        self.weight.data.copy_(centers)
        self.cluster_size.data.copy_(counts)
        self.initted.data.fill_(1.0)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return F.embedding(ids, self.weight)

    def cluster_size_ema_update(self, new_cluster_size: torch.Tensor):
        self.cluster_size.data.mul_(self.decay).add_(new_cluster_size, alpha=1 - self.decay)

    def embed_avg_ema_update(self, new_embed_avg: torch.Tensor):
        self.embed_avg.data.mul_(self.decay).add_(new_embed_avg, alpha=1 - self.decay)

    def weight_update(self, num_tokens: int):
        total = self.cluster_size.sum()
        smoothed = (self.cluster_size + self.eps) / (total + num_tokens * self.eps) * total
        self.weight.data.copy_(self.embed_avg / smoothed.unsqueeze(1))


# ---------------------------------------------------------------------------
# Vector quantizer
# ---------------------------------------------------------------------------

class NormEMAVectorQuantizer(nn.Module):
    def __init__(
        self,
        n_embed: int,
        embedding_dim: int,
        beta: float,
        decay: float = 0.99,
        eps: float = 1e-5,
        statistic_code_usage: bool = True,
        kmeans_init: bool = False,
        codebook_init_path: str = '',
    ):
        super().__init__()
        self.codebook_dim = embedding_dim
        self.num_tokens = n_embed
        self.beta = beta
        self.decay = decay

        self.embedding = EmbeddingEMA(
            n_embed, embedding_dim, decay, eps, kmeans_init, codebook_init_path,
        )
        self.statistic_code_usage = statistic_code_usage
        if statistic_code_usage:
            self.register_buffer('cluster_size', torch.zeros(n_embed))

        if dist.is_available() and dist.is_initialized():
            print("ddp is enable, so use ddp_reduce to sync the statistic_code_usage for each gpu!")
            self.all_reduce_fn = dist.all_reduce
        else:
            self.all_reduce_fn = nn.Identity()

    def reset_cluster_size(self, device: torch.device):
        if self.statistic_code_usage:
            self.cluster_size = torch.zeros(self.num_tokens, device=device)

    def forward(self, z: torch.Tensor):
        z = _l2_normalize(rearrange(z, 'b c h w -> b h w c'))
        z_flat = z.reshape(-1, self.codebook_dim)
        self.embedding.init_embed_(z_flat)

        dists = torch.cdist(z_flat, self.embedding.weight, p=2).pow(2)
        indices = dists.argmin(dim=1)
        z_q = self.embedding(indices).view(z.shape)

        one_hot = F.one_hot(indices, self.num_tokens).type(z.dtype)

        if not self.training:
            with torch.no_grad():
                usage = one_hot.sum(0)
                self.all_reduce_fn(usage)
                _ema_step(self.cluster_size, usage, self.decay)

        if self.training and self.embedding.update:
            bin_counts = one_hot.sum(0)
            self.all_reduce_fn(bin_counts)
            _ema_step(self.cluster_size, bin_counts, self.decay)

            safe_counts = bin_counts.masked_fill(bin_counts == 0, 1.0)
            new_embed = _l2_normalize((z_flat.t() @ one_hot / safe_counts.unsqueeze(0)).t())
            merged = torch.where((bin_counts == 0)[..., None], self.embedding.weight, new_embed)
            _ema_step_normed(self.embedding.weight, merged, self.decay)

        commit_loss = self.beta * F.mse_loss(z_q.detach(), z)
        z_q = z + (z_q - z).detach()
        return rearrange(z_q, 'b h w c -> b c h w'), commit_loss, indices


# ---------------------------------------------------------------------------
# VQ-NSP model
# ---------------------------------------------------------------------------

class VQNSP(nn.Module):
    def __init__(
        self,
        encoder_config: dict,
        decoder_config: dict,
        n_embed: int = 8192,
        embed_dim: int = 32,
        decay: float = 0.99,
        quantize_kmeans_init: bool = True,
        decoder_out_dim: int = 200,
        smooth_l1_loss: bool = False,
        **kwargs,
    ):
        super().__init__()
        print(kwargs)
        if decoder_config['in_chans'] != embed_dim:
            print(f"Rewrite the in_chans in decoder from {decoder_config['in_chans']} to {embed_dim}")
            decoder_config['in_chans'] = embed_dim

        print('Final encoder config', encoder_config)
        self.encoder = NeuralTransformer(**encoder_config)

        print('Final decoder config', decoder_config)
        self.decoder = NeuralTransformer(**decoder_config)

        self.quantize = NormEMAVectorQuantizer(
            n_embed=n_embed,
            embedding_dim=embed_dim,
            beta=1.0,
            kmeans_init=quantize_kmeans_init,
            decay=decay,
        )

        self.patch_size = encoder_config['patch_size']
        self.token_shape = (62, encoder_config['EEG_size'] // self.patch_size)
        self.decoder_out_dim = decoder_out_dim

        enc_dim = encoder_config['embed_dim']
        dec_dim = decoder_config['embed_dim']

        self.encode_task_layer = nn.Sequential(
            nn.Linear(enc_dim, enc_dim),
            nn.Tanh(),
            nn.Linear(enc_dim, embed_dim),
        )
        self.decode_task_layer = nn.Sequential(
            nn.Linear(dec_dim, dec_dim),
            nn.Tanh(),
            nn.Linear(dec_dim, decoder_out_dim),
        )
        self.decode_task_layer_angle = nn.Sequential(
            nn.Linear(dec_dim, dec_dim),
            nn.Tanh(),
            nn.Linear(dec_dim, decoder_out_dim),
        )
        # signal reconstruction head: predicts raw time-domain signal per patch
        self.signal_rec_head = nn.Sequential(
            nn.Linear(dec_dim, dec_dim),
            nn.Tanh(),
            nn.Linear(dec_dim, decoder_out_dim),
        )
        self.contrastive_proj = nn.Sequential(
            nn.Linear(enc_dim, enc_dim),
            nn.GELU(),
            nn.Linear(enc_dim, 128),
        )

        for head in [
            self.encode_task_layer,
            self.decode_task_layer,
            self.decode_task_layer_angle,
            self.signal_rec_head,
            self.contrastive_proj,
        ]:
            head.apply(self._init_weights)

        self.loss_fn = F.smooth_l1_loss if smooth_l1_loss else F.mse_loss
        self.kwargs = kwargs

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {
            'quantize.embedding.weight',
            'decoder.cls_token', 'decoder.pos_embed', 'decoder.time_embed',
            'encoder.cls_token', 'encoder.pos_embed', 'encoder.time_embed',
        }

    @property
    def device(self):
        return self.decoder.cls_token.device

    def get_number_of_tokens(self):
        return self.quantize.n_e

    def get_tokens(self, data, input_chans=None, **kwargs):
        q, ids, _ = self.encode(data, input_chans=input_chans)
        return {
            'token': ids.view(data.shape[0], -1),
            'input_img': data,
            'quantize': rearrange(q, 'b d a c -> b (a c) d'),
        }

    def encode(self, x, input_chans=None):
        B, n, a, t = x.shape
        feat = self.encoder(x, input_chans, return_patch_tokens=True)
        with torch.cuda.amp.autocast(enabled=False):
            feat_q = self.encode_task_layer(feat.type_as(self.encode_task_layer[-1].weight))
        h, w = n, feat_q.shape[1] // n
        feat_q = rearrange(feat_q, 'b (h w) c -> b c h w', h=h, w=w)
        q, loss, ids = self.quantize(feat_q)
        return q, ids, loss

    def decode(self, quantize, input_chans=None, **kwargs):
        dec_feat = self.decoder(quantize, input_chans, return_patch_tokens=True)
        return (
            self.decode_task_layer(dec_feat),
            self.decode_task_layer_angle(dec_feat),
            self.signal_rec_head(dec_feat),
        )

    def get_codebook_indices(self, x, input_chans=None, **kwargs):
        return self.get_tokens(x, input_chans, **kwargs)['token']

    def _spectral_recon_loss(self, rec, target):
        return self.loss_fn(rec, rearrange(target, 'b n a c -> b (n a) c'))

    def _patch_normalize(self, x: torch.Tensor) -> torch.Tensor:
        mu = x.mean(dim=(1, 2, 3), keepdim=True)
        sigma = x.std(dim=(1, 2, 3), keepdim=True)
        return (x - mu) / sigma

    def _contrastive_embed(self, x, input_chans=None):
        feat = self.encoder(x, input_chans, return_patch_tokens=True)
        return _l2_normalize(self.contrastive_proj(feat.mean(dim=1)))

    @staticmethod
    def nt_xent(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.1):
        B = z1.shape[0]
        all_z = torch.cat([z1, z2], dim=0)
        sim = torch.mm(all_z, all_z.t()) / temperature
        sim.masked_fill_(torch.eye(2 * B, dtype=torch.bool, device=z1.device), float('-inf'))
        targets = torch.cat([
            torch.arange(B, 2 * B, device=z1.device),
            torch.arange(B, device=z1.device),
        ])
        return F.cross_entropy(sim, targets)

    def forward(
        self,
        x,
        x_aug=None,
        input_chans=None,
        contrastive_weight: float = 1.0,
        contrastive_temperature: float = 0.1,
        signal_rec_weight: float = 0.0,
        **kwargs,
    ):
        x = rearrange(x, 'B N (A T) -> B N A T', T=200)

        spectrum = torch.fft.fft(x, dim=-1)
        amp_target = self._patch_normalize(torch.abs(spectrum))
        phase_target = self._patch_normalize(torch.angle(spectrum))

        z_q, ids, vq_loss = self.encode(x, input_chans)
        amp_pred, phase_pred, signal_pred = self.decode(z_q, input_chans)

        loss_amp = self._spectral_recon_loss(amp_pred, amp_target)
        loss_phase = self._spectral_recon_loss(phase_pred, phase_target)
        total = vq_loss + loss_amp + loss_phase

        split = "train" if self.training else "val"
        log = {
            f'{split}/quant_loss': vq_loss.detach().mean(),
            f'{split}/rec_loss': loss_amp.detach().mean(),
            f'{split}/rec_angle_loss': loss_phase.detach().mean(),
        }

        if signal_rec_weight > 0.0:
            sig_target = self._patch_normalize(x)
            loss_signal = self._spectral_recon_loss(signal_pred, sig_target)
            total = total + signal_rec_weight * loss_signal
            log[f'{split}/signal_rec_loss'] = loss_signal.detach().mean()

        if x_aug is not None:
            x_aug = rearrange(x_aug, 'B N (A T) -> B N A T', T=200)
            with torch.cuda.amp.autocast(enabled=False):
                z1 = self._contrastive_embed(
                    x.type_as(self.contrastive_proj[-1].weight), input_chans)
                z2 = self._contrastive_embed(
                    x_aug.type_as(self.contrastive_proj[-1].weight), input_chans)
            cont = self.nt_xent(z1, z2, contrastive_temperature)
            total = total + contrastive_weight * cont
            log[f'{split}/contrastive_loss'] = cont.detach().mean()

        log[f'{split}/total_loss'] = total.detach().mean()
        return total, log


# ---------------------------------------------------------------------------
# Helper: default encoder/decoder config
# ---------------------------------------------------------------------------

def _base_arch_cfg():
    return dict(
        EEG_size=1600, patch_size=200, in_chans=1, num_classes=1000,
        embed_dim=200, depth=12, num_heads=10, mlp_ratio=4.0,
        qkv_bias=True, qk_scale=None, drop_rate=0.0, attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        init_values=0.0, use_abs_pos_emb=True, use_rel_pos_bias=False,
        use_shared_rel_pos_bias=False, use_mean_pooling=True, init_scale=0.001,
    )


def _load_pretrained(model: nn.Module, weight_path: str):
    if weight_path.startswith('https'):
        state = torch.hub.load_state_dict_from_url(
            weight_path, map_location='cpu', check_hash=True)
    else:
        state = torch.load(weight_path, map_location='cpu')
    state = state.get('model', state.get('state_dict', state))
    drop_keys = [k for k in list(state.keys())
                 if k.startswith(('loss', 'teacher', 'scaling'))]
    for k in drop_keys:
        del state[k]
    model.load_state_dict(state, strict=False)


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

@register_model
def vqnsp_encoder_base_decoder_3x200x12(
    pretrained=False, pretrained_weight=None, as_tokenzer=False,
    EEG_size=1600, n_code=8192, code_dim=32, **kwargs
):
    enc_cfg = _base_arch_cfg()
    dec_cfg = _base_arch_cfg()

    enc_cfg['EEG_size'] = EEG_size
    enc_cfg['num_classes'] = 0

    dec_cfg['EEG_size'] = EEG_size // dec_cfg['patch_size']
    dec_cfg['patch_size'] = 1
    dec_cfg['in_chans'] = code_dim
    dec_cfg['num_classes'] = 0
    dec_cfg['depth'] = 3

    model = VQNSP(enc_cfg, dec_cfg, n_code, code_dim, decoder_out_dim=200, **kwargs)

    if as_tokenzer:
        assert pretrained and pretrained_weight is not None
        _load_pretrained(model, pretrained_weight)
    return model


@register_model
def vqnsp_encoder_large_decoder_3x200x24(
    pretrained=False, pretrained_weight=None, as_tokenzer=False,
    EEG_size=1600, n_code=8192, code_dim=32, **kwargs
):
    enc_cfg = _base_arch_cfg()
    dec_cfg = _base_arch_cfg()

    enc_cfg['EEG_size'] = EEG_size
    enc_cfg['num_classes'] = 0
    enc_cfg['depth'] = 24

    dec_cfg['EEG_size'] = EEG_size // dec_cfg['patch_size']
    dec_cfg['patch_size'] = 1
    dec_cfg['in_chans'] = code_dim
    dec_cfg['num_classes'] = 0
    dec_cfg['depth'] = 3

    model = VQNSP(enc_cfg, dec_cfg, n_code, code_dim, decoder_out_dim=200, **kwargs)

    if as_tokenzer:
        assert pretrained and pretrained_weight is not None
        _load_pretrained(model, pretrained_weight)
    return model


if __name__ == '__main__':
    pass
