import math
from functools import partial
from typing import List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import drop_path, trunc_normal_
from timm.models import register_model
from einops import rearrange


def _cfg(url: str = '', **kwargs):
    base = dict(
        num_classes=1000,
        input_size=(3, 224, 224),
        pool_size=None,
        crop_pct=0.9,
        interpolation='bicubic',
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
    )
    base.update(kwargs)
    base['url'] = url
    return base


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return drop_path(x, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return f'p={self.drop_prob}'


class Mlp(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer=nn.GELU,
        drop: float = 0.0,
    ):
        super().__init__()
        hidden_features = hidden_features or in_features
        out_features = out_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.fc1(x))
        return self.drop(self.fc2(x))


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm=None,
        qk_scale: Optional[float] = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        window_size=None,
        attn_head_dim: Optional[int] = None,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = attn_head_dim if attn_head_dim is not None else dim // num_heads
        all_head_dim = head_dim * num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, all_head_dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.v_bias = None

        self.q_norm = qk_norm(head_dim) if qk_norm is not None else None
        self.k_norm = qk_norm(head_dim) if qk_norm is not None else None

        if window_size:
            self.window_size = window_size
            n_rel = (2 * window_size[0] - 1) * (2 * window_size[1] - 1) + 3
            self.num_relative_distance = n_rel
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros(n_rel, num_heads))

            h_idx = torch.arange(window_size[0])
            w_idx = torch.arange(window_size[1])
            grid = torch.stack(torch.meshgrid([h_idx, w_idx]))
            grid_flat = torch.flatten(grid, 1)
            rel = grid_flat[:, :, None] - grid_flat[:, None, :]
            rel = rel.permute(1, 2, 0).contiguous()
            rel[:, :, 0] += window_size[0] - 1
            rel[:, :, 1] += window_size[1] - 1
            rel[:, :, 0] *= 2 * window_size[1] - 1
            n_patches = window_size[0] * window_size[1]
            rel_idx = torch.zeros((n_patches + 1,) * 2, dtype=rel.dtype)
            rel_idx[1:, 1:] = rel.sum(-1)
            rel_idx[0, 0:] = n_rel - 3
            rel_idx[0:, 0] = n_rel - 2
            rel_idx[0, 0] = n_rel - 1
            self.register_buffer("relative_position_index", rel_idx)
        else:
            self.window_size = None
            self.relative_position_bias_table = None
            self.relative_position_index = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(
        self,
        x: torch.Tensor,
        rel_pos_bias: Optional[torch.Tensor] = None,
        return_attention: bool = False,
        return_qkv: bool = False,
    ):
        B, N, _ = x.shape

        bias = None
        if self.q_bias is not None:
            bias = torch.cat([
                self.q_bias,
                torch.zeros_like(self.v_bias, requires_grad=False),
                self.v_bias,
            ])

        qkv = F.linear(x, self.qkv.weight, bias)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        if self.q_norm is not None:
            q = self.q_norm(q).type_as(v)
        if self.k_norm is not None:
            k = self.k_norm(k).type_as(v)

        scores = (q * self.scale) @ k.transpose(-2, -1)

        if self.relative_position_bias_table is not None:
            n = self.window_size[0] * self.window_size[1]
            rpb = self.relative_position_bias_table[
                self.relative_position_index.view(-1)
            ].view(n + 1, n + 1, -1).permute(2, 0, 1).contiguous()
            scores = scores + rpb.unsqueeze(0)

        if rel_pos_bias is not None:
            scores = scores + rel_pos_bias

        attn = self.attn_drop(scores.softmax(dim=-1))

        if return_attention:
            return attn

        out = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        out = self.proj_drop(self.proj(out))

        if return_qkv:
            return out, qkv
        return out


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_norm=None,
        qk_scale: Optional[float] = None,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        init_values=None,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        window_size=None,
        attn_head_dim: Optional[int] = None,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            window_size=window_size,
            attn_head_dim=attn_head_dim,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=drop,
        )
        if init_values is not None and init_values > 0:
            self.gamma_1 = nn.Parameter(init_values * torch.ones(dim), requires_grad=True)
            self.gamma_2 = nn.Parameter(init_values * torch.ones(dim), requires_grad=True)
        else:
            self.gamma_1, self.gamma_2 = None, None

    def forward(
        self,
        x: torch.Tensor,
        rel_pos_bias: Optional[torch.Tensor] = None,
        return_attention: bool = False,
        return_qkv: bool = False,
    ):
        normed = self.norm1(x)

        if return_attention:
            return self.attn(normed, rel_pos_bias=rel_pos_bias, return_attention=True)

        if return_qkv:
            attn_out, qkv = self.attn(normed, rel_pos_bias=rel_pos_bias, return_qkv=True)
            x = x + self.drop_path(self.gamma_1 * attn_out)
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
            return x, qkv

        attn_out = self.attn(normed, rel_pos_bias=rel_pos_bias)
        if self.gamma_1 is not None:
            x = x + self.drop_path(self.gamma_1 * attn_out)
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(attn_out)
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


# ---------------------------------------------------------------------------
# Patch / temporal embedding
# ---------------------------------------------------------------------------

class PatchEmbed(nn.Module):
    def __init__(
        self,
        EEG_size: int = 1600,
        patch_size: int = 200,
        in_chans: int = 1,
        embed_dim: int = 200,
    ):
        super().__init__()
        num_patches = 62 * (EEG_size // patch_size)
        self.patch_shape = (1, EEG_size // patch_size)
        self.EEG_size = EEG_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.proj = nn.Conv2d(
            in_chans, embed_dim,
            kernel_size=(1, patch_size),
            stride=(1, patch_size),
        )

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.proj(x).flatten(2).transpose(1, 2)


class TemporalConv(nn.Module):
    def __init__(self, in_chans: int = 1, out_chans: int = 8):
        super().__init__()
        self.conv1 = nn.Conv2d(in_chans, out_chans, kernel_size=(1, 15), stride=(1, 8), padding=(0, 7))
        self.gelu1 = nn.GELU()
        self.norm1 = nn.GroupNorm(4, out_chans)
        self.conv2 = nn.Conv2d(out_chans, out_chans, kernel_size=(1, 3), padding=(0, 1))
        self.gelu2 = nn.GELU()
        self.norm2 = nn.GroupNorm(4, out_chans)
        self.conv3 = nn.Conv2d(out_chans, out_chans, kernel_size=(1, 3), padding=(0, 1))
        self.norm3 = nn.GroupNorm(4, out_chans)
        self.gelu3 = nn.GELU()

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        x = rearrange(x, 'B N A T -> B (N A) T').unsqueeze(1)
        x = self.gelu1(self.norm1(self.conv1(x)))
        x = self.gelu2(self.norm2(self.conv2(x)))
        x = self.gelu3(self.norm3(self.conv3(x)))
        return rearrange(x, 'B C NA T -> B NA (T C)')


# ---------------------------------------------------------------------------
# Main transformer model
# ---------------------------------------------------------------------------

class NeuralTransformer(nn.Module):
    def __init__(
        self,
        EEG_size: int = 1600,
        patch_size: int = 200,
        in_chans: int = 1,
        out_chans: int = 8,
        num_classes: int = 1000,
        embed_dim: int = 200,
        depth: int = 12,
        num_heads: int = 10,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_norm=None,
        qk_scale: Optional[float] = None,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        norm_layer=nn.LayerNorm,
        init_values=None,
        use_abs_pos_emb: bool = True,
        use_rel_pos_bias: bool = False,
        use_shared_rel_pos_bias: bool = False,
        use_mean_pooling: bool = True,
        init_scale: float = 0.001,
        **kwargs,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim

        self.patch_embed = (
            TemporalConv(out_chans=out_chans)
            if in_chans == 1
            else PatchEmbed(
                EEG_size=EEG_size,
                patch_size=patch_size,
                in_chans=in_chans,
                embed_dim=embed_dim,
            )
        )
        self.time_window = EEG_size // patch_size
        self.patch_size = patch_size

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = (
            nn.Parameter(torch.zeros(1, 128 + 1, embed_dim), requires_grad=True)
            if use_abs_pos_emb else None
        )
        self.time_embed = nn.Parameter(torch.zeros(1, 16, embed_dim), requires_grad=True)
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.rel_pos_bias = None

        stoch_depth_rates = [v.item() for v in torch.linspace(0, drop_path_rate, depth)]
        self.use_rel_pos_bias = use_rel_pos_bias
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_norm=qk_norm,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=stoch_depth_rates[i],
                norm_layer=norm_layer,
                init_values=init_values,
                window_size=None,
            )
            for i in range(depth)
        ])
        self.norm = nn.Identity() if use_mean_pooling else norm_layer(embed_dim)
        self.fc_norm = norm_layer(embed_dim) if use_mean_pooling else None
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        if self.pos_embed is not None:
            trunc_normal_(self.pos_embed, std=0.02)
        if self.time_embed is not None:
            trunc_normal_(self.time_embed, std=0.02)
        trunc_normal_(self.cls_token, std=0.02)
        if isinstance(self.head, nn.Linear):
            trunc_normal_(self.head.weight, std=0.02)
        self.apply(self._init_weights)
        self._rescale_proj()
        if isinstance(self.head, nn.Linear):
            self.head.weight.data.mul_(init_scale)
            self.head.bias.data.mul_(init_scale)

    def _rescale_proj(self):
        """Rescale projection weights at initialisation (Rezero / depth-scaled init)."""
        for depth_idx, layer in enumerate(self.blocks):
            scale = math.sqrt(2.0 * (depth_idx + 1))
            layer.attn.proj.weight.data.div_(scale)
            layer.mlp.fc2.weight.data.div_(scale)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def depth(self) -> int:
        """Number of transformer blocks."""
        return len(self.blocks)

    # kept for API compatibility
    def get_num_layers(self) -> int:
        return self.depth()

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'time_embed'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes: int, global_pool: str = ''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(
        self,
        x,
        input_chans=None,
        return_patch_tokens: bool = False,
        return_all_tokens: bool = False,
        **kwargs,
    ):
        B, n_ch, n_win, t = x.shape
        time_steps = n_win if t == self.patch_size else t

        tokens = self.patch_embed(x)
        cls_tok = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls_tok, tokens], dim=1)

        if self.pos_embed is not None:
            pos = self.pos_embed[:, input_chans] if input_chans is not None else self.pos_embed
            spatial_pos = pos[:, 1:].unsqueeze(2).expand(B, -1, time_steps, -1).flatten(1, 2)
            full_pos = torch.cat([pos[:, :1].expand(B, -1, -1), spatial_pos], dim=1)
            tokens = tokens + full_pos

        if self.time_embed is not None:
            n_spatial = n_ch if t == self.patch_size else n_win
            t_emb = self.time_embed[:, :time_steps].unsqueeze(1).expand(B, n_spatial, -1, -1).flatten(1, 2)
            tokens[:, 1:] = tokens[:, 1:] + t_emb

        tokens = self.pos_drop(tokens)
        for blk in self.blocks:
            tokens = blk(tokens, rel_pos_bias=None)
        tokens = self.norm(tokens)

        if self.fc_norm is not None:
            patch_tokens = tokens[:, 1:]
            if return_all_tokens:
                return self.fc_norm(tokens)
            if return_patch_tokens:
                return self.fc_norm(patch_tokens)
            return self.fc_norm(patch_tokens.mean(1))

        if return_all_tokens:
            return tokens
        if return_patch_tokens:
            return tokens[:, 1:]
        return tokens[:, 0]

    def forward(self, x, input_chans=None, return_patch_tokens=False, return_all_tokens=False, **kwargs):
        out = self.forward_features(
            x,
            input_chans=input_chans,
            return_patch_tokens=return_patch_tokens,
            return_all_tokens=return_all_tokens,
            **kwargs,
        )
        return self.head(out)

    def extract_layer(
        self,
        x,
        layer_id: Union[int, List[int]] = 12,
        norm_output: bool = False,
    ):
        """
        Return intermediate token representations at a specific layer or layers.

        layer_id : int  → output after that block's norm1
                   list → patch tokens at each listed layer index
        """
        tokens = self.patch_embed(x)
        B = tokens.shape[0]
        cls_tok = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls_tok, tokens], dim=1)

        if self.pos_embed is not None:
            sp = self.pos_embed[:, 1:].unsqueeze(2).expand(B, -1, self.time_window, -1).flatten(1, 2)
            sp = torch.cat([self.pos_embed[:, :1].expand(B, -1, -1), sp], dim=1)
            tokens = tokens + sp
        if self.time_embed is not None:
            te = self.time_embed.unsqueeze(1).expand(B, 62, -1, -1).flatten(1, 2)
            tokens[:, 1:] = tokens[:, 1:] + te
        tokens = self.pos_drop(tokens)

        rpb = self.rel_pos_bias() if self.rel_pos_bias is not None else None

        if isinstance(layer_id, list):
            collected = []
            for i, blk in enumerate(self.blocks):
                tokens = blk(tokens, rel_pos_bias=rpb)
                if i in layer_id:
                    out = (
                        self.fc_norm(self.norm(tokens[:, 1:]))
                        if norm_output else tokens[:, 1:]
                    )
                    collected.append(out)
            return collected

        for i, blk in enumerate(self.blocks):
            if i < layer_id:
                tokens = blk(tokens, rel_pos_bias=rpb)
            elif i == layer_id:
                tokens = blk.norm1(tokens)
                break
        return tokens[:, 1:]

    # kept for API compatibility
    def forward_intermediate(self, x, layer_id=12, norm_output=False):
        return self.extract_layer(x, layer_id=layer_id, norm_output=norm_output)

    def all_hidden_states(self, x, use_last_norm: bool = False) -> List[torch.Tensor]:
        """Return hidden states after every transformer block."""
        tokens = self.patch_embed(x)
        B = tokens.shape[0]
        cls_tok = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls_tok, tokens], dim=1)

        if self.pos_embed is not None:
            sp = self.pos_embed[:, 1:].unsqueeze(2).expand(B, -1, self.time_window, -1).flatten(1, 2)
            sp = torch.cat([self.pos_embed[:, :1].expand(B, -1, -1), sp], dim=1)
            tokens = tokens + sp
        if self.time_embed is not None:
            te = self.time_embed.unsqueeze(1).expand(B, 62, -1, -1).flatten(1, 2)
            tokens[:, 1:] = tokens[:, 1:] + te
        tokens = self.pos_drop(tokens)

        rpb = self.rel_pos_bias() if self.rel_pos_bias is not None else None
        layer_outs = []
        for blk in self.blocks:
            tokens = blk(tokens, rpb)
            layer_outs.append(self.norm(tokens) if use_last_norm else tokens)
        return layer_outs

    # kept for API compatibility
    def get_intermediate_layers(self, x, use_last_norm: bool = False):
        return self.all_hidden_states(x, use_last_norm=use_last_norm)


# ---------------------------------------------------------------------------
# Masked EEG modeling
# ---------------------------------------------------------------------------

class NeuralTransformerForMaskedEEGModeling(nn.Module):
    def __init__(
        self,
        EEG_size=1600, patch_size=200, in_chans=1, out_chans=8,
        vocab_size=8192, embed_dim=200, depth=12, num_heads=12,
        mlp_ratio=4., qkv_bias=True, qk_norm=None, qk_scale=None,
        drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
        norm_layer=None, init_values=None, attn_head_dim=None,
        use_abs_pos_emb=True, use_rel_pos_bias=False,
        use_shared_rel_pos_bias=False, init_std=0.02,
    ):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim

        self.patch_embed = TemporalConv(out_chans=out_chans)
        self.num_heads = num_heads
        self.patch_size = patch_size

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = (
            nn.Parameter(torch.zeros(1, 128 + 1, embed_dim))
            if use_abs_pos_emb else None
        )
        self.time_embed = nn.Parameter(torch.zeros(1, 16, embed_dim), requires_grad=True)
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.rel_pos_bias = None

        stoch_depth = [v.item() for v in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_norm=qk_norm, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=stoch_depth[i], norm_layer=norm_layer,
                init_values=init_values, window_size=None,
                attn_head_dim=attn_head_dim,
            )
            for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim)

        self.init_std = init_std
        self.lm_head = nn.Linear(embed_dim, vocab_size)

        if self.pos_embed is not None:
            trunc_normal_(self.pos_embed, std=self.init_std)
        trunc_normal_(self.time_embed, std=self.init_std)
        trunc_normal_(self.cls_token, std=self.init_std)
        trunc_normal_(self.mask_token, std=self.init_std)
        trunc_normal_(self.lm_head.weight, std=self.init_std)
        self.apply(self._init_weights)
        self._rescale_proj()

    def _rescale_proj(self):
        """Depth-scaled initialisation for attention projection and MLP output weights."""
        for depth_idx, layer in enumerate(self.blocks):
            scale = math.sqrt(2.0 * (depth_idx + 1))
            layer.attn.proj.weight.data.div_(scale)
            layer.mlp.fc2.weight.data.div_(scale)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'time_embed'}

    def get_num_layers(self) -> int:
        return len(self.blocks)

    def forward_features(self, x, input_chans, bool_masked_pos):
        B, n_ch, n_win, _ = x.size()
        seq = self.patch_embed(x)
        seq_len = seq.shape[1]

        # replace masked positions with learnable mask token
        mask_w = bool_masked_pos.unsqueeze(-1).type_as(seq)
        seq = seq * (1 - mask_w) + self.mask_token.expand(B, seq_len, -1) * mask_w

        cls_tok = self.cls_token.expand(B, -1, -1)
        seq = torch.cat([cls_tok, seq], dim=1)

        if self.pos_embed is not None:
            pos = self.pos_embed[:, input_chans] if input_chans is not None else self.pos_embed
            spatial_pos = pos[:, 1:].unsqueeze(2).expand(B, -1, n_win, -1).flatten(1, 2)
            full_pos = torch.cat([spatial_pos[:, :1].expand(B, -1, -1), spatial_pos], dim=1)
            seq = seq + full_pos

        if self.time_embed is not None:
            t_emb = self.time_embed[:, :n_win].unsqueeze(1).expand(B, n_ch, -1, -1).flatten(1, 2)
            seq[:, 1:] = seq[:, 1:] + t_emb

        seq = self.pos_drop(seq)
        rpb = self.rel_pos_bias() if self.rel_pos_bias is not None else None
        for blk in self.blocks:
            seq = blk(seq, rel_pos_bias=rpb)
        return self.norm(seq)

    def forward(
        self,
        x,
        input_chans=None,
        bool_masked_pos=None,
        return_all_tokens: bool = False,
        return_patch_tokens: bool = False,
        return_all_patch_tokens: bool = False,
    ):
        if bool_masked_pos is None:
            bool_masked_pos = torch.zeros(
                (x.shape[0], x.shape[1] * x.shape[2]),
                dtype=torch.bool, device=x.device,
            )
        hidden = self.forward_features(x, input_chans=input_chans, bool_masked_pos=bool_masked_pos)
        if return_all_patch_tokens:
            return hidden
        patch_tokens = hidden[:, 1:]
        if return_patch_tokens:
            return patch_tokens
        if return_all_tokens:
            return self.lm_head(patch_tokens)
        return self.lm_head(patch_tokens[bool_masked_pos])

    def forward_return_qkv(self, x, bool_masked_pos=None, split_out_as_qkv=False):
        if bool_masked_pos is None:
            bool_masked_pos = torch.zeros(
                (x.shape[0], x.shape[1] * x.shape[2]),
                dtype=torch.bool, device=x.device,
            )
        hidden = self.patch_embed(x, bool_masked_pos=bool_masked_pos)
        B, L, _ = hidden.size()

        w = bool_masked_pos.unsqueeze(-1).type_as(hidden)
        hidden = hidden * (1 - w) + self.mask_token.expand(B, L, -1) * w

        cls = self.cls_token.expand(B, -1, -1)
        hidden = torch.cat([cls, hidden], dim=1)
        if self.pos_embed is not None:
            hidden = hidden + self.pos_embed
        hidden = self.pos_drop(hidden)

        rel_pos_bias = self.rel_pos_bias() if self.rel_pos_bias is not None else None
        for i, blk in enumerate(self.blocks):
            if i < len(self.blocks) - 1:
                hidden = blk(hidden, rel_pos_bias=rel_pos_bias)
            else:
                hidden, qkv = blk(hidden, rel_pos_bias=rel_pos_bias, return_qkv=True)

        if split_out_as_qkv:
            hidden = self.norm(hidden)
            hidden = self.lm_head(hidden)
            q, k, v = hidden.chunk(3, dim=-1)
            b, n, c = q.shape
            q = q.reshape(b, n, self.num_heads, -1).permute(0, 2, 1, 3)
            k = k.reshape(b, n, self.num_heads, -1).permute(0, 2, 1, 3)
            v = v.reshape(b, n, self.num_heads, -1).permute(0, 2, 1, 3)
            return hidden, q, k, v

        hidden = self.norm(hidden)
        out = self.lm_head(hidden[:, 1:][bool_masked_pos])
        q, k, v = qkv[0], qkv[1], qkv[2]
        return out, q, k, v

    def get_last_selfattention(self, x):
        hidden = self.patch_embed(x)
        B, _, _ = hidden.size()
        cls = self.cls_token.expand(B, -1, -1)
        hidden = torch.cat([cls, hidden], dim=1)
        if self.pos_embed is not None:
            hidden = hidden + self.pos_embed
        hidden = self.pos_drop(hidden)
        rel_pos_bias = self.rel_pos_bias() if self.rel_pos_bias is not None else None
        for i, blk in enumerate(self.blocks):
            if i < len(self.blocks) - 1:
                hidden = blk(hidden, rel_pos_bias=rel_pos_bias)
            else:
                return blk(hidden, rel_pos_bias=rel_pos_bias, return_attention=True)


class NeuralTransformerForMEM(nn.Module):
    def __init__(
        self,
        EEG_size=1600, patch_size=200, in_chans=1, out_chans=8,
        vocab_size=8192, embed_dim=200, depth=12, num_heads=10,
        mlp_ratio=4., qkv_bias=True, qk_norm=None, qk_scale=None,
        drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
        norm_layer=None, init_values=None, attn_head_dim=None,
        use_abs_pos_emb=True, use_rel_pos_bias=False,
        use_shared_rel_pos_bias=False, init_std=0.02, **kwargs,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.student = NeuralTransformerForMaskedEEGModeling(
            EEG_size, patch_size, in_chans, out_chans, vocab_size,
            embed_dim, depth, num_heads, mlp_ratio, qkv_bias, qk_norm,
            qk_scale, drop_rate, attn_drop_rate, drop_path_rate,
            norm_layer, init_values, attn_head_dim,
            use_abs_pos_emb, use_rel_pos_bias, use_shared_rel_pos_bias, init_std,
        )
        self.lm_head = nn.Linear(embed_dim, vocab_size)
        self.projection_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
        )
        trunc_normal_(self.lm_head.weight, std=init_std)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'student.cls_token', 'student.pos_embed', 'student.time_embed'}

    def forward(self, x, input_chans=None, bool_masked_pos=None):
        all_tokens = self.student(x, input_chans, bool_masked_pos, return_all_patch_tokens=True)
        x_rec = self.lm_head(all_tokens[:, 1:][bool_masked_pos])

        inv_mask = ~bool_masked_pos
        all_tokens_sym = self.student(x, input_chans, inv_mask, return_all_patch_tokens=True)
        x_rec_sym = self.lm_head(all_tokens_sym[:, 1:][inv_mask])

        return x_rec, x_rec_sym


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

@register_model
def base_patch200_200(pretrained=False, **kwargs):
    model = NeuralTransformer(
        patch_size=200, embed_dim=200, depth=12, num_heads=10, mlp_ratio=4,
        qk_norm=partial(nn.LayerNorm, eps=1e-6),
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    model.default_cfg = _cfg()
    return model


@register_model
def large_patch200_200(pretrained=False, **kwargs):
    model = NeuralTransformer(
        patch_size=200, embed_dim=400, depth=24, num_heads=16, mlp_ratio=4,
        out_chans=16, qk_norm=partial(nn.LayerNorm, eps=1e-6),
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    model.default_cfg = _cfg()
    return model


@register_model
def huge_patch200_200(pretrained=False, **kwargs):
    model = NeuralTransformer(
        patch_size=200, embed_dim=800, depth=48, num_heads=16, mlp_ratio=4,
        out_chans=32, qk_norm=partial(nn.LayerNorm, eps=1e-6),
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    model.default_cfg = _cfg()
    return model


@register_model
def base_patch200_1600_8k_vocab(pretrained=False, **kwargs):
    kwargs.pop("num_classes", None)
    vocab_size = kwargs.pop("vocab_size", 8192)
    model = NeuralTransformerForMEM(
        patch_size=200, embed_dim=200, depth=12, num_heads=10, mlp_ratio=4,
        qkv_bias=False, qk_norm=partial(nn.LayerNorm, eps=1e-6),
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        vocab_size=vocab_size, **kwargs,
    )
    model.default_cfg = _cfg()
    if pretrained:
        ckpt = torch.load(kwargs["init_ckpt"], map_location="cpu")
        model.load_state_dict(ckpt["model"])
    return model


@register_model
def large_patch200_1600_8k_vocab(pretrained=False, **kwargs):
    kwargs.pop("num_classes", None)
    vocab_size = kwargs.pop("vocab_size", 8192)
    model = NeuralTransformerForMEM(
        patch_size=200, embed_dim=400, depth=24, num_heads=16, mlp_ratio=4,
        qkv_bias=False, out_chans=16, qk_norm=partial(nn.LayerNorm, eps=1e-6),
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        vocab_size=vocab_size, **kwargs,
    )
    model.default_cfg = _cfg()
    if pretrained:
        ckpt = torch.load(kwargs["init_ckpt"], map_location="cpu")
        model.load_state_dict(ckpt["model"])
    return model


@register_model
def huge_patch200_1600_8k_vocab(pretrained=False, **kwargs):
    kwargs.pop("num_classes", None)
    vocab_size = kwargs.pop("vocab_size", 8192)
    model = NeuralTransformerForMEM(
        patch_size=200, embed_dim=800, depth=48, num_heads=16, mlp_ratio=4,
        qkv_bias=False, out_chans=32, qk_norm=partial(nn.LayerNorm, eps=1e-6),
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        vocab_size=vocab_size, **kwargs,
    )
    model.default_cfg = _cfg()
    if pretrained:
        ckpt = torch.load(kwargs["init_ckpt"], map_location="cpu")
        model.load_state_dict(ckpt["model"])
    return model
