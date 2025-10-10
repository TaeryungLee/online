from dataclasses import dataclass
from typing import Optional
import math 
from models.resnet import CausalResnet1D

import torch
import torch.nn as nn
import torch.nn.functional as F



class Conv1D_MLP_causal(nn.Module):
    def __init__(self, d_in, d_out, mlp_hidden_times, resid_pdrop=0.1):
        super().__init__()
        self.conv1 = CausalConv1d(in_channels=d_in, out_channels=int(mlp_hidden_times * d_in), kernel_size=3, stride=1)
        self.act = nn.GELU()
        self.conv2 = CausalConv1d(in_channels=int(mlp_hidden_times * d_in), out_channels=d_out, kernel_size=3, stride=1)
        self.dropout = nn.Dropout(resid_pdrop)

    def forward(self, x):
        B, T, D = x.shape
        # x = rearrange(x, 'b t d -> b d t')
        x = x.transpose(1, 2)
        x = self.conv2(self.act(self.conv1(x)))
        x = x.transpose(1, 2)
        # x = rearrange(x, 'b d t -> b t d')
        return self.dropout(x)

class Conv1D_MLP_causal_single(nn.Module):
    def __init__(self, d_in, d_out, mlp_hidden_times, resid_pdrop=0.1):
        super().__init__()
        self.conv1 = CausalConv1d(in_channels=d_in, out_channels=d_out, kernel_size=3, stride=1)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(resid_pdrop)

    def forward(self, x):
        B, T, D = x.shape
        # x = rearrange(x, 'b t d -> b d t')
        x = x.transpose(1, 2)
        x = self.act(self.conv1(x))
        x = x.transpose(1, 2)
        # x = rearrange(x, 'b d t -> b t d')
        return self.dropout(x)
    

class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1):
        super(CausalConv1d, self).__init__()
        self.pad = (kernel_size - 1) * dilation + (1 - stride)         
        self.conv = nn.Conv1d(
            in_channels, 
            out_channels, 
            kernel_size,                        
            stride=stride, 
            padding=0,                          # no padding here
            dilation=dilation
        )

    def forward(self, x):
        x = nn.functional.pad(x, (self.pad, 0))  # only pad on the left
        return self.conv(x)
    

class ConvCausalEncoder(nn.Module):
    def __init__(self,
                 input_emb_width = 272,
                 hidden_size = 1024,
                 down_t = 2,
                 stride_t = 2,
                 width = 1024,
                 depth = 3,
                 dilation_growth_rate = 3,
                 activation='relu',
                 norm=None,
                 latent_dim=16,
                 clip_range = [-30,20]
                 ):
        super().__init__()
        self.clip_range = clip_range
        self.proj = nn.Linear(width, latent_dim*2)

        blocks = []
        filter_t, pad_t = stride_t * 2, stride_t // 2      


        blocks.append(CausalConv1d(input_emb_width, width, 3, 1, 1))
        blocks.append(nn.ReLU())
        
        for i in range(down_t):   
            input_dim = width
            block = nn.Sequential(
                CausalConv1d(input_dim, width, filter_t, stride_t, 1),
                CausalResnet1D(width, depth, dilation_growth_rate, activation=activation, norm=norm),
            )
            blocks.append(block)
        blocks.append(CausalConv1d(width, hidden_size, 3, 1, 1))
        self.model = nn.Sequential(*blocks)


    def forward(self, x):
        x = self.model(x)
        x = x.transpose(1, 2)  
        x = self.proj(x)        
        mu, logvar = x.chunk(2, dim=2)             
        logvar = torch.clamp(logvar, self.clip_range[0], self.clip_range[1])
        # z = self.reparameterize(mu, logvar) 

        # return z, mu, logvar
        return mu, logvar

class ConvCausalEncoder_NoComp(nn.Module):
    def __init__(self,
                 input_emb_width = 272,
                 hidden_size = 1024,
                 down_t = 2,
                 width = 1024,
                 depth = 3,
                 dilation_growth_rate = 3,
                 activation='relu',
                 norm=None,
                 latent_dim=16,
                 clip_range = [-30,20]
                 ):
        super().__init__()
        self.clip_range = clip_range

        blocks = []
        stride_t = 1
        filter_t, pad_t = stride_t * 2, stride_t // 2

        self.in_proj = nn.Linear(input_emb_width, width)


        blocks.append(CausalConv1d(width, width, 3, 1, 1))
        blocks.append(nn.ReLU())
        
        for i in range(down_t):   
            input_dim = width
            block = nn.Sequential(
                CausalConv1d(input_dim, width, filter_t, stride_t, 1),
                CausalResnet1D(width, depth, dilation_growth_rate, activation=activation, norm=norm),
            )
            blocks.append(block)
        blocks.append(CausalConv1d(width, hidden_size, 3, 1, 1))
        self.model = nn.Sequential(*blocks)

        self.proj = nn.Linear(hidden_size, latent_dim*2)



    def forward(self, x):
        x = self.in_proj(x)
        x = x.transpose(1, 2) 
        x = self.model(x)
        x = x.transpose(1, 2) 
        x = self.proj(x)        
        mu, logvar = x.chunk(2, dim=2)             
        logvar = torch.clamp(logvar, self.clip_range[0], self.clip_range[1])
        # z = self.reparameterize(mu, logvar) 

        # return z, mu, logvar
        return mu, logvar

class ConvCausalDecoder(nn.Module):
    def __init__(self,
                 input_emb_width = 272,
                 hidden_size = 1024,
                 down_t = 2,
                 stride_t = 2,
                 width = 1024,
                 depth = 3,
                 dilation_growth_rate = 3, 
                 activation='relu',
                 norm=None
                 ):
        super().__init__()
        blocks = []
        
        filter_t, pad_t = stride_t * 2, stride_t // 2
        blocks.append(CausalConv1d(hidden_size, width, 3, 1, 1))
        blocks.append(nn.ReLU())
        for i in range(down_t):
            out_dim = width
            block = nn.Sequential(
                CausalResnet1D(width, depth, dilation_growth_rate, reverse_dilation=True, activation=activation, norm=norm),
                nn.Upsample(scale_factor=2, mode='nearest'),
                CausalConv1d(width, out_dim, 3, 1, 1)
            )
            blocks.append(block)
        blocks.append(CausalConv1d(width, width, 3, 1, 1))
        blocks.append(nn.ReLU())
        blocks.append(CausalConv1d(width, input_emb_width, 3, 1, 1))

        self.model = nn.Sequential(*blocks)

    def forward(self, z):
        z = z.transpose(1, 2)
        return self.model(z)








class MLPBlock(nn.Module):
    def __init__(
        self, d, expand=4, act="gelu", dropout=0.1,
        norm="ln", layerscale_init=1e-5, droppath=0.0
    ):
        super().__init__()
        self.norm = nn.LayerNorm(d) if norm.lower().startswith("ln") else nn.BatchNorm1d(d)

        self.fc1 = nn.Linear(d, d * expand)
        self.fc2 = nn.Linear(d * expand, d)

        if act.lower() == "gelu":
            self.act = nn.GELU()
        elif act.lower() == "relu":
            self.act = nn.ReLU(inplace=True)
        elif act.lower() == "silu":
            self.act = nn.SiLU()
        else:
            raise ValueError(f"Unknown act: {act}")

        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)

        # LayerScale: 아주 깊을 때 수렴 안정화에 도움
        self.gamma = nn.Parameter(layerscale_init * torch.ones(d)) if layerscale_init else None

        self.reset_parameters()

    def reset_parameters(self):
        # ReLU/SiLU/GELU 계열은 Kaiming(He) 계열이 보편적으로 안전
        nn.init.kaiming_normal_(self.fc1.weight, nonlinearity="relu")
        nn.init.zeros_(self.fc1.bias)
        nn.init.kaiming_normal_(self.fc2.weight, nonlinearity="relu")
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        y = self.norm(x)
        y = self.fc1(y)
        y = self.act(y)
        y = self.drop1(y)
        y = self.fc2(y)
        y = self.drop2(y)
        if self.gamma is not None:
            y = y * self.gamma  # [d] -> broadcast
        return x + y


class MLPEncoder(nn.Module):
    """
    in_dim -> Linear -> [MLPBlock x depth] -> (optional norm) -> out
    """
    def __init__(
        self, input_emb_width=272, out_dim=128,
        width=1024, depth=16, expand=4, act="gelu",
        dropout=0.1, norm="ln", layerscale_init=1e-5,
        final_norm=True,
        clip_range = [-30,20]
    ):
        super().__init__()
        self.in_proj = nn.Linear(input_emb_width, width)
        self.clip_range = clip_range
        blocks = []
        # droppath을 depth에 따라 선형 증가(딥 네트 안정화에 흔히 쓰는 스케줄)
        for i in range(depth):
            blocks.append(
                MLPBlock(
                    d=width, expand=expand, act=act, dropout=dropout,
                    norm=norm, layerscale_init=layerscale_init
                )
            )
        self.blocks = nn.Sequential(*blocks)
        self.out_norm = nn.LayerNorm(width) if (final_norm and norm.lower().startswith("ln")) else nn.Identity()
        self.out = nn.Linear(width, out_dim*2)

        # 입력/출력 가중치도 안전 초기화
        nn.init.kaiming_normal_(self.in_proj.weight, nonlinearity="relu")
        nn.init.zeros_(self.in_proj.bias)
        nn.init.zeros_(self.out.bias)

    def forward(self, x):
        x = self.in_proj(x)
        x = self.blocks(x)
        x = self.out_norm(x)
        x = self.out(x)
        mu, logvar = x.chunk(2, dim=2)
        logvar = torch.clamp(logvar, self.clip_range[0], self.clip_range[1])
        return mu, logvar









# ---------------------------- RoPE utilities ---------------------------- #

def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate last-dim features in pairs: (x0,x1) -> (-x1, x0).
    Expects even last dimension. Shape preserved.
    """
    x = x.view(*x.shape[:-1], -1, 2)
    x1, x2 = x.unbind(dim=-1)  # (..., d/2)
    x_rot = torch.stack((-x2, x1), dim=-1)
    return x_rot.flatten(-2)


def build_rope_cos_sin(seq_len: int, rotary_dim: int, base: float, device, dtype):
    """Build cos/sin tables for RoPE with interleaved pairs.

    Returns:
        cos, sin with shape (seq_len, rotary_dim) for broadcasting.
    """
    assert rotary_dim % 2 == 0, "rotary_dim must be even"
    # inv_freq: (rotary_dim/2,)
    inv_freq = 1.0 / (base ** (torch.arange(0, rotary_dim, 2, device=device, dtype=torch.float32) / rotary_dim))
    # positions: (seq_len,)
    t = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = torch.einsum("t,f->tf", t, inv_freq)  # (T, rotary_dim/2)
    # Expand to interleaved pairs: (..., 2 * (rotary_dim/2)) = (..., rotary_dim)
    cos = torch.cos(freqs).repeat_interleave(2, dim=-1).to(dtype)
    sin = torch.sin(freqs).repeat_interleave(2, dim=-1).to(dtype)
    return cos, sin


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply RoPE to the *first* rotary_dim features of x.

    Args:
        x:   (B, H, T, D_head) or (T, D_head)
        cos: (T, rotary_dim)
        sin: (T, rotary_dim)
    Returns:
        x with RoPE applied on the first rotary_dim dims.
    """
    B, H, T, Dh = x.shape
    rotary_dim = cos.shape[-1]
    x_rot = x[..., :rotary_dim]
    x_pass = x[..., rotary_dim:]

    cos = cos.view(1, 1, T, rotary_dim)
    sin = sin.view(1, 1, T, rotary_dim)

    x_rotated = x_rot * cos + _rotate_half(x_rot) * sin
    return torch.cat([x_rotated, x_pass], dim=-1)


# ----------------------- Local causal self-attention --------------------- #

class LocalCausalSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        attn_window: int = 128,
        rope_fraction: float = 1.0,
        rope_base: float = 10000.0,
        dropout: float = 0.1,
        bias: bool = True,
    ):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.attn_window = attn_window
        self.rope_fraction = rope_fraction
        self.rope_base = rope_base

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=bias)
        self.proj = nn.Linear(d_model, d_model, bias=bias)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

    def _make_local_causal_mask(self, T: int, device: torch.device) -> torch.Tensor:
        """
        Returns a boolean mask of shape (T, T) where True = MASKED (disallowed).
        Allowed: keys j where 0 <= i-j < attn_window (i is query index).
        """
        idx = torch.arange(T, device=device)
        rel = idx[:, None] - idx[None, :]  # (T, T), rel[i,j] = i-j
        allowed = (rel >= 0) & (rel < self.attn_window)
        return ~allowed  # True = mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        device = x.device
        dtype = x.dtype

        # qkv projection
        qkv = self.qkv(x)  # (B, T, 3*C)
        q, k, v = qkv.chunk(3, dim=-1)

        # (B, T, H, Dh) -> (B, H, T, Dh)
        def reshape_heads(t):
            return t.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        q = reshape_heads(q)
        k = reshape_heads(k)
        v = reshape_heads(v)

        # RoPE on q,k (first rotary_dim dims)
        rotary_dim = int(self.head_dim * self.rope_fraction)
        rotary_dim -= rotary_dim % 2  # even
        if rotary_dim > 0:
            cos, sin = build_rope_cos_sin(T, rotary_dim, self.rope_base, device, dtype)
            # pad cos/sin to head_dim if rope_fraction < 1
            if rotary_dim < self.head_dim:
                pad = self.head_dim - rotary_dim
                cos = F.pad(cos, (0, pad))
                sin = F.pad(sin, (0, pad))
            q = apply_rope(q, cos, sin)
            k = apply_rope(k, cos, sin)

        # SDPA expects masks with True = masked.
        attn_mask = self._make_local_causal_mask(T, device)  # (T, T)
        # Scaled dot-product attention (PyTorch native)
        attn_out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,  # broadcast to (B,H,T,T)
            dropout_p=self.attn_drop.p if self.training else 0.0,
            is_causal=False,  # we already provide a mask
        )  # (B,H,T,Dh)

        # Merge heads
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.proj_drop(self.proj(attn_out))
        return out


# ---------------------------- MLP feed-forward --------------------------- #

class FeedForward(nn.Module):
    def __init__(self, d_model: int, expand: int = 4, dropout: float = 0.1, bias: bool = True):
        super().__init__()
        hidden = d_model * expand
        self.fc1 = nn.Linear(d_model, hidden, bias=bias)
        self.fc2 = nn.Linear(hidden, d_model, bias=bias)
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)
        self.act = nn.GELU()
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.fc1.weight, nonlinearity="relu")
        nn.init.zeros_(self.fc1.bias)
        nn.init.kaiming_normal_(self.fc2.weight, nonlinearity="relu")
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


# ------------------------------- Decoder -------------------------------- #

@dataclass
class DecoderConfig:
    d_model: int = 512
    n_heads: int = 8
    depth: int = 6
    attn_window: int = 128
    rope_fraction: float = 1.0
    rope_base: float = 10000.0
    mlp_expand: int = 4
    dropout: float = 0.1
    final_norm: bool = True
    bias: bool = True


class LocalCausalBlock(nn.Module):
    def __init__(self, cfg: DecoderConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = LocalCausalSelfAttention(
            d_model=cfg.d_model,
            n_heads=cfg.n_heads,
            attn_window=cfg.attn_window,
            rope_fraction=cfg.rope_fraction,
            rope_base=cfg.rope_base,
            dropout=cfg.dropout,
            bias=cfg.bias,
        )
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.ff = FeedForward(cfg.d_model, expand=cfg.mlp_expand, dropout=cfg.dropout, bias=cfg.bias)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop(self.attn(self.ln1(x)))
        x = x + self.drop(self.ff(self.ln2(x)))
        return x


class LocalCausalDecoder(nn.Module):
    def __init__(self, d_in: int, d_out: int, cfg: DecoderConfig = DecoderConfig(), conv_mlp=False):
        super().__init__()
        self.conv_mlp = conv_mlp
        self.cfg = cfg
        self.in_proj = nn.Linear(d_in, cfg.d_model, bias=cfg.bias)
        self.blocks = nn.ModuleList([LocalCausalBlock(cfg) for _ in range(cfg.depth)])
        self.out_norm = nn.LayerNorm(cfg.d_model) if cfg.final_norm else nn.Identity()
        
        if conv_mlp:
            # self.conv_mlp = Conv1D_MLP_causal_single(cfg.d_model, d_out, 4, resid_pdrop=0.1)
            self.conv_mlp = Conv1D_MLP_causal(cfg.d_model, d_out, 4, resid_pdrop=0.1)
        else:
            self.out_proj = nn.Linear(cfg.d_model, d_out, bias=cfg.bias)
            nn.init.zeros_(self.out_proj.bias)
        # self.root_norm = nn.LayerNorm(6)
        # self.root_scale_shift = nn.Linear(6, (d_out-6)*2, bias=cfg.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, D_in) -> (B, T, D_out)"""
        x = self.in_proj(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.out_norm(x)
        if self.conv_mlp:
            x = self.conv_mlp(x)
        else:
            x = self.out_proj(x)

        # x_root = x[:, :, :6]
        # x_pose = x[:, :, 6:]
        # x_root = self.root_norm(x_root)
        # scale, shift = self.root_scale_shift(x_root).chunk(2, dim=-1)
        # x_pose = x_pose * scale + shift
        # x = torch.cat([x_root, x_pose], dim=-1)

        return x

EncoderConfig = DecoderConfig


class LocalCausalEncoder(nn.Module):
    def __init__(self, d_in: int, d_out: int, cfg: EncoderConfig = EncoderConfig(), clip_range = [-30,20]):
        super().__init__()
        self.cfg = cfg
        self.clip_range = clip_range
        self.in_proj = nn.Linear(d_in, cfg.d_model, bias=cfg.bias)
        self.blocks = nn.ModuleList([LocalCausalBlock(cfg) for _ in range(cfg.depth)])
        self.out_norm = nn.LayerNorm(cfg.d_model) if cfg.final_norm else nn.Identity()
        self.out_proj = nn.Linear(cfg.d_model, d_out * 2, bias=cfg.bias)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.in_proj(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.out_norm(x)
        x = self.out_proj(x)
        mu, logvar = x.chunk(2, dim=-1)
        logvar = torch.clamp(logvar, self.clip_range[0], self.clip_range[1])
        return mu, logvar
