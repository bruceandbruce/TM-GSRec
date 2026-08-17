import os
import time
import math
import copy
from functools import partial
from typing import Optional, Callable, Any
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, trunc_normal_
from fvcore.nn import FlopCountAnalysis, flop_count_str, flop_count, parameter_count

DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"
# train speed is slower after enabling this opts.
# torch.backends.cudnn.enabled = True
# torch.backends.cudnn.benchmark = True
# torch.backends.cudnn.deterministic = True

try:
    from .csm_triton import cross_scan_fn, cross_merge_fn
except:
    from csm_triton import cross_scan_fn, cross_merge_fn

try:
    from .csms6s import selective_scan_fn, selective_scan_flop_jit
except:
    from csms6s import selective_scan_fn, selective_scan_flop_jit

# FLOPs counter not prepared fro mamba2
try:
    from .mamba2.ssd_minimal import selective_scan_chunk_fn
except:
    from mamba2.ssd_minimal import selective_scan_chunk_fn


# =====================================================
# we have this class as linear and conv init differ from each other
# this function enable loading from both conv2d or linear
class Linear2d(nn.Linear):
    def forward(self, x: torch.Tensor):
        # B, C, H, W = x.shape
        return F.conv2d(x, self.weight[:, :, None, None], self.bias)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        state_dict[prefix + "weight"] = state_dict[prefix + "weight"].view(self.weight.shape)
        return super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)


class LayerNorm2d(nn.LayerNorm):
    def forward(self, x: torch.Tensor):
        x = x.permute(0, 2, 3, 1)
        x = nn.functional.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.permute(0, 3, 1, 2)
        return x


class PatchEmbed2D(nn.Module):
    r""" Image to Patch Embedding
    Args:
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """
    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None, **kwargs):
        super().__init__()
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = self.proj(x).permute(0, 2, 3, 1)
        if self.norm is not None:
            x = self.norm(x)
        return x
    

class PatchMerging2D(nn.Module):
    def __init__(self, dim, out_dim=-1, norm_layer=nn.LayerNorm, channel_first=False):
        super().__init__()
        self.dim = dim
        Linear = Linear2d if channel_first else nn.Linear
        self._patch_merging_pad = self._patch_merging_pad_channel_first if channel_first else self._patch_merging_pad_channel_last
        self.reduction = Linear(4 * dim, (2 * dim) if out_dim < 0 else out_dim, bias=False)
        self.norm = norm_layer(4 * dim)

    @staticmethod
    def _patch_merging_pad_channel_last(x: torch.Tensor):
        H, W, _ = x.shape[-3:]
        if (W % 2 != 0) or (H % 2 != 0):
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))
        x0 = x[..., 0::2, 0::2, :]  # ... H/2 W/2 C
        x1 = x[..., 1::2, 0::2, :]  # ... H/2 W/2 C
        x2 = x[..., 0::2, 1::2, :]  # ... H/2 W/2 C
        x3 = x[..., 1::2, 1::2, :]  # ... H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # ... H/2 W/2 4*C
        return x

    @staticmethod
    def _patch_merging_pad_channel_first(x: torch.Tensor):
        H, W = x.shape[-2:]
        if (W % 2 != 0) or (H % 2 != 0):
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))
        x0 = x[..., 0::2, 0::2]  # ... H/2 W/2
        x1 = x[..., 1::2, 0::2]  # ... H/2 W/2
        x2 = x[..., 0::2, 1::2]  # ... H/2 W/2
        x3 = x[..., 1::2, 1::2]  # ... H/2 W/2
        x = torch.cat([x0, x1, x2, x3], 1)  # ... H/2 W/2 4*C
        return x

    def forward(self, x):
        x = self._patch_merging_pad(x)
        x = self.norm(x)
        x = self.reduction(x)

        return x


class Permute(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.args = args

    def forward(self, x: torch.Tensor):
        return x.permute(*self.args)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.,channels_first=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        Linear = Linear2d if channels_first else nn.Linear
        self.fc1 = Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class gMlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.,channels_first=False):
        super().__init__()
        self.channel_first = channels_first
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        Linear = Linear2d if channels_first else nn.Linear
        self.fc1 = Linear(in_features, 2 * hidden_features)
        self.act = act_layer()
        self.fc2 = Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor):
        x = self.fc1(x)
        x, z = x.chunk(2, dim=(1 if self.channel_first else -1))
        x = self.fc2(x * self.act(z))
        x = self.drop(x)
        return x


class SoftmaxSpatial(nn.Softmax):
    def forward(self, x: torch.Tensor):
        if self.dim == -1:
            B, C, H, W = x.shape
            return super().forward(x.view(B, C, -1)).view(B, C, H, W)
        elif self.dim == 1:
            B, H, W, C = x.shape
            return super().forward(x.view(B, -1, C)).view(B, H, W, C)
        else:
            raise NotImplementedError


class SwappingScan_multiview(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, x2: torch.Tensor):
        # B, C, H, W -> B, 2, C, H*W
        B, C, H, W = x.shape
        ctx.shape = (B, C, H, W)
        
        x = x.view(B, C, -1)  # shape [B, C, H*W]
        x2 = x2.view(B, C, -1)  # shape [B, C, H*W]
        exchange_mask = torch.arange(C) % 2 == 0   # shape [C]
        exchange_mask = exchange_mask.unsqueeze(0).expand(B, -1)  # shape [B, C]
        
        out_x = torch.zeros_like(x)  # shape [B, C, N]
        out_x2 = torch.zeros_like(x2)  # shape [B, C, N]

        out_x[~exchange_mask, ...] = x[~exchange_mask, ...]
        out_x2[~exchange_mask, ...] = x2[~exchange_mask, ...]
       
        out_x[exchange_mask, ...] = x2[exchange_mask, ...]
        out_x2[exchange_mask, ...] = x[exchange_mask, ...]
        
        xs_fuse = x.new_empty((B, 2, C, H * W))
        xs_fuse[:, 0] = out_x
        xs_fuse[:, 1] = out_x2
        
        return xs_fuse

    @staticmethod
    def backward(ctx, ys: torch.Tensor):
        # out: (b, 2, d, h, w)
        B, C, H, W = ctx.shape
        
        return ys[:, 0].view(B, -1, H, W), ys[:, 1].view(B, -1, H, W)


class SwappingMerge_multiview(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ys: torch.Tensor):
        B, K, C, L = ys.shape
        # ctx.shape = (H, W)
        out_x  = ys[:, 0].contiguous()  # shape [B, C, L]
        out_x2 = ys[:, 1].contiguous()  # shape [B, C, L]
        return out_x, out_x2
    
    @staticmethod
    def backward(ctx, x: torch.Tensor, x2: torch.Tensor):
        # B, D, L = x.shape
        # out: (b, k, d, l)
        B, C, L = x.shape
        xs = x.new_empty((B, 2, C, L))
        xs[:, 0] = x
        xs[:, 1] = x2
        return xs, None, None


class ConcatScan_multimodal(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x: torch.Tensor, x2: torch.Tensor):
            # B, C, H, W -> B, 2, C, 2 * H * W
            B, C, H, W = x.shape
            ctx.shape = (B, C, H, W)
            xs_fuse = x.new_empty((B, 2, C, 2 * H * W))
            xs_fuse[:, 0] = torch.concat([x.flatten(2, 3), x2.flatten(2, 3)], dim=2)
            xs_fuse[:, 1] = torch.flip(xs_fuse[:, 0], dims=[-1])
            return xs_fuse

        @staticmethod
        def backward(ctx, ys: torch.Tensor):
            # out: (b, 2, d, l)
            B, C, H, W = ctx.shape
            L = 2 * H * W
            ys = ys[:, 0] + ys[:, 1].flip(dims=[-1]) # B, d, 2 * H * W
            # get B, d, H*W
            return ys[:, :, 0:H*W].view(B, -1, H, W), ys[:, :, H*W:2*H*W].view(B, -1, H, W) 
    
         
class ConcatMerge_multimodal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ys: torch.Tensor):
        B, K, D, L = ys.shape
        # ctx.shape = (H, W)
        # ys = ys.view(B, K, D, -1)
        ys = ys[:, 0] + ys[:, 1].flip(dims=[-1]) # B, d, 2 * H * W, broadcast
        # y = ys[:, :, 0:L//2] + ys[:, :, L//2:L]
        return ys[:, :, 0:L//2], ys[:, :, L//2:L]
    
    @staticmethod
    def backward(ctx, x1: torch.Tensor, x2: torch.Tensor):
        # B, D, L = x.shape
        # out: (b, k, d, l)
        # H, W = ctx.shape
        B, C, L = x1.shape
        xs = x1.new_empty((B, 2, C, 2*L))
        xs[:, 0] = torch.cat([x1, x2], dim=2)
        xs[:, 1] = torch.flip(xs[:, 0], dims=[-1])
        xs = xs.view(B, 2, C, 2*L)
        return xs, None, None


# =====================================================
class mamba_init:
    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        # dt_proj.bias._no_reinit = True
        
        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=-1, device=None, merge=True):
        # S4D real initialization
        A = torch.arange(1, d_state + 1, dtype=torch.float32, device=device).view(1, -1).repeat(d_inner, 1).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 0:
            A_log = A_log[None].repeat(copies, 1, 1).contiguous()
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=-1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 0:
            D = D[None].repeat(copies, 1).contiguous()
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    @classmethod
    def init_dt_A_D(cls, d_state, dt_rank, d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, k_group=4):
        # dt proj ============================
        dt_projs = [
            cls.dt_init(dt_rank, d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor)
            for _ in range(k_group)
        ]
        dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in dt_projs], dim=0)) # (K, inner, rank)
        dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in dt_projs], dim=0)) # (K, inner)
        del dt_projs
            
        # A, D =======================================
        A_logs = cls.A_log_init(d_state, d_inner, copies=k_group, merge=True) # (K * D, N)
        Ds = cls.D_init(d_inner, copies=k_group, merge=True) # (K * D)  
        return A_logs, Ds, dt_projs_weight, dt_projs_bias

class FreqBranch(nn.Module):
    def __init__(self, in_channels=128, hidden_channels=128, kernel_size=3):
        """
        Frequency domain dual-branch module for [B, C, H, W] input.
        
        Process flow:
        1. Input: [B, C, H, W] (C=128, H=256, W=256)
        2. FFT along time dimension (W) → complex spectrum [B, C, H, F]
        3. Split into magnitude and phase
        4. Two parallel branches (each: Conv → ReLU → Conv)
        5. Residual connection
        6. IFFT back to time domain
        
        Args:
            in_channels: input channels (128)
            hidden_channels: hidden dimension in conv layers (default: 256)
            kernel_size: conv kernel size (default: 3)
        """
        super().__init__()
        
        # Phase branch: Conv → ReLU → Conv
        self.phase_conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size, padding=kernel_size//2)
        self.phase_relu1 = nn.ReLU()
        self.phase_conv2 = nn.Conv2d(hidden_channels, in_channels, kernel_size, padding=kernel_size//2)
        
        # Amplitude branch: Conv → ReLU → Conv
        self.amp_conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size, padding=kernel_size//2)
        self.amp_relu1 = nn.ReLU()
        self.amp_conv2 = nn.Conv2d(hidden_channels, in_channels, kernel_size, padding=kernel_size//2)

    def forward(self, x):
        """
        Input:  [B, C, H, W] = [B, 128, 256, 256]
        Output: [B, C, H, W] = [B, 128, 256, 256]
        """
        B, C, H, W = x.shape
        
        # Step 1: FFT along time dimension (W)
        # Input: [B, C, H, W] → FFT along dim=-1 → [B, C, H, F] where F = W//2 + 1
        X = torch.fft.rfft(x, dim=-1)  # [B, C, H, F], F = 129 (since W=256)
        # print(x.shape)
        
        # Step 2: Extract magnitude and phase
        mag = torch.abs(X)          # [B, C, H, F]
        phase = torch.angle(X)      # [B, C, H, F]
        
        # Step 3: Process magnitude branch
        mag_out = self.amp_conv1(mag)      # [B, hidden, H, F]
        mag_out = self.amp_relu1(mag_out)
        mag_out = self.amp_conv2(mag_out)  # [B, C, H, F]
        
        # Step 4: Process phase branch
        phase_out = self.phase_conv1(phase)      # [B, hidden, H, F]
        phase_out = self.phase_relu1(phase_out)
        phase_out = self.phase_conv2(phase_out)  # [B, C, H, F]
        
        # Step 5: Residual connection
        mag_res = mag + mag_out          # [B, C, H, F]
        phase_res = phase + phase_out    # [B, C, H, F]
        
        # Step 6: Reconstruct complex spectrum
        X_enh = mag_res * torch.exp(1j * phase_res)  # [B, C, H, F]
        
        # Step 7: IFFT back to time domain
        y = torch.fft.irfft(X_enh, n=W, dim=-1)  # [B, C, H, W]
        
        return y

class Cross_SS2Dv5(nn.Module):
    def __init__(
        self,
        # basic dims ===========
        d_model=96,
        d_state=16,
        ssm_ratio=2.0,
        dt_rank="auto",
        act_layer=nn.SiLU,
        # dwconv ===============
        d_conv=3, # < 2 means no conv 
        conv_bias=True,
        # ======================
        dropout=0.0,
        bias=False,
        # dt init ==============
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        initialize="v0",
        # ======================
        forward_type="v2",
        channel_first=False,
        # ======================
        **kwargs,    
    ):
        factory_kwargs = {"device": None, "dtype": None}
        super().__init__()
        self.k_group = 4
        self.d_model = int(d_model)
        self.d_state = int(d_state)
        self.d_inner = int(ssm_ratio * d_model)
        self.dt_rank = int(math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank)
        self.channel_first = channel_first
        self.with_dconv = d_conv > 1
        Linear = Linear2d if channel_first else nn.Linear

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.in_proj_sec = nn.Linear(self.d_model, self.d_inner, bias=bias, **factory_kwargs)
        self.act: nn.Module = act_layer()
        # self.conv1 = nn.Conv2d(in_channels=128, out_channels=2, kernel_size=3, padding=1)
        # self.conv2 = nn.Sequential(nn.Conv2d(in_channels=2, out_channels=32, kernel_size=3, padding=1),
        #                            nn.Conv2d(in_channels=32, out_channels=2, kernel_size=3, padding=1))
        # self.conv3 = nn.Conv2d(in_channels=2, out_channels=128, kernel_size=3, padding=1)
        self.conv_fre = FreqBranch()
        # conv =======================================
        if self.with_dconv:
            self.conv2d = nn.Conv2d(
                in_channels=self.d_inner,
                out_channels=self.d_inner,
                groups=self.d_inner,
                bias=conv_bias,
                kernel_size=d_conv,
                padding=(d_conv - 1) // 2,
                **factory_kwargs,
            )

        # x proj ============================
        self.x_proj = [
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False)
            for _ in range(self.k_group)
        ]
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0)) # (K, N, inner)
        del self.x_proj
        
        # out proj =======================================
        self.out_norm = nn.LayerNorm(self.d_inner)
        # self.out_act = nn.GELU() if self.oact else nn.Identity()
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

        if initialize in ["v0"]:
            self.A_logs, self.Ds, self.dt_projs_weight, self.dt_projs_bias = mamba_init.init_dt_A_D(
                self.d_state, self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, k_group=self.k_group,
            )
        elif initialize in ["v1"]:
            # simple init dt_projs, A_logs, Ds
            self.Ds = nn.Parameter(torch.ones((self.k_group * self.d_inner)))
            self.A_logs = nn.Parameter(torch.randn((self.k_group * self.d_inner, self.d_state))) # A == -A_logs.exp() < 0; # 0 < exp(A * dt) < 1
            self.dt_projs_weight = nn.Parameter(0.1 * torch.randn((self.k_group, self.d_inner, self.dt_rank))) # 0.1 is added in 0430
            self.dt_projs_bias = nn.Parameter(0.1 * torch.randn((self.k_group, self.d_inner))) # 0.1 is added in 0430
        elif initialize in ["v2"]:
            # simple init dt_projs, A_logs, Ds
            self.Ds = nn.Parameter(torch.ones((self.k_group * self.d_inner)))
            self.A_logs = nn.Parameter(torch.zeros((self.k_group * self.d_inner, self.d_state))) # A == -A_logs.exp() < 0; # 0 < exp(A * dt) < 1
            self.dt_projs_weight = nn.Parameter(0.1 * torch.rand((self.k_group, self.d_inner, self.dt_rank)))
            self.dt_projs_bias = nn.Parameter(0.1 * torch.rand((self.k_group, self.d_inner)))

    def forward_corev2(
        self,
        x: torch.Tensor=None, 
        x2: torch.Tensor=None,
        x_fuse: torch.Tensor=None,  
        # ==============================
        force_fp32=False, # True: input fp32
        # ==============================
        ssoflex=True, # True: input 16 or 32 output 32 False: output dtype as input
        no_einsum=False, # replace einsum with linear or conv1d to raise throughput
        # ==============================
        selective_scan_backend = None,
        # ==============================
        scan_mode = "cross2d",
        scan_force_torch = False,
        # ==============================
        **kwargs,
    ):
        assert selective_scan_backend in [None, "oflex", "mamba", "torch"]
        _scan_mode = dict(cross2d=0, unidi=1, bidi=2, cascade2d=-1).get(scan_mode, None) if isinstance(scan_mode, str) else scan_mode # for debug
        assert isinstance(_scan_mode, int)
        delta_softplus = True
        out_norm = self.out_norm
        channel_first = self.channel_first
        to_fp32 = lambda *args: (_a.to(torch.float32) for _a in args)

        B, D, H, W = x.shape
        N = self.d_state
        K, D, R = self.k_group, self.d_inner, self.dt_rank
        L = H * W

        def selective_scan(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=True):
            return selective_scan_fn(u, delta, A, B, C, D, delta_bias, delta_softplus, ssoflex, backend=selective_scan_backend)
        
        x_proj_bias = getattr(self, "x_proj_bias", None)
        
        # ==============================
        # print("x_fuse3=", x_fuse.shape)
        xs_fuse = cross_scan_fn(x_fuse, in_channel_first=True, out_channel_first=True, scans=_scan_mode, force_torch=scan_force_torch)
        # print("xs_fuse4=", xs_fuse.shape)
        if no_einsum:
            x_fuse_dbl = F.conv1d(xs_fuse.view(B, -1, L), self.x_proj_weight.view(-1, D, 1), bias=(x_proj_bias.view(-1) if x_proj_bias is not None else None), groups=K)
            dts_fuse, Bs_fuse, Cs_fuse = torch.split(x_fuse_dbl.view(B, K, -1, L), [R, N, N], dim=2)
            if hasattr(self, "dt_projs_weight"):
                dts_fuse = F.conv1d(dts_fuse.contiguous().view(B, -1, L), self.dt_projs_weight.view(K * D, -1, 1), groups=K)
        else:
            x_fuse_dbl = torch.einsum("b k d l, k c d -> b k c l", xs_fuse, self.x_proj_weight)
            if x_proj_bias is not None:
                x_fuse_dbl = x_fuse_dbl + x_proj_bias.view(1, K, -1, 1)
            dts_fuse, Bs_fuse, Cs_fuse = torch.split(x_fuse_dbl, [R, N, N], dim=2)
            if hasattr(self, "dt_projs_weight"):
                dts_fuse = torch.einsum("b k r l, k d r -> b k d l", dts_fuse, self.dt_projs_weight)

        xs_fuse = xs_fuse.view(B, -1, L)
        dts_fuse = dts_fuse.contiguous().view(B, -1, L)
        As = -self.A_logs.to(torch.float).exp() # (k * c, d_state)
        Ds = self.Ds.to(torch.float) # (K * c)
        Bs_fuse = Bs_fuse.contiguous().view(B, K, N, L)
        Cs_fuse = Cs_fuse.contiguous().view(B, K, N, L)
        delta_bias = self.dt_projs_bias.view(-1).to(torch.float)

        x_fuse = self.conv_fre(x_fuse)

        # x_fuse = self.conv1(x_fuse)
        # x_fuse = x_fuse.permute(0, 2, 3, 1)
        # x_fuse = torch.complex(x_fuse[...,0],x_fuse[...,1])
        # x_fuse = torch.fft.fft2(x_fuse, dim=(-2,-1))
        # x_fuse = torch.stack((x_fuse.real,x_fuse.imag), dim=-1)
        # x_fuse = x_fuse.permute(0, 3, 1, 2)
        # x_fuse = self.conv2(x_fuse)
        # # torch.Size([1, 128, 320, 320])
        # # print(1111111111)
        # # print(x_fuse.shape)


        # x_fuse = x_fuse.permute(0, 2, 3, 1)
        # x_fuse =torch.complex(x_fuse[...,0], x_fuse[...,1])
        
        # x_fuse = torch.fft.ifft2(x_fuse, dim=(-2,-1))
        # x_fuse = torch.stack((x_fuse.real,x_fuse.imag), dim=-1)
        # x_fuse = x_fuse.permute(0, 3, 1, 2)
        # x_fuse = self.conv3(x_fuse)
        y_fuse = x_fuse.permute(0, 2, 3, 1)

        # print("x_fuse5 = ",x_fuse.shape)
        # print("xs_fuse=", xs_fuse.shape)
        # ys_fuse: torch.Tensor = selective_scan(
        #     xs_fuse, dts_fuse, As, Bs_fuse, Cs_fuse, Ds, delta_bias, delta_softplus
        # ).view(B, K, -1, H, W)

        # print("ys_fuse=", ys_fuse.shape)
        
        # y_fuse: torch.Tensor = cross_merge_fn(ys_fuse, in_channel_first=True, out_channel_first=True, scans=_scan_mode, force_torch=scan_force_torch)
        # print("y_fuse=", y_fuse.shape)

        # y_fuse = y_fuse.view(B, -1, H, W)
        # print("y_fuse=", y_fuse.shape)
        # if not channel_first:
        #     y_fuse = y_fuse.view(B, -1, H * W).transpose(dim0=1, dim1=2).contiguous().view(B, H, W, -1) # (B, L, C)
        # y_fuse = out_norm(y_fuse)
        
        # ==============================

        # print(xs_fuse.shape)
        xs = cross_scan_fn(x, in_channel_first=True, out_channel_first=True, scans=_scan_mode, force_torch=scan_force_torch)
        if no_einsum:
            x_dbl = F.conv1d(xs.view(B, -1, L), self.x_proj_weight.view(-1, D, 1), bias=(x_proj_bias.view(-1) if x_proj_bias is not None else None), groups=K)
            dts, Bs, Cs = torch.split(x_dbl.view(B, K, -1, L), [R, N, N], dim=2)
            if hasattr(self, "dt_projs_weight"):
                dts = F.conv1d(dts.contiguous().view(B, -1, L), self.dt_projs_weight.view(K * D, -1, 1), groups=K)
        else:
            x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, self.x_proj_weight)
            if x_proj_bias is not None:
                x_dbl = x_dbl + x_proj_bias.view(1, K, -1, 1)
            dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
            if hasattr(self, "dt_projs_weight"):
                dts = torch.einsum("b k r l, k d r -> b k d l", dts, self.dt_projs_weight)

        xs = xs.view(B, -1, L)
        dts = dts.contiguous().view(B, -1, L)
        Bs = Bs.contiguous().view(B, K, N, L)
        Cs = Cs.contiguous().view(B, K, N, L)

        ys: torch.Tensor = selective_scan(
            xs, dts, As, Bs, Cs_fuse, Ds, delta_bias, delta_softplus
        ).view(B, K, -1, H, W)
        
        y: torch.Tensor = cross_merge_fn(ys, in_channel_first=True, out_channel_first=True, scans=_scan_mode, force_torch=scan_force_torch)

        y = y.view(B, -1, H, W)
        if not channel_first:
            y = y.view(B, -1, H * W).transpose(dim0=1, dim1=2).contiguous().view(B, H, W, -1) # (B, L, C)
        y = out_norm(y)

        # ==============================
        xs_2 = cross_scan_fn(x2, in_channel_first=True, out_channel_first=True, scans=_scan_mode, force_torch=scan_force_torch)
        if no_einsum:
            x_2_dbl = F.conv1d(xs_2.view(B, -1, L), self.x_proj_weight.view(-1, D, 1), bias=(x_proj_bias.view(-1) if x_proj_bias is not None else None), groups=K)
            dts_2, Bs_2, Cs_2 = torch.split(x_2_dbl.view(B, K, -1, L), [R, N, N], dim=2)
            if hasattr(self, "dt_projs_weight"):
                dts_2 = F.conv1d(dts_2.contiguous().view(B, -1, L), self.dt_projs_weight.view(K * D, -1, 1), groups=K)
        else:
            x_2_dbl = torch.einsum("b k d l, k c d -> b k c l", xs_2, self.x_proj_weight)
            if x_proj_bias is not None:
                x_2_dbl = x_2_dbl + x_proj_bias.view(1, K, -1, 1)
            dts_2, Bs_2, Cs_2 = torch.split(x_2_dbl, [R, N, N], dim=2)
            if hasattr(self, "dt_projs_weight"):
                dts_2 = torch.einsum("b k r l, k d r -> b k d l", dts_2, self.dt_projs_weight)

        xs_2 = xs_2.view(B, -1, L)
        dts_2 = dts_2.contiguous().view(B, -1, L)
        Bs_2 = Bs_2.contiguous().view(B, K, N, L)
        Cs_2 = Cs_2.contiguous().view(B, K, N, L)

        ys_2: torch.Tensor = selective_scan(
            xs_2, dts_2, As, Bs_2, Cs_fuse, Ds, delta_bias, delta_softplus
        ).view(B, K, -1, H, W)
        
        y_2: torch.Tensor = cross_merge_fn(ys_2, in_channel_first=True, out_channel_first=True, scans=_scan_mode, force_torch=scan_force_torch)

        y_2 = y_2.view(B, -1, H, W)
        if not channel_first:
            y_2 = y_2.view(B, -1, H * W).transpose(dim0=1, dim1=2).contiguous().view(B, H, W, -1) # (B, L, C)
        y_2 = out_norm(y_2)

        # print(y_2.to(x2.dtype).shape)
        # torch.Size([1, 320, 320, 128])
        
        return y.to(x.dtype), y_2.to(x2.dtype), y_fuse.to(x_fuse.dtype)

    def forward(self, x, x2: torch.Tensor, **kwargs):
        x_fuse = (x + x2)/2
        
        
        x = self.in_proj_sec(x)
        x2 = self.in_proj_sec(x2)
        x_fuse = self.in_proj_sec(x_fuse)
        
        

        z = self.act(x_fuse)

        if not self.channel_first:
            x = x.permute(0, 3, 1, 2).contiguous()
            x2 = x2.permute(0, 3, 1, 2).contiguous()
            x_fuse = x_fuse.permute(0, 3, 1, 2).contiguous()
        # print('x_fuse1 =',x_fuse.shape)
        if self.with_dconv:
            x = self.conv2d(x) # (b, d, h, w)
            x2 = self.conv2d(x2) # (b, d, h, w)
            x_fuse = self.conv2d(x_fuse)
        # x_fuse = self.conv1(x_fuse)
        # x_fuse = x_fuse.permute(0, 2, 3, 1)
        # x_fuse = torch.complex(x_fuse[...,0],x_fuse[...,1])
        # x_fuse = torch.fft.fft2(x_fuse, dim=(-2,-1))
        # x_fuse = torch.stack((x_fuse.real,x_fuse.imag), dim=-1)
        # x_fuse = x_fuse.permute(0, 3, 1, 2)
        # x_fuse = self.conv2(x_fuse)
        # # torch.Size([1, 128, 320, 320])
        # # print(1111111111)
        # # print(x_fuse.shape)


        # x_fuse = x_fuse.permute(0, 2, 3, 1)
        # x_fuse =torch.complex(x_fuse[...,0], x_fuse[...,1])
        
        # x_fuse = torch.fft.ifft2(x_fuse, dim=(-2,-1))
        # x_fuse = torch.stack((x_fuse.real,x_fuse.imag), dim=-1)
        # x_fuse = x_fuse.permute(0, 3, 1, 2)
        # x_fuse = self.conv3(x_fuse)

        # print(222222222222222222222222222)
        # print(x_fuse.shape)
        x = self.act(x) 
        x2 = self.act(x2)
        x_fuse = self.act(x_fuse)
        # print('x_fuse2 =',x_fuse.shape)

        y, y2, y_fuse = self.forward_corev2(x, x2, x_fuse)
        
        y = y * z
        y2 = y2 * z
        y_fuse = y_fuse * z

        hh1 = y+y_fuse
        hh2 = y2+y_fuse
        hh1 = self.dropout(self.out_proj(hh1))
        hh2 = self.dropout(self.out_proj(hh2))

        # y_fusion3 = y + y2 + y_fuse
        # out = self.dropout(self.out_proj(y_fusion3))
        # return out
        return hh1,hh2


        print(f"Output from layer {i}: {output.shape}")
