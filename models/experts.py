"""
Expert forecasting models for DA-MoE.
Implements diverse backbone architectures covering Transformer-based,
convolutional, spectral, and linear paradigms.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional


# ---------------------------------------------------------------------------
# Shared Backbone + LoRA-style adapters
# ---------------------------------------------------------------------------

class LoRAAdapter(nn.Module):
    """Lightweight low-rank adapter for parameter-efficient expert tuning."""

    def __init__(self, d_model: int, rank: int = 8):
        super().__init__()
        self.down = nn.Linear(d_model, rank, bias=False)
        self.up   = nn.Linear(rank, d_model, bias=False)
        nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.up(F.gelu(self.down(x)))


class SharedBackboneEncoder(nn.Module):
    """
    Shared temporal encoder shared across all experts.
    Extracts latent temporal representations via a lightweight
    convolutional + positional-encoding stack.
    """

    def __init__(self, input_len: int, d_vars: int, d_model: int = 64,
                 num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Linear(d_vars, d_model)
        self.pos_enc    = nn.Parameter(torch.randn(1, input_len, d_model) * 0.02)
        encoder_layer   = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=4, dim_feedforward=d_model * 2,
            dropout=dropout, batch_first=True, norm_first=True
        )
        self.encoder    = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout    = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, d_vars) -> (B, T, d_model)"""
        h = self.input_proj(x) + self.pos_enc[:, :x.size(1)]
        return self.encoder(self.dropout(h))


# ---------------------------------------------------------------------------
# Base Expert class
# ---------------------------------------------------------------------------

class BaseExpert(nn.Module):
    """
    Base class for all DA-MoE experts.
    Wraps a shared backbone + expert-specific adapter + output head.
    Optionally yields predictive uncertainty via MC-Dropout.
    """

    def __init__(self, backbone: SharedBackboneEncoder,
                 input_len: int, pred_len: int, d_vars: int,
                 d_model: int = 64, rank: int = 8, mc_samples: int = 0):
        super().__init__()
        self.backbone   = backbone
        self.adapter    = LoRAAdapter(d_model, rank)
        self.head       = nn.Linear(d_model, d_vars)
        self.pred_len   = pred_len
        self.input_len  = input_len
        self.d_vars     = d_vars
        self.mc_samples = mc_samples          # >0 → MC-Dropout uncertainty
        self.dropout    = nn.Dropout(0.1)

    def _forward_once(self, x: torch.Tensor) -> torch.Tensor:
        h   = self.backbone(x)                # (B, T, d_model)
        h   = self.adapter(h)                 # (B, T, d_model)
        h   = self.dropout(h)
        out = self.head(h[:, -self.pred_len:]) # (B, pred_len, d_vars)
        return out

    def forward(self, x: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            mu    : (B, pred_len, d_vars)  – point / mean forecast
            sigma2: (B, pred_len, d_vars)  – predictive variance (0 if mc=0)
        """
        if self.mc_samples > 0 and self.training is False:
            self.train()
            preds = torch.stack([self._forward_once(x)
                                  for _ in range(self.mc_samples)], dim=0)
            self.eval()
            mu     = preds.mean(0)
            sigma2 = preds.var(0)
        else:
            mu     = self._forward_once(x)
            sigma2 = torch.zeros_like(mu)
        return mu, sigma2


# ---------------------------------------------------------------------------
# Concrete Expert variants (distinct architectural heads)
# ---------------------------------------------------------------------------

class PatchTSTExpert(BaseExpert):
    """
    PatchTST-style expert: patches the input sequence, applies attention,
    then maps to forecast horizon.
    """

    def __init__(self, backbone, input_len, pred_len, d_vars,
                 d_model=64, patch_size=16, rank=8, mc_samples=0):
        super().__init__(backbone, input_len, pred_len, d_vars,
                         d_model, rank, mc_samples)
        self.patch_size = patch_size
        num_patches     = input_len // patch_size
        self.patch_proj = nn.Linear(patch_size * d_vars, d_model)
        enc_layer       = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=4, dim_feedforward=d_model*2,
            dropout=0.1, batch_first=True, norm_first=True
        )
        self.patch_enc  = nn.TransformerEncoder(enc_layer, num_layers=2)
        self.patch_head = nn.Linear(num_patches * d_model, pred_len * d_vars)
        self.num_patches = num_patches

    def _forward_once(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        # patch
        x_p = x[:, :self.num_patches * self.patch_size]
        x_p = x_p.reshape(B, self.num_patches, self.patch_size * C)
        x_p = self.patch_proj(x_p)
        x_p = self.patch_enc(x_p)            # (B, num_patches, d_model)
        out = self.patch_head(x_p.flatten(1)) # (B, pred_len*C)
        return out.reshape(B, self.pred_len, C)


class iTransformerExpert(BaseExpert):
    """
    iTransformer-style expert: transposes the time and variable axes so
    attention runs over variables rather than time steps.
    """

    def __init__(self, backbone, input_len, pred_len, d_vars,
                 d_model=64, rank=8, mc_samples=0):
        super().__init__(backbone, input_len, pred_len, d_vars,
                         d_model, rank, mc_samples)
        self.var_proj   = nn.Linear(input_len, d_model)
        enc_layer       = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=4, dim_feedforward=d_model*2,
            dropout=0.1, batch_first=True, norm_first=True
        )
        self.var_enc    = nn.TransformerEncoder(enc_layer, num_layers=2)
        self.out_proj   = nn.Linear(d_model, pred_len)

    def _forward_once(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C) -> transpose to (B, C, T)
        x_t  = x.permute(0, 2, 1)              # (B, C, T)
        h    = self.var_proj(x_t)              # (B, C, d_model)
        h    = self.var_enc(h)                 # (B, C, d_model)
        out  = self.out_proj(h)                # (B, C, pred_len)
        return out.permute(0, 2, 1)            # (B, pred_len, C)


class TimeMixerExpert(BaseExpert):
    """
    TimeMixer-style expert: multi-scale decomposition with mixing.
    Decomposes via average pooling at several scales, mixes, then forecasts.
    """

    def __init__(self, backbone, input_len, pred_len, d_vars,
                 d_model=64, scales=(1, 2, 4), rank=8, mc_samples=0):
        super().__init__(backbone, input_len, pred_len, d_vars,
                         d_model, rank, mc_samples)
        self.scales  = scales
        self.mix_lin = nn.ModuleList([
            nn.Linear(input_len // s, d_model) for s in scales
        ])
        self.mixer   = nn.Linear(len(scales) * d_model * d_vars, pred_len * d_vars)

    def _forward_once(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        outs = []
        for i, s in enumerate(self.scales):
            xd = F.avg_pool1d(x.permute(0, 2, 1), s).permute(0, 2, 1)  # (B,T//s,C)
            h  = self.mix_lin[i](xd.permute(0, 2, 1))                    # (B,C,d)
            outs.append(h.flatten(1))
        out = self.mixer(torch.cat(outs, dim=-1))
        return out.reshape(B, self.pred_len, C)


class DLinearExpert(BaseExpert):
    """
    DLinear-style expert: separate linear trend + residual mappings.
    """

    def __init__(self, backbone, input_len, pred_len, d_vars,
                 d_model=64, rank=8, mc_samples=0):
        super().__init__(backbone, input_len, pred_len, d_vars,
                         d_model, rank, mc_samples)
        self.trend_lin = nn.Linear(input_len, pred_len)
        self.resid_lin = nn.Linear(input_len, pred_len)

    def _forward_once(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C)
        x_t     = x.permute(0, 2, 1)          # (B, C, T)
        trend   = F.avg_pool1d(x_t, 25, stride=1, padding=12)
        resid   = x_t - trend
        out     = self.trend_lin(trend) + self.resid_lin(resid)
        return out.permute(0, 2, 1)            # (B, pred_len, C)


class FEDformerExpert(BaseExpert):
    """
    FEDformer-style expert: frequency-domain mixing via random Fourier modes.
    """

    def __init__(self, backbone, input_len, pred_len, d_vars,
                 d_model=64, modes=32, rank=8, mc_samples=0):
        super().__init__(backbone, input_len, pred_len, d_vars,
                         d_model, rank, mc_samples)
        self.modes     = min(modes, input_len // 2)
        self.freq_proj = nn.Linear(self.modes * 2, pred_len)  # real+imag

    def _forward_once(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        # FFT over time axis
        xf   = torch.fft.rfft(x, dim=1)               # (B, T//2+1, C)
        xf   = xf[:, :self.modes]                      # keep low modes
        x_ri = torch.cat([xf.real, xf.imag], dim=1)   # (B, 2*modes, C)
        out  = self.freq_proj(x_ri.permute(0, 2, 1))  # (B, C, pred_len)
        return out.permute(0, 2, 1)


class TimesNetExpert(BaseExpert):
    """
    TimesNet-style expert: reshapes 1-D series to 2-D via FFT period
    detection, applies 2-D convolutions, then flattens to prediction.
    """

    def __init__(self, backbone, input_len, pred_len, d_vars,
                 d_model=64, rank=8, mc_samples=0):
        super().__init__(backbone, input_len, pred_len, d_vars,
                         d_model, rank, mc_samples)
        self.conv2d  = nn.Conv2d(d_vars, d_vars, kernel_size=3, padding=1, groups=d_vars)
        self.out_lin = nn.Linear(input_len * d_vars, pred_len * d_vars)

    def _detect_period(self, x):
        fft_vals = torch.abs(torch.fft.rfft(x.mean(-1), dim=1))
        fft_vals[:, 0] = 0
        period = fft_vals[:, 1:].argmax(dim=1) + 2   # (B,)
        return period

    def _forward_once(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        period  = self._detect_period(x)
        p       = int(period.float().mean().item())
        p       = max(p, 2)
        pad     = (p - T % p) % p
        x_pad   = F.pad(x, (0, 0, 0, pad))
        T2      = x_pad.size(1)
        x_2d    = x_pad.reshape(B, T2 // p, p, C).permute(0, 3, 1, 2)  # (B,C,H,W)
        h       = F.gelu(self.conv2d(x_2d))
        h       = h.permute(0, 2, 3, 1).reshape(B, -1)[:, :T * C]
        out     = self.out_lin(F.gelu(h.reshape(B, T * C)[:, :self.input_len * C]
                                       if h.size(-1) >= self.input_len * C
                                       else F.pad(h.reshape(B, -1),
                                                  (0, self.input_len * C - h.size(-1)))))
        return out.reshape(B, self.pred_len, C)


class PAttnExpert(BaseExpert):
    """
    PAttn (Pretrained Attention) expert: pure multi-head self-attention
    applied directly to the raw time series with linear projection head.
    """

    def __init__(self, backbone, input_len, pred_len, d_vars,
                 d_model=64, nhead=4, rank=8, mc_samples=0):
        super().__init__(backbone, input_len, pred_len, d_vars,
                         d_model, rank, mc_samples)
        self.in_proj  = nn.Linear(d_vars, d_model)
        self.attn     = nn.MultiheadAttention(d_model, nhead, batch_first=True, dropout=0.1)
        self.out_proj = nn.Linear(d_model, d_vars)
        self.pool     = nn.AdaptiveAvgPool1d(pred_len)

    def _forward_once(self, x: torch.Tensor) -> torch.Tensor:
        h      = F.gelu(self.in_proj(x))
        h, _   = self.attn(h, h, h)
        h      = h.permute(0, 2, 1)          # (B, d_model, T)
        h      = self.pool(h).permute(0, 2, 1)  # (B, pred_len, d_model)
        return self.out_proj(h)


class TimeXerExpert(BaseExpert):
    """
    TimeXer-style expert: uses exogenous variable embedding alongside
    endogenous, then merges via cross-attention.
    Simplified version: cross-attention between past and future embeddings.
    """

    def __init__(self, backbone, input_len, pred_len, d_vars,
                 d_model=64, rank=8, mc_samples=0):
        super().__init__(backbone, input_len, pred_len, d_vars,
                         d_model, rank, mc_samples)
        self.enc_proj   = nn.Linear(d_vars, d_model)
        self.dec_query  = nn.Parameter(torch.randn(1, pred_len, d_model) * 0.02)
        self.cross_attn = nn.MultiheadAttention(d_model, 4, batch_first=True, dropout=0.1)
        self.out_proj   = nn.Linear(d_model, d_vars)

    def _forward_once(self, x: torch.Tensor) -> torch.Tensor:
        B    = x.size(0)
        mem  = F.gelu(self.enc_proj(x))                          # (B,T,d)
        q    = self.dec_query.expand(B, -1, -1)                  # (B,pred_len,d)
        h, _ = self.cross_attn(q, mem, mem)                      # (B,pred_len,d)
        return self.out_proj(h)


# ---------------------------------------------------------------------------
# Expert registry
# ---------------------------------------------------------------------------

EXPERT_CLASSES = {
    "patchtst":    PatchTSTExpert,
    "itransformer": iTransformerExpert,
    "timemixer":   TimeMixerExpert,
    "dlinear":     DLinearExpert,
    "fedformer":   FEDformerExpert,
    "timesnet":    TimesNetExpert,
    "pattn":       PAttnExpert,
    "timexer":     TimeXerExpert,
}


def build_expert_pool(input_len: int, pred_len: int, d_vars: int,
                      d_model: int = 64, mc_samples: int = 10,
                      device: str = "cpu") -> nn.ModuleDict:
    """
    Instantiates the full candidate expert pool with a shared backbone.
    All experts share the same backbone encoder; only adapters differ.
    """
    backbone = SharedBackboneEncoder(input_len, d_vars, d_model)
    experts  = {}
    for name, cls in EXPERT_CLASSES.items():
        experts[name] = cls(backbone, input_len, pred_len, d_vars,
                            d_model=d_model, mc_samples=mc_samples)
    pool = nn.ModuleDict(experts)
    return pool.to(device)
