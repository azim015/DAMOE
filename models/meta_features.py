"""
Hybrid Multi-Scale Meta-Feature Extractor (Section 3.2 of DA-MoE paper).

Three-branch architecture:
  1. Multi-scale statistical descriptors  (Zstat)
  2. Self-supervised temporal encoder     (zlearned)
  3. Graph-aware multivariate encoder     (zgraph)

These are concatenated into a unified meta-feature vector x* used by the fusor.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple


# ---------------------------------------------------------------------------
# Branch 1: Multi-Scale Statistical Descriptors
# ---------------------------------------------------------------------------

class MultiScaleStatExtractor(nn.Module):
    """
    Computes statistical descriptors (mean, variance, lag-k autocorrelation,
    entropy proxy) at multiple temporal scales (segment sizes).

    Output shape: (B, L, Fs) where L = number of scales, Fs = num stats/scale.
    Pooled to (B, L*Fs) via concat of mean/max.
    """

    def __init__(self, scales: Tuple[int, ...] = (1, 2, 4, 8),
                 lags: Tuple[int, ...] = (1, 2, 4)):
        super().__init__()
        self.scales = scales
        self.lags   = lags
        # per segment: mean, var, lag-k autocorrs (one per lag), entropy proxy
        self.fs     = 2 + len(lags) + 1    # mean + var + acfs + entropy

    def output_dim(self, d_vars: int) -> int:
        return len(self.scales) * self.fs * d_vars * 2  # mean-pool + max-pool

    def _autocorr(self, seg: torch.Tensor, lag: int) -> torch.Tensor:
        """seg: (B, n_segs, seg_len, C) -> (B, n_segs, C)"""
        if seg.size(2) <= lag:
            return torch.zeros(*seg.shape[:2], seg.shape[-1], device=seg.device)
        x0 = seg[:, :, :-lag]  - seg[:, :, :-lag].mean(2, keepdim=True)
        x1 = seg[:, :, lag:]   - seg[:, :, lag:].mean(2, keepdim=True)
        num = (x0 * x1).mean(2)
        den = (x0.std(2) * x1.std(2)).clamp(min=1e-6)
        return num / den

    def _entropy_proxy(self, seg: torch.Tensor) -> torch.Tensor:
        """Approximate entropy via log-variance. seg: (B,n_segs,seg_len,C)->(B,n_segs,C)"""
        return torch.log(seg.var(2).clamp(min=1e-6))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, C) -> (B, output_dim)"""
        B, T, C = x.shape
        scale_feats = []
        for s in self.scales:
            seg_len  = max(T // s, 1)
            n_segs   = T // seg_len
            if n_segs == 0:
                n_segs  = 1
                seg_len = T
            x_trim = x[:, :n_segs * seg_len]                         # (B, n*sl, C)
            segs   = x_trim.reshape(B, n_segs, seg_len, C)           # (B, n, sl, C)

            mu    = segs.mean(2)                                      # (B, n, C)
            var   = segs.var(2)                                       # (B, n, C)
            acfs  = [self._autocorr(segs, lag) for lag in self.lags] # list of (B,n,C)
            ent   = self._entropy_proxy(segs)                         # (B, n, C)

            stat  = torch.stack([mu, var] + acfs + [ent], dim=-1)    # (B,n,C,Fs)
            stat  = stat.reshape(B, n_segs, -1)                      # (B, n, C*Fs)
            mean_pool = stat.mean(1)                                  # (B, C*Fs)
            max_pool  = stat.max(1).values                            # (B, C*Fs)
            scale_feats.append(torch.cat([mean_pool, max_pool], -1))

        return torch.cat(scale_feats, dim=-1)                         # (B, output_dim)


# ---------------------------------------------------------------------------
# Branch 2: Self-Supervised Temporal Encoder (learned embeddings)
# ---------------------------------------------------------------------------

class TemporalEncoder(nn.Module):
    """
    Lightweight Transformer encoder trained with masked-prediction
    (self-supervised) on raw time-series windows.

    For DA-MoE inference, it encodes the input window to a compact
    representation zlearned.
    """

    def __init__(self, input_len: int, d_vars: int, d_model: int = 64,
                 d_embed: int = 128, mask_ratio: float = 0.15):
        super().__init__()
        self.mask_ratio  = mask_ratio
        self.input_proj  = nn.Linear(d_vars, d_model)
        self.pos_enc     = nn.Parameter(torch.randn(1, input_len, d_model) * 0.02)

        encoder_layer    = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=4, dim_feedforward=d_model * 2,
            dropout=0.1, batch_first=True, norm_first=True
        )
        self.encoder     = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.pool        = nn.AdaptiveAvgPool1d(1)
        self.proj        = nn.Sequential(
            nn.Linear(d_model, d_embed),
            nn.GELU(),
            nn.Linear(d_embed, d_embed)
        )
        # Reconstruction head for self-supervised pretraining
        self.recon_head  = nn.Linear(d_model, d_vars)

    @property
    def output_dim(self) -> int:
        return self.proj[-1].out_features

    def _mask(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, C = x.shape
        mask    = torch.rand(B, T, device=x.device) < self.mask_ratio
        x_mask  = x.clone()
        x_mask[mask] = 0.0
        return x_mask, mask

    def forward(self, x: torch.Tensor,
                return_recon: bool = False
                ) -> Tuple[torch.Tensor, ...]:
        """x: (B, T, C) -> zlearned: (B, d_embed)"""
        h = self.input_proj(x) + self.pos_enc[:, :x.size(1)]
        h = self.encoder(h)                           # (B, T, d_model)
        z = self.pool(h.permute(0, 2, 1)).squeeze(-1) # (B, d_model)
        z = self.proj(z)                              # (B, d_embed)
        if return_recon:
            recon = self.recon_head(h)                # (B, T, C)
            return z, recon
        return z

    def ssl_loss(self, x: torch.Tensor) -> torch.Tensor:
        """Masked reconstruction loss for self-supervised pretraining."""
        x = x.float()
        x_mask, mask = self._mask(x)
        h = self.input_proj(x_mask) + self.pos_enc[:, :x.size(1)]
        h = self.encoder(h)
        recon = self.recon_head(h)              # (B, T, C)
        loss  = F.mse_loss(recon[mask], x[mask])
        return loss


# ---------------------------------------------------------------------------
# Branch 3: Graph-Aware Multivariate Dependency Encoder
# ---------------------------------------------------------------------------

class GraphEncoder(nn.Module):
    """
    Builds a lag-aware correlation graph between variables and produces
    a structural embedding via a simple GNN-style message passing.

    G = (V, E): V = variables, edge weights = lag-k Pearson correlations.
    Output: zgraph  (B, dg)
    """

    def __init__(self, d_vars: int, d_model: int = 32, lags: Tuple[int, ...] = (1, 2)):
        super().__init__()
        self.lags      = lags
        self.node_proj = nn.Linear(1, d_model)           # per-variable scalar degree -> feat
        self.msg_proj  = nn.Linear(d_vars * d_model, d_model)
        self.out_proj  = nn.Linear(d_model, d_model)

    @property
    def output_dim(self) -> int:
        return self.out_proj.out_features

    def _build_adj(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute pairwise lag-max-correlation matrix.
        x: (B, T, C) -> adj: (B, C, C)  values in [-1, 1]
        """
        B, T, C = x.shape
        adj = torch.zeros(B, C, C, device=x.device)
        for lag in self.lags:
            if T <= lag:
                continue
            x0  = x[:, :-lag]  - x[:, :-lag].mean(1, keepdim=True)  # (B,T-lag,C)
            x1  = x[:, lag:]   - x[:, lag:].mean(1, keepdim=True)
            num = torch.bmm(x0.permute(0, 2, 1), x1) / (T - lag)    # (B,C,C)
            std = (x0.std(1, keepdim=True).permute(0, 2, 1) *
                   x1.std(1, keepdim=True).permute(0, 2, 1).transpose(1, 2)).clamp(1e-6)
            adj = adj + num / std
        return (adj / len(self.lags)).clamp(-1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, C) -> zgraph: (B, dg)"""
        B, T, C = x.shape
        adj   = self._build_adj(x)                        # (B, C, C)
        # degree as node feature
        deg   = adj.abs().sum(-1, keepdim=True)           # (B, C, 1)
        node  = self.node_proj(deg)                       # (B, C, d_model)
        # one-hop message passing
        msg   = torch.bmm(adj.abs(), node)                # (B, C, d_model)
        msg   = F.gelu(msg).reshape(B, -1)                # (B, C*d_model)
        z     = self.out_proj(F.gelu(self.msg_proj(msg))) # (B, d_model)
        return z


# ---------------------------------------------------------------------------
# Uncertainty descriptor (expert disagreement + variance)
# ---------------------------------------------------------------------------

def compute_uncertainty_descriptor(
    means: torch.Tensor,
    variances: torch.Tensor
) -> torch.Tensor:
    """
    means    : (B, K, pred_len, C)
    variances: (B, K, pred_len, C)
    Returns  : (B, d_unc)  uncertainty descriptor vector
    """
    # predictive variance per expert (mean over pred_len, C)
    avg_var  = variances.mean(-1).mean(-1)   # (B, K)
    # inter-model disagreement
    disagree = means.var(1).mean(-1).mean(-1, keepdim=True).squeeze(-1)  # (B,)  actually scalar
    disagree = disagree.unsqueeze(-1).expand_as(avg_var[:, :1])          # (B, 1)
    return torch.cat([avg_var, disagree], dim=-1)                        # (B, K+1)


# ---------------------------------------------------------------------------
# Full Hybrid Meta-Feature Extractor
# ---------------------------------------------------------------------------

class HybridMetaFeatureExtractor(nn.Module):
    """
    Combines the three branches into a single meta-feature vector x*.

    x* = concat(pool(Zstat), zlearned, zgraph, zuncertainty)
    """

    def __init__(self, input_len: int, d_vars: int,
                 scales: Tuple[int, ...] = (1, 2, 4, 8),
                 lags: Tuple[int, ...] = (1, 2, 4),
                 d_model: int = 64, d_embed: int = 128,
                 graph_d: int = 32):
        super().__init__()
        self.stat_extractor  = MultiScaleStatExtractor(scales, lags)
        self.temporal_enc    = TemporalEncoder(input_len, d_vars, d_model, d_embed)
        self.graph_enc       = GraphEncoder(d_vars, graph_d, lags=(1, 2))

        stat_dim  = self.stat_extractor.output_dim(d_vars)
        learn_dim = self.temporal_enc.output_dim
        graph_dim = self.graph_enc.output_dim
        # uncertainty dim is variable (K+1), handled at fusion time
        self._base_dim = stat_dim + learn_dim + graph_dim

    @property
    def base_output_dim(self) -> int:
        return self._base_dim

    def forward(self, x: torch.Tensor,
                means: torch.Tensor = None,
                variances: torch.Tensor = None
                ) -> torch.Tensor:
        """
        x          : (B, T, d_vars)
        means      : (B, K, pred_len, d_vars) optional expert predictions
        variances  : (B, K, pred_len, d_vars) optional expert uncertainties
        Returns    : x_star: (B, base_dim) or (B, base_dim + K + 1)
        """
        z_stat    = self.stat_extractor(x)   # (B, stat_dim)
        z_learned = self.temporal_enc(x)     # (B, d_embed)
        z_graph   = self.graph_enc(x)        # (B, graph_d)

        parts = [z_stat, z_learned, z_graph]

        if means is not None and variances is not None:
            z_unc = compute_uncertainty_descriptor(means, variances)
            parts.append(z_unc)

        return torch.cat(parts, dim=-1)

    def ssl_pretrain_loss(self, x: torch.Tensor) -> torch.Tensor:
        """SSL masked-reconstruction loss for offline pretraining."""
        return self.temporal_enc.ssl_loss(x)
