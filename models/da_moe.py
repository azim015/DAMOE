"""
DA-MoE: Diversity-Aware Mixture-of-Experts for Time-Series Forecasting
(Full model integration – Algorithm 1 from the paper)

Architecture:
  1. Expert pool (diverse backbones, shared encoder + LoRA adapters)
  2. Hybrid multi-scale meta-feature extractor
  3. Diversity-aware model selector (error correlation + reliability scoring)
  4. Uncertainty-aware fusor with nonlinear residual mixer
  5. Reliability-driven feedback mechanism

Usage:
    model = DAMoE(input_len=96, pred_len=96, d_vars=7)
    y_hat, info = model(x)   # x: (B, T, C)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple

from .experts          import build_expert_pool, EXPERT_CLASSES
from .meta_features    import HybridMetaFeatureExtractor
from .diversity_selection import (DiversityAwareModelSelector,
                                   ReliabilityScoreTracker,
                                   greedy_diversity_selection,
                                   compute_pairwise_error_correlation,
                                   normalise)
from .fusor            import DAMoEFusor, FusionLoss


class DAMoE(nn.Module):
    """
    Full DA-MoE model.

    Parameters
    ----------
    input_len     : look-back window length T_in
    pred_len      : forecast horizon T_out
    d_vars        : number of variables (channels)
    d_model       : shared backbone hidden size
    d_meta_embed  : temporal encoder output dimension
    graph_d       : graph encoder output dimension
    hidden_fusor  : fusor MLP hidden dimension
    max_pool_size : maximum number of experts in the selected pool
    top_k         : number of experts activated per forward pass
    scales        : temporal scales for multi-scale stat extractor
    lags          : autocorrelation lags
    temperature   : gating softmax temperature τ
    eta           : score-adjustment exponent η
    mc_samples    : MC-Dropout samples for uncertainty (0 = disabled)
    dropout       : dropout rate
    device        : computation device
    """

    def __init__(self,
                 input_len:     int   = 96,
                 pred_len:      int   = 96,
                 d_vars:        int   = 7,
                 d_model:       int   = 64,
                 d_meta_embed:  int   = 128,
                 graph_d:       int   = 32,
                 hidden_fusor:  int   = 128,
                 max_pool_size: int   = 5,
                 top_k:         int   = 5,
                 scales:        Tuple = (1, 2, 4, 8),
                 lags:          Tuple = (1, 2, 4),
                 temperature:   float = 1.0,
                 eta:           float = 2.0,
                 mc_samples:    int   = 0,
                 dropout:       float = 0.1,
                 device:        str   = "cpu"):
        super().__init__()
        self.input_len    = input_len
        self.pred_len     = pred_len
        self.d_vars       = d_vars
        self.top_k        = top_k
        self.device_str   = device

        # ---- Expert pool ------------------------------------------------
        self.expert_pool  = build_expert_pool(
            input_len, pred_len, d_vars, d_model, mc_samples, device
        )
        self.expert_names = list(self.expert_pool.keys())
        self.K_total      = len(self.expert_names)

        # ---- Meta-feature extractor -------------------------------------
        self.meta_extractor = HybridMetaFeatureExtractor(
            input_len, d_vars,
            scales=scales, lags=lags,
            d_model=d_model, d_embed=d_meta_embed,
            graph_d=graph_d
        )
        base_meta_dim = self.meta_extractor.base_output_dim

        # ---- Diversity-aware model selector -----------------------------
        self.selector = DiversityAwareModelSelector(
            self.expert_names, max_pool_size=max_pool_size, top_k=top_k
        )
        # Initialise selected pool as all experts
        self.selected_pool: List[str] = self.expert_names[:top_k]

        # ---- Fusor ------------------------------------------------------
        # meta_dim does NOT include the per-expert-unc (added inside fusor).
        # Fusor internally builds aug_dim = meta_dim + num_experts + 1.
        self.fusor = DAMoEFusor(
            meta_dim    = base_meta_dim,
            num_experts = top_k,
            pred_len    = pred_len,
            d_vars      = d_vars,
            hidden_dim  = hidden_fusor,
            temperature = temperature,
            eta         = eta,
            dropout     = dropout
        )

        # ---- Fusion loss ------------------------------------------------
        self.fusion_loss_fn = FusionLoss()

        # ---- State for stability regularisation -------------------------
        self._prev_weights: Optional[torch.Tensor] = None

        self.to(device)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _run_experts(self, x: torch.Tensor,
                     names: List[str]
                     ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Run the specified subset of experts.
        Returns:
            means : (B, K, pred_len, d_vars)
            vars  : (B, K, pred_len, d_vars)
        """
        means, variances = [], []
        for name in names:
            mu, sigma2 = self.expert_pool[name](x)
            means.append(mu)
            variances.append(sigma2)
        return torch.stack(means, dim=1), torch.stack(variances, dim=1)

    def _get_reliability(self, names: List[str]) -> torch.Tensor:
        """Extract reliability scores for the given expert subset."""
        all_scores = self.selector.tracker.get_scores()
        idxs       = [self.expert_names.index(n) for n in names]
        return all_scores[idxs]

    # ------------------------------------------------------------------
    # Algorithm 1: High-Level Inference
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor,
                target: Optional[torch.Tensor] = None,
                update_scores: bool = False
                ) -> Tuple[torch.Tensor, Dict]:
        """
        Parameters
        ----------
        x            : (B, T_in, d_vars)
        target       : (B, T_out, d_vars)  optional, for score updates
        update_scores: if True and target is given, update reliability scores

        Returns
        -------
        y_hat : (B, pred_len, d_vars)
        info  : dict with diagnostics (weights, selected experts, …)
        """

        # Step 1: Compute dynamic meta-representation (without uncertainty first)
        x_star_base = self.meta_extractor(x)   # (B, base_meta_dim)

        # Step 2: Select candidate models (top-k from current pool)
        active_names = self.selected_pool

        # Step 3: Run experts to get predictions + uncertainty
        means, variances = self._run_experts(x, active_names)  # (B,K,L,C) each

        # Step 4: Get reliability scores for active experts
        reliability = self._get_reliability(active_names)      # (K,)

        # Step 5: Fuse predictions
        y_hat, weights, w_raw = self.fusor(
            x_star_base, means, variances, reliability
        )

        # Step 6: Feedback – update reliability scores
        if update_scores and target is not None:
            with torch.no_grad():
                # losses: per-expert per-sample: mean over (L, C) -> (B, K)
                losses = F.huber_loss(
                    means, target.unsqueeze(1).expand_as(means),
                    reduction="none", delta=1.0
                ).mean(dim=(-1, -2))                  # (B, K)
                expert_losses = losses.mean(0)        # (K,)
                # handle K=1 edge case
                if expert_losses.dim() == 0:
                    expert_losses = expert_losses.unsqueeze(0)
                # map back to full expert list for tracker update
                full_losses = torch.zeros(self.K_total, device=x.device)
                for i, name in enumerate(active_names):
                    full_losses[self.expert_names.index(name)] = expert_losses[i]
                self.selector.update_scores(full_losses)

        info = {
            "selected_experts": active_names,
            "gate_weights":     weights.detach(),
            "raw_weights":      w_raw.detach(),
            "reliability":      reliability.detach(),
            "expert_means":     means.detach(),
            "expert_variances": variances.detach(),
        }
        return y_hat, info

    # ------------------------------------------------------------------
    # Training loss (called by the training loop)
    # ------------------------------------------------------------------

    def compute_loss(self,
                     x:      torch.Tensor,
                     target: torch.Tensor
                     ) -> Tuple[torch.Tensor, Dict]:
        """Full forward + multi-objective fusion loss."""
        y_hat, info = self.forward(x, target, update_scores=True)

        weights     = info["gate_weights"]
        prev_w      = self._prev_weights
        loss, linfo = self.fusion_loss_fn(
            y_hat, target, weights, weights_prev=prev_w
        )
        # stability prior from fusor
        loss = loss + self.fusor.stability_regulariser()
        self._prev_weights = weights.detach()

        return loss, {**linfo, "total": loss.item()}

    # ------------------------------------------------------------------
    # Offline: build expert pool from validation errors
    # ------------------------------------------------------------------

    @torch.no_grad()
    def build_pool_from_validation(
        self,
        val_loader,
        device: Optional[str] = None
    ):
        """
        1. Run all experts on the validation set.
        2. Collect per-sample losses.
        3. Compute pairwise error correlations.
        4. Run greedy diversity selection.
        5. Update self.selected_pool.
        """
        import numpy as np
        device = device or self.device_str
        losses = {n: [] for n in self.expert_names}

        for x_batch, y_batch in val_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            for name in self.expert_names:
                mu, _ = self.expert_pool[name](x_batch)
                l     = F.huber_loss(mu, y_batch, reduction="none",
                                     delta=1.0).mean((-1, -2))  # (B,)
                losses[name].append(l.cpu().numpy())

        val_losses = {n: np.concatenate(losses[n]) for n in self.expert_names}
        self.selected_pool = self.selector.build_pool_from_validation(val_losses)
        print(f"[DA-MoE] Selected expert pool: {self.selected_pool}")

    # ------------------------------------------------------------------
    # Adaptive fusor update (online continual learning)
    # ------------------------------------------------------------------

    def update_fusor(self,
                     x:        torch.Tensor,
                     target:   torch.Tensor,
                     optimizer: torch.optim.Optimizer,
                     beta:      float = 0.001) -> float:
        """
        Single gradient step on the fusor with stability prior.
        Implements: Θ ← Θ − η∇Θ Lfuse + γ(Θ − Θ0)
        """
        optimizer.zero_grad()
        loss, info = self.compute_loss(x, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.fusor.parameters(), max_norm=1.0)
        optimizer.step()
        return info["total"]


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

def build_da_moe(config: Dict, device: str = "cpu") -> DAMoE:
    """Build DA-MoE from a flat config dict."""
    return DAMoE(
        input_len    = config.get("input_len",     96),
        pred_len     = config.get("pred_len",      96),
        d_vars       = config.get("d_vars",         7),
        d_model      = config.get("d_model",        64),
        d_meta_embed = config.get("d_meta_embed",  128),
        graph_d      = config.get("graph_d",        32),
        hidden_fusor = config.get("hidden_fusor",  128),
        max_pool_size= config.get("max_pool_size",   5),
        top_k        = config.get("top_k",           5),
        scales       = config.get("scales",        (1, 2, 4, 8)),
        lags         = config.get("lags",          (1, 2, 4)),
        temperature  = config.get("temperature",   1.0),
        eta          = config.get("eta",           2.0),
        mc_samples   = config.get("mc_samples",      0),
        dropout      = config.get("dropout",        0.1),
        device       = device
    )
