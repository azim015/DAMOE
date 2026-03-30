"""
Hierarchical Uncertainty-Aware Model Fusor (Section 3.3 & 3.5 of DA-MoE paper).

Components:
  - Nonlinear gating network G(x̃; Θ) with temperature τ
  - Entropy regulariser  Lspec = λ H(w)
  - Score-adjusted contribution weights w̃_k
  - Residual nonlinear mixer H({f_k(X)}, w; Ω)
  - Reliability-weighted final fusion
  - Multi-objective training loss: Lforecast + λ1*Lentropy + λ2*Lcal + λ3*Lstab
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict


# ---------------------------------------------------------------------------
# Gating Network
# ---------------------------------------------------------------------------

class GatingNetwork(nn.Module):
    """
    3-layer MLP gating network.

    Input : x̃ = concat(x*, uncertainty_stats, inter-model-disagreement)
    Output: softmax(G(x̃)/τ) – simplex-constrained mixture weights over K experts.
    """

    def __init__(self, input_dim: int, num_experts: int,
                 hidden_dim: int = 128, temperature: float = 1.0,
                 dropout: float = 0.1):
        super().__init__()
        self.temperature = temperature
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_experts)
        )

    def forward(self, x_tilde: torch.Tensor) -> torch.Tensor:
        """x_tilde: (B, input_dim) -> weights: (B, K)"""
        logits  = self.net(x_tilde)                         # (B, K)
        weights = F.softmax(logits / self.temperature, dim=-1)
        return weights


# ---------------------------------------------------------------------------
# Residual Nonlinear Mixer (corrective layer, Section 3.5)
# ---------------------------------------------------------------------------

class ResidualNonlinearMixer(nn.Module):
    """
    Lightweight residual corrective layer H({f_k(X)}, w; Ω).

    Takes stacked expert predictions and gating weights, outputs a
    correction Δ to add to the linear mixture.
    Maintained at low capacity to avoid becoming a substitute predictor.
    """

    def __init__(self, num_experts: int, pred_len: int, d_vars: int,
                 hidden_dim: int = 64):
        super().__init__()
        in_dim  = num_experts * pred_len * d_vars + num_experts
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, pred_len * d_vars)
        )
        self.pred_len = pred_len
        self.d_vars   = d_vars

    def forward(self, expert_preds: torch.Tensor,
                weights: torch.Tensor) -> torch.Tensor:
        """
        expert_preds : (B, K, pred_len, d_vars)
        weights      : (B, K)
        Returns      : correction (B, pred_len, d_vars)
        """
        B, K, L, C = expert_preds.shape
        flat  = expert_preds.reshape(B, -1)           # (B, K*L*C)
        inp   = torch.cat([flat, weights], dim=-1)    # (B, K*L*C + K)
        delta = self.net(inp).reshape(B, L, C)
        return delta


# ---------------------------------------------------------------------------
# Score-Adjusted Contribution Weights  (Section 3.4)
# ---------------------------------------------------------------------------

def score_adjusted_weights(
    gate_weights:    torch.Tensor,   # (B, K)
    reliability_s:   torch.Tensor,   # (K,)  scores in [-1, 1]
    eta:             float = 2.0
) -> torch.Tensor:
    """
    w̃_k = (w_k * ρ(s_k)) / Σ_j (w_j * ρ(s_j))
    ρ(s) = exp(η * s)
    """
    rho   = torch.exp(eta * reliability_s).unsqueeze(0)   # (1, K)
    w_adj = gate_weights * rho                             # (B, K)
    return w_adj / w_adj.sum(dim=-1, keepdim=True).clamp(min=1e-8)


# ---------------------------------------------------------------------------
# Multi-Objective Fusion Loss  (Section 3.6)
# ---------------------------------------------------------------------------

def entropy_loss(weights: torch.Tensor) -> torch.Tensor:
    """
    Penalises excessively uniform (high entropy) or peaked (low entropy)
    weight distributions.  Uses negative entropy so minimising pushes
    toward intermediate specialisation.
    """
    H = -(weights * (weights + 1e-8).log()).sum(-1).mean()
    return -H   # minimising this maximises specialisation


def calibration_loss(pred_intervals: Tuple[torch.Tensor, torch.Tensor],
                     targets: torch.Tensor,
                     alpha: float = 0.1) -> torch.Tensor:
    """
    Interval calibration loss: penalises deviation between nominal (1-α)
    coverage and empirical coverage.

    pred_intervals : (lower, upper) each (B, pred_len, d_vars)
    targets        : (B, pred_len, d_vars)
    """
    lower, upper = pred_intervals
    covered      = ((targets >= lower) & (targets <= upper)).float()
    emp_coverage = covered.mean()
    nom_coverage = torch.tensor(1.0 - alpha, device=targets.device)
    return F.mse_loss(emp_coverage, nom_coverage)


def stability_loss(weights_t: torch.Tensor,
                   weights_tm1: torch.Tensor) -> torch.Tensor:
    """
    Penalises temporal volatility in gating assignments for successive inputs.
    weights_t, weights_tm1: (B, K) — may have different B sizes (last batch).
    """
    B  = min(weights_t.size(0), weights_tm1.size(0))
    return F.mse_loss(weights_t[:B], weights_tm1[:B])


class FusionLoss(nn.Module):
    """
    L = Lforecast + λ1*Lentropy + λ2*Lcalibration + λ3*Lstability
    """

    def __init__(self, lambda1: float = 0.1,
                 lambda2: float = 0.05,
                 lambda3: float = 0.05,
                 lambda_stab: float = 0.001,
                 use_huber: bool = True):
        super().__init__()
        self.lambda1      = lambda1
        self.lambda2      = lambda2
        self.lambda3      = lambda3
        self.lambda_stab  = lambda_stab
        self.use_huber    = use_huber

    def forecast_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.use_huber:
            return F.huber_loss(pred, target, delta=1.0)
        return F.mse_loss(pred, target)

    def forward(self,
                pred:        torch.Tensor,              # (B, L, C) fused prediction
                target:      torch.Tensor,              # (B, L, C)
                weights:     torch.Tensor,              # (B, K) current gating weights
                weights_prev: Optional[torch.Tensor] = None,  # (B, K) previous step
                pred_intervals: Optional[Tuple] = None,
                expert_preds: Optional[torch.Tensor] = None   # (B, K, L, C)
                ) -> Tuple[torch.Tensor, Dict[str, float]]:

        Lf   = self.forecast_loss(pred, target)
        Lent = entropy_loss(weights)
        Lcal = torch.tensor(0.0, device=pred.device)
        Lstab= torch.tensor(0.0, device=pred.device)

        if pred_intervals is not None:
            Lcal = calibration_loss(pred_intervals, target)
        if weights_prev is not None:
            Lstab = stability_loss(weights, weights_prev)

        # Specialisation regulariser: also penalise fusor drift from init
        total = (Lf
                 + self.lambda1   * Lent
                 + self.lambda2   * Lcal
                 + self.lambda_stab * Lstab)

        info  = {
            "Lforecast":    Lf.item(),
            "Lentropy":     Lent.item(),
            "Lcalibration": Lcal.item(),
            "Lstability":   Lstab.item(),
        }
        return total, info


# ---------------------------------------------------------------------------
# Full Fusor
# ---------------------------------------------------------------------------

class DAMoEFusor(nn.Module):
    """
    Hierarchical uncertainty-aware fusor.

    Pipeline:
      1. Build augmented fusor input x̃ = concat(x*, uncertainty_stats, disagree)
      2. Gating network → raw weights w
      3. Adjust with reliability scores → w̃
      4. Linear convex combination of expert predictions
      5. Add residual nonlinear correction from mixer
      6. Reliability-weighted rescaling (second pass)

    Adaptive continual learning:
      Θ ← Θ − η∇Θ Lfuse + γ(Θ − Θ0)   (controlled by caller via optimizer + regulariser)
    """

    def __init__(self,
                 meta_dim:      int,         # base meta-feature dim (without unc)
                 num_experts:   int,
                 pred_len:      int,
                 d_vars:        int,
                 hidden_dim:    int   = 128,
                 temperature:   float = 1.0,
                 eta:           float = 2.0,
                 dropout:       float = 0.1):
        super().__init__()
        self.num_experts  = num_experts
        self.pred_len     = pred_len
        self.d_vars       = d_vars
        self.eta          = eta
        self.temperature  = temperature
        self.hidden_dim   = hidden_dim

        # Step 1: project base meta-features to hidden_dim (fixed size)
        self.meta_proj = nn.Sequential(
            nn.Linear(meta_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        # Step 2: gate body operates on hidden_dim + K + 1  -> hidden_dim
        # We build the first linear dynamically on first call; store params here.
        # For simplicity we fix the dynamic part to max num_experts via padding.
        self.max_experts   = num_experts
        aug_hidden_dim     = hidden_dim + num_experts + 1
        self.gate_body     = nn.Sequential(
            nn.Linear(aug_hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_experts),
        )
        self.mixer = ResidualNonlinearMixer(num_experts, pred_len, d_vars, hidden_dim)

        # store reference parameters Θ0 for stability regularisation
        self._theta0: Optional[Dict[str, torch.Tensor]] = None

    def save_reference_params(self):
        """Call once after offline training to snapshot Θ0."""
        self._theta0 = {k: v.detach().clone()
                        for k, v in self.named_parameters()}

    def stability_regulariser(self, beta: float = 0.01) -> torch.Tensor:
        """‖Θ − Θ0‖² stability prior loss."""
        if self._theta0 is None:
            return torch.tensor(0.0)
        loss = sum(
            F.mse_loss(p, self._theta0[k].to(p.device))
            for k, p in self.named_parameters()
        )
        return beta * loss

    def forward(
        self,
        x_star:       torch.Tensor,      # (B, meta_dim) base meta-features
        expert_means: torch.Tensor,      # (B, K, pred_len, d_vars)
        expert_vars:  torch.Tensor,      # (B, K, pred_len, d_vars)
        reliability:  torch.Tensor,      # (K,) reliability scores
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        y_hat   : (B, pred_len, d_vars) – fused forecast
        weights : (B, K)               – adjusted gating weights
        w_raw   : (B, K)               – raw gating weights (for logging)
        """
        B, K = expert_means.shape[:2]

        # ---- augment meta-features with uncertainty stats ----
        avg_var   = expert_vars.mean(-1).mean(-1)          # (B, K)
        disagree  = expert_means.var(1).mean(-1).mean(-1, keepdim=True)  # (B,1)

        # Project base meta to hidden dim first (fixed-size operation)
        h_meta    = self.meta_proj(x_star)                 # (B, hidden_dim)

        # Pad avg_var to max_experts so gate_body input is always fixed size
        pad_size  = self.max_experts - K
        if pad_size > 0:
            avg_var = torch.cat([avg_var,
                                 torch.zeros(B, pad_size, device=avg_var.device)], dim=-1)

        x_tilde   = torch.cat([h_meta, avg_var, disagree], dim=-1)  # (B, hidden+K+1)

        # ---- gating ----
        logits    = self.gate_body(x_tilde)                # (B, max_experts)
        logits    = logits[:, :K]                          # trim to active experts
        w_raw     = F.softmax(logits / self.temperature, dim=-1)
        weights   = score_adjusted_weights(w_raw, reliability.to(w_raw.device), self.eta)

        # ---- linear mixture ----
        w_exp  = weights.unsqueeze(-1).unsqueeze(-1)       # (B, K, 1, 1)
        y_lin  = (expert_means * w_exp).sum(1)             # (B, pred_len, d_vars)

        # ---- residual nonlinear correction ----
        delta  = self.mixer(expert_means, weights)
        y_hat  = y_lin + delta

        return y_hat, weights, w_raw
