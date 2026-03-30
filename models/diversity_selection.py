"""
Diversity-Aware Expert Pool Construction (Section 3.1 of DA-MoE paper).

1. Greedy subset selection that minimises mean pairwise error correlation.
2. Regime specialisation via k-means on meta-feature space.
3. Reliability score tracking: sk ∈ [-1, 1] per expert, updated online.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.cluster import KMeans


# ---------------------------------------------------------------------------
# Pairwise error correlation & greedy subset selection
# ---------------------------------------------------------------------------

def compute_pairwise_error_correlation(
    errors: Dict[str, np.ndarray]
) -> np.ndarray:
    """
    Compute pairwise Pearson correlation matrix over validation errors.

    Parameters
    ----------
    errors : dict  name -> (N,) array of per-sample losses

    Returns
    -------
    rho : (K, K) correlation matrix
    names : list of model names in row/col order
    """
    names = list(errors.keys())
    K     = len(names)
    E     = np.stack([errors[n] for n in names], axis=0)   # (K, N)
    rho   = np.corrcoef(E)                                   # (K, K)
    return rho, names


def greedy_diversity_selection(
    rho: np.ndarray,
    names: List[str],
    max_pool_size: int = 5,
    seed_idx: int = 0
) -> List[str]:
    """
    Greedy subset selection that minimises mean pairwise error correlation.

    Starts from the model at seed_idx and iteratively adds the candidate
    that minimises the average correlation with already-selected models.

    Parameters
    ----------
    rho            : (K, K) correlation matrix
    names          : list of K model names
    max_pool_size  : target pool size
    seed_idx       : index of the first expert to include

    Returns
    -------
    selected : list of selected model names
    """
    K        = len(names)
    selected = [seed_idx]
    remaining = list(set(range(K)) - {seed_idx})

    while len(selected) < max_pool_size and remaining:
        best_idx  = None
        best_corr = float("inf")
        for idx in remaining:
            avg_corr = np.mean([abs(rho[idx, s]) for s in selected])
            if avg_corr < best_corr:
                best_corr = avg_corr
                best_idx  = idx
        selected.append(best_idx)
        remaining.remove(best_idx)

    return [names[i] for i in selected]


# ---------------------------------------------------------------------------
# Regime specialisation via k-means clustering on meta-features
# ---------------------------------------------------------------------------

class RegimeClusterer:
    """
    Clusters the meta-feature space into R regimes using k-means.
    Returns per-sample regime assignments used to train specialist experts.
    """

    def __init__(self, n_regimes: int = 4, random_state: int = 42):
        self.n_regimes    = n_regimes
        self.random_state = random_state
        self.kmeans       = None

    def fit(self, meta_features: np.ndarray):
        """meta_features: (N, d) array of meta-feature vectors."""
        self.kmeans = KMeans(n_clusters=self.n_regimes,
                             random_state=self.random_state, n_init=10)
        self.kmeans.fit(meta_features)
        return self

    def predict(self, meta_features: np.ndarray) -> np.ndarray:
        """Returns (N,) regime labels in {0,...,R-1}."""
        assert self.kmeans is not None, "Call fit() first."
        return self.kmeans.predict(meta_features)

    def regime_sample_weights(self, labels: np.ndarray,
                               regime: int) -> np.ndarray:
        """
        Sampling weights for regime-r specialist training.
        Samples in regime r get weight 1, others get small weight.
        """
        weights = np.where(labels == regime, 1.0, 0.1)
        return weights / weights.sum()


# ---------------------------------------------------------------------------
# Reliability Score Tracker  (Section 3.4)
# ---------------------------------------------------------------------------

class ReliabilityScoreTracker(nn.Module):
    """
    Maintains a bounded reliability score sk ∈ [-1, 1] for each expert.

    Update rule (exponential moving average with tanh activation):
        a_k,t = (b_t - ℓ_k,t) / (b_t + ε)          (normalised advantage)
        s_k,t = clip((1-β)*s_k,t-1 + β*tanh(γ*a_k,t), -1, 1)

    where b_t = median_j(ℓ_j,t) is the robust cross-model baseline.
    """

    def __init__(self, expert_names: List[str],
                 beta: float = 0.05,     # EMA update rate
                 gamma: float = 5.0,     # sensitivity
                 eps: float = 1e-6):
        super().__init__()
        self.expert_names = expert_names
        self.beta         = beta
        self.gamma        = gamma
        self.eps          = eps

        K = len(expert_names)
        self.register_buffer("scores", torch.zeros(K))

    @property
    def score_dict(self) -> Dict[str, float]:
        return {n: self.scores[i].item()
                for i, n in enumerate(self.expert_names)}

    @torch.no_grad()
    def update(self, losses: torch.Tensor):
        """
        losses : (K,) per-expert mean loss on the current batch
        """
        baseline = losses.median()
        adv      = (baseline - losses) / (baseline + self.eps)   # (K,)
        delta    = torch.tanh(self.gamma * adv)
        new_s    = (1 - self.beta) * self.scores + self.beta * delta
        self.scores = new_s.clamp(-1.0, 1.0)

    def get_scores(self) -> torch.Tensor:
        return self.scores.clone()


# ---------------------------------------------------------------------------
# Composite Utility for final expert ranking  (Section 3.4)
# ---------------------------------------------------------------------------

def composite_utility(
    acc_scores:  torch.Tensor,   # (K,)  lower loss → higher acc score
    div_scores:  torch.Tensor,   # (K,)  diversity contribution
    unc_scores:  torch.Tensor,   # (K,)  lower uncertainty is better
    rel_scores:  torch.Tensor,   # (K,)  reliability from tracker
    lambdas: Tuple[float, float, float, float] = (0.4, 0.2, 0.2, 0.2)
) -> torch.Tensor:
    """
    U_k = λ1*Acc_k + λ2*Div_k - λ3*Unc_k + λ4*s_k
    All inputs should be normalised to [0,1] before calling.
    """
    l1, l2, l3, l4 = lambdas
    return l1 * acc_scores + l2 * div_scores - l3 * unc_scores + l4 * rel_scores


def normalise(x: torch.Tensor) -> torch.Tensor:
    """Min-max normalise a 1-D tensor to [0, 1]."""
    lo, hi = x.min(), x.max()
    if (hi - lo).abs() < 1e-8:
        return torch.zeros_like(x)
    return (x - lo) / (hi - lo)


class DiversityAwareModelSelector:
    """
    End-to-end module that:
      1. Runs validation forward pass for all candidate experts.
      2. Computes pairwise error correlations + greedy selection.
      3. Maintains reliability scores updated after each prediction batch.
      4. Selects the top-k experts via the composite utility function.
    """

    def __init__(self, expert_names: List[str],
                 max_pool_size: int = 5,
                 top_k: int = 5,
                 beta: float = 0.05, gamma: float = 5.0,
                 lambdas: Tuple = (0.4, 0.2, 0.2, 0.2)):
        self.expert_names  = expert_names
        self.max_pool_size = max_pool_size
        self.top_k         = top_k
        self.lambdas       = lambdas
        self.tracker       = ReliabilityScoreTracker(expert_names, beta, gamma)
        self.selected_pool: List[str] = expert_names[:top_k]  # initialised to first k

    def build_pool_from_validation(
        self,
        val_losses: Dict[str, np.ndarray]
    ) -> List[str]:
        """
        Given per-sample validation losses per expert, run greedy
        correlation-based subset selection.
        """
        rho, names  = compute_pairwise_error_correlation(val_losses)
        initial     = greedy_diversity_selection(rho, names, self.max_pool_size)
        self.selected_pool = initial
        return initial

    @torch.no_grad()
    def select_top_k(
        self,
        pool_losses: torch.Tensor,   # (K_pool,)  current batch losses for pool experts
        pool_variances: torch.Tensor # (K_pool,)  mean predictive variance per expert
    ) -> Tuple[List[str], torch.Tensor]:
        """
        Select the top-k experts from the current pool using the composite
        utility function. Returns selected names + their reliability scores.
        """
        K_pool     = len(self.selected_pool)
        # Accuracy score = inverted normalised loss
        acc        = normalise(1.0 / (pool_losses + 1e-6))
        # Diversity score = 1 - mean pairwise correlation proxy (uniform for now)
        div        = torch.ones(K_pool, device=pool_losses.device) * 0.5
        # Uncertainty score (lower is better, so we pass raw to composite - it subtracts)
        unc        = normalise(pool_variances)
        # Reliability scores from tracker
        rel_scores = self.tracker.get_scores().to(pool_losses.device)
        # Only keep scores for selected pool
        pool_idxs  = [self.expert_names.index(n) for n in self.selected_pool]
        rel        = normalise((rel_scores[pool_idxs] + 1) / 2)  # map [-1,1] -> [0,1]

        U          = composite_utility(acc, div, unc, rel, self.lambdas)
        top_idxs   = U.topk(min(self.top_k, K_pool)).indices.tolist()
        selected   = [self.selected_pool[i] for i in top_idxs]
        selected_rel = rel_scores[pool_idxs][top_idxs]
        return selected, selected_rel

    def update_scores(self, losses: torch.Tensor):
        """Forward per-batch losses to the reliability score tracker."""
        self.tracker.update(losses)
