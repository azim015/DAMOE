"""
Data utilities for DA-MoE experiments.

Provides:
  - Synthetic time-series dataset generators (matching ETT-like structure)
  - Dataset normalisation
  - DataLoader creation
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Optional, List


# ---------------------------------------------------------------------------
# Normalisers
# ---------------------------------------------------------------------------

class StandardScaler:
    def __init__(self):
        self.mean = None
        self.std  = None

    def fit(self, data: np.ndarray):
        self.mean = data.mean(0)
        self.std  = data.std(0) + 1e-8
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        return (data - self.mean) / self.std

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        return data * self.std + self.mean


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class TimeSeriesDataset(Dataset):
    """
    Sliding-window dataset for time-series forecasting.

    Parameters
    ----------
    data      : (N, C) numpy array – full time series
    input_len : look-back window length
    pred_len  : forecast horizon
    scaler    : optional fitted scaler applied in __getitem__
    """

    def __init__(self, data: np.ndarray,
                 input_len: int = 96,
                 pred_len:  int = 96,
                 scaler:    Optional[StandardScaler] = None):
        self.data      = data.astype(np.float32)
        self.input_len = input_len
        self.pred_len  = pred_len
        self.scaler    = scaler

    def __len__(self) -> int:
        return len(self.data) - self.input_len - self.pred_len + 1

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.data[idx          : idx + self.input_len]
        y = self.data[idx + self.input_len : idx + self.input_len + self.pred_len]
        if self.scaler is not None:
            x = self.scaler.transform(x)
            y = self.scaler.transform(y)
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Synthetic data generators matching the paper's benchmark characteristics
# ---------------------------------------------------------------------------

def synthetic_ett_like(
    n_samples:  int   = 2000,
    n_vars:     int   = 7,
    seed:       int   = 42,
    noise:      float = 0.05,
    n_periods:  int   = 3
) -> np.ndarray:
    """
    ETT-like multivariate time series with:
      - Multiple seasonal components
      - Long-range trends
      - Cross-variable correlations
      - Mild nonstationarity
    """
    rng  = np.random.RandomState(seed)
    t    = np.linspace(0, 4 * np.pi, n_samples)
    data = np.zeros((n_samples, n_vars))

    periods = [24, 48, 168][:n_periods]
    for c in range(n_vars):
        phase = rng.uniform(0, 2 * np.pi, n_periods)
        amp   = rng.uniform(0.5, 2.0,     n_periods)
        sig   = sum(amp[i] * np.sin(2 * np.pi * t / periods[i % n_periods] + phase[i])
                    for i in range(n_periods))
        trend = 0.01 * c * t / (4 * np.pi)
        sig  += trend + rng.randn(n_samples) * noise
        data[:, c] = sig

    # cross-variable correlations
    mixing = rng.randn(n_vars, n_vars) * 0.3
    np.fill_diagonal(mixing, 1.0)
    data = data @ mixing
    return data


def synthetic_traffic_like(
    n_samples: int   = 2000,
    n_vars:    int   = 7,
    seed:      int   = 0
) -> np.ndarray:
    """
    Traffic-like series with rapid shifts (rush-hour bursts).
    """
    rng  = np.random.RandomState(seed)
    t    = np.arange(n_samples)
    data = np.zeros((n_samples, n_vars))
    for c in range(n_vars):
        base  = 0.5 + 0.3 * np.sin(2 * np.pi * t / 24 + rng.uniform(0, 2*np.pi))
        burst = (np.sin(2 * np.pi * t / 24) > 0.8).astype(float) * rng.uniform(0.5, 1.5)
        noise = rng.randn(n_samples) * 0.05
        data[:, c] = base + burst + noise
    return data


def synthetic_electricity_like(
    n_samples: int   = 2000,
    n_vars:    int   = 7,
    seed:      int   = 1
) -> np.ndarray:
    """
    Electricity-like series with daily and weekly seasonality.
    """
    rng  = np.random.RandomState(seed)
    t    = np.arange(n_samples)
    data = np.zeros((n_samples, n_vars))
    for c in range(n_vars):
        daily  = np.sin(2 * np.pi * t / 24  + rng.uniform(0, 2*np.pi))
        weekly = np.sin(2 * np.pi * t / 168 + rng.uniform(0, 2*np.pi)) * 0.5
        spikes = (rng.rand(n_samples) < 0.02) * rng.exponential(2, n_samples)
        noise  = rng.randn(n_samples) * 0.1
        data[:, c] = daily + weekly + spikes + noise
    return data


DATASET_GENERATORS = {
    "ETTh1":       lambda: synthetic_ett_like(2000, 7, 42),
    "ETTh2":       lambda: synthetic_ett_like(2000, 7, 43, noise=0.1),
    "ETTm1":       lambda: synthetic_ett_like(4000, 7, 44),
    "ETTm2":       lambda: synthetic_ett_like(4000, 7, 45, noise=0.08),
    "Traffic":     lambda: synthetic_traffic_like(2000, 7, 0),
    "Electricity": lambda: synthetic_electricity_like(2000, 7, 1),
    "Weather":     lambda: synthetic_ett_like(2000, 21, 100, noise=0.15, n_periods=3),
    "Solar":       lambda: synthetic_electricity_like(2000, 7, 99),
}


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def make_loaders(
    dataset_name: str   = "ETTh1",
    input_len:    int   = 96,
    pred_len:     int   = 96,
    batch_size:   int   = 32,
    train_ratio:  float = 0.7,
    val_ratio:    float = 0.1,
) -> Tuple[DataLoader, DataLoader, DataLoader, StandardScaler]:
    """
    Returns train / val / test DataLoaders and the fitted scaler.
    """
    data = DATASET_GENERATORS.get(dataset_name, DATASET_GENERATORS["ETTh1"])()

    N     = len(data)
    n_tr  = int(N * train_ratio)
    n_val = int(N * val_ratio)

    train_raw = data[:n_tr]
    val_raw   = data[n_tr : n_tr + n_val]
    test_raw  = data[n_tr + n_val:]

    scaler = StandardScaler().fit(train_raw)

    train_ds = TimeSeriesDataset(train_raw, input_len, pred_len, scaler)
    val_ds   = TimeSeriesDataset(val_raw,   input_len, pred_len, scaler)
    test_ds  = TimeSeriesDataset(test_raw,  input_len, pred_len, scaler)

    kw = dict(batch_size=batch_size, num_workers=0, pin_memory=False)
    return (
        DataLoader(train_ds, shuffle=True,  **kw),
        DataLoader(val_ds,   shuffle=False, **kw),
        DataLoader(test_ds,  shuffle=False, **kw),
        scaler
    )
