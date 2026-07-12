"""
KDE wrapper for VRP instance generation.

Wraps sklearn KernelDensity to expose a .sample(n) -> (n, 2) interface
that clips samples to [0,1]², resampling outliers until the quota is filled.
"""

import numpy as np
from sklearn.neighbors import KernelDensity


class SPKDE:
    """
    Fitted KDE on normalized SP customer coordinates.

    Attributes:
        kde:       fitted sklearn KernelDensity
        bandwidth: bandwidth used
        n_train:   number of points used for fitting
    """

    def __init__(self, kde: KernelDensity, bandwidth: float, n_train: int):
        self.kde       = kde
        self.bandwidth = bandwidth
        self.n_train   = n_train

    def sample(self, n: int, max_tries: int = 20) -> np.ndarray:
        """
        Sample n points in [0,1]².
        Oversamples and filters; retries if needed.
        Returns (n, 2) float32.
        """
        collected = []
        remaining = n
        for _ in range(max_tries):
            raw = self.kde.sample(remaining * 3)          # oversample 3x
            valid = raw[(raw >= 0).all(axis=1) & (raw <= 1).all(axis=1)]
            collected.append(valid[:remaining])
            remaining -= len(valid[:remaining])
            if remaining <= 0:
                break

        pts = np.vstack(collected)[:n]

        # Fallback: fill any missing slots with uniform samples
        if len(pts) < n:
            fill = np.random.uniform(0, 1, (n - len(pts), 2))
            pts  = np.vstack([pts, fill])

        return pts.astype(np.float32)

    def __repr__(self):
        return (f"SPKDE(bandwidth={self.bandwidth:.4f}, "
                f"n_train={self.n_train:,})")
