"""
EDA 06 — Add a KDE density feature to the per-zone data (Part II, Hypothesis 2)
=============================================================================
For each zone, fit a KDE on the *training pool* and attach a standardized
log-density `log p(x, y)` as a third node feature. This is the defensible use of
a KDE: not to *generate* data (H1 samples real points directly), but to *describe*
local density so the model knows whether a node sits in a dense or sparse area.

Writes back into the existing files (H1 keys are preserved):
  artifacts/instances/olist_sp_c{k}_train_pool.pt
      + coords_dens (M, 3)  = [x, y, density_z],  depot_dens (3,)
  artifacts/instances/olist_sp_c{k}_vrp20_heldout_n*.pt
      + coords_dens (B, N, 3)

Density is fit on the TRAIN pool only and evaluated on the held-out nodes — no
leakage. Standardization (z-score) uses the train-pool stats for both.

Run:
    uv run eda/06_add_density.py
"""

import numpy as np
import torch
from pathlib import Path
from sklearn.neighbors import KernelDensity

INST = Path("artifacts/instances")
CLUSTERS = [1, 2, 3, 4, 5]
BANDWIDTH = 0.05   # smooth enough to encode the density gradient (not a spiky per-point spike)

for c in CLUSTERS:
    pool_path = INST / f"olist_sp_c{c}_train_pool.pt"
    eval_matches = sorted(INST.glob(f"olist_sp_c{c}_vrp20_heldout_n*_seed*.pt"))
    if not pool_path.exists() or not eval_matches:
        print(f"C{c}: missing pool or eval file — run eda/05_split_train_test.py first; skipping")
        continue
    eval_path = eval_matches[0]

    # --- Fit KDE on the TRAIN pool ---
    pool = torch.load(pool_path, map_location="cpu", weights_only=False)
    coords = pool["coords"].numpy().astype(np.float64)     # (M, 2)
    depot  = pool["depot"].numpy().astype(np.float64)      # (2,)

    kde = KernelDensity(kernel="gaussian", bandwidth=BANDWIDTH).fit(coords)
    dens = kde.score_samples(coords)                        # (M,) log-density
    mu, sd = float(dens.mean()), float(dens.std() + 1e-8)   # standardization stats (train only)

    def zdens(pts):
        return (kde.score_samples(pts.reshape(-1, 2)) - mu) / sd

    # --- Train pool: coords + standardized density ---
    pool_z = zdens(coords)
    coords_dens = np.concatenate([coords, pool_z[:, None]], axis=1).astype(np.float32)   # (M, 3)
    depot_dens = np.array([depot[0], depot[1], zdens(depot[None])[0]], dtype=np.float32)  # (3,)
    pool["coords_dens"] = torch.from_numpy(coords_dens)
    pool["depot_dens"]  = torch.from_numpy(depot_dens)
    pool["density_bw"], pool["density_mu"], pool["density_sd"] = BANDWIDTH, mu, sd
    torch.save(pool, pool_path)

    # --- Held-out eval: coords + standardized density (same KDE) ---
    ev = torch.load(eval_path, map_location="cpu", weights_only=False)
    ec = ev["coords"].numpy().astype(np.float64)           # (B, N, 2)
    B, N, _ = ec.shape
    ec_z = zdens(ec).reshape(B, N)
    ev["coords_dens"] = torch.from_numpy(
        np.concatenate([ec.astype(np.float32), ec_z[..., None].astype(np.float32)], axis=-1)  # (B, N, 3)
    )
    torch.save(ev, eval_path)

    print(f"C{c}: pool {coords.shape}+dens, eval {ec.shape}+dens  "
          f"(bw={BANDWIDTH}, density z-range [{pool_z.min():.2f}, {pool_z.max():.2f}])")

print("\nDone. Train with: uv run experiments/part2_sp/train_zone_density.py")
