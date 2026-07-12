"""
EDA 05 — Fit KDE on SP Customer Coordinates
=============================================
Fits a Kernel Density Estimator on the normalized lat/lng of SP customers.
The resulting SPKDE object is used during model training (distribution="kde").

Output:
    models/sp_kde.pkl       — fitted SPKDE object (used by trainer)
    eda/plots/05_kde.png    — real data vs KDE samples (visual check)

Run:
    uv run eda/05_fit_kde.py
"""

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.kde_utils import SPKDE

DATA_DIR  = Path("data/olist")
PLOTS_DIR = Path("eda/plots")
MODELS_DIR = Path("models")
PLOTS_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

# ------------------------------------------------------------------
# 1. Load SP customer coordinates
# ------------------------------------------------------------------
customers = pd.read_csv(DATA_DIR / "olist_customers_dataset.csv")
geoloc    = pd.read_csv(DATA_DIR / "olist_geolocation_dataset.csv")

sp_customers = customers[
    customers["customer_city"].str.strip().str.lower() == "sao paulo"
].copy()

geo_mean = (
    geoloc[geoloc["geolocation_state"] == "SP"]
    .groupby("geolocation_zip_code_prefix")[["geolocation_lat", "geolocation_lng"]]
    .mean()
    .reset_index()
    .rename(columns={
        "geolocation_zip_code_prefix": "customer_zip_code_prefix",
        "geolocation_lat": "lat",
        "geolocation_lng": "lng",
    })
)

sp = sp_customers.merge(geo_mean, on="customer_zip_code_prefix", how="inner")
sp = sp[(sp["lat"].between(-24.0, -23.3)) & (sp["lng"].between(-46.9, -46.3))]

print(f"SP customers loaded : {len(sp):,}")

# ------------------------------------------------------------------
# 2. Normalize coordinates to [0,1]² (global SP bounding box)
# ------------------------------------------------------------------
LAT_MIN, LAT_MAX = sp["lat"].min(), sp["lat"].max()
LNG_MIN, LNG_MAX = sp["lng"].min(), sp["lng"].max()

PAD = 0.02  # 2% padding so extreme points aren't right on the border
lat_lo = LAT_MIN - PAD * (LAT_MAX - LAT_MIN)
lat_hi = LAT_MAX + PAD * (LAT_MAX - LAT_MIN)
lng_lo = LNG_MIN - PAD * (LNG_MAX - LNG_MIN)
lng_hi = LNG_MAX + PAD * (LNG_MAX - LNG_MIN)

x_norm = (sp["lng"].values - lng_lo) / (lng_hi - lng_lo)  # lng -> x
y_norm = (sp["lat"].values - lat_lo) / (lat_hi - lat_lo)  # lat -> y

coords_norm = np.stack([x_norm, y_norm], axis=1)  # (N, 2)
print(f"Normalized range    : x=[{x_norm.min():.3f}, {x_norm.max():.3f}]  "
      f"y=[{y_norm.min():.3f}, {y_norm.max():.3f}]")

# ------------------------------------------------------------------
# 3. Select bandwidth via cross-validation (grid search)
# ------------------------------------------------------------------
print("\nSearching bandwidth via CV (this may take ~1 min)...")
bandwidths = np.logspace(-2.5, -0.5, 15)  # ~0.003 to 0.32
grid = GridSearchCV(
    KernelDensity(kernel="gaussian"),
    {"bandwidth": bandwidths},
    cv=5,
    n_jobs=-1,
)
grid.fit(coords_norm)
best_bw = grid.best_params_["bandwidth"]
print(f"Best bandwidth      : {best_bw:.4f}")

# ------------------------------------------------------------------
# 4. Fit KDEs: base (best_bw) and bw2x (2 * best_bw)
# ------------------------------------------------------------------
variants = {
    "sp_kde":      best_bw,
    "sp_kde_bw2x": best_bw * 2,
}

kdes = {}
for name, bw in variants.items():
    kde_sk = KernelDensity(kernel="gaussian", bandwidth=bw)
    kde_sk.fit(coords_norm)
    kdes[name] = SPKDE(kde=kde_sk, bandwidth=bw, n_train=len(coords_norm))
    print(f"Fitted {name:<16}: {kdes[name]}")

# ------------------------------------------------------------------
# 5. Save
# ------------------------------------------------------------------
for name, sp_kde in kdes.items():
    out_pkl = MODELS_DIR / f"{name}.pkl"
    with open(out_pkl, "wb") as f:
        pickle.dump(sp_kde, f)
    print(f"Saved               : {out_pkl}")

# ------------------------------------------------------------------
# 6. Verification plot: real data vs KDE samples
# ------------------------------------------------------------------
N_SAMPLES = len(sp)
kde_samples = sp_kde.sample(N_SAMPLES)

fig, axes = plt.subplots(1, 2, figsize=(13, 6))

for ax, pts, title, color in [
    (axes[0], coords_norm,  f"Real SP customers  (n={len(coords_norm):,})", "#2980b9"),
    (axes[1], kde_samples,  f"KDE samples  (n={N_SAMPLES:,}, bw={best_bw:.3f})", "#e74c3c"),
]:
    ax.scatter(pts[:, 0], pts[:, 1], s=1.5, alpha=0.25, color=color)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("x (lng normalized)", fontsize=10)
    ax.set_ylabel("y (lat normalized)", fontsize=10)
    ax.set_title(title, fontsize=11)
    ax.set_aspect("equal")

plt.suptitle("SP KDE — real data vs synthetic samples", fontsize=13)
plt.tight_layout()

out_plot = PLOTS_DIR / "05_kde.png"
fig.savefig(out_plot, dpi=130, bbox_inches="tight")
plt.close(fig)
print(f"Plot saved          : {out_plot}")

# Print % of samples inside [0,1]²
for name, sp_kde in kdes.items():
    samples = sp_kde.sample(N_SAMPLES)
    inside  = ((samples >= 0) & (samples <= 1)).all(axis=1).mean()
    print(f"Samples inside [0,1]^2 ({name}): {100*inside:.1f}%")
