"""
Part II methodology figures.

Generates:
    artifacts/figures/train_test_split.png    — leakage-free train/test split (real C5)

Zone: C5 / Centro-Sul (the zone we train in Part II).
Training data is sampled DIRECTLY from the real train pool (no KDE) — see notes.md.

Run:
    uv run experiments/part2_sp/plot_methodology.py
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.cluster import KMeans

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

OUT = Path("artifacts/figures"); OUT.mkdir(parents=True, exist_ok=True)
DATA = Path("data/olist")
SEED, PAD, TEST_FRAC, CLUSTER_LABEL = 42, 0.05, 0.15, 4  # label 4 == C5


def load_c5():
    """Return (X, depot) — real C5 customer coords normalized to [0,1]², plus centroid depot."""
    customers = pd.read_csv(DATA / "olist_customers_dataset.csv")
    geoloc    = pd.read_csv(DATA / "olist_geolocation_dataset.csv")
    orders    = pd.read_csv(DATA / "olist_orders_dataset.csv")
    items     = pd.read_csv(DATA / "olist_order_items_dataset.csv")
    products  = pd.read_csv(DATA / "olist_products_dataset.csv")
    spc = customers[customers["customer_city"].str.strip().str.lower() == "sao paulo"].copy()
    gm = (geoloc[geoloc["geolocation_state"] == "SP"]
          .groupby("geolocation_zip_code_prefix")[["geolocation_lat", "geolocation_lng"]].mean()
          .reset_index().rename(columns={"geolocation_zip_code_prefix": "customer_zip_code_prefix",
                                         "geolocation_lat": "lat", "geolocation_lng": "lng"}))
    soi = spc.merge(orders[["order_id", "customer_id"]], on="customer_id")["order_id"]
    ow = (items[items["order_id"].isin(soi)].merge(products[["product_id", "product_weight_g"]], on="product_id", how="left")
          .groupby("order_id")["product_weight_g"].sum().rename("w").reset_index())
    sp = (spc.merge(gm, on="customer_zip_code_prefix", how="inner")
          .merge(orders[["order_id", "customer_id"]], on="customer_id", how="left")
          .merge(ow, on="order_id", how="left"))
    sp = sp[(sp["lat"].between(-24.0, -23.3)) & (sp["lng"].between(-46.9, -46.3))]
    sp = sp.dropna(subset=["w"]); sp = sp[(sp["w"] > 0) & (sp["w"] <= 20000)].reset_index(drop=True)
    km = KMeans(n_clusters=5, random_state=SEED, n_init=10)
    sp["cluster"] = km.fit_predict(sp[["lat", "lng"]].values)
    cdf = sp[sp["cluster"] == CLUSTER_LABEL].reset_index(drop=True)
    lat, lng = cdf["lat"].values, cdf["lng"].values
    la_lo, la_hi = lat.min()-PAD*(lat.max()-lat.min()), lat.max()+PAD*(lat.max()-lat.min())
    ln_lo, ln_hi = lng.min()-PAD*(lng.max()-lng.min()), lng.max()+PAD*(lng.max()-lng.min())
    X = np.stack([(lng-ln_lo)/(ln_hi-ln_lo), (lat-la_lo)/(la_hi-la_lo)], axis=1)
    c = km.cluster_centers_[CLUSTER_LABEL]
    depot = np.array([(c[1]-ln_lo)/(ln_hi-ln_lo), (c[0]-la_lo)/(la_hi-la_lo)])
    return X, depot


def fig_split(X, depot):
    n = len(X); rng = np.random.default_rng(SEED)
    test_size = max(int(round(TEST_FRAC*n)), 20)
    perm = rng.permutation(n); test_idx, train_idx = perm[:test_size], perm[test_size:]

    TRAIN = "#95a5a6"   # grey
    VAL   = "#2980b9"   # blue — validation, distinct from the orange depot
    DEPOT = "#D4651C"   # orange

    fig, ax = plt.subplots(1, 3, figsize=(18, 6.2))

    # Panel 0 — the split: training pool + held-out test + depot
    ax[0].scatter(X[train_idx, 0], X[train_idx, 1], s=8, alpha=0.30, color=TRAIN, label=f"train pool ({len(train_idx):,})")
    ax[0].scatter(X[test_idx, 0], X[test_idx, 1], s=12, alpha=0.55, color=VAL, edgecolors="none", label=f"held-out test ({test_size})")
    ax[0].scatter(*depot, s=70, marker="s", color=DEPOT, edgecolors="k", linewidth=1.0, zorder=6, label="depot (centroid)")
    ax[0].set_title(f"Train / test split — {TEST_FRAC:.0%} held out", fontsize=12)
    ax[0].legend(fontsize=10, loc="upper right")

    # Panels 1-2 — two validation instances recombined from the held-out pool
    for j, axi in enumerate([ax[1], ax[2]], start=1):
        inst = rng.choice(test_idx, size=20, replace=False)
        axi.scatter(X[test_idx, 0], X[test_idx, 1], s=8, alpha=0.12, color=VAL)          # faint held-out pool
        axi.scatter(X[inst, 0], X[inst, 1], s=45, color=VAL, edgecolors="k", linewidth=0.5, zorder=5, label="20 customers")
        axi.scatter(*depot, s=70, marker="s", color=DEPOT, edgecolors="k", linewidth=1.0, zorder=6, label="depot")
        axi.set_title(f"Validation instance #{j}  (20 of {test_size} held-out)", fontsize=12)
        axi.legend(fontsize=10, loc="upper right")

    for a in ax:
        a.set_xlim(-0.05, 1.05); a.set_ylim(-0.05, 1.05); a.set_aspect("equal")
        a.set_xlabel("x (lng)"); a.set_ylabel("y (lat)")
    plt.suptitle(
        f"Zone C5: {n:,} customers → {len(train_idx):,} train / {test_size} held-out — "
        f"the held-out set recombines into many VRP20 validation instances",
        fontsize=13, y=1.02,
    )
    plt.tight_layout(); fig.savefig(OUT/"train_test_split.png", dpi=120, bbox_inches="tight"); plt.close(fig)
    print("saved train_test_split.png")


if __name__ == "__main__":
    X, depot = load_c5()
    fig_split(X, depot)
    print(f"\nFigure saved to {OUT}/")
