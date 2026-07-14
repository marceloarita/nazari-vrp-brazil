"""
EDA 05 — Leakage-free train/test split per SP zone
==================================================
Part II. For each zone, splits real customers into a training pool and a
disjoint held-out test set, so training instances (sampled from the train pool)
and eval instances (sampled from the test pool) never share a customer.

Per zone:
  1. K-means (K=5, seed=42) on raw lat/lng     — identical to eda/04
  2. Full-cluster bounding-box normalization to [0,1]² (padding 0.05) — fixed frame
  3. Split customers into a held-out test set (TEST_FRAC of the zone) + train (disjoint)
  4. Save the TRAIN pool of real coords       -> artifacts/instances/olist_sp_c{k}_train_pool.pt
  5. Build eval instances from the TEST pool  -> artifacts/instances/olist_sp_c{k}_vrp20_heldout_n{N}_seed{S}.pt

Training samples 20 real customers directly from the train pool (no KDE).
Evaluate with: scripts/eval_sp.py --split heldout

Run:
    uv run eda/05_split_train_test.py
"""

import numpy as np
import pandas as pd
import torch
from pathlib import Path
from sklearn.cluster import KMeans

# ------------------------------------------------------------------
# Parameters — must match eda/04 for the shared bits
# ------------------------------------------------------------------
K_CLUSTERS          = 5
SEED                = 42
TEST_FRAC           = 0.15   # fraction of each zone held out for the eval (10-15%)
N_INSTANCES         = 32     # eval instances per zone
N_CUSTOMERS         = 20
VEHICLE_CAPACITY    = 30
MAX_ORDER_WEIGHT_KG = 20.0
COORD_PADDING       = 0.05

DATA_DIR = Path("data/olist")
INST_DIR = Path("artifacts/instances")
INST_DIR.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(SEED)

# ------------------------------------------------------------------
# 1. Load + join + filters (mirror eda/04)
# ------------------------------------------------------------------
print("Loading data...")
customers = pd.read_csv(DATA_DIR / "olist_customers_dataset.csv")
geoloc    = pd.read_csv(DATA_DIR / "olist_geolocation_dataset.csv")
orders    = pd.read_csv(DATA_DIR / "olist_orders_dataset.csv")
items     = pd.read_csv(DATA_DIR / "olist_order_items_dataset.csv")
products  = pd.read_csv(DATA_DIR / "olist_products_dataset.csv")

sp_customers = customers[
    customers["customer_city"].str.strip().str.lower() == "sao paulo"
].copy()

geo_mean = (
    geoloc[geoloc["geolocation_state"] == "SP"]
    .groupby("geolocation_zip_code_prefix")[["geolocation_lat", "geolocation_lng"]]
    .mean().reset_index()
    .rename(columns={"geolocation_zip_code_prefix": "customer_zip_code_prefix",
                     "geolocation_lat": "lat", "geolocation_lng": "lng"})
)

sp_order_ids = sp_customers.merge(orders[["order_id", "customer_id"]], on="customer_id")["order_id"]
order_weight = (
    items[items["order_id"].isin(sp_order_ids)]
    .merge(products[["product_id", "product_weight_g"]], on="product_id", how="left")
    .groupby("order_id")["product_weight_g"].sum()
    .rename("total_weight_g").reset_index()
)
sp = (sp_customers.merge(geo_mean, on="customer_zip_code_prefix", how="inner")
      .merge(orders[["order_id", "customer_id"]], on="customer_id", how="left")
      .merge(order_weight, on="order_id", how="left"))
sp = sp[(sp["lat"].between(-24.0, -23.3)) & (sp["lng"].between(-46.9, -46.3))]
sp = sp.dropna(subset=["total_weight_g"])
sp = sp[(sp["total_weight_g"] > 0) & (sp["total_weight_g"] <= MAX_ORDER_WEIGHT_KG * 1000)]
sp = sp.reset_index(drop=True)
print(f"SP orders after filters : {len(sp):,}")

# Global demand mapping (mirror eda/04)
_, bin_edges = pd.qcut(sp["total_weight_g"], q=9, retbins=True, duplicates="drop")
bin_edges[0] = 0.0; bin_edges[-1] = MAX_ORDER_WEIGHT_KG * 1000 + 1
sp["demand_raw"] = pd.cut(sp["total_weight_g"], bins=bin_edges, labels=False,
                          include_lowest=True).astype(float) + 1

# K-means (identical to eda/04)
km = KMeans(n_clusters=K_CLUSTERS, random_state=SEED, n_init=10)
sp["cluster"] = km.fit_predict(sp[["lat", "lng"]].values)
centroids = km.cluster_centers_

# ------------------------------------------------------------------
# 2. Per-zone split -> train pool + held-out eval
# ------------------------------------------------------------------
for k in range(K_CLUSTERS):
    cdf = sp[sp["cluster"] == k].reset_index(drop=True)
    n = len(cdf)

    # Fixed frame: full-cluster bbox with padding (same as eda/04)
    lat_min, lat_max = cdf["lat"].min(), cdf["lat"].max()
    lng_min, lng_max = cdf["lng"].min(), cdf["lng"].max()
    lat_lo = lat_min - COORD_PADDING * (lat_max - lat_min)
    lat_hi = lat_max + COORD_PADDING * (lat_max - lat_min)
    lng_lo = lng_min - COORD_PADDING * (lng_max - lng_min)
    lng_hi = lng_max + COORD_PADDING * (lng_max - lng_min)

    def normalize(lat, lng):
        return ((lng - lng_lo) / (lng_hi - lng_lo),
                (lat - lat_lo) / (lat_hi - lat_lo))

    depot_x, depot_y = normalize(centroids[k, 0], centroids[k, 1])
    depot = torch.tensor([depot_x, depot_y], dtype=torch.float32)

    # Disjoint split (test = TEST_FRAC of the zone, at least N_CUSTOMERS)
    test_size = max(int(round(TEST_FRAC * n)), N_CUSTOMERS)
    perm = rng.permutation(n)
    test_idx  = perm[:test_size]
    train_idx = perm[test_size:]
    train_df = cdf.iloc[train_idx]
    test_df  = cdf.iloc[test_idx].reset_index(drop=True)

    # --- TRAIN pool: real coords the model samples from ---
    xtr, ytr = normalize(train_df["lat"].values, train_df["lng"].values)
    pool = torch.tensor(np.stack([xtr, ytr], axis=1), dtype=torch.float32)
    pool_path = INST_DIR / f"olist_sp_c{k+1}_train_pool.pt"
    torch.save({
        "coords":       pool,             # (n_train, 2)
        "depot":        depot,            # (2,)
        "cluster_id":   k + 1,
        "centroid_lat": float(centroids[k, 0]),
        "centroid_lng": float(centroids[k, 1]),
        "n_train":      len(pool),
    }, pool_path)

    # --- Held-out eval instances from the TEST pool ---
    all_coords, all_demands = [], []
    for _ in range(N_INSTANCES):
        idx = rng.choice(len(test_df), size=N_CUSTOMERS, replace=False)
        s = test_df.iloc[idx]
        xs, ys = normalize(s["lat"].values, s["lng"].values)
        cust = torch.tensor(np.stack([xs, ys], axis=1), dtype=torch.float32)
        d = torch.tensor(s["demand_raw"].values.astype(np.float32) / VEHICLE_CAPACITY)
        depot_coord = torch.tensor([[depot_x, depot_y]], dtype=torch.float32)
        all_coords.append(torch.cat([depot_coord, cust], dim=0))
        all_demands.append(torch.cat([torch.zeros(1), d], dim=0))

    out = INST_DIR / f"olist_sp_c{k+1}_vrp{N_CUSTOMERS}_heldout_n{N_INSTANCES}_seed{SEED}.pt"
    torch.save({
        "coords":           torch.stack(all_coords),
        "demands":          torch.stack(all_demands),
        "vehicle_capacity": 1.0,
        "cluster_id":       k + 1,
        "centroid_lat":     float(centroids[k, 0]),
        "centroid_lng":     float(centroids[k, 1]),
        "n_customers":      N_CUSTOMERS,
        "n_instances":      N_INSTANCES,
        "seed":             SEED,
        "test_pool_size":   test_size,
        "max_weight_kg":    MAX_ORDER_WEIGHT_KG,
    }, out)

    print(f"C{k+1}: n={n:>5,}  train={len(train_df):>5,}  test={test_size}  "
          f"->  {pool_path.name} + {out.name}")

print("\nDone.")
print("  Train pools  : artifacts/instances/olist_sp_c*_train_pool.pt")
print("  Held-out eval: artifacts/instances/olist_sp_c*_vrp20_heldout_*.pt")
print("Evaluate with: uv run scripts/eval_sp.py --split heldout --checkpoint <ckpt>")
