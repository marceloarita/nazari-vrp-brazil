"""
EDA 04 — Build Real VRP Instances from Olist SP Data
=====================================================
Generates VRP instances using real customer locations and order weights from SP.

Pipeline per cluster:
  1. K-means cluster SP customers (K zones)
  2. Sample N_CUSTOMERS per instance, N_INSTANCES times
  3. Normalize lat/lng to [0,1]² (per cluster bounding box)
  4. Map weight -> demand [1-9] via quantile bins, normalize by VEHICLE_CAPACITY
  5. Depot = cluster centroid (normalized)
  6. Save as .pt — same format as eval sets

Run:
    uv run eda/04_build_instances.py
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.cluster import KMeans

# ------------------------------------------------------------------
# Parameters (edit here)
# ------------------------------------------------------------------
N_INSTANCES        = 32     # instances per cluster
N_CUSTOMERS        = 20     # VRP size
K_CLUSTERS         = 5      # number of SP zones
SEED               = 42     # reproducibility
VEHICLE_CAPACITY   = 30     # same as VRP20 training
MAX_ORDER_WEIGHT_KG = 20.0  # motoboy hard limit
COORD_PADDING      = 0.05   # 5% padding around bounding box per cluster

DATA_DIR  = Path("data/olist")
OUT_DIR   = Path("artifacts/instances")
OUT_DIR.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(SEED)

# ------------------------------------------------------------------
# 1. Load and join: customers + geolocation + orders + items + products
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
    .mean()
    .reset_index()
    .rename(columns={
        "geolocation_zip_code_prefix": "customer_zip_code_prefix",
        "geolocation_lat": "lat",
        "geolocation_lng": "lng",
    })
)

sp_order_ids = sp_customers.merge(
    orders[["order_id", "customer_id"]], on="customer_id"
)["order_id"]

order_weight = (
    items[items["order_id"].isin(sp_order_ids)]
    .merge(products[["product_id", "product_weight_g"]], on="product_id", how="left")
    .groupby("order_id")["product_weight_g"]
    .sum()
    .rename("total_weight_g")
    .reset_index()
)

sp = (
    sp_customers
    .merge(geo_mean, on="customer_zip_code_prefix", how="inner")
    .merge(orders[["order_id", "customer_id"]], on="customer_id", how="left")
    .merge(order_weight, on="order_id", how="left")
)

# Filters
sp = sp[(sp["lat"].between(-24.0, -23.3)) & (sp["lng"].between(-46.9, -46.3))]
sp = sp.dropna(subset=["total_weight_g"])
sp = sp[sp["total_weight_g"] > 0]
sp = sp[sp["total_weight_g"] <= MAX_ORDER_WEIGHT_KG * 1000]
sp = sp.reset_index(drop=True)

print(f"SP orders after filters : {len(sp):,}")

# ------------------------------------------------------------------
# 2. Demand mapping: weight -> [1-9] via quantile bins
# ------------------------------------------------------------------
_, bin_edges = pd.qcut(sp["total_weight_g"], q=9, retbins=True, duplicates="drop")
bin_edges[0]  = 0.0       # include minimum
bin_edges[-1] = MAX_ORDER_WEIGHT_KG * 1000 + 1  # include maximum

sp["demand_raw"] = pd.cut(
    sp["total_weight_g"], bins=bin_edges, labels=False, include_lowest=True
).astype(float) + 1  # 1-indexed

print(f"Demand range : {sp['demand_raw'].min():.0f} - {sp['demand_raw'].max():.0f}  "
      f"(avg {sp['demand_raw'].mean():.2f})")

# ------------------------------------------------------------------
# 3. K-means clustering on lat/lng
# ------------------------------------------------------------------
coords_all = sp[["lat", "lng"]].values
km = KMeans(n_clusters=K_CLUSTERS, random_state=SEED, n_init=10)
sp["cluster"] = km.fit_predict(coords_all)
centroids = km.cluster_centers_  # (K, 2) — [lat, lng]

# ------------------------------------------------------------------
# 4. Build instances per cluster
# ------------------------------------------------------------------
saved = []

for k in range(K_CLUSTERS):
    cluster_df = sp[sp["cluster"] == k].reset_index(drop=True)
    n_available = len(cluster_df)

    print(f"\nCluster {k+1}  |  {n_available:,} customers  |  "
          f"centroid lat={centroids[k,0]:.4f} lng={centroids[k,1]:.4f}")

    if n_available < N_CUSTOMERS:
        print(f"  WARNING: only {n_available} customers — sampling with replacement")

    # Coordinate normalization bounds (per cluster, with padding)
    lat_min, lat_max = cluster_df["lat"].min(), cluster_df["lat"].max()
    lng_min, lng_max = cluster_df["lng"].min(), cluster_df["lng"].max()
    lat_range = max(lat_max - lat_min, 1e-6)
    lng_range = max(lng_max - lng_min, 1e-6)
    lat_lo = lat_min - COORD_PADDING * lat_range
    lat_hi = lat_max + COORD_PADDING * lat_range
    lng_lo = lng_min - COORD_PADDING * lng_range
    lng_hi = lng_max + COORD_PADDING * lng_range

    def normalize(lat, lng):
        x = (lng - lng_lo) / (lng_hi - lng_lo)  # lng -> x
        y = (lat - lat_lo) / (lat_hi - lat_lo)  # lat -> y
        return x, y

    # Depot = cluster centroid (normalized)
    depot_x, depot_y = normalize(centroids[k, 0], centroids[k, 1])
    depot_coord = torch.tensor([[depot_x, depot_y]], dtype=torch.float32)

    all_coords  = []
    all_demands = []

    for _ in range(N_INSTANCES):
        idx = rng.choice(n_available, size=N_CUSTOMERS, replace=(n_available < N_CUSTOMERS))
        sampled = cluster_df.iloc[idx]

        xs, ys = normalize(sampled["lat"].values, sampled["lng"].values)
        customer_coords = torch.tensor(np.stack([xs, ys], axis=1), dtype=torch.float32)

        # Demand normalized by vehicle capacity
        raw_d = sampled["demand_raw"].values.astype(np.float32)
        customer_demands = torch.tensor(raw_d / VEHICLE_CAPACITY, dtype=torch.float32)

        coords_instance  = torch.cat([depot_coord, customer_coords], dim=0)   # (N+1, 2)
        depot_demand     = torch.zeros(1, dtype=torch.float32)
        demands_instance = torch.cat([depot_demand, customer_demands], dim=0) # (N+1,)

        all_coords.append(coords_instance)
        all_demands.append(demands_instance)

    coords_tensor  = torch.stack(all_coords)   # (N_INSTANCES, N+1, 2)
    demands_tensor = torch.stack(all_demands)  # (N_INSTANCES, N+1)

    avg_demand = demands_tensor[:, 1:].mean().item() * VEHICLE_CAPACITY
    print(f"  coords  : {tuple(coords_tensor.shape)}  range [{coords_tensor.min():.3f}, {coords_tensor.max():.3f}]")
    print(f"  demands : {tuple(demands_tensor.shape)}  avg raw demand {avg_demand:.2f}")

    out_path = OUT_DIR / f"olist_sp_c{k+1}_vrp{N_CUSTOMERS}_n{N_INSTANCES}_seed{SEED}.pt"
    torch.save({
        "coords":           coords_tensor,
        "demands":          demands_tensor,
        "vehicle_capacity": 1.0,           # demands already normalized
        "cluster_id":       k + 1,
        "centroid_lat":     float(centroids[k, 0]),
        "centroid_lng":     float(centroids[k, 1]),
        "n_customers":      N_CUSTOMERS,
        "n_instances":      N_INSTANCES,
        "seed":             SEED,
        "max_weight_kg":    MAX_ORDER_WEIGHT_KG,
    }, out_path)
    saved.append(out_path)
    print(f"  Saved -> {out_path}")

# ------------------------------------------------------------------
# 5. Summary
# ------------------------------------------------------------------
print(f"\n{'='*55}")
print(f"Generated {len(saved)} instance files:")
for p in saved:
    size_kb = p.stat().st_size // 1024
    print(f"  {p.name}  ({size_kb} KB)")
