"""
EDA 02 — Geographic Distribution
=================================
Question: Where are SP customers located? Are there natural geographic clusters?
          How many average daily deliveries per cluster?

Run:
    uv run eda/02_geographic_distribution.py
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from pathlib import Path
from sklearn.cluster import KMeans

DATA_DIR  = Path("data/olist")
PLOTS_DIR = Path("eda/plots")
PLOTS_DIR.mkdir(exist_ok=True)

# ------------------------------------------------------------------
# 1. Load SP customers + geolocation + orders (for date range)
# ------------------------------------------------------------------
customers = pd.read_csv(DATA_DIR / "olist_customers_dataset.csv")
geoloc    = pd.read_csv(DATA_DIR / "olist_geolocation_dataset.csv")
orders    = pd.read_csv(DATA_DIR / "olist_orders_dataset.csv",
                        parse_dates=["order_purchase_timestamp"])

sp_customers = customers[
    customers["customer_city"].str.strip().str.lower() == "sao paulo"
].copy()

# Aggregate geolocation: mean lat/lng per CEP prefix
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

sp = (
    sp_customers
    .merge(geo_mean, on="customer_zip_code_prefix", how="inner")
    .merge(orders[["order_id", "customer_id", "order_purchase_timestamp"]],
           on="customer_id", how="left")
)

print(f"SP customers with lat/lng : {len(sp):,}  (out of {len(sp_customers):,})")

# Remove outliers outside SP bounding box
sp = sp[(sp["lat"].between(-24.0, -23.3)) & (sp["lng"].between(-46.9, -46.3))]
print(f"After bounding box filter : {len(sp):,}")

# Working days in dataset period (for daily average)
date_min = sp["order_purchase_timestamp"].min().normalize()
date_max = sp["order_purchase_timestamp"].max().normalize()
working_days = len(pd.bdate_range(date_min, date_max))
print(f"Working days in period    : {working_days:,}")

coords = sp[["lat", "lng"]].values

# ------------------------------------------------------------------
# 2. Plot: scatter + daily distribution bar for K = 3, 5, 8
# ------------------------------------------------------------------
K_VALUES = [3, 5, 8]

fig, axes = plt.subplots(2, len(K_VALUES), figsize=(18, 11),
                         gridspec_kw={"height_ratios": [2, 1]})

for col, k in enumerate(K_VALUES):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(coords)
    centers = km.cluster_centers_
    colors = cm.tab10(np.linspace(0, 1, k))

    # ---- top: scatter map ----
    ax_map = axes[0, col]
    for i in range(k):
        mask = labels == i
        ax_map.scatter(
            sp["lng"].values[mask], sp["lat"].values[mask],
            s=3, alpha=0.45, color=colors[i],
        )
    for i, (clat, clng) in enumerate(centers):
        dark = tuple(c * 0.55 for c in colors[i][:3]) + (1.0,)
        ax_map.scatter(clng, clat, s=260, marker="s", color=dark,
                       edgecolors="white", linewidths=0.8, zorder=5)
        ax_map.text(clng, clat, str(i + 1), ha="center", va="center",
                    fontsize=7, fontweight="bold", color="white", zorder=6)

    ax_map.set_title(f"K={k} clusters", fontsize=11)
    ax_map.set_xlabel("Longitude", fontsize=8)
    ax_map.set_ylabel("Latitude", fontsize=8)
    ax_map.set_aspect("equal")

    # ---- bottom: daily avg per cluster ----
    ax_bar = axes[1, col]
    cluster_labels = [f"C{i+1}" for i in range(k)]
    weekly_avgs = [round((labels == i).sum() / working_days * 5, 1) for i in range(k)]

    bars = ax_bar.bar(cluster_labels, weekly_avgs, color=colors, edgecolor="black",
                      linewidth=0.6)
    for bar, val in zip(bars, weekly_avgs):
        ax_bar.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                    f"{val}", ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax_bar.set_ylabel("Avg orders / week", fontsize=8)
    ax_bar.set_title(f"Weekly avg per cluster (total {sum(weekly_avgs):.0f}/week)", fontsize=9)
    ax_bar.set_ylim(0, max(weekly_avgs) * 1.25)

plt.suptitle(f"SP customer geographic clusters  (n={len(sp):,}, {working_days} working days)",
             fontsize=13, y=1.01)
plt.tight_layout()

out = PLOTS_DIR / "02_geographic_clusters.png"
fig.savefig(out, dpi=130, bbox_inches="tight")
plt.close(fig)
print(f"\nPlot saved -> {out}")

# ------------------------------------------------------------------
# 3. Summary table for K=5
# ------------------------------------------------------------------
km5 = KMeans(n_clusters=5, random_state=42, n_init=10)
labels5 = km5.fit_predict(coords)
print("\nK=5 summary:")
print(f"{'Cluster':<9} {'lat':>8} {'lng':>8} {'orders':>8} {'%':>6} {'avg/day':>9}")
for i, (clat, clng) in enumerate(km5.cluster_centers_):
    n = int((labels5 == i).sum())
    print(f"  C{i+1}     {clat:>8.4f} {clng:>8.4f} {n:>8,} {100*n/len(sp):>5.1f}%  {n/working_days:>8.1f}")
