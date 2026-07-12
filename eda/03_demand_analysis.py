"""
EDA 03 — Demand Analysis
=========================
Question: What is the weight distribution of SP orders?
          How can we map weight to CVRP demand values [1-9]?

Run:
    uv run eda/03_demand_analysis.py
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from pathlib import Path

DATA_DIR  = Path("data/olist")
PLOTS_DIR = Path("eda/plots")
PLOTS_DIR.mkdir(exist_ok=True)

MAX_ORDER_WEIGHT_KG = 20.0  # motoboy hard limit — orders above this are excluded

# ------------------------------------------------------------------
# 1. Load SP orders + items + products
# ------------------------------------------------------------------
customers  = pd.read_csv(DATA_DIR / "olist_customers_dataset.csv")
orders     = pd.read_csv(DATA_DIR / "olist_orders_dataset.csv")
items      = pd.read_csv(DATA_DIR / "olist_order_items_dataset.csv")
products   = pd.read_csv(DATA_DIR / "olist_products_dataset.csv")

sp_order_ids = (
    customers[customers["customer_city"].str.strip().str.lower() == "sao paulo"]
    .merge(orders[["order_id", "customer_id"]], on="customer_id")
    ["order_id"]
)

# Items for SP orders, joined with product weights
sp_items = (
    items[items["order_id"].isin(sp_order_ids)]
    .merge(products[["product_id", "product_weight_g"]], on="product_id", how="left")
)

# Total weight per order (sum of all items)
order_weight = (
    sp_items.groupby("order_id")["product_weight_g"]
    .sum()
    .rename("total_weight_g")
    .reset_index()
)

# Drop orders with missing weight
order_weight = order_weight.dropna(subset=["total_weight_g"])
order_weight = order_weight[order_weight["total_weight_g"] > 0]

n_before = len(order_weight)
order_weight = order_weight[order_weight["total_weight_g"] <= MAX_ORDER_WEIGHT_KG * 1000]
n_excluded = n_before - len(order_weight)

print(f"SP orders with weight data : {n_before:,}")
print(f"Excluded (> {MAX_ORDER_WEIGHT_KG:.0f} kg)  : {n_excluded:,}  ({100*n_excluded/n_before:.1f}%)")
print(f"Orders used for modeling   : {len(order_weight):,}")

w = order_weight["total_weight_g"]
print(f"\nWeight distribution (grams):")
print(f"  Mean   : {w.mean():>8.0f} g  ({w.mean()/1000:.2f} kg)")
print(f"  Median : {w.median():>8.0f} g  ({w.median()/1000:.2f} kg)")
print(f"  Std    : {w.std():>8.0f} g")
print(f"  Min    : {w.min():>8.0f} g")
print(f"  Max    : {w.max():>8.0f} g")
print(f"\nPercentiles:")
for p in [25, 50, 75, 90, 95, 99]:
    v = np.percentile(w, p)
    print(f"  p{p:<3} : {v:>8.0f} g  ({v/1000:.2f} kg)")

# ------------------------------------------------------------------
# 2. Map weight to demand [1-9] using quantile bins
# ------------------------------------------------------------------
# Quantile-based: each bin has equal number of orders
bins = pd.qcut(order_weight["total_weight_g"], q=9, labels=False, duplicates="drop")
order_weight["demand"] = bins + 1  # 1-indexed

print(f"\nDemand mapping (quantile bins, 1-9):")
demand_counts = order_weight["demand"].value_counts().sort_index()
bin_edges = pd.qcut(order_weight["total_weight_g"], q=9, retbins=True, duplicates="drop")[1]
for d in sorted(demand_counts.index):
    lo = bin_edges[int(d) - 1]
    hi = bin_edges[int(d)]
    print(f"  demand={int(d)}  weight {lo/1000:.2f}-{hi/1000:.2f} kg  |  {demand_counts[d]:,} orders")

print(f"\nAvg demand : {order_weight['demand'].mean():.2f}  (model trains with avg ~5)")

# ------------------------------------------------------------------
# 3. Plot: weight histogram + demand distribution
# ------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

# Left: weight histogram (already filtered at MAX_ORDER_WEIGHT_KG)
ax1.hist(w / 1000, bins=50, color="#2980b9", edgecolor="white", linewidth=0.4)
ax1.axvline(w.mean() / 1000, color="#e74c3c", linestyle="--", linewidth=1.2,
            label=f"mean {w.mean()/1000:.2f} kg")
ax1.axvline(w.median() / 1000, color="#e67e22", linestyle="--", linewidth=1.2,
            label=f"median {w.median()/1000:.2f} kg")
ax1.axvline(MAX_ORDER_WEIGHT_KG, color="#8e44ad", linestyle="-", linewidth=1.2,
            label=f"max {MAX_ORDER_WEIGHT_KG:.0f} kg (motoboy limit)")
ax1.set_xlabel("Order weight (kg)", fontsize=11)
ax1.set_ylabel("Number of orders", fontsize=11)
ax1.set_title(f"SP order weight distribution\n(filtered: <= {MAX_ORDER_WEIGHT_KG:.0f} kg)", fontsize=11)
ax1.legend(fontsize=9)
ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

# Right: demand distribution after quantile mapping
demand_vals = sorted(order_weight["demand"].dropna().unique())
demand_counts_plot = [order_weight[order_weight["demand"] == d].shape[0] for d in demand_vals]
ax2.bar([int(d) for d in demand_vals], demand_counts_plot,
        color="#27ae60", edgecolor="black", linewidth=0.5)
ax2.set_xlabel("Demand value", fontsize=11)
ax2.set_ylabel("Number of orders", fontsize=11)
ax2.set_title("Demand distribution after quantile mapping\n(capacity = 30)", fontsize=11)
ax2.set_xticks([int(d) for d in demand_vals])
ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

plt.suptitle(f"SP order demand analysis  (n={len(order_weight):,})", fontsize=13)
plt.tight_layout()

out = PLOTS_DIR / "03_demand_analysis.png"
fig.savefig(out, dpi=130, bbox_inches="tight")
plt.close(fig)
print(f"\nPlot saved -> {out}")
