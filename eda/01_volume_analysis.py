"""
EDA 01 — Volume Analysis
========================
Question: How many daily deliveries does São Paulo have,
          and which CEP zones have the most demand?

Run:
    uv run eda/01_volume_analysis.py
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path

DATA_DIR  = Path("data/olist")
PLOTS_DIR = Path("eda/plots")
PLOTS_DIR.mkdir(exist_ok=True)

# ------------------------------------------------------------------
# 1. Load and filter to São Paulo city
# ------------------------------------------------------------------
orders    = pd.read_csv(DATA_DIR / "olist_orders_dataset.csv", parse_dates=["order_purchase_timestamp"])
customers = pd.read_csv(DATA_DIR / "olist_customers_dataset.csv")

sp = (
    customers[customers["customer_city"].str.strip().str.lower() == "sao paulo"]
    .merge(orders[["order_id", "customer_id", "order_purchase_timestamp"]], on="customer_id")
)

print(f"Total SP orders : {len(sp):,}")
print(f"Date range      : {sp['order_purchase_timestamp'].min().date()} to {sp['order_purchase_timestamp'].max().date()}")

# ------------------------------------------------------------------
# 2. Average daily orders (working days only, Mon–Fri)
# ------------------------------------------------------------------
date_min = sp["order_purchase_timestamp"].min().normalize()
date_max = sp["order_purchase_timestamp"].max().normalize()
working_days = pd.bdate_range(date_min, date_max)  # business days only

sp_weekdays = sp[sp["order_purchase_timestamp"].dt.dayofweek < 5]
avg_daily   = len(sp_weekdays) / len(working_days)

print(f"\nWorking days in period : {len(working_days):,}")
print(f"Orders on weekdays     : {len(sp_weekdays):,}")
print(f"Avg daily orders (SP)  : {avg_daily:.1f}")
print(f"\n→ VRP size suggestion  : ", end="")
if avg_daily <= 15:
    print("VRP10 or VRP20")
elif avg_daily <= 40:
    print("VRP20 or VRP50")
else:
    print(f"VRP50+ (need zone partitioning — {avg_daily:.0f} orders/day is too many for one vehicle)")

# ------------------------------------------------------------------
# 3. Top zones by CEP prefix
# ------------------------------------------------------------------
sp["cep_prefix"] = sp["customer_zip_code_prefix"].astype(str).str.zfill(5)
by_cep = (
    sp.groupby("cep_prefix")
    .size()
    .rename("total_orders")
    .sort_values(ascending=False)
)

print(f"\nTop 10 CEP prefixes in Sao Paulo:")
print(by_cep.head(10).to_string())

top3 = by_cep.head(3)
print(f"\nTop 3 zones (CEP prefix):")
for rank, (cep, count) in enumerate(top3.items(), 1):
    daily_est = count / len(working_days)
    print(f"  #{rank}  CEP {cep}  |  {count:,} total orders  |  ~{daily_est:.1f}/day")

# ------------------------------------------------------------------
# 4. Plot: top 20 CEP prefixes
# ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(12, 5))
top20 = by_cep.head(20)
bars = ax.bar(range(len(top20)), top20.values, color="#2980b9", edgecolor="#2980b9")
ax.set_xticks(range(len(top20)))
ax.set_xticklabels(top20.index, rotation=45, ha="right", fontsize=9)
ax.set_xlabel("CEP prefix (5 digits)", fontsize=12)
ax.set_ylabel("Total orders", fontsize=12)
date_min_str = sp["order_purchase_timestamp"].min().strftime("%b/%Y")
date_max_str = sp["order_purchase_timestamp"].max().strftime("%b/%Y")
ax.set_title(f"Top 20 CEP prefixes - Sao Paulo orders\n{date_min_str} to {date_max_str}  |  {len(sp):,} total orders", fontsize=12)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

# annotate top 3
for i in range(3):
    ax.bar(i, top20.values[i], color="#e74c3c", edgecolor="#e74c3c")
    ax.text(i, top20.values[i] + 5, f"#{i+1}", ha="center", va="bottom", fontsize=9, fontweight="bold", color="#e74c3c")

plt.tight_layout()
out = PLOTS_DIR / "01_orders_by_cep.png"
fig.savefig(out, dpi=130)
plt.close(fig)
print(f"\nPlot saved → {out}")
