# EDA — Olist VRP Brazil

Exploratory analysis of the [Olist Brazilian E-Commerce dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)
to inform the modeling of a real-world CVRP (Capacitated Vehicle Routing Problem) for São Paulo.

## Methodology

Each script is focused on a single topic. Decisions are made incrementally — the output of one
script informs the design of the next.

## Scripts

### 01_volume_analysis.py
**Question:** How many daily deliveries does São Paulo have, and which zones have the most demand?

**Approach:**
1. Filter orders to São Paulo city
2. Calculate total orders and date range
3. Estimate average daily orders using working days only (Mon–Fri, excluding weekends)
4. Group by CEP prefix (5 digits) → identify top zones by order volume
5. Decide: which VRP size fits the daily volume? (VRP10 / VRP20 / VRP50)

**Output:** `artifacts/figures/eda/01_orders_by_cep.png`

---

*Further scripts to be added based on findings from each step.*

## Data

Raw CSVs in `../data/olist/` (not versioned). Download with:
```bash
uv run scripts/download_olist.py
```
