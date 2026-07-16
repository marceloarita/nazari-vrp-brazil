# Neural Vehicle Routing: From Nazari (2018) to Real São Paulo Deliveries

The Capacitated Vehicle Routing Problem (CVRP) asks: given a depot and N customers — each with a location and demand — find the shortest set of routes for vehicles with capacity Q that serves everyone exactly once. With 20 customers the search space exceeds 10¹⁷ candidate solutions. Classical solvers work by pruning that space; learned heuristics try to shortcut it entirely.

This project reproduces an attention-based routing policy trained with reinforcement learning, then confronts it with real delivery data — where the interesting problems live. It is organized in two parts.

## The two parts

| | Part I | Part II |
|---|---|---|
| **Focus** | Reproduce [Nazari et al. (2018)](refs/1802.04240v2.pdf) on synthetic CVRP | Close the real-world distribution-shift gap on São Paulo |
| **Data** | Synthetic `Uniform[0,1]²` instances | Real Olist SP deliveries, per zone |
| **Question** | Can we reproduce attention-based routing with REINFORCE? | Can the model learn São Paulo itself? |
| **Headline** | Within ~9% of OR-Tools, thousands of times faster — but **zero-shot on real data fails (26.9% gap)** | Gap on the hardest zone cut from **36.6% → 3.8%** |

📄 **[Read Part I → Reproducing Nazari on Synthetic CVRP](PART_I.md)**
📄 **[Read Part II → Learning São Paulo: Per-Zone Routing](PART_II.md)**

## Results at a glance

The story in one line — the gap to OR-Tools on São Paulo, from a model that never saw the city to one trained on it (best-of-128 decoding, hardest zone C5):

| Stage | Gap to OR-Tools |
|---|---|
| Zero-shot (uniform-trained model) | 36.6% |
| Per-zone training | 6.6% |
| + POMO learning signal | 5.5% |
| + 2-opt cleanup | **3.8%** |

The bottleneck was never model capacity — it was the training distribution. Matching it, plus a better learning signal and a cheap geometric cleanup, does the rest, all at a fraction of OR-Tools' solve time.

## Pipeline

```
eda/01_volume_analysis.py           → SP order volume, daily scope
eda/02_geographic_distribution.py   → K-means zones, weekly volume per cluster
eda/03_demand_analysis.py           → Weight distribution, demand mapping [1–9]
eda/04_build_instances.py           → VRP20 instance generation from real SP data
eda/05_split_train_test.py          → Per-zone train/test split, held-out eval (Part II)
eda/06_add_density.py               → Optional KDE density feature (Part II / H2)

scripts/train_kaggle.py             → VRP10/20 training — all baseline variants (Kaggle T4)
experiments/part2_sp/               → Per-zone training, POMO, 2-opt, eval & plots (Part II)

scripts/eval_sp.py                  → Zero-shot + OR-Tools eval on SP instances
scripts/eval.py                     → Eval against OR-Tools on synthetic instances
```

## Acknowledgements

All model training was done on **NVIDIA Tesla T4 GPUs** provided free of charge by [Kaggle](https://www.kaggle.com). Without Kaggle's free GPU quota this project would not have been feasible.
