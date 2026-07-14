# Part II — São Paulo

Builds on tag `v1.0-nazari-reproduction` (Part I: Nazari reproduction + zero-shot).
Goal: close the distribution gap on SP by training **per zone**, with train and
eval drawn from the same zone (train == eval).

## Approach (4 levers)
- **A1 — Data**: train per zone by sampling real customers directly from the zone's
  train pool (no KDE). Train and test pools are disjoint (no leakage).
- **A2 — Features**: optional density feature per node → `[x, y, demand, log p(x,y)]`
  (a KDE fitted only to *describe* density, not to generate data).
- **A3 — Baseline**: POMO (mean of N multi-start rollouts; replaces Kool, no critic).
- **A4 — Decoding**: 2-opt post-processing + sampling at eval.

## Experiment naming
```
{size}_{dist}_{zone}__{baseline}__{features}
e.g. vrp20_real_c5__kool__base
```
- Checkpoints: `artifacts/checkpoints/<exp>/`
- Results:     `artifacts/results/<...>.csv` (column `exp` = ID)
- Instances:   `artifacts/instances/` (train pools + held-out eval)

## Data prep
```
uv run eda/05_split_train_test.py     # per-zone train pool + held-out eval
uv run experiments/part2_sp/train_zone.py   # train on one zone (Kaggle T4)
uv run scripts/eval_sp.py --split heldout --checkpoint <ckpt>
```

`notes.md` in this folder is a living draft (not versioned).
