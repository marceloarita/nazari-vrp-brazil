# nazari-vrp-brazil

Reproduction of the Reinforcement Learning model for the Vehicle Routing Problem (VRP) from Nazari et al. (2018), with extensions for Brazilian geographic distribution.

## Reference paper

> Nazari et al. — *Reinforcement Learning for Solving the Vehicle Routing Problem*, NeurIPS 2018  
> [`refs/1802.04240v2.pdf`](refs/1802.04240v2.pdf)

## Architecture

- **Static encoder**: Conv1d(2→128) over node coordinates (x, y) — computed once per episode
- **Dynamic encoder**: Conv1d(2→128) over [demand, remaining capacity] — recomputed each step
- **Decoder**: LSTMCell + 2-pass glimpse attention mechanism (Nazari 2018)
- **Baseline**: greedy rollout (Kool et al. 2019) — intentional departure from Nazari 2018, which uses an exponential moving average of rewards

## Project structure

```
src/          # model, environment, and utilities
scripts/      # train_kaggle.py, test_training.py
tests/        # test_environment.py
refs/         # original paper
```

## How to run

```bash
uv pip install -e . --no-deps
uv run scripts/test_training.py   # local smoke test (CPU, VRP10)
uv run scripts/train_kaggle.py    # full training (GPU, VRP50)
```
