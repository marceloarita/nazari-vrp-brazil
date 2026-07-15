# Part II — Learning São Paulo: Per-Zone Routing

> Continuation of [Part I](README.md), which reproduced Nazari et al. (2018) and exposed the cost of distribution shift.

## The challenge of real data

Part I trained the policy on synthetic `Uniform[0,1]²` instances and then applied it **zero-shot** to real São Paulo deliveries. The gap to OR-Tools jumped from ~20% on synthetic data to **63.8% (greedy) / 26.9% (best-of-128)** on real SP zones. The diagnosis was unambiguous: the bottleneck is **distribution shift, not model capacity**. That zero-shot model is the **baseline** Part II works to beat.

Real data breaks the assumptions synthetic data hides: customers cluster along streets and centers, density varies sharply across a zone, and every zone has its own shape. Part II asks a single question — **can we close the gap by making the model learn São Paulo itself?**

## Validation frame

To keep the effort focused and every claim measurable:

- **Target zone — C5 / Centro-Sul.** We concentrate on the densest, most central zone, which is also the hardest under best-of-128 (36.6% gap). The bet: if an approach works on the hardest, most non-uniform zone, it should carry over to the easier ones.
- **Held-out benchmark.** Every number is measured on a leakage-free held-out test set — 15% of the zone's real customers, never seen in training — the *same* instances used for the Part I zero-shot baseline, so results are directly comparable.
- **A list of hypotheses.** Part II is run as a sequence of hypotheses, each a lever on a different stage of the pipeline (`data → features → learning signal → decoding`). For each, the decision taken and the alternative we discarded:

| Lever | Decision | Alternative discarded (why) | Status |
|---|---|---|---|
| **Data** | Sample real customers directly, per zone | KDE-generated instances — redundant given thousands of real points; the CV bandwidth collapses to a near-copy of the data | ✅ tested (H1) |
| **Features** | KDE **density** per node — `log p(x,y)` | Polar coordinates / distance-to-depot — derivable from the raw `(x,y)` the model already sees | ⬜ planned |
| **Learning signal** | POMO multi-start baseline | A learned critic — an extra network for marginal gain over a rollout baseline | ⬜ planned |
| **Decoding** | Best-of-N sampling, then 2-opt | — | ◐ sampling done, 2-opt planned |

### The held-out benchmark, visualized

The zone's real customers are split once into a training pool and a disjoint held-out test set. Validation instances are then built by sampling 20 customers at a time from the held-out pool — the same 424 points recombine into many distinct VRP20 problems the model never trained on.

<img src="docs/images/train_test_split.png" width="100%">

*Zone C5 — left: the train/test split (2,405 train in grey, 424 held-out in blue, depot in orange). Middle and right: two VRP20 validation instances, each 20 customers drawn from the held-out pool. The 424 held-out points recombine into many distinct instances, and the model never trains on any of them — so a good score cannot be memorization.*

---

## Hypothesis 1 — Train per zone

**Claim:** zero-shot failed because the model never saw clustered geography. The most direct fix is to **train on the target zone's own customers**, matching the training distribution to the evaluation exactly.

Three design choices make "train per zone" concrete:

- **Per-zone normalization.** Each zone's lat/lng is normalized by *its own* bounding box, so the model always sees points in `[0,1]²` framed to that zone — the identical framing the held-out eval uses. (Part I's failed KDE attempt mismatched a global frame against per-zone eval; here the frames agree by construction.)
- **Leakage-free train/test split.** 15% of the zone's customers are held out for evaluation; the model trains only on the remaining 85%. Train and test never share a customer, so a good score cannot be memorization.
- **Direct real-point sampling.** Each training instance draws 20 real customers from the train pool, with the zone centroid as the depot. With ~2,400 real points, this yields effectively unlimited distinct instances — no need to synthesize data.

### Result

| Model | Greedy gap | Best-of-128 gap |
|---|---|---|
| Zero-shot uniform (Part I baseline) | 86.0% | 36.6% |
| **Per-zone (C5)** | **14.0%** | **7.3%** |

<img src="docs/images/vrp20_routes_sp_c5.png" width="100%">

*Zone C5 (held-out) — the per-zone model: greedy vs best-of-128 vs OR-Tools. Greedy already tracks the OR-Tools structure (~13% per instance) and best-of-128 nearly matches it (top instance within 0.7%) — a world apart from the zero-shot baseline, whose greedy routes sprawled at ~87%.*

Per-zone training alone collapses the gap — from **86% → 14%** greedy, and **37% → 7%** under best-of-128 — with no change to the model architecture. This confirms Part I's diagnosis directly: the barrier was the training distribution, not the network. Best-of-128 then lands the model within a few percent of OR-Tools, at a fraction of the runtime.

This per-zone model is the **baseline the remaining hypotheses (density feature, POMO, 2-opt) build on.**
