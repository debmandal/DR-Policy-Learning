# Distributionally Robust Policy Learning from Observational Data
## (Logistic-regression nuisance estimators)

Reference implementation of **Algorithm 1** from the NeurIPS 2026 paper
*"Distributionally-Robust Policy Learning with Observational Data"*, with
DR-trained logistic regression for the nuisance estimators (no neural
networks). For the MLP / PyTorch variants, see the separate
`dr_policy` package.

## What the code does

Given observational data `{(X_i, W_i, Y_i)}_{i=1..n}` (covariates,
binary treatment, binary outcome), it solves

```
   max_{pi in Π}  inf_{Q : D_f(Q || P_n) <= rho}   sum_i  q_i (2 pi(X_i) - 1) Γ_i
```

where `Γ_i` is a doubly-robust score for the conditional treatment
effect (Athey & Wager 2021), and `D_f` is one of three f-divergences:

| name | `f(t)` | `f*(s)`                          |
|------|--------|----------------------------------|
| KL   | `t log t`              | `exp(s - 1)`        |
| χ²   | `(t-1)² / 2`           | `(max(1+s,0)² - 1)/2` |
| CVaR | `1{t ≤ 1/α}`           | `max(s, 0) / α`     |

The conditional outcome `m̂(x,w)` and propensity `ê(x)` are themselves
learnt with **distributionally robust** logistic regression. We use a
*primal* minibatch SGD: at each step, given a minibatch `B` of size
`b` with per-sample losses `ℓ_i = ℓ(θ; X_i, Y_i)`, we

1. solve the inner DRO problem on the batch in closed form,
   `q* = argmax_{q ∈ Δ(b) : (1/b) Σ_j f(b q_j) ≤ ρ} Σ_j q_j ℓ_j`,
   using the same `worst_case_Q` routine as the outer problem,
2. take a primal step `θ ← θ - lr · (Σ_j q*_j ∇ℓ_j + λ_2 θ)`.

This is the natural primal counterpart of the dual minibatch method of
Levy, Carmon, Duchi & Sidford (NeurIPS 2020). Because the inner
problem is convex on a small simplex, exact per-batch resolution
removes the need for any auxiliary dual variables `(λ, η)` and avoids
the bias of naive dual SGD.

## Layout

```
dr_logistic/
├── README.md
├── dr_logistic/
│   ├── __init__.py
│   ├── dro.py                # f-divergence dual: dr_value, worst_case_Q, dual_solution
│   ├── estimators.py         # DR-trained logistic regression (primal SGD)
│   └── policy.py             # Algorithm 1: cross-fit DR scores + tree policy fit
├── experiments/
│   ├── simulation.py         # synthetic Athey–Wager-style DGP
│   ├── real_data.py          # semi-synthetic California GAIN-like dataset
│   └── visualize_trees.py    # learnt-tree diagrams (sklearn plot_tree)
└── results/                  # produced by the experiment scripts (created at runtime)
```

## Install / requirements

```
pip install numpy scipy scikit-learn pandas matplotlib
```

Plain NumPy + sklearn's `DecisionTreeClassifier` for the policy class.
No GPU needed.

## Running the experiments

All three scripts have a `--help` flag listing every knob. Default
settings reproduce the figures from the paper appendix; the most useful
flags to vary are listed below each command.

### Simulation

```
python experiments/simulation.py
python experiments/simulation.py --help

# common customisations:
python experiments/simulation.py \
    --n 2000 --p 8 --n-seeds 10 --max-depth 3 \
    --rho-grid 0.05 0.1 0.2 0.4 0.8 \
    --shift-grid 0 1 2 3 4 5 \
    --n-epochs 50 --batch-size 128 --lr 0.05 \
    --shift-rho 0.2 --cvar-alpha 0.2

# skip one of the two sub-experiments:
python experiments/simulation.py --skip-rho-sweep
python experiments/simulation.py --skip-shift-experiment
```

Outputs:
* `simulation_dr_value_vs_rho.{csv,png}`
* `simulation_shift_robustness.{csv,png}`

### Real-data (California GAIN-like)

```
python experiments/real_data.py
python experiments/real_data.py --help

# common customisations:
python experiments/real_data.py \
    --train-counties Riverside SanDiego Tulare \
    --test-counties Alameda Butte LosAngeles \
    --depths 1 2 3 4 \
    --rho 0.2 --cvar-alpha 0.2 \
    --n-seeds 10 --n-epochs 50
```

Available counties: `Alameda Butte LosAngeles Riverside SanDiego Tulare`.

Outputs: `real_data_summary.csv`, `real_data_county_split.png`.

### Tree visualization

```
python experiments/visualize_trees.py
python experiments/visualize_trees.py --depth 3 --rho 0.3
```

Outputs:
* `real_data_trees_depth{D}.png` — 4-method comparison grid
* `real_data_tree_chi2_depth{D}.png` — standalone DR(χ²) diagram
* `real_data_trees.txt` — text-format trees for the appendix

## Quick-start API

```python
import numpy as np
from dr_logistic import doubly_robust_scores, dr_policy_tree

X = np.random.uniform(0, 1, (1000, 5))
W = (np.random.random(1000) < 0.5).astype(int)
Y = (np.random.random(1000) < 0.5).astype(float)

# Build Γ̂ via 3-fold cross-fitting with DR-trained logistic nuisances
fitted = doubly_robust_scores(
    X, Y, W,
    rho=0.2, divergence="chi2",
    estimator_kwargs=dict(n_epochs=60, batch_size=128, lr=0.05, l2=1e-3),
)

# Solve the outer max–min over depth-2 decision trees via fictitious play
res = dr_policy_tree(
    X, fitted.Gamma,
    rho=0.2, divergence="chi2",
    max_depth=2, n_iters=15,
)
print("DR value:", res.dr_value)
print("Tree:    ", res.policy)
```

## Implementation notes

* **DRO dual solver (`dro.py`).** KL is reduced to a 1D problem after
  closed-form elimination of the `η` dual variable; χ² uses a nested
  1D outer search over `λ` with inner bisection on `η` enforcing
  `Σ q_i = 1`; CVaR has the closed-form sort-based solution. All three
  satisfy KKT exactly (`χ²(Q*) = ρ` and `KL(Q*||P̂_n) = ρ` at the
  optimum) at machine precision.

* **Cross-fitting.** Three-fold rotation: each sample receives a
  Γ̂ score from estimators trained on the *other two* folds.

* **Outer minimax.** Fictitious play: at iteration `t`, fit a weighted
  classification tree (labels = `1{Γ ≥ 0}`, weights = `n · Q_t · |Γ|`),
  then update `Q` to the worst case against the *running average* of
  per-sample `(2 π^s(X) − 1) Γ`. We return the iterate with the largest
  empirical DR value.
