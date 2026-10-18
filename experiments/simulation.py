"""
Simulation experiment for Algorithm 1 (DR logistic-regression nuisances).

Usage
-----
Default settings:
    python experiments/simulation.py

Customise:
    python experiments/simulation.py \\
        --n 2000 --p 8 --n-seeds 10 --max-depth 3 \\
        --rho-grid 0.05 0.1 0.2 0.4 0.8 \\
        --shift-grid 0 1 2 3 4 5 \\
        --n-epochs 50 --batch-size 128 --lr 0.05 \\
        --shift-rho 0.2 --cvar-alpha 0.2 \\
        --output-dir results/

Two sub-experiments are run, both writing PNG + CSV to --output-dir:

    1. dr-value-vs-rho      -- empirical DR value of the learnt policy
                                across rho for each divergence
    2. distribution-shift   -- evaluate ERM vs DR(KL/chi2/CVaR) policies
                                under a tilted test distribution

Skip an experiment with --skip-rho-sweep or --skip-shift-experiment.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

# allow running as a script from the project root
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_THIS_DIR, ".."))

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dr_logistic import (
    doubly_robust_scores,
    dr_policy_tree,
    evaluate_policy_dr,
    evaluate_policy_mean,
)


# ============================================================================
# data-generating process  (Athey & Wager 2021, Section 5 style)
# ============================================================================
def sample_dataset(n: int, p: int, seed: int):
    """Draw observational data (X, W, Y) and return covariates,
    treatment, outcome and the *true* heterogeneous treatment effect tau(X)."""
    rng = np.random.default_rng(seed)

    X = rng.uniform(0.0, 1.0, size=(n, p))

    # propensity: depends on X1 (overlap maintained: e in [0.2, 0.8])
    e = 0.2 + 0.6 * X[:, 0]
    W = (rng.random(n) < e).astype(int)

    # baseline outcome
    mu = 0.4 + 0.3 * X[:, 1] - 0.2 * X[:, 2]

    # heterogeneous treatment effect: positive for some sub-populations,
    # negative for others
    tau = 0.4 * (X[:, 0] - 0.5) + 0.3 * (X[:, 1] - 0.3)

    mu1 = np.clip(mu + tau / 2.0, 0.02, 0.98)
    mu0 = np.clip(mu - tau / 2.0, 0.02, 0.98)

    Y = np.where(
        W == 1,
        (rng.random(n) < mu1).astype(float),
        (rng.random(n) < mu0).astype(float),
    )
    return X, W, Y, tau


def shifted_test_set(n: int, p: int, shift_strength: float, seed: int):
    """Acceptance-rejection sample from Q(x) ~ P0(x) * exp(-s * x_1)."""
    rng = np.random.default_rng(seed)
    accepted = []
    while len(accepted) < n:
        cand = rng.uniform(0.0, 1.0, size=(2 * n, p))
        w = np.exp(-shift_strength * cand[:, 0])
        w = w / w.max()
        keep = rng.random(2 * n) < w
        accepted.extend(cand[keep])
    X = np.array(accepted[:n])
    tau = 0.4 * (X[:, 0] - 0.5) + 0.3 * (X[:, 1] - 0.3)
    return X, tau


def true_value(policy, X: np.ndarray, tau: np.ndarray) -> float:
    """E_X[(2 pi(X) - 1) tau(X)] = the true value of the policy on this X."""
    pi = policy.predict(X)
    return float(np.mean((2.0 * pi - 1.0) * tau))


# ============================================================================
# experiment 1:  DR value as a function of rho
# ============================================================================
def experiment_dr_value_vs_rho(args):
    print("\n=== Experiment 1: DR value vs rho ===")
    rows = []
    est_kw = dict(n_epochs=args.n_epochs, batch_size=args.batch_size,
                   lr=args.lr, l2=args.l2)

    for seed in range(args.n_seeds):
        X, W, Y, tau = sample_dataset(args.n, args.p, seed=seed)
        for div in ("kl", "chi2", "cvar"):
            for rho in args.rho_grid:
                # for CVaR we re-interpret rho as alpha
                alpha = rho if div == "cvar" else 0.1
                rho_eff = rho if div != "cvar" else 0.0
                fitted = doubly_robust_scores(
                    X, Y, W,
                    rho=rho_eff,
                    divergence=div,
                    alpha=alpha,
                    estimator_kwargs=est_kw,
                    seed=seed,
                )
                res = dr_policy_tree(
                    X, fitted.Gamma,
                    rho=rho_eff, divergence=div, alpha=alpha,
                    max_depth=args.max_depth, n_iters=args.tree_iters, seed=seed,
                )
                pop_val = true_value(res.policy, X, tau)
                rows.append(dict(
                    seed=seed, divergence=div, rho=rho,
                    dr_value=res.dr_value,
                    plain_mean=evaluate_policy_mean(res.policy, X, fitted.Gamma),
                    population_value=pop_val,
                ))
        print(f"  seed {seed}: done")

    df = pd.DataFrame(rows)
    csv_path = os.path.join(args.output_dir, "simulation_dr_value_vs_rho.csv")
    df.to_csv(csv_path, index=False)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    palette = dict(kl="#1b6ca8", chi2="#cc5500", cvar="#2c8a3e")
    label = dict(kl=r"KL", chi2=r"$\chi^2$", cvar=r"CVaR")

    g = df.groupby(["divergence", "rho"]).agg(
        dr_mean=("dr_value", "mean"),
        dr_se=("dr_value", lambda v: v.std(ddof=1) / np.sqrt(len(v))),
        pop_mean=("population_value", "mean"),
        pop_se=("population_value", lambda v: v.std(ddof=1) / np.sqrt(len(v))),
    ).reset_index()

    for div, sub in g.groupby("divergence"):
        axes[0].errorbar(sub["rho"], sub["dr_mean"], yerr=sub["dr_se"],
                         marker="o", color=palette[div], label=label[div], capsize=2)
        axes[1].errorbar(sub["rho"], sub["pop_mean"], yerr=sub["pop_se"],
                         marker="o", color=palette[div], label=label[div], capsize=2)

    axes[0].set_xlabel(r"$\rho$ / $\alpha$ (CVaR)")
    axes[0].set_ylabel(r"DR value $\hat A_{n,f}^{est}(\hat\pi_n)$")
    axes[0].set_title("Empirical DR value of learnt policy")
    axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].set_xlabel(r"$\rho$ / $\alpha$ (CVaR)")
    axes[1].set_ylabel(r"True value $\mathbb{E}_{P_0}[(2\hat\pi_n-1)\tau]$")
    axes[1].set_title("Population value on training distribution")
    axes[1].legend(); axes[1].grid(alpha=0.3)
    fig.tight_layout()

    out_png = os.path.join(args.output_dir, "simulation_dr_value_vs_rho.png")
    fig.savefig(out_png, dpi=130)
    plt.close(fig)
    print(f"  saved {csv_path} and {out_png}")
    return df


# ============================================================================
# experiment 2: performance under distribution shift
# ============================================================================
def experiment_distribution_shift(args):
    print("\n=== Experiment 2: distribution-shift robustness ===")
    rows = []
    est_kw = dict(n_epochs=args.n_epochs, batch_size=args.batch_size,
                   lr=args.lr, l2=args.l2)

    for seed in range(args.n_seeds):
        X_tr, W_tr, Y_tr, _ = sample_dataset(args.n, args.p, seed=seed)

        # ERM = no DRO at all (rho=~0, no chi^2 wrapping)
        fitted_erm = doubly_robust_scores(
            X_tr, Y_tr, W_tr, rho=0.01, divergence="kl",
            estimator_kwargs=est_kw, seed=seed,
        )
        erm_res = dr_policy_tree(
            X_tr, fitted_erm.Gamma, rho=0.01, divergence="kl",
            max_depth=args.max_depth, n_iters=args.tree_iters, seed=seed,
        )

        policies = {"ERM (non-DR)": erm_res.policy}
        for div, alpha in (("kl", 0.1), ("chi2", 0.1), ("cvar", args.cvar_alpha)):
            rho_eff = 0.0 if div == "cvar" else args.shift_rho
            fitted = doubly_robust_scores(
                X_tr, Y_tr, W_tr,
                rho=rho_eff, divergence=div, alpha=alpha,
                estimator_kwargs=est_kw, seed=seed,
            )
            res = dr_policy_tree(
                X_tr, fitted.Gamma, rho=rho_eff, divergence=div, alpha=alpha,
                max_depth=args.max_depth, n_iters=args.tree_iters, seed=seed,
            )
            label = {"kl": "DR (KL)", "chi2": r"DR ($\chi^2$)", "cvar": "DR (CVaR)"}[div]
            policies[label] = res.policy

        for s in args.shift_grid:
            X_te, tau_te = shifted_test_set(args.n_test, args.p,
                                              shift_strength=s,
                                              seed=1000 + seed)
            for name, pi in policies.items():
                v = true_value(pi, X_te, tau_te)
                rows.append(dict(seed=seed, policy=name, shift=s, value=v))
        print(f"  seed {seed}: done")

    df = pd.DataFrame(rows)
    csv_path = os.path.join(args.output_dir, "simulation_shift_robustness.csv")
    df.to_csv(csv_path, index=False)

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    palette = {"ERM (non-DR)": "#888888", "DR (KL)": "#1b6ca8",
                r"DR ($\chi^2$)": "#cc5500", "DR (CVaR)": "#2c8a3e"}
    for policy, sub in df.groupby("policy"):
        agg = sub.groupby("shift").agg(
            mean=("value", "mean"),
            se=("value", lambda v: v.std(ddof=1) / np.sqrt(len(v))),
        )
        ax.errorbar(agg.index, agg["mean"], yerr=agg["se"],
                    marker="o", label=policy, color=palette.get(policy),
                    capsize=2)
    ax.axhline(0, color="black", lw=0.6, alpha=0.4)
    ax.set_xlabel("Shift strength  $s$  (X tilted toward small $X_1$)")
    ax.set_ylabel(r"True value  $\mathbb{E}_{Q_s}[(2\hat\pi - 1)\tau(X)]$")
    ax.set_title(rf"Out-of-distribution policy value (rho={args.shift_rho}, "
                  f"depth={args.max_depth})")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()

    out_png = os.path.join(args.output_dir, "simulation_shift_robustness.png")
    fig.savefig(out_png, dpi=130)
    plt.close(fig)
    print(f"  saved {csv_path} and {out_png}")
    return df


# ============================================================================
def parse_args():
    p = argparse.ArgumentParser(
        description="Algorithm 1 simulation experiment (DR logistic nuisances).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # data
    p.add_argument("--n", type=int, default=1500,
                    help="training sample size")
    p.add_argument("--p", type=int, default=6,
                    help="number of covariates")
    p.add_argument("--n-test", type=int, default=4000,
                    help="size of each shifted test set (Experiment 2)")
    p.add_argument("--n-seeds", type=int, default=5,
                    help="number of independent seeds for averaging")

    # nuisance estimator (DR logistic regression)
    p.add_argument("--n-epochs", type=int, default=25,
                    help="SGD epochs per nuisance fit")
    p.add_argument("--batch-size", type=int, default=128,
                    help="minibatch size for SGD")
    p.add_argument("--lr", type=float, default=0.05,
                    help="SGD learning rate")
    p.add_argument("--l2", type=float, default=1e-3,
                    help="L2 regularisation strength")

    # policy class & outer optimisation
    p.add_argument("--max-depth", type=int, default=2,
                    help="decision-tree policy depth")
    p.add_argument("--tree-iters", type=int, default=15,
                    help="fictitious-play iterations for the outer minimax")

    # DRO grids
    p.add_argument("--rho-grid", type=float, nargs="+",
                    default=[0.02, 0.05, 0.1, 0.2, 0.4],
                    help="rho values for Experiment 1 (also used as alpha for CVaR)")
    p.add_argument("--shift-grid", type=float, nargs="+",
                    default=[0.0, 1.0, 2.0, 3.0, 4.0],
                    help="shift strengths for Experiment 2")
    p.add_argument("--shift-rho", type=float, default=0.2,
                    help="rho used in Experiment 2 (KL / chi^2)")
    p.add_argument("--cvar-alpha", type=float, default=0.2,
                    help="alpha used for CVaR in Experiment 2")

    # control
    p.add_argument("--output-dir", default=os.path.join(_THIS_DIR, "..", "results"),
                    help="directory for CSV / PNG outputs")
    p.add_argument("--skip-rho-sweep", action="store_true",
                    help="skip Experiment 1")
    p.add_argument("--skip-shift-experiment", action="store_true",
                    help="skip Experiment 2")

    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Args: {vars(args)}")

    t0 = time.time()
    if not args.skip_rho_sweep:
        experiment_dr_value_vs_rho(args)
    if not args.skip_shift_experiment:
        experiment_distribution_shift(args)
    print(f"\nTotal time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
