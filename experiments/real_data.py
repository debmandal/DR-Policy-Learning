"""
Real-data experiment for Algorithm 1 (DR logistic-regression nuisances).

Semi-synthetic California GAIN-like dataset: 6 counties with realistic
feature distributions and county-specific treatment effects calibrated
to the qualitative findings of Hotz, Imbens & Klerman (2006) -- e.g.
Riverside has a large positive TE, LA is near zero.

Usage
-----
Default settings:
    python experiments/real_data.py

Customise:
    python experiments/real_data.py \\
        --train-counties Riverside SanDiego Tulare \\
        --test-counties Alameda Butte LosAngeles \\
        --depths 1 2 3 4 \\
        --rho 0.2 --cvar-alpha 0.2 \\
        --n-seeds 10 --n-epochs 50 \\
        --output-dir results/

Available counties:
    Alameda, Butte, LosAngeles, Riverside, SanDiego, Tulare
"""
from __future__ import annotations

import argparse
import os
import sys
import time

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
# semi-synthetic California GAIN data
# ============================================================================
COUNTIES = ["Alameda", "Butte", "LosAngeles", "Riverside", "SanDiego", "Tulare"]

COUNTY_PARAMS = {
    "Alameda":    dict(n=900,  prop=0.50,  age=33, edu=12.2,  earn_lo=2.5,
                       te_intercept=0.05,  te_age=-0.001, te_edu=0.012),
    "Butte":      dict(n=600,  prop=0.55,  age=35, edu=11.5,  earn_lo=1.8,
                       te_intercept=0.03,  te_age=-0.002, te_edu=0.008),
    "LosAngeles": dict(n=1500, prop=0.45,  age=34, edu=10.6,  earn_lo=2.1,
                       te_intercept=-0.03, te_age=-0.001, te_edu=0.005),
    "Riverside":  dict(n=1100, prop=0.48,  age=32, edu=11.4,  earn_lo=2.0,
                       te_intercept=0.18,  te_age=-0.0005, te_edu=0.015),
    "SanDiego":   dict(n=900,  prop=0.52,  age=34, edu=12.0,  earn_lo=2.4,
                       te_intercept=0.10,  te_age=-0.0008, te_edu=0.011),
    "Tulare":     dict(n=700,  prop=0.50,  age=36, edu=10.2,  earn_lo=1.6,
                       te_intercept=0.07,  te_age=-0.0015, te_edu=0.009),
}

BETA_BASE = np.array([-0.02, 0.18, 0.10, 0.05, 0.20])
INTERCEPT_BASE = -0.10


def generate_county_data(county: str, seed: int):
    p = COUNTY_PARAMS[county]
    rng = np.random.default_rng(seed)
    n = p["n"]

    age = rng.normal(p["age"], 6.0, size=n).clip(18, 60)
    edu = rng.normal(p["edu"], 2.0, size=n).clip(6, 18)
    log_earn = rng.normal(p["earn_lo"], 1.0, size=n).clip(0, 6)
    female = (rng.random(n) < 0.55).astype(float)
    married = (rng.random(n) < 0.40).astype(float)

    X = np.column_stack([age, edu, log_earn, female, married])

    e = np.clip(p["prop"] + 0.02 * (edu - p["edu"]), 0.15, 0.85)
    W = (rng.random(n) < e).astype(int)

    tau = (
        p["te_intercept"]
        + p["te_age"] * (age - p["age"])
        + p["te_edu"] * (edu - p["edu"])
    )

    X_centered = np.column_stack([
        age - p["age"], edu - p["edu"], log_earn - p["earn_lo"], female, married
    ])
    mu = INTERCEPT_BASE + X_centered @ BETA_BASE

    mu1 = np.clip(mu + tau / 2.0, 0.02, 0.98)
    mu0 = np.clip(mu - tau / 2.0, 0.02, 0.98)

    Y = np.where(
        W == 1,
        (rng.random(n) < mu1).astype(float),
        (rng.random(n) < mu0).astype(float),
    )

    return dict(X=X, W=W, Y=Y, tau=tau, county=county)


def build_pooled_dataset(counties, seed: int):
    parts = [generate_county_data(c, seed=seed * 100 + i) for i, c in enumerate(counties)]
    X = np.vstack([p["X"] for p in parts])
    W = np.concatenate([p["W"] for p in parts])
    Y = np.concatenate([p["Y"] for p in parts])
    tau = np.concatenate([p["tau"] for p in parts])
    county_idx = np.concatenate([
        np.full(len(parts[i]["X"]), i, dtype=int) for i in range(len(parts))
    ])
    return X, W, Y, tau, county_idx


def true_value(policy, X: np.ndarray, tau: np.ndarray) -> float:
    pi = policy.predict(X)
    return float(np.mean((2.0 * pi - 1.0) * tau))


# ============================================================================
# core experiment
# ============================================================================
def run_train_test_split(args):
    rows = []
    est_kw = dict(n_epochs=args.n_epochs, batch_size=args.batch_size,
                   lr=args.lr, l2=args.l2)

    for seed in range(args.n_seeds):
        X_tr, W_tr, Y_tr, tau_tr, _ = build_pooled_dataset(args.train_counties,
                                                             seed=seed)
        test_data = {c: generate_county_data(c, seed=10_000 + seed * 50 + i)
                     for i, c in enumerate(args.test_counties)}

        for depth in args.depths:
            for label, div, alpha, rho_eff in [
                ("ERM",       "kl",   0.1,             0.0),
                ("DR (KL)",   "kl",   0.1,             args.rho),
                (r"DR ($\chi^2$)", "chi2", 0.1,        args.rho),
                ("DR (CVaR)", "cvar", args.cvar_alpha, 0.0),
            ]:
                fitted = doubly_robust_scores(
                    X_tr, Y_tr, W_tr,
                    rho=rho_eff if rho_eff > 0 else 0.01,
                    divergence=div, alpha=alpha,
                    estimator_kwargs=est_kw, seed=seed,
                )
                res = dr_policy_tree(
                    X_tr, fitted.Gamma,
                    rho=rho_eff, divergence=div, alpha=alpha,
                    max_depth=depth, n_iters=args.tree_iters, seed=seed,
                )
                rows.append(dict(
                    seed=seed, depth=depth, policy=label, county="TRAIN",
                    value=true_value(res.policy, X_tr, tau_tr),
                    dr_value=res.dr_value,
                ))
                for c, td in test_data.items():
                    rows.append(dict(
                        seed=seed, depth=depth, policy=label, county=c,
                        value=true_value(res.policy, td["X"], td["tau"]),
                        dr_value=np.nan,
                    ))
        print(f"  seed {seed}: done")

    return pd.DataFrame(rows)


def plot_county_split(df, args):
    fig, axes = plt.subplots(1, len(args.depths),
                              figsize=(4.0 * len(args.depths), 4.4),
                              sharey=True)
    if len(args.depths) == 1:
        axes = [axes]

    palette = {
        "ERM":            "#888888",
        "DR (KL)":        "#1b6ca8",
        r"DR ($\chi^2$)": "#cc5500",
        "DR (CVaR)":      "#2c8a3e",
    }
    policies = ["ERM", "DR (KL)", r"DR ($\chi^2$)", "DR (CVaR)"]

    for ax, depth in zip(axes, args.depths):
        sub = df[df.depth == depth]
        agg = sub.groupby(["policy", "county"]).value.agg(["mean", "std", "count"]).reset_index()
        agg["se"] = agg["std"] / np.sqrt(agg["count"])

        ordered_counties = ["TRAIN"] + list(args.test_counties)
        x_pos = np.arange(len(ordered_counties))
        bar_w = 0.20

        for i, pol in enumerate(policies):
            sub2 = agg[agg.policy == pol].set_index("county").reindex(ordered_counties)
            ax.bar(x_pos + (i - 1.5) * bar_w, sub2["mean"], width=bar_w,
                    yerr=sub2["se"], color=palette[pol],
                    label=pol if depth == args.depths[0] else None,
                    capsize=2)

        ax.axhline(0, color="black", lw=0.5)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(ordered_counties, rotation=30, ha="right")
        ax.set_title(f"depth = {depth}")
        ax.grid(axis="y", alpha=0.3)
        if depth == args.depths[0]:
            ax.set_ylabel(r"True policy value  $\mathbb{E}[(2\pi - 1)\tau]$")

    axes[0].legend(loc="upper left", fontsize=9)
    fig.suptitle("Train on three counties; evaluate on held-out counties\n"
                 + f"Train counties: {', '.join(args.train_counties)}",
                 fontsize=10)
    fig.tight_layout()

    out_png = os.path.join(args.output_dir, "real_data_county_split.png")
    fig.savefig(out_png, dpi=130)
    plt.close(fig)
    print(f"  saved {out_png}")


# ============================================================================
def parse_args():
    p = argparse.ArgumentParser(
        description="Algorithm 1 real-data experiment (DR logistic nuisances).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # county split
    p.add_argument("--train-counties", nargs="+",
                    default=["Riverside", "SanDiego", "Tulare"],
                    choices=COUNTIES,
                    help="counties used for training")
    p.add_argument("--test-counties", nargs="+",
                    default=["Alameda", "Butte", "LosAngeles"],
                    choices=COUNTIES,
                    help="counties used for held-out evaluation")

    # nuisance estimator
    p.add_argument("--n-epochs", type=int, default=25,
                    help="SGD epochs per nuisance fit")
    p.add_argument("--batch-size", type=int, default=128,
                    help="minibatch size for SGD")
    p.add_argument("--lr", type=float, default=0.05,
                    help="SGD learning rate")
    p.add_argument("--l2", type=float, default=1e-3,
                    help="L2 regularisation")

    # policy class
    p.add_argument("--depths", type=int, nargs="+", default=[1, 2, 3],
                    help="decision-tree policy depths to try")
    p.add_argument("--tree-iters", type=int, default=15,
                    help="fictitious-play iterations")

    # DRO
    p.add_argument("--rho", type=float, default=0.2,
                    help="rho for KL / chi^2 DRO")
    p.add_argument("--cvar-alpha", type=float, default=0.2,
                    help="alpha for CVaR DRO")

    # control
    p.add_argument("--n-seeds", type=int, default=5,
                    help="number of seeds for averaging")
    p.add_argument("--output-dir", default=os.path.join(_THIS_DIR, "..", "results"),
                    help="directory for CSV / PNG outputs")

    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    overlap = set(args.train_counties) & set(args.test_counties)
    if overlap:
        raise ValueError(f"train and test counties overlap: {overlap}")
    print(f"Args: {vars(args)}")

    print("\n=== Real-data (California GAIN-like) experiment ===")
    print(f"Training on: {args.train_counties}")
    print(f"Testing on:  {args.test_counties}\n")

    t0 = time.time()
    df = run_train_test_split(args)
    csv_path = os.path.join(args.output_dir, "real_data_summary.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n  saved {csv_path}")

    if 2 in args.depths:
        print("\nMean true value per policy x county (depth=2):")
        print(df[df.depth == 2].groupby(["policy", "county"]).value.mean()
              .unstack().round(3))

    plot_county_split(df, args)
    print(f"\nTotal time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
