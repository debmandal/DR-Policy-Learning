"""
Visualize the decision-tree policies learnt by Algorithm 1 on the
California GAIN-like dataset (DR logistic-regression nuisances).

Usage
-----
    python experiments/visualize_trees.py
    python experiments/visualize_trees.py --depth 3 --rho 0.3
    python experiments/visualize_trees.py --train-counties Riverside SanDiego \\
        --depth 2 --output-dir results/
"""
from __future__ import annotations

import argparse
import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_THIS_DIR, ".."))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree, export_text

from dr_logistic import doubly_robust_scores, dr_policy_tree
from experiments.real_data import (
    build_pooled_dataset,
    generate_county_data,
    true_value,
    COUNTIES,
)

FEATURE_NAMES = ["age", "edu_yrs", "log_earn", "female", "married"]
CLASS_NAMES = ["do not treat", "treat"]


def train_one(X, Y, W, *, divergence, alpha, rho, depth, seed, est_kw, tree_iters):
    rho_eff = 0.0 if divergence == "cvar" else rho
    fitted = doubly_robust_scores(
        X, Y, W,
        rho=rho_eff if rho_eff > 0 else 0.01,
        divergence=divergence, alpha=alpha,
        estimator_kwargs=est_kw, seed=seed,
    )
    res = dr_policy_tree(
        X, fitted.Gamma,
        rho=rho_eff, divergence=divergence, alpha=alpha,
        max_depth=depth, n_iters=tree_iters, seed=seed,
    )
    return res, fitted


def parse_args():
    p = argparse.ArgumentParser(
        description="Visualise learnt tree policies on California GAIN-like data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--train-counties", nargs="+",
                    default=["Riverside", "SanDiego", "Tulare"], choices=COUNTIES)
    p.add_argument("--test-counties", nargs="+",
                    default=["Alameda", "Butte", "LosAngeles"], choices=COUNTIES)
    p.add_argument("--depth", type=int, default=2,
                    help="depth for the 4-method comparison grid")
    p.add_argument("--depth-spotlight", type=int, default=3,
                    help="depth for the standalone DR(chi^2) spotlight figure")
    p.add_argument("--rho", type=float, default=0.2)
    p.add_argument("--cvar-alpha", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-epochs", type=int, default=80,
                    help="more epochs than other scripts since we're doing a single fit")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--l2", type=float, default=1e-3)
    p.add_argument("--tree-iters", type=int, default=20)
    p.add_argument("--output-dir", default=os.path.join(_THIS_DIR, "..", "results"))
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Args: {vars(args)}")

    est_kw = dict(n_epochs=args.n_epochs, batch_size=args.batch_size,
                   lr=args.lr, l2=args.l2)

    X_tr, W_tr, Y_tr, tau_tr, _ = build_pooled_dataset(args.train_counties, seed=args.seed)
    test_data = {c: generate_county_data(c, seed=999 + i)
                 for i, c in enumerate(args.test_counties)}

    methods = [
        ("ERM",            "kl",   0.1,             0.0),
        ("DR (KL)",        "kl",   0.1,             args.rho),
        (r"DR ($\chi^2$)", "chi2", 0.1,             args.rho),
        ("DR (CVaR)",      "cvar", args.cvar_alpha, 0.0),
    ]

    # --- comparison grid at args.depth ---
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    axes = axes.flatten()
    for ax, (label, div, alpha, rho) in zip(axes, methods):
        res, _ = train_one(X_tr, Y_tr, W_tr, divergence=div, alpha=alpha, rho=rho,
                            depth=args.depth, seed=args.seed,
                            est_kw=est_kw, tree_iters=args.tree_iters)
        train_val = true_value(res.policy, X_tr, tau_tr)
        test_vals = {c: true_value(res.policy, td["X"], td["tau"])
                     for c, td in test_data.items()}

        plot_tree(res.policy, feature_names=FEATURE_NAMES, class_names=CLASS_NAMES,
                  filled=True, rounded=True, impurity=False,
                  proportion=True, precision=2, ax=ax, fontsize=9)
        test_str = "  ".join(f"{c[:3]}={v:+.3f}" for c, v in test_vals.items())
        ax.set_title(
            f"{label}   (depth {args.depth})\n"
            f"DR value={res.dr_value:+.3f}   train value={train_val:+.3f}\n"
            f"test:  {test_str}", fontsize=10,
        )

    fig.suptitle(
        f"Decision-tree policies learnt on California GAIN-like data (depth {args.depth})\n"
        f"trained on {{{', '.join(args.train_counties)}}}, seed={args.seed}",
        fontsize=11, y=0.995,
    )
    fig.tight_layout()
    out_png = os.path.join(args.output_dir, f"real_data_trees_depth{args.depth}.png")
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_png}")

    # --- standalone DR(chi^2) spotlight at args.depth_spotlight ---
    res, _ = train_one(X_tr, Y_tr, W_tr, divergence="chi2", alpha=0.1, rho=args.rho,
                        depth=args.depth_spotlight, seed=args.seed,
                        est_kw=est_kw, tree_iters=args.tree_iters)
    train_val = true_value(res.policy, X_tr, tau_tr)
    test_vals = {c: true_value(res.policy, td["X"], td["tau"])
                 for c, td in test_data.items()}

    fig, ax = plt.subplots(figsize=(18, 9))
    plot_tree(res.policy, feature_names=FEATURE_NAMES, class_names=CLASS_NAMES,
              filled=True, rounded=True, impurity=False,
              proportion=True, precision=2, ax=ax, fontsize=11)
    test_str = "   ".join(f"{c}={v:+.3f}" for c, v in test_vals.items())
    ax.set_title(
        rf"DR ($\chi^2$, $\rho={args.rho}$) -- depth-{args.depth_spotlight} policy"
        + " on California GAIN-like data\n"
        f"DR value = {res.dr_value:+.3f}   "
        f"train ({', '.join(args.train_counties)}) = {train_val:+.3f}\n"
        f"held-out test counties:   {test_str}", fontsize=12,
    )
    fig.tight_layout()
    out_png = os.path.join(args.output_dir,
                            f"real_data_tree_chi2_depth{args.depth_spotlight}.png")
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_png}")

    # --- text dumps for the appendix ---
    text_lines = []
    for label, div, alpha, rho in methods:
        res, _ = train_one(X_tr, Y_tr, W_tr, divergence=div, alpha=alpha, rho=rho,
                            depth=args.depth_spotlight, seed=args.seed,
                            est_kw=est_kw, tree_iters=args.tree_iters)
        text = export_text(res.policy, feature_names=FEATURE_NAMES, show_weights=False)
        text = text.replace("class: 1", "TREAT      ").replace(
            "class: 0", "DO NOT TREAT")
        text_lines.append(f"--- {label} ---\n{text}")
    out_txt = os.path.join(args.output_dir, "real_data_trees.txt")
    with open(out_txt, "w") as fh:
        fh.write("\n".join(text_lines))
    print(f"saved {out_txt}")


if __name__ == "__main__":
    main()
