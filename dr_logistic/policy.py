"""
Algorithm 1: Distributionally Robust Policy Learning from Observational Data.

Two pieces:

    1. doubly_robust_scores(...)  -- builds Gamma_i with sample-splitting D1/D2/D3
                                     using DRLogisticOutcome / DRLogisticPropensity
                                     trained on D2 / D3.
    2. dr_policy_tree(...)        -- solves the outer minimax problem
                                     max_pi inf_{Q in U} (1/n) sum_i Q_i (2 pi(X_i) - 1) Gamma_i
                                     over decision-tree policies of bounded depth.

The outer problem is convex in Q for fixed pi (and concave on the simplex with
the f-divergence constraint), but combinatorial in pi when we restrict to a
finite policy class.  We solve it with no-regret best-response dynamics:

    Q^{t+1}  = best response of the inner player to pi^t (worst-case Q under pi^t)
    pi^{t+1} = best response of the outer player to a *running average* of Q
               (weighted classification with sample weights Q_t * |Gamma|,
                labels sign(Gamma)).

We then return the iterate with the largest DR value.  This is an instance of
fictitious play; for our (compact) tree class, empirically it converges to a
near-saddle within a few rounds.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.tree import DecisionTreeClassifier

from .dro import dr_value, worst_case_Q
from .estimators import DRLogisticOutcome, DRLogisticPropensity


# ----------------------------------------------------------------------------
# Doubly robust scores  (Algorithm 1, lines 2-6)
# ----------------------------------------------------------------------------
@dataclass
class FittedScores:
    Gamma: np.ndarray
    X: np.ndarray
    Y: np.ndarray
    W: np.ndarray
    outcome: DRLogisticOutcome
    propensity: DRLogisticPropensity


def doubly_robust_scores(
    X: np.ndarray,
    Y: np.ndarray,
    W: np.ndarray,
    *,
    rho: float = 0.1,
    divergence: str = "kl",
    alpha: float = 0.1,
    propensity_clip: float = 0.05,
    estimator_kwargs: dict | None = None,
    outcome_cls=None,
    propensity_cls=None,
    seed: int = 0,
) -> FittedScores:
    """
    Build doubly robust scores Gamma_i for the *whole* dataset using
    cross-fitting with three folds (D1 receives the scores; D2 trains m_hat;
    D3 trains e_hat — and the role of each fold is rotated so every sample
    eventually receives a Gamma).

    Parameters
    ----------
    X : (n, d) covariates
    Y : (n,)   outcomes (binary)
    W : (n,)   treatments (0/1)
    rho, divergence, alpha :
        DRO parameters used when training the *nuisance* estimators
        (m_hat and e_hat) so they have low DR L2 error.
    estimator_kwargs : dict
        passed through to the underlying nuisance estimator.
    outcome_cls, propensity_cls :
        classes implementing the (fit, predict) interface of
        DRLogisticOutcome / DRLogisticPropensity respectively.
        Default uses the logistic-regression DR estimators; pass
        DRMLPOutcome / DRMLPPropensity for the MLP variant.

    Returns
    -------
    FittedScores
    """
    if outcome_cls is None:
        outcome_cls = DRLogisticOutcome
    if propensity_cls is None:
        propensity_cls = DRLogisticPropensity

    estimator_kwargs = dict(estimator_kwargs or {})
    estimator_kwargs.setdefault("divergence", divergence)
    estimator_kwargs.setdefault("rho", rho)
    estimator_kwargs.setdefault("alpha", alpha)

    n = len(X)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    folds = np.array_split(perm, 3)
    Gamma = np.zeros(n)

    # cross-fitting: for each "score" fold, train m_hat on one of the other
    # two folds and e_hat on the remaining fold
    for k in range(3):
        score_idx = folds[k]
        m_idx = folds[(k + 1) % 3]
        e_idx = folds[(k + 2) % 3]

        outcome = outcome_cls(**estimator_kwargs)
        outcome.fit(X[m_idx], W[m_idx], Y[m_idx])

        prop = propensity_cls(**estimator_kwargs)
        prop.fit(X[e_idx], W[e_idx])

        Xs, Ys, Ws = X[score_idx], Y[score_idx], W[score_idx]
        m1 = outcome.predict(Xs, 1)
        m0 = outcome.predict(Xs, 0)
        e_hat = prop.predict(Xs, clip=propensity_clip)

        # Gamma_i = m(X,1) - m(X,0) +  (W - e) / (e (1-e)) * (Y - m(X, W))
        m_w = np.where(Ws == 1, m1, m0)
        residual = Ys - m_w
        ipw = (Ws - e_hat) / (e_hat * (1.0 - e_hat))
        Gamma[score_idx] = m1 - m0 + ipw * residual

    # for downstream prediction (e.g. real-data evaluation) also fit one final
    # outcome / propensity on *all* data using the same DR config
    outcome_full = outcome_cls(**estimator_kwargs).fit(X, W, Y)
    prop_full = propensity_cls(**estimator_kwargs).fit(X, W)

    return FittedScores(
        Gamma=Gamma, X=X, Y=Y, W=W, outcome=outcome_full, propensity=prop_full
    )


# ----------------------------------------------------------------------------
# DR policy optimization over decision trees  (Algorithm 1, line 7)
# ----------------------------------------------------------------------------
@dataclass
class DRPolicyResult:
    policy: DecisionTreeClassifier
    dr_value: float
    history: list[float]
    all_policies: list[DecisionTreeClassifier]


def _fit_tree_policy(
    X: np.ndarray, Gamma: np.ndarray, Q: np.ndarray, max_depth: int, seed: int
) -> DecisionTreeClassifier:
    """
    Best-response policy: weighted decision-tree classifier with
       label   = 1{Gamma_i >= 0}
       weight  = n * Q_i * |Gamma_i|
    """
    n = len(X)
    labels = (Gamma >= 0).astype(int)
    weights = n * Q * np.abs(Gamma)
    if weights.sum() < 1e-10:
        weights = np.ones(n)
    tree = DecisionTreeClassifier(max_depth=max_depth, random_state=seed)
    tree.fit(X, labels, sample_weight=weights)
    return tree


def dr_policy_tree(
    X: np.ndarray,
    Gamma: np.ndarray,
    *,
    rho: float = 0.1,
    divergence: str = "kl",
    alpha: float = 0.1,
    max_depth: int = 2,
    n_iters: int = 15,
    seed: int = 0,
) -> DRPolicyResult:
    """
    Best-response dynamics for
        max_{pi in tree class}  inf_{Q: D_f(Q||Pn) <= rho}  mean_i Q_i (2 pi(X_i)-1) Gamma_i.

    Iteration t:
        1. fit tree pi^t to weighted classification with sample weights Q_t * |Gamma|
        2. update Q to the worst-case distribution under the *averaged* policy

    The averaged policy is taken in the *prediction* sense: we average the
    per-sample assignments (2 pi^s(X_i) - 1) Gamma_i over s.  This is the
    standard fictitious-play update in zero-sum games.
    """
    n = len(X)
    Q = np.ones(n) / n
    avg_signed = np.zeros(n)  # running mean of (2 pi^s(X) - 1) Gamma
    policies: list[DecisionTreeClassifier] = []
    values: list[float] = []

    for t in range(n_iters):
        # outer best response
        tree = _fit_tree_policy(X, Gamma, Q, max_depth=max_depth, seed=seed + t)
        policies.append(tree)

        pi_pred = tree.predict(X)
        z_t = (2.0 * pi_pred - 1.0) * Gamma
        # update fictitious-play average
        avg_signed = (avg_signed * t + z_t) / (t + 1)

        # inner best response against the averaged outer strategy
        Q = worst_case_Q(avg_signed, rho=rho, divergence=divergence, alpha=alpha)

        # bookkeeping: DR value of the *current* tree
        v = dr_value(z_t, rho=rho, divergence=divergence, alpha=alpha)
        values.append(v)

    # pick the iterate with the largest DR value
    best_idx = int(np.argmax(values))
    return DRPolicyResult(
        policy=policies[best_idx],
        dr_value=values[best_idx],
        history=values,
        all_policies=policies,
    )


# ----------------------------------------------------------------------------
# Convenience: evaluate any policy under the DR objective
# ----------------------------------------------------------------------------
def evaluate_policy_dr(
    policy,
    X: np.ndarray,
    Gamma: np.ndarray,
    *,
    rho: float = 0.1,
    divergence: str = "kl",
    alpha: float = 0.1,
) -> float:
    """DR value of a fitted sklearn-style policy on (X, Gamma)."""
    pi = policy.predict(X)
    z = (2.0 * pi - 1.0) * Gamma
    return dr_value(z, rho=rho, divergence=divergence, alpha=alpha)


def evaluate_policy_mean(policy, X: np.ndarray, Gamma: np.ndarray) -> float:
    """Plain (non-robust) average advantage."""
    pi = policy.predict(X)
    return float(np.mean((2.0 * pi - 1.0) * Gamma))
