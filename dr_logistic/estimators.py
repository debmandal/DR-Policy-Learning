"""
Distributionally robust logistic regression via *primal* minibatch SGD.

For each minibatch B = {i_1, ..., i_b} we

    1. compute per-sample losses     l_i = ell(theta; X_i, Y_i),
    2. solve the inner DRO problem on the batch
           q* = argmax_{q in Delta(b) : (1/b) sum_j f(b q_j) <= rho}
                    sum_j q_j l_j
       in closed form via dr_value / worst_case_Q (KL/chi2: dual,
       CVaR: sort-based),
    3. take a primal gradient step
           theta <- theta - lr * (sum_j q*_j  nabla l_j  + l2 * theta).

This is the natural primal counterpart of the dual minibatch method of
Levy, Carmon, Duchi & Sidford (NeurIPS 2020).  Because the inner problem
is convex and small (b unknowns), we solve it exactly per batch rather
than tracking dual variables across iterations.  The per-batch q*
already accounts for both the divergence constraint and the simplex
constraint, so there are no (lambda, eta) hyperparameters or learning
rates to tune.

For CVaR, the same worst_case_Q routine returns the sort-based weights
q_j = 1 / (alpha b) on the worst alpha-fraction of losses (with a
fractional last weight); no rho is used.

Two estimators are exposed:

    DRLogisticOutcome    --  E[Y | X, W] for binary Y
                              (one logistic head per arm)
    DRLogisticPropensity --  P(W = 1 | X)
"""
from __future__ import annotations

import numpy as np
from scipy.special import expit

from .dro import worst_case_Q


# ----------------------------------------------------------------------------
# helper: numerically stable cross-entropy and its per-sample gradient
# ----------------------------------------------------------------------------
def _bce_loss_and_grad(X: np.ndarray, y: np.ndarray, beta: np.ndarray):
    """
    Per-sample binary cross-entropy and its gradient.

    Returns
    -------
    loss : (B,) array
    grad : (B, p) array of dloss_i / dbeta
    """
    z = X @ beta
    # log(1 + exp(z)) - y * z  is the stable form
    loss = np.logaddexp(0.0, z) - y * z
    grad = (expit(z) - y)[:, None] * X
    return loss, grad


# ----------------------------------------------------------------------------
# generic DR-trained logistic regression  (primal minibatch SGD)
# ----------------------------------------------------------------------------
class _DRLogistic:
    """Internal base class — fits a binary logistic regression with DRO
    via primal minibatch SGD.  At each step the inner sup_q is solved
    *exactly* on the current batch via worst_case_Q."""

    def __init__(
        self,
        divergence: str = "kl",
        rho: float = 0.1,
        alpha: float = 0.1,
        lr: float = 0.05,
        n_epochs: int = 200,
        batch_size: int = 128,
        l2: float = 1e-3,
        seed: int = 0,
    ):
        self.divergence = divergence
        self.rho = rho
        self.alpha = alpha
        self.lr = lr
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.l2 = l2
        self.seed = seed

    # ------------------------------------------------------------------
    def fit(self, X: np.ndarray, y: np.ndarray):
        rng = np.random.default_rng(self.seed)
        n, d = X.shape
        # add intercept
        X_aug = np.hstack([X, np.ones((n, 1))])
        p = d + 1

        beta = np.zeros(p)
        bs = min(self.batch_size, n)

        for _ in range(self.n_epochs):
            shuffle = rng.permutation(n)
            for s in range(0, n, bs):
                batch = shuffle[s : s + bs]
                Xb, yb = X_aug[batch], y[batch]

                loss_i, grad_i = _bce_loss_and_grad(Xb, yb, beta)

                # ------ inner DRO step: q* = argmax_q sum_i q_i l_i ------
                # Note worst_case_Q solves the *minimisation* inf_q E_Q[z],
                # so for the maximisation we negate.
                q_star = worst_case_Q(
                    -loss_i,
                    rho=self.rho,
                    divergence=self.divergence,
                    alpha=self.alpha,
                )

                # ------ primal gradient step ------
                # gradient = sum_i q_i grad_i  +  l2 * beta
                grad_beta = (q_star[:, None] * grad_i).sum(axis=0) + self.l2 * beta
                beta -= self.lr * grad_beta

        self.beta_ = beta
        return self

    # ------------------------------------------------------------------
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_aug = np.hstack([X, np.ones((len(X), 1))])
        return expit(X_aug @ self.beta_)


# ----------------------------------------------------------------------------
# public-facing wrappers
# ----------------------------------------------------------------------------
class DRLogisticPropensity:
    """Distributionally robust propensity score estimator e(X) = P(W=1|X)."""

    def __init__(self, **kwargs):
        self._model = _DRLogistic(**kwargs)

    def fit(self, X: np.ndarray, W: np.ndarray):
        self._model.fit(X, W.astype(float))
        return self

    def predict(self, X: np.ndarray, clip: float = 0.05) -> np.ndarray:
        p = self._model.predict_proba(X)
        return np.clip(p, clip, 1.0 - clip)


class DRLogisticOutcome:
    """
    Distributionally robust outcome estimator m(x, w) = E[Y | X=x, W=w]
    for binary Y.  Fits one DR-logistic per treatment arm.
    """

    def __init__(self, **kwargs):
        self._kwargs = kwargs
        self._models: dict[int, _DRLogistic | None] = {}

    def fit(self, X: np.ndarray, W: np.ndarray, Y: np.ndarray):
        for w in (0, 1):
            mask = W == w
            if mask.sum() == 0:
                # degenerate: no samples for this arm -- return constant 0.5
                self._models[w] = None
                continue
            self._models[w] = _DRLogistic(**self._kwargs).fit(X[mask], Y[mask].astype(float))
        return self

    def predict(self, X: np.ndarray, w: int) -> np.ndarray:
        if self._models.get(w) is None:
            return 0.5 * np.ones(X.shape[0])
        return self._models[w].predict_proba(X)
