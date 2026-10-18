"""
Core DRO utilities for f-divergence ambiguity sets.

Supports three divergences:
  * 'kl'   : f(t) = t log t,        f*(s) = exp(s - 1)
  * 'chi2' : f(t) = (t-1)^2 / 2,    f*(s) = max(s + s^2/2, -1/2)
  * 'cvar' : CVaR at level alpha (constraint q_i <= 1/(alpha n), no rho)

Public API
----------
    dr_value(z, ...)    --> inf_{Q: D_f(Q||Pn) <= rho} (1/n) sum_i Q_i z_i
    worst_case_Q(z,...) --> the minimising probability vector

The dual identity used throughout is

    inf_Q E_Q[z]  =  sup_{lambda >= 0, eta} { -lambda mean f*((-z - eta)/lambda)
                                              - lambda rho - eta }.

For KL we eliminate eta in closed form,

    inf_Q E_Q[z]  =  sup_{lambda > 0} { -lambda rho - lambda log mean exp(-z / lambda) },

so KL collapses to a 1D problem and the worst-case Q is the
exponentially-tilted distribution  q_i \\propto exp(-z_i / lambda*).
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize, minimize_scalar


# ----------------------------------------------------------------------------
# conjugate functions and their derivatives
# ----------------------------------------------------------------------------
def f_conjugate(s: np.ndarray, divergence: str, alpha: float = 0.1) -> np.ndarray:
    if divergence == "kl":
        return np.exp(np.clip(s - 1.0, -50.0, 50.0))
    if divergence == "chi2":
        # f(t) = (t-1)^2/2 for t >= 0, +inf else.  Hence
        #     f*(s) = sup_{t>=0} { s t - (t-1)^2/2 }
        #           = (max(1+s, 0)^2 - 1) / 2
        # (interior optimum at t = 1+s when s >= -1; boundary t = 0 with value -1/2 else).
        clipped = np.maximum(1.0 + s, 0.0)
        return 0.5 * (clipped * clipped - 1.0)
    if divergence == "cvar":
        return np.maximum(s, 0.0) / alpha
    raise ValueError(f"unknown divergence {divergence!r}")


def f_conjugate_grad(s: np.ndarray, divergence: str, alpha: float = 0.1) -> np.ndarray:
    if divergence == "kl":
        return np.exp(np.clip(s - 1.0, -50.0, 50.0))
    if divergence == "chi2":
        return np.maximum(1.0 + s, 0.0)
    if divergence == "cvar":
        return (s > 0.0).astype(float) / alpha
    raise ValueError(f"unknown divergence {divergence!r}")


# ----------------------------------------------------------------------------
# CVaR closed form
# ----------------------------------------------------------------------------
def _cvar_value(z: np.ndarray, alpha: float) -> float:
    n = len(z)
    sorted_z = np.sort(z)
    threshold = alpha * n
    k = int(np.floor(threshold))
    r = threshold - k
    if k == 0:
        return float(sorted_z[0])
    total = sorted_z[:k].sum() / threshold
    if k < n and r > 1e-12:
        total += sorted_z[k] * r / threshold
    return float(total)


def _cvar_weights(z: np.ndarray, alpha: float) -> np.ndarray:
    n = len(z)
    order = np.argsort(z)
    threshold = alpha * n
    k = int(np.floor(threshold))
    r = threshold - k
    q = np.zeros(n)
    if k > 0:
        q[order[:k]] = 1.0 / threshold
    if k < n and r > 1e-12:
        q[order[k]] = r / threshold
    return q


# ----------------------------------------------------------------------------
# KL: 1D dual after eliminating eta
# ----------------------------------------------------------------------------
def _kl_neg_dual(log_lam: float, z: np.ndarray, rho: float) -> float:
    """Negative of (-lambda*rho - lambda*log E[exp(-z/lambda)]), so we minimise."""
    lam = float(np.exp(log_lam))
    a = -z / lam
    m = a.max()
    log_mean = m + np.log(np.mean(np.exp(a - m)))
    return lam * rho + lam * log_mean


def _kl_solve(z: np.ndarray, rho: float):
    res = minimize_scalar(
        _kl_neg_dual,
        args=(z, rho),
        method="bounded",
        bounds=(-12.0, 12.0),
        options={"xatol": 1e-9},
    )
    log_lam = float(res.x)
    lam = float(np.exp(log_lam))
    val = -float(res.fun)  # primal: inf_Q E_Q[z]
    a = -z / lam
    a = a - a.max()
    q = np.exp(a)
    q = q / q.sum()
    return val, lam, q


# ----------------------------------------------------------------------------
# chi2: nested 1D dual.  Outer: line search over lambda > 0.  Inner: bisect
# on eta to enforce sum_i max(1 - (z_i + eta)/lambda, 0) / n = 1, which is
# the KKT stationarity condition d G / d eta = 0.  This is more robust than
# joint 2D optimization, which struggles when the constraint q_i >= 0 binds.
# ----------------------------------------------------------------------------
def _chi2_eta_for_lambda(z: np.ndarray, lam: float) -> float:
    """
    Find eta such that mean_i max(1 - (z_i + eta) / lambda, 0) = 1.

    The LHS is a continuous, monotonically *decreasing*, piecewise-linear
    function of eta with a finite range, so we bisect.
    """
    n = len(z)
    # bounds: when eta = -z.max() + lam, ratios are <= 1 (mean <= 1 may not hold).
    # widen until the desired root is bracketed.
    lo, hi = float(-z.max() - 5.0 * lam - 1.0), float(-z.min() + lam + 1.0)

    def g(eta):
        return np.maximum(1.0 - (z + eta) / lam, 0.0).mean() - 1.0

    g_lo, g_hi = g(lo), g(hi)
    # widen if needed
    while g_lo < 0:
        lo -= 5.0 * lam
        g_lo = g(lo)
    while g_hi > 0:
        hi += 5.0 * lam
        g_hi = g(hi)
    # bisection
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        gm = g(mid)
        if gm > 0:
            lo, g_lo = mid, gm
        else:
            hi, g_hi = mid, gm
        if hi - lo < 1e-12 * max(abs(lo), abs(hi), 1.0):
            break
    return 0.5 * (lo + hi)


def _chi2_dual_at_lambda(log_lam: float, z: np.ndarray, rho: float) -> float:
    """Negative dual value at fixed lambda (with optimal eta plugged in)."""
    lam = float(np.exp(log_lam))
    eta = _chi2_eta_for_lambda(z, lam)
    s = (-z - eta) / lam
    fst = f_conjugate(s, "chi2")
    return lam * fst.mean() + lam * rho + eta


def _chi2_solve(z: np.ndarray, rho: float):
    z = np.asarray(z, dtype=float)
    n = len(z)
    res = minimize_scalar(
        _chi2_dual_at_lambda,
        args=(z, rho),
        method="bounded",
        bounds=(-12.0, 12.0),
        options={"xatol": 1e-9},
    )
    log_lam = float(res.x)
    lam = float(np.exp(log_lam))
    eta = _chi2_eta_for_lambda(z, lam)
    val = -float(res.fun)  # primal: inf_Q E_Q[z]
    ratios = np.maximum(1.0 - (z + eta) / lam, 0.0)
    q = ratios / n  # sum is 1 by construction
    return val, lam, q


# ----------------------------------------------------------------------------
# public API
# ----------------------------------------------------------------------------
def dr_value(z, rho=0.1, divergence="kl", alpha=0.1) -> float:
    """inf_{Q: D_f(Q || Pn) <= rho}  (1/n) sum_i Q_i z_i."""
    z = np.asarray(z, dtype=float)
    if divergence == "cvar":
        return _cvar_value(z, alpha)
    if rho <= 0.0:
        return float(z.mean())
    if divergence == "kl":
        v, _, _ = _kl_solve(z, rho)
        return v
    if divergence == "chi2":
        v, _, _ = _chi2_solve(z, rho)
        return v
    raise ValueError(f"unknown divergence {divergence!r}")


def worst_case_Q(z, rho=0.1, divergence="kl", alpha=0.1) -> np.ndarray:
    """The minimising probability vector for `dr_value`."""
    z = np.asarray(z, dtype=float)
    n = len(z)
    if divergence == "cvar":
        return _cvar_weights(z, alpha)
    if rho <= 0.0:
        # constraint forces Q = P_n  ==>  uniform weights
        return np.ones(n) / n
    if divergence == "kl":
        _, _, q = _kl_solve(z, rho)
        return q
    if divergence == "chi2":
        _, _, q = _chi2_solve(z, rho)
        return q
    raise ValueError(f"unknown divergence {divergence!r}")


def dual_solution(z, rho=0.1, divergence="kl", alpha=0.1):
    """Return (lambda*, q*) of the inner DRO dual (KL/chi2)."""
    if divergence == "cvar":
        raise ValueError("CVaR has no lambda; use worst_case_Q for the weights.")
    if divergence == "kl":
        _, lam, q = _kl_solve(z, rho)
    else:
        _, lam, q = _chi2_solve(z, rho)
    return lam, q
