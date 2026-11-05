"""The few scipy functions the web app needs, in plain numpy (keeps the Vercel bundle small).

Each one is tested against scipy in tests/test_mathx.py.
"""
from __future__ import annotations

import math
from itertools import permutations

import numpy as np

_erf = np.frompyfunc(math.erf, 1, 1)


def norm_cdf(x) -> np.ndarray:
    """Standard normal CDF, Phi(x)."""
    x = np.asarray(x, float)
    return (0.5 * (1 + _erf(x / math.sqrt(2)))).astype(float)


# Acklam's rational approximation of the inverse normal CDF (relative error < 1.2e-9).
_A = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
      1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
_B = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
      6.680131188771972e+01, -1.328068155288572e+01]
_C = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
      -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
_D = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00, 3.754408661907416e+00]
_LOW = 0.02425


def norm_ppf(p) -> np.ndarray:
    """Inverse of the standard normal CDF, Phi^-1(p), for 0 < p < 1."""
    p = np.asarray(p, float)
    out = np.empty_like(p)
    lo, hi = p < _LOW, p > 1 - _LOW
    mid = ~(lo | hi)

    q = p[mid] - 0.5
    r = q * q
    out[mid] = ((((((_A[0] * r + _A[1]) * r + _A[2]) * r + _A[3]) * r + _A[4]) * r + _A[5]) * q /
                (((((_B[0] * r + _B[1]) * r + _B[2]) * r + _B[3]) * r + _B[4]) * r + 1))

    def tail(pp):
        q = np.sqrt(-2 * np.log(pp))
        return ((((((_C[0] * q + _C[1]) * q + _C[2]) * q + _C[3]) * q + _C[4]) * q + _C[5]) /
                ((((_D[0] * q + _D[1]) * q + _D[2]) * q + _D[3]) * q + 1))

    out[lo] = tail(p[lo])
    out[hi] = -tail(1 - p[hi])
    return out


def minimize_bounded(f, lo: float, hi: float, tol: float = 1e-6) -> float:
    """Minimise a one-variable function on [lo, hi] by golden-section search."""
    g = (math.sqrt(5) - 1) / 2
    a, b = lo, hi
    c, d = b - g * (b - a), a + g * (b - a)
    fc, fd = f(c), f(d)
    while b - a > tol:
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - g * (b - a)
            fc = f(c)
        else:
            a, c, fc = c, d, fd
            d = a + g * (b - a)
            fd = f(d)
    return (a + b) / 2


def best_assignment(score: np.ndarray) -> list[tuple[int, int]]:
    """Rows-to-columns assignment maximising the total score (square, tiny matrices only).

    Tries every permutation, so it is meant for the 3x3 and 4x4 seeding problems.
    """
    n = score.shape[0]
    best = max(permutations(range(n)), key=lambda perm: sum(score[i, perm[i]] for i in range(n)))
    return list(enumerate(best))
