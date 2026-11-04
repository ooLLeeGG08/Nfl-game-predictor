"""Betting-market (Vegas) win probabilities. Shared by evaluate.py and the app.

* Moneylines present: implied probability of each side, normalised so the two
  sum to 1 (removes the bookmaker's margin, the "vig").
* Spread only: P(home) = Phi(spread_line / sigma), sigma fitted on the
  training seasons to match moneyline probabilities.
* Pick'em (spread 0): 0.5. No line at all: NaN.

nflverse convention: positive spread_line = home team favored.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import norm


def moneyline_prob(ml) -> np.ndarray:
    """American odds -> implied probability (still includes the vig)."""
    ml = np.asarray(ml, float)
    # np.where evaluates both branches; an even-money line (-100) makes the
    # unused one divide by zero, which is harmless.
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(ml < 0, -ml / (-ml + 100), 100 / (ml + 100))


def vigfree(d: pd.DataFrame) -> np.ndarray:
    ph, pa = moneyline_prob(d.home_moneyline), moneyline_prob(d.away_moneyline)
    return ph / (ph + pa)


def fit_spread_sigma(df: pd.DataFrame, seasons) -> float:
    """Pick sigma so Phi(spread / sigma) best matches moneyline probabilities."""
    d = df[df.season.isin(list(seasons))].dropna(subset=["spread_line", "home_moneyline", "away_moneyline"])
    target = vigfree(d)
    res = minimize_scalar(lambda s: np.mean((norm.cdf(d.spread_line / s) - target) ** 2), bounds=(5, 25), method="bounded")
    return float(res.x)


def vegas_prob(df: pd.DataFrame, sigma: float) -> pd.Series:
    """Home win probability implied by the market. NaN when no line exists."""
    p = pd.Series(np.nan, index=df.index)
    has_ml = df.home_moneyline.notna() & df.away_moneyline.notna()
    p[has_ml] = vigfree(df[has_ml])
    spread_only = ~has_ml & df.spread_line.notna()
    p[spread_only] = norm.cdf(df.spread_line[spread_only] / sigma)
    p[df.spread_line == 0] = 0.5  # pick'em
    return p
