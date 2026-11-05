"""The numpy replacements in mathx.py must agree with scipy (dev dependency)."""
import numpy as np
import pytest
from scipy.optimize import linear_sum_assignment, minimize_scalar
from scipy.special import ndtri
from scipy.stats import norm

import mathx


def test_norm_cdf_matches_scipy():
    x = np.linspace(-8, 8, 2001)
    np.testing.assert_allclose(mathx.norm_cdf(x), norm.cdf(x), atol=1e-15)


def test_norm_ppf_matches_scipy():
    p = np.concatenate([np.linspace(1e-9, 1e-3, 500), np.linspace(1e-3, 1 - 1e-3, 2000), np.linspace(1 - 1e-3, 1 - 1e-9, 500)])
    np.testing.assert_allclose(mathx.norm_ppf(p), ndtri(p), rtol=2e-9, atol=1e-9)


def test_minimize_bounded_matches_scipy():
    f = lambda s: (np.log(s) - 2.3) ** 2 + 0.01 * s
    ours = mathx.minimize_bounded(f, 5, 25)
    theirs = minimize_scalar(f, bounds=(5, 25), method="bounded").x
    assert ours == pytest.approx(theirs, abs=1e-4)


@pytest.mark.parametrize("n", [3, 4])
def test_best_assignment_matches_hungarian(n):
    rng = np.random.default_rng(n)
    for _ in range(50):
        score = rng.random((n, n))
        ours = sum(score[r, c] for r, c in mathx.best_assignment(score))
        rows, cols = linear_sum_assignment(-score)
        assert ours == pytest.approx(score[rows, cols].sum())


def test_vegas_sigma_unchanged_after_dropping_scipy():
    import json
    from train import REPORTS
    import market
    from data.fetch import TRAIN_SEASONS
    from features import build_features
    from live import ELO
    from train import load_raw
    sched, stats = load_raw()
    sigma = market.fit_spread_sigma(build_features(sched, stats, ELO), TRAIN_SEASONS)
    assert sigma == pytest.approx(json.loads((REPORTS / "info_test.json").read_text())["spread_sigma"], abs=0.01)
