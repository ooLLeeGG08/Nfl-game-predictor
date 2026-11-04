"""Prediction-time behaviour of the saved model (no data download needed)."""
import numpy as np
import pandas as pd
import pytest

from features import MODEL_FEATURES
from live import PIPE, home_win_prob


def _rows(**overrides):
    base = {"elo_diff": 0.0, "pd_diff": 0.0, "to_diff": 0.0, "rest_diff": 0.0, "div_game": 0, "neutral_site": 0}
    return pd.DataFrame([{**base, **overrides}])[MODEL_FEATURES]


def test_equal_teams_at_neutral_site_are_a_coin_flip():
    assert home_win_prob(_rows(neutral_site=1))[0] == pytest.approx(0.5)


def test_neutral_prediction_ignores_which_team_is_listed_home():
    a_home = _rows(elo_diff=120, pd_diff=4, to_diff=0.6, rest_diff=3, neutral_site=1)
    b_home = _rows(elo_diff=-120, pd_diff=-4, to_diff=-0.6, rest_diff=-3, neutral_site=1)
    assert home_win_prob(a_home)[0] == pytest.approx(1 - home_win_prob(b_home)[0])


def test_non_neutral_games_use_the_model_unchanged():
    X = pd.concat([_rows(elo_diff=d) for d in (-200, 0, 150)], ignore_index=True)
    np.testing.assert_allclose(home_win_prob(X), PIPE.predict_proba(X)[:, 1])
    assert home_win_prob(_rows())[0] > 0.5  # home advantage still applies


def test_fast_array_probability_matches_dataframe_version():
    from live import home_win_prob_array
    rng = np.random.default_rng(0)
    X = np.column_stack([rng.normal(0, 150, 300), rng.normal(0, 8, 300), rng.normal(0, 1, 300),
                         rng.integers(-7, 8, 300), rng.integers(0, 2, 300), rng.integers(0, 2, 300)]).astype(float)
    np.testing.assert_allclose(home_win_prob_array(X), home_win_prob(pd.DataFrame(X, columns=MODEL_FEATURES)), atol=1e-12)
