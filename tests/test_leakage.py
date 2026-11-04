"""Leakage guard: a game's features may only depend on games with an earlier gameday."""
import numpy as np
import pandas as pd
import pytest

from data.fetch import TEST_SEASON, HeldOutSeasonError, load_schedules
from features import MODEL_FEATURES, build_features

FEATURE_COLS = MODEL_FEATURES + [
    "home_elo", "away_elo", "home_pf5", "away_pf5", "home_to5", "away_to5",
    "home_games_played", "away_games_played",
]


def _sample_games(schedules, n=25, seed=0):
    """Random games plus the tricky cases: season openers, playoffs, neutral sites."""
    rng = np.random.default_rng(seed)
    picks = set(rng.choice(schedules.game_id, size=n, replace=False))
    picks |= set(schedules[schedules.week == 1].groupby("season").game_id.first())
    picks |= set(schedules[schedules.game_type != "REG"].game_id.head(5))
    picks |= set(schedules[schedules.location == "Neutral"].game_id.head(3))
    return sorted(picks)


def test_features_use_only_earlier_gamedays(schedules, team_stats):
    full = build_features(schedules, team_stats).set_index("game_id")

    for gid in _sample_games(schedules):
        game = schedules[schedules.game_id == gid].iloc[0]
        # Keep only strictly earlier games, plus this game with its result erased.
        past = schedules[schedules.gameday < game.gameday]
        target = game.to_frame().T.astype(schedules.dtypes.to_dict())
        target[["home_score", "away_score"]] = np.nan
        truncated = build_features(pd.concat([past, target]), team_stats).set_index("game_id")

        pd.testing.assert_series_equal(
            full.loc[gid, FEATURE_COLS].astype(float),
            truncated.loc[gid, FEATURE_COLS].astype(float),
            check_names=False,
            obj=f"features of {gid}",
        )


def test_future_results_do_not_change_past_features(schedules, team_stats):
    """Scramble every result from a cutoff date on; features up to that date must not move."""
    cutoff = pd.Timestamp("2020-11-01")
    scrambled = schedules.astype({"home_score": float, "away_score": float})
    future = scrambled.gameday >= cutoff
    rng = np.random.default_rng(1)
    scrambled.loc[future, "home_score"] = rng.integers(0, 50, future.sum())
    scrambled.loc[future, "away_score"] = rng.integers(0, 50, future.sum())

    a = build_features(schedules, team_stats).set_index("game_id")
    b = build_features(scrambled, team_stats).set_index("game_id")
    up_to = a[a.gameday <= cutoff].index  # includes games ON the cutoff day
    pd.testing.assert_frame_equal(a.loc[up_to, FEATURE_COLS], b.loc[up_to, FEATURE_COLS])
    # Sanity check that scrambling did something: later features should differ.
    assert not a.loc[a.gameday > cutoff, "elo_diff"].equals(b.loc[b.gameday > cutoff, "elo_diff"])


def test_test_season_is_locked():
    with pytest.raises(HeldOutSeasonError):
        load_schedules([TEST_SEASON])
