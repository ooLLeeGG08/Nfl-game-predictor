"""Correctness checks for individual features, against independent computations."""
import numpy as np
import pandas as pd
import pytest

from features import EloParams, build_features, elo_expected, elo_update, turnover_margins


@pytest.fixture(scope="module")
def feats(schedules, team_stats):
    return build_features(schedules, team_stats, return_state=True)


def _long(df):
    """One row per (team, game) with the team's own perspective."""
    home = df.assign(team=df.home_team, pf=df.home_score, pa=df.away_score, pf5=df.home_pf5, gp=df.home_games_played)
    away = df.assign(team=df.away_team, pf=df.away_score, pa=df.home_score, pf5=df.away_pf5, gp=df.away_games_played)
    return pd.concat([home, away]).sort_values(["team", "gameday"])


def test_rolling_matches_plain_last5_mean_from_game6(feats):
    df, _ = feats
    lg = _long(df)
    # Independent computation: within each team-season, mean of the previous 5 games.
    lg["expected"] = lg.groupby(["team", "season"]).pf.transform(lambda s: s.shift(1).rolling(5).mean())
    late = lg[lg.gp >= 5]
    np.testing.assert_allclose(late.pf5, late.expected)


def test_week1_uses_prior_season_final(feats):
    df, _ = feats
    lg = _long(df)
    lg["last5_end"] = lg.groupby(["team", "season"]).pf.transform(lambda s: s.tail(5).mean())
    final = lg.groupby(["team", "season"]).last5_end.first().rename("prior").reset_index()
    final["season"] += 1
    opener = lg[lg.gp == 0].merge(final, on=["team", "season"])
    assert len(opener) > 0
    np.testing.assert_allclose(opener.pf5, opener.prior)


def test_blend_weight_week3(feats):
    df, _ = feats
    lg = _long(df)
    lg["cur"] = lg.groupby(["team", "season"]).pf.transform(lambda s: s.shift(1).expanding().mean())
    lg["last5_end"] = lg.groupby(["team", "season"]).pf.transform(lambda s: s.tail(5).mean())
    final = lg.groupby(["team", "season"]).last5_end.first().rename("prior").reset_index()
    final["season"] += 1
    g2 = lg[lg.gp == 2].merge(final, on=["team", "season"])
    np.testing.assert_allclose(g2.pf5, 0.4 * g2.cur + 0.6 * g2.prior)


def test_turnover_margin_is_symmetric(team_stats):
    margins = turnover_margins(team_stats)
    per_game = pd.Series(margins).groupby(level=0).sum()  # level 0 = game_id
    assert len(per_game) > 500
    assert (per_game == 0).all()


def test_elo_is_zero_sum_and_regresses():
    p = EloParams()
    h, a = elo_update(1550, 1450, 27, 10, neutral=False, params=p)
    assert h > 1550 and a < 1450 and h + a == pytest.approx(3000)
    assert elo_expected(1500, 1500, 0) == pytest.approx(0.5)
    assert elo_expected(1500, 1500, p.hfa) > 0.5


def test_elo_mean_stays_at_1500(feats):
    _, teams = feats
    assert np.mean([t.elo for t in teams.values()]) == pytest.approx(1500, abs=1e-6)


def test_neutral_flag_covers_super_bowls(feats):
    df, _ = feats
    assert df[df.game_type == "SB"].neutral_site.all()


def test_unplayed_games_get_features_but_do_not_update(schedules, team_stats):
    last = schedules.sort_values("gameday").iloc[-1]
    blanked = schedules.copy()
    blanked.loc[blanked.game_id == last.game_id, ["home_score", "away_score"]] = np.nan
    df, teams = build_features(blanked, team_stats, return_state=True)
    row = df[df.game_id == last.game_id].iloc[0]
    assert not row.played and np.isnan(row.home_win) and not np.isnan(row.elo_diff)
    assert teams[last.home_team].elo == pytest.approx(row.home_elo)
