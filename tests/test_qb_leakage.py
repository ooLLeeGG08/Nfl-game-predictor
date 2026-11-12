"""Leakage tests for the v2 QB rating (qb.py). Run offline on data/qb_raw/.

1. Audit: every dated input used for a game (past games' player stats, past games'
   lead QB, injury-report rows, depth-chart snapshots) is strictly before kickoff.
   For 2009-2024 this covers the injury reports' date_modified; 2025+ reports have
   no timestamp in nflverse (see README, Known limitations).
2. Perturbation: scramble everything published at/after a cutoff kickoff C and check
   that no QB feature of a game kicking off at or before C moves. Games AT C have
   their own player stats scrambled, so a feature reading its own game is caught.
3. The actual starter (schedules qb_id) is never used.
4. The perturbation test catches a planted leak (so a passing run means something).
"""
import numpy as np
import pandas as pd
import pytest

import qb
from data.fetch import load_schedules

COLS = ["expected", "starter_out", "qb_rating", "qb_plays_prior"]
CUTOFFS = ["2015-11-01", "2021-10-15", "2025-12-01", "2026-10-20"]


@pytest.fixture(scope="module")
def base():
    sched = load_schedules(range(2002, 2027), allow_test=True)
    src = qb.load_sources(qb.QB_RAW, sched)
    audit = []
    feats = qb.build_qb_features(sched, *src, audit=audit)
    return sched, src, feats, audit


def test_every_dated_input_is_before_kickoff(base):
    *_, audit = base
    a = pd.DataFrame(audit, columns=["kick", "source", "ts"])
    assert set(a.source) >= {"pstats", "injury", "depth_snapshot", "prev_game"}
    late = a[pd.to_datetime(a.ts, utc=True) >= pd.to_datetime(a.kick, utc=True)]
    assert late.empty, late.head()


def _scrambled(sched, src, C):
    ps, inj, dw, ds = src
    rng = np.random.default_rng(1)
    kick = qb.team_games(sched).drop_duplicates("game_id").set_index("game_id").kick

    ps2 = ps.astype({"plays": float, "epa": float})
    late = (ps2.game_id.map(kick) >= C).to_numpy()
    ps2.loc[late, "epa"] = rng.normal(0, 30, late.sum())
    ps2.loc[late, "plays"] = rng.integers(1, 80, late.sum()).astype(float)
    ps2.loc[late, "player_id"] = rng.permutation(ps2.loc[late, "player_id"].to_numpy())

    # Injury rows by publication time: timestamped rows dated at/after C; rows
    # without a timestamp (2025+) when their game is after C. A game's own final
    # report is pre-game information and stays.
    wk_kick = qb.team_games(sched).set_index(["season", "week", "team"]).kick.to_dict()
    game_kick = pd.to_datetime(pd.Series([wk_kick.get(t) for t in zip(inj.season, inj.week, inj.team)]), utc=True)
    dm = pd.to_datetime(inj.date_modified.reset_index(drop=True), utc=True)
    late_inj = np.where(dm.isna(), game_kick.gt(C) | game_kick.isna(), dm.ge(C))
    inj2 = inj.copy()
    inj2.loc[late_inj, "report_status"] = rng.choice(["Out", "Doubtful", "Questionable"], late_inj.sum())

    ds2 = ds.copy()
    late_d = (ds2.dt >= C).to_numpy()
    ds2.loc[late_d, "gsis_id"] = rng.permutation(ds2.loc[late_d, "gsis_id"].to_numpy())
    return ps2, inj2, dw, ds2


def _cutoff(sched, cutoff):
    kick = qb.team_games(sched).kick
    return kick[kick >= pd.Timestamp(cutoff, tz="UTC")].min()


def _moved(before: pd.DataFrame, after: pd.DataFrame, mask) -> int:
    b = before[mask].set_index(["game_id", "side"])[COLS].sort_index()
    a = after.set_index(["game_id", "side"]).loc[b.index][COLS]
    same = (a == b) | (a.isna() & b.isna())
    return int((~same).any(axis=1).sum())


@pytest.mark.parametrize("cutoff", CUTOFFS)
def test_future_data_does_not_move_past_features(base, cutoff):
    sched, src, feats, _ = base
    C = _cutoff(sched, cutoff)
    assert (feats.kick == C).sum() >= 2
    after = qb.build_qb_features(sched, *_scrambled(sched, src, C))
    assert _moved(feats, after, feats.kick <= C) == 0


def test_actual_starter_is_never_used(base):
    sched, src, feats, _ = base
    s2 = sched.copy()
    s2["home_qb_id"], s2["away_qb_id"] = "X", "Y"
    assert _moved(feats, qb.build_qb_features(s2, *src), np.ones(len(feats), bool)) == 0


def test_perturbation_catches_a_planted_leak(base):
    sched, src, _, _ = base
    C = _cutoff(sched, CUTOFFS[1])
    leaky = qb.build_qb_features(sched, *src, _planted_leak=True)
    leaky_after = qb.build_qb_features(sched, *_scrambled(sched, src, C), _planted_leak=True)
    assert _moved(leaky, leaky_after, leaky.kick <= C) > 0
