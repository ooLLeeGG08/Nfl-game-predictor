"""QB context (no probabilities) and the shadow-mode v2.

* No "QB-adjusted" text is rendered anywhere; v2 is only logged.
* The headline probability (and pick) is v1 in every case.
* QB lines: "Backup starting", Q/D badge and "(low sample)" when they apply.
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

import live
from features import MODEL_FEATURES

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

KICK = pd.Timestamp("2026-11-15T18:00:00Z")


def entry(diff=0.0, report_days=1.0, depth_days=1.0, status=None, replaces=None, report=True, depth=True,
          usual="00-1", starts=50):
    side = lambda name, st=None, rep=None: {"team": "X", "qb_id": "00-1", "name": name, "status": st,
                                            "rating": 0.05, "plays_prior": 900, "replaces": rep,
                                            "usual_starter": usual, "career_starts": starts}
    return {
        "kickoff": KICK.isoformat(),
        "report_as_of": (KICK - pd.Timedelta(days=report_days)).isoformat() if report else None,
        "depth_as_of": (KICK - pd.Timedelta(days=depth_days)).isoformat() if depth else None,
        "home": side("Home QB", status, replaces), "away": side("Away QB"),
        "qb_rating_diff": diff, "frozen": False,
    }


ROW = SimpleNamespace(game_id="G", home_team="HOM", away_team="AWY",
                      **dict(zip(MODEL_FEATURES, [40.0, 2.0, 0.0, 0.0, 0, 0])))


def v1(row=ROW):
    return float(live.home_win_prob(pd.DataFrame([{c: getattr(row, c) for c in MODEL_FEATURES}]))[0])


# ---------------------------------------------------------------- freshness --

@pytest.mark.parametrize("kw, ok", [
    ({}, True),
    ({"report_days": 3.99}, True),
    ({"report_days": 4.01}, False),          # report older than 4 days before kickoff
    ({"depth_days": 5}, False),              # depth chart too old
    ({"report": False}, False),              # no report yet
    ({"depth": False}, False),
    ({"report_days": -0.1}, False),          # seen after kickoff
])
def test_freshness_rule(kw, ok):
    assert live.qb_fresh(entry(**kw))[0] is ok


def test_missing_entry_is_unavailable():
    assert live.qb_fresh(None) == (False, "no QB data for this game")


# ----------------------------------------------------- shadow-mode v2 --

def test_v2_matches_the_frozen_pipeline():
    import joblib
    pipe = joblib.load(ROOT / "models" / "v2" / "pipeline.joblib")
    X = pd.DataFrame([{**{c: getattr(ROW, c) for c in MODEL_FEATURES}, "qb_rating_diff": 0.3}])
    assert live.home_win_prob_v2(X)[0] == pytest.approx(pipe.predict_proba(X)[0, 1], abs=1e-12)


# ------------------------------------------------- headline is always v1 --

@pytest.fixture
def extreme_qb(monkeypatch):
    """Fresh QB data with a huge rating gap for every game of the current week."""
    f = live.snapshot().features
    week = live.current_week()
    games = f[(f.season == live.CURRENT_SEASON) & (f.week == week)]
    data = {"games": {}}
    for i, g in enumerate(games.itertuples(index=False)):
        kick = live._kickoff(g)
        e = entry(diff=1.5 if i % 2 else -1.5, status="Questionable", replaces={"name": "Old QB", "status": "Out"})
        if kick is not None:
            k = pd.Timestamp(kick).tz_convert("UTC")
            e["kickoff"] = k.isoformat()
            e["report_as_of"] = (k - pd.Timedelta(days=1)).isoformat()
            e["depth_as_of"] = (k - pd.Timedelta(days=1)).isoformat()
        data["games"][g.game_id] = e
    monkeypatch.setattr(live, "qb_data", lambda: data)
    return games


PAGES = ["/", "/week", "/?home=BUF&away=SF"]


def test_week_headline_is_v1(extreme_qb):
    w = live.week_games()
    v1_home = dict(zip(extreme_qb.game_id, live.home_win_prob(extreme_qb)))
    for r in w["games"]:
        ph = v1_home[r["game_id"]]
        pick, p = (r["home"], ph) if ph >= 0.5 else (r["away"], 1 - ph)
        assert r["model_team"] == pick and r["model_p"] == pytest.approx(p, abs=1e-12)
        assert r["qb"]["available"] and "flag" not in r["qb"] and "p_v2_home" not in r["qb"]


def test_no_qb_adjusted_text_anywhere(extreme_qb):
    import app as app_module
    c = app_module.app.test_client()
    urls = PAGES + [f"/game/{gid}" for gid in extreme_qb.game_id]
    for u in urls:
        html = c.get(u).get_data(as_text=True)
        assert "QB-adjusted" not in html and "adds each starter" not in html, u


def test_game_headline_is_v1(extreme_qb):
    import app as app_module
    c = app_module.app.test_client()
    for g in extreme_qb.itertuples(index=False):
        p1 = float(live.home_win_prob(extreme_qb[extreme_qb.game_id == g.game_id])[0])
        d = live.game_detail(g.game_id)
        assert d["p_home"] == pytest.approx(p1, abs=1e-12)
        assert f"{p1 * 100:.1f}%" in c.get(f"/game/{g.game_id}").get_data(as_text=True)


def test_qb_line_labels(monkeypatch, extreme_qb):
    data = live.qb_data()
    g0, g1 = list(data["games"])[:2]
    data["games"][g0] = entry(status="Questionable", usual="00-9", starts=3)   # backup, Q, low sample
    data["games"][g1] = entry(status="Out", usual="00-1", starts=50)          # usual starter; Out is not badged
    for gid in (g0, g1):
        real = extreme_qb[extreme_qb.game_id == gid].iloc[0]
        kick = pd.Timestamp(live._kickoff(real)).tz_convert("UTC")
        data["games"][gid].update(kickoff=kick.isoformat(), report_as_of=(kick - pd.Timedelta(days=1)).isoformat(),
                                  depth_as_of=(kick - pd.Timedelta(days=1)).isoformat())
    import app as app_module
    c = app_module.app.test_client()
    page0 = c.get(f"/game/{g0}").get_data(as_text=True)
    assert 'class="qb-backup">Backup starting<' in page0 and 'class="qb-st">Q<' in page0
    assert 'class="qb-low">(low sample)<' in page0 and "EPA/play" in page0
    page1 = c.get(f"/game/{g1}").get_data(as_text=True)
    assert "Backup starting" not in page1 and "(low sample)" not in page1 and 'class="qb-st"' not in page1


def test_unavailable_text(monkeypatch):
    monkeypatch.setattr(live, "qb_data", lambda: {"games": {}})
    import app as app_module
    html = app_module.app.test_client().get("/week").get_data(as_text=True)
    assert "QB data unavailable" in html


# -------------------------------------------------------- append-only log --

def test_log_is_append_only(tmp_path, monkeypatch):
    import build_qb
    monkeypatch.setattr(build_qb, "LOG", tmp_path / "log.csv")
    now = pd.Timestamp("2026-11-12T12:00:00Z")
    e = {**entry(diff=0.1), "available": True, "p_v1": 0.6, "p_v2": 0.62}
    data = {"built_at": now.isoformat(), "games": {"G1": e}}
    assert build_qb.append_log(data) == 1
    assert build_qb.append_log({**data, "built_at": (now + pd.Timedelta(hours=6)).isoformat()}) == 0  # unchanged
    before = (tmp_path / "log.csv").read_text()
    e2 = {**e, "p_v2": 0.66}
    assert build_qb.append_log({"built_at": (now + pd.Timedelta(hours=12)).isoformat(), "games": {"G1": e2}}) == 1
    after = (tmp_path / "log.csv").read_text()
    assert after.startswith(before) and len(after.splitlines()) == 3


# ------------------------------------------------------------ shadow report --

def test_shadow_report_uses_last_fresh_pre_kickoff_row():
    import shadow_report as sr
    k = "2026-11-15T18:00:00+00:00"
    log = pd.DataFrame([
        # G1: two fresh rows before kickoff -> the later one counts; a row after kickoff is ignored
        {"logged_at": "2026-11-13T12:00:00+00:00", "game_id": "G1", "kickoff": k, "available": True, "p_v1": 0.6, "p_v2": 0.5},
        {"logged_at": "2026-11-14T12:00:00+00:00", "game_id": "G1", "kickoff": k, "available": True, "p_v1": 0.6, "p_v2": 0.7},
        {"logged_at": "2026-11-15T19:00:00+00:00", "game_id": "G1", "kickoff": k, "available": True, "p_v1": 0.6, "p_v2": 0.1},
        # G2: last pre-kickoff row has no fresh QB data -> the earlier fresh row counts
        {"logged_at": "2026-11-13T12:00:00+00:00", "game_id": "G2", "kickoff": k, "available": True, "p_v1": 0.4, "p_v2": 0.45},
        {"logged_at": "2026-11-14T12:00:00+00:00", "game_id": "G2", "kickoff": k, "available": False, "p_v1": 0.4, "p_v2": 0.2},
        # G3: never fresh -> excluded; G4: tie -> excluded from scoring
        {"logged_at": "2026-11-14T12:00:00+00:00", "game_id": "G3", "kickoff": k, "available": False, "p_v1": 0.5, "p_v2": 0.5},
        {"logged_at": "2026-11-14T12:00:00+00:00", "game_id": "G4", "kickoff": k, "available": True, "p_v1": 0.5, "p_v2": 0.9},
    ])
    rows = sr.select(log)
    assert sorted(rows.game_id) == ["G1", "G2", "G4"]
    assert rows.set_index("game_id").loc["G1", "p_v2"] == 0.7 and rows.set_index("game_id").loc["G2", "p_v2"] == 0.45
    res = pd.DataFrame({"game_id": ["G1", "G2", "G4"], "home_score": [24, 10, 17], "away_score": [20, 13, 17]})
    r = sr.score(rows, res)
    import math
    assert r["games_scored"] == 2
    assert r["log_likelihood"]["v1"] == pytest.approx(math.log(0.6) + math.log(0.6))
    assert r["log_likelihood"]["v2"] == pytest.approx(math.log(0.7) + math.log(0.55))
    assert sr.score(rows.iloc[:0], res)["games_scored"] == 0
