"""Live predictions for the app: current ratings and matchup probabilities.

The app never retrains. It loads the saved pipeline + Elo config, replays all
games from 2002 through the latest completed 2026 week to get each team's
current Elo and last-5 stats, and scores matchups from that state.

Data refresh: historical seasons are committed in data/raw/ and never change.
The 2026 files are re-downloaded when older than the TTL (see data.fetch), and
the in-memory state is rebuilt on the same TTL.
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pandas as pd

from data.fetch import (BURN_IN_START, CURRENT_SEASON, CURRENT_TTL_SECONDS, FIRST_SEASON, TRAIN_SEASONS,
                        load_schedules, load_team_stats)
import market
from features import MODEL_FEATURES, EloParams, TeamState, build_features
from teams import NAMES, same_division
from paths import CONFIG_PATH, MODEL_JSON, ON_VERCEL, WRITABLE_DIR

# The saved model as plain numbers (exported by train.py from models/pipeline.joblib),
# so the web app needs neither scikit-learn nor joblib's pickle of it.
_MODEL = json.loads(MODEL_JSON.read_text())
assert _MODEL["features"] == MODEL_FEATURES, "models/model.json is out of date: run python train.py --export-json"
_MEAN, _SCALE, _COEF = (np.array(_MODEL[k]) for k in ("mean", "scale", "coef"))
_INTERCEPT = float(_MODEL["intercept"])
ELO = EloParams(**json.loads(CONFIG_PATH.read_text())["elo"])


@dataclass
class Snapshot:
    built_at: float
    as_of_week: int                 # latest fully completed 2026 week
    teams: dict[str, TeamState]     # state after that week
    schedule: pd.DataFrame          # full 2026 schedule (played + upcoming)
    features: pd.DataFrame          # pre-kickoff features for every game through 2026
    spread_sigma: float             # spread -> probability mapping, fitted on training seasons
    teams_now: dict[str, TeamState]  # state after every played game (season simulation)


_lock = threading.Lock()
_snapshot: Snapshot | None = None


def last_completed_week(season_sched: pd.DataFrame) -> int:
    """Highest week where every game has a final score (0 if none)."""
    done = season_sched.groupby("week").home_score.apply(lambda s: s.notna().all())
    done = done[done]
    # Weeks must be complete consecutively from week 1.
    week = 0
    for w in sorted(season_sched.week.unique()):
        if w in done.index:
            week = w
        else:
            break
    return int(week)


def _build() -> Snapshot:
    # The 2025 test season was evaluated once and is now ordinary history,
    # needed so 2026 ratings carry over correctly.
    seasons = range(BURN_IN_START, CURRENT_SEASON + 1)
    sched = load_schedules(seasons, allow_test=True)
    stats = load_team_stats(seasons, allow_test=True)

    current = sched[sched.season == CURRENT_SEASON]
    week = last_completed_week(current)
    # Ratings "as of the latest completed week": drop later results, even if some are in.
    through = sched[(sched.season < CURRENT_SEASON) | (sched.week <= week)]
    _, teams = build_features(through, stats, ELO, return_state=True)
    for t in teams.values():  # before week 1 completes, apply the off-season regression
        if t.season != CURRENT_SEASON:
            t.start_season(CURRENT_SEASON, ELO)
    features, teams_now = build_features(sched, stats, ELO, return_state=True)  # all games, incl. upcoming
    gametime = sched.set_index("game_id").gametime
    features["gametime"] = features.game_id.map(gametime)
    sigma = market.fit_spread_sigma(features, TRAIN_SEASONS)
    return Snapshot(time.time(), week, teams, current.copy(), features, sigma, teams_now)


def snapshot() -> Snapshot:
    """Current snapshot, rebuilt when older than the TTL."""
    global _snapshot
    with _lock:
        if _snapshot is None or time.time() - _snapshot.built_at > CURRENT_TTL_SECONDS:
            _snapshot = _build()
        return _snapshot


# Features that flip sign when home and away are swapped. div_game and
# neutral_site are the same from either side.
ANTISYMMETRIC = ["elo_diff", "pd_diff", "to_diff", "rest_diff"]


def home_win_prob(X: pd.DataFrame) -> np.ndarray:
    """Model probability that the listed home team wins, one per row.

    At a neutral site "home" is only a label, but the model learned a spurious
    penalty for it from ~51 neutral training games (equal teams got 46%, not
    50%). For neutral rows we therefore average the two labelings:
        p = (p(A listed home) + 1 - p(B listed home)) / 2
    which makes equal teams exactly 50% and the answer independent of which
    team is listed first. Non-neutral rows use the model unchanged.
    """
    return home_win_prob_array(X[MODEL_FEATURES].to_numpy(float))


def team_summary(team: str) -> dict:
    t = snapshot().teams[team]
    r = t.rolling()
    return {"team": team, "name": NAMES[team], "elo": t.elo, "pd5": r["pd"], "to5": r["to"], "pf5": r["pf"], "pa5": r["pa"]}


def predict_matchup(home: str, away: str, neutral: bool) -> dict:
    """Home win probability for a hypothetical game played next.

    Rest is unknown for a hypothetical game, so both teams are assumed to have
    equal rest (rest_diff = 0).
    """
    h, a = team_summary(home), team_summary(away)
    row = {
        "elo_diff": h["elo"] - a["elo"],
        "pd_diff": h["pd5"] - a["pd5"],
        "to_diff": h["to5"] - a["to5"],
        "rest_diff": 0,
        "div_game": int(same_division(home, away)),
        "neutral_site": int(neutral),
    }
    p = float(home_win_prob(pd.DataFrame([row]))[0])
    return {"home": h, "away": a, "p_home": p, "p_away": 1 - p, "inputs": row, "neutral": neutral}


def games_modeled() -> int:
    """Played, non-tie games from the first training season through today."""
    f = snapshot().features
    return int((f.played & ~f.is_tie & (f.season >= FIRST_SEASON)).sum())


# ------------------------------------------------------------- this week --

ET = ZoneInfo("America/New_York")  # nflverse kickoff times are US Eastern
ROUND_NAMES = {"WC": "Wild Card round", "DIV": "Divisional round", "CON": "Conference championships", "SB": "Super Bowl"}
DISAGREE = 0.10  # model vs Vegas gap (in home-win probability) that highlights a row


def current_week() -> int:
    """Earliest week of the current season with an unplayed game (else the last week)."""
    c = snapshot().features
    c = c[c.season == CURRENT_SEASON]
    upcoming = c[~c.played]
    return int(upcoming.week.min() if len(upcoming) else c.week.max())


def _kickoff(row) -> datetime | None:
    if not isinstance(row.gametime, str) or ":" not in row.gametime:
        return None  # kickoff not scheduled yet
    hh, mm = map(int, row.gametime.split(":"))
    return row.gameday.to_pydatetime().replace(hour=hh, minute=mm, tzinfo=ET)


def _spread_text(row) -> str:
    s = row.spread_line
    if pd.isna(s):
        return "no line"
    if s == 0:
        return "PK"
    fav = row.home_team if s > 0 else row.away_team
    return f"{fav} \u2212{abs(s):g}"


def _pick(row, p_home: float) -> tuple[str, float]:
    return (row.home_team, p_home) if p_home >= 0.5 else (row.away_team, 1 - p_home)


def _scored(games: pd.DataFrame) -> pd.DataFrame:
    games = games.copy()
    games["p_model"] = home_win_prob(games)
    games["p_vegas"] = market.vegas_prob(games, snapshot().spread_sigma)
    return games


def week_games(week: int | None = None) -> dict:
    """Every game of a week, with the model next to the market."""
    f = snapshot().features
    week = current_week() if week is None else week
    games = f[(f.season == CURRENT_SEASON) & (f.week == week)].sort_values(["gameday", "gametime", "game_id"])
    games = _scored(games)

    rows = []
    for g in games.itertuples(index=False):
        model_team, model_p = _pick(g, g.p_model)
        has_line = not math.isnan(g.p_vegas)
        vegas_team, vegas_p = _pick(g, g.p_vegas) if has_line else (None, None)
        kickoff = _kickoff(g)
        winner = None
        if g.played and g.home_score != g.away_score:
            winner = g.home_team if g.home_score > g.away_score else g.away_team
        rows.append({
            "game_id": g.game_id, "home": g.home_team, "away": g.away_team,
            "neutral": bool(g.neutral_site), "played": bool(g.played),
            "home_score": None if not g.played else int(g.home_score),
            "away_score": None if not g.played else int(g.away_score),
            "kickoff_iso": kickoff.isoformat() if kickoff else None,
            "kickoff_et": f"{kickoff:%a %H:%M} ET" if kickoff else "TBD",
            "model_team": model_team, "model_p": model_p,
            "vegas_team": vegas_team, "vegas_p": vegas_p, "spread": _spread_text(g),
            "disagree": has_line and abs(g.p_model - g.p_vegas) > DISAGREE,
            "model_right": None if winner is None else model_team == winner,
            "vegas_right": None if (winner is None or not has_line) else vegas_team == winner,
        })

    types = set(games.game_type)
    title = f"Week {week}" if types <= {"REG"} else ROUND_NAMES.get(next(iter(types - {"REG"})), f"Week {week}")
    final = [r for r in rows if r["played"] and r["model_right"] is not None]
    return {
        "week": week, "title": title, "games": rows,
        "n_final": len(final),
        "model_right": sum(r["model_right"] for r in final),
        "vegas_right": sum(bool(r["vegas_right"]) for r in final),
        "n_upcoming": sum(not r["played"] for r in rows),
    }


def game_detail(game_id: str) -> dict | None:
    """One game's actual pre-kickoff inputs and both probabilities."""
    f = snapshot().features
    match = f[f.game_id == game_id]
    if match.empty:
        return None
    g = _scored(match).iloc[0]

    def side(prefix: str, team: str) -> dict:
        return {"team": team, "name": NAMES.get(team, team), "elo": g[f"{prefix}_elo"],
                "pd5": g[f"{prefix}_pd5"], "to5": g[f"{prefix}_to5"], "rest": g[f"{prefix}_rest"]}

    kickoff = _kickoff(g)
    return {
        "game_id": game_id, "season": int(g.season), "week": int(g.week),
        "home": side("home", g.home_team), "away": side("away", g.away_team),
        "p_home": float(g.p_model), "p_away": 1 - float(g.p_model),
        "p_vegas": None if math.isnan(g.p_vegas) else float(g.p_vegas), "spread": _spread_text(g),
        "neutral": bool(g.neutral_site), "inputs": {"div_game": int(g.div_game)},
        "played": bool(g.played),
        "score": f"{g.away_team} {int(g.away_score)} \u2013 {int(g.home_score)} {g.home_team}" if g.played else None,
        "kickoff_iso": kickoff.isoformat() if kickoff else None,
        "kickoff_et": f"{kickoff:%a %b %-d, %H:%M} ET" if kickoff else "TBD",
    }


def next_game() -> dict | None:
    """The next game to kick off (earliest unplayed game of the current week)."""
    upcoming = [g for g in week_games()["games"] if not g["played"]]
    if not upcoming:
        return None
    g = dict(upcoming[0])
    g["p_home"] = g["model_p"] if g["model_team"] == g["home"] else 1 - g["model_p"]
    # Nicknames ("Jaguars") fit on one line in the hero card; full names wrap.
    g["home_name"], g["away_name"] = NAMES[g["home"]].split()[-1], NAMES[g["away"]].split()[-1]
    return g


# ------------------------------------------------------------ season odds --
#
# The Monte Carlo (10,000 seasons) never runs inside a request: it is computed
# in a background thread whenever the data changes, cached in memory and on
# disk (data/cache/), and pages read the cache. While a new run is computing,
# pages keep showing the previous result.

log = logging.getLogger(__name__)
ODDS_CACHE_DIR = WRITABLE_DIR / "cache"

_ANTI_COLS = [MODEL_FEATURES.index(c) for c in ANTISYMMETRIC]
_NEUTRAL_COL = MODEL_FEATURES.index("neutral_site")


def home_win_prob_array(X: np.ndarray) -> np.ndarray:
    """home_win_prob for a plain array in MODEL_FEATURES order (no pandas overhead;
    the simulator calls it ~150 times per run)."""
    def raw(M):
        z = ((M - _MEAN) / _SCALE) @ _COEF + _INTERCEPT
        return 1 / (1 + np.exp(-z))

    X = np.asarray(X, float)
    p = raw(X)
    neutral = X[:, _NEUTRAL_COL] == 1
    if neutral.any():
        swapped = X[neutral].copy()
        swapped[:, _ANTI_COLS] = -swapped[:, _ANTI_COLS]
        p[neutral] = (p[neutral] + 1 - raw(swapped)) / 2
    return p


def _fingerprint(snap: Snapshot) -> str:
    """Changes whenever a result, the schedule, the model or the engine changes."""
    import simulate
    import ui

    f = snap.features[snap.features.season == CURRENT_SEASON]
    played = f[f.played][["game_id", "home_score", "away_score"]].astype(str).agg("|".join, axis=1)
    parts = [ui.model_version(), simulate.ENGINE_VERSION, str(simulate.N_SIMS), str(simulate.SEED),
             *sorted(played), *sorted(f[~f.played].game_id)]
    return hashlib.sha1("\n".join(parts).encode()).hexdigest()[:12]


def compute_season_odds(snap: Snapshot | None = None) -> pd.DataFrame:
    """Run the Monte Carlo now (blocking). Used by the background thread and the CLI."""
    import simulate

    snap = snap or snapshot()
    inp, record = simulate.inputs_from_snapshot(snap, home_win_prob_array)
    odds = simulate.monte_carlo(inp, record)
    odds.attrs["bracket"] = simulate.average_bracket(odds)
    return odds


_odds_lock = threading.Lock()
_compute_lock = threading.Lock()
_odds: dict = {"fp": None, "odds": None, "computing": None}


def _odds_path(fp: str) -> Path:
    return ODDS_CACHE_DIR / f"season_odds_{fp}.joblib"


def _compute_in_background(snap: Snapshot, fp: str) -> None:
    try:
        t0 = time.time()
        odds = compute_season_odds(snap)
        ODDS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        tmp = _odds_path(fp).with_suffix(".tmp")
        joblib.dump((odds, dict(odds.attrs)), tmp)
        tmp.replace(_odds_path(fp))
        log.info("season odds %s computed in %.1fs", fp, time.time() - t0)
        with _odds_lock:
            _odds.update(fp=fp, odds=odds)
    except Exception:
        log.exception("season odds computation failed")
    finally:
        with _odds_lock:
            _odds["computing"] = None


def season_odds() -> pd.DataFrame | None:
    """Latest Monte Carlo odds.

    Returns the cached result for the current data if it exists (memory, then
    disk). Otherwise, on a long-running server, starts a background
    computation and returns the previous result (None on a cold start). On
    Vercel it computes inside the request instead, once per instance per data
    version; concurrent requests wait for that single computation.
    """
    snap = snapshot()
    fp = _fingerprint(snap)
    with _odds_lock:
        if _odds["fp"] == fp:
            return _odds["odds"]
        path = _odds_path(fp)
        if path.exists():
            odds, attrs = joblib.load(path)
            odds.attrs.update(attrs)
            _odds.update(fp=fp, odds=odds)
            return odds
        if ON_VERCEL:
            # Serverless: work started after the response may be frozen, so compute
            # now (a few seconds, once per instance per data version) and cache it.
            _odds["computing"] = fp
        elif _odds["computing"] != fp:
            _odds["computing"] = fp
            threading.Thread(target=_compute_in_background, args=(snap, fp), daemon=True).start()
            return _odds["odds"]  # previous (stale) result, or None on a cold start
        else:
            return _odds["odds"]
    with _compute_lock:  # Vercel: one computation; other requests wait, then reuse it
        with _odds_lock:
            if _odds["fp"] == fp:
                return _odds["odds"]
        _compute_in_background(snap, fp)
    with _odds_lock:
        return _odds["odds"]


def season_odds_ready() -> bool:
    with _odds_lock:
        return _odds["fp"] is not None and _odds["computing"] is None


@lru_cache(maxsize=64)
def _single(fp: str, seed: int, chalk: bool) -> dict:
    import simulate

    inp, _ = simulate.inputs_from_snapshot(snapshot(), home_win_prob_array)
    return simulate.single_season(inp, seed, chalk)


def single_season(seed: int, chalk: bool = False) -> dict:
    """One simulated season (cheap: a single run). Cached per data version and seed."""
    return _single(_fingerprint(snapshot()), seed, chalk)
