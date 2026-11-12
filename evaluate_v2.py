"""Evaluate v2 (v1 + qb_rating_diff) against v1.

    python evaluate_v2.py --rolling    # rolling origin 2019-2024 -> reports/v2_rolling.json
    python evaluate_v2.py --holdout    # ONCE: 2026 so far + 2025 ("previously seen") -> reports/v2_holdout.json

Rolling origin: for Y in 2019..2024, refit v1 and v2 on regular-season games
2009..Y-1 and score every non-tie game of Y. Paired bootstrap over games (and over
whole season-weeks as a check) gives a 95% CI of the per-game loss difference.

Holdout: the frozen models as saved (models/pipeline.joblib = v1, models/v2/ = v2),
no refitting. 2025 was v1's test season and has been looked at, so it is reported
as "previously seen"; 2026 games played so far are the clean holdout. Runs once:
a marker file stops a second run.

"Mismatch" games: the point-in-time expected starter was not the QB who actually
started (schedules qb_id, used here only for this diagnostic, never as a feature).
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd

import market
from data.fetch import CURRENT_SEASON, TEST_SEASON, load_schedules
from features import MODEL_FEATURES
from paths import MODEL_PATH, REPORTS
from train import TRAIN_SEASONS, make_pipeline, model_rows
from train_v2 import V2_DIR, V2_FEATURES, v2_frame

ROLL_YEARS = range(2019, 2025)
ROLL_START = 2009
HOLDOUT_MARKER = REPORTS / ".v2_holdout_done"
N_BOOT = 10_000


def logloss_i(y, p):
    p = np.clip(np.asarray(p, float), 1e-15, 1 - 1e-15)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def brier_i(y, p):
    return (np.asarray(p, float) - y) ** 2


def accuracy(y, p):
    p = np.asarray(p, float)
    return float(np.mean(np.where(p == 0.5, 0.5, (p > 0.5) == (y == 1))))


def paired_ci(diff: np.ndarray, groups: np.ndarray | None = None, seed: int = 42) -> dict:
    """95% percentile CI of mean(diff): over games, and over groups (e.g. season-week) if given."""
    rng = np.random.default_rng(seed)
    n = len(diff)
    boots = np.array([diff[rng.integers(0, n, n)].mean() for _ in range(N_BOOT)])
    out = {"mean": float(diff.mean()), "ci95": [float(x) for x in np.percentile(boots, [2.5, 97.5])], "n": int(n)}
    if groups is not None:
        idx = pd.Series(range(n)).groupby(groups).indices
        keys = list(idx)
        cb = [diff[np.concatenate([idx[keys[k]] for k in rng.integers(0, len(keys), len(keys))])].mean()
              for _ in range(2000)]
        out["ci95_week_cluster"] = [float(x) for x in np.percentile(cb, [2.5, 97.5])]
    out["excludes_0"] = bool(out["ci95"][1] < 0 or out["ci95"][0] > 0)
    return out


def summary(y, preds: dict) -> dict:
    return {k: {"log_loss": float(logloss_i(y, p).mean()), "brier": float(brier_i(y, p).mean()),
                "accuracy": accuracy(y, p)} for k, p in preds.items()}


def mismatch_block(games: pd.DataFrame, sched: pd.DataFrame) -> dict:
    """Expected starter != actual starter: counts and losses on just those games."""
    act = sched.set_index("game_id")[["home_qb_id", "away_qb_id"]]
    g = games.join(act, on="game_id")
    mis = (g.home_expected != g.home_qb_id) | (g.away_expected != g.away_qb_id)
    team_mis = int((g.home_expected != g.home_qb_id).sum() + (g.away_expected != g.away_qb_id).sum())
    y = g.home_win.to_numpy(float)
    out = {"games": int(len(g)), "games_with_mismatch": int(mis.sum()),
           "team_games_mismatched": team_mis, "team_games": int(2 * len(g))}
    for name, m in (("mismatch", mis.to_numpy()), ("match", ~mis.to_numpy())):
        if m.any():
            out[name] = summary(y[m], {k: g[k].to_numpy()[m] for k in ("v1", "v2", "vegas")})
    if mis.any():
        d = logloss_i(y[mis.to_numpy()], g.v2.to_numpy()[mis.to_numpy()]) - logloss_i(y[mis.to_numpy()], g.v1.to_numpy()[mis.to_numpy()])
        out["mismatch_v2_minus_v1_log_loss"] = paired_ci(d)
    return out


def rolling() -> dict:
    df = v2_frame(max(ROLL_YEARS))
    sched = load_schedules(range(ROLL_START, max(ROLL_YEARS) + 1))
    parts = []
    for Y in ROLL_YEARS:
        tr = model_rows(df, range(ROLL_START, Y), regular_only=True)
        te = model_rows(df, [Y], regular_only=False).copy()
        assert tr[V2_FEATURES].notna().all().all() and te[V2_FEATURES].notna().all().all()
        te["vegas"] = market.vegas_prob(te, market.fit_spread_sigma(df, range(ROLL_START, Y)))
        te["v1"] = make_pipeline().fit(tr[MODEL_FEATURES], tr.home_win).predict_proba(te[MODEL_FEATURES])[:, 1]
        te["v2"] = make_pipeline().fit(tr[V2_FEATURES], tr.home_win).predict_proba(te[V2_FEATURES])[:, 1]
        parts.append(te)
    P = pd.concat(parts, ignore_index=True)
    y = P.home_win.to_numpy(float)
    per_season = {int(s): summary(g.home_win.to_numpy(float), {k: g[k] for k in ("v1", "v2", "vegas")})
                  for s, g in P.groupby("season")}
    beats_vegas = {s: m["v2"]["log_loss"] < m["vegas"]["log_loss"] or m["v1"]["log_loss"] < m["vegas"]["log_loss"]
                   for s, m in per_season.items()}
    weeks = (P.season.astype(str) + "-" + P.week.astype(str)).to_numpy()
    return {
        "setup": f"train regular season {ROLL_START}..Y-1, score all non-tie games of Y, Y = {ROLL_YEARS.start}-{ROLL_YEARS.stop - 1}",
        "games": int(len(P)),
        "pooled": summary(y, {k: P[k] for k in ("v1", "v2", "vegas")}),
        "per_season": per_season,
        "any_season_model_beats_vegas": any(beats_vegas.values()),
        "v2_minus_v1": {
            "log_loss": paired_ci(logloss_i(y, P.v2) - logloss_i(y, P.v1), weeks),
            "brier": paired_ci(brier_i(y, P.v2) - brier_i(y, P.v1), weeks),
        },
        "mismatch": mismatch_block(P, sched),
    }


def holdout() -> dict:
    if HOLDOUT_MARKER.exists():
        sys.exit(f"The v2 holdout already ran ({HOLDOUT_MARKER.read_text().strip()}). It runs once.")
    v1 = joblib.load(MODEL_PATH)
    v2 = joblib.load(V2_DIR / "pipeline.joblib")
    df = v2_frame(CURRENT_SEASON, allow_test=True)
    sched = load_schedules(range(TEST_SEASON, CURRENT_SEASON + 1), allow_test=True)
    sigma = market.fit_spread_sigma(df, TRAIN_SEASONS)
    out = {"v1_model": "models/pipeline.joblib (trained 2006-2023)", "v2_model": "models/v2/pipeline.joblib (trained 2009-2024)",
           "note": "frozen pipelines, no neutral-site averaging (same as the v1 test evaluation)"}
    preds = []
    for season, label in ((CURRENT_SEASON, "holdout"), (TEST_SEASON, "previously seen")):
        g = model_rows(df, [season], regular_only=False).copy()
        assert g[V2_FEATURES].notna().all().all()
        g["v1"] = v1.predict_proba(g[MODEL_FEATURES])[:, 1]
        g["v2"] = v2.predict_proba(g[V2_FEATURES])[:, 1]
        g["vegas"] = market.vegas_prob(g, sigma)
        g = g[g.vegas.notna()] if g.vegas.isna().any() else g
        y = g.home_win.to_numpy(float)
        weeks = g.week.to_numpy()
        out[str(season)] = {
            "label": label, "games": int(len(g)), "weeks": [int(g.week.min()), int(g.week.max())],
            "metrics": summary(y, {k: g[k] for k in ("v1", "v2", "vegas")}),
            "v2_minus_v1": {"log_loss": paired_ci(logloss_i(y, g.v2) - logloss_i(y, g.v1), weeks),
                            "brier": paired_ci(brier_i(y, g.v2) - brier_i(y, g.v1), weeks)},
            "v2_beats_vegas_log_loss": bool(logloss_i(y, g.v2).mean() < logloss_i(y, g.vegas).mean()),
            "mismatch": mismatch_block(g, sched),
        }
        preds.append(g[["game_id", "season", "week", "home_win", "v1", "v2", "vegas", "qb_rating_diff"]])
    pd.concat(preds).to_csv(REPORTS / "predictions_v2_holdout.csv", index=False)
    HOLDOUT_MARKER.write_text(f"ran {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--rolling", action="store_true")
    g.add_argument("--holdout", action="store_true")
    a = ap.parse_args()
    name, res = ("v2_rolling.json", rolling()) if a.rolling else ("v2_holdout.json", holdout())
    (REPORTS / name).write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
