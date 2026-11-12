"""Shadow-mode v2 vs v1 on finished games from the pre-kickoff log.

    python scripts/shadow_report.py        -> prints the report, writes reports/v2_shadow.json

Uses exactly the pre-registered selection (README, "Pre-registered evaluation"):
for every game in data/qb/v2_log.csv with fresh QB data, its LAST row logged
before kickoff; finished, non-tie games only. Scores are per game, from the
logged probabilities (nothing is recomputed).

During the season this is for watching only: the decision is made once, at the
end of 2026, on the full set (paired-bootstrap 95% CI of v2 - v1 log loss
entirely below 0 -> consider promoting v2). Nothing on the site changes.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.fetch import CURRENT_SEASON, load_schedules  # noqa: E402
from evaluate_v2 import accuracy, brier_i, logloss_i, paired_ci  # noqa: E402

LOG = ROOT / "data" / "qb" / "v2_log.csv"
OUT = ROOT / "reports" / "v2_shadow.json"


def select(log: pd.DataFrame) -> pd.DataFrame:
    """Last row logged before kickoff, per game, among rows with fresh QB data."""
    log = log.copy()
    log["logged_at"] = pd.to_datetime(log.logged_at, utc=True)
    log["kickoff"] = pd.to_datetime(log.kickoff, utc=True)
    avail = log.available.astype(str).str.lower() == "true"
    pre = log[avail & (log.logged_at < log.kickoff)]
    return pre.sort_values("logged_at").drop_duplicates("game_id", keep="last")


def score(rows: pd.DataFrame, results: pd.DataFrame) -> dict:
    """rows: selected log rows; results: game_id, home_score, away_score (finished games)."""
    g = rows.merge(results, on="game_id")
    g = g[g.home_score.notna() & (g.home_score != g.away_score)]
    out = {"games_logged": int(len(rows)), "games_scored": int(len(g))}
    if g.empty:
        return out
    y = (g.home_score > g.away_score).to_numpy(float)
    p1, p2 = g.p_v1.to_numpy(float), g.p_v2.to_numpy(float)
    ll1, ll2 = -logloss_i(y, p1), -logloss_i(y, p2)      # per-game log-likelihood
    out.update({
        "log_likelihood": {"v1": float(ll1.sum()), "v2": float(ll2.sum()), "v2_minus_v1": float((ll2 - ll1).sum())},
        "log_loss": {"v1": float(-ll1.mean()), "v2": float(-ll2.mean())},
        "brier": {"v1": float(brier_i(y, p1).mean()), "v2": float(brier_i(y, p2).mean())},
        "accuracy": {"v1": accuracy(y, p1), "v2": accuracy(y, p2)},
        "v2_minus_v1_log_loss": paired_ci(logloss_i(y, p2) - logloss_i(y, p1)) if len(g) >= 2 else None,
        "first_kickoff": g.kickoff.min().isoformat(), "last_kickoff": g.kickoff.max().isoformat(),
    })
    return out


def main() -> None:
    if not LOG.exists():
        sys.exit(f"{LOG.relative_to(ROOT)} not found: run scripts/build_qb.py first")
    rows = select(pd.read_csv(LOG))
    sched = load_schedules([CURRENT_SEASON])
    rep = {"generated_at": datetime.now(timezone.utc).isoformat(), "season": CURRENT_SEASON,
           "status": "interim (watching only; the decision is made at the end of the season)",
           **score(rows, sched[["game_id", "home_score", "away_score"]])}
    OUT.write_text(json.dumps(rep, indent=2))
    print(f"shadow v2 vs v1: {rep['games_scored']} finished of {rep['games_logged']} logged games")
    if rep["games_scored"]:
        ll, ci = rep["log_likelihood"], rep["v2_minus_v1_log_loss"]
        print(f"  log-likelihood  v1 {ll['v1']:.3f}  v2 {ll['v2']:.3f}  (v2 - v1 {ll['v2_minus_v1']:+.3f})")
        print(f"  log loss        v1 {rep['log_loss']['v1']:.4f}  v2 {rep['log_loss']['v2']:.4f}")
        if ci:
            print(f"  v2 - v1 log loss {ci['mean']:+.4f}  95% CI [{ci['ci95'][0]:+.4f}, {ci['ci95'][1]:+.4f}]")


if __name__ == "__main__":
    main()
