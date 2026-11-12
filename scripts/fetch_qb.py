"""Download the nflverse data behind the v2 QB rating (dev machine / scheduled CI job only).

    python scripts/fetch_qb.py                 # current season only (what the cron job runs)
    python scripts/fetch_qb.py --all           # every season 2002..current (first setup)

Writes small QB-only subsets to data/qb_raw/ (committed, so training, evaluation
and the leakage test run offline):
    pstats_{season}.parquet  QB player stats per game (plays, EPA)        2002+
    inj_{season}.parquet     QB rows of the final weekly injury report     2009+
    depth_{season}.parquet   QB depth chart (weekly to 2024, timestamped snapshots 2025+)

For the current season it also saves a timestamped snapshot of the latest week's
injury report, all positions, to data/injury_snapshots/{season}/. nflverse gives
2025+ reports without a timestamp; these snapshots record when we saw each report,
so later games can be checked against kickoff. A snapshot is written only when the
report changed since the previous one.

The web app never runs this: it only reads files built from it (scripts/build_qb.py).
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from datetime import datetime, timezone
from pathlib import Path

import nflreadpy as nr
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from data.fetch import BURN_IN_START, CURRENT_SEASON  # noqa: E402

RAW = ROOT / "data" / "qb_raw"
SNAP = ROOT / "data" / "injury_snapshots"
INJURY_FIRST = 2009
DEPTH_FIRST = 2009

PSTATS_COLS = ["game_id", "season", "week", "season_type", "team", "player_id", "player_display_name", "position",
               "attempts", "sacks_suffered", "carries", "passing_epa", "rushing_epa"]
INJ_COLS = ["season", "game_type", "week", "team", "gsis_id", "full_name", "position", "report_status", "date_modified"]
DEPTH_W_COLS = ["season", "week", "game_type", "club_code", "gsis_id", "full_name", "position", "formation", "depth_team"]
DEPTH_S_COLS = ["dt", "team", "gsis_id", "player_name", "pos_abb", "pos_rank"]


def _pd(df) -> pd.DataFrame:
    return df.to_pandas()


def fetch_season(y: int) -> None:
    ps = _pd(nr.load_player_stats(y))
    ps = ps[ps.position == "QB"][PSTATS_COLS]
    ps.to_parquet(RAW / f"pstats_{y}.parquet", index=False)
    if y >= INJURY_FIRST:
        inj = _pd(nr.load_injuries(y))
        inj = inj[inj.position == "QB"][[c for c in INJ_COLS if c in inj]]
        inj.to_parquet(RAW / f"inj_{y}.parquet", index=False)
    if y >= DEPTH_FIRST:
        d = _pd(nr.load_depth_charts(y))
        if "club_code" in d:
            d = d[d.position == "QB"][DEPTH_W_COLS]
        else:
            d = d[d.pos_abb == "QB"][DEPTH_S_COLS]
        d.to_parquet(RAW / f"depth_{y}.parquet", index=False)
    print(f"{y}: pstats {len(ps)} rows")


def snapshot_injuries(y: int) -> Path | None:
    """Latest week's full injury report (all positions), stamped with the fetch time."""
    inj = _pd(nr.load_injuries(y))
    week = int(inj.week.max())
    rep = inj[inj.week == week].sort_values(["team", "gsis_id"]).reset_index(drop=True)
    digest = hashlib.sha256(pd.util.hash_pandas_object(rep, index=False).values.tobytes()).hexdigest()[:16]
    folder = SNAP / str(y)
    folder.mkdir(parents=True, exist_ok=True)
    if any(p.stem.endswith(digest) for p in folder.glob(f"week{week:02d}_*.parquet")):
        print(f"injury report {y} week {week}: unchanged")
        return None
    now = datetime.now(timezone.utc)
    rep["fetched_at"] = now
    path = folder / f"week{week:02d}_{now:%Y%m%dT%H%M%SZ}_{digest}.parquet"
    rep.to_parquet(path, index=False)
    print(f"injury report {y} week {week}: saved {path.relative_to(ROOT)}")
    return path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help=f"fetch {BURN_IN_START}..{CURRENT_SEASON}")
    ap.add_argument("--no-snapshot", action="store_true")
    a = ap.parse_args()
    RAW.mkdir(parents=True, exist_ok=True)
    for y in (range(BURN_IN_START, CURRENT_SEASON + 1) if a.all else [CURRENT_SEASON]):
        fetch_season(y)
    if not a.no_snapshot:
        snapshot_injuries(CURRENT_SEASON)


if __name__ == "__main__":
    main()
