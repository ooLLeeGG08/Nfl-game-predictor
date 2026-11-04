"""Download NFLverse data and cache it locally as parquet.

Design notes
------------
* One parquet file per (dataset, season). Past seasons never change, so once
  cached they are never re-downloaded unless ``refresh=True`` (``--refresh``).
* The in-progress season *does* change week to week, so its cache file gets a
  time-to-live (TTL). This matters on Render, where the disk is wiped on every
  deploy/restart: the app simply re-fetches the current season when needed.
* nflreadpy returns Polars DataFrames. We convert to pandas right here, at the
  load boundary, so the rest of the project only ever sees pandas.
* Test-season guard: the 2025 test season must not be touched while we are
  developing. Loading it requires an explicit ``allow_test=True``.

Run ``python -m data.fetch --help`` to pre-populate the cache.
"""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).resolve().parent / "raw"

FIRST_SEASON = 2006       # first season used for training
BURN_IN_START = 2002      # extra seasons used only to warm up Elo / prior stats
VALID_SEASON = 2024
TEST_SEASON = 2025
CURRENT_SEASON = 2026
CURRENT_TTL_SECONDS = 6 * 60 * 60

# Old team codes -> current franchise codes, so Elo follows a franchise
# through relocations (Rams STL->LA, Chargers SD->LAC, Raiders OAK->LV).
TEAM_MAP = {"STL": "LA", "SD": "LAC", "OAK": "LV"}

TURNOVER_SOURCE = "nflreadpy.load_team_stats(summary_level='week')"


class HeldOutSeasonError(RuntimeError):
    """Raised when code tries to read the held-out test season during dev."""


def _guard(seasons, allow_test: bool) -> list[int]:
    seasons = sorted(set(int(s) for s in seasons))
    if TEST_SEASON in seasons and not allow_test:
        raise HeldOutSeasonError(
            f"Season {TEST_SEASON} is the held-out test season. "
            "Pass allow_test=True only for the single final evaluation."
        )
    return seasons


def _is_fresh(path: Path, season: int) -> bool:
    if not path.exists():
        return False
    if season < CURRENT_SEASON:
        return True  # historical data is immutable
    return (time.time() - path.stat().st_mtime) < CURRENT_TTL_SECONDS


def _download(kind: str, season: int) -> pd.DataFrame:
    """Fetch one season from nflreadpy, falling back to nfl_data_py."""
    try:
        import nflreadpy as nfl

        if kind == "schedules":
            df = nfl.load_schedules([season])
        else:
            df = nfl.load_team_stats([season], summary_level="week")
        return df.to_pandas()
    except Exception as exc:  # network error, API change, missing package...
        log.warning("nflreadpy failed for %s %s (%s); trying nfl_data_py", kind, season, exc)
        if kind != "schedules":
            # nfl_data_py has no team-level weekly stats; turnovers would have
            # to be derived from play-by-play. Not needed so far, so fail loudly.
            raise
        try:
            import nfl_data_py as ndp  # optional fallback dependency, not installed by default
        except ImportError:
            raise exc from None
        return ndp.import_schedules([season])


def _load(kind: str, seasons, refresh: bool, allow_test: bool) -> pd.DataFrame:
    seasons = _guard(seasons, allow_test)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    frames = []
    for season in seasons:
        path = CACHE_DIR / f"{kind}_{season}.parquet"
        if refresh or not _is_fresh(path, season):
            log.info("downloading %s %s", kind, season)
            try:
                df = _download(kind, season)
            except Exception as exc:
                if not path.exists():
                    raise
                # Serving slightly stale data beats crashing; the next TTL check retries.
                log.warning("refresh of %s %s failed (%s); using cached copy", kind, season, exc)
            else:
                # Write then rename: a crash mid-write can never leave a half-written cache file.
                tmp = path.with_suffix(".tmp")
                df.to_parquet(tmp, index=False)
                tmp.replace(path)
        frames.append(pd.read_parquet(path))
    return pd.concat(frames, ignore_index=True)


def load_schedules(seasons, refresh: bool = False, allow_test: bool = False) -> pd.DataFrame:
    df = _load("schedules", seasons, refresh, allow_test)
    for col in ("home_team", "away_team"):
        df[col] = df[col].replace(TEAM_MAP)
    df["gameday"] = pd.to_datetime(df["gameday"])
    return df


def load_team_stats(seasons, refresh: bool = False, allow_test: bool = False) -> pd.DataFrame:
    df = _load("team_stats", seasons, refresh, allow_test)
    for col in ("team", "opponent_team"):
        df[col] = df[col].replace(TEAM_MAP)
    return df


def main() -> None:
    p = argparse.ArgumentParser(description="Fetch and cache NFLverse data.")
    p.add_argument("--start", type=int, default=BURN_IN_START)
    p.add_argument("--end", type=int, default=VALID_SEASON)
    p.add_argument("--refresh", action="store_true", help="re-download even if cached")
    p.add_argument("--allow-test", action="store_true", help=f"permit loading {TEST_SEASON}")
    a = p.parse_args()
    seasons = range(a.start, a.end + 1)
    s = load_schedules(seasons, a.refresh, a.allow_test)
    t = load_team_stats(seasons, a.refresh, a.allow_test)
    print(f"schedules: {len(s)} rows, team_stats: {len(t)} rows, cached in {CACHE_DIR}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
