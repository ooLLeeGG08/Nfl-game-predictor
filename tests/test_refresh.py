"""Data-refresh rules (no network): what gets re-downloaded, and which week is 'complete'."""
import os
import time

import numpy as np
import pandas as pd

from data.fetch import CURRENT_SEASON, CURRENT_TTL_SECONDS, _is_fresh
from live import last_completed_week


def _touch(path, age_seconds):
    path.write_bytes(b"x")
    t = time.time() - age_seconds
    os.utime(path, (t, t))


def test_past_seasons_are_never_refetched(tmp_path):
    p = tmp_path / "old.parquet"
    _touch(p, age_seconds=365 * 24 * 3600)
    assert _is_fresh(p, CURRENT_SEASON - 1)


def test_current_season_expires_after_ttl(tmp_path):
    p = tmp_path / "now.parquet"
    _touch(p, age_seconds=CURRENT_TTL_SECONDS - 60)
    assert _is_fresh(p, CURRENT_SEASON)
    _touch(p, age_seconds=CURRENT_TTL_SECONDS + 60)
    assert not _is_fresh(p, CURRENT_SEASON)


def test_missing_cache_is_never_fresh(tmp_path):
    assert not _is_fresh(tmp_path / "missing.parquet", CURRENT_SEASON - 1)


def test_last_completed_week_ignores_partially_played_weeks():
    sched = pd.DataFrame({
        "week": [1, 1, 2, 2, 3],
        "home_score": [20, 17, 24, np.nan, np.nan],  # week 2 still has a game to play
    })
    assert last_completed_week(sched) == 1
    assert last_completed_week(sched.assign(home_score=np.nan)) == 0


def test_failed_refresh_falls_back_to_stale_cache(tmp_path, monkeypatch):
    import data.fetch as fetch

    monkeypatch.setattr(fetch, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(fetch, "RUNTIME_DIR", tmp_path)
    stale = pd.DataFrame({"game_id": ["g1"], "home_team": ["KC"], "away_team": ["BUF"], "gameday": ["2026-09-10"]})
    path = tmp_path / f"schedules_{CURRENT_SEASON}.parquet"
    stale.to_parquet(path, index=False)
    _age = time.time() - CURRENT_TTL_SECONDS - 60
    os.utime(path, (_age, _age))

    def offline(kind, season):
        raise ConnectionError("network down")

    monkeypatch.setattr(fetch, "_download", offline)
    df = fetch.load_schedules([CURRENT_SEASON])
    assert list(df.game_id) == ["g1"]


def test_failed_download_without_cache_raises(tmp_path, monkeypatch):
    # No cached copy and no committed seed: nothing to fall back on, so the error surfaces.
    import data.fetch as fetch
    import pytest

    monkeypatch.setattr(fetch, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(fetch, "RUNTIME_DIR", tmp_path)
    monkeypatch.setattr(fetch, "SEED_DIR", tmp_path / "no-seed")
    monkeypatch.setattr(fetch, "_download", lambda kind, season: (_ for _ in ()).throw(ConnectionError("down")))
    with pytest.raises(ConnectionError):
        fetch.load_schedules([CURRENT_SEASON])


def test_empty_cache_and_no_network_falls_back_to_committed_seed(tmp_path, monkeypatch):
    # A fresh serverless instance (empty /tmp) while GitHub is unreachable.
    import data.fetch as fetch

    seed_dir = tmp_path / "seed"
    seed_dir.mkdir()
    pd.DataFrame({"game_id": ["s1"], "home_team": ["KC"], "away_team": ["BUF"], "gameday": ["2026-09-10"]}) \
        .to_parquet(seed_dir / f"schedules_{CURRENT_SEASON}.parquet", index=False)
    monkeypatch.setattr(fetch, "RUNTIME_DIR", tmp_path / "empty")
    monkeypatch.setattr(fetch, "SEED_DIR", seed_dir)
    monkeypatch.setattr(fetch, "_download", lambda kind, season: (_ for _ in ()).throw(ConnectionError("down")))
    assert list(fetch.load_schedules([CURRENT_SEASON]).game_id) == ["s1"]


def test_download_retries_transient_errors(monkeypatch):
    import urllib.request

    import data.fetch as fetch

    calls = {"n": 0}
    good = __import__("io").BytesIO()
    pd.DataFrame({"season": [CURRENT_SEASON], "game_id": ["g"]}).to_parquet(good, index=False)

    class Resp:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self): return good.getvalue()

    def flaky(req, timeout):
        calls["n"] += 1
        if calls["n"] < 3:
            raise ConnectionResetError(54, "Connection reset by peer")
        return Resp()

    monkeypatch.setattr(urllib.request, "urlopen", flaky)
    monkeypatch.setattr(fetch.time, "sleep", lambda s: None)
    assert list(fetch._download("schedules", CURRENT_SEASON).game_id) == ["g"] and calls["n"] == 3
