import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.fetch import load_schedules, load_team_stats  # noqa: E402

# A few dev seasons are enough to exercise season boundaries, playoffs and
# neutral sites while keeping the tests fast. Never includes the test season.
SEASONS = range(2018, 2022)


@pytest.fixture(scope="session")
def schedules():
    return load_schedules(SEASONS)


@pytest.fixture(scope="session")
def team_stats():
    return load_team_stats(SEASONS)
