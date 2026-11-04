"""Feature engineering: Elo ratings + rolling last-5-game stats.

Every feature describes what was known *before kickoff*. We walk through games
in chronological order and, for each game:

    1. read each team's current state  -> these are the game's features
    2. if the game has been played, update the state with its result

Because step 1 always happens before step 2, a game can never see its own
result or any later game. Unplayed (upcoming) games still get features in
step 1; they just skip step 2. The leakage tests in tests/ verify this.

Run ``python features.py`` to build data/features.parquet.
"""
from __future__ import annotations

import logging
import math
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from data.fetch import BURN_IN_START, TURNOVER_SOURCE, VALID_SEASON, load_schedules, load_team_stats

log = logging.getLogger(__name__)

FEATURES_PATH = Path(__file__).resolve().parent / "data" / "features.parquet"

WINDOW = 5
ROLLING_STATS = ("pf", "pa", "pd", "to")  # points for/against, point diff, turnover margin
# Prior used before a team has any history (only matters in the 2002 burn-in).
NO_HISTORY_PRIOR = {"pf": 21.0, "pa": 21.0, "pd": 0.0, "to": 0.0}

# Columns fed to the model (all home-minus-away, plus game-context flags).
# pf_diff and pa_diff are stored but not used: pd_diff = pf_diff - pa_diff
# exactly, so using all three would split one effect across collinear
# coefficients and make feature importance meaningless.
MODEL_FEATURES = [
    "elo_diff",
    "pd_diff",
    "to_diff",
    "rest_diff",
    "div_game",
    "neutral_site",
]


@dataclass(frozen=True)
class EloParams:
    k: float = 20.0           # how far one game moves a rating
    hfa: float = 48.0         # home-field advantage in Elo points (used only inside the Elo update)
    mov: bool = True          # scale updates by margin of victory
    regress: float = 1 / 3    # fraction pulled back to the mean between seasons (0 = full carryover, 1 = reset)
    mean: float = 1500.0


@dataclass
class TeamState:
    elo: float
    season: int | None = None
    recent: deque = field(default_factory=lambda: deque(maxlen=WINDOW))  # current-season games only
    prior: dict = field(default_factory=lambda: dict(NO_HISTORY_PRIOR))   # last season's final last-5 values

    def rolling(self) -> dict:
        """Last-5 averages, blended with last season's final values early in a season.

        With n games played this season (n < 5): w = n/5,
        value = w * mean(this season's games) + (1 - w) * prior season final.
        From game 6 on, w = 1 and this is a plain last-5 average.
        """
        out = {}
        for stat in ROLLING_STATS:
            vals = [g[stat] for g in self.recent if not math.isnan(g[stat])]
            w = len(vals) / WINDOW
            current = float(np.mean(vals)) if vals else 0.0
            out[stat] = w * current + (1 - w) * self.prior[stat]
        out["games_played"] = len(self.recent)
        return out

    def start_season(self, season: int, params: EloParams) -> None:
        if self.season is not None:
            self.prior = {s: v for s, v in self.rolling().items() if s in ROLLING_STATS}
            self.elo = params.mean + (1 - params.regress) * (self.elo - params.mean)
        self.recent.clear()
        self.season = season


def elo_expected(elo_home: float, elo_away: float, hfa: float) -> float:
    """Probability the home team wins according to Elo alone."""
    return 1.0 / (1.0 + 10 ** (-(elo_home - elo_away + hfa) / 400))


def elo_update(elo_home, elo_away, home_score, away_score, neutral, params: EloParams):
    """Return new (elo_home, elo_away) after a game. Zero-sum: points won = points lost."""
    hfa = 0.0 if neutral else params.hfa
    expected = elo_expected(elo_home, elo_away, hfa)
    margin = home_score - away_score
    actual = 1.0 if margin > 0 else 0.0 if margin < 0 else 0.5

    mult = 1.0
    if params.mov and margin != 0:
        # FiveThirtyEight-style multiplier: bigger wins move ratings more, but
        # the bonus shrinks when the favourite wins (avoids rating inflation).
        winner_edge = (elo_home - elo_away + hfa) * (1 if margin > 0 else -1)
        mult = math.log(abs(margin) + 1) * 2.2 / (winner_edge * 0.001 + 2.2)

    delta = params.k * mult * (actual - expected)
    return elo_home + delta, elo_away - delta


def turnover_margins(team_stats: pd.DataFrame) -> dict[tuple[str, str], float]:
    """(game_id, team) -> takeaways minus giveaways.

    Giveaways = interceptions thrown + all fumbles lost. Takeaways are defined
    as the *opponent's* giveaways in the same game, so the two margins in a
    game always sum to zero.
    """
    ts = team_stats[["game_id", "team", "opponent_team", "passing_interceptions", "fumbles_lost_total"]].copy()
    ts["give"] = ts["passing_interceptions"].fillna(0) + ts["fumbles_lost_total"].fillna(0)
    opp = ts[["game_id", "team", "give"]].rename(columns={"team": "opponent_team", "give": "take"})
    ts = ts.merge(opp, on=["game_id", "opponent_team"], how="left")
    ts["margin"] = ts["take"] - ts["give"]
    return {(g, t): m for g, t, m in zip(ts["game_id"], ts["team"], ts["margin"])}


def build_features(
    schedules: pd.DataFrame,
    team_stats: pd.DataFrame,
    params: EloParams = EloParams(),
    return_state: bool = False,
):
    """Compute pre-kickoff features for every game in ``schedules``.

    Returns one row per game. With ``return_state=True`` also returns the
    final per-team state (used by the app for "current" ratings).
    """
    games = schedules.sort_values(["gameday", "gametime", "game_id"], na_position="last")
    to_margin = turnover_margins(team_stats)
    teams: dict[str, TeamState] = {}
    rows = []

    for g in games.itertuples(index=False):
        home = teams.setdefault(g.home_team, TeamState(elo=params.mean))
        away = teams.setdefault(g.away_team, TeamState(elo=params.mean))
        for t in (home, away):
            if t.season != g.season:
                t.start_season(g.season, params)

        # ---- 1. features: read state only ----
        hr, ar = home.rolling(), away.rolling()
        neutral = int(g.location == "Neutral")
        played = not (pd.isna(g.home_score) or pd.isna(g.away_score))
        row = {
            "game_id": g.game_id,
            "season": g.season,
            "game_type": g.game_type,
            "week": g.week,
            "gameday": g.gameday,
            "home_team": g.home_team,
            "away_team": g.away_team,
            "home_score": g.home_score,
            "away_score": g.away_score,
            "played": played,
            "is_tie": played and g.home_score == g.away_score,
            "home_win": (float(g.home_score > g.away_score) if played else np.nan),
            "spread_line": g.spread_line,
            "home_moneyline": g.home_moneyline,
            "away_moneyline": g.away_moneyline,
            "neutral_site": neutral,
            "div_game": int(g.div_game),
            "home_rest": g.home_rest,
            "away_rest": g.away_rest,
            "rest_diff": g.home_rest - g.away_rest,
            "home_elo": home.elo,
            "away_elo": away.elo,
            "elo_diff": home.elo - away.elo,
            "home_games_played": hr["games_played"],
            "away_games_played": ar["games_played"],
        }
        for stat in ROLLING_STATS:
            row[f"home_{stat}5"] = hr[stat]
            row[f"away_{stat}5"] = ar[stat]
            row[f"{stat}_diff"] = hr[stat] - ar[stat]
        rows.append(row)

        # ---- 2. update state with the result (played games only) ----
        if not played:
            continue
        home.elo, away.elo = elo_update(home.elo, away.elo, g.home_score, g.away_score, neutral, params)
        hs, as_ = float(g.home_score), float(g.away_score)
        home.recent.append({"pf": hs, "pa": as_, "pd": hs - as_, "to": to_margin.get((g.game_id, g.home_team), np.nan)})
        away.recent.append({"pf": as_, "pa": hs, "pd": as_ - hs, "to": to_margin.get((g.game_id, g.away_team), np.nan)})

    df = pd.DataFrame(rows)
    return (df, teams) if return_state else df


def main() -> None:
    seasons = range(BURN_IN_START, VALID_SEASON + 1)  # test season deliberately excluded
    sched = load_schedules(seasons)
    stats = load_team_stats(seasons)
    df = build_features(sched, stats)

    ties = int(df["is_tie"].sum())
    missing_to = int(df[df.played][["home_to5", "away_to5"]].isna().any(axis=1).sum())
    log.info("built %d games, seasons %d-%d", len(df), df.season.min(), df.season.max())
    log.info("ties: %d (kept here for Elo/rolling state; excluded from training/evaluation)", ties)
    log.info("turnover source: %s; games with missing turnover features: %d", TURNOVER_SOURCE, missing_to)

    FEATURES_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(FEATURES_PATH, index=False)
    log.info("wrote %s", FEATURES_PATH)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
