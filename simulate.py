"""Season simulation engine: one random season, a "chalk" season, or 10,000 (Monte Carlo).

    python simulate.py            # Monte Carlo -> reports/season_odds_2026.csv (+ disk cache)

One engine serves all three modes (`simulate()`):

1. Played games keep their real results.
2. Remaining regular-season games are played in schedule order. For each one
   the frozen model gives P(home win) (neutral sites symmetrized). A final
   margin is drawn as  margin ~ Normal(sigma * Phi^-1(p), sigma),  so the home
   team wins (margin > 0) with probability exactly p. sigma is the spread
   scale fitted for the Vegas comparison (market.py). Margins are rounded and
   never 0: no simulated ties. "Chalk" mode uses the expected margin instead,
   so the favorite always wins.
3. After every simulated game, Elo and the last-5 stats update ("hot"
   simulation), exactly like features.build_features does for real games.
   Turnovers cannot be simulated, so simulated games add a turnover margin
   of 0 (its expected value).
4. Standings and seeding (simplified): division winners are seeds 1-4, the
   best three other teams seeds 5-7. Ties on record are broken by
   head-to-head among the tied teams, then division record, then conference
   record, then at random. (The official NFL procedure has more steps and
   conditions, e.g. head-to-head applies only to sweeps in multi-team ties.)
5. Playoffs: seed 1 has a bye (+7 rest days in its divisional game); wild card
   2v7, 3v6, 4v5; re-seed after each round so the top seed left plays the
   lowest; higher seed hosts; neutral Super Bowl. Playoff games update
   ratings too.

All simulations run at once as numpy arrays (in chunks, to bound memory),
so 10,000 seasons take seconds. The same seed always gives the same result.
"""
from __future__ import annotations

import copy
import logging
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from data.fetch import CURRENT_SEASON
from features import MODEL_FEATURES, EloParams
from mathx import best_assignment, norm_ppf
from teams import CONFERENCE_OF, DIVISIONS, NAMES

log = logging.getLogger(__name__)

N_SIMS = 10_000
SEED = 2026
CHUNK = 2_500            # sims per batch: bounds memory on Render's 512 MB
BYE_REST_DAYS = 7
WINDOW = 5
ENGINE_VERSION = "4-average-season"

TEAMS = sorted(NAMES)
IDX = {t: i for i, t in enumerate(TEAMS)}
DIV_NAMES = list(DIVISIONS)
DIV_MEMBERS = np.array([[IDX[t] for t in DIVISIONS[d]] for d in DIV_NAMES])  # 8 x 4
CONF_DIVS = {c: [k for k, d in enumerate(DIV_NAMES) if d.startswith(c)] for c in ("AFC", "NFC")}
CONF_MEMBERS = {c: np.concatenate([DIV_MEMBERS[d] for d in divs]) for c, divs in CONF_DIVS.items()}
DIVISION_OF_IDX = np.empty(len(TEAMS), dtype=int)
for _d, _members in enumerate(DIV_MEMBERS):
    DIVISION_OF_IDX[_members] = _d
CONF_OF_IDX = np.array([0 if CONFERENCE_OF[t] == "AFC" else 1 for t in TEAMS])
ROUNDS = ["Wild Card", "Divisional", "Conference", "Super Bowl"]


@dataclass
class SeasonInputs:
    """Everything the engine needs; built from live data or by hand in tests."""
    elo: np.ndarray        # (32,) current ratings
    pd_buf: np.ndarray     # (32, 5) this season's last 5 point diffs, oldest first, NaN-padded
    to_buf: np.ndarray     # (32, 5) same for turnover margin
    prior_pd: np.ndarray   # (32,) last season's final last-5 values (weeks 1-4 blend)
    prior_to: np.ndarray
    games: pd.DataFrame    # this season's regular-season games in schedule order:
                           # game_id, week, home, away (team index), played, result
                           # (1 home win / 0 / 0.5 tie), rest_diff, div_game, neutral_site
    prob: Callable[[np.ndarray], np.ndarray]  # X in MODEL_FEATURES order -> P(home win)
    elo_params: EloParams
    sigma: float


# ------------------------------------------------------------------ games --

class _State:
    """Per-simulation ratings and last-5 buffers, shape (S, 32[, 5])."""

    def __init__(self, inp: SeasonInputs, n: int):
        self.elo = np.tile(inp.elo, (n, 1)).astype(float)
        self.pd_buf = np.tile(inp.pd_buf, (n, 1, 1)).astype(float)
        self.to_buf = np.tile(inp.to_buf, (n, 1, 1)).astype(float)
        self.rows = np.arange(n)


def _rolling(buf: np.ndarray, prior) -> np.ndarray:
    """Same rule as TeamState.rolling: w = n_valid/5, w*mean + (1-w)*prior."""
    valid = (~np.isnan(buf)).sum(axis=-1)
    return np.nansum(buf, axis=-1) / WINDOW + (1 - valid / WINDOW) * prior


def _push(buf, rows, team, value):
    b = buf[rows, team]
    b[:, :-1] = b[:, 1:]
    b[:, -1] = value
    buf[rows, team] = b


def _play(st: _State, inp: SeasonInputs, home, away, rest_diff, neutral: int, rng, chalk: bool):
    """Play one game in every simulation at once; update ratings. Returns (p_home, margin)."""
    r = st.rows
    eh, ea = st.elo[r, home], st.elo[r, away]
    cols = {
        "elo_diff": eh - ea,
        "pd_diff": _rolling(st.pd_buf[r, home], inp.prior_pd[home]) - _rolling(st.pd_buf[r, away], inp.prior_pd[away]),
        "to_diff": _rolling(st.to_buf[r, home], inp.prior_to[home]) - _rolling(st.to_buf[r, away], inp.prior_to[away]),
        "rest_diff": rest_diff,
        "div_game": (DIVISION_OF_IDX[home] == DIVISION_OF_IDX[away]).astype(float),
        "neutral_site": neutral,
    }
    X = np.column_stack([np.broadcast_to(np.asarray(cols[c], float), (len(r),)) for c in MODEL_FEATURES])
    p = inp.prob(X)

    expected_margin = inp.sigma * norm_ppf(np.clip(p, 1e-9, 1 - 1e-9))
    margin = expected_margin if chalk else expected_margin + inp.sigma * rng.standard_normal(len(r))
    pts = np.rint(margin)
    pts = np.where(pts == 0, np.where(margin < 0, -1.0, 1.0), pts)  # no ties
    home_won = pts > 0

    # Elo update: same formula as features.elo_update
    prm = inp.elo_params
    hfa = 0.0 if neutral else prm.hfa
    exp_home = 1 / (1 + 10 ** (-(eh - ea + hfa) / 400))
    mult = 1.0
    if prm.mov:
        edge = (eh - ea + hfa) * np.where(home_won, 1, -1)
        mult = np.log(np.abs(pts) + 1) * 2.2 / (edge * 0.001 + 2.2)
    delta = prm.k * mult * (home_won - exp_home)
    st.elo[r, home] = eh + delta
    st.elo[r, away] = ea - delta
    _push(st.pd_buf, r, home, pts)
    _push(st.pd_buf, r, away, -pts)
    _push(st.to_buf, r, home, 0.0)
    _push(st.to_buf, r, away, 0.0)
    return p, pts


# -------------------------------------------------------------- standings --

class _Standings:
    def __init__(self, games: pd.DataFrame, res: np.ndarray):
        n_teams = len(TEAMS)
        h, a = games.home.to_numpy(), games.away.to_numpy()
        H = np.zeros((len(games), n_teams)); H[np.arange(len(games)), h] = 1
        A = np.zeros((len(games), n_teams)); A[np.arange(len(games)), a] = 1
        div = (DIVISION_OF_IDX[h] == DIVISION_OF_IDX[a])
        conf = (CONF_OF_IDX[h] == CONF_OF_IDX[a])

        def tally(mask):
            Hm, Am, rm = H[mask], A[mask], res[:, mask]
            won = (rm == 1) @ Hm + (rm == 0) @ Am
            tied = (rm == 0.5) @ Hm + (rm == 0.5) @ Am
            played = Hm.sum(0) + Am.sum(0)
            return won, tied, played

        all_mask = np.ones(len(games), bool)
        self.w, self.t, self.g = tally(all_mask)
        self.div_w, self.div_t, self.div_g = tally(div)
        self.conf_w, self.conf_t, self.conf_g = tally(conf)
        pct = lambda w, t, g: np.where(g > 0, (w + 0.5 * t) / np.maximum(g, 1), 0.5)
        self.pct = pct(self.w, self.t, self.g)
        self.div_pct = pct(self.div_w, self.div_t, self.div_g)
        self.conf_pct = pct(self.conf_w, self.conf_t, self.conf_g)

        self.h2h = np.zeros((len(res), n_teams, n_teams), dtype=np.float32)  # points (win 1, tie 0.5)
        self.h2h_games = np.zeros((n_teams, n_teams))
        for j, (hi, ai) in enumerate(zip(h, a)):
            self.h2h[:, hi, ai] += res[:, j]
            self.h2h[:, ai, hi] += 1 - res[:, j]
            self.h2h_games[hi, ai] += 1
            self.h2h_games[ai, hi] += 1

    def rank(self, cands: np.ndarray, rng, valid=None) -> np.ndarray:
        """Order candidates best-to-worst for each sim. cands: (S, k) team indices."""
        S, k = cands.shape
        rows = np.arange(S)[:, None]
        wp = self.pct[rows, cands]
        if valid is not None:
            wp = np.where(valid, wp, -1.0)
        tied = (wp[:, :, None] == wp[:, None, :]) & ~np.eye(k, dtype=bool)[None]
        if valid is not None:
            tied &= valid[:, :, None] & valid[:, None, :]
        pts = (self.h2h[np.arange(S)[:, None, None], cands[:, :, None], cands[:, None, :]] * tied).sum(-1)
        gms = (self.h2h_games[cands[:, :, None], cands[:, None, :]] * tied).sum(-1)
        h2h = np.where(gms > 0, pts / np.maximum(gms, 1), 0.5)
        keys = (rng.random((S, k)), -self.conf_pct[rows, cands], -self.div_pct[rows, cands], -h2h, -wp)
        return np.lexsort(keys, axis=-1)   # last key is primary


# ----------------------------------------------------------------- engine --

def _simulate_chunk(inp: SeasonInputs, n: int, rng, chalk: bool, detail: bool) -> dict:
    g = inp.games
    st = _State(inp, n)
    res = np.tile(g.result.fillna(0).to_numpy(float), (n, 1))
    todo = np.flatnonzero(~g.played.to_numpy())
    sim_p = np.full((n, len(todo)), np.nan)
    sim_pts = np.zeros((n, len(todo)))
    for k, j in enumerate(todo):
        row = g.iloc[j]
        p, pts = _play(st, inp, int(row.home), int(row.away), float(row.rest_diff), int(row.neutral_site), rng, chalk)
        res[:, j] = pts > 0
        sim_p[:, k], sim_pts[:, k] = p, pts

    stand = _Standings(g, res)
    div_order = np.stack([stand.rank(np.tile(DIV_MEMBERS[d], (n, 1)), rng) for d in range(8)], axis=1)  # (n, 8, 4)
    div_ranked = DIV_MEMBERS[np.arange(8)[None, :, None], div_order]                                   # teams
    winners = div_ranked[:, :, 0]

    seeds, playoff_games, champs = {}, [], {}
    for c, conf in enumerate(("AFC", "NFC")):
        dw = winners[:, CONF_DIVS[conf]]
        top4 = np.take_along_axis(dw, stand.rank(dw, rng), axis=1)
        members = np.tile(CONF_MEMBERS[conf], (n, 1))
        is_dw = (members[:, :, None] == dw[:, None, :]).any(-1)
        wild = np.take_along_axis(members, stand.rank(members, rng, valid=~is_dw), axis=1)[:, :3]
        seeds[conf] = np.concatenate([top4, wild], axis=1)

    div_round, conf_round = [], []

    def game(rnd, conf, home, away, hseed, aseed, rest=0.0, neutral=0):
        p, pts = _play(st, inp, home, away, rest, neutral, rng, chalk)
        winner = np.where(pts > 0, home, away)
        playoff_games.append({"round": rnd, "conf": conf, "home": home, "away": away, "home_seed": hseed,
                              "away_seed": aseed, "p_home": p, "margin": pts, "winner": winner})
        return winner, np.where(pts > 0, hseed, aseed)

    for conf in ("AFC", "NFC"):
        s = seeds[conf]
        alive_t, alive_s = [s[:, 0]], [np.ones(n, int)]
        for hi, lo in ((2, 7), (3, 6), (4, 5)):
            w, ws = game("Wild Card", conf, s[:, hi - 1], s[:, lo - 1], np.full(n, hi), np.full(n, lo))
            alive_t.append(w); alive_s.append(ws)
        t, sd = np.stack(alive_t, 1), np.stack(alive_s, 1)
        order = np.argsort(sd, axis=1)                         # re-seed: best first
        t, sd = np.take_along_axis(t, order, 1), np.take_along_axis(sd, order, 1)
        div_round.append(t)
        w1, s1 = game("Divisional", conf, t[:, 0], t[:, 3], sd[:, 0], sd[:, 3], rest=BYE_REST_DAYS)
        w2, s2 = game("Divisional", conf, t[:, 1], t[:, 2], sd[:, 1], sd[:, 2])
        conf_round.append(np.stack([w1, w2], 1))
        first = s1 < s2
        champs[conf], _ = game("Conference", conf, np.where(first, w1, w2), np.where(first, w2, w1),
                               np.minimum(s1, s2), np.maximum(s1, s2))
    champion, _ = game("Super Bowl", "SB", champs["AFC"], champs["NFC"],
                       np.zeros(n, int), np.zeros(n, int), neutral=1)

    out = {"wins": stand.w + 0.5 * stand.t, "div_wins": stand.div_w + 0.5 * stand.div_t, "div_games": stand.div_g,
           "games": stand.g, "winners": winners, "seeds": seeds,
           "div_round": np.concatenate(div_round, 1), "conf_round": np.concatenate(conf_round, 1),
           "afc": champs["AFC"], "nfc": champs["NFC"], "champion": champion}
    if detail:
        out.update(stand=stand, div_ranked=div_ranked, sim_idx=todo, sim_p=sim_p, sim_pts=sim_pts,
                   playoff_games=playoff_games)
    return out


def simulate(inp: SeasonInputs, n_sims: int = 1, seed: int = SEED, chalk: bool = False, detail: bool = False) -> list[dict]:
    """Run n_sims seasons. Returns one result dict per chunk (deterministic for a given seed)."""
    children = np.random.SeedSequence(seed).spawn(-(-n_sims // CHUNK))
    out, left = [], n_sims
    for child in children:
        n = min(CHUNK, left)
        out.append(_simulate_chunk(inp, n, np.random.default_rng(child), chalk, detail))
        left -= n
    return out


# ------------------------------------------------------------ Monte Carlo --

def monte_carlo(inp: SeasonInputs, record: pd.DataFrame, n_sims=N_SIMS, seed=SEED) -> pd.DataFrame:
    """Per-team odds over n_sims seasons. record: current wins/losses/ties per team."""
    chunks = simulate(inp, n_sims, seed)

    def share(key_fn):
        counts = sum(np.bincount(np.asarray(key_fn(c)).ravel(), minlength=len(TEAMS)) for c in chunks)
        return counts / n_sims

    out = pd.DataFrame({
        "team": TEAMS,
        "name": [NAMES[t] for t in TEAMS],
        "conference": [CONFERENCE_OF[t] for t in TEAMS],
        "division": [DIV_NAMES[DIVISION_OF_IDX[i]] for i in range(len(TEAMS))],
    })
    out = out.join(record.reset_index(drop=True))
    out["proj_wins"] = sum(c["wins"].sum(axis=0) for c in chunks) / n_sims
    out["proj_div_wins"] = sum(c["div_wins"].sum(axis=0) for c in chunks) / n_sims
    out["games"] = chunks[0]["games"]
    out["div_games"] = chunks[0]["div_games"]
    out["p_playoffs"] = share(lambda c: np.concatenate([c["seeds"]["AFC"], c["seeds"]["NFC"]], axis=1))
    out["p_division"] = share(lambda c: c["winners"])
    out["p_top_seed"] = share(lambda c: np.concatenate([c["seeds"]["AFC"][:, 0], c["seeds"]["NFC"][:, 0]]))
    out["p_div_round"] = share(lambda c: c["div_round"])
    out["p_conf_round"] = share(lambda c: c["conf_round"])
    out["p_reach_sb"] = share(lambda c: np.concatenate([c["afc"], c["nfc"]]))
    out["p_sb"] = share(lambda c: c["champion"])
    for k in range(7):  # share of sims in which the team was seed k+1 of its conference
        out[f"p_seed_{k + 1}"] = share(lambda c, k=k: np.concatenate([c["seeds"]["AFC"][:, k], c["seeds"]["NFC"][:, k]]))
    pairs = sum(np.bincount(c["afc"] * len(TEAMS) + c["nfc"], minlength=len(TEAMS) ** 2) for c in chunks)
    # Plain nested list, not a numpy array: pandas compares attrs with ==, which arrays break.
    out.attrs["sb_pairs"] = (pairs / n_sims).reshape(len(TEAMS), len(TEAMS)).tolist()  # [AFC champ][NFC champ]
    out.attrs["games_left"] = int((~inp.games.played).sum())
    return out


# ----------------------------------------------------------- single season --

def _roman(n: int) -> str:
    vals = [(50, "L"), (40, "XL"), (10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I")]
    s = ""
    for v, r in vals:
        while n >= v:
            s += r
            n -= v
    return s


def single_season(inp: SeasonInputs, seed: int, chalk: bool = False) -> dict:
    """One simulated season, unpacked for display."""
    c = simulate(inp, 1, seed, chalk=chalk, detail=True)[0]
    st = c["stand"]
    seed_of, badge = {}, {}
    for conf in ("AFC", "NFC"):
        for k, t in enumerate(c["seeds"][conf][0], 1):
            seed_of[int(t)] = k
            badge[int(t)] = "z" if k == 1 else "y" if k <= 4 else "x"  # bye / division / wild card

    def rec(w, t, g):
        w, t = int(w), int(t)
        return f"{w}–{int(g) - w - t}" + (f"–{t}" if t else "")

    standings = {}
    for conf, divs in CONF_DIVS.items():
        standings[conf] = []
        for d in divs:
            rows = []
            for i in c["div_ranked"][0, d]:
                i = int(i)
                rows.append({"team": TEAMS[i], "record": rec(st.w[0, i], st.t[0, i], st.g[i]),
                             "div_record": rec(st.div_w[0, i], st.div_t[0, i], st.div_g[i]),
                             "seed": seed_of.get(i), "badge": badge.get(i)})
            standings[conf].append({"division": DIV_NAMES[d], "teams": rows})

    rounds = {r: [] for r in ROUNDS}
    for pg in c["playoff_games"]:
        h, a, w = int(pg["home"][0]), int(pg["away"][0]), int(pg["winner"][0])
        rounds[pg["round"]].append({
            "conf": pg["conf"], "home": TEAMS[h], "away": TEAMS[a],
            "home_seed": int(pg["home_seed"][0]) or None, "away_seed": int(pg["away_seed"][0]) or None,
            "p_home": float(pg["p_home"][0]), "p_away": 1 - float(pg["p_home"][0]),
            "winner": TEAMS[w], "margin": int(abs(pg["margin"][0])),
        })

    games = inp.games
    sim_games = []
    for k, j in enumerate(c["sim_idx"]):
        g = games.iloc[j]
        pts = c["sim_pts"][0, k]
        sim_games.append({"week": int(g.week), "home": TEAMS[int(g.home)], "away": TEAMS[int(g.away)],
                          "p_home": float(c["sim_p"][0, k]), "winner": TEAMS[int(g.home if pts > 0 else g.away)],
                          "margin": int(abs(pts))})

    return {"seed": seed, "chalk": chalk, "standings": standings, "rounds": [(r, rounds[r]) for r in ROUNDS],
            "champion": TEAMS[int(c["champion"][0])], "sim_games": sim_games,
            "super_bowl": f"Super Bowl {_roman(CURRENT_SEASON - 1965)}"}


# --------------------------------------------------------- average bracket --

def average_bracket(odds: pd.DataFrame) -> dict:
    """One bracket that summarises all Monte Carlo seasons.

    Field: each division's most frequent winner, and the three teams that won a
    wild card (any of seeds 5-7) most often. Within each group, teams are put on
    seeds so that the total share of sims in which each team held its seed is
    as large as possible (an assignment problem). Choosing the field first
    matters: a team that is usually a wild card but spread over seeds 5, 6 and
    7 might top none of them individually. Games: the team that reached the next round in more simulations
    advances. Percentages are those shares, so the two teams in a game do not
    sum to 100%: each counts every path, not just this matchup.
    """
    o = odds.set_index("team")

    def record(t):
        w, l, tie = int(o.loc[t, "wins"]), int(o.loc[t, "losses"]), int(o.loc[t, "ties"])
        return f"{w}\u2013{l}" + (f"\u2013{tie}" if tie else "")

    def assign(teams, seeds):
        share = o.loc[teams, [f"p_seed_{k}" for k in seeds]].to_numpy()
        return {seeds[c]: teams[r] for r, c in best_assignment(share)}

    def game(home, away, seed_of, key):
        winner = home if o.loc[home, key] >= o.loc[away, key] else away
        return {"home": home, "away": away, "home_seed": seed_of.get(home), "away_seed": seed_of.get(away),
                "winner": winner, "p_home": float(o.loc[home, key]), "p_away": float(o.loc[away, key])}

    out = {}
    for conf, divs in CONF_DIVS.items():
        winners = [o.loc[DIVISIONS[DIV_NAMES[d]], "p_division"].idxmax() for d in divs]
        others = o[(o.conference == conf) & ~o.index.isin(winners)]
        wild = list((others.p_playoffs - others.p_division).nlargest(3).index)  # P(wild card)
        by_seed = {**assign(winners, [1, 2, 3, 4]), **assign(wild, [5, 6, 7])}
        seed_of = {t: k for k, t in by_seed.items()}

        wc = [game(by_seed[hi], by_seed[lo], seed_of, "p_div_round") for hi, lo in ((2, 7), (3, 6), (4, 5))]
        alive = sorted([by_seed[1]] + [g["winner"] for g in wc], key=seed_of.get)
        div = [game(alive[0], alive[3], seed_of, "p_conf_round"), game(alive[1], alive[2], seed_of, "p_conf_round")]
        a, b = sorted((g["winner"] for g in div), key=seed_of.get)
        final = game(a, b, seed_of, "p_reach_sb")
        out[conf] = {
            "seeds": [{"seed": k, "team": by_seed[k], "record": record(by_seed[k]), "division_winner": k <= 4,
                       "p_seed": float(o.loc[by_seed[k], f"p_seed_{k}"]),
                       "p_playoffs": float(o.loc[by_seed[k], "p_playoffs"])} for k in range(1, 8)],
            "rounds": [("Wild card", wc), ("Divisional", div), ("Conference", [final])],
            "champion": final["winner"],
        }

    afc, nfc = out["AFC"]["champion"], out["NFC"]["champion"]
    sb = game(afc, nfc, {}, "p_sb")
    sb.update(reach_home=float(o.loc[afc, "p_reach_sb"]), reach_away=float(o.loc[nfc, "p_reach_sb"]))
    pairs = odds.attrs["sb_pairs"]
    return {
        "AFC": out["AFC"], "NFC": out["NFC"], "super_bowl": sb,
        "p_matchup": float(pairs[IDX[afc]][IDX[nfc]]),
        "top_champion": o.p_sb.idxmax(), "p_top_champion": float(o.p_sb.max()),
    }


def average_season(odds: pd.DataFrame) -> dict:
    """The Monte Carlo average, shaped like single_season() so the same page layout shows it.

    Standings: average record over all simulations, each division ordered by
    average wins. Seeds and games: the average bracket. Game percentages are
    the share of simulations in which each team reached the next round.
    """
    b = odds.attrs["bracket"]
    o = odds.set_index("team")
    seed_of = {s["team"]: s["seed"] for conf in ("AFC", "NFC") for s in b[conf]["seeds"]}

    def avg(w, g):
        return f"{w:.1f}\u2013{g - w:.1f}"

    standings = {}
    for conf, divs in CONF_DIVS.items():
        standings[conf] = []
        for d in divs:
            teams = sorted(DIVISIONS[DIV_NAMES[d]], key=lambda t: -o.loc[t, "proj_wins"])
            rows = []
            for t in teams:
                k = seed_of.get(t)
                rows.append({"team": t, "record": avg(o.loc[t, "proj_wins"], o.loc[t, "games"]),
                             "div_record": avg(o.loc[t, "proj_div_wins"], o.loc[t, "div_games"]),
                             "seed": k, "badge": None if k is None else "z" if k == 1 else "y" if k <= 4 else "x"})
            standings[conf].append({"division": DIV_NAMES[d], "teams": rows})

    def as_game(g, rnd, conf):
        return {**g, "round": rnd, "conf": conf, "margin": None}

    rounds = []
    for k, name in enumerate(["Wild Card", "Divisional", "Conference"]):
        games = [as_game(g, name, conf) for conf in ("AFC", "NFC") for g in b[conf]["rounds"][k][1]]
        rounds.append((name, games))
    sb = b["super_bowl"]
    rounds.append(("Super Bowl", [as_game({**sb, "home_seed": None, "away_seed": None}, "Super Bowl", "SB")]))
    champ = sb["winner"]
    return {"standings": standings, "rounds": rounds, "champion": champ, "sim_games": [],
            "super_bowl": f"Super Bowl {_roman(CURRENT_SEASON - 1965)}", "average": True, "chalk": False,
            "p_champion": float(o.loc[champ, "p_sb"]), "p_matchup": b["p_matchup"],
            "top_champion": b["top_champion"], "p_top_champion": b["p_top_champion"]}


# ----------------------------------------------------------- live wrapper --

def inputs_from_snapshot(snap, prob_array, sigma=None) -> tuple[SeasonInputs, pd.DataFrame]:
    """Engine inputs + current record (wins/losses/ties) from a live.Snapshot."""
    from live import ELO  # frozen Elo settings from models/config.json

    teams = {}
    for t in TEAMS:
        ts = snap.teams_now[t]
        if ts.season != CURRENT_SEASON:  # season not started for this team: apply off-season step
            ts = copy.deepcopy(ts)
            ts.start_season(CURRENT_SEASON, ELO)
        teams[t] = ts

    def buf(stat):
        b = np.full((len(TEAMS), WINDOW), np.nan)
        for i, t in enumerate(TEAMS):
            vals = [g[stat] for g in teams[t].recent]
            if vals:
                b[i, -len(vals):] = vals
        return b

    f = snap.features
    season = f[(f.season == CURRENT_SEASON) & (f.game_type == "REG")].sort_values(["gameday", "gametime", "game_id"])
    result = np.where(season.home_score > season.away_score, 1.0,
                      np.where(season.home_score < season.away_score, 0.0, 0.5))
    games = pd.DataFrame({
        "game_id": season.game_id.to_numpy(), "week": season.week.to_numpy(),
        "home": season.home_team.map(IDX).to_numpy(), "away": season.away_team.map(IDX).to_numpy(),
        "played": season.played.to_numpy(), "result": np.where(season.played, result, np.nan),
        "rest_diff": season.rest_diff.fillna(0).to_numpy(float),
        "div_game": season.div_game.to_numpy(), "neutral_site": season.neutral_site.to_numpy(),
    })
    inp = SeasonInputs(
        elo=np.array([teams[t].elo for t in TEAMS]),
        pd_buf=buf("pd"), to_buf=buf("to"),
        prior_pd=np.array([teams[t].prior["pd"] for t in TEAMS]),
        prior_to=np.array([teams[t].prior["to"] for t in TEAMS]),
        games=games, prob=prob_array, elo_params=ELO,
        sigma=snap.spread_sigma if sigma is None else sigma,
    )

    played = games[games.played]
    rec = pd.DataFrame(0, index=range(len(TEAMS)), columns=["wins", "losses", "ties"])
    for g in played.itertuples(index=False):
        if g.result == 1:
            rec.loc[g.home, "wins"] += 1; rec.loc[g.away, "losses"] += 1
        elif g.result == 0:
            rec.loc[g.away, "wins"] += 1; rec.loc[g.home, "losses"] += 1
        else:
            rec.loc[g.home, "ties"] += 1; rec.loc[g.away, "ties"] += 1
    return inp, rec


def main() -> None:
    import live
    from train import REPORTS

    odds = live.compute_season_odds()
    path = REPORTS / f"season_odds_{CURRENT_SEASON}.csv"
    odds.sort_values("p_sb", ascending=False).to_csv(path, index=False, float_format="%.4f")
    show = odds.sort_values("p_sb", ascending=False).head(12)
    print(show[["team", "wins", "losses", "ties", "proj_wins", "p_playoffs", "p_division", "p_sb"]].round(3).to_string(index=False))
    print(f"\n{N_SIMS:,} seasons, seed {SEED}, {odds.attrs['games_left']} games left. Wrote {path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
