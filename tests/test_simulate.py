"""Season simulation engine: reproducibility, invariants, tiebreakers, hot updates (no network)."""
import numpy as np
import pandas as pd
import pytest

import simulate as sim
from features import EloParams, elo_update
from teams import CONFERENCE_OF, DIVISIONS

N = len(sim.TEAMS)
I = sim.IDX


def elo_prob(X):
    """Toy model: P(home) from Elo difference only (column 0)."""
    return 1 / (1 + 10 ** (-X[:, 0] / 400))


def _inputs(games, elo=None):
    if elo is None:
        elo = 1500 + np.random.default_rng(0).normal(0, 80, N)  # varied team strengths
    return sim.SeasonInputs(
        elo=elo,
        pd_buf=np.zeros((N, 5)), to_buf=np.zeros((N, 5)), prior_pd=np.zeros(N), prior_to=np.zeros(N),
        games=games, prob=elo_prob, elo_params=EloParams(k=25, hfa=0, mov=True), sigma=11.7,
    )


def _random_schedule(n_games=200, played_share=0.4, seed=1):
    rng = np.random.default_rng(seed)
    home = rng.integers(0, N, n_games)
    away = (home + rng.integers(1, N, n_games)) % N
    played = np.arange(n_games) < int(n_games * played_share)
    result = np.where(played, rng.integers(0, 2, n_games).astype(float), np.nan)
    return pd.DataFrame({"game_id": [f"g{i}" for i in range(n_games)], "week": np.arange(n_games) // 16 + 1,
                         "home": home, "away": away, "played": played, "result": result,
                         "rest_diff": 0.0, "div_game": 0, "neutral_site": 0})


def _played(results):
    """Fully played schedule from [(home, away, home_won)] using team codes."""
    rows = [{"game_id": f"g{k}", "week": 1, "home": I[h], "away": I[a], "played": True,
             "result": float(w), "rest_diff": 0.0, "div_game": 0, "neutral_site": 0}
            for k, (h, a, w) in enumerate(results)]
    return pd.DataFrame(rows)


# ------------------------------------------------------------ reproducible --

def test_same_seed_same_season_and_champion():
    inp = _inputs(_random_schedule())
    a, b = sim.single_season(inp, seed=11), sim.single_season(inp, seed=11)
    assert a["champion"] == b["champion"]
    assert a == b


def test_different_seed_changes_some_results():
    inp = _inputs(_random_schedule())
    a, b = sim.single_season(inp, seed=11), sim.single_season(inp, seed=12)
    winners_a = [g["winner"] for g in a["sim_games"]]
    winners_b = [g["winner"] for g in b["sim_games"]]
    assert winners_a != winners_b


def test_chalk_is_deterministic_and_favorites_always_win():
    inp = _inputs(_random_schedule())
    a, b = sim.single_season(inp, seed=1, chalk=True), sim.single_season(inp, seed=99, chalk=True)
    assert a["sim_games"] == b["sim_games"] and a["champion"] == b["champion"]
    for g in a["sim_games"] + [g for _, games in a["rounds"] for g in games]:
        fav = g["home"] if g["p_home"] >= 0.5 else g["away"]
        assert g["winner"] == fav


# -------------------------------------------------------------- invariants --

def test_monte_carlo_probabilities_add_up():
    inp = _inputs(_random_schedule())
    rec = pd.DataFrame(0, index=range(N), columns=["wins", "losses", "ties"])
    out = sim.monte_carlo(inp, rec, n_sims=3000, seed=1)
    assert out.p_playoffs.sum() == pytest.approx(14)
    assert out.p_division.sum() == pytest.approx(8)
    assert out.p_top_seed.sum() == pytest.approx(2)
    assert out.p_reach_sb.sum() == pytest.approx(2)
    assert out.p_sb.sum() == pytest.approx(1)
    cols = ["p_playoffs", "p_division", "p_top_seed", "p_reach_sb", "p_sb"]
    assert ((out[cols] >= 0) & (out[cols] <= 1)).all().all()
    assert ((out.p_playoffs >= out.p_division) & (out.p_division >= out.p_top_seed)).all()


def test_every_bracket_has_seven_distinct_teams_including_division_winners():
    inp = _inputs(_random_schedule())
    c = sim.simulate(inp, n_sims=800, seed=4)[0]
    for conf, divs in sim.CONF_DIVS.items():
        s = c["seeds"][conf]
        assert all(len(set(row)) == 7 for row in s)
        assert all(set(w) <= set(row[:4]) for w, row in zip(c["winners"][:, divs], s))
        assert all(CONFERENCE_OF[sim.TEAMS[t]] == conf for row in s for t in row)


def test_no_simulated_ties():
    inp = _inputs(_random_schedule())
    season = sim.single_season(inp, seed=3)
    assert all(g["margin"] > 0 for g in season["sim_games"])


# ------------------------------------------------------------- tiebreakers --

def _division_winner(results, n=400):
    c = sim.simulate(_inputs(_played(results)), n_sims=n, seed=0)[0]
    east = sim.DIV_NAMES.index("AFC East")
    return pd.Series(c["winners"][:, east]).map(dict(enumerate(sim.TEAMS))).value_counts(normalize=True)


# Fillers so the other AFC East teams finish below the tied pair.
_BOTTOM = [("NE", "SF", 0), ("NYJ", "SEA", 0)]


def test_head_to_head_breaks_a_tie():
    # BUF and MIA are both 1-1 overall AND 1-1 in the division, so only their
    # meeting (BUF won) separates them. NE and NYJ finish below.
    shares = _division_winner([
        ("BUF", "MIA", 1),   # BUF beats MIA
        ("NYJ", "BUF", 1),   # BUF loses to NYJ (division)
        ("MIA", "NE", 1),    # MIA beats NE (division)
        ("NYJ", "SF", 0), ("NYJ", "SEA", 0), ("NYJ", "LA", 0),  # NYJ ends 1-3
    ])
    assert shares.get("BUF") == 1


def test_division_record_breaks_a_tie_without_head_to_head():
    # Both 1-1, never met; BUF's win came against a division rival.
    shares = _division_winner([("BUF", "NE", 1), ("BUF", "KC", 0), ("MIA", "DEN", 1), ("MIA", "LV", 0),
                               ("NYJ", "SEA", 0)])
    assert shares.get("BUF") == 1


def test_identical_teams_are_split_at_random():
    # Mirror-image records: nothing separates BUF and MIA, so a coin flip decides.
    shares = _division_winner([("BUF", "KC", 1), ("BUF", "LV", 0), ("MIA", "DEN", 1), ("MIA", "LAC", 0)] + _BOTTOM,
                              n=2000)
    assert 0.4 < shares.get("BUF", 0) < 0.6 and 0.4 < shares.get("MIA", 0) < 0.6


# -------------------------------------------------------------- hot updates --

def test_simulated_game_updates_elo_like_a_real_game():
    games = pd.DataFrame([{"game_id": "g", "week": 1, "home": I["KC"], "away": I["BUF"], "played": False,
                           "result": np.nan, "rest_diff": 0.0, "div_game": 0, "neutral_site": 0}])
    elo = np.full(N, 1500.0); elo[I["KC"]], elo[I["BUF"]] = 1600, 1450
    inp = _inputs(games, elo=elo)
    st = sim._State(inp, 1)
    _, pts = sim._play(st, inp, I["KC"], I["BUF"], 0.0, 0, np.random.default_rng(0), chalk=True)
    margin = int(pts[0])
    exp_home, exp_away = elo_update(1600, 1450, margin, 0, neutral=0, params=inp.elo_params)
    assert st.elo[0, I["KC"]] == pytest.approx(exp_home)
    assert st.elo[0, I["BUF"]] == pytest.approx(exp_away)
    assert st.pd_buf[0, I["KC"], -1] == margin and st.pd_buf[0, I["BUF"], -1] == -margin


def test_ratings_carry_through_the_season():
    # KC hosts BUF twice. With chalk KC (the favorite) wins game 1, so its rating
    # rises and it must be an even bigger favorite in game 2. A "cold"
    # simulation would give both games the same probability.
    row = {"week": 1, "home": I["KC"], "away": I["BUF"], "played": False, "result": np.nan,
           "rest_diff": 0.0, "div_game": 0, "neutral_site": 0}
    games = pd.DataFrame([{**row, "game_id": "g1"}, {**row, "game_id": "g2", "week": 2}])
    elo = np.full(N, 1500.0); elo[I["KC"]] = 1550
    c = sim.simulate(_inputs(games, elo=elo), n_sims=1, seed=0, chalk=True, detail=True)[0]
    p1, p2 = c["sim_p"][0]
    assert p1 > 0.5 and p2 > p1


# ---------------------------------------------------------- average bracket --

def _mc_odds():
    inp = _inputs(_random_schedule())
    rec = pd.DataFrame(0, index=range(N), columns=["wins", "losses", "ties"])
    return sim.monte_carlo(inp, rec, n_sims=3000, seed=2)


def test_round_reach_shares_add_up():
    out = _mc_odds()
    assert out.p_div_round.sum() == pytest.approx(8)
    assert out.p_conf_round.sum() == pytest.approx(4)
    for k in range(1, 8):
        assert out.groupby("conference")[f"p_seed_{k}"].sum().tolist() == pytest.approx([1, 1])
    assert np.sum(out.attrs["sb_pairs"]) == pytest.approx(1)


def test_average_bracket_is_a_valid_bracket():
    out = _mc_odds()
    o = out.set_index("team")
    b = sim.average_bracket(out)
    for conf, divs in sim.CONF_DIVS.items():
        seeds = [s["team"] for s in b[conf]["seeds"]]
        assert len(set(seeds)) == 7 and all(CONFERENCE_OF[t] == conf for t in seeds)
        # one winner per division, each division's most frequent winner
        assert {sim.DIVISION_OF_IDX[I[t]] for t in seeds[:4]} == set(divs)
        for d in divs:
            assert o.loc[DIVISIONS[sim.DIV_NAMES[d]], "p_division"].idxmax() in seeds[:4]
        # wild cards: the three most frequent wild-card teams
        others = o[(o.conference == conf) & ~o.index.isin(seeds[:4])]
        assert set(seeds[4:]) == set((others.p_playoffs - others.p_division).nlargest(3).index)
        # every game's winner is the team that advanced more often, and the higher seed hosts
        for _, games in b[conf]["rounds"]:
            for g in games:
                assert g["home_seed"] < g["away_seed"]
                assert g["winner"] == (g["home"] if g["p_home"] >= g["p_away"] else g["away"])
    assert b["super_bowl"]["winner"] in (b["AFC"]["champion"], b["NFC"]["champion"])


def test_average_season_matches_the_page_shape_and_is_consistent():
    out = _mc_odds()
    out.attrs["bracket"] = sim.average_bracket(out)
    avg = sim.average_season(out)
    single = sim.single_season(_inputs(_random_schedule()), seed=1)
    assert set(single) - {"seed", "chalk"} <= set(avg)  # same layout keys
    assert [r for r, _ in avg["rounds"]] == sim.ROUNDS
    assert [len(g) for _, g in avg["rounds"]] == [6, 4, 2, 1]
    for conf in ("AFC", "NFC"):
        for div in avg["standings"][conf]:
            wins = [float(t["record"].split("\u2013")[0]) for t in div["teams"]]
            assert wins == sorted(wins, reverse=True)           # ordered by average wins
        seeded = [t for div in avg["standings"][conf] for t in div["teams"] if t["seed"]]
        assert sorted(t["seed"] for t in seeded) == list(range(1, 8))
    assert avg["champion"] == avg["rounds"][-1][1][0]["winner"]
