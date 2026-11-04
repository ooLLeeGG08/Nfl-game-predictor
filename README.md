# NFL game outcome predictor

Home-team win probability for NFL games from a logistic regression on
pre-kickoff features (Elo + last-5-game form), evaluated honestly against
"always pick home" and the Vegas line.

## Results: 2025 test season (evaluated once)

284 games (regular season + playoffs; 1 tie excluded; every game had a line).

| Method | Accuracy | Log loss | Brier |
|---|---|---|---|
| **Model** | **0.655** | **0.624** | **0.218** |
| Always home | 0.535 | 0.692 | 0.249 |
| Vegas (moneyline, vig removed) | 0.660 | 0.607 | 0.211 |

The model beats the home baseline on every metric and trails Vegas on every
metric, which is the expected ordering: the market knows about injuries,
quarterback changes and weather; this model does not. Playoffs alone: 13 games,
accuracy 0.769, too few to read anything into.

Validation season (2024, 285 games) for comparison: model 0.674 / 0.622 / 0.216,
home 0.547 / 0.689 / 0.248, Vegas 0.707 / 0.589 / 0.201.

![Calibration on 2025](reports/calibration_test.png)

Calibration is reasonable above 0.5. Below about 0.4 the model is
under-confident about home underdogs (they won more often than predicted),
but with 284 games spread over 10 bins, the low-probability bins are small.

## Model

`StandardScaler` + `LogisticRegression` (seed 42), trained on 2006-2023
regular-season games. Features are home-minus-away differences; there is no
home/away flag, because the intercept carries home advantage.

| Feature | Meaning |
|---|---|
| `elo_diff` | Elo rating difference before kickoff |
| `pd_diff` | last-5-game point differential |
| `to_diff` | last-5-game turnover margin |
| `rest_diff` | days of rest |
| `div_game` | division game flag |
| `neutral_site` | `location == "Neutral"` (international games, Super Bowl) |

**Points for/against vs point differential.** `pd_diff` equals
`pf_diff - pa_diff` exactly, so using all three would split one effect across
collinear coefficients. We keep `pd_diff` and drop `pf_diff`/`pa_diff`.

**Neutral sites at prediction time.** With only 51 neutral-site training
games, the model learned a spurious penalty for the team *listed* as home
(equal teams got 46%, not 50%). The app therefore averages both labelings for
neutral games, `p = (p(A listed home) + 1 - p(B listed home)) / 2`, so equal
teams get exactly 50% and the answer does not depend on which team is listed
first (`live.home_win_prob`, tested in `tests/test_live.py`). Non-neutral
games use the model unchanged. The test-season metrics above were computed
before this adjustment; it affects only the ~5 neutral games per season.

![Feature importance](reports/feature_importance.png)

Coefficients are on standardized features (log-odds per 1 SD). Elo dominates.
`to_diff` is slightly *negative*: turnover margin is largely luck, so among
teams with the same Elo and point differential, the one with the better
turnover margin is a bit overrated.

### Elo

| Parameter | Value | How chosen |
|---|---|---|
| K | 25 | grid {10,15,20,25,30,40}, lowest 2024 log loss |
| Home-field (Elo update only) | 0 | grid {0,15,25,35,48,65}; flat (log loss range ~0.0006) |
| Margin-of-victory multiplier | on (FiveThirtyEight form) | better in every grid cell |
| Between-season regression | 1/3 toward 1500 | experiment below |

Elo starts at 1500 in 2002. Seasons 2002-2005 only warm up the ratings and are
never training rows. Franchise codes are unified (STL→LA, SD→LAC, OAK→LV) so
ratings follow teams through relocations.

### Rolling stats

Last 5 games. In weeks 1-4 the value is blended with last season's final
value: `w * current + (1 - w) * prior_final`, `w = games_played / 5`.
Turnovers come from `nflreadpy.load_team_stats(summary_level="week")`:
giveaways = interceptions thrown + fumbles lost; takeaways = the opponent's
giveaways in the same game.

## Experiments (2024 validation only)

| Experiment | Setting | Accuracy | Log loss | Brier |
|---|---|---|---|---|
| Training window | last 5 seasons | 0.667 | 0.624 | 0.217 |
| Training window | last 10 seasons | 0.677 | 0.619 | 0.215 |
| Training window | **last 18 seasons (chosen)** | 0.674 | 0.622 | 0.216 |
| Elo carryover | full | 0.667 | 0.627 | 0.218 |
| Elo carryover | **1/3 regression (chosen)** | 0.674 | 0.622 | 0.216 |
| Elo carryover | reset to 1500 | 0.642 | 0.639 | 0.224 |

The 10-season window scored slightly better, but the gap is within the noise
of one 285-game season, so we kept the planned 2006-2023 window rather than
tune further on validation.

## Evaluation protocol

- Time-based split: Elo warm-up 2002-2005, train 2006-2023, validate 2024, test 2025.
- All tuning and experiments use 2024. Loading 2025 raises an error unless
  `allow_test=True`, and `evaluate.py --final-test` refuses to run twice.
- Ties are excluded from training and evaluation (14 in 2002-2024, 1 in 2025).
- Accuracy gives a 0.5 probability half credit.
- **Vegas:** vig-free probability from moneylines when present; otherwise
  `Phi(spread_line / sigma)`, with sigma = 11.7 fitted on 2006-2023 to match
  moneyline probabilities. Positive `spread_line` = home favored. Pick'em = 0.5.
  Games without a line are dropped from the Vegas comparison only (0 in 2024 and 2025).
- **Home baseline probability:** the 2006-2023 home win rate.
- **Leakage guard:** `tests/test_leakage.py` rebuilds features from only
  earlier-gameday games and checks they match; a second test scrambles future
  scores and checks past features do not move. Both tests were confirmed to
  fail on a deliberately planted leak.

## Usage

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
python -m data.fetch          # cache 2002-2024 in data/raw/ (--refresh to re-download)
python features.py            # data/features.parquet
python -m pytest              # leakage + feature tests
python train.py               # tune Elo on 2024, fit, save models/pipeline.joblib
python evaluate.py            # validation report + experiments -> reports/
python evaluate.py --replot   # redraw light (README) and dark (app) plots
python simulate.py            # season odds -> reports/season_odds_2026.csv
gunicorn app:app --workers 1 --threads 4   # site on http://127.0.0.1:8000
```

The app loads `models/pipeline.joblib` and never retrains. Historical raw data
(2002-2025) is committed in `data/raw/`; only the current season is downloaded
at runtime and refreshed every 6 hours. Render settings are in `render.yaml`: one worker with 4 threads (each worker
holds ~300 MB, so two would exceed the 512 MB free tier) and no `--preload`
(forking after numpy/polars/pyarrow start their thread pools segfaulted on
macOS and can hang on Linux).

## Web app

- **Home** (`/`): any matchup from current ratings (as of the latest fully
  completed week, equal rest assumed).
- **This week** (`/week`): the earliest 2026 week with an unplayed game. Each
  row shows the model's pick and probability next to the Vegas spread and its
  implied probability (`market.py`, same method as the evaluation). Rows where
  the two differ by more than 10 points are highlighted. Finished games show
  the score and whether each pick was right. Kickoffs are stored in US Eastern
  and converted to the visitor's time zone in the browser.
- **Game** (`/game/<game_id>`): the exact pre-kickoff inputs for one game,
  including real rest days.
- **Season** (`/season`): Monte Carlo season odds (below), sortable.
- **Simulate** (`/simulate`): one simulated season (or chalk), with standings, bracket and champion.
- **About** (`/about`): results, calibration, feature importance, experiments.

## Season simulation (`simulate.py`)

One engine (`simulate.simulate`) powers three modes: a single random season
(`/simulate?seed=N`), a "chalk" season where every favorite wins
(`/simulate?mode=chalk`), and the 10,000-season Monte Carlo behind `/season`.

- **Games.** Played games keep their real results. Each remaining game, in
  schedule order, gets P(home win) from the frozen model (neutral sites
  symmetrized). A margin is drawn as `Normal(sigma * Phi^-1(p), sigma)` with
  the spread scale sigma = 11.7 from `market.py`, so the home team wins with
  probability exactly p. Margins are rounded and never 0 (no simulated ties).
  Chalk mode uses the expected margin, so the favorite always wins.
- **Hot ratings.** After every simulated game, Elo and the last-5 stats update
  with the same formulas as real games. Turnovers cannot be simulated, so
  simulated games add a turnover margin of 0 (its expected value).
- **Seeding (simplified tiebreakers).** Division winners are seeds 1-4, the best
  three other teams seeds 5-7. Ties on record: head-to-head among the tied
  teams, then division record, then conference record, then random. The
  official NFL procedure has more steps and conditions (for example,
  head-to-head counts in multi-team ties only with a sweep).
- **Playoffs.** Seed 1 has a bye (+7 rest days in its divisional game); wild
  card 2v7, 3v6, 4v5; re-seed after each round; higher seed hosts; neutral
  Super Bowl. Playoff games update ratings too.
- **Reproducible.** A seed fully determines a run; `/simulate` links carry it.
- **Performance.** All simulations run at once as numpy arrays, in chunks of
  2,500 to bound memory (about 2 s for 10,000 seasons locally). The Monte
  Carlo never runs inside a request: it starts in a background thread after
  each data refresh, is cached in memory and in `data/cache/`, and pages show
  the previous result (or a "computing" note on a cold start) until it is done.

`python simulate.py` writes `reports/season_odds_2026.csv` with, per team:
current record, projected wins, P(playoffs), P(division), P(top seed),
P(reach Super Bowl) and P(win Super Bowl). `tests/test_simulate.py` covers
reproducibility (same seed, same champion; different seed, different
results), chalk, invariants (14 playoff spots, 8 division winners, 1 champion,
7 distinct teams per bracket), each tiebreaker on hand-built leagues, and the
in-season Elo update; each test was checked against a deliberately broken
engine. `tests/test_pages.py` renders every page.

The Season page also shows an **average bracket** (`simulate.average_bracket`)
that summarises all 10,000 seasons: each division goes to its most frequent
winner, the three most frequent wild-card teams fill seeds 5-7, and within each
group seeds are assigned to maximise how often each team held its exact seed.
In every game, the team that reached the next round in more simulations
advances. Each team's percentage counts all its paths to that round, so the
two teams in a game do not sum to 100%; the page also shows how often the
displayed Super Bowl matchup actually occurred. The deterministic "every
favorite wins" bracket is the chalk mode of `/simulate`.

**Data refresh.** Render's disk is ephemeral. Seasons 2002-2025 are committed;
2026 is downloaded at startup and again whenever the cached copy is older than
6 hours (`data/fetch.py`), and the in-memory ratings are rebuilt on the same
schedule (`live.snapshot`). If a refresh fails (e.g. GitHub is unreachable)
the last cached copy is served instead of crashing. `tests/test_refresh.py`
covers these rules.

## Limitations

- No injuries, quarterback changes or weather.
- Small playoff sample (13 games per season).
- Simplified simulator tiebreakers; turnovers are not simulated.
- Elo home-field and training-window choices are within validation noise.
- The neutral-site effect is poorly estimated (51 training games); neutral
  predictions are symmetrized at prediction time, see above.
