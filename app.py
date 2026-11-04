"""Flask app. Loads the saved model via live.py; never retrains.

Local:   gunicorn app:app --workers 1 --threads 4
Render:  see render.yaml (same settings; do not add --preload, see the comment there)
"""
from __future__ import annotations

import os
import secrets
from datetime import datetime, timezone

from flask import Flask, abort, redirect, render_template, request, url_for

import live
import simulate
import ui
from data.fetch import CURRENT_SEASON
from teams import NAMES, TEAM_COLORS

app = Flask(__name__)

TEAM_CHOICES = sorted(NAMES)


@app.context_processor
def site_globals():
    built = datetime.fromtimestamp(live.snapshot().built_at, tz=timezone.utc)
    return {
        "icon": ui.icon,
        "model_version": ui.model_version(),
        "refreshed": f"{built:%Y-%m-%d %H:%M} UTC",
        "repo_url": os.environ.get("REPO_URL"),  # optional, set in Render's dashboard
        "current_season": CURRENT_SEASON,
    }


@app.route("/")
def home():
    home_team = request.args.get("home", "KC")
    away_team = request.args.get("away", "BUF")
    neutral = request.args.get("neutral") == "1"
    error = result = None
    if home_team not in NAMES or away_team not in NAMES:
        error = "Unknown team."
    elif home_team == away_team:
        error = "Pick two different teams."
    else:
        result = live.predict_matchup(home_team, away_team, neutral)

    snap = live.snapshot()
    test = ui.reports()["test"]
    built = datetime.fromtimestamp(snap.built_at, tz=timezone.utc)
    stats = {
        "accuracy": ui.metric(test, "Model", "accuracy"),
        "log_loss": ui.metric(test, "Model", "log_loss"),
        "vegas_log_loss": ui.metric(test, "Vegas", "log_loss"),
        "games": live.games_modeled(),
        "refresh_time": f"{built:%H:%M}",
        "refresh_date": f"{built:%b %-d} UTC",
    }
    return render_template(
        "home.html",
        teams=TEAM_CHOICES, names=NAMES, colors=TEAM_COLORS,
        home=home_team, away=away_team, neutral=neutral, result=result, error=error,
        season=CURRENT_SEASON, week=snap.as_of_week, stats=stats, next_game=live.next_game(),
    )


@app.route("/week")
def week():
    return render_template("week.html", w=live.week_games(), colors=TEAM_COLORS, season=CURRENT_SEASON)


@app.route("/game/<game_id>")
def game(game_id):
    g = live.game_detail(game_id)
    if g is None:
        abort(404)
    return render_template("game.html", g=g, colors=TEAM_COLORS)


@app.route("/season")
def season():
    odds = live.season_odds()
    if odds is None:  # first Monte Carlo after a cold start is still running in the background
        return render_template("season.html", teams=None, season=CURRENT_SEASON, n_sims=simulate.N_SIMS)
    odds = odds.sort_values(["p_sb", "p_playoffs"], ascending=False)
    avg = simulate.average_season(odds)
    return render_template(
        "season.html", teams=odds.to_dict("records"), colors=TEAM_COLORS, names=NAMES, season=CURRENT_SEASON,
        games_left=odds.attrs["games_left"], n_sims=simulate.N_SIMS, s=avg, champ_odds=avg["p_champion"],
        stale=not live.season_odds_ready(),
    )


MAX_SEED = 999_999


@app.route("/simulate")
def simulate_page():
    mode = request.args.get("mode")
    if mode == "average":
        odds = live.season_odds()  # cached Monte Carlo; never computed in the request
        if odds is None:
            return render_template("soon.html", tag="Simulate", heading="The average season",
                                   note=f"The {simulate.N_SIMS:,}-season simulation is still running. Reload in a minute.")
        s = simulate.average_season(odds)
        return render_template("simulate.html", s=s, colors=TEAM_COLORS, names=NAMES, season=CURRENT_SEASON,
                               champ_odds=s["p_champion"], n_sims=simulate.N_SIMS,
                               new_seed=secrets.randbelow(MAX_SEED + 1))
    chalk = mode == "chalk"
    seed = request.args.get("seed", type=int)
    if not chalk and (seed is None or not 0 <= seed <= MAX_SEED):
        return redirect(url_for("simulate_page", seed=secrets.randbelow(MAX_SEED + 1)))
    s = live.single_season(0 if chalk else seed, chalk)
    odds = live.season_odds()
    champ_odds = None if odds is None else float(odds.set_index("team").loc[s["champion"], "p_sb"])
    return render_template(
        "simulate.html", s=s, colors=TEAM_COLORS, names=NAMES, season=CURRENT_SEASON,
        champ_odds=champ_odds, n_sims=simulate.N_SIMS, new_seed=secrets.randbelow(MAX_SEED + 1),
    )


@app.route("/about")
def about():
    return render_template("about.html", r=ui.reports())


@app.route("/health")
def health():
    return {"ok": True, "as_of_week": live.snapshot().as_of_week, "model": ui.model_version()}


# Build the snapshot at import so the first visitor doesn't wait for it, and
# start the first Monte Carlo in the background (never inside a request).
live.snapshot()
live.season_odds()
