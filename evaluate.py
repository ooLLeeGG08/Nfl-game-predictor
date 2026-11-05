"""Evaluate the model against baselines, plot calibration, run experiments.

    python evaluate.py               # validation (2024): metrics, plots, experiments
    python evaluate.py --final-test  # the ONE test-season (2025) evaluation

Baselines
---------
* Home: always pick the home team. Its probability is the home-win rate in the
  training seasons, so log loss / Brier are defined for it too.
* Vegas: vig-free probability from the moneylines when present; otherwise the
  spread mapped through a normal curve, P(home) = Phi(spread / sigma), with sigma
  fitted on the training seasons. Pick'em lines (spread 0) count as 0.5. Games
  with no line at all are excluded from the Vegas comparison only.

A probability of exactly 0.5 is a coin flip, so it earns half credit in accuracy.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, log_loss

import market
from data.fetch import TEST_SEASON, VALID_SEASON
from features import MODEL_FEATURES, EloParams, build_features
from paths import CONFIG_PATH, MODEL_PATH, REPORTS, STATIC_DIR
from train import TRAIN_SEASONS, fit, load_raw, model_rows, valid_logloss

log = logging.getLogger(__name__)

TEST_MARKER = REPORTS / ".test_run_done"
STATIC_PLOTS = STATIC_DIR / "plots"
PRED_COLS = ["game_id", "season", "game_type", "home_win", "p_model", "p_home", "p_vegas"]

# Light: reference palette from the dataviz skill (README). Dark: the app's
# monochrome palette, where model vs Vegas is told apart by marker shape, line
# style and the legend, not by color alone.
THEMES = {
    "light": {"surface": "#fcfcfb", "ink": "#0b0b0b", "muted": "#52514e", "grid": "#e4e3df",
              "model": "#2a78d6", "vegas": "#eb6834", "font": "sans-serif", "suffix": ""},
    "dark": {"surface": "#000000", "ink": "#F2F2F2", "muted": "#7A7A7A", "grid": "#2A2A2A",
             "model": "#F2F2F2", "vegas": "#7A7A7A", "font": "monospace", "suffix": "_dark"},
}


# ---------------------------------------------------------------- baselines --

def fit_spread_sigma(df: pd.DataFrame) -> float:
    return market.fit_spread_sigma(df, TRAIN_SEASONS)


vegas_prob = market.vegas_prob


# ------------------------------------------------------------------ metrics --

def metrics(y, p) -> dict:
    y, p = np.asarray(y, float), np.asarray(p, float)
    correct = np.where(p == 0.5, 0.5, (p > 0.5) == (y == 1))
    return {
        "n": len(y),
        "accuracy": correct.mean(),
        "log_loss": log_loss(y, p, labels=[0, 1]),
        "brier": brier_score_loss(y, p),
    }


def evaluate_season(df: pd.DataFrame, pipe, season: int, sigma: float, home_rate: float):
    """Metrics table for one season (regular season + playoffs, ties excluded)."""
    games = model_rows(df, [season], regular_only=False).copy()
    games["p_model"] = pipe.predict_proba(games[MODEL_FEATURES])[:, 1]
    games["p_home"] = home_rate
    games["p_vegas"] = vegas_prob(games, sigma)
    lined = games[games.p_vegas.notna()]

    rows = [
        {"method": "Model", **metrics(games.home_win, games.p_model)},
        {"method": "Always home", **metrics(games.home_win, games.p_home)},
        {"method": "Vegas", **metrics(lined.home_win, lined.p_vegas)},
        {"method": "Model (games with a line)", **metrics(lined.home_win, lined.p_model)},
    ]
    po = games[games.game_type != "REG"]
    if len(po):
        rows.append({"method": "Model (playoffs only)", **metrics(po.home_win, po.p_model)})
    info = {
        "season": season,
        "ties_excluded": int(df[(df.season == season) & df.is_tie].shape[0]),
        "missing_lines": int(games.p_vegas.isna().sum()),
        "pickem_games": int((games.spread_line == 0).sum()),
        "vegas_from_moneyline": int((games.home_moneyline.notna() & games.away_moneyline.notna()).sum()),
    }
    return pd.DataFrame(rows), games, info


# -------------------------------------------------------------------- plots --

def _style(ax, th):
    ax.set_facecolor(th["surface"])
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(th["muted"])
    ax.tick_params(colors=th["muted"], labelsize=9)
    ax.grid(color=th["grid"], linewidth=0.8)
    ax.set_axisbelow(True)


def plot_calibration(games: pd.DataFrame, season: int, path, theme: str = "light") -> None:
    th = THEMES[theme]
    with plt.rc_context({"font.family": th["font"]}):
        fig, ax = plt.subplots(figsize=(6, 5.5), facecolor=th["surface"])
        _style(ax, th)
        ax.plot([0, 1], [0, 1], color=th["muted"], linestyle=":", linewidth=1, label="Perfect calibration")
        series = (("p_model", "Model", th["model"], "s", "-"), ("p_vegas", "Vegas", th["vegas"], "o", "--"))
        for col, name, color, marker, ls in series:
            g = games.dropna(subset=[col])
            frac, mean_pred = calibration_curve(g.home_win, g[col], n_bins=10, strategy="uniform")
            ax.plot(mean_pred, frac, marker=marker, markersize=7, linewidth=2, linestyle=ls, color=color,
                    label=name, markeredgecolor=th["surface"], markeredgewidth=1.5)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Predicted home win probability (10 bins)", color=th["ink"])
        ax.set_ylabel("Observed home win rate", color=th["ink"])
        ax.set_title(f"Calibration, {season} season (n={len(games)})", color=th["ink"], loc="left", fontsize=12)
        ax.legend(frameon=False, labelcolor=th["ink"], fontsize=9, loc="upper left")
        fig.tight_layout()
        fig.savefig(path, dpi=150, facecolor=th["surface"])
        plt.close(fig)


def coefficients(pipe) -> pd.Series:
    """Coefficients on standardized features: log-odds change per 1 SD of each feature."""
    return pd.Series(pipe.named_steps["logreg"].coef_[0], index=MODEL_FEATURES).sort_values()


def plot_importance(coefs: pd.Series, path, theme: str = "light") -> None:
    th = THEMES[theme]
    with plt.rc_context({"font.family": th["font"]}):
        fig, ax = plt.subplots(figsize=(6.5, 3.6), facecolor=th["surface"])
        _style(ax, th)
        ax.grid(axis="y", visible=False)
        ax.barh(coefs.index, coefs.values, color=th["model"], height=0.6)
        ax.axvline(0, color=th["muted"], linewidth=1)
        for i, v in enumerate(coefs.values):
            ax.text(v + (0.01 if v >= 0 else -0.01), i, f"{v:+.3f}", va="center",
                    ha="left" if v >= 0 else "right", color=th["ink"], fontsize=9)
        lo, hi = coefs.min(), coefs.max()
        pad = 0.15 * (hi - min(lo, 0))
        ax.set_xlim(min(lo, 0) - pad, hi + pad)
        ax.set_xlabel("Standardized coefficient (log-odds per 1 SD, home-minus-away)", color=th["ink"])
        ax.set_title("Feature importance", color=th["ink"], loc="left", fontsize=12)
        fig.tight_layout()
        fig.savefig(path, dpi=150, facecolor=th["surface"])
        plt.close(fig)


def save_plots(games: pd.DataFrame, season: int, tag: str, pipe) -> None:
    """Light plots for the README (reports/), dark plots for the app (static/plots/)."""
    STATIC_PLOTS.mkdir(parents=True, exist_ok=True)
    coefs = coefficients(pipe)
    for theme, folder in (("light", REPORTS), ("dark", STATIC_PLOTS)):
        sfx = THEMES[theme]["suffix"]
        plot_calibration(games, season, folder / f"calibration_{tag}{sfx}.png", theme)
        plot_importance(coefs, folder / f"feature_importance{sfx}.png", theme)


# -------------------------------------------------------------- experiments --

def run_experiments(sched, stats, params: EloParams) -> pd.DataFrame:
    """All on 2024 validation. One factor varied at a time; the rest stay at the tuned config."""
    rows = []
    df = build_features(sched, stats, params)
    for n in (5, 10, 18):
        seasons = range(VALID_SEASON - n, VALID_SEASON)
        pipe = fit(df, seasons)
        val = model_rows(df, [VALID_SEASON], regular_only=False)
        m = metrics(val.home_win, pipe.predict_proba(val[MODEL_FEATURES])[:, 1])
        rows.append({"experiment": "training window", "setting": f"last {n} seasons ({min(seasons)}-{max(seasons)})", **m})
    for name, r in (("full carryover", 0.0), ("1/3 regression", 1 / 3), ("reset to 1500", 1.0)):
        p = EloParams(k=params.k, hfa=params.hfa, mov=params.mov, regress=r)
        d = build_features(sched, stats, p)
        val = model_rows(d, [VALID_SEASON], regular_only=False)
        m = metrics(val.home_win, fit(d).predict_proba(val[MODEL_FEATURES])[:, 1])
        rows.append({"experiment": "Elo carryover", "setting": name, **m})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------- main --

def load_model():
    cfg = json.loads(CONFIG_PATH.read_text())
    return joblib.load(MODEL_PATH), cfg, EloParams(**cfg["elo"])


def home_rate(df: pd.DataFrame) -> float:
    return float(model_rows(df, TRAIN_SEASONS, regular_only=True).home_win.mean())


def run(season: int, allow_test: bool, with_experiments: bool) -> None:
    pipe, cfg, params = load_model()
    sched, stats = load_raw(last_season=season, allow_test=allow_test)
    df = build_features(sched, stats, params)
    sigma = fit_spread_sigma(df)
    table, games, info = evaluate_season(df, pipe, season, sigma, home_rate(df))
    info.update({"spread_sigma": round(sigma, 2), "elo": cfg["elo"], "train_seasons": cfg["train_seasons"]})

    tag = "test" if season == TEST_SEASON else "validation"
    REPORTS.mkdir(exist_ok=True)
    table.to_csv(REPORTS / f"metrics_{tag}.csv", index=False)
    (REPORTS / f"info_{tag}.json").write_text(json.dumps(info, indent=2))
    games[PRED_COLS].to_csv(REPORTS / f"predictions_{tag}.csv", index=False)
    coefs = coefficients(pipe)
    coefs.rename("coef").to_csv(REPORTS / "feature_importance.csv")
    save_plots(games, season, tag, pipe)

    print(f"\n=== {tag.upper()} ({season}) ===")
    print(table.round(4).to_string(index=False))
    print(json.dumps(info, indent=2))
    print("\nStandardized coefficients:\n" + coefs.round(4).to_string())

    if with_experiments:
        exp = run_experiments(sched, stats, params)
        exp.to_csv(REPORTS / "experiments.csv", index=False)
        print("\nExperiments (validation):\n" + exp.round(4).to_string(index=False))

    if season == TEST_SEASON:
        model_acc = table.set_index("method").loc["Model (games with a line)", "accuracy"]
        vegas_acc = table.set_index("method").loc["Vegas", "accuracy"]
        if model_acc > vegas_acc:
            print(f"\n!!! Model accuracy {model_acc:.4f} beats Vegas {vegas_acc:.4f}: "
                  "treat as a PROBABLE LEAK and investigate before reporting.")


def replot() -> None:
    """Redraw all plots from saved predictions; computes no metrics.

    The test run predates predictions_test.csv, so it is regenerated once from
    the frozen model. Allowed only after the single test run: the config can
    no longer change, so this cannot influence any modelling decision.
    """
    pipe, _, params = load_model()
    for tag, season in (("validation", VALID_SEASON), ("test", TEST_SEASON)):
        path = REPORTS / f"predictions_{tag}.csv"
        if not path.exists():
            if tag == "test" and not TEST_MARKER.exists():
                continue
            sched, stats = load_raw(last_season=season, allow_test=(tag == "test"))
            df = build_features(sched, stats, params)
            games = model_rows(df, [season], regular_only=False).copy()
            games["p_model"] = pipe.predict_proba(games[MODEL_FEATURES])[:, 1]
            games["p_home"] = home_rate(df)
            games["p_vegas"] = vegas_prob(games, fit_spread_sigma(df))
            games[PRED_COLS].to_csv(path, index=False)
        save_plots(pd.read_csv(path), season, tag, pipe)
        print(f"replotted {tag}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--final-test", action="store_true", help=f"evaluate on {TEST_SEASON} (allowed once)")
    ap.add_argument("--replot", action="store_true", help="redraw plots from saved predictions")
    a = ap.parse_args()
    if a.replot:
        replot()
        return
    if not a.final_test:
        run(VALID_SEASON, allow_test=False, with_experiments=True)
        return
    if TEST_MARKER.exists():
        sys.exit(f"Test set already evaluated ({TEST_MARKER.read_text().strip()}). Refusing to run again.")
    run(TEST_SEASON, allow_test=True, with_experiments=False)
    TEST_MARKER.write_text(f"run at {pd.Timestamp.now():%Y-%m-%d %H:%M} with {CONFIG_PATH.read_text()}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
