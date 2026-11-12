"""Run from the repo root: python scripts/explain_model.py

Numbers for the README section "How the model calculates a probability".
Read-only with respect to the model: it loads the frozen models/pipeline.joblib
and never refits, retrains or overwrites it, and it changes no reported metric.

Writes
    reports/model_explained.json   every number the README section quotes
    reports/logit_curve.png        predicted p vs linear score z, 2025 test games

* Coefficients, intercept and scaler come from the saved scikit-learn pipeline.
* scikit-learn reports no standard errors, so SE, z and p-values come from an
  unpenalized statsmodels Logit on the same standardized 2006-2023 rows. The
  sklearn fit has a small L2 penalty (C = 1.0), so its coefficients differ
  slightly from the statsmodels ones; the README shows the sklearn ones.
* The 2025 test season was evaluated once (evaluate.py --final-test). Loading
  it here only re-reads that season for the worked example and the plot, the
  same way the web app does.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import statsmodels.api as sm  # noqa: E402
from scipy.stats import chi2  # noqa: E402

import live  # noqa: E402  (the app's own prediction function)
import market  # noqa: E402
from data.fetch import TEST_SEASON  # noqa: E402
from evaluate import THEMES, _style  # noqa: E402
from features import MODEL_FEATURES, EloParams, build_features  # noqa: E402
from paths import CONFIG_PATH, MODEL_PATH, REPORTS  # noqa: E402
from train import TRAIN_SEASONS, load_raw, model_rows  # noqa: E402

OUT_JSON = REPORTS / "model_explained.json"
OUT_PLOT = REPORTS / "logit_curve.png"

MEANING = {
    "elo_diff": "Elo rating difference",
    "pd_diff": "last-5 point differential difference",
    "to_diff": "last-5 turnover margin difference",
    "rest_diff": "days of rest difference",
    "div_game": "division game (0/1)",
    "neutral_site": "neutral site (0/1)",
}


def log_lik(y: np.ndarray, p: np.ndarray) -> float:
    """sum over home wins of log(p) + sum over home losses of log(1 - p)."""
    return float(np.sum(np.log(p[y == 1])) + np.sum(np.log(1 - p[y == 0])))


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def main() -> None:
    pipe = joblib.load(MODEL_PATH)
    scaler, logreg = pipe.named_steps["scale"], pipe.named_steps["logreg"]
    params = EloParams(**json.loads(CONFIG_PATH.read_text())["elo"])
    mean, sd = scaler.mean_, scaler.scale_
    w, b0 = logreg.coef_[0], float(logreg.intercept_[0])

    # The app scores games from models/model.json; it must be the same model.
    mj = json.loads((ROOT / "models" / "model.json").read_text())
    assert np.allclose(mj["coef"], w, rtol=0, atol=1e-15) and mj["intercept"] == b0

    sched, stats = load_raw(last_season=TEST_SEASON, allow_test=True)
    df = build_features(sched, stats, params)
    df["gametime"] = df.game_id.map(sched.set_index("game_id").gametime)  # kickoff order, as in live.py

    # ---- training data: must be exactly the rows the scaler was fitted on ----
    train = model_rows(df, TRAIN_SEASONS, regular_only=True)
    X, y = train[MODEL_FEATURES].to_numpy(float), train.home_win.to_numpy(int)
    n = len(y)
    assert n == scaler.n_samples_seen_, (n, scaler.n_samples_seen_)
    assert np.allclose(X.mean(0), mean, rtol=0, atol=1e-9) and np.allclose(X.std(0), sd, rtol=0, atol=1e-9)
    Xs = scaler.transform(train[MODEL_FEATURES])

    # ---- the model (sklearn) on its training data ----
    p_fit = pipe.predict_proba(train[MODEL_FEATURES])[:, 1]
    ll_fit = log_lik(y, p_fit)
    home_rate = float(y.mean())
    ll_null = log_lik(y, np.full(n, home_rate))
    k = len(MODEL_FEATURES)
    lr_chi2 = 2 * (ll_fit - ll_null)

    # ---- unpenalized fit, used only for standard errors / z / p ----
    sm_fit = sm.Logit(y, sm.add_constant(Xs)).fit(disp=0, method="newton", maxiter=100)
    assert sm_fit.mle_retvals["converged"]
    se, z_sm, p_sm = sm_fit.bse, sm_fit.tvalues, sm_fit.pvalues
    names = ["intercept", *MODEL_FEATURES]

    coefficients = []
    for i, name in enumerate(names):
        wi = b0 if i == 0 else float(w[i - 1])
        coefficients.append({
            "term": name,
            "meaning": "baseline log-odds (all features at their training mean)" if i == 0 else MEANING[name],
            "one_sd": None if i == 0 else float(sd[i - 1]),
            "train_mean": None if i == 0 else float(mean[i - 1]),
            "coef_sklearn": wi,
            "odds_ratio": float(np.exp(wi)),
            "coef_unpenalized": float(sm_fit.params[i]),
            "se": float(se[i]), "z": float(z_sm[i]), "p_value": float(p_sm[i]),
            "wald_chi2": float(z_sm[i] ** 2),
        })

    # Equal teams, same rest, not a division game, not neutral: every raw input is 0.
    z_equal = float(b0 + np.sum(w * (0 - mean) / sd))

    # ---- worked example: first game of week 10, 2025 ----
    wk10 = df[(df.season == TEST_SEASON) & (df.week == 10)].sort_values(["gameday", "gametime", "game_id"])
    g = wk10.iloc[0]
    x_raw = g[MODEL_FEATURES].to_numpy(float)
    x_std = (x_raw - mean) / sd
    contrib = w * x_std
    z_ex = float(b0 + contrib.sum())
    p_ex = float(sigmoid(z_ex))
    p_app = float(live.home_win_prob(wk10.iloc[[0]])[0])  # the app's function (incl. neutral-site rule)
    p_pipe = float(pipe.predict_proba(wk10.iloc[[0]][MODEL_FEATURES])[:, 1][0])
    sigma = market.fit_spread_sigma(df, TRAIN_SEASONS)
    p_vegas = float(market.vegas_prob(wk10.iloc[[0]], sigma).iloc[0])
    assert abs(p_ex - p_app) < 1e-12 and abs(p_ex - p_pipe) < 1e-12

    example = {
        "game_id": g.game_id, "gameday": str(g.gameday), "home": g.home_team, "away": g.away_team,
        "home_score": int(g.home_score), "away_score": int(g.away_score), "home_win": int(g.home_win),
        "neutral_site": int(g.neutral_site),
        "inputs": {"home_elo": float(g.home_elo), "away_elo": float(g.away_elo),
                   "home_pd5": float(g.home_pd5), "away_pd5": float(g.away_pd5),
                   "home_to5": float(g.home_to5), "away_to5": float(g.away_to5),
                   "home_rest": float(g.home_rest), "away_rest": float(g.away_rest)},
        "features": [{"feature": f, "raw": float(r), "mean": float(m), "sd": float(s), "scaled": float(xs),
                      "coef": float(wi), "contribution": float(c)}
                     for f, r, m, s, xs, wi, c in zip(MODEL_FEATURES, x_raw, mean, sd, x_std, w, contrib)],
        "intercept": b0, "z": z_ex, "p_home": p_ex, "p_app": p_app,
        "spread_line": None if pd.isna(g.spread_line) else float(g.spread_line),
        "home_moneyline": None if pd.isna(g.home_moneyline) else float(g.home_moneyline),
        "away_moneyline": None if pd.isna(g.away_moneyline) else float(g.away_moneyline),
        "p_vegas": p_vegas,
    }

    # ---- test season: log loss = -LL / n, using the frozen predictions ----
    test = pd.read_csv(REPORTS / "predictions_test.csv")
    yt, pt = test.home_win.to_numpy(int), test.p_model.to_numpy(float)
    ll_test = log_lik(yt, pt)
    # Same McFadden idea on the test season, null = the training home-win rate
    # (the "always home" baseline), so model and Vegas can be compared.
    ll_test_null = log_lik(yt, test.p_home.to_numpy(float))
    ll_test_vegas = log_lik(yt, test.p_vegas.to_numpy(float))

    out = {
        "source": "models/pipeline.joblib (frozen; not refitted)",
        "train": {"seasons": [min(TRAIN_SEASONS), max(TRAIN_SEASONS)], "games": n,
                  "home_wins": int(y.sum()), "home_win_rate": home_rate},
        "sklearn": {"C": float(logreg.C), "penalty": "l2", "solver": logreg.solver},
        "intercept": b0,
        "intercept_prob": float(sigmoid(b0)),
        "equal_teams": {"z": z_equal, "p": float(sigmoid(z_equal)),
                        "note": "all raw diffs 0, not a division game, not neutral"},
        "coefficients": coefficients,
        "fit": {
            "ll_fit": ll_fit, "ll_null": ll_null,
            "ll_null_check": float(n * (home_rate * np.log(home_rate) + (1 - home_rate) * np.log(1 - home_rate))),
            "ll_fit_unpenalized": float(sm_fit.llf),
            "mcfadden_r2": (ll_null - ll_fit) / ll_null,
            "lr_chi2": lr_chi2, "df": k, "lr_p_value": float(chi2.sf(lr_chi2, k)),
        },
        "example": example,
        "test": {"games": len(yt), "ll": ll_test, "log_loss": -ll_test / len(yt),
                 "ll_null": ll_test_null, "ll_vegas": ll_test_vegas,
                 "mcfadden_r2_model": (ll_test_null - ll_test) / ll_test_null,
                 "mcfadden_r2_vegas": (ll_test_null - ll_test_vegas) / ll_test_null},
    }
    OUT_JSON.write_text(json.dumps(out, indent=2))

    # ---- plot: p vs z on the test games (frozen pipeline, as evaluated) ----
    tg = model_rows(df, [TEST_SEASON], regular_only=False)
    zt = pipe.decision_function(tg[MODEL_FEATURES])
    th = THEMES["dark"]
    rng = np.random.default_rng(0)
    with plt.rc_context({"font.family": th["font"]}):
        fig, ax = plt.subplots(figsize=(6.5, 4.2), facecolor=th["surface"])
        _style(ax, th)
        grid = np.linspace(min(-3.5, zt.min() - 0.2), max(3.5, zt.max() + 0.2), 400)
        ax.plot(grid, sigmoid(grid), color=th["model"], linewidth=2, label="p = 1 / (1 + e^-z)", zorder=3)
        yj = tg.home_win.to_numpy(float) + rng.uniform(-0.025, 0.025, len(tg))
        ax.scatter(zt, yj, s=14, color=th["muted"], alpha=0.8, linewidths=0,
                   label=f"Actual result (1 = home win), n={len(tg)}", zorder=2)
        ax.axhline(0.5, color=th["muted"], linestyle=":", linewidth=1)
        ax.axvline(0, color=th["muted"], linestyle=":", linewidth=1)
        ax.scatter([z_ex], [p_ex], s=60, facecolor=th["surface"], edgecolor=th["ink"], linewidth=1.8, zorder=4)
        ax.annotate(f"{example['away']} @ {example['home']}: z = {z_ex:.3f}, p = {p_ex:.3f}", (z_ex, p_ex),
                    xytext=(-12, 10), textcoords="offset points", ha="right", color=th["ink"], fontsize=9)
        ax.set_ylim(-0.08, 1.08)
        ax.set_xlabel("Linear score z = log-odds of a home win", color=th["ink"])
        ax.set_ylabel("Predicted home win probability", color=th["ink"])
        ax.set_title(f"From line to S-curve, {TEST_SEASON} test games", color=th["ink"], loc="left", fontsize=12)
        ax.legend(frameon=False, labelcolor=th["ink"], fontsize=9, loc="center left", bbox_to_anchor=(0.0, 0.72))
        fig.tight_layout()
        fig.savefig(OUT_PLOT, dpi=150, facecolor=th["surface"])
        plt.close(fig)

    print(json.dumps({k: out[k] for k in ("train", "intercept", "intercept_prob", "equal_teams", "fit", "test")}, indent=2))
    print(f"wrote {OUT_JSON.relative_to(ROOT)} and {OUT_PLOT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
