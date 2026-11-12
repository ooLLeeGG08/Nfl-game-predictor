"""Fit and freeze the v2 model: v1's six features + qb_rating_diff.

    python train_v2.py

Writes models/v2/{pipeline.joblib, model.json, config.json}. v1 (models/*.json,
models/pipeline.joblib) is not touched.

Training rows: regular-season games 2009-2024. 2009 is the first season with
injury reports, so the expected starter is defined the same way on every row;
2025 and 2026 are kept out for the holdout (evaluate_v2.py --holdout).
Elo settings are v1's (models/config.json); nothing is re-tuned.
"""
from __future__ import annotations

import json
from datetime import date

import joblib
import pandas as pd

import qb
from data.fetch import load_schedules, load_team_stats
from features import MODEL_FEATURES, EloParams, build_features
from paths import CONFIG_PATH, ROOT
from train import SEED, make_pipeline, model_rows

V2_DIR = ROOT / "models" / "v2"
V2_FEATURES = MODEL_FEATURES + ["qb_rating_diff"]
V2_TRAIN = range(2009, 2025)
FIRST_DATA_SEASON = 2002


def v2_frame(last_season: int, allow_test: bool = False) -> pd.DataFrame:
    """v1 features + qb_rating_diff for every game 2002..last_season."""
    params = EloParams(**json.loads(CONFIG_PATH.read_text())["elo"])
    seasons = range(FIRST_DATA_SEASON, last_season + 1)
    sched = load_schedules(seasons, allow_test=allow_test)
    stats = load_team_stats(seasons, allow_test=allow_test)
    df = build_features(sched, stats, params)
    qbf = qb.to_game_features(qb.build_qb_features(sched, *qb.load_sources(qb.QB_RAW, sched)))
    return df.merge(qbf, on="game_id", how="left")


def export_json(pipe, path) -> None:
    scale, logreg = pipe.named_steps["scale"], pipe.named_steps["logreg"]
    path.write_text(json.dumps({
        "version": "v2",
        "features": V2_FEATURES,
        "mean": scale.mean_.tolist(),
        "scale": scale.scale_.tolist(),
        "coef": logreg.coef_[0].tolist(),
        "intercept": float(logreg.intercept_[0]),
    }, indent=2))


def main() -> None:
    df = v2_frame(max(V2_TRAIN))
    train = model_rows(df, V2_TRAIN, regular_only=True)
    assert train[V2_FEATURES].notna().all().all()
    pipe = make_pipeline().fit(train[V2_FEATURES], train.home_win)

    V2_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, V2_DIR / "pipeline.joblib")
    export_json(pipe, V2_DIR / "model.json")
    v1cfg = json.loads(CONFIG_PATH.read_text())
    (V2_DIR / "config.json").write_text(json.dumps({
        "version": "v2",
        "frozen": str(date.today()),
        "features": V2_FEATURES,
        "train_seasons": [min(V2_TRAIN), max(V2_TRAIN)],
        "train_games": int(len(train)),
        "game_types": "regular season only",
        "elo": v1cfg["elo"],
        "seed": SEED,
        "logistic_regression": {"C": 1.0, "penalty": "l2", "solver": "lbfgs"},
        "qb": {
            "rating": "EPA per play (pass attempts + sacks + carries) in games before kickoff, "
                      "current + previous 2 seasons, shrunk to the league mean of earlier seasons",
            "k_shrink_plays": qb.K_SHRINK,
            "window_seasons": 3,
            "expected_starter": "week 1: depth-chart QB1; else the QB with most plays in the team's previous "
                                "game; if he is Out/Doubtful on the final pre-game injury report, the next "
                                "depth-chart QB who is not Out/Doubtful",
            "injury_status_used": ["Out", "Doubtful"],
        },
        "selection": "rolling origin 2019-2024 (evaluate_v2.py --rolling); kept only features whose "
                     "paired-bootstrap 95% CI of the log-loss difference vs v1 excludes 0",
    }, indent=2))
    print(f"v2 trained on {len(train)} games; wrote {V2_DIR.relative_to(ROOT)}/")
    for f, c in zip(V2_FEATURES, pipe.named_steps["logreg"].coef_[0]):
        print(f"  {f:16s} {c:+.4f}")


if __name__ == "__main__":
    main()
