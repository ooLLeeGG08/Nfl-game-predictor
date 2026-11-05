"""Split, tune Elo on the validation season, fit the model, save it.

Split (time-based, never shuffled):
    2002-2005  Elo warm-up only, never training rows
    2006-2023  training  (regular season only)
    2024       validation (tuning + experiments only, never fitted on)
    2025       test      (touched once, by ``evaluate.py --final-test``)

Run ``python train.py`` to tune, fit and write models/pipeline.joblib.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict
from itertools import product

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from data.fetch import BURN_IN_START, TRAIN_SEASONS, VALID_SEASON, load_schedules, load_team_stats
from features import MODEL_FEATURES, EloParams, build_features
from paths import CONFIG_PATH, MODEL_JSON, MODEL_PATH, REPORTS

log = logging.getLogger(__name__)

SEED = 42

ELO_GRID = {
    "k": [10, 15, 20, 25, 30, 40],
    "hfa": [0, 15, 25, 35, 48, 65],
    "mov": [False, True],
}


def make_pipeline() -> Pipeline:
    # Scaling puts every feature in standard deviations, which (a) helps the
    # solver and (b) makes coefficients comparable as "importance".
    return Pipeline([
        ("scale", StandardScaler()),
        ("logreg", LogisticRegression(random_state=SEED)),
    ])


def load_raw(last_season: int = VALID_SEASON, allow_test: bool = False):
    seasons = range(BURN_IN_START, last_season + 1)
    return load_schedules(seasons, allow_test=allow_test), load_team_stats(seasons, allow_test=allow_test)


def model_rows(df: pd.DataFrame, seasons, regular_only: bool) -> pd.DataFrame:
    """Played, non-tie games from the given seasons."""
    m = df.season.isin(list(seasons)) & df.played & ~df.is_tie
    if regular_only:
        m &= df.game_type == "REG"
    return df[m]


def fit(df: pd.DataFrame, seasons=TRAIN_SEASONS) -> Pipeline:
    train = model_rows(df, seasons, regular_only=True)  # train on regular season only
    return make_pipeline().fit(train[MODEL_FEATURES], train.home_win)


def valid_logloss(df: pd.DataFrame, pipe: Pipeline) -> float:
    val = model_rows(df, [VALID_SEASON], regular_only=False)
    return log_loss(val.home_win, pipe.predict_proba(val[MODEL_FEATURES])[:, 1])


def tune_elo(sched, stats, regress: float = 1 / 3) -> pd.DataFrame:
    """Grid-search Elo settings by the *full model's* log loss on 2024.

    We judge each setting by log loss rather than accuracy: log loss rewards
    good probabilities, not just right picks, and it is far less noisy on a
    ~285-game season.
    """
    results = []
    for k, hfa, mov in product(ELO_GRID["k"], ELO_GRID["hfa"], ELO_GRID["mov"]):
        params = EloParams(k=k, hfa=hfa, mov=mov, regress=regress)
        df = build_features(sched, stats, params)
        results.append({"k": k, "hfa": hfa, "mov": mov, "val_log_loss": valid_logloss(df, fit(df))})
    return pd.DataFrame(results).sort_values("val_log_loss").reset_index(drop=True)


def export_model_json(pipe: Pipeline) -> None:
    """Write the fitted pipeline as plain numbers so the web app needs no scikit-learn.

    A logistic regression after standard scaling is just
        p = 1 / (1 + exp(-(((x - mean) / scale) . coef + intercept)))
    """
    scale, logreg = pipe.named_steps["scale"], pipe.named_steps["logreg"]
    MODEL_JSON.write_text(json.dumps({
        "features": MODEL_FEATURES,
        "mean": scale.mean_.tolist(),
        "scale": scale.scale_.tolist(),
        "coef": logreg.coef_[0].tolist(),
        "intercept": float(logreg.intercept_[0]),
    }, indent=2))


def save(pipe: Pipeline, params: EloParams, train_seasons) -> None:
    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(pipe, MODEL_PATH)
    export_model_json(pipe)
    CONFIG_PATH.write_text(json.dumps({
        "elo": asdict(params),
        "features": MODEL_FEATURES,
        "train_seasons": [min(train_seasons), max(train_seasons)],
        "seed": SEED,
    }, indent=2))


def main() -> None:
    sched, stats = load_raw()
    REPORTS.mkdir(exist_ok=True)

    grid = tune_elo(sched, stats)
    grid.to_csv(REPORTS / "elo_tuning.csv", index=False)
    best = grid.iloc[0]
    params = EloParams(k=float(best.k), hfa=float(best.hfa), mov=bool(best.mov))
    log.info("best Elo on validation: %s (log loss %.4f)", params, best.val_log_loss)

    df = build_features(sched, stats, params)
    pipe = fit(df)
    save(pipe, params, TRAIN_SEASONS)
    log.info("saved %s", MODEL_PATH)


if __name__ == "__main__" and "--export-json" in __import__("sys").argv:
    export_model_json(joblib.load(MODEL_PATH))  # re-export without retraining
    print(f"wrote {MODEL_JSON}")
elif __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
