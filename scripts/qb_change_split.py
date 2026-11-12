"""Post-hoc breakdown: where does v2 differ from v1, games with vs without a QB change?

    python scripts/qb_change_split.py   -> reports/v2_qb_change_split.json

Explains the QB-adjusted tag; it does not change the pre-registered evaluation rule.

"QB change" for a game: either team's expected starter is not its established
starter, the QB who led the team in the most of its previous 4 games (ties: the
more recent one). Both are known before kickoff.

    2019-2024  rolling origin (refit on 2009..Y-1 each season), as in evaluate_v2.py --rolling
    2025       frozen models (previously seen), as in reports/predictions_v2_holdout.csv
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import qb  # noqa: E402
from data.fetch import load_schedules  # noqa: E402
from evaluate_v2 import ROLL_START, ROLL_YEARS, logloss_i, paired_ci  # noqa: E402
from features import MODEL_FEATURES  # noqa: E402
from paths import REPORTS  # noqa: E402
from train import make_pipeline, model_rows  # noqa: E402
from train_v2 import V2_FEATURES, v2_frame  # noqa: E402

FLAG = 0.05


def established_change(feats: pd.DataFrame) -> pd.Series:
    """game_id -> True if either team's expected starter differs from its established starter."""
    out = {}
    for _, g in feats.sort_values(["team", "kick", "game_id"]).groupby("team"):
        leads = g.lead_prev.shift(-1).tolist()  # lead of game i = lead_prev of game i+1
        rows = g.to_dict("records")
        for i, r in enumerate(rows):
            last4 = [q for q in leads[max(0, i - 4):i] if isinstance(q, str)]
            est = max(set(last4), key=lambda q: (last4.count(q), -last4[::-1].index(q))) if last4 else None
            changed = isinstance(r["expected"], str) and est is not None and r["expected"] != est
            out[r["game_id"]] = out.get(r["game_id"], False) or changed
    return pd.Series(out)


def block(P: pd.DataFrame, v1: str, v2: str) -> dict:
    y = P.home_win.to_numpy(float)
    d = logloss_i(y, P[v2]) - logloss_i(y, P[v1])
    ch = P.changed.to_numpy(bool)
    flagged = (P[v2] - P[v1]).abs().to_numpy() > FLAG
    res = {"games": int(len(P)), "v2_minus_v1_all": paired_ci(d)}
    for name, m in (("qb_change", ch), ("no_qb_change", ~ch),
                    ("flagged_qb_change", flagged & ch), ("flagged_no_qb_change", flagged & ~ch)):
        res[name] = {**paired_ci(d[m]), "share_of_total_difference": float(d[m].sum() / d.sum())}
    return res


def main() -> None:
    df = v2_frame(2025, allow_test=True)
    sched = load_schedules(range(2002, 2026), allow_test=True)
    changed = established_change(qb.build_qb_features(sched, *qb.load_sources(qb.QB_RAW, sched)))

    parts = []
    for Y in ROLL_YEARS:
        tr = model_rows(df, range(ROLL_START, Y), regular_only=True)
        te = model_rows(df, [Y], regular_only=False).copy()
        te["v1"] = make_pipeline().fit(tr[MODEL_FEATURES], tr.home_win).predict_proba(te[MODEL_FEATURES])[:, 1]
        te["v2"] = make_pipeline().fit(tr[V2_FEATURES], tr.home_win).predict_proba(te[V2_FEATURES])[:, 1]
        parts.append(te[["game_id", "home_win", "v1", "v2"]])
    R = pd.concat(parts, ignore_index=True)
    H = pd.read_csv(REPORTS / "predictions_v2_holdout.csv")
    H = H[H.season == 2025]
    for P in (R, H):
        P["changed"] = P.game_id.map(changed).fillna(False).astype(bool)

    out = {"definition": __doc__.split("\n\n")[2].strip(), "flag_threshold": FLAG,
           "rolling_2019_2024": block(R, "v1", "v2"), "season_2025_previously_seen": block(H, "v1", "v2")}
    (REPORTS / "v2_qb_change_split.json").write_text(json.dumps(out, indent=2))
    for k in ("rolling_2019_2024", "season_2025_previously_seen"):
        print(k)
        for name in ("qb_change", "no_qb_change", "flagged_qb_change", "flagged_no_qb_change"):
            b = out[k][name]
            print(f"  {name:22s} n={b['n']:5d}  {b['mean']:+.4f} [{b['ci95'][0]:+.4f}, {b['ci95'][1]:+.4f}]  share {b['share_of_total_difference']:.0%}")


if __name__ == "__main__":
    main()
