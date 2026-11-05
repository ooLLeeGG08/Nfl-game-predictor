"""Small presentation helpers: pixel icons and the data shown on the site."""
from __future__ import annotations

import hashlib
import json
from functools import lru_cache

import pandas as pd
from markupsafe import Markup

from paths import CONFIG_PATH, MODEL_JSON, REPORTS

# 8x8 pixel icons, drawn in the same style as the field scene.
_ICONS = {
    "target": ["..####..", ".#....#.", "#..##..#", "#.#..#.#", "#.#..#.#", "#..##..#", ".#....#.", "..####.."],
    "chart": ["#.......", "#.....#.", "#....#..", "#.#.#...", "#..#....", "#.......", "########", "........"],
    "stack": ["########", "#......#", "########", "#......#", "########", "#......#", "########", "........"],
    "clock": ["..####..", ".#.#..#.", "#..#...#", "#..###.#", "#......#", "#......#", ".#....#.", "..####.."],
    # nav
    "home": ["...##...", "..#..#..", ".#....#.", "#......#", ".#....#.", ".#.##.#.", ".#.##.#.", ".######."],
    "calendar": [".#....#.", "########", "#......#", "########", "#.#.#..#", "#......#", "#.#.#..#", "########"],
    "trophy": ["########", "#.####.#", "#.####.#", ".#.##.#.", "...##...", "...##...", "..####..", ".######."],
    "dice": ["########", "#......#", "#.#..#.#", "#......#", "#...#..#", "#......#", "#.#..#.#", "########"],
    "info": ["..####..", ".#.##.#.", "#......#", "#..##..#", "#..##..#", "#..##..#", ".#....#.", "..####.."],
    "arrow": ["........", "....#...", ".....#..", "#######.", ".....#..", "....#...", "........", "........"],
    # model inputs
    "plusminus": ["...#....", "...#....", ".#####..", "...#....", "...#....", "........", ".#####..", "........"],
    "swap": ["..#.....", ".##.....", "#######.", ".##.....", ".....##.", "#######.", ".....##.", ".....#.."],
    "moon": ["...###..", "..##....", ".##.....", ".##.....", ".##.....", "..##....", "...###..", "........"],
    "grid": ["###.###.", "#.#.#.#.", "###.###.", "........", "###.###.", "#.#.#.#.", "###.###.", "........"],
    "globe": ["..####..", ".#.##.#.", "#..##..#", "########", "#..##..#", "#..##..#", ".#.##.#.", "..####.."],
    "shield": ["########", "#......#", "#.....##", "#....#.#", "##..#..#", ".##.#.#.", "..###...", "...##..."],
}


def icon(name: str, size: int = 16) -> Markup:
    rects = "".join(
        f'<rect x="{x}" y="{y}" width="1" height="1"/>'
        for y, row in enumerate(_ICONS[name]) for x, v in enumerate(row) if v == "#"
    )
    return Markup(
        f'<svg class="icon" width="{size}" height="{size}" viewBox="0 0 8 8" fill="currentColor" '
        f'shape-rendering="crispEdges" aria-hidden="true">{rects}</svg>'
    )


@lru_cache(maxsize=1)
def model_version() -> str:
    return "v1-" + hashlib.sha256(MODEL_JSON.read_bytes()).hexdigest()[:7]


@lru_cache(maxsize=1)
def reports() -> dict:
    """Everything the About page and stat strip need, read once from reports/."""
    def table(name):
        return pd.read_csv(REPORTS / name).to_dict("records")

    return {
        "test": table("metrics_test.csv"),
        "validation": table("metrics_validation.csv"),
        "experiments": table("experiments.csv"),
        "config": json.loads(CONFIG_PATH.read_text()),
        "test_info": json.loads((REPORTS / "info_test.json").read_text()),
    }


def metric(rows: list[dict], method: str, key: str) -> float:
    return next(r[key] for r in rows if r["method"] == method)
