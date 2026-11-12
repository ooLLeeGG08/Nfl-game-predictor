"""Vercel rejects vercel.json if functions.<file>.excludeFiles is longer than 256 characters."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_exclude_files_within_vercel_limit():
    cfg = json.loads((ROOT / "vercel.json").read_text())
    for name, fn in cfg.get("functions", {}).items():
        assert len(fn.get("excludeFiles", "")) <= 256, name


def test_app_runtime_files_are_not_ignored():
    """The app reads these at runtime; they must stay in the deployment."""
    ignored = [l.strip().strip("/") for l in (ROOT / ".vercelignore").read_text().splitlines()
               if l.strip() and not l.startswith("#")]
    for needed in ("data/qb/upcoming.json", "models/v2/model.json", "models/model.json"):
        assert not any(needed == i or needed.startswith(i + "/") for i in ignored), needed
