"""File locations shared by training and the web app (no heavy imports here).

On Vercel the project directory is read-only; only /tmp is writable, so
everything the app writes at runtime goes under WRITABLE_DIR.
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "pipeline.joblib"   # scikit-learn pipeline (training / evaluation)
MODEL_JSON = ROOT / "models" / "model.json"        # same model as plain numbers (used by the app)
CONFIG_PATH = ROOT / "models" / "config.json"
REPORTS = ROOT / "reports"
STATIC_DIR = ROOT / "public" / "static"            # Vercel serves public/** from its CDN

ON_VERCEL = bool(os.environ.get("VERCEL"))
WRITABLE_DIR = Path("/tmp/nfl-predictor") if ON_VERCEL else ROOT / "data"
