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

# Vercel sets VERCEL=1 (and VERCEL_REGION at runtime) when "Enable access to
# System Environment Variables" is on in the project settings.
ON_VERCEL = bool(os.environ.get("VERCEL") or os.environ.get("VERCEL_REGION"))


def _writable(path: Path) -> bool:
    """True if we can actually create files there (catches read-only deployments too)."""
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".write-test"
        probe.write_text("ok")
        probe.unlink()
        return True
    except OSError:
        return False


# Even if the variables above are missing, never try to write into a read-only project folder.
WRITABLE_DIR = ROOT / "data" if not ON_VERCEL and _writable(ROOT / "data") else Path("/tmp/nfl-predictor")
