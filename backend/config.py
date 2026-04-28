"""Backend configuration."""

from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = ROOT_DIR / "model" / "artifacts"
MODEL_PATH = ARTIFACT_DIR / "model.h5"
PREPROCESSOR_PATH = ARTIFACT_DIR / "preprocessor.joblib"
METADATA_PATH = ARTIFACT_DIR / "metadata.json"
DEMO_MODEL_PATH = ARTIFACT_DIR / "demo_model.joblib"
DEMO_METADATA_PATH = ARTIFACT_DIR / "demo_metadata.json"
FRONTEND_DIR = ROOT_DIR / "frontend"
MAX_UPLOAD_SIZE_MB = 16
