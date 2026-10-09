from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "data"
MODEL_PATH = PROJECT_ROOT / "models" / "deepfake_full.pt"

IMAGE_SIZE = 224
MAX_VIDEO_SECONDS = 60.0
MAX_FRAME_PIXELS = 1920 * 1080
MAX_SAMPLED_FRAMES = 12
DEFAULT_SAMPLED_FRAMES = 8
MIN_FACE_FRAMES = 3

# Demo-only routing bands. Calibrate with held-out data before any real use.
LOWER_RISK_CEILING = 0.35
REVIEW_FLOOR = 0.50