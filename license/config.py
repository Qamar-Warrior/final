import os
from pathlib import Path

BASE_DIR = Path(__file__).parent

# Paths
DB_PATH = os.environ.get("LPR_DB_PATH", str(BASE_DIR / "plates.db"))
MODEL_PATH = os.environ.get("LPR_MODEL_PATH", str(BASE_DIR / "models" / "plate_detector.pt"))
MODEL_FALLBACK_URL = "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt"

# Detection tuning
YOLO_CONFIDENCE = float(os.environ.get("LPR_YOLO_CONF", "0.45"))
OCR_LANGUAGES = os.environ.get("LPR_OCR_LANGS", "en").split(",")
VIDEO_FRAME_SKIP = int(os.environ.get("LPR_FRAME_SKIP", "3"))

# Deduplication window in seconds (skip re-saving same plate within this window)
DEDUP_WINDOW_SECONDS = int(os.environ.get("LPR_DEDUP_SECONDS", "2"))

# Minimum confidence to save a detection to the database (0.0 - 1.0)
MIN_SAVE_CONFIDENCE = float(os.environ.get("LPR_MIN_SAVE_CONF", "0.60"))

# API server
API_HOST = os.environ.get("LPR_HOST", "0.0.0.0")
API_PORT = int(os.environ.get("LPR_PORT", "8000"))

# GPU usage for EasyOCR (set to "1" to enable)
USE_GPU = os.environ.get("LPR_GPU", "1") == "1"
