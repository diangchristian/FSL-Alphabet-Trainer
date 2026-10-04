"""Paths and constants shared by every step of the pipeline."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# Override per machine with FSL_DATASET / FSL_PWA environment variables.
DATASET = Path(os.environ.get("FSL_DATASET", r"C:\Users\chris\Downloads\archive\Collated"))
HAND_MODEL = ROOT / "assets" / "hand_landmarker.task"
HAND_MODEL_SHA256 = "fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1"
HAND_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
CACHE = ROOT / "cache"
FIXTURES = ROOT / "fixtures"
REPORTS = ROOT / "reports"
RELEASES = ROOT / "releases"
FSL_PWA = Path(os.environ.get("FSL_PWA", r"D:\Programming\fsl-pwa"))

# Must match the app (client/src/features/fsl-recognition/category-core.js).
FRAMES = 32
FEATURES = 128
FORMAT = "fsl-conv1d-bilstm-v1"
NORMALIZATION = "hand-relative-v1"
DETECTION_CONFIDENCE = 0.5
NUM_HANDS = 2
SWAP_HANDS = False

SEED = 42
VALIDATION_FRACTION = 0.15
TEST_FRACTION = 0.15
