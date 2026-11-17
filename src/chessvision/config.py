"""Global system parameters (paths, camera, calibration)."""
from pathlib import Path

# --- Paths ---------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"

CALIB_DIR = DATA_DIR / "calibration"
CALIB_IMAGES_DIR = CALIB_DIR / "images"      # pattern photos (input)
CALIB_CORNERS_DIR = CALIB_DIR / "corners"    # drawn corners (for the report)
CALIB_FILE = CALIB_DIR / "calibration.npz"   # K, dist, RMS (output)

# --- Camera (Camo) -------------------------------------------------------
# Camo shows up in Windows as just another webcam. If the laptop has a
# built-in webcam, it is usually index 0 and Camo is index 1.
# Check it with:  python scripts/calibrate.py probe
CAMERA_INDEX = 1
FRAME_WIDTH = 1280          # resolution selected in the Camo app
FRAME_HEIGHT = 720

# --- Calibration pattern -------------------------------------------------
# Number of INNER corners (columns, rows), not squares.
# The standard OpenCV pattern (10x7 squares) has 9x6 inner corners.
CHESSBOARD_SHAPE = (9, 6)
SQUARE_SIZE_MM = 24.0        # measured side of a square on the PRINTED pattern
