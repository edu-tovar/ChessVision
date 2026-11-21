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

# --- Security: piece recognition with SIFT -------------------------------
SECURITY_DIR = DATA_DIR / "security"          # reference photos: <set>/<piece>/NNN.jpg
SECURITY_REFERENCE_SET = "upright"            # set used as references by the app and scripts/security.py
PIECE_TYPES = ("pawn", "knight", "bishop", "rook", "queen", "king")
SECURITY_ROI = (440, 160, 400, 400)           # (x, y, w, h) in the undistorted frame
SIFT_FEATURES = "silhouette"  # silhouette | texture | raw (see security/piece_matcher.py)
SIFT_RATIO = 0.75            # Lowe's ratio test
SIFT_MIN_MATCHES = 6         # good matches needed before running RANSAC
SIFT_MIN_INLIERS = 6         # RANSAC inliers needed to accept a piece type
SIFT_SCALE_RANGE = (0.5, 2.0)  # plausible query/reference scale (pieces shown at a similar distance)
# Colour mask: SIFT only looks at the (light, yellowish) piece, never at the hand or the wall.
# OpenCV HSV ranges: H 0-180, S 0-255, V 0-255. Skin is H 0-10, the light pieces H 18-24.
# Min S = 100: a white wall under warm light reaches H 14-19 / S 48-62, the pieces S 150-180.
PIECE_HSV_LOW = (14, 100, 40)
PIECE_HSV_HIGH = (35, 255, 255)
PIECE_MASK_DILATE = 15       # px added around the piece so that contour keypoints are kept
