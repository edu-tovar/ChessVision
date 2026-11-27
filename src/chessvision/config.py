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

# --- Board corners (board/corners.py) ------------------------------------
# Selected once with the mouse (fixed stand) and reused. Not uploaded: it depends on the set-up.
BOARD_CORNERS_FILE = DATA_DIR / "board" / "corners.json"
BOARD_MIN_MATCH = 56          # squares (of 64) the CNN must match with the known position to trust the orientation

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
SIFT_CROP_MARGIN = 32        # px around the silhouette bounding box where SIFT runs
# Colour mask: SIFT only looks at the (light, yellowish) piece, never at the hand or the wall.
# OpenCV HSV ranges: H 0-180, S 0-255, V 0-255. Skin is H 0-10, the light pieces H 18-24.
# Min S = 100: a white wall under warm light reaches H 14-19 / S 48-62, the pieces S 150-180.
PIECE_HSV_LOW = (14, 100, 40)
PIECE_HSV_HIGH = (35, 255, 255)
PIECE_MASK_DILATE = 15       # px added around the piece so that contour keypoints are kept

# --- Security: key decoder (security/decoder.py) --------------------------
SECURITY_KEY = ("knight", "rook", "bishop", "queen")   # secret: 4 distinct piece types, in order
# Presence of a piece = area (px) of its colour mask in the ROI, with hysteresis.
# Smallest piece measured: ~3500 px (pawn); empty ROI: ~0 px.
DECODER_AREA_ON = 2000       # area above which a piece is in front of the camera...
DECODER_ON_FRAMES = 2        # ... for this many consecutive frames
DECODER_AREA_OFF = 1000      # area below which the piece has been removed...
DECODER_OFF_FRAMES = 4       # ... for this many consecutive frames
DECODER_MAX_SHIFT = 15       # px the mask centroid may move between frames; above it the frame does not vote
DECODER_VOTE_FRAMES = 8      # recognised (still) frames needed to decide
DECODER_MIN_FRAMES = 3       # fewer recognised frames than this -> no decision ("show it longer")
DECODER_MAX_TIME = 4.0       # s showing a piece; then decide with the frames collected so far
DECODER_MIN_SHARE = 0.6      # share of the inlier-weighted vote the winner needs
# Fraction of the still frames in which some piece reached SIFT_MIN_INLIERS. Guards against an
# object that only now and then looks like a piece. 0.5 rejected a real rook held in a fist.
DECODER_MIN_RECOGNISED = 0.25
DECODER_IDLE_RESET = 10.0    # s without any piece -> the partially entered key is cleared
