"""Parámetros globales del sistema (rutas, cámara, calibración)."""
from pathlib import Path

# --- Rutas ---------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"

CALIB_DIR = DATA_DIR / "calibration"
CALIB_IMAGES_DIR = CALIB_DIR / "images"      # fotos del patrón (entrada)
CALIB_CORNERS_DIR = CALIB_DIR / "corners"    # esquinas dibujadas (para el informe)
CALIB_FILE = CALIB_DIR / "calibration.npz"   # K, dist, RMS (salida)

# --- Cámara (Camo) -------------------------------------------------------
# Camo aparece en Windows como una webcam más. Si el portátil tiene webcam
# integrada, ésta suele ser el índice 0 y Camo el 1.
# Comprueba cuál es con:  python scripts/calibrate.py probe
CAMERA_INDEX = 1
FRAME_WIDTH = 1280          # resolución elegida en la aplicación Camo
FRAME_HEIGHT = 720

# --- Patrón de calibración ----------------------------------------------
# Número de esquinas INTERIORES (columnas, filas), no de casillas.
# El patrón estándar de OpenCV (10x7 casillas) tiene 9x6 esquinas interiores.
CHESSBOARD_SHAPE = (9, 6)
SQUARE_SIZE_MM = 24.0        # medida real de casilla del patrón IMPRESO
