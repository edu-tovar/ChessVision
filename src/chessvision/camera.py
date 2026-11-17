"""Video capture and lens distortion correction using the calibration parameters."""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

from chessvision import config


# --- Capture ---------------------------------------------------------------
def open_camera(index: int = config.CAMERA_INDEX,
                width: int = config.FRAME_WIDTH,
                height: int = config.FRAME_HEIGHT) -> cv2.VideoCapture:
    """Open the camera and set its resolution. Raises an error if it cannot be opened."""
    backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
    cap = cv2.VideoCapture(index, backend)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera {index}. "
                           "Is Camo running on the PC and the phone connected?")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    real = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    if real != (width, height):
        print(f"[camera] Warning: requested {width}x{height} but the camera gives "
              f"{real[0]}x{real[1]}. Check the resolution in Camo.")
    return cap


# --- Image read/write --------------------------------------------------------
# cv2.imread/cv2.imwrite fail on Windows when the path has non-ASCII characters
# (e.g. "3º"). These versions go through NumPy and work with any path.
def read_image(path: Path) -> np.ndarray | None:
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None


def write_image(path: Path, img: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(path.suffix or ".jpg", img)
    if not ok:
        raise IOError(f"Could not encode image {path}")
    buf.tofile(str(path))


# --- Calibration parameters --------------------------------------------------
def save_calibration(path: Path, K: np.ndarray, dist: np.ndarray, rms: float,
                     image_size: tuple[int, int], **extra) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, K=K, dist=dist, rms=rms, image_size=np.array(image_size), **extra)


def load_calibration(path: Path = config.CALIB_FILE) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist. Run first: python scripts/calibrate.py calibrate")
    with np.load(path) as data:
        return {k: data[k] for k in data.files}


# --- Distortion correction ---------------------------------------------------
class Undistorter:
    """Corrects distortion with precomputed maps (initUndistortRectifyMap + remap).

    The maps are computed only once; each frame only needs a remap,
    which is cheap and does not hurt the FPS.

    alpha = 0 -> crops so that no black borders remain (recommended).
    alpha = 1 -> keeps every original pixel (black borders appear).
    """

    def __init__(self, K: np.ndarray, dist: np.ndarray, image_size: tuple[int, int],
                 alpha: float = 0.0):
        self.image_size = tuple(int(v) for v in image_size)          # (w, h)
        self.K, self.dist = K, dist
        self.new_K, self.roi = cv2.getOptimalNewCameraMatrix(K, dist, self.image_size,
                                                             alpha, self.image_size)
        self.map1, self.map2 = cv2.initUndistortRectifyMap(K, dist, None, self.new_K,
                                                           self.image_size, cv2.CV_16SC2)

    @classmethod
    def from_file(cls, path: Path = config.CALIB_FILE, alpha: float = 0.0) -> "Undistorter":
        calib = load_calibration(path)
        return cls(calib["K"], calib["dist"], tuple(calib["image_size"]), alpha)

    def __call__(self, frame: np.ndarray) -> np.ndarray:
        size = (frame.shape[1], frame.shape[0])
        if size != self.image_size:
            raise ValueError(f"The frame is {size[0]}x{size[1]} but the calibration was done at "
                             f"{self.image_size[0]}x{self.image_size[1]}. Use the same resolution.")
        return cv2.remap(frame, self.map1, self.map2, cv2.INTER_LINEAR)
