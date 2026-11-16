"""Captura de vídeo y corrección de la distorsión con los parámetros de calibración."""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

from chessvision import config


# --- Captura ---------------------------------------------------------------
def open_camera(index: int = config.CAMERA_INDEX,
                width: int = config.FRAME_WIDTH,
                height: int = config.FRAME_HEIGHT) -> cv2.VideoCapture:
    """Abre la cámara y fija la resolución. Lanza un error si no se puede abrir."""
    backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
    cap = cv2.VideoCapture(index, backend)
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir la cámara {index}. "
                           "¿Está Camo abierto en el PC y el móvil conectado?")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    real = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    if real != (width, height):
        print(f"[camera] Aviso: se pidió {width}x{height} pero la cámara da {real[0]}x{real[1]}. "
              "Revisa la resolución en Camo.")
    return cap


# --- Lectura/escritura de imágenes -------------------------------------------
# cv2.imread/cv2.imwrite fallan en Windows si la ruta tiene caracteres no ASCII
# (p. ej. "3º"). Estas versiones pasan por NumPy y funcionan con cualquier ruta.
def read_image(path: Path) -> np.ndarray | None:
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None


def write_image(path: Path, img: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(path.suffix or ".jpg", img)
    if not ok:
        raise IOError(f"No se pudo codificar la imagen {path}")
    buf.tofile(str(path))


# --- Parámetros de calibración ---------------------------------------------
def save_calibration(path: Path, K: np.ndarray, dist: np.ndarray, rms: float,
                     image_size: tuple[int, int], **extra) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, K=K, dist=dist, rms=rms, image_size=np.array(image_size), **extra)


def load_calibration(path: Path = config.CALIB_FILE) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"No existe {path}. Ejecuta antes: python scripts/calibrate.py")
    with np.load(path) as data:
        return {k: data[k] for k in data.files}


# --- Corrección de la distorsión -------------------------------------------
class Undistorter:
    """Corrige la distorsión con mapas precalculados (initUndistortRectifyMap + remap).

    Los mapas se calculan una sola vez; en cada frame solo se hace un remap,
    que es barato y no penaliza los FPS.

    alpha = 0 -> recorta para que no queden bordes negros (recomendado).
    alpha = 1 -> conserva todos los píxeles originales (aparecen bordes negros).
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
            raise ValueError(f"El frame es {size[0]}x{size[1]} pero la calibración se hizo a "
                             f"{self.image_size[0]}x{self.image_size[1]}. Usa la misma resolución.")
        return cv2.remap(frame, self.map1, self.map2, cv2.INTER_LINEAR)
