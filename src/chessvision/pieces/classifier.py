"""Clasificador de piezas: models/pieces.onnx con cv2.dnn (sin PyTorch).

Replica exactamente el preprocesado del entrenamiento (training/export_onnx.py),
leyendo los parámetros de models/pieces.json.
"""
import json
from pathlib import Path

import cv2
import numpy as np

from chessvision.board.geometry import crop_squares

MODELS_DIR = Path(__file__).resolve().parents[3] / "models"


class PieceClassifier:
    def __init__(self, onnx_path: Path = MODELS_DIR / "pieces.onnx",
                 json_path: Path | None = None):
        onnx_path = Path(onnx_path)
        json_path = Path(json_path) if json_path else onnx_path.with_suffix(".json")
        self.meta = json.loads(json_path.read_text(encoding="utf-8"))
        self.net = cv2.dnn.readNetFromONNX(np.fromfile(str(onnx_path), dtype=np.uint8))
        self.classes = self.meta["classes"]
        self.w = self.meta["input_size"]["width"]
        self.h = self.meta["input_size"]["height"]
        self.mean = np.array(self.meta["mean"], np.float32)
        self.std = np.array(self.meta["std"], np.float32)
        self.square_px = self.meta["crop"]["square_px"]
        self.height_squares = self.meta["crop"]["height_squares"]

    def preprocess(self, crops_bgr: list[np.ndarray]) -> np.ndarray:
        batch = []
        for img in crops_bgr:
            x = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            x = cv2.resize(x, (self.w, self.h), interpolation=cv2.INTER_AREA)
            x = (x.astype(np.float32) / 255.0 - self.mean) / self.std
            batch.append(x.transpose(2, 0, 1))
        return np.ascontiguousarray(np.stack(batch), dtype=np.float32)

    def predict(self, crops_bgr: list[np.ndarray]) -> np.ndarray:
        """Devuelve probabilidades [N, 13]."""
        self.net.setInput(self.preprocess(crops_bgr))
        logits = self.net.forward()
        e = np.exp(logits - logits.max(1, keepdims=True))
        return e / e.sum(1, keepdims=True)

    def crops(self, frame_bgr: np.ndarray, corners: dict) -> dict[str, np.ndarray]:
        return crop_squares(frame_bgr, corners, self.square_px, self.height_squares)

    def classify_board(self, frame_bgr: np.ndarray, corners: dict) -> dict[str, tuple[str, float]]:
        """frame ya corregido de distorsión. corners = {"a1","h1","h8","a8"} -> (x, y).

        Devuelve {casilla: (clase, confianza)} para las 64 casillas.
        """
        probs = self.board_probs(frame_bgr, corners)
        return {n: (self.classes[int(p.argmax())], float(p.max())) for n, p in probs.items()}

    def board_probs(self, frame_bgr: np.ndarray, corners: dict) -> dict[str, np.ndarray]:
        """Full probabilities {square: [13]} (order of self.classes) for the 64 squares."""
        crops = self.crops(frame_bgr, corners)
        names = list(crops)
        probs = self.predict([crops[n] for n in names])
        return dict(zip(names, probs))