"""Exporta el checkpoint a models/pieces.onnx + models/pieces.json para la app.

Verifica que la salida de OpenCV (cv2.dnn, lo que usa la app) coincide con PyTorch.

Uso:
    python training/export_onnx.py
    python training/export_onnx.py --ckpt training/runs/<run>/best.pt
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from config import (CLASSES, CROP_HEIGHT_SQUARES, INPUT_H, INPUT_W, MEAN,
                    MODELS_DIR, STD, WARP_SQUARE_PX)
from evaluate import load_model

OPSET = 17


def preprocess(crops_bgr: list[np.ndarray]) -> np.ndarray:
    """Mismo preprocesado que dataset.py (sin aumentos). La app debe replicarlo.

    BGR uint8 (cualquier tamaño) -> RGB -> resize W×H -> [0,1] -> (x - mean) / std -> NCHW float32
    """
    batch = []
    for img in crops_bgr:
        x = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        x = cv2.resize(x, (INPUT_W, INPUT_H), interpolation=cv2.INTER_AREA)
        x = (x.astype(np.float32) / 255.0 - MEAN) / STD
        batch.append(x.transpose(2, 0, 1))
    return np.ascontiguousarray(np.stack(batch), dtype=np.float32)


def export(model, out_path: Path) -> None:
    dummy = torch.randn(1, 3, INPUT_H, INPUT_W)
    kwargs = dict(input_names=["input"], output_names=["logits"], opset_version=OPSET,
                  dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}})
    try:
        # Exportador clásico (TorchScript): no necesita onnxscript y es el más compatible con cv2.dnn
        torch.onnx.export(model, dummy, str(out_path), dynamo=False, **kwargs)
    except TypeError:
        # Versiones de torch sin el argumento "dynamo"
        torch.onnx.export(model, dummy, str(out_path), **kwargs)


def verify(model, onnx_path: Path, n: int = 64) -> float:
    """Compara PyTorch vs cv2.dnn con un batch aleatorio de recortes. Devuelve la diferencia máxima."""
    rng = np.random.default_rng(0)
    crops = [rng.integers(0, 256, (INPUT_H, INPUT_W, 3), dtype=np.uint8) for _ in range(n)]
    blob = preprocess(crops)

    with torch.no_grad():
        ref = model(torch.from_numpy(blob)).numpy()

    net = cv2.dnn.readNetFromONNX(str(onnx_path))
    net.setInput(blob)
    out = net.forward()

    diff = float(np.abs(ref - out).max())
    same = float((ref.argmax(1) == out.argmax(1)).mean())
    print(f"Verificación cv2.dnn: diferencia máx. logits = {diff:.2e} | misma predicción = {same:.0%}")
    return diff


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, default=MODELS_DIR / "pieces_best.pt")
    ap.add_argument("--out", type=Path, default=MODELS_DIR / "pieces.onnx")
    args = ap.parse_args()

    model, ckpt = load_model(args.ckpt, torch.device("cpu"))
    export(model, args.out)
    print(f"ONNX guardado en {args.out} ({args.out.stat().st_size / 1e6:.1f} MB)")

    meta = {
        "model_name": ckpt["model_name"],
        "epoch": ckpt["epoch"],
        "val_metrics": ckpt.get("val"),
        "onnx": args.out.name,
        "opset": OPSET,
        "input_name": "input",
        "output_name": "logits",
        "classes": CLASSES,
        "input_size": {"width": INPUT_W, "height": INPUT_H},
        "color_order": "RGB",
        "scale": 1 / 255,
        "mean": list(MEAN),
        "std": list(STD),
        "crop": {"square_px": WARP_SQUARE_PX, "height_squares": CROP_HEIGHT_SQUARES},
        "preprocess": "BGR->RGB, resize a input_size, x/255, (x-mean)/std, NCHW float32",
    }
    json_path = args.out.with_suffix(".json")
    json_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Metadatos guardados en {json_path}")

    if verify(model, args.out) > 1e-3:
        print("⚠️  La salida de cv2.dnn difiere de PyTorch más de lo esperado. Revisa la versión de OpenCV.")


if __name__ == "__main__":
    main()