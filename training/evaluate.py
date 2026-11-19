"""Evalúa un checkpoint: métricas por clase, matriz de confusión y acierto por tablero.

Uso:
    python training/evaluate.py                       # test de ChessReD
    python training/evaluate.py --split val
    python training/evaluate.py --real                # data/real_test/<clase>/*.png
    python training/evaluate.py --ckpt training/runs/<run>/best.pt
"""
import argparse
import csv
import shutil
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import CLASSES, DATA_DIR, MODELS_DIR, NUM_WORKERS, REAL_TEST_DIR, RUNS_DIR
from dataset import PiecesDataset
from model import build_model
from train import get_device


def load_model(ckpt_path: Path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    assert ckpt["classes"] == CLASSES, "Las clases del checkpoint no coinciden con config.CLASSES"
    model = build_model(ckpt["model_name"], pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval(), ckpt


@torch.no_grad()
def predict(model, ds, device, workers):
    dl = DataLoader(ds, batch_size=256, shuffle=False, num_workers=workers)
    preds, probs = [], []
    for x, _ in tqdm(dl, desc="Evaluando"):
        p = torch.softmax(model(x.to(device)), 1).cpu()
        probs.append(p.max(1).values)
        preds.append(p.argmax(1))
    return torch.cat(preds).numpy(), torch.cat(probs).numpy()


def plot_confusion(cm, out_path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cm_norm = cm / cm.sum(1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(9, 8))
    ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(CLASSES)), CLASSES, rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(len(CLASSES)), CLASSES, fontsize=8)
    ax.set_xlabel("Predicción"); ax.set_ylabel("Real"); ax.set_title(title)
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            if cm[i, j]:
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=6,
                        color="white" if cm_norm[i, j] > 0.5 else "black")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def board_metrics(ds, y_true, y_pred):
    """Acierto por tablero usando manifest.csv (solo splits de ChessReD)."""
    manifest = DATA_DIR / "manifest.csv"
    if not manifest.exists():
        return None
    with open(manifest, encoding="utf-8") as f:
        image_of = {r["path"]: r["image_id"] for r in csv.DictReader(f)}

    errors = defaultdict(int)
    squares = Counter()
    for (path, _), t, p in zip(ds.samples, y_true, y_pred):
        img = image_of.get(Path(path).relative_to(DATA_DIR).as_posix())
        if img is None:
            continue
        squares[img] += 1
        errors[img] += int(t != p)
    boards = [i for i, n in squares.items() if n == 64]   # solo tableros completos
    if not boards:
        return None
    errs = np.array([errors[i] for i in boards])
    return {"n": len(boards), "perfect": float((errs == 0).mean()),
            "le1": float((errs <= 1).mean()), "mean_err": float(errs.mean())}


def save_errors(ds, y_true, y_pred, out_dir, max_per_pair=20):
    """Copia ejemplos mal clasificados a errors/<real>__como__<pred>/."""
    count = Counter()
    for (path, _), t, p in zip(ds.samples, y_true, y_pred):
        if t == p:
            continue
        key = f"{CLASSES[t]}__como__{CLASSES[p]}"
        if count[key] >= max_per_pair:
            continue
        count[key] += 1
        d = out_dir / key
        d.mkdir(parents=True, exist_ok=True)
        shutil.copy(path, d / Path(path).name)
    return count


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, default=MODELS_DIR / "pieces_best.pt")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--real", action="store_true", help="evaluar en data/real_test/")
    ap.add_argument("--workers", type=int, default=NUM_WORKERS)
    args = ap.parse_args()

    device = get_device()
    model, ckpt = load_model(args.ckpt, device)
    if args.real:
        ds, name = PiecesDataset("", root=REAL_TEST_DIR, train=False), "real"
    else:
        ds, name = PiecesDataset(args.split, train=False), args.split
    print(f"Modelo: {ckpt['model_name']} (época {ckpt['epoch']}) | {name}: {len(ds)} recortes")

    y_true = np.array([label for _, label in ds.samples])
    y_pred, conf = predict(model, ds, device, args.workers)

    out = RUNS_DIR / f"eval_{name}_{datetime.now():%Y%m%d_%H%M%S}"
    out.mkdir(parents=True, exist_ok=True)

    labels = list(range(len(CLASSES)))
    report = classification_report(y_true, y_pred, labels=labels, target_names=CLASSES,
                                   digits=3, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    acc = (y_true == y_pred).mean()
    recall = cm.diagonal() / cm.sum(1).clip(min=1)
    bal_acc = recall[cm.sum(1) > 0].mean()

    lines = [f"Checkpoint: {args.ckpt}", f"Conjunto: {name} ({len(ds)} recortes)",
             f"Accuracy: {acc:.4f}", f"Accuracy balanceada: {bal_acc:.4f}", "", report]

    # Confusiones más frecuentes
    pairs = [(cm[i, j], CLASSES[i], CLASSES[j])
             for i in labels for j in labels if i != j and cm[i, j]]
    lines.append("Confusiones más frecuentes (real -> predicción):")
    lines += [f"  {n:>5}  {a} -> {b}" for n, a, b in sorted(pairs, reverse=True)[:10]]

    b = None if args.real else board_metrics(ds, y_true, y_pred)
    if b:
        lines += ["", f"Tableros completos: {b['n']}",
                  f"  Perfectos (64/64): {b['perfect']:.1%}",
                  f"  Con <= 1 error:    {b['le1']:.1%}",
                  f"  Errores medios por tablero: {b['mean_err']:.2f}"]

    text = "\n".join(lines)
    print("\n" + text)
    (out / "report.txt").write_text(text, encoding="utf-8")
    plot_confusion(cm, out / "confusion_matrix.png", f"{ckpt['model_name']} — {name}")
    n_err = save_errors(ds, y_true, y_pred, out / "errors")
    print(f"\nResultados en {out} ({sum(n_err.values())} ejemplos de error guardados)")


if __name__ == "__main__":
    main()