"""Entrena la CNN de piezas y guarda el mejor modelo en models/pieces_best.pt.

Fase 1: backbone congelado, solo se entrena la cabeza.
Fase 2: fine-tuning de toda la red con early stopping sobre la pérdida de validación.

Uso:
    python training/train.py
    python training/train.py --model simple_cnn --no-pretrained
    python training/train.py --epochs-finetune 2 --workers 0     # prueba rápida
"""
import argparse
import csv
import random
import shutil
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from config import (CLASSES, EPOCHS_FINETUNE, EPOCHS_HEAD, INPUT_H, INPUT_W,
                    LR_FINETUNE, LR_HEAD, MODEL_NAME, MODELS_DIR, NUM_WORKERS,
                    PATIENCE, RUNS_DIR, SEED)
from dataset import class_weights, make_loaders
from model import build_model, set_backbone_trainable


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def run_epoch(model, loader, criterion, device, optimizer=None, scaler=None):
    """Una época de train (si hay optimizer) o de validación. Devuelve métricas."""
    train = optimizer is not None
    model.train(train)
    n_cls = len(CLASSES)
    conf = torch.zeros(n_cls, n_cls, dtype=torch.long)
    total_loss, n = 0.0, 0

    with torch.set_grad_enabled(train):
        for x, y in tqdm(loader, leave=False, desc="train" if train else "val"):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device.type, enabled=scaler is not None):
                logits = model(x)
                loss = criterion(logits, y)
            if train:
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
            total_loss += loss.item() * y.size(0)
            n += y.size(0)
            pred = logits.argmax(1).cpu()
            conf += torch.bincount(y.cpu() * n_cls + pred, minlength=n_cls ** 2).view(n_cls, n_cls)

    acc = conf.diag().sum().item() / max(n, 1)
    recall = conf.diag().float() / conf.sum(1).clamp(min=1).float()
    present = conf.sum(1) > 0
    bal_acc = recall[present].mean().item()       # media del acierto por clase
    return {"loss": total_loss / max(n, 1), "acc": acc, "bal_acc": bal_acc}


def save_checkpoint(path, model, args, epoch, metrics):
    torch.save({
        "state_dict": model.state_dict(),
        "model_name": args.model,
        "classes": CLASSES,
        "input_size": (INPUT_W, INPUT_H),
        "epoch": epoch,
        "val": metrics,
    }, path)


def plot_history(rows, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ep = [r["epoch"] for r in rows]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    a1.plot(ep, [r["train_loss"] for r in rows], label="train")
    a1.plot(ep, [r["val_loss"] for r in rows], label="val")
    a1.set_title("Pérdida"); a1.set_xlabel("época"); a1.legend()
    a2.plot(ep, [r["val_acc"] for r in rows], label="val acc")
    a2.plot(ep, [r["val_bal_acc"] for r in rows], label="val acc balanceada")
    a2.set_title("Validación"); a2.set_xlabel("época"); a2.legend()
    fase2 = next((r["epoch"] for r in rows if r["phase"] == 2), None)
    for ax in (a1, a2):
        if fase2 and fase2 > 1:
            ax.axvline(fase2 - 0.5, color="gray", ls="--", lw=0.8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL_NAME)
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--epochs-head", type=int, default=EPOCHS_HEAD)
    ap.add_argument("--epochs-finetune", type=int, default=EPOCHS_FINETUNE)
    ap.add_argument("--loss-weights", action="store_true",
                    help="pesos por clase en la pérdida en vez de sampler balanceado")
    ap.add_argument("--workers", type=int, default=NUM_WORKERS)
    args = ap.parse_args()

    seed_everything(SEED)
    device = get_device()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    run_dir = RUNS_DIR / f"{datetime.now():%Y%m%d_%H%M%S}_{args.model}"
    run_dir.mkdir(parents=True, exist_ok=True)
    run_best = run_dir / "best.pt"            # copia por ejecución (para comparar modelos)
    best_path = MODELS_DIR / "pieces_best.pt"

    train_dl, val_dl, train_ds = make_loaders(balanced=not args.loss_weights,
                                              num_workers=args.workers)
    print(f"Dispositivo: {device} | train: {len(train_ds)} | val: {len(val_dl.dataset)}")

    pretrained = not args.no_pretrained and args.model != "simple_cnn"
    model = build_model(args.model, pretrained=pretrained).to(device)

    weight = class_weights(train_ds).to(device) if args.loss_weights else None
    criterion = nn.CrossEntropyLoss(weight=weight, label_smoothing=0.1)
    scaler = torch.amp.GradScaler() if device.type == "cuda" else None

    # Fase 1 solo tiene sentido con pesos preentrenados
    phases = []
    if pretrained and args.epochs_head > 0:
        phases.append((1, args.epochs_head, LR_HEAD, False))
    phases.append((2, args.epochs_finetune, LR_FINETUNE, True))

    history, best_loss, epoch = [], float("inf"), 0
    for phase, n_epochs, lr, trainable in phases:
        set_backbone_trainable(model, trainable)
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
        stale = 0
        print(f"\n--- Fase {phase}: {n_epochs} épocas, lr={lr}, "
              f"{'toda la red' if trainable else 'solo cabeza'} ---")

        for _ in range(n_epochs):
            epoch += 1
            t0 = time.time()
            tr = run_epoch(model, train_dl, criterion, device, optimizer, scaler)
            va = run_epoch(model, val_dl, criterion, device)
            scheduler.step()

            row = {"epoch": epoch, "phase": phase, "lr": optimizer.param_groups[0]["lr"],
                   "train_loss": tr["loss"], "train_acc": tr["acc"],
                   "val_loss": va["loss"], "val_acc": va["acc"], "val_bal_acc": va["bal_acc"]}
            history.append(row)

            mark = ""
            if va["loss"] < best_loss:
                best_loss, stale, mark = va["loss"], 0, "  * guardado"
                save_checkpoint(run_best, model, args, epoch, va)
            else:
                stale += 1
            print(f"Ép {epoch:>2} | train loss {tr['loss']:.4f} acc {tr['acc']:.3f} | "
                  f"val loss {va['loss']:.4f} acc {va['acc']:.3f} bal {va['bal_acc']:.3f} | "
                  f"{time.time() - t0:.0f}s{mark}")

            if phase == 2 and stale >= PATIENCE:
                print(f"Early stopping: {PATIENCE} épocas sin mejorar")
                break

    with open(run_dir / "history.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=history[0].keys())
        w.writeheader()
        w.writerows(history)
    plot_history(history, run_dir / "curves.png")
    shutil.copy(run_best, best_path)
    print(f"\nMejor val loss: {best_loss:.4f} -> {best_path}")
    print(f"Historial y gráficas en {run_dir}")


if __name__ == "__main__":
    main()