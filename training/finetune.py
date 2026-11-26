"""Fine-tuning del modelo de piezas con recortes propios (data/pieces_own/).
 
Parte de models/pieces_best.pt (entrenado con ChessReD) y lo ajusta a tu tablero.
Cada batch mezcla recortes propios y de ChessReD para que no olvide lo aprendido.
 
Antes de empezar guarda una copia del modelo original en models/pieces_chessred.pt.
El resultado sustituye a models/pieces_best.pt (luego: export_onnx.py).
 
Uso (desde la raíz del repo):
    python training/prepare_own.py
    python training/finetune.py
    python training/finetune.py --own-frac 0.8 --epochs 30 --workers 0
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
from torch.utils.data import ConcatDataset, DataLoader, WeightedRandomSampler
from tqdm import tqdm
 
from config import (BATCH_SIZE, CLASSES, DATA_DIR, INPUT_H, INPUT_W, MODELS_DIR,
                    NUM_WORKERS, ROOT, RUNS_DIR, SEED)
from dataset import PiecesDataset
from model import build_model
 
OWN_DATA_DIR = ROOT / "data" / "pieces_cenital"
 
 
def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
 
 
def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
 
 
def domain_weights(ds: PiecesDataset, total: float) -> list[float]:
    """Pesos por muestra: clases balanceadas dentro del dominio y suma total = `total`."""
    counts = ds.class_counts()
    present = sum(1 for c in counts if c > 0)
    return [total / (present * counts[label]) for _, label in ds.samples]
 
 
def run_epoch(model, loader, criterion, device, optimizer=None, scaler=None):
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
    bal = recall[conf.sum(1) > 0].mean().item()
    return {"loss": total_loss / max(n, 1), "acc": acc, "bal_acc": bal}
 
 
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--init", default=None, help="checkpoint de partida (por defecto el de ChessReD)")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--own-frac", type=float, default=0.7,
                    help="fracción de cada batch con recortes propios (resto ChessReD)")
    ap.add_argument("--samples-per-epoch", type=int, default=8000)
    ap.add_argument("--workers", type=int, default=NUM_WORKERS)
    args = ap.parse_args()
 
    seed_everything(SEED)
    device = get_device()
 
    # Copia de seguridad del modelo de ChessReD (solo la primera vez)
    best_path = MODELS_DIR / "pieces_best.pt"
    base_path = MODELS_DIR / "pieces_chessred.pt"
    if not base_path.exists():
        if not best_path.exists():
            raise SystemExit(f"No existe {best_path}. Entrena antes con training.py")
        shutil.copy(best_path, base_path)
        print(f"Copia del modelo original en {base_path}")
    init_path = args.init or base_path
 
    ckpt = torch.load(init_path, map_location="cpu")
    assert ckpt["classes"] == CLASSES, "Las clases del checkpoint no coinciden con config.CLASSES"
    model = build_model(ckpt["model_name"], pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
 
    # Datos
    own_train = PiecesDataset("train", root=OWN_DATA_DIR, train=True)
    own_val = PiecesDataset("val", root=OWN_DATA_DIR, train=False)
    parts, weights = [own_train], domain_weights(own_train, args.own_frac)
    if (DATA_DIR / "train").exists() and args.own_frac < 1:
        cr_train = PiecesDataset("train", root=DATA_DIR, train=True)
        parts.append(cr_train)
        weights += domain_weights(cr_train, 1 - args.own_frac)
    else:
        print("Sin recortes de ChessReD en data/pieces/train: se entrena solo con los propios")
    train_ds = ConcatDataset(parts)
    sampler = WeightedRandomSampler(weights, num_samples=args.samples_per_epoch, replacement=True)
    pin = device.type == "cuda"
    train_dl = DataLoader(train_ds, BATCH_SIZE, sampler=sampler, num_workers=args.workers, pin_memory=pin)
    val_dl = DataLoader(own_val, BATCH_SIZE, shuffle=False, num_workers=args.workers, pin_memory=pin)
    print(f"Dispositivo: {device} | propios train {len(own_train)}, val {len(own_val)}"
          + (f" | ChessReD train {len(parts[1])}" if len(parts) > 1 else "")
          + f" | {args.own_frac:.0%} propios por batch")
 
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler() if device.type == "cuda" else None
 
    run_dir = RUNS_DIR / f"{datetime.now():%Y%m%d_%H%M%S}_finetune"
    run_dir.mkdir(parents=True, exist_ok=True)
    run_best = run_dir / "best.pt"
 
    va = run_epoch(model, val_dl, criterion, device)
    print(f"Antes del fine-tuning | val propio: loss {va['loss']:.4f} acc {va['acc']:.3f} bal {va['bal_acc']:.3f}")
 
    history, best_loss, stale = [], float("inf"), 0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr = run_epoch(model, train_dl, criterion, device, optimizer, scaler)
        va = run_epoch(model, val_dl, criterion, device)
        scheduler.step()
        history.append({"epoch": epoch, "train_loss": tr["loss"], "train_acc": tr["acc"],
                        "val_loss": va["loss"], "val_acc": va["acc"], "val_bal_acc": va["bal_acc"]})
        mark = ""
        if va["loss"] < best_loss:
            best_loss, stale, mark = va["loss"], 0, "  * guardado"
            torch.save({"state_dict": model.state_dict(), "model_name": ckpt["model_name"],
                        "classes": CLASSES, "input_size": (INPUT_W, INPUT_H), "epoch": epoch,
                        "val": va, "finetuned_from": str(init_path)}, run_best)
        else:
            stale += 1
        print(f"Ép {epoch:>2} | train loss {tr['loss']:.4f} acc {tr['acc']:.3f} | "
              f"val loss {va['loss']:.4f} acc {va['acc']:.3f} bal {va['bal_acc']:.3f} | "
              f"{time.time() - t0:.0f}s{mark}")
        if stale >= args.patience:
            print(f"Early stopping: {args.patience} épocas sin mejorar")
            break
 
    with open(run_dir / "history.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=history[0].keys())
        w.writeheader()
        w.writerows(history)
    shutil.copy(run_best, best_path)
    print(f"\nMejor modelo (val loss {best_loss:.4f}) copiado a {best_path}")
    print("Siguiente: python training/evaluate.py --real  y  python training/export_onnx.py")
 
 
if __name__ == "__main__":
    main()