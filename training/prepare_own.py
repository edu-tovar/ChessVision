"""Genera recortes etiquetados a partir de fotos propias con su FEN.
 
Entrada (una carpeta, por defecto data/own_photos/):
    000.jpeg, 001.jpeg, ...           fotos del tablero
    000.jpeg.corners.json, ...        esquinas a1/h1/h8/a8 (las genera scripts/test_photos.py)
    *.csv                             columnas photo,fen  (o archivo,fen)
 
Salida:
    data/pieces_own/{train,val,test}/<clase>/<foto>_<casilla>.png
    data/pieces_own/splits.csv        qué foto ha ido a cada split
    data/real_test/<clase>/...        copia del split test (para evaluate.py --real)
 
El split se hace POR FOTO (todas las casillas de una foto van al mismo split).
 
Uso (desde la raíz del repo):
    python training/prepare_own.py
    python training/prepare_own.py --src data/own_photos --val 0.15 --test 0.15
"""
import argparse
import csv
import json
import random
import shutil
from collections import Counter
from pathlib import Path
 
import cv2
import numpy as np
 
from chessvision.board.geometry import crop_squares
from config import CLASSES, CROP_HEIGHT_SQUARES, REAL_TEST_DIR, ROOT, SEED, WARP_SQUARE_PX
 
OWN_PHOTOS_DIR = ROOT / "data" / "cenital_photos"
OWN_DATA_DIR = ROOT / "data" / "pieces_cenital"
 
FEN_TO_CLASS = {"P": "white_pawn", "N": "white_knight", "B": "white_bishop",
                "R": "white_rook", "Q": "white_queen", "K": "white_king",
                "p": "black_pawn", "n": "black_knight", "b": "black_bishop",
                "r": "black_rook", "q": "black_queen", "k": "black_king"}
 
 
def fen_to_classes(fen: str) -> dict[str, str]:
    """FEN -> {casilla: clase} con las 64 casillas. Lanza ValueError si está mal escrita."""
    rows = fen.split()[0].split("/")
    if len(rows) != 8:
        raise ValueError(f"tiene {len(rows)} filas, no 8")
    board = {}
    for r, row in enumerate(rows):
        c = 0
        for ch in row:
            if ch.isdigit():
                for _ in range(int(ch)):
                    board[f"{'abcdefgh'[c]}{8 - r}"] = "empty"
                    c += 1
            elif ch in FEN_TO_CLASS:
                if c < 8:
                    board[f"{'abcdefgh'[c]}{8 - r}"] = FEN_TO_CLASS[ch]
                c += 1
            else:
                raise ValueError(f"carácter no válido '{ch}' en la fila {8 - r}")
        if c != 8:
            raise ValueError(f"la fila {8 - r} ('{row}') tiene {c} casillas, no 8")
    return board
 
 
def read_fens(src: Path) -> dict[str, str]:
    csvs = sorted(src.glob("*.csv"))
    if not csvs:
        raise SystemExit(f"No hay ningún .csv con las FEN en {src}")
    fens = {}
    for path in csvs:
        with open(path, encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                row = {k.strip().lower(): (v or "").strip() for k, v in row.items() if k}
                name = row.get("photo") or row.get("archivo") or row.get("file")
                fen = row.get("fen")
                if name and fen:
                    fens[name] = fen
    return fens
 
 
def read_image(path: Path):
    return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
 
 
def write_png(path: Path, img) -> None:
    ok, buf = cv2.imencode(".png", img)
    if ok:
        buf.tofile(str(path))
 
 
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, default=OWN_PHOTOS_DIR)
    ap.add_argument("--out", type=Path, default=OWN_DATA_DIR)
    ap.add_argument("--val", type=float, default=0.15, help="fracción de fotos para validación")
    ap.add_argument("--test", type=float, default=0.15, help="fracción de fotos para test")
    ap.add_argument("--empty-keep", type=float, default=0.5,
                    help="fracción de casillas vacías que se guardan en train")
    ap.add_argument("--no-real-test", action="store_true", help="no copiar el test a data/real_test")
    args = ap.parse_args()
 
    fens = read_fens(args.src)
    photos, problems = [], []
    for name, fen in sorted(fens.items()):
        p = args.src / name
        if not p.exists():
            problems.append(f"{name}: no existe la foto")
        elif not (args.src / f"{name}.corners.json").exists():
            problems.append(f"{name}: faltan las esquinas (márcalas con scripts/test_photos.py)")
        else:
            try:
                fen_to_classes(fen)
                photos.append(name)
            except ValueError as e:
                problems.append(f"{name}: FEN incorrecta, {e}")
    for msg in problems:
        print("  ⚠️ ", msg)
    if not photos:
        raise SystemExit("No hay fotos válidas")
 
    # Split por foto, reproducible
    rng = random.Random(SEED)
    order = photos[:]
    rng.shuffle(order)
    n_test = max(1, round(len(order) * args.test))
    n_val = max(1, round(len(order) * args.val))
    split_of = {n: "test" for n in order[:n_test]}
    split_of.update({n: "val" for n in order[n_test:n_test + n_val]})
    split_of.update({n: "train" for n in order[n_test + n_val:]})
 
    if args.out.exists():
        shutil.rmtree(args.out)
    for s in ("train", "val", "test"):
        for c in CLASSES:
            (args.out / s / c).mkdir(parents=True, exist_ok=True)
 
    counts = {s: Counter() for s in ("train", "val", "test")}
    for name in photos:
        img = read_image(args.src / name)
        if img is None:
            print(f"  ⚠️  {name}: no se puede leer")
            continue
        corners = {k: tuple(v) for k, v in
                   json.loads((args.src / f"{name}.corners.json").read_text()).items()}
        labels = fen_to_classes(fens[name])
        crops = crop_squares(img, corners, WARP_SQUARE_PX, CROP_HEIGHT_SQUARES)
        split = split_of[name]
        stem = Path(name).stem
        for sq, crop in crops.items():
            cls = labels[sq]
            if split == "train" and cls == "empty" and rng.random() > args.empty_keep:
                continue
            write_png(args.out / split / cls / f"{stem}_{sq}.png", crop)
            counts[split][cls] += 1
 
    with open(args.out / "splits.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["photo", "split"])
        for n in sorted(split_of):
            w.writerow([n, split_of[n]])
 
    if not args.no_real_test:
        if REAL_TEST_DIR.exists():
            shutil.rmtree(REAL_TEST_DIR)
        shutil.copytree(args.out / "test", REAL_TEST_DIR)
 
    print(f"\nFotos: {len(photos)}  (train {sum(v == 'train' for v in split_of.values())}, "
          f"val {sum(v == 'val' for v in split_of.values())}, "
          f"test {sum(v == 'test' for v in split_of.values())})")
    print(f"{'clase':<14}{'train':>7}{'val':>7}{'test':>7}")
    for c in CLASSES:
        print(f"{c:<14}{counts['train'][c]:>7}{counts['val'][c]:>7}{counts['test'][c]:>7}")
    print(f"\nRecortes en {args.out}" + ("" if args.no_real_test else f" (test copiado a {REAL_TEST_DIR})"))
 
 
if __name__ == "__main__":
    main()