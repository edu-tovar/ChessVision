"""ChessReD2K -> recortes de casillas por clase en data/pieces/.

Para cada imagen con esquinas anotadas:
  filtro de ángulo -> homografía -> 64 recortes -> data/pieces/<split>/<clase>/

Uso:
    python training/prepare_data.py
    python training/prepare_data.py --limit 20     # prueba rápida
    python training/prepare_data.py --overwrite    # borra data/pieces/ antes
"""
import argparse
import csv
import json
import random
import shutil
from multiprocessing import Pool

import cv2
from tqdm import tqdm

from config import (CHESSRED_DIR, CLASSES, CROP_HEIGHT_SQUARES, DATA_DIR,
                    EMPTY_KEEP_RATIO, MIN_PERSPECTIVE_RATIO, NUM_WORKERS, SEED,
                    WARP_SQUARE_PX)
from geometry import crop_squares, perspective_ratio

# Esquinas de ChessReD (vista de las blancas) -> casilla
CORNER_NAMES = {"bottom_left": "a1", "bottom_right": "h1",
                "top_right": "h8", "top_left": "a8"}
SPLITS = ("train", "val", "test")
READ_SCALE = 0.5  # se lee a la mitad (IMREAD_REDUCED_COLOR_2): más rápido y sobra resolución


def load_annotations():
    with open(CHESSRED_DIR / "annotations.json", encoding="utf-8") as f:
        ann = json.load(f)

    # "white-pawn" -> "white_pawn"
    cat_to_class = {c["id"]: c["name"].replace("-", "_") for c in ann["categories"]}
    assert set(cat_to_class.values()) == set(CLASSES), "Las clases no coinciden con config.CLASSES"

    images = {im["id"]: im for im in ann["images"]}
    corners = {c["image_id"]: {CORNER_NAMES[k]: v for k, v in c["corners"].items()}
               for c in ann["annotations"]["corners"]}
    labels = {}  # image_id -> {casilla: clase}
    for p in ann["annotations"]["pieces"]:
        labels.setdefault(p["image_id"], {})[p["chessboard_position"]] = cat_to_class[p["category_id"]]

    split_of = {i: s for s in SPLITS for i in ann["splits"]["chessred2k"][s]["image_ids"]}
    return images, corners, labels, split_of


def process(job):
    """Recorta una imagen y guarda sus casillas. Devuelve filas del manifest."""
    im, corners, labels, split, seed = job
    img = cv2.imread(str(CHESSRED_DIR / im["path"]), cv2.IMREAD_REDUCED_COLOR_2)
    if img is None:
        return []
    corners = {k: (x * READ_SCALE, y * READ_SCALE) for k, (x, y) in corners.items()}
    crops = crop_squares(img, corners, WARP_SQUARE_PX, CROP_HEIGHT_SQUARES)

    rng = random.Random(seed)
    rows = []
    for square, crop in crops.items():
        cls = labels.get(square, "empty")
        # Solo se submuestrean las vacías en train; val/test se quedan completos
        if cls == "empty" and split == "train" and rng.random() > EMPTY_KEEP_RATIO:
            continue
        name = f"G{im['game_id']:03d}_M{im['move_id']:03d}_{square}.png"
        out = DATA_DIR / split / cls / name
        cv2.imwrite(str(out), crop)
        rows.append((split, cls, out.relative_to(DATA_DIR).as_posix(),
                     im["game_id"], im["id"], square))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="máx. imágenes por split (0 = todas)")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    if args.overwrite and DATA_DIR.exists():
        shutil.rmtree(DATA_DIR)
    for s in SPLITS:
        for c in CLASSES:
            (DATA_DIR / s / c).mkdir(parents=True, exist_ok=True)

    images, corners, labels, split_of = load_annotations()

    jobs, skipped = [], 0
    per_split = {s: 0 for s in SPLITS}
    for img_id, split in sorted(split_of.items()):
        if args.limit and per_split[split] >= args.limit:
            continue
        if perspective_ratio(corners[img_id]) < MIN_PERSPECTIVE_RATIO:
            skipped += 1
            continue
        per_split[split] += 1
        jobs.append((images[img_id], corners[img_id], labels.get(img_id, {}),
                     split, SEED + img_id))
    print(f"Imágenes: {per_split} | descartadas por ángulo: {skipped}")

    rows = []
    if NUM_WORKERS > 0:
        with Pool(NUM_WORKERS) as pool:
            for r in tqdm(pool.imap_unordered(process, jobs, chunksize=8), total=len(jobs)):
                rows.extend(r)
    else:
        for job in tqdm(jobs):
            rows.extend(process(job))

    with open(DATA_DIR / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["split", "class", "path", "game_id", "image_id", "square"])
        w.writerows(sorted(rows))

    # Resumen por clase
    print(f"\n{'clase':<14}" + "".join(f"{s:>8}" for s in SPLITS))
    for c in CLASSES:
        print(f"{c:<14}" + "".join(f"{sum(1 for r in rows if r[0] == s and r[1] == c):>8}"
                                   for s in SPLITS))
    print(f"\nRecortes guardados en {DATA_DIR}")


if __name__ == "__main__":
    main()