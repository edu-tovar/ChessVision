"""Descarga ChessReD (annotations.json + images.zip) en data/chessred/.

Uso:
    python training/download_chessred.py            # todo
    python training/download_chessred.py --only-annotations

Licencia del dataset: CC BY-NC-SA 4.0 (Masouris, 2023).
"""
import argparse
import hashlib
import sys
import urllib.request
import zipfile
from pathlib import Path

from config import CHESSRED_DIR

BASE = "https://data.4tu.nl/file/99b5c721-280b-450b-b058-b2900b69a90f"
FILES = {
    "annotations.json": (f"{BASE}/3cae6364-daca-4967-b426-1e4b68cdb64c",
                         "d34bca5ad46ec7a8df96a1d3c36784f3"),
    "images.zip": (f"{BASE}/6329e969-616e-48e3-b893-a0379d1c15ba",
                   "32e23ed535239dc517a6499762d9847e"),
}
CHUNK = 1 << 20  # 1 MiB


def md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def download(url: str, dest: Path) -> None:
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length", 0))
        done = 0
        while block := r.read(CHUNK):
            f.write(block)
            done += len(block)
            if total:
                print(f"\r  {dest.name}: {done / total:6.1%} "
                      f"({done >> 20} / {total >> 20} MiB)", end="", flush=True)
    print()
    tmp.rename(dest)


def fetch(name: str) -> Path:
    url, expected = FILES[name]
    dest = CHESSRED_DIR / name
    if dest.exists() and md5(dest) == expected:
        print(f"  {name}: ya descargado, se omite")
        return dest
    print(f"Descargando {name}...")
    download(url, dest)
    if md5(dest) != expected:
        dest.unlink()
        sys.exit(f"Error: MD5 incorrecto en {name}. Vuelve a intentarlo.")
    return dest


def extract(zip_path: Path) -> None:
    if (CHESSRED_DIR / "images").exists():
        print("  images/: ya extraído, se omite")
        return
    print("Extrayendo images.zip...")
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(CHESSRED_DIR)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--only-annotations", action="store_true")
    p.add_argument("--keep-zip", action="store_true")
    args = p.parse_args()

    CHESSRED_DIR.mkdir(parents=True, exist_ok=True)
    fetch("annotations.json")
    if args.only_annotations:
        return

    zip_path = fetch("images.zip")
    extract(zip_path)
    if not args.keep_zip:
        zip_path.unlink()
    print(f"Listo: {CHESSRED_DIR}")


if __name__ == "__main__":
    main()