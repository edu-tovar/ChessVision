"""Prueba rápida del modelo de piezas con fotos sueltas.

Para cada foto:
  1. Marca con clic las esquinas a1, h1, h8, a8 (en ese orden). r = reiniciar, Enter = aceptar.
     Se guardan en <foto>.corners.json y se reutilizan la próxima vez (--recalibrate para repetir).
  2. Clasifica las 64 casillas y muestra el tablero en texto + FEN.
  3. Guarda una imagen con el tablero enderezado y la predicción de cada casilla.

Uso (desde la raíz del repo):
    python scripts/test_photos.py foto1.jpg foto2.jpg
    python scripts/test_photos.py data/test_photos/            # todas las fotos de una carpeta
    python scripts/test_photos.py foto.jpg --initial           # compara con la posición inicial
    python scripts/test_photos.py foto.jpg --fen "<FEN>"       # compara con otra posición
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from chessvision.board.geometry import warp_board
from chessvision.pieces.classifier import PieceClassifier

CLICK_ORDER = ("a1", "h1", "h8", "a8")
SYMBOL = {"empty": ".", "white_pawn": "P", "white_knight": "N", "white_bishop": "B",
          "white_rook": "R", "white_queen": "Q", "white_king": "K",
          "black_pawn": "p", "black_knight": "n", "black_bishop": "b",
          "black_rook": "r", "black_queen": "q", "black_king": "k"}
INITIAL_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".heic"}
MAX_VIEW = 1000  # lado máximo de la ventana de clic


def read_image(path: Path):
    data = np.fromfile(str(path), dtype=np.uint8)   # soporta rutas con tildes en Windows
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def write_image(path: Path, img) -> None:
    ok, buf = cv2.imencode(path.suffix, img)
    if ok:
        buf.tofile(str(path))


def click_corners(img) -> dict | None:
    scale = min(1.0, MAX_VIEW / max(img.shape[:2]))
    view = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    pts = []

    def on_mouse(event, x, y, *_):
        if event == cv2.EVENT_LBUTTONDOWN and len(pts) < 4:
            pts.append((x, y))

    win = "Clic en a1, h1, h8, a8  |  r = reiniciar  |  Enter = aceptar  |  Esc = saltar"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(win, on_mouse)
    while True:
        canvas = view.copy()
        for i, p in enumerate(pts):
            cv2.circle(canvas, p, 6, (0, 0, 255), -1)
            cv2.putText(canvas, CLICK_ORDER[i], (p[0] + 8, p[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        if len(pts) > 1:
            cv2.polylines(canvas, [np.int32(pts)], len(pts) == 4, (0, 255, 0), 2)
        msg = f"Siguiente: {CLICK_ORDER[len(pts)]}" if len(pts) < 4 else "Enter para aceptar"
        cv2.putText(canvas, msg, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        cv2.imshow(win, canvas)
        key = cv2.waitKey(20) & 0xFF
        if key == ord("r"):
            pts.clear()
        elif key in (13, 10) and len(pts) == 4:
            break
        elif key == 27:
            cv2.destroyWindow(win)
            return None
    cv2.destroyWindow(win)
    return {k: (x / scale, y / scale) for k, (x, y) in zip(CLICK_ORDER, pts)}


def get_corners(photo: Path, img, recalibrate: bool) -> dict | None:
    cfile = photo.with_suffix(photo.suffix + ".corners.json")
    if cfile.exists() and not recalibrate:
        return {k: tuple(v) for k, v in json.loads(cfile.read_text()).items()}
    corners = click_corners(img)
    if corners:
        cfile.write_text(json.dumps(corners, indent=2))
    return corners


def fen_to_board(fen: str) -> dict[str, str]:
    """FEN (solo colocación) -> {casilla: símbolo}."""
    board, rows = {}, fen.split()[0].split("/")
    for r, row in enumerate(rows):
        c = 0
        for ch in row:
            if ch.isdigit():
                for _ in range(int(ch)):
                    board[f"{'abcdefgh'[c]}{8 - r}"] = "."
                    c += 1
            else:
                board[f"{'abcdefgh'[c]}{8 - r}"] = ch
                c += 1
    return board


def board_to_fen(sym: dict[str, str]) -> str:
    rows = []
    for rank in range(8, 0, -1):
        row, empty = "", 0
        for f in "abcdefgh":
            s = sym[f"{f}{rank}"]
            if s == ".":
                empty += 1
            else:
                row += (str(empty) if empty else "") + s
                empty = 0
        rows.append(row + (str(empty) if empty else ""))
    return "/".join(rows)


def print_board(sym, conf, expected=None, low=0.6):
    print("    a  b  c  d  e  f  g  h")
    for rank in range(8, 0, -1):
        line = f" {rank} "
        for f in "abcdefgh":
            sq = f"{f}{rank}"
            mark = " "
            if expected and expected[sq] != sym[sq]:
                mark = "x"          # fallo respecto a la posición esperada
            elif conf[sq] < low:
                mark = "?"          # poca confianza
            line += f" {sym[sq]}{mark}"
        print(line)
    print("    (x = fallo, ? = confianza < %.1f)" % low)


def draw_result(img, corners, pred, clf, expected=None):
    s, m = clf.square_px * 2, clf.height_squares - 1          # warp a 128 px/casilla para verlo bien
    warped, names = warp_board(img, corners, s, clf.height_squares)
    for r in range(8):
        for c in range(8):
            sq = names[r][c]
            cls, p = pred[sq]
            x, y = c * s, (r + m) * s
            wrong = expected is not None and expected[sq] != SYMBOL[cls]
            color = (0, 0, 255) if wrong else ((0, 200, 255) if p < 0.6 else (0, 255, 0))
            cv2.rectangle(warped, (x, y), (x + s - 1, y + s - 1), color, 1)
            label = f"{sq} {SYMBOL[cls]} {p:.2f}"
            cv2.putText(warped, label, (x + 4, y + s - 8), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, (0, 0, 0), 3)
            cv2.putText(warped, label, (x + 4, y + s - 8), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, color, 1)
    return warped


def collect(paths):
    out = []
    for p in map(Path, paths):
        if p.is_dir():
            out += sorted(f for f in p.iterdir() if f.suffix.lower() in IMG_EXT)
        else:
            out.append(p)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("photos", nargs="+", help="fotos o carpetas")
    ap.add_argument("--initial", action="store_true", help="comparar con la posición inicial")
    ap.add_argument("--fen", help="comparar con esta posición (FEN)")
    ap.add_argument("--recalibrate", action="store_true", help="volver a marcar las esquinas")
    ap.add_argument("--model", type=Path, default=None, help="ruta a pieces.onnx")
    ap.add_argument("--no-show", action="store_true", help="no mostrar la imagen de resultado")
    args = ap.parse_args()

    clf = PieceClassifier(args.model) if args.model else PieceClassifier()
    expected = fen_to_board(args.fen or INITIAL_FEN) if (args.initial or args.fen) else None

    for photo in collect(args.photos):
        img = read_image(photo)
        if img is None:
            print(f"No se puede leer {photo}")
            continue
        corners = get_corners(photo, img, args.recalibrate)
        if corners is None:
            print(f"Saltada: {photo}")
            continue

        pred = clf.classify_board(img, corners)
        sym = {sq: SYMBOL[c] for sq, (c, _) in pred.items()}
        conf = {sq: p for sq, (_, p) in pred.items()}

        print(f"\n=== {photo.name} ===")
        print_board(sym, conf, expected)
        print(f"FEN: {board_to_fen(sym)}")
        if expected:
            errors = [sq for sq in sym if sym[sq] != expected[sq]]
            print(f"Aciertos: {64 - len(errors)}/64" +
                  (f"  | fallos: {', '.join(f'{sq} {expected[sq]}->{sym[sq]}' for sq in errors)}"
                   if errors else "  | tablero perfecto"))
        print(f"Confianza media: {np.mean(list(conf.values())):.3f}")

        result = draw_result(img, corners, pred, clf, expected)
        out = photo.with_name(photo.stem + "_pred.png")
        write_image(out, result)
        print(f"Imagen: {out}")
        if not args.no_show:
            cv2.imshow("Prediccion (verde = ok, naranja = dudosa, rojo = fallo)  |  tecla = seguir", result)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
