"""Calibración offline de la cámara (método de Zhang)

Uso (desde la raíz del repositorio):
    python scripts/calibrate.py probe        # ¿qué índice tiene la cámara de Camo?
    python scripts/calibrate.py capture      # captura las fotos del patrón (ESPACIO = guardar)
    python scripts/calibrate.py calibrate    # calcula K, dist y RMS -> data/calibration/calibration.npz
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from chessvision import config                                    # noqa: E402
from chessvision.camera import (Undistorter, open_camera, read_image,  # noqa: E402
                                save_calibration, write_image)

FIND_FLAGS = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
SUBPIX_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


# --- Funciones de la Práctica 2 ------------------------------------------------
def get_chessboard_points(chessboard_shape, dx, dy) -> np.ndarray:
    """Coordenadas 3D de las esquinas en el sistema de referencia del patrón (Z = 0)."""
    cols, rows = chessboard_shape
    points = np.zeros((rows * cols, 3), np.float32)
    points[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * np.array([dx, dy])
    return points


def corner_spacing(corners, pattern_size) -> float:
    """Distancia mínima [px] entre esquinas adyacentes, en filas y columnas."""
    pts = corners.reshape(pattern_size[1], pattern_size[0], 2)
    d_rows = np.linalg.norm(np.diff(pts, axis=1), axis=-1)
    d_cols = np.linalg.norm(np.diff(pts, axis=0), axis=-1)
    return float(min(d_rows.min(), d_cols.min()))


def subpix_window(corners, pattern_size, image_shape=None, max_half: int = 11) -> tuple[int, int]:
    """Semiventana de cv2.cornerSubPix() adaptada a la separación entre esquinas.

    Ventana total (2*half+1) ~ una casilla: no llega a la esquina vecina.
    Probado con las imágenes de la Práctica 2: misma RMS que la ventana (5,5) fija.
    """
    half = int(corner_spacing(corners, pattern_size) / 2)
    if image_shape is not None:
        h, w = image_shape[:2]
        xy = corners.reshape(-1, 2)
        margin = np.min([xy[:, 0], xy[:, 1], w - 1 - xy[:, 0], h - 1 - xy[:, 1]])
        half = min(half, int(margin) - 1)
    return (int(np.clip(half, 2, max_half)),) * 2


# --- probe: buscar la cámara ---------------------------------------------------
def probe(args) -> None:
    backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
    print("Buscando cámaras (índices 0-5)...")
    for i in range(6):
        cap = cv2.VideoCapture(i, backend)
        if cap.isOpened() and cap.read()[0]:
            w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(f"  índice {i}: OK ({w}x{h})")
        cap.release()
    print("Tapa el objetivo del móvil o mira la vista previa con 'capture --camera N' "
          "para saber cuál es Camo, y pon ese índice en config.CAMERA_INDEX.")


# --- capture: tomar fotos del patrón -------------------------------------------
def capture(args) -> None:
    out_dir = Path(args.images)
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = [int(p.stem.split("_")[1]) for p in out_dir.glob("calib_*.jpg")
                if p.stem.split("_")[1].isdigit()]
    idx = max(existing, default=-1) + 1            # nunca sobrescribe fotos anteriores
    pattern = args.pattern
    cap = open_camera(args.camera, args.width, args.height)

    scale = 0.5            # la detección en vivo se hace a media resolución para ir fluido
    coverage = None        # zonas de la imagen ya cubiertas por el patrón
    print("ESPACIO = guardar foto (solo si se detecta el patrón) | Q/ESC = salir")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("No llegan frames de la cámara.")
            break
        if coverage is None:
            coverage = np.zeros_like(frame)

        small = cv2.cvtColor(cv2.resize(frame, None, fx=scale, fy=scale), cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(small, pattern,
                                                   FIND_FLAGS + cv2.CALIB_CB_FAST_CHECK)

        view = cv2.add(frame, coverage)
        if found:
            corners = corners / scale
            cv2.drawChessboardCorners(view, pattern, corners, found)
        color = (0, 200, 0) if found else (0, 0, 255)
        cv2.putText(view, f"Fotos: {idx}   Patron: {'SI' if found else 'NO'}", (15, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
        cv2.imshow("Calibracion - captura", view)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord(" "):
            if not found:
                print("  Patrón no detectado: no se guarda.")
                continue
            path = out_dir / f"calib_{idx:03d}.jpg"
            write_image(path, frame)                     # se guarda el frame LIMPIO
            layer = np.zeros_like(frame)
            hull = cv2.convexHull(corners.reshape(-1, 2).astype(np.int32))
            cv2.fillConvexPoly(layer, hull, (0, 50, 0))
            coverage = cv2.add(coverage, layer)
            idx += 1
            print(f"  Guardada {path.name}")

    cap.release()
    cv2.destroyAllWindows()
    print(f"{idx} fotos en {out_dir}")


# --- calibrate: calcular K, dist y RMS -----------------------------------------
def calibrate(args) -> None:
    img_dir, out_file, pattern = Path(args.images), Path(args.output), args.pattern
    corners_dir = out_file.parent / "corners"
    paths = sorted(p for ext in ("*.jpg", "*.png") for p in img_dir.glob(ext))
    if not paths:
        sys.exit(f"No hay imágenes en {img_dir}. Ejecuta antes: python scripts/calibrate.py capture")

    # 1) Detección de esquinas + 2) refinado subpíxel
    obj_template = get_chessboard_points(pattern, args.square, args.square)
    obj_points, img_points, used, image_size = [], [], [], None
    for p in paths:
        img = read_image(p)
        if img is None:
            print(f"  {p.name}: no se pudo leer")
            continue
        size = (img.shape[1], img.shape[0])
        if image_size is None:
            image_size = size
        elif size != image_size:
            sys.exit(f"{p.name} mide {size} y las anteriores {image_size}: "
                     "todas las fotos deben tener la misma resolución.")

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(gray, pattern, FIND_FLAGS)
        if not found:
            print(f"  {p.name}: patrón NO detectado -> se descarta")
            continue
        win = subpix_window(corners, pattern, gray.shape)
        corners = cv2.cornerSubPix(gray, corners, win, (-1, -1), SUBPIX_CRITERIA)

        obj_points.append(obj_template)
        img_points.append(corners)
        used.append(p)
        # 3) Comprobación visual: se guardan las esquinas dibujadas (útil para el informe)
        write_image(corners_dir / p.name, cv2.drawChessboardCorners(img.copy(), pattern, corners, True))

    print(f"\nDetecciones válidas: {len(used)}/{len(paths)} | resolución {image_size[0]}x{image_size[1]}")
    if len(used) < 3:
        sys.exit("Se necesitan al menos 3 imágenes válidas (recomendado 15-25).")
    if len(used) < 10:
        print("Aviso: con menos de 10 imágenes la calibración puede ser inestable.")

    # 4) Calibración (Zhang)
    flags = cv2.CALIB_FIX_K3 if args.fix_k3 else 0
    rms, K, dist, rvecs, tvecs, std_int, _, per_view = cv2.calibrateCameraExtended(
        obj_points, img_points, image_size, None, None, flags=flags)
    per_view = per_view.ravel()
    std_int = std_int.ravel()

    np.set_printoptions(suppress=True, precision=4)
    print("\nMatriz intrínseca K:\n", K)
    print(f"fx = {K[0, 0]:.2f} ± {std_int[0]:.2f} px | fy = {K[1, 1]:.2f} ± {std_int[1]:.2f} px")
    print(f"cx = {K[0, 2]:.2f} ± {std_int[2]:.2f} px | cy = {K[1, 2]:.2f} ± {std_int[3]:.2f} px "
          f"(centro de la imagen: {image_size[0] / 2:.0f}, {image_size[1] / 2:.0f})")
    print("Distorsión (k1, k2, p1, p2, k3):", dist.ravel())
    print(f"RMS de reproyección: {rms:.4f} px")

    print("\nError por imagen [px]:")
    median = float(np.median(per_view))
    for p, e in sorted(zip(used, per_view), key=lambda t: -t[1]):
        flag = "  <-- revisar (> 2x mediana)" if e > 2 * median else ""
        print(f"  {p.name}: {e:.4f}{flag}")

    # 5) Guardado
    save_calibration(out_file, K, dist, rms, image_size,
                     per_view_errors=per_view, std_intrinsics=std_int,
                     image_names=np.array([p.name for p in used]),
                     chessboard_shape=np.array(pattern), square_size=args.square)
    print(f"\nGuardado en {out_file}")

    # Ejemplo antes/después de la corrección (para el informe)
    sample = read_image(used[0])
    after = Undistorter(K, dist, image_size, alpha=0.0)(sample)
    write_image(out_file.parent / "undistort_example.jpg", np.hstack([sample, after]))
    print(f"Ejemplo original | corregida en {out_file.parent / 'undistort_example.jpg'}")


# --- CLI ----------------------------------------------------------------------
def parse_pattern(text: str) -> tuple[int, int]:
    cols, rows = text.lower().split("x")
    return int(cols), int(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("probe", help="lista las cámaras disponibles")
    p.set_defaults(func=probe)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--images", default=str(config.CALIB_IMAGES_DIR),
                        help="carpeta de las fotos del patrón")
    common.add_argument("--pattern", type=parse_pattern,
                        default=config.CHESSBOARD_SHAPE,
                        help="esquinas interiores COLxFIL, p. ej. 9x6")

    p = sub.add_parser("capture", parents=[common], help="captura fotos del patrón")
    p.add_argument("--camera", type=int, default=config.CAMERA_INDEX)
    p.add_argument("--width", type=int, default=config.FRAME_WIDTH)
    p.add_argument("--height", type=int, default=config.FRAME_HEIGHT)
    p.set_defaults(func=capture)

    p = sub.add_parser("calibrate", parents=[common], help="calcula K, dist y RMS")
    p.add_argument("--square", type=float, default=config.SQUARE_SIZE_MM,
                   help="lado de la casilla en mm")
    p.add_argument("--output", default=str(config.CALIB_FILE))
    p.add_argument("--fix-k3", action="store_true",
                   help="fija k3 = 0 (útil si k3 sale enorme con lentes poco distorsionadas)")
    p.set_defaults(func=calibrate)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
