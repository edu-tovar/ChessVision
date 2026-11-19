"""Geometría del tablero: homografía cenital y recorte de las 64 casillas.

Lo usan tanto training/prepare_data.py como la app, para que el recorte sea
idéntico en entrenamiento e inferencia.

Convención de esquinas: dict con claves "a1", "h1", "h8", "a8" -> (x, y) en
píxeles de la imagen original.

El tablero se endereza de forma que el lado más cercano a la cámara quede
abajo. Así la parte alta de cada pieza (que con vista oblicua "invade" la
casilla de detrás) queda siempre por encima de su casilla, y el recorte alto
la incluye sea cual sea el lado desde el que se hizo la foto.
"""
import cv2
import numpy as np

FILES = "abcdefgh"
CORNER_ORDER = ("a1", "h1", "h8", "a8")          # orden cíclico alrededor del tablero
BOARD_COORDS = np.float32([[0, 0], [8, 0], [8, 8], [0, 8]])  # (columna, fila) de cada esquina


def _orient(corners: dict) -> tuple[np.ndarray, np.ndarray]:
    """Rota las esquinas para que el lado más cercano a la cámara quede abajo.

    Devuelve (puntos_imagen, coords_tablero) en orden BL, BR, TR, TL.
    """
    pts = np.float32([corners[k] for k in CORNER_ORDER])
    # Lado más cercano a la cámara = el de mayor y media en la imagen
    k = max(range(4), key=lambda i: pts[i, 1] + pts[(i + 1) % 4, 1])
    idx = [(k + i) % 4 for i in range(4)]
    return pts[idx], BOARD_COORDS[idx]


def warp_board(img: np.ndarray, corners: dict, square_px: int = 64,
               crop_h_squares: int = 2) -> tuple[np.ndarray, list[list[str]]]:
    """Endereza el tablero a vista cenital.

    Deja (crop_h_squares - 1) casillas de margen por arriba para que los
    recortes de la fila superior no se salgan.

    Devuelve:
        warped: imagen de (8*square_px) x ((8 + margen)*square_px)
        names:  names[fila][col] = nombre de la casilla ("e4") en el warp
    """
    m = crop_h_squares - 1
    s = square_px
    src, board = _orient(corners)
    # Destino BL, BR, TR, TL (y hacia abajo), con margen m arriba
    dst = np.float32([[0, (8 + m) * s], [8 * s, (8 + m) * s],
                      [8 * s, m * s], [0, m * s]])
    H = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, H, (8 * s, (8 + m) * s),
                                 flags=cv2.INTER_AREA)

    # Correspondencia celda del warp -> casilla real (es una rotación exacta)
    grid = np.float32([[0, 8], [8, 8], [8, 0], [0, 0]])
    G = cv2.getPerspectiveTransform(grid, board)
    centers = np.float32([[[c + 0.5, r + 0.5]] for r in range(8) for c in range(8)])
    fr = cv2.perspectiveTransform(centers, G).reshape(8, 8, 2)
    names = [[f"{FILES[int(fr[r, c, 0])]}{int(fr[r, c, 1]) + 1}" for c in range(8)]
             for r in range(8)]
    return warped, names


def crop_squares(img: np.ndarray, corners: dict, square_px: int = 64,
                 crop_h_squares: int = 2) -> dict[str, np.ndarray]:
    """Devuelve {casilla: recorte} con las 64 casillas.

    Cada recorte mide square_px de ancho x (crop_h_squares * square_px) de alto:
    la casilla abajo y lo que hay "detrás" de ella arriba.
    """
    warped, names = warp_board(img, corners, square_px, crop_h_squares)
    s, h = square_px, crop_h_squares * square_px
    out = {}
    for r in range(8):
        for c in range(8):
            # La casilla (r, c) ocupa y en [(r+m)s, (r+m+1)s]; el recorte empieza m casillas antes
            out[names[r][c]] = warped[r * s: r * s + h, c * s: (c + 1) * s]
    return out


def perspective_ratio(corners: dict) -> float:
    """1.0 = vista cenital; cuanto menor, más oblicua."""
    a, b, c, d = (np.float32(corners[k]) for k in CORNER_ORDER)
    n = np.linalg.norm
    r1 = min(n(b - a), n(c - d)) / max(n(b - a), n(c - d))
    r2 = min(n(d - a), n(c - b)) / max(n(d - a), n(c - b))
    return float(min(r1, r2))