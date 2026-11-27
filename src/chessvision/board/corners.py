"""Board corners in the live app: click them once in order, save them, reuse them.

The camera stand is fixed, so the corners are selected once at start-up and reused.
The user clicks the 4 outer corners of the playing area in the order a1, h1, h8, a8.

Convention (same as board/geometry.py): {"a1", "h1", "h8", "a8"} -> (x, y) in the UNDISTORTED frame.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from chessvision import config
from chessvision.board.geometry import CORNER_ORDER, FILES

INITIAL_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"
SYMBOL_TO_CLASS = {"P": "white_pawn", "N": "white_knight", "B": "white_bishop",
                   "R": "white_rook", "Q": "white_queen", "K": "white_king",
                   "p": "black_pawn", "n": "black_knight", "b": "black_bishop",
                   "r": "black_rook", "q": "black_queen", "k": "black_king"}


# --- Geometry ------------------------------------------------------------------
def board_to_image_homography(corners: dict) -> np.ndarray:
    """H that maps board coordinates (file, rank) in [0, 8]^2 to image pixels.

    (0, 0) = outer corner of a1, (8, 8) = outer corner of h8. The centre of e4 is (4.5, 3.5).
    """
    src = np.float32([[0, 0], [8, 0], [8, 8], [0, 8]])          # a1, h1, h8, a8
    dst = np.float32([corners[k] for k in CORNER_ORDER])
    return cv2.getPerspectiveTransform(src, dst)


def square_center(corners: dict, square: str, H: np.ndarray | None = None) -> tuple[int, int]:
    """Pixel position of the centre of a square (e.g. for arrows on the real image)."""
    H = board_to_image_homography(corners) if H is None else H
    f, r = FILES.index(square[0]) + 0.5, int(square[1]) - 0.5
    x, y = cv2.perspectiveTransform(np.float32([[[f, r]]]), H)[0, 0]
    return int(round(x)), int(round(y))


def square_polygon(corners: dict, square: str, H: np.ndarray | None = None) -> np.ndarray:
    """Image polygon (4x2 int32) of a square: mask of the square in the frame."""
    H = board_to_image_homography(corners) if H is None else H
    f, r = FILES.index(square[0]), int(square[1]) - 1
    pts = np.float32([[[f, r], [f + 1, r], [f + 1, r + 1], [f, r + 1]]])
    return np.int32(np.round(cv2.perspectiveTransform(pts, H)[0]))


def board_mask(corners: dict, shape: tuple[int, ...], margin: float = 0.0) -> np.ndarray:
    """Binary mask (uint8 0/255) of the board in the frame, optionally enlarged by `margin` squares."""
    H = board_to_image_homography(corners)
    m = margin
    pts = np.float32([[[-m, -m], [8 + m, -m], [8 + m, 8 + m], [-m, 8 + m]]])
    poly = np.int32(np.round(cv2.perspectiveTransform(pts, H)[0]))
    mask = np.zeros(shape[:2], np.uint8)
    cv2.fillConvexPoly(mask, poly, 255)
    return mask


def draw_grid(img: np.ndarray, corners: dict, color=(0, 255, 0), labels: bool = True) -> np.ndarray:
    """Draws the 8x8 grid and the corner names on a copy of the image."""
    out = img.copy()
    H = board_to_image_homography(corners)
    for i in range(9):
        for a, b in (((i, 0), (i, 8)), ((0, i), (8, i))):
            p = cv2.perspectiveTransform(np.float32([[a, b]]), H)[0]
            cv2.line(out, tuple(np.int32(p[0])), tuple(np.int32(p[1])), color, 1, cv2.LINE_AA)
    if labels:
        for sq in ("a1", "h1", "h8", "a8"):
            x, y = square_center(corners, sq, H)
            cv2.putText(out, sq, (x - 12, y + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
            cv2.putText(out, sq, (x - 12, y + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
    return out


# --- Known positions ------------------------------------------------------------
def fen_to_classes(fen: str) -> dict[str, str]:
    """FEN (placement field is enough) -> {square: class name}."""
    out = {}
    for r, row in enumerate(fen.split()[0].split("/")):
        c = 0
        for ch in row:
            if ch.isdigit():
                for _ in range(int(ch)):
                    out[f"{FILES[c]}{8 - r}"] = "empty"
                    c += 1
            else:
                out[f"{FILES[c]}{8 - r}"] = SYMBOL_TO_CLASS[ch]
                c += 1
    return out


# --- Persistence -----------------------------------------------------------------
def save_corners(corners: dict, frame_size: tuple[int, int], path: Path = config.BOARD_CORNERS_FILE) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {"frame_size": list(frame_size), "corners": {k: list(corners[k]) for k in CORNER_ORDER}}
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def load_corners(path: Path = config.BOARD_CORNERS_FILE,
                 frame_size: tuple[int, int] | None = None) -> dict | None:
    """Saved corners, or None if there are none (or they were taken at another resolution)."""
    path = Path(path)
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    if frame_size is not None and tuple(data["frame_size"]) != tuple(frame_size):
        print(f"[board] Saved corners are for {data['frame_size']}, frame is {list(frame_size)}: ignored.")
        return None
    return {k: tuple(v) for k, v in data["corners"].items()}


# --- Interactive selection on the live camera -------------------------------------
def select_corners(cap, undistort, window: str = "Board setup") -> dict | None:
    """Live window to click the 4 board corners in the order a1, h1, h8, a8.

    Keys: click = next corner | BACKSPACE = undo last | R = restart | ENTER = accept | ESC = cancel.
    Once the 4 corners are in, the 8x8 grid is drawn to check it before accepting.
    Returns the corners (undistorted frame) or None if cancelled.
    """
    pts: list[tuple[int, int]] = []

    def on_mouse(event, x, y, *_):
        if event == cv2.EVENT_LBUTTONDOWN and len(pts) < 4:
            pts.append((x, y))

    cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(window, on_mouse)
    while True:
        ok, raw = cap.read()
        if not ok:
            print("[board] No frames from the camera.")
            cv2.destroyWindow(window)
            return None
        frame = undistort(raw)

        if len(pts) == 4:
            corners = {k: (float(x), float(y)) for k, (x, y) in zip(CORNER_ORDER, pts)}
            view = draw_grid(frame, corners)
            hint = "Check the grid  |  ENTER = save  |  BACKSPACE = undo  |  R = restart"
        else:
            view = frame.copy()
            for name, p in zip(CORNER_ORDER, pts):
                cv2.circle(view, p, 5, (0, 0, 255), -1)
                cv2.putText(view, name, (p[0] + 8, p[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            if len(pts) > 1:
                cv2.polylines(view, [np.int32(pts)], False, (0, 255, 0), 1)
            hint = f"Click {CORNER_ORDER[len(pts)]}  (order a1, h1, h8, a8)  |  BACKSPACE = undo  |  ESC = cancel"
        cv2.putText(view, hint, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
        cv2.putText(view, hint, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
        cv2.imshow(window, view)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            cv2.destroyWindow(window)
            return None
        if key in (ord("r"), ord("R")):
            pts.clear()
        elif key == 8 and pts:                          # backspace
            pts.pop()
        elif key in (13, 10) and len(pts) == 4:
            cv2.destroyWindow(window)
            return {k: (float(x), float(y)) for k, (x, y) in zip(CORNER_ORDER, pts)}


def get_corners(cap, undistort, recalibrate: bool = False,
                path: Path = config.BOARD_CORNERS_FILE) -> dict | None:
    """Saved corners if they exist (same resolution); otherwise opens the selection window and saves them."""
    size = undistort.image_size
    if not recalibrate:
        corners = load_corners(path, size)
        if corners is not None:
            print(f"[board] Corners loaded from {path}")
            return corners
    corners = select_corners(cap, undistort)
    if corners is not None:
        save_corners(corners, size, path)
        print(f"[board] Corners saved to {path}")
    return corners