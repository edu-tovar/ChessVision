"""Board set-up tool: select the corners on the live camera and check the CNN on the stream.

Usage (from the repository root):
    python scripts/board.py setup              # click the 4 corners in the order a1, h1, h8, a8
    python scripts/board.py live               # grid + CNN prediction per square + FPS (uses the saved corners)
    python scripts/board.py live --fen "<FEN>" # ... and count the squares that match this position

The corners are saved in data/board/corners.json (undistorted frame) and reused while the stand does not move.
Live keys:  C = classify now | A = classify continuously on/off | ESC/Q = quit
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from chessvision import config                                                  # noqa: E402
from chessvision.board.corners import draw_grid, fen_to_classes, get_corners, square_center  # noqa: E402
from chessvision.board.geometry import perspective_ratio                        # noqa: E402
from chessvision.camera import Undistorter, open_camera                         # noqa: E402
from chessvision.pieces.classifier import PieceClassifier                       # noqa: E402

SYMBOL = {"empty": ".", "white_pawn": "P", "white_knight": "N", "white_bishop": "B",
          "white_rook": "R", "white_queen": "Q", "white_king": "K",
          "black_pawn": "p", "black_knight": "n", "black_bishop": "b",
          "black_rook": "r", "black_queen": "q", "black_king": "k"}


def put(img, text, org, color=(0, 255, 255), scale=0.6):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1)


def cmd_setup(args, cap, undistort):
    corners = get_corners(cap, undistort, recalibrate=True)
    if corners is None:
        print("Cancelled.")
        return
    print(f"Perspective ratio: {perspective_ratio(corners):.2f} (1.0 = overhead)")


def cmd_live(args, cap, undistort):
    corners = get_corners(cap, undistort)   # opens set-up if none saved
    if corners is None:
        return
    clf = PieceClassifier()
    expected = fen_to_classes(args.fen) if args.fen else None
    pred, cnn_ms, continuous = None, 0.0, False
    t_prev, fps = time.perf_counter(), 0.0
    win = "Board live  |  C = classify, A = continuous, ESC = quit"

    while True:
        ok, raw = cap.read()
        if not ok:
            print("No frames from the camera.")
            break
        frame = undistort(raw)

        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            break
        if key == ord("a"):
            continuous = not continuous
        if key == ord("c") or continuous:
            t0 = time.perf_counter()
            pred = clf.classify_board(frame, corners)
            cnn_ms = (time.perf_counter() - t0) * 1000

        view = draw_grid(frame, corners, labels=pred is None)
        if pred is not None:
            n_err = 0
            for sq, (cls, p) in pred.items():
                wrong = expected is not None and expected[sq] != cls
                n_err += wrong
                color = (0, 0, 255) if wrong else ((0, 165, 255) if p < 0.6 else (0, 255, 0))
                x, y = square_center(corners, sq)
                put(view, SYMBOL[cls], (x - 6, y + 6), color, 0.6)
            info = f"CNN {cnn_ms:.0f} ms"
            if expected is not None:
                info += f"  |  {64 - n_err}/64 vs FEN"
            put(view, info, (10, 56))

        t = time.perf_counter()
        fps = 0.9 * fps + 0.1 / max(t - t_prev, 1e-6)
        t_prev = t
        put(view, f"FPS {fps:.1f}" + ("  [continuous]" if continuous else ""), (10, 28))
        cv2.imshow(win, view)

    cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=("setup", "live"))
    ap.add_argument("--fen", help="live: position on the board, to count matching squares")
    ap.add_argument("--camera", type=int, default=config.CAMERA_INDEX)
    args = ap.parse_args()

    cap = open_camera(args.camera)
    undistort = Undistorter.from_file()
    try:
        {"setup": cmd_setup, "live": cmd_live}[args.command](args, cap, undistort)
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()