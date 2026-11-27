"""Hand tracker tool: live test on the camera or on a recorded video, and recording of test videos.

Usage (from the repository root; the board corners must be set: python scripts/board.py setup):
    python scripts/tracker.py live                       # camera: contour, Kalman, tip square, state, FPS
    python scripts/tracker.py live --video test.mp4      # same on a recorded (undistorted) video
    python scripts/tracker.py record test.mp4            # record undistorted camera video to test offline later

Keys: SPACE = pause | B = re-learn background | M = show/hide foreground mask | ESC/Q = quit
Events (hand_in, hand_out, settled) are printed with the frame number.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from chessvision import config                                    # noqa: E402
from chessvision.board.corners import draw_grid, load_corners     # noqa: E402
from chessvision.camera import Undistorter, open_camera           # noqa: E402
from chessvision.tracker.hand import HandTracker, draw_track      # noqa: E402

STATE_COLOR = {"idle": (0, 255, 0), "hand": (0, 0, 255), "settling": (0, 200, 255)}


def put(img, text, org, color=(0, 255, 255), scale=0.6):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1)


def frames(args):
    """Yields undistorted frames from the camera or (already undistorted) frames from a video."""
    if args.video:
        cap = cv2.VideoCapture(str(args.video))
        undistort = None
    else:
        cap = open_camera(args.camera)
        undistort = Undistorter.from_file()
    try:
        while True:
            ok, f = cap.read()
            if not ok:
                return
            yield undistort(f) if undistort else f
    finally:
        cap.release()


def cmd_live(args):
    it = frames(args)
    first = next(it, None)
    if first is None:
        print("No frames.")
        return
    h, w = first.shape[:2]
    corners = load_corners(frame_size=(w, h))
    if corners is None:
        print("No board corners for this resolution. Run: python scripts/board.py setup")
        return
    tracker = HandTracker(corners, first.shape)
    show_mask, paused, n = False, False, 0
    t_prev, fps = time.perf_counter(), 0.0
    last_event = ""
    frame = first

    while True:
        if not paused:
            frame = next(it, None)
            if frame is None:
                break
            n += 1
            res = tracker.update(frame)
            if res.event:
                last_event = f"{res.event} @ {n}"
                print(f"[{n:5d}] {res.event}" + (f"  (tip on {res.tip_square})" if res.tip_square else ""))
            view = draw_track(draw_grid(frame, corners, labels=False), res)
            if show_mask:
                view[res.fg_mask > 0] = (0.5 * view[res.fg_mask > 0] + (0, 0, 127)).astype("uint8")
            t = time.perf_counter()
            fps = 0.9 * fps + 0.1 / max(t - t_prev, 1e-6)
            t_prev = t
            put(view, f"FPS {fps:.1f}", (10, 28))
            put(view, f"state: {res.state}", (10, 56), STATE_COLOR[res.state])
            put(view, f"last event: {last_event}", (10, 84))
            cv2.imshow("Hand tracker  |  SPACE pause, B relearn, M mask, ESC quit", view)

        key = cv2.waitKey(1 if not args.video else 30) & 0xFF
        if key in (27, ord("q")):
            break
        if key == ord(" "):
            paused = not paused
        elif key == ord("m"):
            show_mask = not show_mask
        elif key == ord("b"):
            tracker.relearn(frame)
    cv2.destroyAllWindows()


def cmd_record(args):
    cap = open_camera(args.camera)
    undistort = Undistorter.from_file()
    out = None
    print("Recording... ESC/Q to stop.")
    try:
        while True:
            ok, raw = cap.read()
            if not ok:
                break
            frame = undistort(raw)
            if out is None:
                h, w = frame.shape[:2]
                out = cv2.VideoWriter(str(args.output), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (w, h))
            out.write(frame)
            view = frame.copy()
            put(view, "REC", (10, 28), (0, 0, 255))
            cv2.imshow("Recording  |  ESC = stop", view)
            if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                break
    finally:
        cap.release()
        if out is not None:
            out.release()
        cv2.destroyAllWindows()
    print(f"Saved {args.output}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    p = sub.add_parser("live")
    p.add_argument("--video", type=Path, help="recorded video (already undistorted) instead of the camera")
    p.add_argument("--camera", type=int, default=config.CAMERA_INDEX)
    p = sub.add_parser("record")
    p.add_argument("output", type=Path)
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--camera", type=int, default=config.CAMERA_INDEX)
    args = ap.parse_args()
    {"live": cmd_live, "record": cmd_record}[args.command](args)


if __name__ == "__main__":
    main()