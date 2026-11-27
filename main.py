"""ChessVision: full real-time system with the modules implemented so far.
 
    1. Security   - show the 4 key pieces in the ROI, one at a time (skip with --no-security)
    2. Board      - saved corners, or click a1, h1, h8, a8 (redo with --recalibrate)
    3. Live       - hand tracker (MOG2 + Kalman); every time the hand leaves the board and the
                    scene is still ("settled") the move detector (moves/detector.py) re-reads only
                    the squares that changed in the image, applies the piece-count rules and gets
                    the move. The digital board shows the position; the move squares are highlighted.
 
Usage (from the repository root):
    python main.py
    python main.py --no-security            # straight to the board
    python main.py --recalibrate            # click the board corners again
    python main.py --scale 0.6              # smaller window
    python main.py --fen initial            # assume the initial position instead of reading it with the CNN
    python main.py --fen "<FEN>"            # start from a given position
 
Live keys:  C = read the board now | R = re-read the whole position | B = re-learn background |
            M = foreground mask | ESC/Q = quit
Security keys: R = reset the key | S = skip | ESC = quit
"""
from __future__ import annotations
 
import argparse
import sys
import time
from collections import deque
from pathlib import Path
 
import cv2
import numpy as np
 
# So that Python finds src/chessvision wherever it is run from
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from chessvision import config                                          # noqa: E402
from chessvision.board.corners import INITIAL_FEN, draw_grid, get_corners  # noqa: E402
from chessvision.board.geometry import FILES                            # noqa: E402
from chessvision.camera import Undistorter, open_camera, read_image     # noqa: E402
from chessvision.moves.detector import MoveDetector                     # noqa: E402
from chessvision.pieces.classifier import PieceClassifier               # noqa: E402
from chessvision.security import decoder as dec                         # noqa: E402
from chessvision.security.piece_matcher import PieceMatcher             # noqa: E402
from chessvision.tracker.hand import HandTracker, draw_track            # noqa: E402
 
WINDOW = "ChessVision"
SYMBOL = {"empty": "", "white_pawn": "P", "white_knight": "N", "white_bishop": "B",
          "white_rook": "R", "white_queen": "Q", "white_king": "K",
          "black_pawn": "p", "black_knight": "n", "black_bishop": "b",
          "black_rook": "r", "black_queen": "q", "black_king": "k"}
STATE_COLOR = {"idle": (0, 255, 0), "hand": (0, 0, 255), "settling": (0, 200, 255)}
LOW_CONF = 0.6
 
 
# --- Drawing helpers ------------------------------------------------------------------
def put(img, text, org, color=(0, 255, 255), scale=0.7, thick=2):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)
 
 
def draw_digital_board(pred: dict | None, changed: set[str], size: int) -> np.ndarray:
    """Digital board (White at the bottom) with the CNN reading of each square."""
    m = size // 16                                     # margin for the coordinates
    s = (size - 2 * m) // 8
    img = np.full((size, size, 3), 40, np.uint8)
    for r in range(8):                                 # r = 0 -> rank 8 (top)
        for c in range(8):
            sq = f"{FILES[c]}{8 - r}"
            x, y = m + c * s, m + r * s
            light = (r + c) % 2 == 0
            cv2.rectangle(img, (x, y), (x + s, y + s), (181, 217, 240) if light else (99, 136, 181), -1)
            if sq in changed:
                cv2.rectangle(img, (x + 2, y + 2), (x + s - 2, y + s - 2), (0, 220, 255), 3)
            if pred is None:
                continue
            cls, p = pred[sq]
            if cls == "empty":
                continue
            white = cls.startswith("white")
            cx, cy = x + s // 2, y + s // 2
            cv2.circle(img, (cx, cy), int(s * 0.38), (245, 245, 245) if white else (30, 30, 30), -1)
            cv2.circle(img, (cx, cy), int(s * 0.38), (0, 165, 255) if p < LOW_CONF else (90, 90, 90), 2)
            letter = SYMBOL[cls].upper()
            (tw, th), _ = cv2.getTextSize(letter, cv2.FONT_HERSHEY_SIMPLEX, s / 70, 2)
            cv2.putText(img, letter, (cx - tw // 2, cy + th // 2), cv2.FONT_HERSHEY_SIMPLEX, s / 70,
                        (30, 30, 30) if white else (245, 245, 245), 2, cv2.LINE_AA)
    for i in range(8):
        cv2.putText(img, FILES[i], (m + i * s + s // 2 - 6, size - m // 4), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (220, 220, 220), 1, cv2.LINE_AA)
        cv2.putText(img, str(8 - i), (m // 4, m + i * s + s // 2 + 6), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (220, 220, 220), 1, cv2.LINE_AA)
    return img
 
 
def board_to_fen(pred: dict) -> str:
    rows = []
    for rank in range(8, 0, -1):
        row, empty = "", 0
        for f in FILES:
            s = SYMBOL[pred[f"{f}{rank}"][0]]
            if not s:
                empty += 1
            else:
                row += (str(empty) if empty else "") + s
                empty = 0
        rows.append(row + (str(empty) if empty else ""))
    return "/".join(rows)
 
 
class Fps:
    def __init__(self):
        self.prev, self.value = time.perf_counter(), 0.0
 
    def tick(self) -> float:
        now = time.perf_counter()
        inst = 1.0 / max(now - self.prev, 1e-6)
        self.value = inst if self.value == 0 else 0.9 * self.value + 0.1 * inst
        self.prev = now
        return self.value
 
 
# --- 1. Security ------------------------------------------------------------------------
def build_matcher() -> PieceMatcher | None:
    set_dir = config.SECURITY_DIR / config.SECURITY_REFERENCE_SET
    matcher, n = PieceMatcher(), 0
    for label in config.PIECE_TYPES:
        for p in sorted((set_dir / label).glob("*.jpg")):
            img = read_image(p)
            if img is not None:
                matcher.add_reference(label, img, ref_id=n)
                n += 1
    return matcher if n else None
 
 
def run_security(cap, undistort, scale: float) -> bool:
    """Returns True when unlocked (or skipped with S), False if the user quits."""
    matcher = build_matcher()
    if matcher is None:
        print("[security] No reference photos in data/security/: security skipped.")
        return True
    decoder = dec.KeyDecoder(matcher)
    x, y, w, h = config.SECURITY_ROI
    fps, event, event_t = Fps(), None, 0.0
    text = {dec.IDLE: "Show a key piece in the box", dec.OBSERVING: "Hold the piece still...",
            dec.WAIT_REMOVAL: "Remove the piece", dec.UNLOCKED: "UNLOCKED"}
    print("[security] Show the 4 key pieces one at a time (R = reset, S = skip, ESC = quit)")
 
    while True:
        ok, raw = cap.read()
        if not ok:
            return False
        frame = undistort(raw)
        roi = frame[y:y + h, x:x + w]
        new = decoder.update(roi)
        if new is not None:
            event, event_t = new, time.monotonic()
            print(f"[security] {new.kind}: {new.message}")
 
        view = frame.copy()
        mask = decoder.matcher.piece_mask(roi)
        if mask is not None:
            view[y:y + h, x:x + w][mask == 0] //= 2
        color = {dec.IDLE: (200, 200, 200), dec.OBSERVING: (0, 255, 255),
                 dec.WAIT_REMOVAL: (255, 255, 0), dec.UNLOCKED: (0, 255, 0)}[decoder.state]
        cv2.rectangle(view, (x, y), (x + w, y + h), color, 2)
        if decoder.progress > 0:
            cv2.rectangle(view, (x, y + h + 8), (x + int(w * decoder.progress), y + h + 20), (0, 255, 255), -1)
        put(view, f"SECURITY - {text[decoder.state]}", (15, 35), color, 0.9)
        put(view, f"FPS {fps.tick():.1f}", (15, 70))
        for i in range(4):
            x0 = 15 + i * 150
            filled = i < len(decoder.entries)
            cv2.rectangle(view, (x0, 600), (x0 + 140, 640), (0, 255, 0) if filled else (120, 120, 120), 2)
            put(view, "*" if filled else "?", (x0 + 60, 632), (255, 255, 255), 0.9)   # the key is not shown
        if event is not None and time.monotonic() - event_t < 3.0:
            ok_ev = event.kind in ("accepted", "unlocked")
            put(view, event.message, (15, 690), (0, 255, 0) if ok_ev else (0, 0, 255), 0.8)
        cv2.imshow(WINDOW, cv2.resize(view, None, fx=scale, fy=scale))
 
        if new is not None and new.kind == "unlocked":
            cv2.waitKey(1000)                          # let the user see it
            return True
        k = cv2.waitKey(1) & 0xFF
        if k in (27, ord("q")):
            return False
        if k == ord("r"):
            decoder.reset()
        if k == ord("s"):
            print("[security] Skipped.")
            return True
 
 
# --- 3. Live ------------------------------------------------------------------------------
def run_live(cap, undistort, corners: dict, scale: float, start_fen: str) -> None:
    clf = PieceClassifier()
    ok, raw = cap.read()
    if not ok:
        return
    frame = undistort(raw)
    tracker = HandTracker(corners, frame.shape)
    detector = MoveDetector(clf, corners)
    recent = deque([frame], maxlen=config.MOVE_AVG_FRAMES)   # last frames, averaged by the CNN
    fps = Fps()
    show_mask, last_event, n = False, "", 0
    h = frame.shape[0]
 
    def start_position(frames) -> None:
        if start_fen == "read":                        # CNN + rules on the 64 squares
            detector.reset(frames[-1], detector.read_position(list(frames)))
        else:
            detector.reset(frames[-1], INITIAL_FEN if start_fen == "initial" else start_fen)
 
    def as_pred(conf: dict[str, float] | None = None) -> dict:
        """Detector position in the format of draw_digital_board / board_to_fen."""
        conf = conf or {}
        return {sq: (c, conf.get(sq, 1.0)) for sq, c in detector.position.items()}
 
    while len(recent) < config.MOVE_AVG_FRAMES:        # a few frames for the initial reading
        ok, raw = cap.read()
        if not ok:
            return
        recent.append(undistort(raw))
    start_position(recent)
    pred, changed, cnn_ms = as_pred(), set(), 0.0
    moves: list[str] = []
    last_msg, msg_color = "", (255, 255, 255)
    print(f"[board] start: {board_to_fen(pred)}")
    print("[live] C = read now, R = re-read position, B = re-learn background, M = mask, ESC = quit")
 
    while True:
        ok, raw = cap.read()
        if not ok:
            break
        frame = undistort(raw)
        recent.append(frame)
        n += 1
        res = tracker.update(frame)
 
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            break
        if key == ord("m"):
            show_mask = not show_mask
        if key == ord("b"):
            tracker.relearn(frame)
        if key == ord("r"):                            # resynchronise: whole position from the CNN + rules
            detector.reset(frame, detector.read_position(list(recent)))
            pred, changed = as_pred(), set()
            last_msg, msg_color = "position re-read", (0, 255, 255)
            print(f"[board] re-read: {board_to_fen(pred)}")
        if res.event:
            last_event = res.event
        if res.event == "settled" or key == ord("c"):
            t0 = time.perf_counter()
            mv = detector.update(list(recent))
            cnn_ms = (time.perf_counter() - t0) * 1000
            if mv.ok and mv.move:
                moves.append(mv.move)
                changed = {sq for sq, _, _ in mv.changes}
                pred = as_pred({sq: mv.confidence for sq in changed})
                last_msg = f"{len(moves)}. {mv.move} {mv.piece}  (conf {mv.confidence:.2f})"
                msg_color = (0, 165, 255) if mv.confidence < LOW_CONF or mv.overridden else (0, 255, 0)
                print(f"[move] {last_msg}   {board_to_fen(pred)}   ({cnn_ms:.0f} ms)")
                for sq, (cnn, rule) in mv.overridden.items():
                    print(f"[move]   rules changed {sq}: CNN {cnn} -> {rule}")
            elif mv.ok:
                changed = set()
                last_msg, msg_color = "no change", (200, 200, 200)
            else:
                changed = set(mv.changed_squares)
                last_msg, msg_color = f"unclear: {mv.reason}", (0, 0, 255)
                print(f"[move] not committed: {mv.reason}   changed in image: {', '.join(mv.changed_squares)}")
                for sq, (cnn, rule) in mv.overridden.items():
                    print(f"[move]   rules changed {sq}: CNN {cnn} -> {rule}")
 
        view = draw_track(draw_grid(frame, corners, labels=False), res)
        if show_mask:
            sel = res.fg_mask > 0
            view[sel] = (0.5 * view[sel] + (0, 0, 127)).astype(np.uint8)
        put(view, f"FPS {fps.tick():.1f}", (15, 35))
        put(view, f"hand: {res.state}", (15, 70), STATE_COLOR[res.state])
        if res.tip_square and res.state == "hand":
            put(view, f"over {res.tip_square}", (15, 105), (0, 255, 255))
        if last_msg:
            put(view, last_msg, (15, 140), msg_color)
        put(view, f"last event: {last_event}   |   move read {cnn_ms:.0f} ms   |   moves: {len(moves)}",
            (15, h - 20), (255, 255, 255), 0.6, 1)
 
        panel = draw_digital_board(pred, changed, h)
        if res.state != "idle":                         # the reading is waiting for the hand to leave
            put(panel, "waiting for the hand...", (20, 25), (0, 200, 255), 0.6, 1)
        out = np.hstack([view, panel])
        cv2.imshow(WINDOW, cv2.resize(out, None, fx=scale, fy=scale) if scale != 1 else out)
 
 
# --- main ---------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-security", action="store_true", help="skip the security step")
    ap.add_argument("--recalibrate", action="store_true", help="click the board corners again")
    ap.add_argument("--scale", type=float, default=0.75, help="window scale (default 0.75)")
    ap.add_argument("--camera", type=int, default=config.CAMERA_INDEX)
    ap.add_argument("--fen", default="read",
                    help='start position: "read" = CNN + rules (default), "initial", or a FEN')
    args = ap.parse_args()
 
    cap = open_camera(args.camera)
    undistort = Undistorter.from_file()
    try:
        if not args.no_security and not run_security(cap, undistort, args.scale):
            return
        cv2.destroyAllWindows()
        corners = get_corners(cap, undistort, recalibrate=args.recalibrate)
        if corners is None:
            print("No board corners: quitting.")
            return
        run_live(cap, undistort, corners, args.scale, args.fen)
    finally:
        cap.release()
        cv2.destroyAllWindows()
 
 
if __name__ == "__main__":
    main()