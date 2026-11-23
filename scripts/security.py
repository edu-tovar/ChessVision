"""Security system tool: reference photos, evaluation and live test of the SIFT piece recognition.

Usage (from the repository root; --set defaults to the reference set in config.py):
    python scripts/security.py capture                     # take photos of each piece in the ROI
    python scripts/security.py capture --set upright_test  # ... into another set (e.g. a test set)
    python scripts/security.py evaluate                    # leave-one-out accuracy + confusion matrix
    python scripts/security.py evaluate --test upright_test  # accuracy on an independent test set
    python scripts/security.py live                        # real-time recognition with FPS
    python scripts/security.py unlock                      # full security system: enter the 4-piece key
    python scripts/security.py unlock --key pawn,rook,king,queen   # ... with another key
Add --features {silhouette,texture,raw} to any command to choose what SIFT looks at
(default in config.py; the area outside the piece mask is shown darkened).

Capture keys:  P N B R Q K = pawn knight bishop rook queen king | X = empty ROI (no piece) | ESC = quit
Photos are saved as data/security/<set>/<piece>/NNN.jpg (only the ROI crop).
Unlock keys:   R = reset the key | ESC = quit
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from chessvision import config                                       # noqa: E402
from chessvision.camera import Undistorter, open_camera, read_image, write_image  # noqa: E402
from chessvision.security import decoder as dec                      # noqa: E402
from chessvision.security.piece_matcher import FEATURE_MODES, PieceMatcher  # noqa: E402

KEYS = {ord("p"): "pawn", ord("n"): "knight", ord("b"): "bishop", ord("r"): "rook",
        ord("q"): "queen", ord("k"): "king", ord("x"): "none"}
LABELS = list(config.PIECE_TYPES) + ["none"]


# --- Helpers -------------------------------------------------------------------
def crop_roi(frame: np.ndarray) -> np.ndarray:
    x, y, w, h = config.SECURITY_ROI
    return frame[y:y + h, x:x + w]


def draw_roi(view: np.ndarray, color=(255, 255, 0)) -> None:
    x, y, w, h = config.SECURITY_ROI
    cv2.rectangle(view, (x, y), (x + w, y + h), color, 2)


def show_roi(view: np.ndarray, roi: np.ndarray, mask, kp=None) -> None:
    """Paint the ROI into the view, darkening what lies outside the piece mask."""
    shown = roi.copy()
    if mask is not None:
        shown[mask == 0] //= 3
    if kp is not None:
        shown = cv2.drawKeypoints(shown, kp, None, (0, 255, 0))
    x, y, _, _ = config.SECURITY_ROI
    view[y:y + roi.shape[0], x:x + roi.shape[1]] = shown


def get_undistorter():
    try:
        return Undistorter.from_file()
    except FileNotFoundError:
        print("Warning: no calibration found, frames are not undistorted.")
        return lambda frame: frame


def load_set(set_dir: Path) -> list[tuple[str, Path, np.ndarray]]:
    data = []
    for label in LABELS:
        for p in sorted((set_dir / label).glob("*.jpg")):
            img = read_image(p)
            if img is not None:
                data.append((label, p, img))
    return data


def build_matcher(data, features: str = config.SIFT_FEATURES) -> PieceMatcher:
    matcher = PieceMatcher(features=features)
    for i, (label, _, img) in enumerate(data):
        if label != "none":
            matcher.add_reference(label, img, ref_id=i)
    return matcher


# --- capture -------------------------------------------------------------------
def capture(args) -> None:
    set_dir = config.SECURITY_DIR / args.set
    undistort = get_undistorter()
    sift = PieceMatcher(features=args.features)   # only used to show live keypoints
    cap = open_camera()
    print(__doc__.split("Capture keys:")[1].split("Photos")[0].strip())

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = undistort(frame)
        roi = crop_roi(frame)
        kp, _ = sift.features(roi)

        view = frame.copy()
        show_roi(view, roi, sift.piece_mask(roi), kp)   # the saved photo is the clean ROI
        draw_roi(view)
        cv2.putText(view, f"Keypoints in ROI: {len(kp)}", (15, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        cv2.imshow("Security - capture", view)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        if key in KEYS:
            label = KEYS[key]
            folder = set_dir / label
            numbers = [int(p.stem) for p in folder.glob("*.jpg") if p.stem.isdigit()]
            idx = max(numbers, default=-1) + 1          # never overwrites previous photos
            write_image(folder / f"{idx:03d}.jpg", roi)
            print(f"  Saved {label}/{idx:03d}.jpg  ({len(kp)} keypoints)")

    cap.release()
    cv2.destroyAllWindows()


# --- evaluate ------------------------------------------------------------------
def evaluate(args) -> None:
    ref_data = load_set(config.SECURITY_DIR / args.set)
    if args.test:
        test_data = load_set(config.SECURITY_DIR / args.test)
        mode = f"references '{args.set}' vs test '{args.test}'"
    else:
        test_data = ref_data
        mode = f"leave-one-out on '{args.set}'"
    if not ref_data or not test_data:
        sys.exit("No photos found. Run 'capture' first.")

    matcher = build_matcher(ref_data, args.features)
    print(f"Mode: {mode} | features '{args.features}' | "
          f"{len(matcher.refs)} reference images | {len(test_data)} test images\n")

    # Votes are computed once; the decision threshold is swept afterwards
    results = []                               # (true label, votes, n keypoints)
    for i, (label, _, img) in enumerate(test_data):
        kp, des = matcher.features(img)
        exclude = i if not args.test else None   # leave-one-out: never match an image with itself
        results.append((label, matcher.scores(kp, des, exclude_id=exclude), len(kp)))

    # 1) Keypoints per piece type
    print("Keypoints per image:")
    for label in LABELS:
        n = [k for t, _, k in results if t == label]
        if n:
            print(f"  {label:7s}: {np.mean(n):6.1f} mean | {min(n):4d} min | {len(n)} images")

    # 2) Votes of the correct type vs the best wrong type (separation)
    print("\nInliers: correct type vs best wrong type (mean):")
    for label in config.PIECE_TYPES:
        rows = [v for t, v, _ in results if t == label]
        if rows:
            right = np.mean([v[label] for v in rows])
            wrong = np.mean([max(c for l, c in v.items() if l != label) for v in rows])
            print(f"  {label:7s}: {right:6.1f} vs {wrong:6.1f}")

    # 3) Threshold sweep
    def metrics(th):
        pieces = [(t, matcher.decide(v, th)) for t, v, _ in results if t != "none"]
        empties = [matcher.decide(v, th) for t, v, _ in results if t == "none"]
        acc = np.mean([t == p for t, p in pieces]) if pieces else float("nan")
        fp = np.mean([p is not None for p in empties]) if empties else float("nan")
        return acc, fp

    print("\nThreshold sweep (min inliers -> piece accuracy | false positives on empty ROI):")
    sweep = []
    for th in range(4, 41, 2):
        acc, fp = metrics(th)
        sweep.append((th, acc, fp))
        print(f"  {th:3d}: {acc * 100:5.1f} % | {fp * 100:5.1f} %" if not np.isnan(fp)
              else f"  {th:3d}: {acc * 100:5.1f} % | (no empty photos)")
    best_th = max(sweep, key=lambda s: (s[1] - (0 if np.isnan(s[2]) else s[2]), -s[0]))[0]

    # 4) Confusion matrix at the best threshold
    print(f"\nConfusion matrix (threshold = {best_th}) rows = true, columns = predicted:")
    cols = LABELS
    print(" " * 9 + "".join(f"{c[:6]:>7s}" for c in cols))
    for label in LABELS:
        rows = [matcher.decide(v, best_th) or "none" for t, v, _ in results if t == label]
        if rows:
            print(f"  {label:7s}" + "".join(f"{rows.count(c):7d}" for c in cols))

    acc, fp = metrics(best_th)
    print(f"\nPiece accuracy: {acc * 100:.1f} %" +
          ("" if np.isnan(fp) else f" | false positives on empty ROI: {fp * 100:.1f} %"))
    verdict = "VIABLE" if acc >= 0.9 and (np.isnan(fp) or fp <= 0.05) else "NOT VIABLE as it is"
    print(f"Verdict: {verdict} (criterion: accuracy >= 90 % and false positives <= 5 %)")
    if not args.test:
        print("Note: leave-one-out with very similar consecutive photos is optimistic. "
              "Capture a second set in another session and use --test.")


# --- live ----------------------------------------------------------------------
def live(args) -> None:
    data = load_set(config.SECURITY_DIR / args.set)
    if not data:
        sys.exit("No photos found. Run 'capture' first.")
    matcher = build_matcher(data, args.features)
    undistort = get_undistorter()
    cap = open_camera()
    prev, fps = time.perf_counter(), 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = undistort(frame)
        roi = crop_roi(frame)
        label, votes, n_kp = matcher.classify(roi)

        now = time.perf_counter()
        fps = 0.9 * fps + 0.1 / (now - prev) if fps else 1.0 / (now - prev)   # smoothed
        prev = now
        view = frame.copy()
        show_roi(view, roi, matcher.piece_mask(roi))
        draw_roi(view, (0, 255, 0) if label else (0, 0, 255))
        cv2.putText(view, f"{label or '-'}  |  {n_kp} kp  |  {fps:4.1f} FPS", (15, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        for j, (name, v) in enumerate(sorted(votes.items(), key=lambda t: -t[1])):
            cv2.putText(view, f"{name}: {v}", (15, 75 + 30 * j),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.imshow("Security - live", view)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


# --- unlock --------------------------------------------------------------------
STATE_COLOURS = {dec.IDLE: (160, 160, 160), dec.OBSERVING: (0, 255, 255),
                 dec.WAIT_REMOVAL: (255, 255, 0), dec.UNLOCKED: (0, 255, 0)}
STATE_TEXT = {dec.IDLE: "Show a piece", dec.OBSERVING: "Hold the piece still...",
              dec.WAIT_REMOVAL: "Remove the piece", dec.UNLOCKED: "UNLOCKED (R = lock again)"}
EVENT_COLOURS = {"accepted": (0, 255, 0), "unlocked": (0, 255, 0), "rejected": (0, 0, 255),
                 "wrong_key": (0, 0, 255)}


def put_text(view, text, org, scale, colour) -> None:
    """Text with a dark outline, readable on any background."""
    cv2.putText(view, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 5)
    cv2.putText(view, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, colour, 2)


def draw_unlock(view, decoder, fps, event, event_time) -> None:
    x, y, w, h = config.SECURITY_ROI
    draw_roi(view, STATE_COLOURS[decoder.state])
    put_text(view, f"{STATE_TEXT[decoder.state]}   {fps:4.1f} FPS", (15, 35), 1.0,
             STATE_COLOURS[decoder.state])
    # Progress of the current observation, under the ROI
    if decoder.progress > 0:
        cv2.rectangle(view, (x, y + h + 8), (x + int(w * decoder.progress), y + h + 20), (0, 255, 255), -1)
    # What the last frame saw
    if decoder.state == dec.OBSERVING:
        seen = "moving" if decoder.moving else (decoder.frame_label or "-")
        n = decoder.frame_votes.get(decoder.frame_label, 0) if decoder.frame_label else 0
        put_text(view, f"frame: {seen}" + (f" ({n})" if n else ""), (x, y - 12), 0.7, (255, 255, 255))
    # Entered key: 4 slots
    for i in range(len(decoder.key)):
        x0 = 15 + i * 150
        filled = i < len(decoder.entries)
        cv2.rectangle(view, (x0, 600), (x0 + 140, 640), (0, 255, 0) if filled else (120, 120, 120), 2)
        put_text(view, decoder.entries[i] if filled else "?", (x0 + 10, 630), 0.8, (255, 255, 255))
    # Last event, for 3 s
    if event is not None and time.monotonic() - event_time < 3.0:
        put_text(view, event.message, (15, 690), 0.9, EVENT_COLOURS.get(event.kind, (0, 165, 255)))


def unlock(args) -> None:
    data = load_set(config.SECURITY_DIR / args.set)
    if not data:
        sys.exit("No photos found. Run 'capture' first.")
    key = tuple(args.key.split(",")) if args.key else config.SECURITY_KEY
    decoder = dec.KeyDecoder(build_matcher(data, args.features), key)
    print(f"Key: {len(key)} pieces (set in config.SECURITY_KEY or with --key). R = reset, ESC = quit")
    undistort = get_undistorter()
    cap = open_camera()
    prev, fps = time.perf_counter(), 0.0
    event, event_time = None, 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = undistort(frame)
        roi = crop_roi(frame)
        new = decoder.update(roi)
        if new is not None:
            event, event_time = new, time.monotonic()
            print(f"  [{time.strftime('%H:%M:%S')}] {new.kind:9s} {new.message}"
                  + (f"\n{'':22s}({new.stats})" if new.stats else ""))

        now = time.perf_counter()
        fps = 0.9 * fps + 0.1 / (now - prev) if fps else 1.0 / (now - prev)
        prev = now
        view = frame.copy()
        show_roi(view, roi, decoder.matcher.piece_mask(roi))
        draw_unlock(view, decoder, fps, event, event_time)
        cv2.imshow("Security - unlock", view)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break
        if k == ord("r"):
            decoder.reset()
            event, event_time = dec.Event("reset", message="Key reset"), time.monotonic()

    cap.release()
    cv2.destroyAllWindows()


# --- CLI -----------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    for name, func, text in (("capture", capture, "capture photos of each piece in the ROI"),
                             ("evaluate", evaluate, "offline accuracy and confusion matrix"),
                             ("live", live, "real-time recognition"),
                             ("unlock", unlock, "full security system: enter the 4-piece key")):
        p = sub.add_parser(name, help=text)
        p.add_argument("--set", default=config.SECURITY_REFERENCE_SET,
                       help="photo set (subfolder of data/security); the reference set for evaluate/live")
        p.add_argument("--features", choices=FEATURE_MODES, default=config.SIFT_FEATURES,
                       help="silhouette = binary mask of the piece, texture = grey image inside "
                            "the mask, raw = grey image of the whole ROI")
        if name == "evaluate":
            p.add_argument("--test", help="independent test set (otherwise leave-one-out)")
        if name == "unlock":
            p.add_argument("--key", help="comma-separated key, e.g. pawn,rook,king,queen "
                                         "(default: SECURITY_KEY in config.py)")
        p.set_defaults(func=func)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
