"""Security system tool: reference photos, evaluation and live test of the SIFT piece recognition.

Usage (from the repository root; --set defaults to the reference set in config.py):
    python scripts/security.py capture                     # take photos of each piece in the ROI
    python scripts/security.py capture --set upright_test  # ... into another set (e.g. a test set)
    python scripts/security.py evaluate                    # leave-one-out accuracy + confusion matrix
    python scripts/security.py evaluate --test upright_test  # accuracy on an independent test set
    python scripts/security.py live                        # real-time recognition with FPS
Add --features {silhouette,texture,raw} to any command to choose what SIFT looks at
(default in config.py; the area outside the piece mask is shown darkened).

Capture keys:  P N B R Q K = pawn knight bishop rook queen king | X = empty ROI (no piece) | ESC = quit
Photos are saved as data/security/<set>/<piece>/NNN.jpg (only the ROI crop).
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
    prev = time.perf_counter()

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = undistort(frame)
        roi = crop_roi(frame)
        label, votes, n_kp = matcher.classify(roi)

        now = time.perf_counter()
        fps, prev = 1.0 / (now - prev), now
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


# --- CLI -----------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    for name, func, text in (("capture", capture, "capture photos of each piece in the ROI"),
                             ("evaluate", evaluate, "offline accuracy and confusion matrix"),
                             ("live", live, "real-time recognition")):
        p = sub.add_parser(name, help=text)
        p.add_argument("--set", default=config.SECURITY_REFERENCE_SET,
                       help="photo set (subfolder of data/security); the reference set for evaluate/live")
        p.add_argument("--features", choices=FEATURE_MODES, default=config.SIFT_FEATURES,
                       help="silhouette = binary mask of the piece, texture = grey image inside "
                            "the mask, raw = grey image of the whole ROI")
        if name == "evaluate":
            p.add_argument("--test", help="independent test set (otherwise leave-one-out)")
        p.set_defaults(func=func)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
