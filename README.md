# ChessVision

Real-time digitisation of a chess game played on a physical board, using a single camera and classical computer vision.

## Camera calibration

Offline calibration (Zhang's method) of the camera. The repo shows an example of calibration parameters obtained with the images of an iPhone 12 streamed to the PC with **Camo**. It is run once, before the system; at runtime every frame is undistorted with precomputed maps.

### Files involved

| File | Role |
|---|---|
| `src/chessvision/config.py` | Parameters: camera index, resolution, pattern shape, square size, paths |
| `scripts/calibrate.py` | Command-line tool: find the camera index, capture photos, calibrate |
| `src/chessvision/camera.py` | Utils needed: `open_camera()`, save/load calibration, `Undistorter` (runtime correction) |

### Workflow

Run from the repository root:

| Step | Command | Result |
|---|---|---|
| 1. Find the camera | `python scripts/calibrate.py probe` | Lists available camera indices → set `CAMERA_INDEX` in `config.py` |
| 2. Capture photos | `python scripts/calibrate.py capture` | ~20 photos in `data/calibration/images/` (SPACE = save, Q/ESC = quit) |
| 3. Calibrate | `python scripts/calibrate.py calibrate` | `data/calibration/calibration.npz` (K, dist, RMS) + `corners/` + `undistort_example.jpg` |

### Parameter configuration through console

> Parameters used in execution can be modified using flags in the call from the terminal.

| Option | Commands | Default | Description |
|---|---|---|---|
| `--images DIR` | `capture`, `calibrate` | `data/calibration/images` | Folder with the pattern photos |
| `--pattern CxR` | `capture`, `calibrate` | `9x6` | Inner corners (columns x rows), not squares, of the pattern used to calibrate |
| `--camera N` | `capture` | `1` | Camera index |
| `--width W` / `--height H` | `capture` | `1280` / `720` | Capture resolution (must match runtime) |
| `--square MM` | `calibrate` | `24.0` | Side of a printed square in mm |
| `--output FILE` | `calibrate` | `data/calibration/calibration.npz` | Output file |
| `--fix-k3` | `calibrate` | off | Fix k3 = 0 (if k3 blows up / edges fold) |

Defaults come from `config.py`.


> The calibration is only valid for the same lens, resolution, zoom and focus used during capture.

## Security system

The user unlocks the system by showing 4 distinct real chess pieces, one by one and in order, inside a fixed ROI.

- **Piece recognition (per frame).** Each piece type is recognised with **SIFT on the silhouette of the piece**: colour mask (HSV) → SIFT on the binary mask → brute-force matching + Lowe's ratio test (one-to-one) → RANSAC similarity with a scale check → vote per type. On an independent test set captured in another session it reaches **100 % accuracy with 0 % false positives** (~50 ms per frame).
- **Key decoder (over time).** A single frame can be wrong, so each piece is decided over several frames: presence by the area of the colour mask (with hysteresis), frames where the piece moves do not vote, and each still frame adds the inliers of its winning type. The piece is accepted if the winner gets ≥ 60 % of the votes in ≥ 8 recognised frames (or what was collected in 4 s). It must be removed before the next one; after 4 pieces the sequence is compared with the key.

### Files involved

| File | Role |
|---|---|
| `src/chessvision/config.py` | Parameters: ROI, reference set, HSV range of the pieces, SIFT thresholds, key and decoder thresholds |
| `src/chessvision/security/piece_matcher.py` | `PieceMatcher`: SIFT features, references, matching and per-frame decision |
| `src/chessvision/security/decoder.py` | `KeyDecoder`: state machine that turns per-frame recognitions into the 4-piece key |
| `scripts/security.py` | Command-line tool: capture reference photos, evaluate, live test, unlock |
| `data/security/<set>/<piece>/NNN.jpg` | Photo sets (ROI crops). The reference set *is* the model: it is loaded at runtime |

### Workflow

Run from the repository root. Show each piece **upright, centred in the ROI and held by its base**.

| Step | Command | Result |
|---|---|---|
| 1. Capture references | `python scripts/security.py capture` | 15–20 photos per piece in `data/security/upright/` (P N B R Q K = piece, X = no piece, ESC = quit) |
| 2. Capture a test set (another day / light) | `python scripts/security.py capture --set upright_test` | 8–10 photos per piece + 5–8 of objects that pass the colour mask (X) |
| 3. Evaluate | `python scripts/security.py evaluate --test upright_test` | Accuracy, false positives on empty ROI, threshold sweep, confusion matrix |
| 4. Live test | `python scripts/security.py live` | Per-frame recognised piece, votes and FPS on the camera stream |
| 5. Unlock | `python scripts/security.py unlock` | Full system: show the 4 key pieces (`SECURITY_KEY`), one at a time, removing each one (R = reset, ESC = quit) |

Without `--test`, `evaluate` runs leave-one-out on the reference set (optimistic: consecutive photos are very similar).

### Options

| Option | Commands | Default | Description |
|---|---|---|---|
| `--set NAME` | all | `upright` | Photo set (subfolder of `data/security`); the reference set for `evaluate` / `live` / `unlock` |
| `--test NAME` | `evaluate` | none (leave-one-out) | Independent test set |
| `--key A,B,C,D` | `unlock` | `knight,rook,bishop,queen` | Key: 4 distinct piece types, in order |
| `--features MODE` | all | `silhouette` | What SIFT looks at: `silhouette` (binary mask), `texture` (grey image inside the mask), `raw` (whole ROI) |

Defaults come from `config.py`.

> The colour mask is tuned for light (yellow) pieces. If the hand or the wall appear lit in the `capture` window, adjust `PIECE_HSV_LOW` / `PIECE_HSV_HIGH`.
