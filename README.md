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
