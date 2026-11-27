"""Hand tracker: MOG2 background subtraction + morphology + largest contour + Kalman filter.

Only the board (plus a margin, where the arm comes in) is analysed. The tracker also tells
the app WHEN to read the board:

    IDLE --hand appears--> HAND --hand leaves--> SETTLING --scene still N frames--> IDLE
                                                    |  emits "settled": read the position now
                                                    +--hand comes back--> HAND

After "settled" the background model is re-initialised with the current frame, so the pieces
that were just moved become background and do not show up as foreground in the next move.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from chessvision import config
from chessvision.board.corners import board_mask, board_to_image_homography
from chessvision.board.geometry import FILES


@dataclass
class TrackResult:
    state: str                                  # "idle" | "hand" | "settling"
    event: str | None = None                    # "hand_in" | "hand_out" | "settled" | None
    detected: bool = False                      # hand measured in this frame
    contour: np.ndarray | None = None
    bbox: tuple[int, int, int, int] | None = None
    centroid: tuple[float, float] | None = None     # measured
    position: tuple[float, float] | None = None     # Kalman estimate (also when briefly lost)
    velocity: tuple[float, float] | None = None     # px/frame (Kalman)
    tip: tuple[int, int] | None = None              # contour point deepest inside the board
    tip_square: str | None = None                   # square under the tip ("e4")
    fg_mask: np.ndarray | None = field(default=None, repr=False)


def make_kalman(dt: float = 1.0, q: float = config.KALMAN_PROCESS_NOISE,
                r: float = config.KALMAN_MEASUREMENT_NOISE) -> cv2.KalmanFilter:
    """Constant-velocity model. State [x, y, vx, vy], measurement [x, y]."""
    kf = cv2.KalmanFilter(4, 2)
    kf.transitionMatrix = np.array([[1, 0, dt, 0], [0, 1, 0, dt],
                                    [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
    kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
    kf.processNoiseCov = np.eye(4, dtype=np.float32) * q
    kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * r
    kf.errorCovPost = np.eye(4, dtype=np.float32) * 100
    return kf


class HandTracker:
    def __init__(self, corners: dict, frame_shape: tuple[int, ...]):
        self.corners = corners
        self.roi = board_mask(corners, frame_shape, margin=config.HAND_BOARD_MARGIN)
        self.inner = board_mask(corners, frame_shape, margin=0.0)
        self.H_inv = np.linalg.inv(board_to_image_homography(corners))   # image -> board coords
        m = cv2.moments(self.inner, binaryImage=True)
        self.center = np.float32([m["m10"] / m["m00"], m["m01"] / m["m00"]])
        self.k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (config.HAND_OPEN_K,) * 2)
        self.k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (config.HAND_CLOSE_K,) * 2)
        self.reset()

    # --- state -------------------------------------------------------------------
    def reset(self) -> None:
        self.bg = cv2.createBackgroundSubtractorMOG2(history=config.MOG2_HISTORY,
                                                     varThreshold=config.MOG2_VAR_THRESHOLD,
                                                     detectShadows=True)
        self.kf: cv2.KalmanFilter | None = None
        self.state = "idle"
        self.seen = 0          # consecutive frames with hand
        self.missed = 0        # consecutive frames without hand
        self.still = 0         # consecutive still frames while settling
        self.prev_gray = None
        self.warm = config.MOG2_WARMUP_FRAMES

    def relearn(self, frame: np.ndarray) -> None:
        """Re-initialises the background with this frame (after a move, or on demand)."""
        self.bg.apply(frame, learningRate=1.0)

    # --- per frame -----------------------------------------------------------------
    def foreground(self, frame: np.ndarray, learning_rate: float) -> np.ndarray:
        fg = self.bg.apply(frame, learningRate=learning_rate)
        fg = np.where(fg == 255, 255, 0).astype(np.uint8)      # 127 = shadow -> background
        fg &= self.roi
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, self.k_open)
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, self.k_close)
        return fg

    def _square_at(self, pt) -> str | None:
        f, r = cv2.perspectiveTransform(np.float32([[pt]]), self.H_inv)[0, 0]
        if 0 <= f < 8 and 0 <= r < 8:
            return f"{FILES[int(f)]}{int(r) + 1}"
        return None

    def update(self, frame: np.ndarray) -> TrackResult:
        # While nobody touches the board the background keeps adapting slowly (light changes);
        # while the hand is there it is frozen so that a hand held still is not absorbed.
        if self.warm > 0:
            lr = -1                                   # automatic (fast) while warming up
            self.warm -= 1
        else:
            lr = config.MOG2_LEARNING_RATE if self.state == "idle" else 0.0
        fg = self.foreground(frame, lr)

        res = TrackResult(state=self.state, fg_mask=fg)
        contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        big = max(contours, key=cv2.contourArea) if contours else None
        if big is not None and cv2.contourArea(big) >= config.HAND_MIN_AREA:
            # A hand must reach the board itself, not only the margin around it
            reach = np.zeros_like(fg)
            cv2.drawContours(reach, [big], -1, 255, -1)
            if cv2.countNonZero(reach & self.inner) >= config.HAND_MIN_AREA_INSIDE:
                res.detected = True
                res.contour = big
                res.bbox = cv2.boundingRect(big)
                m = cv2.moments(big)
                res.centroid = (m["m10"] / m["m00"], m["m01"] / m["m00"])
                pts = big.reshape(-1, 2).astype(np.float32)
                tip = pts[np.argmin(np.linalg.norm(pts - self.center, axis=1))]
                res.tip = (int(tip[0]), int(tip[1]))
                res.tip_square = self._square_at(tip)

        self._kalman(res)
        self._state_machine(frame, fg, res)
        res.state = self.state
        return res

    def _kalman(self, res: TrackResult) -> None:
        if res.detected:
            z = np.float32(res.centroid).reshape(2, 1)
            if self.kf is None:
                self.kf = make_kalman()
                self.kf.statePost = np.float32([[z[0, 0]], [z[1, 0]], [0], [0]])
            else:
                self.kf.predict()
                self.kf.correct(z)
        elif self.kf is not None:
            if self.missed < config.HAND_MAX_MISSED:
                self.kf.predict()                      # brief loss: coast on the prediction
            else:
                self.kf = None
        if self.kf is not None:
            s = self.kf.statePost.ravel()
            res.position = (float(s[0]), float(s[1]))
            res.velocity = (float(s[2]), float(s[3]))

    def _state_machine(self, frame: np.ndarray, fg: np.ndarray, res: TrackResult) -> None:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        motion = 0
        if self.prev_gray is not None:
            diff = cv2.absdiff(gray, self.prev_gray)
            motion = cv2.countNonZero((diff > config.SETTLE_DIFF_THRESHOLD).astype(np.uint8) & self.roi)
        self.prev_gray = gray

        if res.detected:
            self.seen += 1
            self.missed = 0
        else:
            self.seen = 0
            self.missed += 1

        if self.state == "idle":
            if self.seen >= config.HAND_ON_FRAMES:
                self.state, res.event = "hand", "hand_in"
        elif self.state == "hand":
            if self.missed >= config.HAND_OFF_FRAMES:
                self.state, res.event, self.still = "settling", "hand_out", 0
        elif self.state == "settling":
            if self.seen >= config.HAND_ON_FRAMES:
                self.state, res.event = "hand", "hand_in"
            elif motion <= config.SETTLE_MAX_MOTION_PX:
                self.still += 1
                if self.still >= config.SETTLE_FRAMES:
                    self.state, res.event = "idle", "settled"
                    self.relearn(frame)                # moved pieces become background
                    self.kf = None
            else:
                self.still = 0


def draw_track(img: np.ndarray, res: TrackResult) -> np.ndarray:
    """Overlay of the tracker result (contour, bbox, Kalman position and velocity, tip)."""
    out = img.copy()
    if res.contour is not None:
        cv2.drawContours(out, [res.contour], -1, (255, 0, 255), 2)
        x, y, w, h = res.bbox
        cv2.rectangle(out, (x, y), (x + w, y + h), (255, 255, 0), 1)
    if res.centroid is not None:
        cv2.circle(out, tuple(map(int, res.centroid)), 4, (0, 0, 255), -1)          # measured
    if res.position is not None:
        p = tuple(map(int, res.position))
        cv2.circle(out, p, 8, (0, 255, 0), 2)                                       # Kalman
        v = res.velocity
        cv2.arrowedLine(out, p, (int(p[0] + 5 * v[0]), int(p[1] + 5 * v[1])), (0, 255, 0), 2)
    if res.tip is not None:
        cv2.circle(out, res.tip, 6, (0, 255, 255), -1)
        if res.tip_square:
            cv2.putText(out, res.tip_square, (res.tip[0] + 8, res.tip[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    return out