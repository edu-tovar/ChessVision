"""Key decoder: turns per-frame piece recognitions into a 4-piece key.

A single frame can be wrong (motion blur, a bad silhouette), so the decision about
which piece is being shown is taken over several frames:

    IDLE --piece appears--> OBSERVING --enough votes / timeout / removed--> WAIT_REMOVAL
      ^                                                                        |
      +----------------------------- piece removed ----------------------------+

- Presence is the area of the colour mask of the piece in the ROI, with hysteresis
  (cheap: SIFT only runs while a piece is being observed).
- While OBSERVING, frames in which the piece moves do not vote. In each still frame
  the winning type adds its number of inliers (a mode weighted by confidence).
- The piece is accepted if the winner has enough recognised frames, a clear share of
  the votes and the piece was recognised in most still frames; otherwise it is rejected
  and the user must remove it and show it again.
- A piece must leave the ROI before the next one counts. When 4 pieces have been
  entered they are compared with the secret key. The partial key is cleared after
  some time without pieces, or with reset().
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from chessvision import config
from chessvision.security.piece_matcher import PieceMatcher

IDLE, OBSERVING, WAIT_REMOVAL, UNLOCKED = "idle", "observing", "wait_removal", "unlocked"


@dataclass
class Event:
    kind: str                  # accepted | repeated | rejected | too_short | unlocked | wrong_key | timeout
    piece: str | None = None
    message: str = ""
    stats: str = ""            # votes behind a decision (for tuning the thresholds)


@dataclass
class Observation:
    """Votes collected while one piece is shown."""
    start: float
    score: dict[str, float] = field(default_factory=lambda: {t: 0.0 for t in config.PIECE_TYPES})
    still: int = 0             # still frames (classified)
    recognised: int = 0        # still frames where some type reached the inlier threshold

    def winner(self) -> tuple[str | None, float]:
        total = sum(self.score.values())
        if total == 0:
            return None, 0.0
        best = max(self.score, key=self.score.get)
        return best, self.score[best] / total


class KeyDecoder:
    def __init__(self, matcher: PieceMatcher, key=config.SECURITY_KEY):
        key = tuple(key)
        if len(key) != 4 or len(set(key)) != 4 or not set(key) <= set(config.PIECE_TYPES):
            raise ValueError(f"The key must be 4 distinct types of {config.PIECE_TYPES}, got {key}")
        self.matcher, self.key = matcher, key
        self.reset()

    def reset(self) -> None:
        self.state = IDLE
        self.entries: list[str] = []
        self.obs: Observation | None = None
        self._on = self._off = 0
        self._prev_centroid = None
        self._last_activity = None
        # Per-frame info for the display
        self.area, self.moving, self.frame_votes, self.frame_label = 0, False, {}, None

    # --- Per-frame update -------------------------------------------------
    def update(self, roi: np.ndarray, t: float | None = None) -> Event | None:
        """Process one ROI frame (BGR). Returns an Event when something happens."""
        t = time.monotonic() if t is None else t
        if self._last_activity is None:
            self._last_activity = t
        self.frame_votes, self.frame_label = {}, None
        if self.state == UNLOCKED:
            return None

        core = self.matcher.piece_core(roi)
        self.area = int(np.count_nonzero(core))
        self._on = self._on + 1 if self.area >= config.DECODER_AREA_ON else 0
        self._off = self._off + 1 if self.area < config.DECODER_AREA_OFF else 0
        present = self.area >= config.DECODER_AREA_OFF and self._off == 0
        removed = self._off >= config.DECODER_OFF_FRAMES

        if present:
            self._last_activity = t

        if self.state == IDLE:
            if self._on >= config.DECODER_ON_FRAMES:
                self.state, self.obs, self._prev_centroid = OBSERVING, Observation(start=t), None
            elif self.entries and t - self._last_activity >= config.DECODER_IDLE_RESET:
                self.entries = []
                return Event("timeout", message="Too long without a piece: key cleared")
            if self.state == IDLE:
                return None

        if self.state == WAIT_REMOVAL:
            if removed:
                self.state = IDLE
            return None

        # --- OBSERVING ---
        if removed:
            return self._decide(early=True)
        self.moving = self._is_moving(core)
        if not self.moving and present:
            votes = self.matcher.scores(*self.matcher.features(roi))
            self.frame_votes = votes
            self.frame_label = self.matcher.decide(votes)
            self.obs.still += 1
            if self.frame_label is not None:
                self.obs.score[self.frame_label] += votes[self.frame_label]
                self.obs.recognised += 1
        if (self.obs.recognised >= config.DECODER_VOTE_FRAMES
                or t - self.obs.start >= config.DECODER_MAX_TIME):
            return self._decide(early=False)
        return None

    def _is_moving(self, core: np.ndarray) -> bool:
        ys, xs = np.nonzero(core)
        if len(xs) == 0:
            return True
        c = np.array([xs.mean(), ys.mean()])
        prev, self._prev_centroid = self._prev_centroid, c
        return prev is None or float(np.linalg.norm(c - prev)) > config.DECODER_MAX_SHIFT

    # --- Decision -----------------------------------------------------------
    def _decide(self, early: bool) -> Event:
        """Close the current observation. early = the piece was removed before the end."""
        obs, self.obs = self.obs, None
        self.state = IDLE if early else WAIT_REMOVAL
        piece, share = obs.winner()
        votes = ", ".join(f"{t} {int(s)}" for t, s in sorted(obs.score.items(), key=lambda x: -x[1]) if s)
        stats = (f"still frames {obs.still}, recognised {obs.recognised} "
                 f"({obs.recognised / max(obs.still, 1):.0%}), share {share:.0%}, inliers: {votes or '-'}")
        if obs.recognised < config.DECODER_MIN_FRAMES:
            if early:
                return Event("too_short", None, "Piece removed too soon: show it again", stats)
            return Event("rejected", None, "Piece not recognised: remove it and try again", stats)
        if share < config.DECODER_MIN_SHARE:
            return Event("rejected", None, f"Not sure: {piece} got only {share:.0%} of the votes. "
                                           "Remove it and try again", stats)
        if obs.recognised / max(obs.still, 1) < config.DECODER_MIN_RECOGNISED:
            return Event("rejected", None, f"{piece} seen in only {obs.recognised}/{obs.still} still frames. "
                                           "Show it again, upright", stats)
        if piece in self.entries:
            return Event("repeated", piece, f"{piece} already entered: show another piece", stats)

        self.entries.append(piece)
        if len(self.entries) < len(self.key):
            return Event("accepted", piece, f"{piece} - {len(self.entries)}/{len(self.key)}", stats)
        if tuple(self.entries) == self.key:
            self.state = UNLOCKED
            return Event("unlocked", piece, "Key correct: unlocked", stats)
        self.entries = []
        return Event("wrong_key", piece, "Wrong key: start again", stats)

    @property
    def progress(self) -> float:
        """0..1 progress of the current observation (for the display)."""
        if self.obs is None:
            return 0.0
        return min(self.obs.recognised / config.DECODER_VOTE_FRAMES, 1.0)
