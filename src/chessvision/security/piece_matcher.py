"""Piece type recognition with SIFT descriptors.

Pipeline for a query image (the ROI where the user shows a piece):
    colour mask -> SIFT keypoints -> FLANN matching against every reference image
    -> Lowe's ratio test -> RANSAC geometric verification -> vote per piece type.

The colour mask keeps only the light (yellowish) piece plus a thin border for its
contour. Without it the hand and its shadow, present in almost every reference photo,
match the hand of an empty ROI and produce false positives.

The geometric model is a similarity (rotation + scale + translation,
cv2.estimateAffinePartial2D) instead of a homography: a piece is a 3D object,
not a plane, and a 4-DOF model needs fewer matches to be estimated reliably.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from chessvision import config


@dataclass
class Reference:
    label: str
    ref_id: int
    points: np.ndarray                 # (N, 2) keypoint coordinates
    matcher: cv2.FlannBasedMatcher     # FLANN index built on this image's descriptors


class PieceMatcher:
    def __init__(self, ratio: float = config.SIFT_RATIO,
                 min_matches: int = config.SIFT_MIN_MATCHES,
                 min_inliers: int = config.SIFT_MIN_INLIERS,
                 use_clahe: bool = True, use_mask: bool = True):
        self.sift = cv2.SIFT_create()
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)) if use_clahe else None
        self.ratio, self.min_matches, self.min_inliers = ratio, min_matches, min_inliers
        self.use_mask = use_mask
        self.refs: list[Reference] = []

    # --- Features ---------------------------------------------------------
    def piece_mask(self, img: np.ndarray) -> np.ndarray | None:
        """Binary mask of the piece (pixels inside the HSV range, dilated). None = whole image."""
        if not self.use_mask or img.ndim != 3:
            return None
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, config.PIECE_HSV_LOW, config.PIECE_HSV_HIGH)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))   # remove specks
        d = config.PIECE_MASK_DILATE
        return cv2.dilate(mask, np.ones((d, d), np.uint8))

    def features(self, img: np.ndarray):
        """SIFT keypoints and descriptors inside the piece mask.
        CLAHE boosts the little texture pieces have."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
        if self.clahe is not None:
            gray = self.clahe.apply(gray)
        return self.sift.detectAndCompute(gray, self.piece_mask(img))

    # --- References -------------------------------------------------------
    def add_reference(self, label: str, img: np.ndarray, ref_id: int | None = None) -> int:
        """Add a reference image of a piece type. Returns its number of keypoints."""
        kp, des = self.features(img)
        if des is None or len(kp) < 2:
            return 0
        matcher = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
        matcher.add([des])
        matcher.train()
        self.refs.append(Reference(label, len(self.refs) if ref_id is None else ref_id,
                                   np.float32([k.pt for k in kp]), matcher))
        return len(kp)

    # --- Matching ---------------------------------------------------------
    def inliers_vs(self, ref: Reference, q_pts: np.ndarray, des: np.ndarray) -> int:
        """RANSAC inliers between the query and one reference (0 if not enough matches
        or if the estimated model is not a plausible view of the same piece)."""
        good = [m for m, n in (p for p in ref.matcher.knnMatch(des, k=2) if len(p) == 2)
                if m.distance < self.ratio * n.distance]
        # One-to-one: keep only the best query match for each reference keypoint.
        # Repetitive textures (a card, a lid, a glove) send many query points to the same
        # reference point, and RANSAC then "explains" them all with a scale-0 model.
        best = {}
        for m in good:
            if m.trainIdx not in best or m.distance < best[m.trainIdx].distance:
                best[m.trainIdx] = m
        good = list(best.values())
        if len(good) < self.min_matches:
            return 0
        src = q_pts[[m.queryIdx for m in good]]
        dst = ref.points[[m.trainIdx for m in good]]
        model, mask = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC,
                                                  ransacReprojThreshold=5.0)
        if model is None:
            return 0
        scale = float(np.hypot(model[0, 0], model[1, 0]))
        smin, smax = config.SIFT_SCALE_RANGE
        return int(mask.sum()) if smin <= scale <= smax else 0

    def scores(self, kp, des, exclude_id: int | None = None) -> dict[str, int]:
        """Vote per piece type = best number of inliers among its reference images."""
        votes = {label: 0 for label in config.PIECE_TYPES}
        if des is None or len(kp) < 2:
            return votes
        q_pts = np.float32([k.pt for k in kp])
        for ref in self.refs:
            if ref.ref_id == exclude_id:
                continue
            votes[ref.label] = max(votes[ref.label], self.inliers_vs(ref, q_pts, des))
        return votes

    def decide(self, votes: dict[str, int], min_inliers: int | None = None) -> str | None:
        """Piece type with most inliers, or None if it does not reach the threshold."""
        threshold = self.min_inliers if min_inliers is None else min_inliers
        best = max(votes, key=votes.get)
        return best if votes[best] >= threshold else None

    def classify(self, img: np.ndarray) -> tuple[str | None, dict[str, int], int]:
        """Returns (piece type or None, votes, number of keypoints)."""
        kp, des = self.features(img)
        votes = self.scores(kp, des)
        return self.decide(votes), votes, len(kp)
