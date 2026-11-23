"""Piece type recognition with SIFT descriptors.

Pipeline for a query image (the ROI where the user shows a piece):
    colour mask -> SIFT keypoints -> nearest neighbours in every reference image
    -> Lowe's ratio test -> RANSAC geometric verification -> vote per piece type.

Matching is exact brute force: the distances to the descriptors of all the references
are one matrix product (a few ms), much faster than one FLANN search per reference.

The colour mask keeps only the light (yellowish) piece: the hand and its shadow,
present in almost every reference photo, would otherwise match the hand of an empty ROI.

What SIFT looks at (`features` mode):
    "silhouette"  the binary mask of the piece (pure shape). Default: pieces have almost
                  no texture, and their shading changes with the light, but the outline
                  of each type does not.
    "texture"     the grey image inside the (dilated) mask. Fails when the light changes.
    "raw"         the grey image of the whole ROI (no mask). Baseline.

The geometric model is a similarity (rotation + scale + translation,
cv2.estimateAffinePartial2D) instead of a homography: a piece is a 3D object,
not a plane, and a 4-DOF model needs fewer matches to be estimated reliably.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from chessvision import config

FEATURE_MODES = ("silhouette", "texture", "raw")


@dataclass
class Reference:
    label: str
    ref_id: int
    points: np.ndarray                 # (N, 2) keypoint coordinates
    descriptors: np.ndarray            # (N, 128) float32 SIFT descriptors


class PieceMatcher:
    def __init__(self, ratio: float = config.SIFT_RATIO,
                 min_matches: int = config.SIFT_MIN_MATCHES,
                 min_inliers: int = config.SIFT_MIN_INLIERS,
                 features: str = config.SIFT_FEATURES):
        if features not in FEATURE_MODES:
            raise ValueError(f"features must be one of {FEATURE_MODES}, got '{features}'")
        self.sift = cv2.SIFT_create()
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self.ratio, self.min_matches, self.min_inliers = ratio, min_matches, min_inliers
        self.mode = features
        self.refs: list[Reference] = []
        self._stack = None             # all reference descriptors stacked (built lazily)

    # --- Features ---------------------------------------------------------
    @staticmethod
    def piece_core(img: np.ndarray) -> np.ndarray:
        """Binary mask of the piece: pixels inside the HSV range, without specks."""
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, config.PIECE_HSV_LOW, config.PIECE_HSV_HIGH)
        return cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

    def piece_mask(self, img: np.ndarray) -> np.ndarray | None:
        """Area where SIFT looks (also used to display it). None = whole image."""
        if self.mode == "raw":
            return None
        d = config.PIECE_MASK_DILATE
        return cv2.dilate(self.piece_core(img), np.ones((d, d), np.uint8))

    def features(self, img: np.ndarray):
        """SIFT keypoints and descriptors of a BGR image, according to the mode."""
        if self.mode == "silhouette":
            core = self.piece_core(img)
            ys, xs = np.nonzero(core)
            if len(xs) == 0:
                return [], None
            # SIFT only on the bounding box of the piece (+ margin): same keypoints, much faster
            m = config.SIFT_CROP_MARGIN
            x0, y0 = max(int(xs.min()) - m, 0), max(int(ys.min()) - m, 0)
            x1, y1 = int(xs.max()) + m + 1, int(ys.max()) + m + 1
            # Slight blur: SIFT needs gradients, a hard 0/255 edge is aliased
            sil = cv2.GaussianBlur(core[y0:y1, x0:x1], (5, 5), 0)
            kp, des = self.sift.detectAndCompute(sil, None)
            for k in kp:                                    # back to image coordinates
                k.pt = (k.pt[0] + x0, k.pt[1] + y0)
            return kp, des
        gray = self.clahe.apply(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))   # CLAHE boosts texture
        return self.sift.detectAndCompute(gray, self.piece_mask(img))

    # --- References -------------------------------------------------------
    def add_reference(self, label: str, img: np.ndarray, ref_id: int | None = None) -> int:
        """Add a reference image of a piece type. Returns its number of keypoints."""
        kp, des = self.features(img)
        if des is None or len(kp) < 2:
            return 0
        self.refs.append(Reference(label, len(self.refs) if ref_id is None else ref_id,
                                   np.float32([k.pt for k in kp]), des.astype(np.float32)))
        self._stack = None
        return len(kp)

    def _stacked(self):
        """All reference descriptors in one matrix, plus a padded (n_refs, max_kp) index
        into it so that the two nearest neighbours in every reference are found at once."""
        if self._stack is None:
            des = np.vstack([r.descriptors for r in self.refs])
            sizes = [len(r.descriptors) for r in self.refs]
            starts = np.cumsum([0] + sizes[:-1])
            cols = np.arange(max(sizes))
            pad_idx = starts[:, None] + cols[None, :]                 # (n_refs, max_kp)
            pad = cols[None, :] >= np.array(sizes)[:, None]           # True = padding
            pad_idx[pad] = 0
            self._stack = (np.ascontiguousarray(des.T), (des ** 2).sum(axis=1), pad_idx, pad)
        return self._stack

    # --- Matching ---------------------------------------------------------
    def _verify(self, ref: Reference, q_pts: np.ndarray, good: np.ndarray,
                nearest: np.ndarray, d1: np.ndarray) -> int:
        """RANSAC inliers between the query and one reference (0 if not enough matches
        or if the estimated model is not a plausible view of the same piece).
        good / nearest / d1: per query keypoint, ratio test passed / index of its nearest
        reference keypoint / distance to it."""
        q_idx = np.flatnonzero(good)
        t_idx, d = nearest[q_idx], d1[q_idx]
        # One-to-one: keep only the best query match for each reference keypoint.
        # Repetitive textures (a card, a lid, a glove) send many query points to the same
        # reference point, and RANSAC then "explains" them all with a scale-0 model.
        order = np.argsort(d)
        _, first = np.unique(t_idx[order], return_index=True)
        keep = order[first]
        if len(keep) < self.min_matches:
            return 0
        src, dst = q_pts[q_idx[keep]], ref.points[t_idx[keep]]
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
        if des is None or len(kp) < 2 or not self.refs:
            return votes
        q_pts = np.float32([k.pt for k in kp])
        q = des.astype(np.float32)
        des_t, sq, pad_idx, pad = self._stacked()

        # Euclidean distances to every reference descriptor in one matrix product
        dist = np.sqrt(np.maximum((q ** 2).sum(axis=1)[:, None] + sq[None, :] - 2.0 * (q @ des_t), 0))
        per_ref = dist[:, pad_idx]                                    # (n_query, n_refs, max_kp)
        per_ref[:, pad] = np.inf
        # Two nearest neighbours in each reference + Lowe's ratio test, for all references at once
        nearest = per_ref.argmin(axis=2)                              # (n_query, n_refs)
        d1 = np.take_along_axis(per_ref, nearest[..., None], axis=2)[..., 0]
        np.put_along_axis(per_ref, nearest[..., None], np.inf, axis=2)
        d2 = per_ref.min(axis=2)
        good = d1 < self.ratio * d2
        counts = good.sum(axis=0)

        # RANSAC per reference, best candidates first. Inliers <= matches, so a reference
        # with no more matches than the current best vote of its type cannot change it.
        for r in np.argsort(-counts):
            if counts[r] < self.min_matches:
                break
            ref = self.refs[r]
            if ref.ref_id == exclude_id or counts[r] <= votes[ref.label]:
                continue
            n = self._verify(ref, q_pts, good[:, r], nearest[:, r], d1[:, r])
            votes[ref.label] = max(votes[ref.label], n)
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
