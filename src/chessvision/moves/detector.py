"""Move detection: image difference + CNN + piece-count rules (no legal-move check).
 
Called on every "settled" event of the hand tracker:
 
    1. Average the CNN probabilities over a few still frames (less noise).
    2. Image difference against the last accepted board -> squares that changed.
    3. constraints.solve() with the UNCHANGED squares fixed to their previous class:
       the CNN only decides the squares that really changed, and the result respects
       the piece counts (1 king, 1 queen, ...).
    4. The move is the difference between the old and the new position:
       1 square emptied + 1 filled = normal move / capture, 2 + 2 = castling,
       2 emptied + 1 filled = en passant. Anything else -> "unclear" (not committed).
       The piece on the destination square is ALWAYS the one the CNN reads there: it is
       never copied from the square that was emptied.
 
If a reading is not committed, the reference frame is kept, so the next "settled" compares
against the last good board and can still recover the move.
"""
from __future__ import annotations
 
from dataclasses import dataclass, field
 
import cv2
import numpy as np
 
from chessvision import config
from chessvision.board.corners import INITIAL_FEN, fen_to_classes, square_polygon
from chessvision.moves import constraints
from chessvision.moves.constraints import SQUARES
from chessvision.pieces.classifier import PieceClassifier
 
 
@dataclass
class MoveResult:
    ok: bool                                    # move understood and committed
    move: str | None = None                     # "e2e4", castling as the king move "e1g1"
    piece: str | None = None                    # class read by the CNN on the destination square
    reason: str = ""                            # why it is not ok / "no change"
    changes: list[tuple[str, str, str]] = field(default_factory=list)   # (square, old, new)
    changed_squares: list[str] = field(default_factory=list)            # from the image difference
    confidence: float = 1.0                     # lowest CNN probability among the changed squares
    overridden: dict[str, tuple[str, str]] = field(default_factory=dict)  # {square: (CNN argmax, after rules)}
    position: dict[str, str] = field(default_factory=dict)              # board after the reading
 
 
def colour(cls: str) -> str | None:
    return None if cls == "empty" else cls.split("_")[0]
 
 
def interpret(changes: list[tuple[str, str, str]]) -> tuple[str | None, str]:
    """Move from the list of changed squares. Returns (move or None, reason)."""
    if not changes:
        return None, "no change"
    emptied = [(sq, old) for sq, old, new in changes if new == "empty"]
    filled = [(sq, new) for sq, old, new in changes if new != "empty"]
 
    if len(emptied) == 1 and len(filled) == 1:              # normal move or capture
        (f, _), (t, _) = emptied[0], filled[0]              # piece on t = CNN reading, not the old piece
        return f + t, ""
 
    if len(emptied) == 2 and len(filled) == 2:              # castling
        kings = [(sq, p) for sq, p in emptied if p.endswith("_king")]
        rooks = [(sq, p) for sq, p in emptied if p.endswith("_rook")]
        k_to = [sq for sq, p in filled if p.endswith("_king")]
        if len(kings) == 1 and len(rooks) == 1 and len(k_to) == 1 and \
                colour(kings[0][1]) == colour(rooks[0][1]):
            return kings[0][0] + k_to[0], ""
 
    if len(emptied) == 2 and len(filled) == 1:              # en passant
        t, new = filled[0]
        own = [sq for sq, p in emptied if p.endswith("_pawn") and colour(p) == colour(new)]
        other = [sq for sq, p in emptied if p.endswith("_pawn") and colour(p) != colour(new)]
        if len(own) == 1 and len(other) == 1:
            return own[0] + t, ""
 
    return None, f"{len(emptied)} squares emptied, {len(filled)} filled"
 
 
class MoveDetector:
    def __init__(self, classifier: PieceClassifier, corners: dict,
                 diff_threshold: int = config.MOVE_DIFF_THRESHOLD,
                 min_changed_fraction: float = config.MOVE_MIN_CHANGED_FRACTION):
        self.clf = classifier
        self.corners = corners
        self.diff_threshold = diff_threshold
        self.min_changed_fraction = min_changed_fraction
        self.position: dict[str, str] = {}
        self.ref_gray: np.ndarray | None = None
        self._masks: dict[str, np.ndarray] | None = None
 
    # --- State ----------------------------------------------------------------
    def reset(self, frame: np.ndarray, position: dict[str, str] | str = INITIAL_FEN) -> None:
        """Start from a known position (FEN or {square: class}) with `frame` as reference."""
        self.position = fen_to_classes(position) if isinstance(position, str) else dict(position)
        self.ref_gray = self._gray(frame)
 
    def read_position(self, frames: list[np.ndarray]) -> dict[str, str]:
        """Position from scratch (no previous position): CNN + rules on all 64 squares."""
        return constraints.solve(self.probabilities(frames), self.clf.classes)
 
    # --- CNN ------------------------------------------------------------------
    def probabilities(self, frames: list[np.ndarray]) -> dict[str, np.ndarray]:
        """CNN probabilities averaged over several still frames: {square: [13]}."""
        acc = None
        for f in frames:
            p = self.clf.board_probs(f, self.corners)
            acc = p if acc is None else {sq: acc[sq] + p[sq] for sq in acc}
        return {sq: v / len(frames) for sq, v in acc.items()}
 
    # --- Image difference -------------------------------------------------------
    @staticmethod
    def _gray(frame: np.ndarray) -> np.ndarray:
        return cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (5, 5), 0)
 
    def changed_squares(self, frame: np.ndarray) -> list[str]:
        """Squares whose pixels changed against the reference frame."""
        gray = self._gray(frame)
        if self._masks is None:
            self._masks = {}
            for sq in SQUARES:
                m = np.zeros(gray.shape, np.uint8)
                cv2.fillConvexPoly(m, square_polygon(self.corners, sq), 255)
                self._masks[sq] = m > 0
        moving = cv2.absdiff(gray, self.ref_gray) > self.diff_threshold
        return [sq for sq, m in self._masks.items() if moving[m].mean() > self.min_changed_fraction]
 
    # --- Main entry point ---------------------------------------------------------
    def update(self, frames: list[np.ndarray]) -> MoveResult:
        """Read the board after a "settled" event. `frames`: a few undistorted still frames."""
        if self.ref_gray is None:
            raise RuntimeError("Call reset() first.")
        changed = self.changed_squares(frames[-1])
        if not changed:
            return MoveResult(True, reason="no change", position=dict(self.position))
 
        probs = self.probabilities(frames)
        fixed = {sq: c for sq, c in self.position.items() if sq not in changed}
        new = constraints.solve(probs, self.clf.classes, fixed=fixed)
 
        changes = [(sq, self.position[sq], new[sq]) for sq in SQUARES if new[sq] != self.position[sq]]
        cidx = {c: i for i, c in enumerate(self.clf.classes)}
        conf = min((float(probs[sq][cidx[new[sq]]]) for sq in changed), default=1.0)
        move, reason = interpret(changes)
        piece = new[move[2:]] if move else None
        cnn = {sq: self.clf.classes[int(probs[sq].argmax())] for sq in changed}
        overridden = {sq: (cnn[sq], new[sq]) for sq in changed if cnn[sq] != new[sq]}
 
        ok = move is not None or reason == "no change"
        if ok:                                       # commit: new position and new reference
            self.position = new
            self.ref_gray = self._gray(frames[-1])
        return MoveResult(ok, move, piece, reason, changes, changed, conf, overridden, new)