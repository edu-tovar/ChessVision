"""Rule-based correction of the CNN output (no promotions).

The CNN classifies each square on its own, so it can return two white kings or a queen
where there should be a king. This module finds the MOST PROBABLE position that respects
the piece counts of a game without promotions:

    per colour: exactly 1 king, at most 1 queen, 2 rooks, 2 knights, 8 pawns,
                1 bishop on light squares and 1 on dark squares,
    no pawns on ranks 1 and 8.

It is solved as an assignment problem (Hungarian algorithm, scipy.optimize.linear_sum_assignment):

    rows    = piece "slots" (16 per colour) + 64 "empty" slots
    columns = 64 squares + 32 "off the board" columns (captured pieces / unused empty slots)
    cost    = -log P_cnn(class | square)   for a slot placed on a square
              0                            for a slot left off the board
              BIG                          for forbidden placements (pawn on rank 1/8,
                                           bishop on the wrong colour, king off the board)

Every square receives exactly one slot, so the total cost is -log of the joint probability
of the position (squares independent) -> the optimum is the most probable legal-count position.

Squares can also be fixed to a class (`fixed`): the move detector uses it to keep the squares
that did not change in the image.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

from chessvision.board.geometry import FILES

BIG = 1e6           # forbidden (finite so the solver always has a feasible solution)
EPS = 1e-9          # floor for probabilities before the log

# (class, number of slots, square restriction)
#   restriction: None | "no_back_ranks" | "light" | "dark"
_SLOTS_PER_COLOUR = [
    ("king", 1, None),
    ("queen", 1, None),
    ("rook", 2, None),
    ("knight", 2, None),
    ("bishop", 1, "light"),
    ("bishop", 1, "dark"),
    ("pawn", 8, "no_back_ranks"),
]

SQUARES = [f"{f}{r}" for r in range(1, 9) for f in FILES]      # a1, b1, ..., h8


def is_light(square: str) -> bool:
    """a1 is dark."""
    return (FILES.index(square[0]) + int(square[1])) % 2 == 0


def _slots() -> list[tuple[str, str | None]]:
    """[(class name, restriction)] for the 32 piece slots."""
    out = []
    for colour in ("white", "black"):
        for piece, n, rule in _SLOTS_PER_COLOUR:
            out += [(f"{colour}_{piece}", rule)] * n
    return out


def _allowed(rule: str | None, square: str) -> bool:
    if rule == "no_back_ranks":
        return square[1] not in "18"
    if rule == "light":
        return is_light(square)
    if rule == "dark":
        return not is_light(square)
    return True


def solve(probs: dict[str, np.ndarray], classes: list[str],
          fixed: dict[str, str] | None = None) -> dict[str, str]:
    """Most probable position that satisfies the rules.

    probs   {square: [13] probabilities in the order of `classes`} for the 64 squares.
    classes class names (models/pieces.json), including "empty".
    fixed   {square: class} squares whose class is imposed.
    Returns {square: class} for the 64 squares.
    """
    fixed = fixed or {}
    cidx = {c: i for i, c in enumerate(classes)}
    logp = {sq: -np.log(np.maximum(np.asarray(probs[sq], np.float64), EPS)) for sq in SQUARES}

    piece_slots = _slots()
    n_piece, n_sq = len(piece_slots), len(SQUARES)
    rows = [(c, rule) for c, rule in piece_slots] + [("empty", None)] * n_sq
    cost = np.zeros((len(rows), n_sq + n_piece))          # 96 x 96

    for i, (cls, rule) in enumerate(rows):
        for j, sq in enumerate(SQUARES):
            if not _allowed(rule, sq) or (sq in fixed and fixed[sq] != cls):
                cost[i, j] = BIG
            else:
                cost[i, j] = logp[sq][cidx[cls]]
        if cls.endswith("_king"):
            cost[i, n_sq:] = BIG                           # kings must be on the board

    r, c = linear_sum_assignment(cost)
    out = {}
    for i, j in zip(r, c):
        if j < n_sq:
            out[SQUARES[j]] = rows[i][0]
    return out


def violations(position: dict[str, str]) -> list[str]:
    """Rules broken by a position (e.g. the raw CNN argmax). Empty list = OK."""
    msgs = []
    for colour in ("white", "black"):
        for piece, limit in (("king", 1), ("queen", 1), ("rook", 2), ("knight", 2), ("pawn", 8)):
            n = sum(v == f"{colour}_{piece}" for v in position.values())
            if n > limit or (piece == "king" and n != 1):
                msgs.append(f"{n} x {colour}_{piece}")
        for light in (True, False):
            n = sum(v == f"{colour}_bishop" and is_light(sq) == light for sq, v in position.items())
            if n > 1:
                msgs.append(f"{n} x {colour}_bishop on {'light' if light else 'dark'} squares")
    for sq, v in position.items():
        if v.endswith("_pawn") and sq[1] in "18":
            msgs.append(f"pawn on {sq}")
    return msgs