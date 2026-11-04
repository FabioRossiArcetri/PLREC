"""Petal geometry of the DM piston modes, the "twin" symmetry of the pupil, and
re-referencing of petal pistons.

The twin of a pupil phase phi(x) is -phi(-x): rotated by 180 degrees and
negated. For a point-symmetric pupil amplitude A(x) = A(-x) its focal-plane
field is the complex conjugate of the original one, conj(E(u)), so the two
have the same focal-plane intensity, and a sensor that responds to intensity
only cannot tell them apart. For petal pistons the 180-degree rotation swaps
each petal with the opposite one, so with petal r as the zero reference the
twin of the pistons p (p_r = 0) is

    p'_k = p_opp(r) - p_opp(k)        (re-referenced to petal r again)

which leaves the piston of the petal opposite the reference unchanged:
p'_opp(r) = p_opp(r) - p_r = p_opp(r).
"""
from __future__ import annotations

import numpy as np


def petal_geometry(mask, modes, n_pistons=5, tol=1e-6):
    """Petals of the first n_pistons piston modes and of the reference petal
    (the pupil pixels that none of them moves).

    mask  : (ny, nx) pupil mask; modes : (n_modes, n_pixels) mode matrix, one
        value per mask pixel in row-major order (as in an influence function).

    Returns a dict: ``petal_map`` (ny, nx) int, -1 outside the pupil, k for the
    petal of mode k, n_pistons for the reference petal; ``angle_deg`` (n_pistons
    + 1,) centroid angle of each petal (counter-clockwise from +x, image rows
    pointing down); ``n_px`` pixels per petal; ``opposite`` (n_pistons + 1,)
    index of the petal at the nearest angle to each petal's + 180 degrees; and
    ``ref`` = n_pistons.
    """
    mask = np.asarray(mask) > 0
    rows, cols = np.nonzero(mask)
    label = np.full(len(rows), n_pistons)
    for k in range(n_pistons):
        on = np.abs(modes[k]) > tol * np.abs(modes[k]).max()
        label[on] = k
    petal_map = np.full(mask.shape, -1)
    petal_map[rows, cols] = label
    cy, cx = (mask.shape[0] - 1) / 2, (mask.shape[1] - 1) / 2
    angle = np.array([np.degrees(np.arctan2(-(rows[label == k].mean() - cy), cols[label == k].mean() - cx))
                      for k in range(n_pistons + 1)])
    diff = (angle[None, :] - angle[:, None] - 180 + 180) % 360 - 180      # angle_j - (angle_i + 180)
    return dict(petal_map=petal_map, angle_deg=angle, ref=n_pistons,
                n_px=np.bincount(label, minlength=n_pistons + 1),
                opposite=np.abs(diff).argmin(axis=1))


def twin_matrix(opposite, ref):
    """(M, M) matrix T with twin(p) = T @ p for pistons p of the M = len(opposite)
    - 1 non-reference petals (petal ``ref`` is the zero reference):
    p'_k = p_opp(ref) - p_opp(k). T is an involution (T @ T = I)."""
    n = len(opposite)
    keep = [k for k in range(n) if k != ref]
    col = {k: i for i, k in enumerate(keep)}
    T = np.zeros((len(keep), len(keep)))
    for i, k in enumerate(keep):
        a, b = opposite[ref], opposite[k]          # p'_k = p_a - p_b, with p_ref = 0
        if a != ref:
            T[i, col[a]] += 1
        if b != ref:
            T[i, col[b]] -= 1
    return T


def rereference(C, ref, new_ref):
    """Pistons C (n, M) of the petals other than ``ref`` (which is 0) expressed
    relative to petal ``new_ref`` instead: (n, M) pistons of the petals other
    than new_ref, in petal order. The physical configuration -- and so the
    lantern signals -- is the same, up to a global piston."""
    n, M = C.shape
    full = np.insert(np.asarray(C, float), ref, 0.0, axis=1)     # (n, M + 1), petal order
    rel = full - full[:, [new_ref]]
    return np.delete(rel, new_ref, axis=1)
