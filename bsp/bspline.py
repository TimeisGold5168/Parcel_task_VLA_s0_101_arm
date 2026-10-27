"""B-spline action targets for B-spline Policy (BSP) on LeRobot data.

Fitting, chunking and decoding follow the official implementation
(github.com/B-spline-policy/bspline-policy, MIT licence,
bspline_policy/common/bspline_action.py), without its replay-buffer
dependency.

Target per frame: a (chunk_size + 2 * degree, 1 + action_dim) matrix.
Column 0 holds knots in frames relative to the current frame; the other
columns hold control points in the dataset action units (degrees).
"""

import numpy as np
from scipy.interpolate import BSpline, generate_knots, make_lsq_spline


def fit_episode(actions, degree=3, max_error=0.5, s=1e-12):
    """Least-squares B-spline with knots added until max error is met.

    Returns (spline, max_abs_error).
    """
    t = np.arange(len(actions), dtype=np.float64)
    data = np.asarray(actions, dtype=np.float64)
    spl, err = None, None
    for knots in generate_knots(t, data, k=degree, s=s):
        spl = make_lsq_spline(t, data, knots, k=degree)
        err = float(np.abs(spl(t) - data).max())
        if err < max_error:
            break
    return spl, err


def chunk_spline(spl, chunk_size=10, stride=1):
    """Split a fitted spline into fixed-size (t, c) parameter chunks."""
    degree = spl.k
    t_full, c_full = spl.t, spl.c
    n_unique = len(t_full) - 2 * degree
    rows = chunk_size + 2 * degree
    chunks = []
    for start in range(0, n_unique - 1, stride):
        lo = start
        hi = min(len(t_full), start + chunk_size + 2 * degree)
        ct, cc = t_full[lo:hi], c_full[lo:hi]
        if len(ct) < rows:
            ct = np.concatenate([ct, np.full(rows - len(ct), ct[-1])])
        if len(cc) < rows:
            pad = np.repeat(cc[-1:], rows - len(cc), axis=0)
            cc = np.concatenate([cc, pad], axis=0)
        chunks.append((ct, cc))
    return chunks


def episode_targets(actions, degree=3, chunk_size=10, max_error=0.5):
    """One parameter matrix per frame of the episode.

    Frame i uses the first chunk whose first interior knot is at or after
    i; frames past the last chunk reuse it. Knots are made relative to i.
    """
    spl, err = fit_episode(actions, degree, max_error)
    chunks = chunk_spline(spl, chunk_size)
    n, dim = len(actions), actions.shape[1]
    out = np.zeros((n, chunk_size + 2 * degree, 1 + dim), np.float32)
    i = 0
    mat = None
    for ct, cc in chunks:
        mat = np.concatenate([ct[:, None], cc], axis=1)
        while i < n and i <= ct[degree]:
            out[i] = mat
            out[i, :, 0] -= i
            i += 1
    while i < n:
        out[i] = mat
        out[i, :, 0] -= i
        i += 1
    n_knots = len(spl.t) - 2 * degree
    return out, err, n_knots


def make_spline(params, degree=3):
    """BSpline from one parameter matrix (knot validity enforced)."""
    p = np.asarray(params, dtype=np.float64)
    knots = np.maximum.accumulate(p[:, 0])
    ctrl = p[: -(degree + 1), 1:]
    return BSpline(knots, ctrl, degree, extrapolate=False)


def segment_range(params, degree=3):
    """(t_min, t_max) of the valid part of a segment, in frames."""
    knots = np.maximum.accumulate(np.asarray(params)[:, 0])
    return float(knots[degree]), float(knots[-(degree + 1)])


def decode(params, num_actions, degree=3):
    """Sample num_actions evenly over the valid part of a segment."""
    t0, t1 = segment_range(params, degree)
    if t1 <= t0:
        raise ValueError(f"invalid B-spline range [{t0}, {t1}]")
    # the curve evaluates to 0 exactly at the last knot (extrapolate=False)
    t = np.linspace(t0, t1 - 1e-6, int(num_actions))
    return make_spline(params, degree)(t).astype(np.float32)
