"""SQUAD quaternion resampling — smooth playback of sparse / low-fps animation.

Ubisoft's Dunia/Disrupt engines store rotation as spline-compressed control
points (Avatar: `HK_SPLINE_COMPRESSED_ANIMATION`; FC4: `eCurveSpline` +
parameterized slerp) and evaluate a SMOOTH curve at the game framerate.  The
importer keys one discrete frame per decoded key and lets Blender interpolate
the four quaternion components independently — which wobbles off the geodesic
and looks choppy on sparse / 15 fps clips.

This module reproduces the engine's smoothing: SQUAD (spherical cubic
interpolation, the quaternion analogue of a Catmull-Rom / cubic spline) through
the decoded key quaternions, sampled densely so Blender just plays the baked
frames.  Pure-Python on (w, x, y, z) tuples so it's testable without Blender.

Reference: Shoemake, "Animating Rotation with Quaternion Curves" (SIGGRAPH '85).
"""

import math
import bisect


def _dot(a, b):
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2] + a[3]*b[3]


def _norm(q):
    n = math.sqrt(_dot(q, q)) or 1.0
    return (q[0]/n, q[1]/n, q[2]/n, q[3]/n)


def _neg(q):
    return (-q[0], -q[1], -q[2], -q[3])


def _mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return (w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2)


def _conj(q):
    return (q[0], -q[1], -q[2], -q[3])


def _qlog(q):
    """log of a UNIT quaternion -> pure quaternion (0, v)."""
    v = math.sqrt(q[1]*q[1] + q[2]*q[2] + q[3]*q[3])
    if v < 1e-9:
        return (0.0, 0.0, 0.0, 0.0)
    s = math.atan2(v, q[0]) / v
    return (0.0, q[1]*s, q[2]*s, q[3]*s)


def _qexp(p):
    """exp of a pure quaternion (0, v) -> unit quaternion."""
    v = math.sqrt(p[1]*p[1] + p[2]*p[2] + p[3]*p[3])
    if v < 1e-9:
        return (1.0, 0.0, 0.0, 0.0)
    s = math.sin(v) / v
    return (math.cos(v), p[1]*s, p[2]*s, p[3]*s)


def _slerp(a, b, t):
    d = _dot(a, b)
    if d < 0.0:                      # take the short way round
        b = _neg(b)
        d = -d
    if d > 0.9995:                   # almost identical -> normalised lerp
        return _norm((a[0] + t*(b[0]-a[0]), a[1] + t*(b[1]-a[1]),
                      a[2] + t*(b[2]-a[2]), a[3] + t*(b[3]-a[3])))
    th = math.acos(max(-1.0, min(1.0, d)))
    s = math.sin(th)
    wa = math.sin((1.0 - t) * th) / s
    wb = math.sin(t * th) / s
    return (wa*a[0] + wb*b[0], wa*a[1] + wb*b[1],
            wa*a[2] + wb*b[2], wa*a[3] + wb*b[3])


def _intermediate(q0, q1, q2):
    """SQUAD control quaternion at q1 (uses neighbours q0, q2)."""
    q1c = _conj(q1)
    l_next = _qlog(_mul(q1c, q2))
    l_prev = _qlog(_mul(q1c, q0))
    e = (0.0,
         -(l_next[1] + l_prev[1]) * 0.25,
         -(l_next[2] + l_prev[2]) * 0.25,
         -(l_next[3] + l_prev[3]) * 0.25)
    return _mul(q1, _qexp(e))


def _squad(q0, q1, s0, s1, t):
    return _slerp(_slerp(q0, q1, t), _slerp(s0, s1, t), 2.0 * t * (1.0 - t))


def build_squad(keys):
    """Pre-process keys for SQUAD evaluation.

    `keys`: sorted list of (frame: float, quat: (w,x,y,z)).  Returns
    (eval_fn, first_frame, last_frame) where eval_fn(t) -> (w,x,y,z) is the
    smooth SQUAD value at source-frame time `t`.  Quaternions are forced into a
    consistent hemisphere first (sign flips break log/slerp).
    """
    frames = [float(f) for f, _ in keys]
    qs = [_norm(keys[0][1])]
    for i in range(1, len(keys)):
        q = _norm(keys[i][1])
        if _dot(q, qs[-1]) < 0.0:
            q = _neg(q)
        qs.append(q)

    n = len(qs)
    ctrl = [None] * n
    for i in range(n):
        q0 = qs[i-1] if i > 0 else qs[i]
        q2 = qs[i+1] if i < n-1 else qs[i]
        ctrl[i] = _intermediate(q0, qs[i], q2)

    def eval_fn(t):
        if t <= frames[0]:
            return qs[0]
        if t >= frames[-1]:
            return qs[-1]
        j = bisect.bisect_right(frames, t) - 1
        j = max(0, min(n - 2, j))
        f0, f1 = frames[j], frames[j + 1]
        u = (t - f0) / (f1 - f0) if f1 > f0 else 0.0
        return _norm(_squad(qs[j], qs[j + 1], ctrl[j], ctrl[j + 1], u))

    return eval_fn, frames[0], frames[-1]


# ---------------------------------------------------------------------------
# Vectorised (NumPy) batch SQUAD — evaluates every output frame at once. Same
# math as the scalar path above; used when NumPy is importable (always inside
# Blender), with the pure-Python path kept as a fallback for standalone tests.
# ---------------------------------------------------------------------------

def _normalize_np(np, q):
    n = np.sqrt(np.sum(q * q, axis=1, keepdims=True))
    return q / np.where(n == 0.0, 1.0, n)


def _slerp_np(np, a, b, t):
    """a, b: (M,4); t: (M,). Mirrors the scalar _slerp (short-way + near-parallel
    lerp), vectorised; result normalised."""
    d = np.sum(a * b, axis=1)
    b = np.where((d < 0.0)[:, None], -b, b)
    d = np.abs(d)
    th = np.arccos(np.clip(d, -1.0, 1.0))
    s = np.sin(th)
    safe = np.where(s == 0.0, 1.0, s)
    near = d > 0.9995
    wa = np.where(near, 1.0 - t, np.sin((1.0 - t) * th) / safe)
    wb = np.where(near, t,       np.sin(t * th) / safe)
    return _normalize_np(np, wa[:, None] * a + wb[:, None] * b)


def _squad_np(np, q0, q1, s0, s1, t):
    a = _slerp_np(np, q0, q1, t)
    b = _slerp_np(np, s0, s1, t)
    return _slerp_np(np, a, b, 2.0 * t * (1.0 - t))


def _resample_numpy(np, keys, mult):
    # Build (hemisphere-fixed, normalised) key quats + SQUAD control points.
    # N (#keys) is small, so this stays scalar; only the dense eval is vectorised.
    frames = [float(f) for f, _ in keys]
    qs = [_norm(keys[0][1])]
    for i in range(1, len(keys)):
        q = _norm(keys[i][1])
        if _dot(q, qs[-1]) < 0.0:
            q = _neg(q)
        qs.append(q)
    n = len(qs)
    ctrl = [_intermediate(qs[i - 1] if i > 0 else qs[i], qs[i],
                          qs[i + 1] if i < n - 1 else qs[i]) for i in range(n)]

    fr = np.asarray(frames)
    Q = np.asarray(qs)
    C = np.asarray(ctrl)
    start = int(round(frames[0] * mult))
    end = int(round(frames[-1] * mult))
    ofs = np.arange(start, end + 1)
    t = ofs / float(mult)
    j = np.clip(np.searchsorted(fr, t, side='right') - 1, 0, n - 2)
    f0 = fr[j]; f1 = fr[j + 1]
    span = f1 - f0
    u = np.where(span > 0.0, (t - f0) / np.where(span > 0.0, span, 1.0), 0.0)
    out = _squad_np(np, Q[j], Q[j + 1], C[j], C[j + 1], u)
    out[t <= frames[0]] = Q[0]
    out[t >= frames[-1]] = Q[-1]
    return [(int(ofs[i]), (float(out[i, 0]), float(out[i, 1]),
                           float(out[i, 2]), float(out[i, 3])))
            for i in range(len(ofs))]


def resample_rotation(keys, mult):
    """SQUAD-resample rotation keys to `mult`× density.

    `keys`: [(frame:int|float, (w,x,y,z)), ...] (need not be sorted).
    Returns [(out_frame:int, (w,x,y,z)), ...] at integer frames `frame*mult`
    (so source frame f -> out frame f*mult; set scene fps to src_fps*mult).
    mult<=1 or <2 keys -> returns the keys normalised+hemisphere-fixed, scaled.
    """
    keys = sorted(keys, key=lambda k: k[0])
    if mult <= 1 or len(keys) < 2:
        out = []
        prev = None
        for f, q in keys:
            q = _norm(q)
            if prev is not None and _dot(q, prev) < 0.0:
                q = _neg(q)
            prev = q
            out.append((int(round(f * max(1, mult))), q))
        return out

    try:                                 # vectorised fast path (Blender bundles NumPy)
        import numpy as np
        return _resample_numpy(np, keys, mult)
    except Exception:
        pass                             # pure-Python fallback (standalone tests)

    eval_fn, f0, f1 = build_squad(keys)
    start = int(round(f0 * mult))
    end = int(round(f1 * mult))
    return [(of, eval_fn(of / float(mult))) for of in range(start, end + 1)]
