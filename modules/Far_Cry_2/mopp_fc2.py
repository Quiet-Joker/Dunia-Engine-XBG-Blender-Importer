"""Reference Python implementation of the Avatar/Dunia MOPP virtual machine.

Decoded byte-for-byte from the interpreter at VA 0x10bfad30 in
Dunia_Retail_1.02_decrypted.dll (see agents.md "MOPP VM fully DECODED").
This is the ORACLE the bytecode emitter (mopp_build.py) must invert: emit a
program, run it here, and confirm it returns the right triangle set.

Two entry points:
  collect_all_keys(code)        -- infinite-query traversal: returns every
                                   triangle key the program can yield, in DFS
                                   order. Exercises all key arithmetic (SCALE /
                                   REOFFSET / split recursion / terminals)
                                   WITHOUT needing the runtime quantization.
  query(code, box, root_state)  -- full query with a quantized AABB; returns the
                                   culled triangle-key set (needs root state).

State block mirrors the engine's `esi`:
  bmax[3]  query box MAX (esi+0x00/04/08)
  bmin[3]  query box MIN (esi+0x10/14/18)
  off[3]   offset accumulators (esi+0x20/24/28)
  base     key base / reoffset accumulator (esi+0x2c)
  exp      accumulated scale exponent (esi+0x30)
  mask     key hi-bit mask (esi+0x34)
The engine copies-on-write into a frame-local block at the first REOFFSET, so a
recursed low-subtree never mutates its parent: we model that by copying the
state at every recursion boundary (pass-by-value into each frame).
"""

import copy
import struct

INVALID = set([0x08, 0x0e, 0x0f, 0x1d, 0x1e, 0x1f, 0x2c, 0x2d, 0x2e, 0x2f]
              + list(range(0x54, 0x60)) + list(range(0x6c, 0x100)))


class State:
    # qmax/qmin = the full-precision query corners (edi+0x10 / edi+0x20),
    # CONSTANT across the whole traversal; SCALE re-derives the local box from
    # them at the new exponent/offset (it reads edi, not the current box).
    __slots__ = ('bmax', 'bmin', 'off', 'base', 'exp', 'mask', 'qmax', 'qmin')

    def __init__(self):
        self.bmax = [1 << 30] * 3      # "infinite" box for collect-all
        self.bmin = [-(1 << 30)] * 3
        self.off = [0, 0, 0]
        self.base = 0
        self.exp = 0
        self.mask = 0
        self.qmax = None               # set by query(); None in collect-all
        self.qmin = None

    def clone(self):
        s = State.__new__(State)
        s.bmax = self.bmax[:]; s.bmin = self.bmin[:]; s.off = self.off[:]
        s.base = self.base; s.exp = self.exp; s.mask = self.mask
        s.qmax = self.qmax; s.qmin = self.qmin
        return s


def _be(code, i, n):
    v = 0
    for k in range(n):
        v = (v << 8) | code[i + k]
    return v


def _run(code, start, S, keys, cull, end=None, depth=0):
    """Execute one call frame from `start`. `cull`=False -> take all branches
    (collect-all). `end` bounds this frame's byte range: a subtree that ends by
    culling (a DOUBLE_CUT chain with no terminal) stops at `end` instead of
    spilling into its sibling. Returns on terminal/RETURN/end."""
    i = start
    n = len(code) if end is None else end
    while 0 <= i < n:
        op = code[i]
        if op in INVALID:
            raise ValueError("invalid op %#x @+%d" % (op, i))

        if op == 0x00:                       # RETURN
            return

        if 0x01 <= op <= 0x04:               # SCALE1..4
            shift = op
            lo = [(S.off[a] + code[i + 1 + a]) << shift for a in range(3)]
            S.off = lo
            S.exp += shift
            if cull and S.qmax is not None:
                cl = 16 - S.exp                       # >> amount (handler: 16-exp)
                S.bmax = [(S.qmax[a] >> cl) - lo[a] + 1 for a in range(3)]
                S.bmin = [(S.qmin[a] >> cl) - lo[a] for a in range(3)]
            # collect-all: box irrelevant (no culling), leave as-is
            i += 4; continue

        if op == 0x05:                       # JUMP8
            i += 2 + code[i + 1]; continue
        if op == 0x06:                       # JUMP16
            i += 3 + _be(code, i + 1, 2); continue
        if op == 0x07:                       # JUMP24
            i += 4 + _be(code, i + 1, 3); continue

        if op == 0x09:                       # REOFFSET8 (base += u8)
            S.base += code[i + 1]; i += 2; continue
        if op == 0x0a:                       # REOFFSET16
            S.base += _be(code, i + 1, 2); i += 3; continue
        if op == 0x0b:                       # REOFFSET32 (base = be32)
            S.base = _be(code, i + 1, 4); i += 5; continue
        if op == 0x0c:                       # JUMP_CHUNK (build mode 2 only)
            raise ValueError("chunked MOPP (op 0x0c) not supported")
        if op == 0x0d:                       # data / no-op
            i += 5; continue

        if 0x10 <= op <= 0x1c:               # SPLIT/CUT planes (4 bytes)
            low = code[i + 3]
            if not cull:
                _run(code, i + 4, S.clone(), keys, cull, i + 4 + low, depth + 1)
            i += 4 + low; continue

        if 0x20 <= op <= 0x22:               # SINGLE_SPLIT X/Y/Z
            axis = op - 0x20
            plane = code[i + 1]
            low = code[i + 2]
            go_low = go_high = True
            if cull:
                if S.bmax[axis] <= plane:
                    go_high = False
                elif S.bmin[axis] > plane:
                    go_low = False
            if go_low:
                _run(code, i + 3, S.clone(), keys, cull, i + 3 + low, depth + 1)
            if not go_high:
                return
            i += 3 + low; continue

        if 0x23 <= op <= 0x25:               # SPLIT2 X/Y/Z (7-byte header)
            # [op B1 B2 A:be16 LOW:be16]; low child @ +7+A, high child @ +7+LOW.
            # B2 = high-side plane (high relevant iff bmax>B2); B1 = low-side
            # plane (low irrelevant iff bmin>=B1). Mirrors handler 0x10bfafc8.
            axis = op - 0x23
            b1 = code[i + 1]; b2 = code[i + 2]
            A = _be(code, i + 3, 2); LOW = _be(code, i + 5, 2)
            lo_off = i + 7 + A
            hi_off = i + 7 + LOW
            maxv = S.bmax[axis]; minv = S.bmin[axis]
            if cull and maxv <= b2:
                if minv >= b1:
                    return                    # cull
                i = lo_off; continue          # low only
            # maxv > b2 (or collect-all): high present
            go_low = (not cull) or (minv < b1)
            if go_low:
                _run(code, lo_off, S.clone(), keys, cull, hi_off, depth + 1)
            i = hi_off; continue              # continue high

        if 0x26 <= op <= 0x28:               # DOUBLE_CUT X/Y/Z (clip, no branch)
            axis = op - 0x26
            lo = code[i + 1]; hi = code[i + 2]
            if cull and (S.bmax[axis] < lo or S.bmin[axis] >= hi):
                return
            i += 3; continue

        if 0x29 <= op <= 0x2b:               # DOUBLE_CUT32 (7 bytes)
            i += 7; continue

        if 0x30 <= op <= 0x4f:               # TERMINAL
            keys.append(S.base + (op - 0x30)); return
        if op == 0x50:
            keys.append(S.base + code[i + 1]); return
        if op == 0x51:
            keys.append(S.base + _be(code, i + 1, 2)); return
        if op == 0x52:
            keys.append(S.base + _be(code, i + 1, 3)); return
        if op == 0x53:
            keys.append(S.base + _be(code, i + 1, 4)); return

        if 0x60 <= op <= 0x63:
            i += 2; continue
        if 0x64 <= op <= 0x67:
            i += 3; continue
        if 0x68 <= op <= 0x6b:
            i += 5; continue
        raise ValueError("unhandled op %#x @+%d" % (op, i))


def collect_all_keys(code):
    keys = []
    _run(code, 0, State(), keys, cull=False)
    return keys


def query(code, qmin, qmax):
    """Run with culling against the FULL-PRECISION quantized query corners
    (qmin/qmax are the (world-offset)*scale ints). Root state mirrors the engine
    setup: exp=0, off=0, box = (Q>>16) (the hi-word coarse grid). SCALE refines.
    Returns the set of triangle keys not culled."""
    S = State()
    S.qmax = list(qmax); S.qmin = list(qmin)
    S.off = [0, 0, 0]; S.base = 0; S.exp = 0
    cl = 16
    S.bmax = [(qmax[a] >> cl) + 1 for a in range(3)]
    S.bmin = [(qmin[a] >> cl) for a in range(3)]
    keys = []
    _run(code, 0, S, keys, cull=True)
    return set(keys)


# ============================================================
# MOPP EMITTER (compiler) — was mopp_build.py. Builds bytecode the
# VM above (was mopp_vm.py) interprets to a conservative tri set.
# ============================================================

GRID = 255
QBITS = 24                         # full-precision quantization (24-bit grid)
ROOT_SHIFT = 16                    # VM: root box = Q >> 16; SCALE refines
# Leave 1 byte of headroom at the top: the root box (Q>>16) maxes at 254 so a
# DOUBLE_CUT upper bound (box.hi+1) is always <=255 and representable. Without
# this, a triangle touching local 255 is wrongly culled for a top-face query.
QMAX = (255 << 16) - 1             # 0xFEFFFF -> (>>16) == 254

# opcodes
OP_RET = 0x00
OP_DCUT = (0x26, 0x27, 0x28)      # DOUBLE_CUT X, Y, Z (clip to [lo, hi))
OP_SCALE = (0x01, 0x02, 0x03, 0x04)   # SCALE by 1..4 bits (index = n-1)
OP_SPLIT = (0x20, 0x21, 0x22)     # X, Y, Z
OP_TERM16 = 0x51


def quantize(verts):
    """verts: list of (x,y,z). Returns (qverts, offset(3), scale) with a single
    uniform scale (matches m_info's single w-scale). 24-bit grid so the VM's
    >>16 root sees the top 8 bits and SCALE peels the lower 16."""
    xs = [v[0] for v in verts]; ys = [v[1] for v in verts]; zs = [v[2] for v in verts]
    mn = (min(xs), min(ys), min(zs))
    mx = (max(xs), max(ys), max(zs))
    ext = max(mx[a] - mn[a] for a in range(3)) or 1.0
    scale = QMAX / ext
    q = []
    for v in verts:
        qq = tuple(min(QMAX, max(0, int(round((v[a] - mn[a]) * scale)))) for a in range(3))
        q.append(qq)
    return q, mn, scale


def tri_boxes(qverts, tris):
    """Per-triangle quantized AABB: list of (lo(3), hi(3))."""
    boxes = []
    for (a, b, c) in tris:
        va, vb, vc = qverts[a], qverts[b], qverts[c]
        lo = tuple(min(va[k], vb[k], vc[k]) for k in range(3))
        hi = tuple(max(va[k], vb[k], vc[k]) for k in range(3))
        boxes.append((lo, hi))
    return boxes


def _emit_leaf(idx):
    return bytes([OP_TERM16, (idx >> 8) & 0xFF, idx & 0xFF])


# module-level flag set when a non-conservative fallback had to be used
_LOSSY = [False]
# count of triangles overhanging a split plane into the sibling region (the
# broad-phase residual that a low-only query may miss; shipped MOPP has it too)
_OVERHANG = [0]

OP_SPLIT2 = (0x23, 0x24, 0x25)            # X, Y, Z   (2-byte lowSize)
MAX_LOW2 = 0xFFFF                          # SPLIT2 low size field is be16


def _emit_split(axis, plane, low_bytes, high_bytes):
    """SINGLE_SPLIT if the inline `low` child fits the 1-byte size field, else
    SPLIT2 (2-byte). SPLIT2 layout [op B1 B2 A:be16 LOW:be16][low][high] with
    A=0, B1=plane+1, B2=plane reproduces a clean median split (see mopp_vm)."""
    nl = len(low_bytes)
    if nl <= 255:
        return bytes([OP_SPLIT[axis], plane, nl]) + low_bytes + high_bytes
    if nl <= MAX_LOW2:
        b1 = min(GRID, plane + 1); b2 = plane
        hdr = bytes([OP_SPLIT2[axis], b1, b2, 0, 0, (nl >> 8) & 0xFF, nl & 0xFF])
        return hdr + low_bytes + high_bytes
    return None                            # >64 KB inline low: caller re-splits


def _local_boxes(items, boxes, exp, off):
    """Per-triangle box in the current subtree's local coordinate frame, exactly
    as the VM computes it: local = (full >> (16-exp)) - off, per axis."""
    sh = ROOT_SHIFT - exp
    lb = {}
    for i in items:
        lo, hi = boxes[i]
        lb[i] = (tuple((lo[a] >> sh) - off[a] for a in range(3)),
                 tuple((hi[a] >> sh) - off[a] for a in range(3)))
    return lb


def _split_sets(items, lboxes, axis, plane):
    """Conservative partition in LOCAL coords: low = box reaches [.,plane], high
    = box reaches (plane,.]. Straddlers -> BOTH (provably conservative)."""
    low = [i for i in items if lboxes[i][0][axis] <= plane]
    high = [i for i in items if lboxes[i][1][axis] > plane]
    return low, high


def _best_split(items, lboxes):
    """(axis, plane, low, high) where BOTH children strictly shrink, most
    balanced; or None if no byte plane separates at this resolution."""
    n = len(items)
    best = None
    for axis in range(3):
        los = sorted(set(lboxes[i][0][axis] for i in items))
        his = sorted(set(lboxes[i][1][axis] for i in items))
        cands = sorted(set(los + his))
        for plane in cands:
            if plane < 0 or plane >= GRID:
                continue
            low, high = _split_sets(items, lboxes, axis, plane)
            if 0 < len(low) < n and 0 < len(high) < n:
                dup = len(low) + len(high) - n
                bal = abs(len(low) - len(high))
                score = bal + dup
                if best is None or score < best[0]:
                    best = (score, axis, plane, low, high)
    return None if best is None else best[1:]


def _emit_scale(items, boxes, exp, off):
    """Emit a SCALE op to refine resolution, then recurse. Picks shift n (1..4)
    and the new offset so the subtree's tris re-center near local 0 at the finer
    level. Returns (bytecode, handled) or (None) if exp is maxed."""
    n = min(4, ROOT_SHIFT - exp)          # cannot exceed exp==16 (sh>=0)
    if n <= 0:
        return None
    newexp = exp + n
    sh = ROOT_SHIFT - newexp
    # b[a] (byte operand) chosen so newoff = (off+b)<<n sits at the tris' min.
    b = []
    for a in range(3):
        tmin = min(boxes[i][0][a] >> sh for i in items)   # tris' min at newexp
        ba = (tmin >> n) - off[a]
        b.append(max(0, min(255, ba)))
    newoff = [(off[a] + b[a]) << n for a in range(3)]
    body = _build(items, boxes, newexp, newoff)
    return bytes([OP_SCALE[n - 1], b[0], b[1], b[2]]) + body


def _envelope(items, lboxes):
    """Union local AABB of items: (lo3, hi3)."""
    lo = [min(lboxes[i][0][a] for i in items) for a in range(3)]
    hi = [max(lboxes[i][1][a] for i in items) for a in range(3)]
    return lo, hi


def _emit_dcut(env):
    """3-axis DOUBLE_CUT clipping the query to the envelope [lo, hi]. DCUT
    continues iff NOT(bmax<lo or bmin>=hi); to keep box [blo,bhi] use lo=blo,
    hi=bhi+1. Axes that span the full byte range add no culling and are skipped."""
    out = bytearray()
    for a in range(3):
        lo = max(0, min(255, env[0][a]))
        hi = max(0, min(255, env[1][a] + 1))
        if lo == 0 and hi >= 255:
            continue                       # spans the cell -> no cull, skip
        out += bytes([OP_DCUT[a], lo, hi])
    return bytes(out)


def _build(items, boxes, exp=0, off=(0, 0, 0)):
    """DOUBLE_CUT-AABB BVH (shipped architecture). Returns bytecode that starts
    with this subtree's envelope clip, then either a leaf terminal or a split
    into two centroid-partitioned children. No duplication."""
    lboxes = _local_boxes(items, boxes, exp, off)

    if len(items) == 1:
        return _emit_dcut(lboxes[items[0]]) + _emit_leaf(items[0])

    env = _envelope(items, lboxes)

    # No-duplication BVH (matches shipped maxrep==1, keeps the file compact):
    # hard-partition triangles by centroid on the axis with the widest spread,
    # one triangle per leaf. Each child's DOUBLE_CUT envelope does the precise
    # culling. The split plane = L's max extent so the LOW branch is never
    # wrongly culled; the residual is a few R-triangles overhanging below the
    # plane that a low-only query can miss (the same broad-phase residual shipped
    # MOPP has, covered by collision tolerance) -> reported via _OVERHANG.
    cents = {i: tuple(lboxes[i][0][a] + lboxes[i][1][a] for a in range(3)) for i in items}
    spreads = [max(cents[i][a] for i in items) - min(cents[i][a] for i in items)
               for a in range(3)]
    axis = max(range(3), key=lambda a: spreads[a])

    if spreads[axis] == 0:
        scaled = _emit_scale(items, boxes, exp, off)
        if scaled is not None:
            return _emit_dcut(env) + scaled
        # exp maxed and centroids still coincident: nest as guarded leaves
        _LOSSY[0] = True
        inner = _emit_dcut(lboxes[items[-1]]) + _emit_leaf(items[-1])
        for i in reversed(items[:-1]):
            leaf = _emit_dcut(lboxes[i]) + _emit_leaf(i)
            node = _emit_split(0, max(0, min(GRID, env[1][0])), leaf, inner)
            inner = node if node is not None else (leaf + inner)
        return _emit_dcut(env) + inner

    order = sorted(items, key=lambda i: cents[i][axis])
    mid = len(order) // 2
    low, high = order[:mid], order[mid:]
    lmax = max(lboxes[i][1][axis] for i in low)
    rmin = min(lboxes[i][0][axis] for i in high)
    if lmax >= rmin:                       # children overlap on axis -> overhang
        _OVERHANG[0] += sum(1 for i in high if lboxes[i][0][axis] <= lmax)
    plane = max(0, min(GRID - 1, lmax))

    lb = _build(low, boxes, exp, off)
    hb = _build(high, boxes, exp, off)
    node = _emit_split(axis, plane, lb, hb)
    if node is None:
        _LOSSY[0] = True
        node = bytes([OP_SPLIT[axis], plane, 255]) + lb + hb
    return _emit_dcut(env) + node


def build_mopp(verts, tris):
    """Returns (mopp_bytecode, m_info(ox,oy,oz,scale), info: dict).
    info = {'lossy': bool, 'overhang': int} — `overhang` counts triangles that
    overhang a split plane into the sibling region (the broad-phase residual a
    low-only query may miss; shipped MOPP has the same kind of residual, covered
    by collision tolerance). `lossy` flags the rarer exp-maxed coincident case.
    A purely conservative result has overhang==0 and lossy==False."""
    _LOSSY[0] = False
    _OVERHANG[0] = 0
    qverts, off, scale = quantize(verts)
    boxes = tri_boxes(qverts, tris)
    code = _build(list(range(len(tris))), boxes)
    code += bytes([OP_RET])           # safety terminator
    return (code, (off[0], off[1], off[2], scale),
            {'lossy': _LOSSY[0], 'overhang': _OVERHANG[0]})


def quantize_query(box_world, m_info):
    """world AABB (wmin, wmax) -> FULL-PRECISION quantized query corners
    (qmin, qmax) = round((world - offset) * scale), clamped to the 24-bit grid.
    These feed mopp_vm.query(), which does the >>16 root + SCALE refinement
    itself (matching the engine setup at 0x10bfb4e0)."""
    ox, oy, oz, scale = m_info
    off = (ox, oy, oz)
    wmin, wmax = box_world
    import math
    qmin = [max(0, min(QMAX, int(math.floor((wmin[a] - off[a]) * scale)))) for a in range(3)]
    qmax = [max(0, min(QMAX, int(math.ceil((wmax[a] - off[a]) * scale)))) for a in range(3)]
    return qmin, qmax
