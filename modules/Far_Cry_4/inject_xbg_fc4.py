"""inject_xbg_fc3.py — FC3 / FC4 XBG mesh injector (in-place vertex patch).

Re-injects Blender objects tagged with 'xbg_fc3_data' back into the source
XBG binary.  Independent from the Avatar/FC2 pipeline (inject_xbg.py).

Workflow
--------
1. Import with "Separate Primitives" ON  → one Blender object per section.
2. Edit the section meshes (move / sculpt verts, edit normals, UVs, colors).
3. Select the section objects and run Inject FC3/FC4.
4. A new .xbg is written at the chosen path.

Design — PRESERVE EVERYTHING, patch only what changed
-----------------------------------------------------
The FC3/FC4 vertex buffer is a flag-driven interleave of up to nine
components (see _VTX_COMPONENTS): i16 position, two i16 UV channels, u8x4
bone weights + u8x4 palette indices, i8 normal, u8x4 colour, i8 tangent and
i8 binormal (each with a handedness/pad byte).  Vertex flags/strides differ
per VB (FC4 characters use flag 0x0bda, stride 40).

Rather than re-encode (which would drop the components Blender can't supply —
bone weights, pos.w, handedness — and silently corrupt skinning), this
injector reads each section's ORIGINAL vertex bytes and overwrites only the
geometry components the user can edit: position, normal, tangent, binormal,
UV0, UV1, vertex colour.  Bone weights/indices, the position w, the
normal/tangent/binormal pad+handedness bytes and any unknown fields survive
byte-for-byte.  Verified byte-exact on a decode→patch round-trip of every
vertex in sabal / firstperson_* (18,987 verts, 0 diffs).

This keeps the vertex COUNT and ORDER (reshape / sculpt / repaint).  Changing
the vertex count routes to a clear "not supported yet" message — a full
rebuild with bone-weight re-binding is the next phase.
"""

import os
import struct

import bpy
import bmesh

from ..Core.debug import VerboseLogger
from . import import_xbg_fc4 as import_xbg_fc3  # FC4-owned parser

vlog = VerboseLogger


# ── Vertex component layout (flag, byte size) in interleave order ────────────
# Mirrors import_xbg_fc3._read_vertex / VertexFlags.COMPONENT_ORDER exactly.
_VTX_COMPONENTS = [
    (0x0001, 12),  # POS_FLOAT
    (0x0002,  8),  # POS_INT16
    (0x0004,  8),  # POS_HALF
    (0x0008,  4),  # UV0
    (0x0800,  4),  # UV1
    (0x1000,  4),  # UV2
    (0x0010,  8),  # BONE_WTS1 (u8x4 weights + u8x4 indices)
    (0x0020,  8),  # BONE_WTS2
    (0x0040,  4),  # NORMAL  (i8x3 + pad)
    (0x0080,  4),  # COLOR   (u8x4 RGBA)
    (0x0100,  4),  # TANGENT (i8x3 + handedness)
    (0x0200,  4),  # BINORMAL(i8x3 + handedness)
    (0x0400,  4),  # UNK_400
]
_POS_FLAGS = (0x0001, 0x0002, 0x0004)


def _vertex_offsets(flag):
    """flag -> {component_flag: byte_offset} for the components present."""
    off = 0
    out = {}
    pos_done = False
    for fl, size in _VTX_COMPONENTS:
        if fl in _POS_FLAGS:
            if pos_done or not (flag & fl):
                continue
            pos_done = True
        elif not (flag & fl):
            continue
        out[fl] = off
        off += size
    return out


def _i16(v):
    return max(-32768, min(32767, int(round(v * 16383.5))))


def _s2u(v):
    """Encode a [-1,1] component to a D3DCOLOR unsigned byte (inverse of the
    importer's b/255*2-1; exactly invertible for all 256 values)."""
    return max(0, min(255, int(round((v + 1.0) * 0.5 * 255.0))))


def _u8(v):
    return max(0, min(255, int(round(v))))


def _patch_vertex(orig, flag, comp):
    """Return `orig` (stride bytes) with the geometry components in `comp`
    overwritten in place.  `comp` keys (any subset): 'p','uv','uv1','n',
    'color','t','b'.  Everything else (bone weights, pos.w, pad/handedness
    bytes, unknowns) is preserved verbatim."""
    o = _vertex_offsets(flag)
    buf = bytearray(orig)
    if 'p' in comp and 0x0002 in o:                       # i16 xyz (keep w)
        p = o[0x0002]
        struct.pack_into('<3h', buf, p,
                         _i16(comp['p'][0]), _i16(comp['p'][1]), _i16(comp['p'][2]))
    elif 'p' in comp and 0x0001 in o:                     # float xyz
        struct.pack_into('<3f', buf, o[0x0001], *comp['p'][:3])
    if 'uv' in comp and 0x0008 in o:
        p = o[0x0008]
        struct.pack_into('<2h', buf, p,
                         _i16(comp['uv'][0] - 1.0), _i16(2.0 - comp['uv'][1]))
    if 'uv1' in comp and 0x0800 in o:
        p = o[0x0800]
        if comp['uv1'] is None:
            struct.pack_into('<2h', buf, p, -32768, -32768)
        else:
            struct.pack_into('<2h', buf, p,
                             _i16(comp['uv1'][0] - 1.0), _i16(2.0 - comp['uv1'][1]))
    # Normal/tangent/binormal/color are D3DCOLOR: unsigned-normalised, BGRA
    # byte order (byte2=x, byte1=y, byte0=z; for colour byte0=B,1=G,2=R,3=A).
    if 'n' in comp and 0x0040 in o:                       # keep byte3 (pad)
        p = o[0x0040]
        n = comp['n']
        struct.pack_into('<3B', buf, p, _s2u(n[2]), _s2u(n[1]), _s2u(n[0]))
    if 'color' in comp and 0x0080 in o:                   # R,G,B,A -> B,G,R,A
        p = o[0x0080]
        c = comp['color']
        struct.pack_into('<4B', buf, p, _u8(c[2]), _u8(c[1]), _u8(c[0]), _u8(c[3]))
    if 't' in comp and 0x0100 in o:                       # keep byte3 (handedness)
        p = o[0x0100]
        t = comp['t']
        struct.pack_into('<3B', buf, p, _s2u(t[2]), _s2u(t[1]), _s2u(t[0]))
    if 'b' in comp and 0x0200 in o:                       # keep byte3 (handedness)
        p = o[0x0200]
        b = comp['b']
        struct.pack_into('<3B', buf, p, _s2u(b[2]), _s2u(b[1]), _s2u(b[0]))
    return bytes(buf)


def _encode_vertex_full(flag, stride, comp, si, sw):
    """Encode a whole vertex from scratch (for rebuilt/added geometry).

    `comp` keys: 'p','uv','uv1','n','color','t','b'.  `si`/`sw` are the u8x4
    palette-local bone indices / weights.  Pad and handedness bytes get sane
    defaults (0 / 0x80) — validated to re-decode with 0 functional diffs.
    """
    o = _vertex_offsets(flag)
    b = bytearray(stride)
    if 0x0002 in o:
        p = comp.get('p', (0.0, 0.0, 0.0))
        struct.pack_into('<4h', b, o[0x0002], _i16(p[0]), _i16(p[1]), _i16(p[2]), 0)
    elif 0x0001 in o:
        struct.pack_into('<3f', b, o[0x0001], *comp.get('p', (0.0, 0.0, 0.0)))
    if 0x0008 in o and 'uv' in comp:
        struct.pack_into('<2h', b, o[0x0008],
                         _i16(comp['uv'][0] - 1.0), _i16(2.0 - comp['uv'][1]))
    if 0x0800 in o:
        u = comp.get('uv1')
        if u:
            struct.pack_into('<2h', b, o[0x0800], _i16(u[0] - 1.0), _i16(2.0 - u[1]))
        else:
            struct.pack_into('<2h', b, o[0x0800], -32768, -32768)
    if 0x0010 in o:
        struct.pack_into('<8B', b, o[0x0010],
                         sw[0], sw[1], sw[2], sw[3], si[0], si[1], si[2], si[3])
    if 0x0040 in o and 'n' in comp:                       # D3DCOLOR BGRA + pad
        n = comp['n']
        struct.pack_into('<4B', b, o[0x0040], _s2u(n[2]), _s2u(n[1]), _s2u(n[0]), 0)
    if 0x0080 in o and 'color' in comp:                   # R,G,B,A -> B,G,R,A
        c = comp['color']
        struct.pack_into('<4B', b, o[0x0080], _u8(c[2]), _u8(c[1]), _u8(c[0]), _u8(c[3]))
    if 0x0100 in o and 't' in comp:                       # D3DCOLOR BGRA + handedness
        t = comp['t']
        struct.pack_into('<4B', b, o[0x0100], _s2u(t[2]), _s2u(t[1]), _s2u(t[0]), 0x80)
    if 0x0200 in o and 'b' in comp:                       # D3DCOLOR BGRA + handedness
        bn = comp['b']
        struct.pack_into('<4B', b, o[0x0200], _s2u(bn[2]), _s2u(bn[1]), _s2u(bn[0]), 0x80)
    return bytes(b)


def _encode_weights(influences, palette, name2idx):
    """{bone_name: weight} -> (si[4], sw[4]) u8 palette-local indices + weights.

    Maps each bone name to its global index, then to the section's palette-local
    index (the reverse of import's palette[local]=global).  Keeps the four
    heaviest influences and normalises the weights to sum 255.  Bones not in the
    section palette are dropped (can't be expressed without growing the palette).
    Validated: re-encoding stock weights reproduces the exact bone set.
    """
    g2l = {}
    for li, gi in enumerate(palette):
        if gi not in g2l:
            g2l[gi] = li
    pairs = []
    for nm, w in influences.items():
        gi = name2idx.get(nm)
        if gi is None or gi not in g2l or w <= 0.0:
            continue
        pairs.append((g2l[gi], float(w)))
    pairs.sort(key=lambda x: -x[1])
    pairs = pairs[:4]
    if not pairs:
        return [0, 0, 0, 0], [0, 0, 0, 0]
    tot = sum(w for _, w in pairs)
    sw = [int(round(w / tot * 255)) for _, w in pairs]
    sw[0] += 255 - sum(sw)                       # fix rounding onto the heaviest
    si = [li for li, _ in pairs]
    while len(si) < 4:
        si.append(0)
        sw.append(0)
    return si, sw


def _fill_missing_tangents(comps, faces):
    """Compute a tangent + binormal from geometry & UVs for any vertex that has
    no authored tangent (key 't' absent or ~zero) — e.g. verts added by an
    extrude.  Standard per-triangle tangent accumulation + Gram-Schmidt against
    the vertex normal; binormal = normal × tangent.  Authored tangents on
    unchanged verts are left untouched.
    """
    def _zero(v):
        return v is None or (abs(v[0]) < 1e-6 and abs(v[1]) < 1e-6 and abs(v[2]) < 1e-6)

    todo = [i for i in range(len(comps))
            if _zero(comps[i].get('t')) or _zero(comps[i].get('b'))]
    if not todo:
        return

    acc = [[0.0, 0.0, 0.0] for _ in range(len(comps))]
    for (a, b, c) in faces:
        ca, cb, cc = comps[a], comps[b], comps[c]
        if 'uv' not in ca or 'uv' not in cb or 'uv' not in cc:
            continue
        p0, p1, p2 = ca['p'], cb['p'], cc['p']
        u0, u1, u2 = ca['uv'], cb['uv'], cc['uv']
        e1 = (p1[0]-p0[0], p1[1]-p0[1], p1[2]-p0[2])
        e2 = (p2[0]-p0[0], p2[1]-p0[1], p2[2]-p0[2])
        d1v = u1[1]-u0[1]
        d2v = u2[1]-u0[1]
        d1u = u1[0]-u0[0]
        d2u = u2[0]-u0[0]
        denom = d1u*d2v - d2u*d1v
        if abs(denom) < 1e-12:
            continue
        r = 1.0 / denom
        t = ((e1[0]*d2v - e2[0]*d1v) * r,
             (e1[1]*d2v - e2[1]*d1v) * r,
             (e1[2]*d2v - e2[2]*d1v) * r)
        for vi in (a, b, c):
            acc[vi][0] += t[0]; acc[vi][1] += t[1]; acc[vi][2] += t[2]

    for vi in todo:
        nrm = comps[vi].get('n', (0.0, 0.0, 1.0))
        t = acc[vi]
        d = nrm[0]*t[0] + nrm[1]*t[1] + nrm[2]*t[2]      # Gram-Schmidt
        t = (t[0]-nrm[0]*d, t[1]-nrm[1]*d, t[2]-nrm[2]*d)
        L = (t[0]*t[0] + t[1]*t[1] + t[2]*t[2]) ** 0.5
        if L < 1e-8:
            tang = (1.0, 0.0, 0.0)
        else:
            tang = (t[0]/L, t[1]/L, t[2]/L)
        binorm = (nrm[1]*tang[2] - nrm[2]*tang[1],
                  nrm[2]*tang[0] - nrm[0]*tang[2],
                  nrm[0]*tang[1] - nrm[1]*tang[0])
        comps[vi]['t'] = tang
        comps[vi]['b'] = binorm


# ── SDOL geometry walk ───────────────────────────────────────────────────────

def _sdol_offset(data):
    """Absolute byte offset of the SDOL chunk, via the validated chunk walk
    (the old byte-stride _find_chunk fails on the FC4 header layout)."""
    for name, off, *_ in import_xbg_fc3.parse_xbg(bytes(data))['chunks']:
        if name == 'SDOL':
            return off
    return -1


def _walk_lod(data, sdol_off, lod_idx):
    """Walk the SDOL to the requested LOD.

    Returns dict:
        'vb_meta'  : [(flag, stride, vcount, offset), ...]
        'vb_base'  : [abs byte offset of each VB's vertex data]
        'vb_voff'  : [cumulative vertex index where each VB begins]
        'indices'  : tuple of u16 local indices (the shared LOD index buffer)
    """
    s = import_xbg_fc3._Stream(data)
    s.setpos(sdol_off + 20)                 # past 20-byte chunk header
    n_lods = s.u32()
    for li in range(n_lods):
        s.f32()                              # lod distance
        vbc = s.u32()
        vb_meta = [(s.u32(), s.u32(), s.u32(), s.u32()) for _ in range(vbc)]
        ne = s.u32()
        s.p += ne * 28                       # entries
        s.u32()                              # vb_size
        s.align(16)
        vb_base = []
        vb_voff = []
        voff = 0
        base = s.p
        for flag, stride, vcount, _ in vb_meta:
            vb_base.append(base)
            vb_voff.append(voff)
            base += vcount * stride
            voff += vcount
        s.p = base
        total_idx = s.u32()
        s.align(16)
        indices = struct.unpack_from('<%dH' % total_idx, data, s.p)
        s.p += total_idx * 2
        if li == lod_idx:
            return {'vb_meta': vb_meta, 'vb_base': vb_base,
                    'vb_voff': vb_voff, 'indices': indices}
    raise IndexError("LOD %d not found in SDOL" % lod_idx)


def _section_global_verts(lod_walk, vb_index, idx_start, idx_end):
    """The sorted global vertex indices a section covers, in the SAME order
    the importer built the section mesh (sorted unique).  Maps Blender vertex
    `lv` -> global index g_verts[lv]."""
    vb_voff = lod_walk['vb_voff'][vb_index]
    indices = lod_walk['indices']
    used = set()
    for k in range(idx_start, min(idx_end, len(indices))):
        used.add(indices[k] + vb_voff)
    return sorted(used)


# ── Blender component extraction ─────────────────────────────────────────────

def _extract_components(obj):
    """Per-vertex geometry components from a Blender object.

    Returns (n_verts, comps) where comps[i] is a dict with keys
    'p','uv','uv1','n','color','t','b' (only those present).  Per-loop data
    (normals/UVs/colour) is collapsed to per-vertex, first-referenced-loop
    wins — matching how the importer collapsed them.
    """
    depsgraph = bpy.context.evaluated_depsgraph_get()
    eval_obj = obj.evaluated_get(depsgraph)
    me = eval_obj.to_mesh()

    n = len(me.vertices)
    comps = [dict() for _ in range(n)]
    for i, v in enumerate(me.vertices):
        comps[i]['p'] = (v.co.x, v.co.y, v.co.z)

    # Normals: prefer the RAW authored vectors stored by the importer in the
    # xbg_normal POINT attribute -- Blender's own corner_normals are
    # NORMALIZED, and authored XBG normals are often slightly non-unit
    # (~0.996), so encoding the normalized value shifted every normal byte
    # by +/-1 (broke byte-exact zero-edit round-trips).  Verts whose
    # attribute entry is zero (e.g. added by an edit) fall back to corner
    # normals.
    na = me.attributes.get('xbg_normal')
    if na and na.domain == 'POINT' and len(na.data) == n:
        for vi in range(n):
            v = na.data[vi].vector
            if v[0] * v[0] + v[1] * v[1] + v[2] * v[2] > 1e-6:
                comps[vi]['n'] = (v[0], v[1], v[2])
    # Split normals: Blender 5.0 exposes them via mesh.corner_normals (the old
    # calc_normals_split() was removed in 4.1 and clobbered custom normals).
    corner_normals = getattr(me, 'corner_normals', None)
    seen_n = [False] * n
    for poly in me.polygons:
        for li in poly.loop_indices:
            vi = me.loops[li].vertex_index
            if not seen_n[vi] and 'n' not in comps[vi]:
                if corner_normals is not None:
                    nl = corner_normals[li].vector
                else:
                    nl = me.loops[li].normal
                comps[vi]['n'] = (nl.x, nl.y, nl.z)
                seen_n[vi] = True

    def _collapse_uv(layer_name, key):
        uvl = me.uv_layers.get(layer_name)
        if not uvl:
            return
        done = [False] * n
        for poly in me.polygons:
            for li in poly.loop_indices:
                vi = me.loops[li].vertex_index
                if not done[vi]:
                    uv = uvl.data[li].uv
                    comps[vi][key] = (uv[0], uv[1])
                    done[vi] = True
    _collapse_uv('UVMap', 'uv')
    _collapse_uv('UV1', 'uv1')

    col = me.color_attributes.get('Col') if hasattr(me, 'color_attributes') else None
    if col is None and me.vertex_colors:
        col = me.vertex_colors.get('Col') or me.vertex_colors[0]
    if col is not None:
        done = [False] * n
        # color attribute may be per-corner (loop) or per-point (vertex)
        per_point = getattr(col, 'domain', 'CORNER') == 'POINT'
        if per_point:
            for vi in range(n):
                c = col.data[vi].color
                comps[vi]['color'] = (c[0]*255, c[1]*255, c[2]*255, c[3]*255)
        else:
            for poly in me.polygons:
                for li in poly.loop_indices:
                    vi = me.loops[li].vertex_index
                    if not done[vi]:
                        c = col.data[li].color
                        comps[vi]['color'] = (c[0]*255, c[1]*255, c[2]*255, c[3]*255)
                        done[vi] = True

    ta = me.attributes.get('xbg_tangent')
    if ta and len(ta.data) == n:
        for vi in range(n):
            vec = ta.data[vi].vector
            comps[vi]['t'] = (vec.x, vec.y, vec.z)
    ba = me.attributes.get('xbg_binormal')
    if ba and len(ba.data) == n:
        for vi in range(n):
            vec = ba.data[vi].vector
            comps[vi]['b'] = (vec.x, vec.y, vec.z)

    # Per-vertex bone influences (name -> weight), from the object's vertex
    # groups; needed only by the rebuild path.
    vg_names = {vg.index: vg.name for vg in obj.vertex_groups}
    weights = [dict() for _ in range(n)]
    for vi, v in enumerate(me.vertices):
        for g in v.groups:
            nm = vg_names.get(g.group)
            if nm and g.weight > 0.0:
                weights[vi][nm] = weights[vi].get(nm, 0.0) + g.weight

    # Fan-triangulate every polygon.  Extrudes / edits create quads and
    # n-gons; keeping only existing triangles would DROP those faces and leave
    # the freshly added vertices unconnected (the "added verts but no faces"
    # bug).  Fan triangulation needs no bmesh, so the custom split normals read
    # above stay intact.
    faces = []
    for p in me.polygons:
        vs = p.vertices
        for i in range(1, len(vs) - 1):
            faces.append((vs[0], vs[i], vs[i + 1]))

    # New verts (e.g. from an extrude) have no authored tangent space — derive
    # one from geometry + UVs so added geometry shades correctly.
    _fill_missing_tangents(comps, faces)

    # Import reversed FC3/FC4's winding for correct Blender display; reverse it
    # back to the original-format convention before writing.  Done AFTER tangent
    # computation (which runs in Blender winding, consistent with the outward
    # Blender normals).
    faces = [(a, c, b) for (a, b, c) in faces]

    eval_obj.to_mesh_clear()
    return n, comps, weights, faces


# ── Rebuild path (changed vertex count) ──────────────────────────────────────

def _section_entry_index(lods, lod_idx, vb_i, sec_i):
    """Index (within the LOD's entry list) of SDOL section (vb_i, sec_i)."""
    cnt = {}
    for k, e in enumerate(lods[lod_idx].get('entries', [])):
        v = e[0]
        c = cnt.get(v, 0)
        if v == vb_i and c == sec_i:
            return k
        cnt[v] = c + 1
    return None


def _section_palette(lods, lod_idx, vb_i, sec_i, palettes):
    """The SULC bone palette (local->global) for SDOL section (vb_i, sec_i)."""
    sulc_start = sum(len(lods[j].get('entries', [])) for j in range(lod_idx))
    k = _section_entry_index(lods, lod_idx, vb_i, sec_i)
    if k is None:
        return []
    gi = sulc_start + k
    return palettes[gi] if gi < len(palettes) else []


def _sync_sulc_counts(data, lods, lod_idx, overrides):
    """Patch the SULC per-section vertex/index counts for rebuilt sections so
    the skinning metadata stays consistent with the new geometry.

    The per-section count fields live at a fixed u16 offset from each 3034
    section marker, but that offset varies between files (palette / metadata
    size differs).  So this CALIBRATES the offset for this file: it finds the
    u16 offset whose value equals the known vertex count of EVERY section, and
    likewise for the index count.  Requiring agreement across all sections (a
    random offset can't match every section's distinct counts) makes a false
    positive astronomically unlikely.

    SELF-VALIDATING: if the marker count is wrong, or no consistent vert/index
    offset exists, it leaves SULC completely untouched and returns False — a
    stale-but-valid SULC is far safer than wrongly-written bytes.  `data` is a
    mutable bytearray; returns True if it patched.
    """
    def u16(off):
        return struct.unpack_from('<H', data, off)[0]

    chunks = import_xbg_fc3.parse_xbg(bytes(data))['chunks']
    dnks = next((c for c in chunks if c[0] == 'DNKS'), None)
    if not dnks:
        return False
    dnks_off = dnks[1]
    dnks_size = struct.unpack_from('<I', data, dnks_off + 8)[0]
    sulc_off = bytes(data).find(b'SULC', dnks_off, dnks_off + dnks_size)
    if sulc_off < 0:
        return False
    pay = sulc_off + 20
    pay_end = pay + struct.unpack_from('<I', data, sulc_off + 12)[0]

    starts = []
    p = pay + 16
    while p < pay_end - 1:
        if u16(p) == 3034:
            starts.append(p)
        p += 2
    if len(starts) != sum(len(L.get('entries', [])) for L in lods):
        return False                              # marker scan unreliable here

    base = sum(len(lods[j].get('entries', [])) for j in range(lod_idx))
    walk = _walk_lod(data, _sdol_offset(data), lod_idx)

    # Gather every section in this LOD with its known original counts.
    secs = {}     # (vb_i, sec_i) -> (marker_start, orig_vcount, orig_icount)
    for vb_i, vb in enumerate(lods[lod_idx]['vbs']):
        for sec_i, (mat_i, idx_s, idx_e) in enumerate(vb.get('sections', [])):
            k = _section_entry_index(lods, lod_idx, vb_i, sec_i)
            si = base + k
            if si >= len(starts):
                return False
            g = _section_global_verts(walk, vb_i, idx_s, idx_e)
            secs[(vb_i, sec_i)] = (starts[si], len(g), idx_e - idx_s)
    if not secs:
        return False

    bounds = {st: (starts[i + 1] if i + 1 < len(starts) else pay_end)
              for i, st in enumerate(starts)}
    sec_list = list(secs.values())
    min_len = min((bounds[st] - st) // 2 for st, _, _ in sec_list)

    def calibrate(value_idx):
        for o in range(1, min_len):
            if all(u16(st + o * 2) == vals[value_idx]
                   for st, *vals in [(s[0], s[1], s[2]) for s in sec_list]):
                return o
        return None
    voff = calibrate(0)        # vertex count
    ioff = calibrate(1)        # index count (= faces × 3)
    if voff is None or ioff is None:
        return False

    for (vb_i, sec_i), ov in overrides.items():
        s = secs.get((vb_i, sec_i))
        if not s:
            continue
        st = s[0]
        struct.pack_into('<H', data, st + voff * 2, min(0xFFFF, len(ov['verts'])))
        struct.pack_into('<H', data, st + ioff * 2, min(0xFFFF, len(ov['faces']) * 3))
    return True


def _build_lod_payload(data, walk, lod, lods, lod_idx, overrides, palettes,
                       bones, name2idx, abs_base):
    """Build one LOD's SDOL payload bytes.  `overrides` may be empty (every
    section then keeps its original vertex bytes — used for non-target LODs so
    they're re-laid-out with correct alignment at their new position)."""
    new_idx = []
    new_entries = []
    enc_vbs = []
    for vb_i, vb in enumerate(lod['vbs']):
        flag, stride, _vc, _o = walk['vb_meta'][vb_i]
        base = walk['vb_base'][vb_i]
        voff = walk['vb_voff'][vb_i]
        vbb = bytearray()
        for sec_i, (mat_i, idx_s, idx_e) in enumerate(vb.get('sections', [])):
            sec_start = len(vbb) // stride
            ov = overrides.get((vb_i, sec_i))
            idx0 = len(new_idx)
            if ov is not None:
                pal = _section_palette(lods, lod_idx, vb_i, sec_i, palettes)
                for (comp, infl) in ov['verts']:
                    si, sw = _encode_weights(infl, pal, name2idx)
                    vbb += _encode_vertex_full(flag, stride, comp, si, sw)
                for (a, b, c) in ov['faces']:
                    new_idx.extend((a + sec_start, b + sec_start, c + sec_start))
                last_v = sec_start + len(ov['verts']) - 1
            else:
                g = _section_global_verts(walk, vb_i, idx_s, idx_e)
                g2l = {gv: lv for lv, gv in enumerate(g)}
                for gv in g:                              # copy original bytes
                    foff = base + (gv - voff) * stride
                    vbb += bytes(data[foff:foff + stride])
                for k in range(idx_s, idx_e):
                    new_idx.append(g2l[walk['indices'][k] + voff] + sec_start)
                last_v = sec_start + len(g) - 1
            new_entries.append([vb_i, mat_i, 0, idx0, last_v,
                                sec_start * stride, 0])
        enc_vbs.append(bytes(vbb))

    def _pad16(buf):
        buf.extend(b'\x00' * ((-(abs_base + len(buf))) % 16))
    pay = bytearray()
    pay += struct.pack('<f', lod['lod_distance'])
    pay += struct.pack('<I', len(enc_vbs))
    vo = 0
    for ev, (flag, stride, _vc, _o) in zip(enc_vbs, walk['vb_meta']):
        nvc = len(ev) // stride
        pay += struct.pack('<4I', flag, stride, nvc, vo)
        vo += nvc
    pay += struct.pack('<I', len(new_entries))
    for e in new_entries:
        pay += struct.pack('<7I', *e)
    pay += struct.pack('<I', sum(len(v) for v in enc_vbs))
    _pad16(pay)
    for ev in enc_vbs:
        pay += ev
    pay += struct.pack('<I', len(new_idx))
    _pad16(pay)
    for ix in new_idx:
        pay += struct.pack('<H', max(0, min(0xFFFF, ix)))
    return bytes(pay)


def _rebuild_and_write(data, sdol_off, lod_idx, overrides, palettes, bones,
                       output_filepath):
    """Rebuild the SDOL, overriding edited sections in the target LOD.

    `overrides`: {(vb_i, sec_i): {'verts':[(comp, influences), ...], 'faces':[...]}}.
    EVERY LOD is rebuilt (non-target LODs keep their original vertex bytes but
    are re-laid-out so each LOD's internal 16-byte alignment is correct at its
    new position — copying them verbatim would misalign once the target LOD
    changes size).  SULC palettes are left untouched (same bones).
    """
    data = bytearray(data)                        # mutable: SULC patched in place
    name2idx = {b['name']: i for i, b in enumerate(bones)}
    lods = import_xbg_fc3.parse_xbg(bytes(data))['lods']
    n_lods = len(lods)

    full = bytearray()
    full += struct.pack('<I', n_lods)
    for li in range(n_lods):
        walk = _walk_lod(data, sdol_off, li)
        abs_base = sdol_off + 20 + len(full)
        ov = overrides if li == lod_idx else {}
        full += _build_lod_payload(data, walk, lods[li], lods, li, ov,
                                   palettes, bones, name2idx, abs_base)

    # Keep the SULC skinning metadata's per-section counts in sync with the new
    # geometry (self-validating — see _sync_sulc_counts).  DNKS/SULC sits before
    # SDOL, so patching `data` here lands in the data[:sdol_off] slice below.
    synced = _sync_sulc_counts(data, lods, lod_idx, overrides)

    chunk = struct.pack('<4sIIII', b'SDOL', 1, 20 + len(full), len(full), 0) + bytes(full)
    old_sz = struct.unpack_from('<I', data, sdol_off + 8)[0]
    new_file = bytes(data[:sdol_off]) + chunk + bytes(data[sdol_off + old_sz:])

    out_dir = os.path.dirname(output_filepath)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(output_filepath, 'wb') as f:
        f.write(new_file)
    return len(new_file)


# ── Main inject entry point ──────────────────────────────────────────────────

def inject_fc4(context, objects, output_filepath, lod_idx=0):
    """Inject FC3/FC4 section objects into the source XBG.

    Routes per source file: if every edited section keeps its original vertex
    count, patches in place (preserving weights byte-for-byte); if any count
    changed, rebuilds the LOD geometry and re-binds weights from vertex groups
    via the SULC palette.

    Returns (status_set, message_string).
    """
    vlog.log("\n%s\nFC3/FC4 INJECT  lod=%d\n%s"
             % ('=' * 60, lod_idx, '=' * 60))

    by_source = {}
    for obj in objects:
        raw = obj.get('xbg_fc3_data')
        if not raw:
            continue
        meta = raw.to_dict() if hasattr(raw, 'to_dict') else dict(raw)
        fp = meta.get('filepath', '')
        if not fp or not os.path.exists(fp):
            vlog.log("  SKIP %s: source missing (%s)" % (obj.name, fp))
            continue
        by_source.setdefault(fp, []).append((obj, meta))

    if not by_source:
        return ({'CANCELLED'}, "No objects with valid xbg_fc3_data found")

    total = 0
    warnings = []
    mode = "in-place"
    for source_fp, obj_list in by_source.items():
        vlog.log("\nSource: %s" % source_fp)
        with open(source_fp, 'rb') as f:
            data = bytearray(f.read())
        sdol_off = _sdol_offset(data)
        if sdol_off < 0:
            return ({'CANCELLED'}, "SDOL chunk not found in %s"
                    % os.path.basename(source_fp))
        try:
            walk = _walk_lod(data, sdol_off, lod_idx)
        except Exception as exc:
            return ({'CANCELLED'}, "SDOL walk failed: %s" % exc)

        # PMCP position post-multiplier: parse_xbg applies it to decoded
        # verts, so Blender coords = raw/16383.5 * pos_scale.  Divide it
        # back out before the *16383.5 i16 encode -- files with a
        # non-baseline PMCP (e.g. FC3's fanceiling_01, pos_scale 0.5)
        # otherwise re-encode every position at the wrong scale (the fan
        # came back HALF SIZE on a zero-edit round-trip).
        try:
            pos_scale = float(import_xbg_fc3.parse_xbg(bytes(data))
                              .get('pos_scale', 1.0)) or 1.0
        except Exception:
            pos_scale = 1.0

        # Extract every section, decide in-place vs rebuild.
        extracted = []      # (meta, n, comps, weights, faces, g_verts)
        need_rebuild = False
        for obj, meta in obj_list:
            if int(meta.get('lod', 0)) != lod_idx:
                vlog.log("  SKIP %s: lod mismatch" % obj.name)
                continue
            vb_i = int(meta.get('vb_index', 0))
            g_verts = _section_global_verts(
                walk, vb_i, int(meta.get('idx_start', 0)),
                int(meta.get('idx_end', 0)))
            try:
                n, comps, weights, faces = _extract_components(obj)
            except Exception as exc:
                warnings.append("%s: extract failed (%s)" % (obj.name, exc))
                continue
            if pos_scale != 1.0:
                inv_ps = 1.0 / pos_scale
                for c in comps:
                    if 'p' in c:
                        p = c['p']
                        c['p'] = (p[0] * inv_ps, p[1] * inv_ps, p[2] * inv_ps)
            extracted.append((obj, meta, vb_i, n, comps, weights, faces, g_verts))
            if n != len(g_verts):
                need_rebuild = True

        if not extracted:
            continue

        if not need_rebuild:
            # ── In-place patch ──
            for obj, meta, vb_i, n, comps, weights, faces, g_verts in extracted:
                flag, stride, _vc, _o = walk['vb_meta'][vb_i]
                base = walk['vb_base'][vb_i]
                voff = walk['vb_voff'][vb_i]
                for lv, gv in enumerate(g_verts):
                    foff = base + (gv - voff) * stride
                    data[foff:foff + stride] = _patch_vertex(
                        bytes(data[foff:foff + stride]), flag, comps[lv])
                total += len(g_verts)
                vlog.log("  Patched VB%d '%s' %d verts" % (vb_i, obj.name, n))
            out_dir = os.path.dirname(output_filepath)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
            with open(output_filepath, 'wb') as f:
                f.write(bytes(data))
        else:
            # ── Rebuild (count changed): re-bind weights via SULC palette ──
            mode = "rebuild"
            m = import_xbg_fc3.parse_xbg(bytes(data))
            bones = m['bones']
            sk = m.get('skinning')
            palettes = [s['bone_assignments'] for s in (sk['sections'] if sk else [])]
            overrides = {}
            for obj, meta, vb_i, n, comps, weights, faces, g_verts in extracted:
                sec_i = int(meta.get('section_index', 0))
                verts = [(comps[i], weights[i]) for i in range(n)]
                overrides[(vb_i, sec_i)] = {'verts': verts, 'faces': faces}
                total += n
                vlog.log("  Rebuild VB%d/sec%d '%s'  %d→%d verts"
                         % (vb_i, sec_i, obj.name, len(g_verts), n))
            try:
                nbytes = _rebuild_and_write(data, sdol_off, lod_idx, overrides,
                                            palettes, bones, output_filepath)
                vlog.log("  Wrote %d B → %s" % (nbytes, output_filepath))
            except Exception as exc:
                import traceback
                vlog.log(traceback.format_exc())
                return ({'CANCELLED'}, "Rebuild failed: %s" % exc)

    if total == 0:
        return ({'CANCELLED'},
                "Nothing injected" + (" — " + warnings[0] if warnings else ""))
    msg = "FC3/FC4 inject [%s]: %d verts → %s" % (
        mode, total, os.path.basename(output_filepath))
    if warnings:
        msg += "  (%d note(s) — see console)" % len(warnings)
    return ({'FINISHED'}, msg)
