"""inject_xbg_fc5.py — Far Cry 5 XBG mesh injector (in-place vertex patch).

Re-injects Blender objects tagged with 'xbg_fc3_data' (the FC5 pipeline
stamps the same metadata dict as FC3/FC4) back into the source FC5 XBG.

Same design as modules/Far_Cry_3/inject_xbg_fc3.py's in-place path —
PRESERVE EVERYTHING, patch only what the user can edit: position, normal,
tangent, binormal, UV0, UV1, vertex colour.  Bone weights/indices, pos.w,
the R10G10B10A2 2-bit A (handedness) fields and any unknown bytes survive
byte-for-byte.

FC5-specific encoding differences vs FC3/FC4 (mirrors import_xbg_fc5):
  * normal / tangent / binormal are R10G10B10A2_UNORM packed in ONE u32
    (three 10-bit unsigned components mapped to [-1,1] + 2-bit A) — NOT the
    FC3 D3DCOLOR byte triple.  The A bits are kept from the original u32.
  * tangent may live in the UNK_400 (0x0400) slot instead of TANGENT
    (0x0100) — flag 0x0c7a does this; accept either, like the reader.
  * the SDOL LOD layout differs (extra u32 + variable entry/palette block);
    the vertex block's absolute offset is taken from parse_xbg's 'vstart'
    (anchored on the trailing index buffer) rather than a 16-align walk.

Same vertex COUNT and ORDER only (reshape / sculpt / repaint).  A changed
count aborts with a clear message — the FC3-style full rebuild needs the
FC5 SULC section layout decoded first, which hasn't been done.
"""

import os
import struct

import bpy

from ..Core.debug import VerboseLogger
from . import import_xbg_fc5

vlog = VerboseLogger


# ── FC5 vertex component encoding ────────────────────────────────────────────

def _i16(v):
    return max(-32768, min(32767, int(round(v * 16383.5))))


def _u8(v):
    return max(0, min(255, int(round(v))))


def _pack_1010102(vec, keep_a_bits):
    """Inverse of import_xbg_fc5._unpack_1010102: [-1,1] xyz -> u32 with
    three 10-bit UNSIGNED components; the 2-bit A field is preserved from
    the original value (handedness — Blender has no edit for it)."""
    def enc(c):
        return max(0, min(1023, int(round((c + 1.0) * 0.5 * 1023.0))))
    return (enc(vec[0]) | (enc(vec[1]) << 10) | (enc(vec[2]) << 20)
            | ((keep_a_bits & 0x3) << 30))


def _patch_vertex_fc5(orig, flag, comp, pos_scale=1.0):
    """Return `orig` (stride bytes) with the geometry components in `comp`
    overwritten.  `comp` keys (any subset): 'p','uv','uv1','n','color',
    't','b'.  Everything else is preserved verbatim.

    `pos_scale` is the PMCP post-multiplier parse_xbg applied to decoded
    positions (1.0 for the baseline 1/16384 files) — Blender coordinates are
    raw/16383.5 * pos_scale, so encode divides it back out first."""
    o = import_xbg_fc5._vertex_offsets(flag)
    buf = bytearray(orig)
    inv_ps = 1.0 / pos_scale if pos_scale else 1.0

    if 'p' in comp and 0x0002 in o:                       # i16 xyz (keep w)
        p = o[0x0002]
        struct.pack_into('<3h', buf, p,
                         _i16(comp['p'][0] * inv_ps),
                         _i16(comp['p'][1] * inv_ps),
                         _i16(comp['p'][2] * inv_ps))
    elif 'p' in comp and 0x0001 in o:                     # float xyz
        struct.pack_into('<3f', buf, o[0x0001],
                         comp['p'][0] * inv_ps, comp['p'][1] * inv_ps,
                         comp['p'][2] * inv_ps)

    # UVs: decode was u = ru/16383.5 + 1.0 ; v = 2.0 - rv/16383.5
    if 'uv' in comp and 0x0008 in o:
        struct.pack_into('<2h', buf, o[0x0008],
                         _i16(comp['uv'][0] - 1.0), _i16(2.0 - comp['uv'][1]))
    if 'uv1' in comp and 0x0800 in o:
        p = o[0x0800]
        if comp['uv1'] is None:
            struct.pack_into('<2h', buf, p, -32768, -32768)
        else:
            struct.pack_into('<2h', buf, p,
                             _i16(comp['uv1'][0] - 1.0),
                             _i16(2.0 - comp['uv1'][1]))

    # Normal / tangent / binormal: R10G10B10A2 (keep the 2-bit A field).
    def _patch_1010102(slot_off, vec):
        old = struct.unpack_from('<I', buf, slot_off)[0]
        struct.pack_into('<I', buf, slot_off,
                         _pack_1010102(vec, (old >> 30) & 0x3))

    if 'n' in comp and 0x0040 in o:
        _patch_1010102(o[0x0040], comp['n'])
    if 't' in comp:
        for _tflag in (0x0100, 0x0400):                   # same probe as reader
            if _tflag in o:
                _patch_1010102(o[_tflag], comp['t'])
                break
    if 'b' in comp and 0x0200 in o:
        _patch_1010102(o[0x0200], comp['b'])

    # Colour: D3DCOLOR — R,G,B,A -> B,G,R,A bytes (same as FC3).
    if 'color' in comp and 0x0080 in o:
        p = o[0x0080]
        c = comp['color']
        struct.pack_into('<4B', buf, p,
                         _u8(c[2]), _u8(c[1]), _u8(c[0]), _u8(c[3]))
    return bytes(buf)


# ── Section -> file-vertex mapping ───────────────────────────────────────────

def _section_global_verts(lod, idx_s, idx_e):
    """Reproduce the importer's g_verts (sorted unique global vertex indices)
    for the section [idx_s, idx_e) — must match blender_pipeline_fc5's
    construction exactly so Blender vertex lv maps to g_verts[lv]."""
    vbs = lod['vbs']
    sec_list = vbs[0].get('sections', [])
    if not sec_list:
        return []
    vb_idx_base = min(s[1] for s in sec_list)
    face_start = (idx_s - vb_idx_base) // 3
    n_faces = (idx_e - idx_s) // 3
    face_slice = vbs[0]['faces'][face_start: face_start + n_faces]
    vert_set = set()
    for fa, fb, fc_v in face_slice:
        vert_set.update((fa, fb, fc_v))
    return sorted(vert_set)


def _vb_layout(lod):
    """[(abs_byte_base, global_vert_start, stride, flag, vcount), ...]"""
    out = []
    base = lod['vstart']
    gstart = 0
    for vb in lod['vbs']:
        out.append((base, gstart, vb['stride'], vb['flag'], vb['vcount']))
        base += vb['stride'] * vb['vcount']
        gstart += vb['vcount']
    return out


def _locate_vert(layout, gv):
    """(abs_byte_offset, stride, flag) of global vertex index gv."""
    for base, gstart, stride, flag, vcount in layout:
        if gstart <= gv < gstart + vcount:
            return base + (gv - gstart) * stride, stride, flag
    return None, None, None


# ── Blender component extraction (per-vertex collapse) ───────────────────────

def _extract_components(obj):
    """Per-vertex geometry components from a Blender object — same collapse
    rules as the FC3 injector (per-loop data -> per-vertex, first loop wins,
    matching how the importer expanded per-vertex data to loops)."""
    depsgraph = bpy.context.evaluated_depsgraph_get()
    eval_obj = obj.evaluated_get(depsgraph)
    me = eval_obj.to_mesh()

    n = len(me.vertices)
    comps = [dict() for _ in range(n)]
    for i, v in enumerate(me.vertices):
        comps[i]['p'] = (v.co.x, v.co.y, v.co.z)

    # Prefer the RAW authored normals stored by the importer in the
    # xbg_normal POINT attribute (Blender's corner_normals are NORMALIZED,
    # which re-quantizes every 10-bit normal code on round-trip); verts
    # without an attribute entry fall back to corner normals.
    na = me.attributes.get('xbg_normal')
    if na and na.domain == 'POINT' and len(na.data) == n:
        for vi in range(n):
            v = na.data[vi].vector
            if v[0] * v[0] + v[1] * v[1] + v[2] * v[2] > 1e-6:
                comps[vi]['n'] = (v[0], v[1], v[2])
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
    if col is None and getattr(me, 'vertex_colors', None):
        col = me.vertex_colors.get('Col') or me.vertex_colors[0]
    if col is not None:
        per_point = getattr(col, 'domain', 'CORNER') == 'POINT'
        if per_point:
            for vi in range(min(n, len(col.data))):
                c = col.data[vi].color
                comps[vi]['color'] = (c[0]*255, c[1]*255, c[2]*255, c[3]*255)
        else:
            done = [False] * n
            for poly in me.polygons:
                for li in poly.loop_indices:
                    vi = me.loops[li].vertex_index
                    if not done[vi]:
                        c = col.data[li].color
                        comps[vi]['color'] = (c[0]*255, c[1]*255,
                                              c[2]*255, c[3]*255)
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

    eval_obj.to_mesh_clear()
    return n, comps


# ── Main entry point ─────────────────────────────────────────────────────────

def inject_fc5(context, objects, output_filepath, lod_idx=0):
    """Patch the selected FC5 section objects' edited vertices back into a
    copy of the source .xbg (in place, same vertex count/order only).

    Returns (status_set, message_string).
    """
    vlog.log("\n%s\nFC5 INJECT  lod=%d\n%s" % ('=' * 60, lod_idx, '=' * 60))

    metas = []
    for obj in objects:
        meta = obj.get('xbg_fc3_data')
        d = meta.to_dict() if hasattr(meta, 'to_dict') else dict(meta or {})
        metas.append((obj, d))

    srcs = {m.get('filepath') for _, m in metas}
    if len(srcs) != 1 or not next(iter(srcs)):
        return {'CANCELLED'}, ("Selected sections come from different "
                               "source files (or none) — select sections "
                               "from ONE imported FC5 model")
    src = next(iter(srcs))
    if not os.path.isfile(src):
        return {'CANCELLED'}, f"Source XBG not found: {src}"

    data = bytearray(open(src, 'rb').read())
    parsed = import_xbg_fc5.parse_xbg(bytes(data))
    lods = parsed['lods']
    pos_scale = float(parsed.get('pos_scale', 1.0)) or 1.0
    if lod_idx >= len(lods):
        return {'CANCELLED'}, (f"LOD {lod_idx} not present "
                               f"({len(lods)} LOD(s) decoded)")
    lod = lods[lod_idx]
    layout = _vb_layout(lod)

    patched_objs = 0
    patched_verts = 0
    skipped = []
    for obj, meta in metas:
        if int(meta.get('lod', 0)) != lod_idx:
            skipped.append((obj.name, "different LOD"))
            continue
        idx_s = int(meta.get('idx_start', 0))
        idx_e = int(meta.get('idx_end', 0))
        g_verts = _section_global_verts(lod, idx_s, idx_e)
        if not g_verts:
            skipped.append((obj.name, "section not found in SDOL"))
            continue

        n, comps = _extract_components(obj)
        if n != len(g_verts):
            skipped.append((obj.name,
                            f"vertex count changed ({n} vs {len(g_verts)}) — "
                            "FC5 inject is in-place only"))
            continue

        for lv, gv in enumerate(g_verts):
            voff, stride, flag = _locate_vert(layout, gv)
            if voff is None:
                continue
            orig = bytes(data[voff:voff + stride])
            data[voff:voff + stride] = _patch_vertex_fc5(
                orig, flag, comps[lv], pos_scale)
            patched_verts += 1
        patched_objs += 1
        vlog.log(f"  patched '{obj.name}': {n} verts")

    if not patched_objs:
        why = "; ".join(f"{nm}: {r}" for nm, r in skipped) or "nothing selected"
        return {'CANCELLED'}, f"No sections injected — {why}"

    with open(output_filepath, 'wb') as f:
        f.write(bytes(data))

    msg = (f"FC5 inject: {patched_objs} section(s), {patched_verts} verts "
           f"patched -> {os.path.basename(output_filepath)}")
    if skipped:
        msg += "  [skipped: " + "; ".join(f"{nm} ({r})" for nm, r in skipped) + "]"
    return {'FINISHED'}, msg
