"""Watch Dogs 1 .xbg geometry injection (in-place vertex editing).

The WD1 xbg is a sequential IBinaryArchive stream (see import_wd.py), so —
unlike Avatar's chunk splicing — geometry edits are written by patching the
vertex bytes IN PLACE inside the shared SGfxBuffers vertex block.  This
keeps the file structure byte-identical everywhere except the vertices the
user actually moved.

PHASE 1 (this module): edit vertex POSITIONS / NORMALS / UVs / COLORS
without changing the vertex or triangle COUNT.  That covers reshaping,
sculpting and re-skinning existing geometry — enough to test custom meshes
in-game before the count-changing rebuild (phase 2: add/remove geometry,
which must repack the shared buffer + drawcall ranges).

The vertex codec is the exact inverse of import_wd._decode_wd1_mesh:
position i16 round-trips bit-exactly (verified on char01), so an unedited
re-export reproduces the original file byte-for-byte.

Per-mesh layout needed for export is stamped onto each imported object by
build_wd_model (keys below); injection reads it back:
    wd_src        source .xbg path
    wd_vb_off     absolute file offset of this mesh's first vertex
    wd_stride     vertex stride in bytes
    wd_format     FVF flag word
    wd_scale      (pos_off, pos_scale, uv_off, uv_scale)  decompression consts
    wd_vcount     vertex count
"""

import os
import struct

try:
    import bpy
    import mathutils
except Exception:
    bpy = None
    mathutils = None


def component_layout(fmt, stride=None):
    """Byte sub-offset + type of each editable component within one vertex.

    Mirrors _decode_wd1_mesh's component walk exactly.  Returns a dict; when
    `stride` is given, asserts the computed stride matches (guards against an
    unknown flag combination silently corrupting the patch)."""
    lay = {}
    k = 0
    if fmt & 0x1:
        lay['pos'] = (k, 'f32'); k += 12
    elif fmt & 0x2:
        lay['pos'] = (k, 'i16'); k += 8          # x,y,z,w(i16) — w kept as-is
    if fmt & 0x4:
        lay['uv'] = (k, 'f32'); k += 8
    elif fmt & 0x8:
        lay['uv'] = (k, 'i16'); k += 4
    if fmt & 0x1000:
        lay['uv2'] = (k, 'i16'); k += 4
    if fmt & 0x2000:
        k += 4                                    # uv_comp3 (passthrough)
    if fmt & 0x10:
        lay['skin'] = (k, 'skin'); k += 8         # not edited in phase 1
    if fmt & 0x20:
        k += 4
    if fmt & 0x80:
        lay['normal'] = (k, 'u8'); k += 4
    elif fmt & 0x4000:
        lay['normal'] = (k, 'f32'); k += 12
    if fmt & 0x100:
        lay['color'] = (k, 'u8c'); k += 4
    if fmt & 0x200:
        lay['tangent'] = (k, 'u8'); k += 4
    if fmt & 0x400:
        lay['binormal'] = (k, 'u8'); k += 4
    if fmt & 0x8000:
        k += 4
    if stride is not None and k != stride:
        raise ValueError("WD1 format 0x%04X: computed stride %d != %d"
                         % (fmt, k, stride))
    return lay


def _clamp_i16(v):
    return max(-32768, min(32767, int(round(v))))


def _enc_u8n(n):
    """Inverse of import_wd._u8n: (x-1)/127 - 1  ->  x."""
    return max(0, min(255, int(round((n + 1.0) * 127.0)) + 1))


def patch_vertex(buf, base, lay, off, *, co=None, normal=None,
                 uv=None, uv2=None, color=None, tangent=None, tangent_w=None,
                 binormal=None, binormal_w=None):
    """Overwrite the editable components of one vertex at file offset `base`.

    Only the kwargs supplied are written; everything else (skin weights,
    padding) is left untouched, so unedited data survives exactly.
    `off` = (pos_off, pos_scale, uv_off, uv_scale)."""
    if co is not None and 'pos' in lay:
        k, t = lay['pos']
        if t == 'f32':
            struct.pack_into('<3f', buf, base + k, co[0], co[1], co[2])
        else:                                     # i16 quantized; keep w
            for i in range(3):
                struct.pack_into('<h', buf, base + k + i * 2,
                                 _clamp_i16((co[i] - off[0]) / off[1]))
    if uv is not None and 'uv' in lay:
        k, t = lay['uv']
        u, v = uv[0], 1.0 - uv[1]                 # decode flips V
        if t == 'f32':
            struct.pack_into('<2f', buf, base + k, u, v)
        else:
            struct.pack_into('<h', buf, base + k,
                             _clamp_i16((u - off[2]) / off[3]))
            struct.pack_into('<h', buf, base + k + 2,
                             _clamp_i16((v - off[2]) / off[3]))
    if uv2 is not None and 'uv2' in lay:
        k, _ = lay['uv2']
        u, v = uv2[0], 1.0 - uv2[1]
        struct.pack_into('<h', buf, base + k, _clamp_i16((u - off[2]) / off[3]))
        struct.pack_into('<h', buf, base + k + 2,
                         _clamp_i16((v - off[2]) / off[3]))
    # Normal/tangent/binormal/color are D3DCOLOR: stored B,G,R,A, so xyz go to
    # bytes 2,1,0 (and colour R,G,B -> bytes 2,1,0).  Must match import_wd's
    # BGRA decode or the round-trip scrambles axes.
    if normal is not None and 'normal' in lay:
        k, t = lay['normal']
        if t == 'f32':
            struct.pack_into('<3f', buf, base + k, *normal)
        else:
            for i in range(3):
                buf[base + k + i] = _enc_u8n(normal[2 - i])
    if color is not None and 'color' in lay:
        k, _ = lay['color']
        for i in range(3):                                  # R,G,B -> bytes 2,1,0
            buf[base + k + i] = max(0, min(255, int(round(color[2 - i] * 255.0))))
        buf[base + k + 3] = max(0, min(255, int(round(color[3] * 255.0))))
    if tangent is not None and 'tangent' in lay:
        k, _ = lay['tangent']
        for i in range(3):
            buf[base + k + i] = _enc_u8n(tangent[2 - i])
        if tangent_w is not None:
            buf[base + k + 3] = _enc_u8n(tangent_w)
    if binormal is not None and 'binormal' in lay:
        k, _ = lay['binormal']
        for i in range(3):
            buf[base + k + i] = _enc_u8n(binormal[2 - i])
        if binormal_w is not None:
            buf[base + k + 3] = _enc_u8n(binormal_w)


# ---------------------------------------------------------------------------
# Blender export
# ---------------------------------------------------------------------------

def _edited_extent(objs):
    """Max |coordinate| over all selected meshes' vertices (object space —
    the same values patch_vertex encodes)."""
    m = 0.0
    for ob in objs:
        for v in ob.data.vertices:
            m = max(m, abs(v.co.x), abs(v.co.y), abs(v.co.z))
    return m


def _maybe_expand_bounds(main_buf, mip_buf, layout, edited_extent,
                         margin=32000.0):
    """Avatar-style bounds/scale expansion for WD1.

    Positions are quantized as i16 with a GLOBAL (pos_off, pos_scale) from
    SceneGeometryParams; geometry bigger than 32767*pos_scale used to be
    hard-clamped to the original bounding box at inject.  When
    `edited_extent` (max |coord| of the edited meshes) exceeds the current
    capacity, this rewrites pos_scale so the new geometry fits, re-encodes
    EVERY i16 position in EVERY buffer (the scale is file-global — unedited
    meshes must be re-quantized too or they'd change size), and updates the
    header bounding box + sphere.  Position w (bone-palette slot on rigid
    meshes!) is never touched.

    Mutates main_buf (and mip_buf when given) in place.  Returns
    (new_off_tuple, expanded_bool)."""
    off = tuple(layout['scale'])
    pos_off, pos_scale = off[0], off[1]
    capacity = 32767.0 * pos_scale
    needed = max(abs(edited_extent - pos_off), abs(-edited_extent - pos_off))
    if needed <= capacity:
        return off, False

    new_scale = needed / margin           # |raw| tops out ~margin < 32767
    ho = layout['header_offs']
    tables = layout['buffer_mesh_tables']
    buf_offsets = layout['buf_offsets']
    mip_present = bool(layout.get('mip_path'))

    lo = [1e30, 1e30, 1e30]
    hi = [-1e30, -1e30, -1e30]
    seen_pools = set()
    for b, table in enumerate(tables):
        in_mip = mip_present and b == 0
        tgt = mip_buf if in_mip else main_buf
        if tgt is None:
            continue                       # mip objects not loaded — nothing to do
        vd = buf_offsets[b] if b < len(buf_offsets) else 0
        for vb_off, stride, fmt, vcount in table:
            if not (fmt & 0x2):            # f32 positions don't quantize
                continue
            key = (b, vb_off)
            if key in seen_pools:          # shared pools: re-encode ONCE
                continue
            seen_pools.add(key)
            base = vd + vb_off
            for vi in range(vcount):
                p = base + vi * stride
                x, y, z = struct.unpack_from('<3h', tgt, p)
                wx = x * pos_scale + pos_off
                wy = y * pos_scale + pos_off
                wz = z * pos_scale + pos_off
                struct.pack_into('<3h', tgt, p,
                                 _clamp_i16((wx - pos_off) / new_scale),
                                 _clamp_i16((wy - pos_off) / new_scale),
                                 _clamp_i16((wz - pos_off) / new_scale))
                for i, w in enumerate((wx, wy, wz)):
                    if w < lo[i]:
                        lo[i] = w
                    if w > hi[i]:
                        hi[i] = w

    # bounds must also cover the EDITED geometry (written after this)
    for i in range(3):
        lo[i] = min(lo[i], -edited_extent)
        hi[i] = max(hi[i], edited_extent)

    # header: new scale + bbox + bounding sphere (center + radius)
    struct.pack_into('<f', main_buf, ho['gp_scale'], new_scale)
    struct.pack_into('<6f', main_buf, ho['bbox'],
                     lo[0], lo[1], lo[2], hi[0], hi[1], hi[2])
    cx = (lo[0] + hi[0]) * 0.5
    cy = (lo[1] + hi[1]) * 0.5
    cz = (lo[2] + hi[2]) * 0.5
    rad = ((hi[0] - cx) ** 2 + (hi[1] - cy) ** 2 + (hi[2] - cz) ** 2) ** 0.5
    struct.pack_into('<4f', main_buf, ho['bsphere'], cx, cy, cz, rad)

    return (pos_off, new_scale, off[2], off[3]), True


def inject_wd1_objects(objects, out_path, source_path=None,
                       recalculate_normals=False):
    """Patch the edited vertices of each WD1-imported object back into a copy
    of the source .xbg.  Returns (n_objects, n_vertices, warnings).

    `recalculate_normals`: when True, use Blender's computed geometry normals
    instead of the stored xbg_normal attribute.  Use after sculpting so the
    new normals match the moved geometry rather than the original import."""
    tagged = [o for o in objects if o.get('wd_src')]
    if not tagged:
        raise RuntimeError("no WD1-imported meshes selected "
                           "(import a Watch Dogs .xbg first)")
    src = source_path or tagged[0]['wd_src']
    tagged = [o for o in tagged if o['wd_src'] == src]
    buf = bytearray(open(src, 'rb').read())
    warnings = []
    n_obj = n_vtx = 0

    # Streamed-LOD0 objects (wd_mip_src): their bytes live in the companion
    # .xbgmip, not the .xbg.  Patch a copy of that file too, written next to
    # the output as "<out_base>.high.xbgmip" — without this, the game streams
    # the pristine hi-res LOD over the edit at close range (the "my edit
    # reverted when I walked up to it" bug).
    mip_srcs = {o['wd_mip_src'] for o in tagged if o.get('wd_mip_src')}
    mip_buf = None
    mip_out = None
    if mip_srcs:
        if len(mip_srcs) > 1:
            raise RuntimeError("selected meshes reference different .xbgmip files")
        mip_src = next(iter(mip_srcs))
        if not os.path.isfile(mip_src):
            raise RuntimeError("companion .xbgmip not found: %s" % mip_src)
        mip_buf = bytearray(open(mip_src, 'rb').read())
        mip_out = os.path.splitext(out_path)[0] + ".high.xbgmip"

    # ── bounds/scale expansion ───────────────────────────────────────────
    # Edited geometry bigger than the original quantization range used to
    # be hard-clamped to the stock bounding box.  Detect and expand first
    # (rewrites the header scale/bounds + re-encodes every stock position).
    new_off = None
    off0 = tuple(tagged[0]['wd_scale'])
    ext = _edited_extent(tagged)
    if ext + abs(off0[0]) > 32767.0 * off0[1]:
        from .import_wd import parse_wd1_xbg
        layout = parse_wd1_xbg(src)['_layout']
        new_off, expanded = _maybe_expand_bounds(buf, mip_buf, layout, ext)
        if expanded:
            warnings.append(
                "geometry exceeds the original bounds — position scale "
                "expanded %.6g -> %.6g (all stock vertices re-encoded, "
                "header bbox/sphere updated); slight precision loss on "
                "unedited meshes is expected" % (off0[1], new_off[1]))

    for ob in tagged:
        me = ob.data
        vb_off = int(ob['wd_vb_off'])
        buf0_off = int(ob.get('wd_buf0_off', 0))
        stride = int(ob['wd_stride'])
        fmt = int(ob['wd_format'])
        vcount = int(ob['wd_vcount'])
        off = new_off if new_off is not None else tuple(ob['wd_scale'])
        if len(me.vertices) != vcount:
            warnings.append(
                "%s: vertex count changed (%d -> %d) — phase-1 inject keeps "
                "the count; skipped (adding/removing geometry needs the "
                "buffer rebuild, not yet available)"
                % (ob.name, vcount, len(me.vertices)))
            continue
        try:
            lay = component_layout(fmt, stride)
        except ValueError as e:
            warnings.append("%s: %s — skipped" % (ob.name, e))
            continue

        # gather per-vertex UV / normal / colour from Blender (loop data
        # averaged to the vertex, matching the per-vertex source layout)
        uvs = _vertex_uvs(me, 0)
        uv2s = _vertex_uvs(me, 1)
        cols = _vertex_colors(me)
        norms = _vertex_normals(me, recalculate=recalculate_normals)
        # tangent + binormal: only recompute when recalculate_normals is on;
        # otherwise leave the original bytes untouched (round-trip fidelity).
        tans, tsigns = (_vertex_tangents(me)
                        if recalculate_normals and 'tangent' in lay
                        else (None, None))

        # the decoded LOD lives wholly in ONE buffer; each vertex is a flat,
        # contiguous slice at buf0_off + vb_off + vi*stride (no straddling).
        # Streamed-LOD0 meshes patch the .xbgmip copy instead of the .xbg.
        tgt = mip_buf if (ob.get('wd_mip_src') and mip_buf is not None) else buf
        base = buf0_off + vb_off
        for vi, v in enumerate(me.vertices):
            foff = base + vi * stride
            vbytes = bytearray(tgt[foff:foff + stride])
            tan = bn = tw = bw = None
            if tans and norms:
                tan = tans[vi]
                tw = tsigns[vi]
                n = mathutils.Vector(norms[vi])
                t = mathutils.Vector(tan)
                bn = tuple((n.cross(t)).normalized() * tw)
                bw = 1.0
            patch_vertex(vbytes, 0, lay, off,
                         co=(v.co.x, v.co.y, v.co.z),
                         normal=norms[vi] if norms else None,
                         uv=uvs[vi] if uvs else None,
                         uv2=uv2s[vi] if uv2s else None,
                         color=cols[vi] if cols else None,
                         tangent=tan, tangent_w=tw,
                         binormal=bn, binormal_w=bw)
            tgt[foff:foff + stride] = vbytes
            n_vtx += 1
        n_obj += 1

    with open(out_path, 'wb') as f:
        f.write(buf)
    if mip_buf is not None and mip_out:
        with open(mip_out, 'wb') as f:
            f.write(mip_buf)
        warnings.append(
            "streamed LOD0 written to %s — keep it NEXT TO the injected .xbg "
            "(and name both like the originals) so the game streams your "
            "edit at close range too" % os.path.basename(mip_out))
    return n_obj, n_vtx, warnings


def _vertex_uvs(me, layer_index):
    if layer_index >= len(me.uv_layers):
        return None
    uvl = me.uv_layers[layer_index].data
    out = [None] * len(me.vertices)
    for loop in me.loops:
        if out[loop.vertex_index] is None:
            uv = uvl[loop.index].uv
            out[loop.vertex_index] = (uv[0], uv[1])
    return out  # None = vertex not referenced by any face; patch_vertex skips it


def _vertex_colors(me):
    if not me.color_attributes:
        return None
    ca = me.color_attributes[0]
    if ca.domain == 'POINT':
        return [tuple(ca.data[i].color) for i in range(len(me.vertices))]
    out = [None] * len(me.vertices)
    for loop in me.loops:
        if out[loop.vertex_index] is None:
            out[loop.vertex_index] = tuple(ca.data[loop.index].color)
    return [c or (1.0, 1.0, 1.0, 1.0) for c in out]


def _vertex_normals(me, recalculate=False):
    # When recalculate=True: use Blender's computed smooth normal so that
    # sculpted/moved geometry gets a matching normal in the file.
    # Default: prefer the round-trip xbg_normal attribute (authored vectors);
    # fall back to the geometry normal for verts where xbg_normal is zero.
    geo = [tuple(v.normal) for v in me.vertices]
    if recalculate:
        return geo
    na = me.attributes.get('xbg_normal')
    if na and na.domain == 'POINT' and len(na.data) == len(me.vertices):
        out = []
        for i in range(len(me.vertices)):
            v = na.data[i].vector
            out.append((v[0], v[1], v[2])
                       if (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]) > 1e-6
                       else geo[i])
        return out
    return geo


def sync_normals_to_geometry(obj):
    """Bake Blender's current smooth normals into the xbg_normal attribute so
    the next injection writes normals that match the sculpted/edited shape.
    Call this in Object Mode before injecting when you've moved vertices."""
    if bpy is None or obj.type != 'MESH':
        return 0
    me = obj.data
    geo = [tuple(v.normal) for v in me.vertices]
    na = me.attributes.get('xbg_normal')
    if na is None or na.domain != 'POINT' or len(na.data) != len(me.vertices):
        try:
            if na:
                me.attributes.remove(na)
            na = me.attributes.new('xbg_normal', 'FLOAT_VECTOR', 'POINT')
        except Exception:
            return 0
    na.data.foreach_set('vector', [c for n in geo for c in n])
    return len(geo)


# ---------------------------------------------------------------------------
# Phase 2 — count-changing rebuild (add / remove geometry)
# ---------------------------------------------------------------------------
# The whole SGfxBuffers vertex+index buffer is rebuilt, then spliced back in
# place of the original (head + new buffers + tail), and every LOD0 drawcall
# field is patched.  Each vertex POOL (group of meshes sharing a vb_offset)
# starts from the original pool bytes — so unchanged meshes round-trip
# byte-for-byte — with moved vertices re-encoded and appended vertices
# encoded fresh.  Index buffers are rebuilt from the meshes' triangles.

def _mesh_triangles(me):
    """[(v0, v1, v2)] in the FILE's winding (import swapped 1<->2, undo it)."""
    me.calc_loop_triangles()
    tris = []
    for lt in me.loop_triangles:
        a, b, c = lt.vertices
        tris.append((a, c, b))           # reverse the import winding swap
    return tris


def _vertex_tangents(me):
    """Per-vertex (tangent, sign) computed from the UV map, Avatar-style —
    so newly added geometry gets a valid tangent frame instead of zeros.
    Returns (tangents, signs) or (None, None) if no UV map."""
    if mathutils is None or not me.uv_layers:
        return None, None
    try:
        me.calc_tangents(uvmap=me.uv_layers[0].name)
    except Exception:
        return None, None
    acc = [mathutils.Vector((0.0, 0.0, 0.0)) for _ in me.vertices]
    sign = [0.0] * len(me.vertices)
    cnt = [0] * len(me.vertices)
    for loop in me.loops:
        vi = loop.vertex_index
        acc[vi] += loop.tangent
        sign[vi] += loop.bitangent_sign
        cnt[vi] += 1
    out_t, out_s = [], []
    for vi in range(len(me.vertices)):
        t = acc[vi]
        out_t.append(tuple(t.normalized()) if t.length > 1e-9 else (1.0, 0.0, 0.0))
        out_s.append(1.0 if (sign[vi] / max(1, cnt[vi])) >= 0 else -1.0)
    return out_t, out_s


def _neutral_color(me, cols):
    """A sensible default colour for NEW vertices that have none, matching
    Avatar's 'don't leave unpainted geometry black' rule.  Use the mesh's
    most common authored colour so new geometry shades like its neighbours;
    fall back to white (no-darken) when nothing is authored."""
    if cols:
        from collections import Counter
        c = Counter(tuple(round(x, 3) for x in v) for v in cols[:2000])
        return c.most_common(1)[0][0]
    return (1.0, 1.0, 1.0, 1.0)


def _encode_skin(buf, base, lay, vgroups, name2pal):
    """Write 4 weight bytes + 4 palette-index bytes for one vertex's groups.

    WD1 skinning is 4-BONE (the body meshes use up to 4 influences) and the
    weights are NOT renormalised to 255 — the shipped data sums to whatever
    the artist painted (e.g. 252/255) and the engine normalises at runtime.
    So each weight is written as round(w * 255) directly: imported weights
    (w = byte/255) round-trip byte-exact.  Drop zero weights, sort
    heaviest-first, keep the top 4.  If the painted weights exceed 255 in
    total (un-normalised input) they are scaled down to fit."""
    if 'skin' not in lay:
        return
    k, _ = lay['skin']
    infl = sorted(((g.weight, g.group) for g in vgroups if g.weight > 0.001),
                  key=lambda t: -t[0])[:4]
    raw = [w * 255.0 for w, _ in infl]
    s = sum(raw)
    if s > 255.0:                       # only rescale if it would overflow
        raw = [r * 255.0 / s for r in raw]
    b = [0, 0, 0, 0]
    ix = [0, 0, 0, 0]
    for i, (w, gi) in enumerate(infl):
        b[i] = max(0, min(255, int(round(raw[i]))))
        ix[i] = max(0, min(255, name2pal.get(gi, 0)))
    for i in range(4):
        buf[base + k + i] = b[i]
        buf[base + k + 4 + i] = ix[i]


def rebuild_wd1_objects(objects, out_path, source_path=None, reskin=False,
                        drop_unselected=False, recalculate_normals=False):
    """Rebuild a WD1 .xbg's geometry buffers from edited meshes, supporting
    vertex/triangle COUNT changes.  When `reskin` is True, every vertex's
    bone weights are re-derived from its vertex groups (Avatar-style).  When
    `drop_unselected` is True, meshes of the source file that are NOT in
    `objects` are REMOVED from the output (their geometry is dropped and the
    drawcall emptied) — `objects` becomes a keep-list, so injecting a subset
    bakes a smaller file (e.g. a head without the eyes).
    Returns (n_meshes, n_verts, warnings)."""
    if bpy is None:
        raise RuntimeError("bpy unavailable")
    from .import_wd import parse_wd1_xbg

    tagged = [o for o in objects if o.get('wd_src') is not None
              and 'wd_mesh_index' in o.keys()]
    if not tagged:
        raise RuntimeError("no WD1-imported meshes with rebuild layout "
                           "(re-import the .xbg with the current addon)")
    src = source_path or tagged[0]['wd_src']
    tagged = [o for o in tagged if o['wd_src'] == src]
    obj_by_mi = {int(o['wd_mesh_index']): o for o in tagged}

    model = parse_wd1_xbg(src)
    L = model['_layout']
    lod0 = L['lod0_meshes']
    off = tuple(L['scale'])
    orig = bytearray(open(src, 'rb').read())
    vdata0_off = L['vdata0_off']
    warnings = []

    # ── bounds/scale expansion (same as the in-place path) ──────────────
    ext = _edited_extent(tagged)
    if ext + abs(off[0]) > 32767.0 * off[1]:
        new_off, expanded = _maybe_expand_bounds(orig, None, L, ext)
        if expanded:
            warnings.append(
                "geometry exceeds the original bounds — position scale "
                "expanded %.6g -> %.6g (all stock vertices re-encoded, "
                "header bbox/sphere updated)" % (off[1], new_off[1]))
            off = new_off
    # keep-list: which LOD0 mesh indices survive.  drop_unselected -> only the
    # supplied objects; otherwise every mesh (unselected ones kept as-is).
    keep = set(obj_by_mi) if drop_unselected else set(range(len(lod0)))
    n_dropped = 0

    # name2pal: vertex-group index -> palette slot, per mesh (reverse of the
    # importer's palette->node->bone resolve).  Built per mesh below.
    bone_name_to_index = {nm: i for i, nm in enumerate(L['bones'])}
    node2bone = {int(k): v for k, v in L['node2bone'].items()}
    bone2node = {v: k for k, v in node2bone.items()}

    # group LOD0 meshes into vertex pools by vb_offset (file order preserved)
    from collections import OrderedDict
    pools = OrderedDict()
    for mi, me in enumerate(lod0):
        pools.setdefault(me['drawcall']['vb_offset'], []).append(mi)

    # ── rebuild the vertex buffer, pool by pool ─────────────────────────
    new_vdata = bytearray()
    pool_newoff = {}        # old vb_off -> new byte offset
    pool_vcount = {}        # old vb_off -> new vertex count
    n_verts = 0
    for vb_off, mis in pools.items():
        # drop the whole pool when none of its meshes survive
        if not any(mi in keep for mi in mis):
            n_dropped += sum(1 for mi in mis)
            continue
        stride = lod0[mis[0]]['stride']
        fmt = lod0[mis[0]]['format']
        old_count = lod0[mis[0]]['drawcall']['vertex_count']
        old_bytes = orig[vdata0_off + vb_off:
                         vdata0_off + vb_off + old_count * stride]
        # pick the vertex-source object: the one whose count changed, else
        # the lowest-index member (shared pools store identical vertices)
        changed = [mi for mi in mis if mi in obj_by_mi
                   and len(obj_by_mi[mi].data.vertices) != old_count]
        if len(changed) > 1:
            warnings.append(
                "pool @vb_off %d edited on %d meshes at once — using mesh %d; "
                "edit a shared pool on ONE object only" % (vb_off, len(changed),
                                                           changed[0]))
        kept_in_pool = [mi for mi in mis if mi in keep and mi in obj_by_mi]
        src_mi = (changed[0] if changed
                  else (kept_in_pool[0] if kept_in_pool else None))
        if src_mi is None:
            new_vdata.extend(old_bytes)        # no object — keep as-is
            pool_newoff[vb_off] = len(new_vdata) - len(old_bytes)
            pool_vcount[vb_off] = old_count
            continue
        ob = obj_by_mi[src_mi]
        me = ob.data
        new_count = len(me.vertices)
        try:
            lay = component_layout(fmt, stride)
        except ValueError as e:
            warnings.append("mesh %d: %s — pool kept unchanged" % (src_mi, e))
            new_vdata.extend(old_bytes)
            pool_newoff[vb_off] = len(new_vdata) - len(old_bytes)
            pool_vcount[vb_off] = old_count
            continue

        pool_newoff[vb_off] = len(new_vdata)
        pool_vcount[vb_off] = new_count
        # start from original bytes (existing verts) + zero-filled new verts
        pool = bytearray(old_bytes)
        if new_count > old_count:
            pool.extend(b'\0' * ((new_count - old_count) * stride))
        elif new_count < old_count:
            del pool[new_count * stride:]

        uvs = _vertex_uvs(me, 0)
        uv2s = _vertex_uvs(me, 1)
        cols = _vertex_colors(me)
        norms = _vertex_normals(me, recalculate=recalculate_normals)
        # tangent frames from UVs: always when recalculate_normals (all verts
        # get a fresh TBN), otherwise only for new verts (Avatar parity).
        tans, tsigns = (_vertex_tangents(me)
                        if ('tangent' in lay
                            and (new_count > old_count or recalculate_normals))
                        else (None, None))
        neutral = _neutral_color(me, cols) if 'color' in lay else None

        def _name2pal(mi_, ob_):
            """vertex-group index -> palette slot for one mesh."""
            bmap = lod0[mi_]['bone_map']
            pal = L['palettes'][bmap] if bmap < len(L['palettes']) else []
            n2s = {nd: s for s, nd in enumerate(pal)}
            out = {}
            for g in ob_.vertex_groups:
                bi = bone_name_to_index.get(g.name)
                if bi is None:
                    continue
                slot = n2s.get(bone2node.get(bi))
                if slot is not None:
                    out[g.index] = slot
            return out

        name2pal = _name2pal(src_mi, ob)
        # Re-skin source per pool VERTEX: in a shared pool each member mesh
        # only has valid weights for the vertex range IT uses (disjoint
        # min..max), decoded through its own palette.  So a vertex's weights
        # must come from the member that OWNS it, not from one source object.
        owner = {}           # pool vertex index -> (mesh data, name2pal)
        if reskin:
            for mi in mis:
                ob_m = obj_by_mi.get(mi)
                if ob_m is None:
                    continue
                n2p_m = _name2pal(mi, ob_m)
                me_m = ob_m.data
                me_m.calc_loop_triangles()
                for lt in me_m.loop_triangles:   # claim the verts it uses
                    for vidx in lt.vertices:
                        owner[vidx] = (me_m, n2p_m)

        for vi, v in enumerate(me.vertices):
            base = vi * stride
            new_vert = vi >= old_count
            # existing verts: patch only authored fields, keep the rest of
            # the original bytes (weights/tangents preserved).  New verts:
            # encode the FULL frame (tangent/binormal from UVs, neutral
            # colour, skin from vertex groups) so nothing is left zeroed.
            col = (cols[vi] if cols else None)
            if new_vert and col is None:
                col = neutral
            tan = bn = tw = bw = None
            if tans and (new_vert or recalculate_normals):
                tan = tans[vi]
                tw = tsigns[vi]
                if norms:
                    n = mathutils.Vector(norms[vi])
                    t = mathutils.Vector(tan)
                    bn = tuple((n.cross(t)).normalized() * tw)
                    bw = 1.0
            patch_vertex(pool, base, lay, off,
                         co=(v.co.x, v.co.y, v.co.z),
                         normal=norms[vi] if norms else None,
                         uv=uvs[vi] if uvs else None,
                         uv2=uv2s[vi] if uv2s else None,
                         color=col,
                         tangent=tan, tangent_w=tw,
                         binormal=bn, binormal_w=bw)
            # New verts are skinned from the edited object's groups.  Existing
            # verts only when re-skinning, and then from the member that OWNS
            # the vertex (shared-pool weights are valid only on each member's
            # own range); fall back to original bytes if no owner is found.
            if new_vert:
                _encode_skin(pool, base, lay, v.groups, name2pal)
            elif reskin:
                own = owner.get(vi)
                if own is not None:
                    src_me, src_n2p = own
                    _encode_skin(pool, base, lay,
                                 src_me.vertices[vi].groups, src_n2p)
            n_verts += 1
        new_vdata.extend(pool)

    # ── rebuild the index buffer + per-mesh drawcall fields ─────────────
    new_idata = bytearray()
    dc_new = {}
    for mi, mesh in enumerate(lod0):
        vb_off = mesh['drawcall']['vb_offset']
        if mi not in keep:
            # DROPPED mesh: empty drawcall (renders nothing; its pool was
            # excluded from the buffer so the file is smaller)
            dc_new[mi] = {'vb_offset': 0, 'vertex_count': 0, 'index_start': 0,
                          'index_count': 0, 'prim_count': 0,
                          'min_index': 0, 'max_index': 0}
            continue
        ob = obj_by_mi.get(mi)
        if ob is not None:
            tris = _mesh_triangles(ob.data)
        else:                                  # kept but not in scene — keep orig
            dc = mesh['drawcall']
            old_idx = struct.unpack_from(
                '<%dH' % dc['index_count'],
                orig, L['buf_frames'][0]['idata_off'] + dc['index_start'] * 2)
            tris = [(old_idx[t], old_idx[t + 2], old_idx[t + 1])
                    for t in range(0, len(old_idx) - 2, 3)]
        start = len(new_idata) // 2
        flat = []
        for a, b, c in tris:
            flat += [a, c, b]                  # restore file winding
        for ix in flat:
            new_idata += struct.pack('<H', ix & 0xFFFF)
        dc_new[mi] = {
            'vb_offset': pool_newoff[vb_off],
            'vertex_count': pool_vcount[vb_off],
            'index_start': start,
            'index_count': len(flat),
            'prim_count': len(tris),
            'min_index': min(flat) if flat else 0,
            'max_index': max(flat) if flat else 0,
        }

    # ── reassemble buffer 0 + splice + patch drawcall fields ────────────
    out = _splice_buffers(orig, L, bytes(new_vdata), bytes(new_idata))
    out = bytearray(out)
    for mi, mesh in enumerate(lod0):
        nf = dc_new[mi]
        for dc in [mesh['drawcall']] + [r['drawcall'] for r in mesh['ranges']]:
            _patch_dc_fields(out, dc['_offs'], nf)
    with open(out_path, 'wb') as f:
        f.write(out)
    if n_dropped:
        warnings.append("dropped %d unselected mesh(es) from the output"
                        % n_dropped)
    return len(obj_by_mi), n_verts, warnings


def _patch_dc_fields(buf, offs, nf):
    struct.pack_into('<I', buf, offs['vb_offset'], nf['vb_offset'])
    struct.pack_into('<I', buf, offs['prim_count'], nf['prim_count'])
    struct.pack_into('<I', buf, offs['index_count'], nf['index_count'])
    struct.pack_into('<I', buf, offs['index_start'], nf['index_start'])
    struct.pack_into('<H', buf, offs['vertex_count'], nf['vertex_count'] & 0xFFFF)
    struct.pack_into('<H', buf, offs['min_index'], nf['min_index'] & 0xFFFF)
    struct.pack_into('<H', buf, offs['max_index'], nf['max_index'] & 0xFFFF)


def _aligned_count(buf, val):
    """IBinaryArchive ndVector count: pad(4) then u32."""
    while len(buf) % 4:
        buf.append(0)
    buf.extend(struct.pack('<I', val))


def _splice_buffers(orig, L, new_vdata0, new_idata0):
    """Rebuild the SGfxBuffers section with buffer 0 replaced, keeping the
    other buffers byte-exact, and splice it into the original file."""
    frames = L['buf_frames']
    section = bytearray()
    _aligned_count(section, len(frames))       # buffer count
    for bi, bf in enumerate(frames):
        if bi == 0:
            vdata, idata = new_vdata0, new_idata0
        else:
            vdata = orig[bf['vdata_off']:bf['vdata_off'] + bf['vsz']]
            idata = orig[bf['idata_off']:bf['idata_off'] + bf['isz']]
        _aligned_count(section, len(vdata))
        section.extend(vdata)
        _aligned_count(section, len(idata))
        section.extend(idata)
    return (orig[:L['buffers_section_start']] + bytes(section)
            + orig[L['buffers_section_end']:])
