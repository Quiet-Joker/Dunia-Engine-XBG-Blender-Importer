"""XBG mesh EXPORT (inject side) — the counterpart to import_mesh.py.

Everything needed to turn an edited Blender mesh into XBG geometry:
  * mesh prep: `_triangulate_and_split_by_material` (triangulate + per-material
    slices + UV-seam split), `_split_mesh_by_face_budget` (uint16 index wall),
  * vertex byte-encoding: `_encode_vertices` (position int16/float, UV0/1/2,
    D3DCOLOR normal, int8 tangent/binormal, vertex colour, bone weights, laid out
    per the format flags) + `_null_vertex` (degenerate filler),
  * index buffer: `_build_index_buffer` (uint16 triangle list, winding reverse).

Stock vertices re-export their authored normals/tangents byte-exact (from the
xbg_normal / xbg_tangent POINT attrs that survive bmesh); new geometry uses
viewport normals + UV-computed tangents. BYTE-EXACT ROUND-TRIP is the contract —
validate any change against the inject md5 baseline (see agents.md).
"""

import struct
import math

import bpy
import bmesh
import mathutils

from ..Core.debug  import TraceLogger
from .bounds_avatar import clamp_to_16bit
from .binary_avatar import LE
from .import_mesh_avatar    import VertexFlags
from .normals_avatar        import build_tbn_lookups, compute_tangents_from_uvs
from .export_uv_avatar      import _split_uv_seams


def _encode_vertices(tri_mesh, vert_format_flags, vert_stride,
                     pos_scale, uv_trans, uv_scale, obj,
                     apply_scale=1.0, ignore_limits=False,
                     color_map=None,
                     neutral_empty_colors=True,
                     weight_map=None,
                     endian=LE):
    """
    Encode a (already-triangulated) Blender mesh into an XBG vertex buffer.

    apply_scale      : uniform pre-multiplier on all positions before encoding.
    Stock verts re-export their authored normals (the `xbg_normal` POINT attr,
    read below); new verts fall back to Blender's corner normals. No recompute
    option — to reset normals, do it in Blender before export.
    color_map     : {vertex_index: (r,g,b,a)} floats from `vertex_colors.build_vertex_color_map()`.
                    None -> full white (255,255,255,255).
    weight_map    : {vertex_index: ([w0,w1,w2,w3], [i0,i1,i2,i3])} from
                    _build_weight_map().  None -> pin every vertex to bone 0.

    Returns (vertex_buffer_bytes, clamped_count).
    """
    # Endianness shortcut for all struct.pack_into calls below.  All multi-
    # byte fields (positions, UVs) follow the target file's byte order.
    # Single-byte fields (normals, tangents, binormals, colors, bone wts)
    # are byte-order-independent so they keep using '<' (or no prefix would
    # work equally well — '<' is kept for diff cleanliness against pre-PS3
    # versions of this function).
    en = endian

    # -- Rotation correction (mirrors the import) -------------------------
    rz = obj.rotation_euler.z
    if abs(rz - math.radians(180)) < 0.01:
        rot_inv = mathutils.Matrix.Rotation(-math.radians(180), 4, 'Z')
    elif abs(rz) > 0.01:
        rot_inv = mathutils.Matrix.Rotation(-rz, 4, 'Z')
    else:
        rot_inv = mathutils.Matrix.Identity(4)

    # -- Component offsets ------------------------------------------------
    _, co = VertexFlags.calculate_stride(vert_format_flags)
    hpf  = bool(vert_format_flags & VertexFlags.POS_FLOAT)
    huv0 = bool(vert_format_flags & VertexFlags.UV0)
    huv1 = bool(vert_format_flags & VertexFlags.UV1)
    huv2 = bool(vert_format_flags & VertexFlags.UV2)
    hnrm = bool(vert_format_flags & VertexFlags.NORMAL)
    hcol = bool(vert_format_flags & VertexFlags.COLOR)
    hbwt = bool(vert_format_flags & VertexFlags.BONE_WTS1)
    htan = bool(vert_format_flags & VertexFlags.TANGENT)
    hbin = bool(vert_format_flags & VertexFlags.BINORMAL)

    mesh    = tri_mesh
    # Per-vertex accumulators: instead of storing every loop value and re-summing
    # per vertex in the encode loop (the old hot path — tens of thousands of
    # sum() calls), accumulate running sums DURING the loop walk that happens
    # anyway. Average = sum/count, byte-identical result.
    uv_map  = {}   # vertex_index -> [sum_u, sum_v, count]            (UV0)
    uv1_map = {}   # vertex_index -> [sum_u, sum_v, count]            (UV1, 2nd layer)
    nrm_map = {}   # vertex_index -> [sum_nx, sum_ny, sum_nz, count]

    # -- Tangent / binormal lookup (per-vertex) ---------------------------
    # Stored at import time as POINT attributes on the SOURCE mesh
    # (obj.data).  tri_mesh has a different vertex index space after
    # material split, so the lookup matches by 3D position — see
    # normals.build_tbn_lookups for the matching logic.
    src_mesh = obj.data
    tan_lookup, bin_lookup = build_tbn_lookups(src_mesh, mesh, htan, hbin)

    # -- Per-vertex UV accumulation ---------------------------------------
    if huv0 and mesh.uv_layers:
        uv_layer = mesh.uv_layers.active
        if uv_layer:
            _ud = uv_layer.data
            _loops = mesh.loops
            for poly in mesh.polygons:
                for li in poly.loop_indices:
                    vi = _loops[li].vertex_index
                    uv = _ud[li].uv
                    a = uv_map.get(vi)
                    if a is None:
                        uv_map[vi] = [uv.x, uv.y, 1]
                    else:
                        a[0] += uv.x; a[1] += uv.y; a[2] += 1

    # -- UV1 source: a SECOND uv layer.  Foliage (tree trunk / leaf /
    # grass) shaders sample UV1; original Avatar character parts have
    # UV1 = sentinel, so we ONLY encode a real UV1 when the mesh
    # actually carries a 2nd UV layer.  Single-UV meshes keep the
    # sentinel path below unchanged (no regression for existing files).
    if huv1 and mesh.uv_layers and len(mesh.uv_layers) >= 2:
        _active = mesh.uv_layers.active
        uv1_layer = next((L for L in mesh.uv_layers if L is not _active),
                         None)
        if uv1_layer:
            _u1d = uv1_layer.data
            _loops = mesh.loops
            for poly in mesh.polygons:
                for li in poly.loop_indices:
                    vi = _loops[li].vertex_index
                    uv = _u1d[li].uv
                    a = uv1_map.get(vi)
                    if a is None:
                        uv1_map[vi] = [uv.x, uv.y, 1]
                    else:
                        a[0] += uv.x; a[1] += uv.y; a[2] += 1

    # -- Per-vertex normal accumulation -----------------------------------
    # CRITICAL: this MUST run before mesh.calc_tangents() below, because
    # calc_tangents internally calls calc_normals_split() which overwrites
    # the custom XBG split normals we stored at import time.  By reading
    # corner_normals first, we snapshot the authored XBG normals before
    # calc_tangents clobbers them.
    if hnrm:
        _corner_normals = getattr(mesh, 'corner_normals', None)
        _loops = mesh.loops
        for poly in mesh.polygons:
            for li in poly.loop_indices:
                vi = _loops[li].vertex_index
                if _corner_normals is not None:
                    n = _corner_normals[li].vector
                    nx_, ny_, nz_ = n[0], n[1], n[2]
                else:
                    n = _loops[li].normal
                    nx_, ny_, nz_ = n.x, n.y, n.z
                a = nrm_map.get(vi)
                if a is None:
                    nrm_map[vi] = [nx_, ny_, nz_, 1]
                else:
                    a[0] += nx_; a[1] += ny_; a[2] += nz_; a[3] += 1

    # -- Compute tangents from UVs for verts without stored XBG data ------
    # NOTE: calc_tangents() (called inside compute_tangents_from_uvs)
    # recomputes split normals as a side-effect — but we already captured
    # the authored XBG normals into nrm_map above, so the encoding loop
    # still writes the correct per-vertex normals.
    missing_verts = (htan or hbin) and any(
        v.index not in tan_lookup for v in mesh.vertices
    )
    if missing_verts:
        computed_tan, computed_bin = compute_tangents_from_uvs(
            mesh, tan_lookup, htan, hbin)
    else:
        computed_tan = {}
        computed_bin = {}

    # -- Build buffer ------------------------------------------------------
    inv_pos  = 1.0 / pos_scale
    n_verts  = len(mesh.vertices)
    buf      = bytearray(n_verts * vert_stride)  # pre-zeroed
    clamped  = 0

    # ── Trace: setup summary for this slice ────────────────────────────
    flag_names = []
    for bit, nm in (
        (VertexFlags.POS_FLOAT, "POS_FLOAT"),
        (VertexFlags.POS_INT16, "POS_INT16"),
        (VertexFlags.UV0,       "UV0"),
        (VertexFlags.UV1,       "UV1"),
        (VertexFlags.UV2,       "UV2"),
        (VertexFlags.BONE_WTS1, "BONE_WTS1"),
        (VertexFlags.BONE_WTS2, "BONE_WTS2"),
        (VertexFlags.NORMAL,    "NORMAL"),
        (VertexFlags.COLOR,     "COLOR"),
        (VertexFlags.TANGENT,   "TANGENT"),
        (VertexFlags.BINORMAL,  "BINORMAL"),
    ):
        if vert_format_flags & bit:
            flag_names.append(nm)
    TraceLogger.kvblock(
        f"_encode_vertices setup  obj='{obj.name}'",
        [
            ("n_verts",            n_verts),
            ("vert_stride",        vert_stride),
            ("vert_format_flags",  f"0x{vert_format_flags:04X} | {' | '.join(flag_names)}"),
            ("pos_scale",          pos_scale),
            ("inv_pos",            inv_pos),
            ("uv_trans",           uv_trans),
            ("uv_scale",           uv_scale),
            ("apply_scale",        apply_scale),
            ("ignore_limits",      bool(ignore_limits)),
            ("nrm_map_coverage",   f"{len(nrm_map)}/{n_verts}"),
            ("uv_map_coverage",    f"{len(uv_map)}/{n_verts}"),
            ("uv1_map_coverage",   f"{len(uv1_map)}/{n_verts}" if huv1 else "n/a"),
            ("tan_lookup_size",    len(tan_lookup) if isinstance(tan_lookup, dict)
                                      else len(tan_lookup or [])),
            ("weight_map_size",    len(weight_map) if weight_map else 0),
            ("16bit_world_half",   f"±{32767 * pos_scale:.4f} world units"),
        ],
        tier="DEBUG",
        event="encode_setup")

    # Pick a small sample set of verts to log in detail (TRACE only).
    # First 5 verts, last 5 verts — covers the boundary cases where
    # bugs in attribute-lookup most often manifest (first vert tends to
    # be untouched by remap; last vert tends to be the new geometry).
    sample_set = set()
    if TraceLogger.trace_enabled() and n_verts > 0:
        for i in range(min(5, n_verts)):
            sample_set.add(i)
        for i in range(max(0, n_verts - 5), n_verts):
            sample_set.add(i)
    # Stats accumulators (always-on counters even when TRACE is off; the
    # numbers are cheap and they unlock the encode-summary table below).
    _stat_zero_normal      = 0
    _stat_weight_sum_warn  = 0  # weights summed outside [254, 256]
    _stat_no_weight_entry  = 0
    _stat_uv0_missing      = 0
    _stat_uv1_missing      = 0
    _stat_pos_out_of_range = 0
    _max_abs_int16 = [0, 0, 0]

    # Per-vertex authored normal that SURVIVES the bmesh split. Custom split
    # normals are LOOP data and bmesh (from_mesh/to_mesh, used by the split-by-
    # material / uv-seam passes) drops them, so reading corner-normals would
    # give the host AVERAGED geometric normals. The importer also stores the
    # normal as the `xbg_normal` POINT attribute (POINT attrs DO survive bmesh,
    # like xbg_tangent); stock verts read it here. New/foreign verts have a
    # zero/absent entry and fall back to corner-normals (nrm_map).
    _stored_nrm_attr = mesh.attributes.get("xbg_normal") if hnrm else None
    _stored_nrm_n = len(_stored_nrm_attr.data) if _stored_nrm_attr else 0

    for v in mesh.vertices:
        base = v.index * vert_stride
        rc = rot_inv @ mathutils.Vector((v.co.x, v.co.y, v.co.z, 1.0))
        fx, fy, fz = rc.x * apply_scale, rc.y * apply_scale, rc.z * apply_scale

        # -- Position ------------------------------------------------------
        if hpf:
            struct.pack_into(f'{en}fff', buf, base + co['pos_float'], fx, fy, fz)
        else:
            px_raw = round(fx * inv_pos)
            py_raw = round(fy * inv_pos)
            pz_raw = round(fz * inv_pos)

            if ignore_limits:
                px = px_raw & 0xFFFF
                py = py_raw & 0xFFFF
                pz = pz_raw & 0xFFFF
                if px > 32767: px -= 65536
                if py > 32767: py -= 65536
                if pz > 32767: pz -= 65536
            else:
                px = clamp_to_16bit(px_raw)
                py = clamp_to_16bit(py_raw)
                pz = clamp_to_16bit(pz_raw)
                if px != px_raw or py != py_raw or pz != pz_raw:
                    clamped += 1
                    _stat_pos_out_of_range += 1

            # Track the largest absolute int16 we've encoded along each axis;
            # this is what tells us at a glance how close we are to the wall.
            for _a, _v in enumerate((px, py, pz)):
                _a_abs = -_v if _v < 0 else _v
                if _a_abs > _max_abs_int16[_a]:
                    _max_abs_int16[_a] = _a_abs

            struct.pack_into(f'{en}hhh', buf, base + co['pos_int16'], px, py, pz)
            # W component must be 1 -- zero W sends every vertex to infinity
            struct.pack_into(f'{en}H', buf, base + co['pos_int16'] + 6, 1)

        # -- UV0 -----------------------------------------------------------
        if huv0:
            a = uv_map.get(v.index)
            if a:
                c = a[2]; au = a[0] / c; av = 1.0 - a[1] / c   # flip V
            else:
                au = 0.0; av = 1.0
            struct.pack_into(f'{en}hh', buf, base + co['uv0'],
                clamp_to_16bit(round((au - uv_trans) / uv_scale)),
                clamp_to_16bit(round((av - uv_trans) / uv_scale)))

        # -- UV1 -- real 2nd-layer UV when present, else sentinel --------
        if huv1:
            uv1s = uv1_map.get(v.index)
            if uv1s:
                c = uv1s[2]; bu = uv1s[0] / c; bv = 1.0 - uv1s[1] / c   # flip V
                struct.pack_into(f'{en}hh', buf, base + co['uv1'],
                    clamp_to_16bit(round((bu - uv_trans) / uv_scale)),
                    clamp_to_16bit(round((bv - uv_trans) / uv_scale)))
            else:
                struct.pack_into(f'{en}hh', buf, base + co['uv1'],
                                 -32768, -32768)
        if huv2:
            struct.pack_into(f'{en}hh', buf, base + co['uv2'], -32768, -32768)

        # -- Normal (D3DCOLOR: 3 × unsigned byte + 1 sign byte) -----------
        if hnrm:
            # Prefer the bmesh-surviving authored normal (xbg_normal POINT
            # attr) for stock verts — custom split normals are lost through the
            # inject's bmesh split, so corner-normals (nrm_map) would give the
            # host AVERAGED geometric normals. Near-zero stored vector = "no
            # data" (new/foreign vert) -> fall back to corner-normals.
            _sn = (_stored_nrm_attr.data[v.index].vector
                   if (_stored_nrm_attr and v.index < _stored_nrm_n) else None)
            if _sn is not None and (_sn.x * _sn.x + _sn.y * _sn.y
                                    + _sn.z * _sn.z) > 1e-8:
                nx, ny, nz = _sn.x, _sn.y, _sn.z
            else:
                nrms = nrm_map.get(v.index)
                if nrms:
                    c = nrms[3]; nx = nrms[0] / c; ny = nrms[1] / c; nz = nrms[2] / c
                else:
                    # Vertex not referenced by any polygon — zero normal
                    nx, ny, nz = 0.0, 0.0, 0.0
            rn = rot_inv @ mathutils.Vector((nx, ny, nz, 0.0))
            nl = mathutils.Vector((rn.x, rn.y, rn.z))
            if nl.length > 1e-4:
                nl.normalize()
            # The normal is NOT negated (see normals.apply_split_normals): it's
            # the outward surface normal; the WINDING reverse (below, in the
            # index buffer) handles the game's front-face convention. Stock
            # round-trips byte-exact because import doesn't negate either.
            nrm_off = co['normal']
            # D3DCOLOR encode: UNSIGNED-normalised (byte = round((c+1)/2*255),
            # inverse of byte/255*2-1; NOT signed*127) + BGRA byte order
            # (byte0=Z, byte1=Y, byte2=X — the GPU presents xyz=byte2,byte1,
            # byte0). round() keeps unchanged data byte-exact.
            struct.pack_into('<BBBB', buf, base + nrm_off,
                max(0, min(255, int(round((nl.z + 1.0) * 127.5)))),   # byte0 = Z
                max(0, min(255, int(round((nl.y + 1.0) * 127.5)))),   # byte1 = Y
                max(0, min(255, int(round((nl.x + 1.0) * 127.5)))),   # byte2 = X
                0x80)   # byte3 = D3DCOLOR alpha (TBN handedness/sign)

        # -- Tangent (3 × int8 + 1 sign byte) -----------------------------
        # Resolution order: stored XBG data → nearest-neighbor → UV-computed → orthogonal → zero
        # tangent_dir keeps the final game-space tangent so the binormal
        # fallback below can build a bitangent that is actually perpendicular
        # to the tangent we just wrote, instead of inventing a new one.
        tangent_dir = None
        if htan:
            t = tan_lookup.get(v.index) or computed_tan.get(v.index)
            # Orthogonal floor is UNCONDITIONAL: a zero tangent has no valid
            # meaning for the engine's DXT5-GA normal mapping — it makes every
            # UV/normal seam render as a hard "cut" line (proven: a foreign mesh
            # injected with all tangent sources missing shipped 100% zero
            # tangents and showed seam cuts on the legs, while every stock
            # submesh has 100% non-zero TBN). Real sources (stored XBG tangent,
            # then UV-computed) are tried first; this only guarantees we never
            # emit a zero tangent when a normal exists.
            if t is None and hnrm:
                # Build a tangent perpendicular to the (just-encoded) normal `nl`.
                # Pick the world axis least parallel to N to avoid degenerate cross.
                if abs(nl.x) < 0.9:
                    perp = mathutils.Vector((1.0, 0.0, 0.0))
                else:
                    perp = mathutils.Vector((0.0, 1.0, 0.0))
                tvec = nl.cross(perp)
                if tvec.length > 1e-6:
                    tvec.normalize()
                # `nl` is already in game space (post-rot_inv), so tvec is too —
                # apply rot_inv ONLY to stored/UV-computed values which are in
                # Blender local space. Mark with sentinel by setting t directly.
                t = (tvec.x, tvec.y, tvec.z, 0x80)
                _tan_in_game_space = True
            else:
                _tan_in_game_space = False
            if t is not None:
                tx, ty, tz, tw = t
                if _tan_in_game_space:
                    rt = mathutils.Vector((tx, ty, tz))
                else:
                    rv = rot_inv @ mathutils.Vector((tx, ty, tz, 0.0))
                    rt = mathutils.Vector((rv.x, rv.y, rv.z))
                tangent_dir = rt   # share with binormal fallback below
                # D3DCOLOR encode: unsigned-normalised + BGRA (see normal note).
                struct.pack_into('<BBBB', buf, base + co['tangent'],
                    max(0, min(255, int(round((rt.z + 1.0) * 127.5)))),   # byte0 = Z
                    max(0, min(255, int(round((rt.y + 1.0) * 127.5)))),   # byte1 = Y
                    max(0, min(255, int(round((rt.x + 1.0) * 127.5)))),   # byte2 = X
                    tw)

        # -- Binormal (3 × int8 + 1 sign byte) ----------------------------
        # Orthogonal fallback: B = N × T using the ACTUAL tangent we just wrote,
        # not a freshly-rebuilt synthetic tangent.  The old code re-ran the
        # axis-pick → cross dance against `nl`, which produced a bitangent that
        # was perpendicular to N but completely unrelated to the tangent in the
        # buffer — silently breaking the TBN basis for vertices where the
        # tangent came from stored data / nearest-neighbor / UV computation.
        if hbin:
            b = bin_lookup.get(v.index) or computed_bin.get(v.index)
            # Unconditional floor (see tangent note above): B = N × T
            # whenever we have a normal and a tangent, so the TBN basis
            # is never left zero/broken.
            if b is None and hnrm and tangent_dir is not None:
                bvec = nl.cross(tangent_dir)
                if bvec.length > 1e-6:
                    bvec.normalize()
                b = (bvec.x, bvec.y, bvec.z, 0x80)
                _bin_in_game_space = True
            else:
                _bin_in_game_space = False
            if b is not None:
                bx, by, bz, bw = b
                if _bin_in_game_space:
                    rb = mathutils.Vector((bx, by, bz))
                else:
                    rv = rot_inv @ mathutils.Vector((bx, by, bz, 0.0))
                    rb = mathutils.Vector((rv.x, rv.y, rv.z))
                # D3DCOLOR encode: unsigned-normalised + BGRA (see normal note).
                struct.pack_into('<BBBB', buf, base + co['binormal'],
                    max(0, min(255, int(round((rb.z + 1.0) * 127.5)))),   # byte0 = Z
                    max(0, min(255, int(round((rb.y + 1.0) * 127.5)))),   # byte1 = Y
                    max(0, min(255, int(round((rb.x + 1.0) * 127.5)))),   # byte2 = X
                    bw)

        # -- Color (= the aaa.fx MASK: r=spec, g=detail/normal2, b=normal
        # strength + diffuse tint, a=AO) ----------------------------------
        # Two things matter here:
        #  (1) D3DCOLOR/BGRA byte order: byte0=B, byte1=G, byte2=R, byte3=A
        #      (the game presents R from byte2). Writing RGBA swaps the spec
        #      and normal-strength mask channels. round() keeps stock byte-exact.
        #  (2) NEVER default to white. White = mask maxed = full spec + full
        #      normal -> any material renders glossy. The importer's neutral is
        #      black (0,0,0,1); use that for verts with no color data.
        if hcol:
            if color_map and v.index in color_map:
                r, g, b, a = color_map[v.index]
                # New geometry added in Blender edit mode initialises its POINT
                # color to (0,0,0,0). A real authored vert is NEVER exactly
                # (0,0,0,0) (decode-verified: stock body has zero such verts,
                # foreign/new verts are ~100% of them; the file DOES author
                # (255,255,255,0) = real AO=0, so we test ALL FOUR channels, not
                # alpha alone). Rewrite that all-zero default to neutral
                # (0,0,0,1) = RGBA: only alpha→1 (AO bright, kills the shadow);
                # R=G=B stay 0.
                #
                # WHY blue stays 0 (do NOT set it to 1): every aaa.fx mask channel
                # is OVERLOADED — blue drives BOTH the normal-map strength
                # (lerp(flat, normalMap, b), line 419) AND the diffuse tint
                # (lerp(DiffuseColorBase, DiffuseColor1, b), line 375). An earlier
                # attempt set blue=1 to "show the normal map", but on materials
                # that read vertex colour that ALSO forced DiffuseColor1 and
                # applied the normal map over the foreign mesh's UV-computed
                # tangents → visible WAVE/QUANTIZATION artifacts on face+body
                # (user-confirmed in-game). blue=0 keeps diffuse at Base and the
                # normal flat — the clean neutral. There is NO blue value that
                # isolates "normal map" from "diffuse tint"; if new geometry needs
                # its normal map, the user must PAINT the blue channel per their
                # material's intent. Gated by "Generate Neutral Vertex Colors"
                # (default ON) — OFF writes raw (0,0,0,0) as painted.
                if (neutral_empty_colors
                        and r == 0.0 and g == 0.0 and b == 0.0 and a == 0.0):
                    a = 1.0
                struct.pack_into('<BBBB', buf, base + co['color'],
                    max(0, min(255, int(round(b * 255)))),   # byte0 = B
                    max(0, min(255, int(round(g * 255)))),   # byte1 = G
                    max(0, min(255, int(round(r * 255)))),   # byte2 = R
                    max(0, min(255, int(round(a * 255)))))   # byte3 = A
            else:
                # Include Vertex Colors OFF (color_map is None) -> neutral for
                # every vert: (0,0,0,255) RGBA. Bytes are B,G,R,A.
                struct.pack_into('<BBBB', buf, base + co['color'],
                                 0, 0, 0, 255)   # = RGBA(0,0,0,255)

        # -- Bone weights --------------------------------------------------
        if hbwt:
            bwt_off = co['bone_wts1']
            if weight_map and v.index in weight_map:
                wb, ib = weight_map[v.index]
                struct.pack_into('<BBBB', buf, base + bwt_off,
                                 wb[0], wb[1], wb[2], wb[3])
                struct.pack_into('<BBBB', buf, base + bwt_off + 4,
                                 ib[0], ib[1], ib[2], ib[3])
                # Stat: weights should sum to ~255 (some rounding error OK).
                _wsum = wb[0] + wb[1] + wb[2] + wb[3]
                if _wsum < 253 or _wsum > 257:
                    _stat_weight_sum_warn += 1
            else:
                # Fallback: pin to bone 0 with full weight
                struct.pack_into('<BBBB', buf, base + bwt_off,     255, 0, 0, 0)
                struct.pack_into('<BBBB', buf, base + bwt_off + 4,   0, 0, 0, 0)
                _stat_no_weight_entry += 1

        # Stats: missing UV / zero-normal entries (cheap, always-on).
        if huv0 and v.index not in uv_map:
            _stat_uv0_missing += 1
        if huv1 and v.index not in uv_map:  # uv1 fallback path is sentinel
            pass  # already counted via uv1_map below
        if huv1 and v.index not in uv_map and v.index not in uv1_map:
            _stat_uv1_missing += 1
        if hnrm and v.index not in nrm_map:
            _stat_zero_normal += 1

        # TRACE: per-vertex dump for the sampled set.  Reads the bytes
        # we JUST wrote so the log shows exactly what landed in the buf.
        if v.index in sample_set:
            vbytes = bytes(buf[base : base + vert_stride])
            sample_row = {
                "obj": obj.name,
                "v_idx": v.index,
                "blender_co": (round(v.co.x, 6), round(v.co.y, 6), round(v.co.z, 6)),
                "rotinv_world": (round(rc.x, 6), round(rc.y, 6), round(rc.z, 6)),
                "scaled_world": (round(fx, 6), round(fy, 6), round(fz, 6)),
            }
            if not hpf:
                sample_row["raw_int16"]   = (px_raw, py_raw, pz_raw)
                sample_row["clamped_int16"] = (px, py, pz)
                sample_row["world_recovered"] = (
                    round(px * pos_scale, 6),
                    round(py * pos_scale, 6),
                    round(pz * pos_scale, 6))
            if hbwt:
                if weight_map and v.index in weight_map:
                    wb, ib = weight_map[v.index]
                    sample_row["weights_u8"] = list(wb)
                    sample_row["palette_slots"] = list(ib)
                else:
                    sample_row["weights_u8"] = [255, 0, 0, 0]
                    sample_row["palette_slots"] = [0, 0, 0, 0]
                    sample_row["weights_fallback"] = True
            if huv0:
                sample_row["uv0_blender"] = tuple(round(x, 5)
                    for x in uv_map.get(v.index, [(0.0, 0.0)])[0])
            if hnrm:
                _nl = nrm_map.get(v.index, [(0.0, 0.0, 0.0)])
                sample_row["normal_loops"] = len(_nl)
                if _nl:
                    sample_row["normal_avg"] = (round(_nl[0][0], 5),
                                                 round(_nl[0][1], 5),
                                                 round(_nl[0][2], 5))
            sample_row["encoded_hex"] = vbytes.hex()
            TraceLogger.struct("encode_sample", sample_row, tier="TRACE")
            TraceLogger.trace(
                f"  [trace] v[{v.index:>5}] {obj.name}  "
                f"co=({v.co.x:+.4f},{v.co.y:+.4f},{v.co.z:+.4f}) → "
                f"int16=({px if not hpf else '-':>6},{py if not hpf else '-':>6},{pz if not hpf else '-':>6}) "
                f"hex={vbytes[:16].hex()}…")

    # ── Encode summary (after loop) ─────────────────────────────────
    TraceLogger.kvblock(
        f"_encode_vertices stats  obj='{obj.name}'",
        [
            ("n_verts",                 n_verts),
            ("clamped_to_int16",        clamped),
            ("pos_out_of_range",        _stat_pos_out_of_range),
            ("max_abs_int16_per_axis",  tuple(_max_abs_int16)),
            ("axis_headroom",           tuple(32767 - a for a in _max_abs_int16)),
            ("zero_normal_count",       _stat_zero_normal),
            ("weight_sum_off_count",    _stat_weight_sum_warn),
            ("no_weight_entry_count",   _stat_no_weight_entry),
            ("uv0_missing_count",       _stat_uv0_missing),
            ("uv1_missing_count",       _stat_uv1_missing),
            ("encoded_bytes",           len(buf)),
        ],
        tier="DEBUG",
        event="encode_stats")
    return bytes(buf), clamped


# ============================================================
# Index buffer
# ============================================================

def _null_vertex(flags, stride, endian):
    """Build one shader-safe null vertex for a given vertex format.

    Produces `stride` bytes where every component is zero EXCEPT:
    - ``POS_INT16``: the W component is set to 1 (prevents the vertex
      being sent to infinity by the vertex shader's W-divide).
    - ``BONE_WTS1``: the first weight byte is 0xFF (100 % on bone 0),
      matching the convention used for all other vertices.

    The game never reads this vertex data for DNKS=0 slots (it skips
    the submesh entirely before touching the VB), so these values only
    matter if the GPU driver validates buffers before the skip.
    """
    buf = bytearray(stride)
    _, offsets = VertexFlags.calculate_stride(flags)

    if flags & VertexFlags.POS_INT16:
        off = offsets.get('pos_int16', 0)
        # (x=0, y=0, z=0, w=1)
        struct.pack_into(f'{endian}4h', buf, off, 0, 0, 0, 1)

    if flags & VertexFlags.BONE_WTS1:
        off = offsets.get('bone_wts1', 0)
        # weights: (0xFF, 0, 0, 0)  indices: (0, 0, 0, 0)
        struct.pack_into('8B', buf, off, 0xFF, 0, 0, 0, 0, 0, 0, 0)

    return bytes(buf)


def _triangulate_and_split_by_material(obj, split_by_material=False):
    """
    Triangulate obj and optionally split it by material slot.

    Returns list of (tri_mesh, material_slot_index, material_name).
    The caller MUST call bpy.data.meshes.remove(tri_mesh) for every mesh
    returned when done with it.

    When split_by_material=False or the mesh has <=1 material, a single-element
    list is returned.  Empty material slices are silently skipped.
    """
    mat_count = len(obj.data.materials)
    src_verts = len(obj.data.vertices)
    src_faces = len(obj.data.polygons)
    TraceLogger.kvblock(
        f"_triangulate_and_split_by_material  obj='{obj.name}'",
        [
            ("split_by_material", bool(split_by_material)),
            ("material_count",    mat_count),
            ("src_verts",         src_verts),
            ("src_faces",         src_faces),
            ("material_slots",    [m.name if m else None
                                   for m in obj.data.materials]),
        ],
        tier="DEBUG",
        event="split_entry")

    # -- Single-mesh path -------------------------------------------------
    if not split_by_material or mat_count <= 1:
        bm = bmesh.new()
        bm.from_mesh(obj.data)
        bmesh.ops.triangulate(bm, faces=bm.faces)
        tri = bpy.data.meshes.new("_xbg_tmp_single")
        bm.to_mesh(tri)
        bm.free()
        before_v = len(tri.vertices)
        before_f = len(tri.polygons)
        _split_uv_seams(tri)
        after_v = len(tri.vertices)
        after_f = len(tri.polygons)
        mat_name = (obj.data.materials[0].name
                    if mat_count > 0 and obj.data.materials[0]
                    else "Default")
        TraceLogger.kvblock(
            "  single-slice (no split)",
            [
                ("material",       mat_name),
                ("verts_after_tri", before_v),
                ("faces_after_tri", before_f),
                ("verts_after_uv_seam_split", after_v),
                ("faces_after_uv_seam_split", after_f),
                ("verts_added_by_seam_split", after_v - before_v),
            ],
            tier="DEBUG",
            event="split_single_done")
        return [(tri, 0, mat_name)]

    # -- Per-material split -----------------------------------------------
    results = []
    bm_orig = bmesh.new()
    bm_orig.from_mesh(obj.data)
    bmesh.ops.triangulate(bm_orig, faces=bm_orig.faces)
    rows = []

    for mat_idx in range(mat_count):
        bm_copy = bm_orig.copy()

        # Delete faces that don't belong to this material slot
        faces_total = len(bm_copy.faces)
        faces_for_mat = sum(1 for f in bm_copy.faces if f.material_index == mat_idx)
        del_faces = [f for f in bm_copy.faces if f.material_index != mat_idx]
        if del_faces:
            bmesh.ops.delete(bm_copy, geom=del_faces, context='FACES')
        if not bm_copy.faces:
            mat = obj.data.materials[mat_idx]
            mat_name = mat.name if mat else f"Material_{mat_idx}"
            rows.append((mat_idx, mat_name, faces_total, faces_for_mat,
                         "<dropped: no faces>"))
            bm_copy.free()
            continue

        # Clean up isolated vertices left after face deletion
        verts_before_clean = len(bm_copy.verts)
        del_verts = [v for v in bm_copy.verts if not v.link_faces]
        if del_verts:
            bmesh.ops.delete(bm_copy, geom=del_verts, context='VERTS')
        verts_after_clean = len(bm_copy.verts)

        tri = bpy.data.meshes.new(f"_xbg_tmp_mat{mat_idx}")
        bm_copy.to_mesh(tri)
        bm_copy.free()
        before_v = len(tri.vertices)
        before_f = len(tri.polygons)
        _split_uv_seams(tri)
        after_v = len(tri.vertices)

        mat = obj.data.materials[mat_idx]
        mat_name = mat.name if mat else f"Material_{mat_idx}"
        rows.append((mat_idx, mat_name, faces_total, faces_for_mat,
                     f"v={before_v}→{after_v} f={before_f}"))
        results.append((tri, mat_idx, mat_name))

    bm_orig.free()
    TraceLogger.table(
        "split-by-material slice breakdown",
        ("mat_idx", "material", "tot_faces", "for_mat", "result"),
        rows, tier="DEBUG", event="split_per_mat_done")
    TraceLogger.kv("slices_returned", len(results), tier="DEBUG",
                   event="split_total_returned")
    return results


# Engine submesh limits: the DNKS/SDOL submesh header stores face_count,
# index_count (= face_count*3) and vert_count as uint16. index_count is
# the first to overflow: 65535/3 = 21845 triangles. A submesh past that
# gets its index_count clamped (chunks.py), so the engine issues a draw
# with a wrong index count against a huge buffer -> out-of-bounds VRAM
# read -> GPU driver reset / black screen. So any slice over budget MUST
# be split into multiple submeshes before encoding.
_MAX_SM_FACES = 21000      # < 65535/3, headroom
_MAX_SM_VERTS = 65534
# Absolute format limits — a built submesh must NEVER reach these.
# Used by the final pre-write wall (refuses to write an overflowing file).
_HARD_MAX_IDX  = 65535
_HARD_MAX_VERT = 65535


def _split_mesh_by_face_budget(tri_mesh, base_name):
    """Split a triangulated mesh into <=uint16 submeshes.

    Uses BFS face-adjacency ordering so each submesh is a spatially
    contiguous region of the mesh.  The old greedy index-order split
    walked faces in arbitrary bmesh storage order, which scattered
    topologically-adjacent triangles into different submeshes and
    produced "floating" disconnected chunks in-game.

    BFS ordering guarantees the cut always falls on a connected edge
    loop (the topological boundary between two flood-fill regions),
    so the two halves each look like a coherent piece — no scattered
    triangles.  Boundary vertices encode to the same int16 value in
    both adjacent submeshes (same pos_scale, same float position), so
    there is no positional crack either.

    Returns a list of meshes preserving UV / normal / color / vertex
    layers.  If already within budget, returns [tri_mesh] unchanged.
    Caller owns every returned mesh (must remove via bpy.data.meshes).
    """
    if (len(tri_mesh.polygons) <= _MAX_SM_FACES
            and len(tri_mesh.vertices) <= _MAX_SM_VERTS):
        return [tri_mesh]

    bm_src = bmesh.new()
    bm_src.from_mesh(tri_mesh)
    bm_src.faces.ensure_lookup_table()
    bm_src.edges.ensure_lookup_table()

    n_faces = len(bm_src.faces)

    # Build face-adjacency: two faces are adjacent if they share an edge.
    # This is what lets BFS produce a spatially local traversal order.
    adj = [[] for _ in range(n_faces)]
    for e in bm_src.edges:
        lf = e.link_faces
        if len(lf) == 2:
            a, b = lf[0].index, lf[1].index
            adj[a].append(b)
            adj[b].append(a)

    # BFS traversal → spatially local face ordering.
    # Faces connected by shared edges are visited consecutively, so
    # the subsequent greedy partition cuts along one edge loop rather
    # than scattering isolated faces across multiple submeshes.
    bfs_order = []
    visited = bytearray(n_faces)
    for seed in range(n_faces):
        if visited[seed]:
            continue
        q = deque([seed])
        visited[seed] = 1
        while q:
            fi = q.popleft()
            bfs_order.append(fi)
            for nb in adj[fi]:
                if not visited[nb]:
                    visited[nb] = 1
                    q.append(nb)

    # Greedy partition on the BFS-ordered list, respecting BOTH the
    # face count and the unique-vertex budget.
    groups = []
    cur, cur_v = [], set()
    for fi in bfs_order:
        vids = [v.index for v in bm_src.faces[fi].verts]
        added = sum(1 for x in vids if x not in cur_v)
        if cur and (len(cur) + 1 > _MAX_SM_FACES
                    or len(cur_v) + added > _MAX_SM_VERTS):
            groups.append(set(cur))
            cur, cur_v = [], set()
        cur.append(fi)
        cur_v.update(vids)
    if cur:
        groups.append(set(cur))

    out = []
    for gi, keep in enumerate(groups):
        bm_c = bm_src.copy()
        bm_c.faces.ensure_lookup_table()
        del_f = [f for i, f in enumerate(bm_c.faces) if i not in keep]
        if del_f:
            bmesh.ops.delete(bm_c, geom=del_f, context='FACES')
        del_v = [v for v in bm_c.verts if not v.link_faces]
        if del_v:
            bmesh.ops.delete(bm_c, geom=del_v, context='VERTS')
        m = bpy.data.meshes.new(f"{base_name}_b{gi}")
        bm_c.to_mesh(m)
        bm_c.free()
        out.append(m)

    bm_src.free()
    return out


# ============================================================

def _build_index_buffer(tri_mesh, endian=LE, reverse_winding=False):
    """
    Build a uint16 index buffer from a triangulated mesh.
    Returns (bytes_data, uint16_count).

    `endian` is '<' for PC or '>' for PS3 — the indices are packed in the
    file's native byte order so the GPU can read them directly.

    `reverse_winding=True` swaps every triangle's (a, b, c) → (a, c, b),
    reversing front/back face orientation.  Required when the import flipped
    face winding to match Blender's viewport convention: without reversing
    back on export, every triangle in the file ends up wound opposite to the
    original, which the game's backface culling then hides (and mouth/eye
    interiors, originally inward-facing, become outward-facing and stick
    out of the head as a "stretched teeth" glitch).

    Raises ValueError if any vertex index exceeds 65534 (uint16 cap).
    """
    indices = []
    _n_dropped_nontri = 0
    for poly in tri_mesh.polygons:
        if len(poly.vertices) != 3:
            _n_dropped_nontri += 1
            continue
        a, b, c = poly.vertices
        if a > 65534 or b > 65534 or c > 65534:
            raise ValueError(
                f"Vertex index {max(a,b,c)} exceeds uint16 limit (65534). "
                "XBG index buffers are always uint16.  "
                "Split the mesh into smaller objects (<= 65534 verts each)."
            )
        if reverse_winding:
            indices.extend((a, c, b))
        else:
            indices.extend((a, b, c))

    buf = struct.pack(f'{endian}{len(indices)}H', *indices)

    # ── Trace summary + sample triangles ─────────────────────────────
    _mn = min(indices) if indices else 0
    _mx = max(indices) if indices else 0
    TraceLogger.kvblock(
        f"_build_index_buffer  '{getattr(tri_mesh, 'name', '?')}'",
        [
            ("triangles_in",       len(tri_mesh.polygons)),
            ("triangles_dropped",  _n_dropped_nontri),
            ("indices_written",    len(indices)),
            ("byte_size",          len(buf)),
            ("index_min",          _mn),
            ("index_max",          _mx),
            ("reverse_winding",    bool(reverse_winding)),
            ("first_tri",          tuple(indices[:3])),
            ("last_tri",           tuple(indices[-3:]) if len(indices) >= 3 else ()),
        ],
        tier="DEBUG",
        event="index_buffer_stats")
    if TraceLogger.trace_enabled() and indices:
        # First/last 5 triangles in TRACE
        rows = []
        for ti in range(min(5, len(indices) // 3)):
            o = ti * 3
            rows.append((f"first[{ti}]", indices[o], indices[o+1], indices[o+2]))
        last_n = min(5, len(indices) // 3)
        for ti in range(last_n):
            o = len(indices) - (last_n - ti) * 3
            rows.append((f"last[{ti}]", indices[o], indices[o+1], indices[o+2]))
        TraceLogger.table("index buffer sample triangles",
                          ("label", "a", "b", "c"), rows, tier="TRACE",
                          event="index_buffer_sample")
    return buf, len(indices)

