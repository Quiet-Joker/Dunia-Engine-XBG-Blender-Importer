"""Normal, tangent and binormal handling for XBG meshes.

Import side
-----------
  * `apply_split_normals()` — copies the XBG per-vertex normals onto a
    Blender mesh as custom *split* normals so the viewport renders the
    authored shading rather than face-angle averages.
  * `store_tangent_attributes()` — stashes the raw XBG tangent/binormal
    vectors (plus their handedness sign byte) on the mesh as POINT
    attributes (`xbg_tangent`, `xbg_tangent_w`, `xbg_binormal`,
    `xbg_binormal_w`).  The injector reads these back on export so
    unchanged vertices round-trip byte-identically.

Inject side
-----------
  * `build_tbn_lookups()` — match triangulated-mesh vertices back to the
    source mesh by 3D position (rounded), populating per-vertex lookups
    from the stored XBG attributes.
  * `compute_tangents_from_uvs()` — Blender's calc_tangents() pass for
    new verts (those not matched to a stored XBG tangent).

Resolution order in the encoder (see inject_xbg._encode_vertices):
  stored XBG tangent  →  UV-computed  →  orthogonal floor (never zero)
"""

from ..Core.debug import VerboseLogger


# ============================================================
# Import side — apply XBG normals / TBN onto the Blender mesh
# ============================================================

def apply_split_normals(me, vert_normal_list, flip):
    """Store XBG per-vertex normals as Blender custom split normals.

    Smooth shading is enabled on every polygon first — Blender 4.1+/5.0
    requires this for `normals_split_custom_set()` to produce correct
    viewport shading.  Normals are normalised to unit length because the XBG
    decode can yield lengths slightly off 1.0.

    The normal is NEVER negated. The XBG normal is the outward surface normal;
    reversing the triangle WINDING (done separately on import to match the
    game's opposite front-face convention) is what orients faces for Blender —
    the normal itself is intrinsic to the surface and must keep pointing
    outward. Negating it inverts the shading normal relative to the (correct)
    winding-derived geometry -> the model renders BLACK in Cycles, grey/flat in
    EEVEE, and glossy in Material Preview (deleting the custom normals "fixes"
    it because Blender then falls back to the correct geometric normal). The
    old `flip` negate was a leftover that only looked OK while the decode was
    scrambled; with the D3DCOLOR codec correct it cleanly inverts the normals.
    The `flip` arg is accepted for call-compat but no longer negates.

    ALSO stores the per-vertex normal as an `xbg_normal` POINT attribute.
    Custom split normals are LOOP data and bmesh (from_mesh/to_mesh, used by
    the inject split-by-material) does NOT carry them — so a host re-encoded
    through the split lost its authored normals and got Blender's averaged
    geometric normals. POINT attributes DO survive bmesh (that's why
    `xbg_tangent` works), so the encoder reads `xbg_normal` for stock verts.
    It stores the same (outward, normalised) value as the custom split normal,
    so it's a drop-in replacement for `corner_normals`.
    """
    if not vert_normal_list:
        return
    sign = 1.0   # never negate — see docstring (winding handles orientation)

    # Per-vertex POINT attribute (survives bmesh, unlike custom split normals).
    try:
        nverts = len(me.vertices)
        na = (me.attributes.get("xbg_normal")
              or me.attributes.new("xbg_normal", 'FLOAT_VECTOR', 'POINT'))
        flat = []
        for vi in range(nverts):
            if vi < len(vert_normal_list):
                n = vert_normal_list[vi]
                nx, ny, nz = n[0] * sign, n[1] * sign, n[2] * sign
                length = (nx * nx + ny * ny + nz * nz) ** 0.5
                if length > 1e-6:
                    nx /= length; ny /= length; nz /= length
                else:
                    nx, ny, nz = 0.0, 0.0, 1.0
            else:
                nx, ny, nz = 0.0, 0.0, 1.0
            flat.extend((nx, ny, nz))
        na.data.foreach_set("vector", flat)
    except Exception as exc:
        VerboseLogger.log(f"  xbg_normal POINT-attr store failed: {exc}")

    loop_normals = []
    for poly in me.polygons:
        for li in poly.loop_indices:
            vi = me.loops[li].vertex_index
            if vi < len(vert_normal_list):
                n = vert_normal_list[vi]
                nx, ny, nz = n[0] * sign, n[1] * sign, n[2] * sign
                length = (nx * nx + ny * ny + nz * nz) ** 0.5
                if length > 1e-6:
                    nx /= length; ny /= length; nz /= length
                else:
                    nx, ny, nz = 0.0, 0.0, 1.0   # degenerate fallback
            else:
                nx, ny, nz = 0.0, 0.0, 1.0
            loop_normals.append((nx, ny, nz))

    # Smooth shading must be active before normals_split_custom_set() so
    # Blender 4.1+/5.0 actually displays the stored custom normals.
    for poly in me.polygons:
        poly.use_smooth = True

    try:
        me.use_auto_smooth = True   # no-op in Blender 4.1+ (attribute removed)
    except AttributeError:
        pass

    me.normals_split_custom_set(loop_normals)
    me.update()


def store_tangent_attributes(me, vert_tangent_list, vert_binormal_list):
    """Stash raw XBG tangent / binormal vectors as POINT attributes.

    Each entry in `vert_tangent_list` / `vert_binormal_list` is
    `(x, y, z, sign_byte)` where x/y/z are floats in roughly [-1, 1]
    (decoded from int8/127) and `sign_byte` is the uint8 handedness flag
    (usually 0x80) that the game uses to reconstruct mirrored tangent
    space.  The injector reads these back on export so unchanged
    vertices round-trip byte-identically.
    """
    n = len(me.vertices)
    if not n:
        return

    if vert_tangent_list:
        t_attr  = me.attributes.new("xbg_tangent",   'FLOAT_VECTOR', 'POINT')
        tw_attr = me.attributes.new("xbg_tangent_w", 'FLOAT',        'POINT')
        for i in range(min(n, len(vert_tangent_list))):
            tx, ty, tz, tw = vert_tangent_list[i]
            t_attr.data[i].vector = (tx, ty, tz)
            tw_attr.data[i].value = float(tw)

    if vert_binormal_list:
        b_attr  = me.attributes.new("xbg_binormal",   'FLOAT_VECTOR', 'POINT')
        bw_attr = me.attributes.new("xbg_binormal_w", 'FLOAT',        'POINT')
        for i in range(min(n, len(vert_binormal_list))):
            bx, by, bz, bw = vert_binormal_list[i]
            b_attr.data[i].vector = (bx, by, bz)
            bw_attr.data[i].value = float(bw)


# ============================================================
# Inject side — TBN resolution helpers
# ============================================================

def _read_tbn_attributes(src_mesh, htan, hbin):
    """Internal: return (src_tan, src_tan_w, src_bin, src_bin_w) attribute
    handles or None for each that's missing / disabled."""
    src_tan   = src_mesh.attributes.get("xbg_tangent")    if htan else None
    src_tan_w = src_mesh.attributes.get("xbg_tangent_w")  if htan else None
    src_bin   = src_mesh.attributes.get("xbg_binormal")   if hbin else None
    src_bin_w = src_mesh.attributes.get("xbg_binormal_w") if hbin else None
    return src_tan, src_tan_w, src_bin, src_bin_w


def build_tbn_lookups(src_mesh, tri_mesh, htan, hbin):
    """Build (tan_lookup, bin_lookup) dicts keyed by tri_mesh vertex index.

    After triangulation + material splitting `tri_mesh` has a different
    vertex index space than `src_mesh` (= `obj.data`).  We match by 3D
    position rounded to 5 decimals — positions are preserved exactly
    through bmesh triangulate / split-by-material, so the rounding only
    guards against floating-point noise.

    Vertices that have no position match in the source are simply absent
    from the returned dicts — that is the intended "new geometry"
    signal for the encoder's resolution chain.
    """
    tan_lookup = {}
    bin_lookup = {}

    src_tan, src_tan_w, src_bin, src_bin_w = _read_tbn_attributes(src_mesh, htan, hbin)
    if not (src_tan or src_bin):
        return tan_lookup, bin_lookup

    # Position → source vertex index (rounded for stability)
    pos_to_src = {}
    for si, sv in enumerate(src_mesh.vertices):
        key = (round(sv.co.x, 5), round(sv.co.y, 5), round(sv.co.z, 5))
        pos_to_src[key] = si

    for tv in tri_mesh.vertices:
        key = (round(tv.co.x, 5), round(tv.co.y, 5), round(tv.co.z, 5))
        si = pos_to_src.get(key)
        if si is None:
            continue
        # A NEAR-ZERO stored vector is the "no XBG data" sentinel, NOT a
        # real tangent: when a foreign mesh is joined into an imported
        # object, Blender fills the new verts' xbg_tangent/xbg_binormal
        # POINT attributes with the default zero vector. Inserting those
        # zeros here would make the encoder think the vert already has
        # valid TBN -> it skips nearest-neighbor / UV-computed / the
        # orthogonal floor and ships a zero tangent (broken normal-map
        # lighting -> hard seam lines). Treat zero as ABSENT so the full
        # resolution chain runs for joined geometry.
        if src_tan and si < len(src_tan.data):
            vec = src_tan.data[si].vector
            if vec.x * vec.x + vec.y * vec.y + vec.z * vec.z > 1e-8:
                tw = int(src_tan_w.data[si].value) if src_tan_w else 0x80
                tan_lookup[tv.index] = (vec.x, vec.y, vec.z, tw & 0xFF)
        if src_bin and si < len(src_bin.data):
            vec = src_bin.data[si].vector
            if vec.x * vec.x + vec.y * vec.y + vec.z * vec.z > 1e-8:
                bw = int(src_bin_w.data[si].value) if src_bin_w else 0x80
                bin_lookup[tv.index] = (vec.x, vec.y, vec.z, bw & 0xFF)

    return tan_lookup, bin_lookup


def compute_tangents_from_uvs(tri_mesh, tan_lookup, htan, hbin):
    """Use Blender's `calc_tangents()` to derive per-vertex TBN from UVs.

    Only fills verts NOT already in `tan_lookup`.  Loops sharing a vertex
    are averaged.  Sign byte is set to the conventional 0x80 since the
    UV-derived tangent space has no authored handedness.

    *** Side effect ***: `calc_tangents()` internally calls
    `calc_normals_split()`, which OVERWRITES any custom split normals.
    Callers MUST snapshot the authored XBG normals BEFORE invoking this
    function (the encoder does this via its `nrm_map` pass).
    """
    computed_tan = {}
    computed_bin = {}
    if not (htan or hbin):
        return computed_tan, computed_bin
    if not (tri_mesh.uv_layers and tri_mesh.uv_layers.active):
        return computed_tan, computed_bin

    try:
        tri_mesh.calc_tangents(uvmap=tri_mesh.uv_layers.active.name)
        # Accumulate running sums during the loop walk (instead of storing every
        # loop value and re-summing per vertex) -> same average, fewer calls.
        accum_t = {}   # vi -> [sx, sy, sz, count]
        accum_b = {}
        loops = tri_mesh.loops
        for poly in tri_mesh.polygons:
            for li in poly.loop_indices:
                vi = loops[li].vertex_index
                if vi in tan_lookup:
                    continue
                loop = loops[li]
                t = loop.tangent; b = loop.bitangent
                a = accum_t.get(vi)
                if a is None:
                    accum_t[vi] = [t.x, t.y, t.z, 1]
                    accum_b[vi] = [b.x, b.y, b.z, 1]
                else:
                    a[0] += t.x; a[1] += t.y; a[2] += t.z; a[3] += 1
                    c = accum_b[vi]
                    c[0] += b.x; c[1] += b.y; c[2] += b.z; c[3] += 1

        for vi, a in accum_t.items():
            k = a[3]
            computed_tan[vi] = (a[0] / k, a[1] / k, a[2] / k, 0x80)
        for vi, a in accum_b.items():
            k = a[3]
            computed_bin[vi] = (a[0] / k, a[1] / k, a[2] / k, 0x80)
    except Exception as exc:
        VerboseLogger.log(f"  [inject] WARNING: calc_tangents failed ({exc}); "
                          f"falling back to orthogonal placeholders if enabled")

    return computed_tan, computed_bin
