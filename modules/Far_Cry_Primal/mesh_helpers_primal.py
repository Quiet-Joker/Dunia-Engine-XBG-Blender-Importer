"""Mesh-building helpers for the Far Cry 4 importer.

These are FC4-OWNED copies of the small generic Blender mesh operations
(split-normal apply, tangent stashing, face-winding flip).  They are
deliberately duplicated from other game folders rather than imported, so
that the Far Cry 4 import path is fully self-contained: editing FC3's
mesh behaviour can never affect Avatar / Far Cry 3 / Watch Dogs, and vice
versa (the one-folder-per-game architecture).
"""

import bmesh
import numpy as np

from ..Core.debug import VerboseLogger


def apply_split_normals(me, vert_normal_list, flip):
    """Store XBG per-vertex normals as Blender custom split normals.

    Smooth shading is enabled on every polygon first — Blender 4.1+/5.0
    requires this for `normals_split_custom_set()` to produce correct
    viewport shading.  Normals are normalised to unit length because the
    XBG decode can yield lengths slightly off 1.0.

    The normal is NEVER negated — reversing the triangle WINDING (done
    separately on import) is what orients faces for Blender; the normal
    itself is intrinsic to the surface and must keep pointing outward.
    The `flip` arg is accepted for call-compat but no longer negates.

    Also stores the per-vertex normal as an `xbg_normal` POINT attribute
    (custom split normals are LOOP data and do NOT survive bmesh, but
    POINT attributes do — so the encoder can read stock-vert normals back).
    """
    if not vert_normal_list:
        return
    # NB: normals are never negated — winding handles orientation.
    nverts = len(me.vertices)

    # Per-vertex normalised normals (vectorised). Missing entries and
    # degenerate (~zero-length) normals fall back to (0,0,1) — identical to
    # the old per-element loop.
    src = np.zeros((nverts, 3), dtype=np.float64)
    m = min(nverts, len(vert_normal_list))
    if m:
        src[:m] = np.array([n[:3] for n in vert_normal_list[:m]], dtype=np.float64)
    lengths = np.sqrt((src * src).sum(axis=1))
    normalized = np.empty((nverts, 3), dtype=np.float64)
    normalized[:] = (0.0, 0.0, 1.0)
    good = lengths > 1e-6
    normalized[good] = src[good] / lengths[good, None]

    # Per-vertex POINT attribute (survives bmesh, unlike custom split normals).
    # Store the RAW authored vector, NOT the normalized one: authored XBG
    # normals are often slightly non-unit (~0.996), and the injector encodes
    # from this attribute — normalizing here shifted every re-encoded normal
    # byte by ±1 and broke byte-exact zero-edit round-trips.
    try:
        na = (me.attributes.get("xbg_normal")
              or me.attributes.new("xbg_normal", 'FLOAT_VECTOR', 'POINT'))
        raw = np.zeros((nverts, 3), dtype=np.float64)
        if m:
            raw[:m] = src[:m]
        na.data.foreach_set("vector", raw.ravel())
    except Exception as exc:
        VerboseLogger.log(f"  xbg_normal POINT-attr store failed: {exc}")

    # Loop normals = gather the per-vertex normalised normal by loop vertex.
    nloops = len(me.loops)
    loop_vi = np.empty(nloops, dtype=np.intp)
    me.loops.foreach_get('vertex_index', loop_vi)
    loop_normals = normalized[loop_vi].tolist()

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

    Each entry is `(x, y, z, sign_byte)` where x/y/z are floats in roughly
    [-1, 1] and `sign_byte` is the uint8 handedness flag.  The injector
    reads these back on export so unchanged vertices round-trip identically.
    """
    n = len(me.vertices)
    if not n:
        return

    # foreach_set in one call beats per-index .data[i] assignment (which was
    # the per-vertex hotspot).  Build full-length (n) flat arrays; entries past
    # the supplied list stay 0 — same as the old loop left them.
    def _store(vec_list, vec_name, w_name):
        va = me.attributes.new(vec_name, 'FLOAT_VECTOR', 'POINT')
        wa = me.attributes.new(w_name,   'FLOAT',        'POINT')
        m = min(n, len(vec_list))
        flat = [0.0] * (n * 3)
        ws   = [0.0] * n
        for i in range(m):
            x, y, z, w = vec_list[i]
            j = i * 3
            flat[j] = x; flat[j + 1] = y; flat[j + 2] = z
            ws[i] = float(w)
        va.data.foreach_set('vector', flat)
        wa.data.foreach_set('value', ws)

    if vert_tangent_list:
        _store(vert_tangent_list, "xbg_tangent", "xbg_tangent_w")
    if vert_binormal_list:
        _store(vert_binormal_list, "xbg_binormal", "xbg_binormal_w")


def flip_face_winding(me):
    """Reverse triangle winding on a single mesh datablock.

    One bmesh round-trip — cheap enough to call per-mesh during import.
    Fixes XBG's handedness convention so faces shade correctly without
    per-loop normal flipping.
    """
    bm = bmesh.new()
    bm.from_mesh(me)
    bmesh.ops.reverse_faces(bm, faces=bm.faces[:])
    bm.to_mesh(me)
    bm.free()
    me.update()
