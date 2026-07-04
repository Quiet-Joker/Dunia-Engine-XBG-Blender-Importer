"""UV layer and face-winding helpers for XBG mesh import.

Centralises three operations that the import pipeline performs on every
mesh:

  * `apply_uv_layer()` — write a list of (u, v) pairs onto a named UV
    layer via `foreach_set` (one C call per layer).  Handles the
    `None`-sentinel convention used by parse_mesh_vertices for "unused"
    UV channels (raw -32768/-32768 markers in the file).
  * `flip_face_winding()` — reverse the winding of every polygon in a
    single mesh datablock.  Used by import to compensate for the XBG
    handedness convention so faces shade correctly in the viewport.
  * `flip_mesh_normals()` — same operation applied to a list of mesh
    objects (used by the post-import "Flip Normals" debug action).

Keep this module the single source of truth for UV layer application.
The importer should never duplicate the foreach_set loop inline.
"""

import bmesh

try:
    from ..Core.debug import VerboseLogger as vlog
except ImportError:
    class vlog:
        @staticmethod
        def log(m): pass


def apply_uv_layer(me, uv_src, name):
    """Create a UV layer named `name` on `me` and fill it from `uv_src`.

    uv_src : list indexed by vertex index.  Each entry is either
             (u, v) or `None` (sentinel for "unused" — the raw file used
             -32768/-32768 to flag this channel as inactive for the vert).

    If every entry is sentinel the layer is not created at all.
    Out-of-range vertex indices and `None` sentinels fall back to (0, 0).

    Uses `foreach_set` for the actual upload (single C call), which is
    significantly faster than per-loop Python assignment on large meshes.
    """
    if not uv_src:
        return
    if all(uv is None for uv in uv_src):
        vlog.log(f"  {name}: all sentinel values, skipping layer")
        return

    try:
        uv_layer = me.uv_layers.new(name=name)

        # Pull loop -> vertex_index in one C call
        loop_vis = [0] * len(me.loops)
        me.loops.foreach_get("vertex_index", loop_vis)

        # Build a flat [u, v, u, v, ...] array; sentinel verts default to (0, 0)
        default = (0.0, 0.0)
        uv_flat = [
            coord
            for vi in loop_vis
            for coord in (uv_src[vi] if vi < len(uv_src) and uv_src[vi] is not None else default)
        ]
        uv_layer.data.foreach_set("uv", uv_flat)
    except Exception as e:
        vlog.log(f"  {name} import failed: {e}")


def flip_face_winding(me):
    """Reverse triangle winding on a single mesh datablock.

    One bmesh round-trip — cheap enough to call per-mesh during import.
    Used to fix XBG's handedness convention so faces shade correctly in
    the viewport without needing per-loop normal flipping.
    """
    bm = bmesh.new()
    bm.from_mesh(me)
    bmesh.ops.reverse_faces(bm, faces=bm.faces[:])
    bm.to_mesh(me)
    bm.free()
    me.update()


def flip_mesh_normals(mesh_objects):
    """Reverse triangle winding on every MESH object in the iterable.

    Thin wrapper over `flip_face_winding` for the post-import bulk action.
    """
    for obj in mesh_objects:
        if obj.type != 'MESH':
            continue
        flip_face_winding(obj.data)
