"""Vertex color (RGBA) handling for XBG meshes.

Import side: `apply_vertex_colors()` writes the incoming uint8 RGBA list
onto a `BYTE_COLOR` Blender color attribute via `foreach_set` (single
C call per mesh).

Inject side: `build_vertex_color_map()` reads the active color attribute
back as `{vertex_index: (r, g, b, a)}` floats, averaging across all loops
that share each vertex.  Used by the encoder when
`inject_vertex_colors=True` to write the COLOR component of the XBG
vertex buffer.

Single source of truth for vertex-color logic — neither the importer
nor the injector should duplicate these loops inline.
"""

try:
    from ..Core.debug import VerboseLogger as vlog
except ImportError:
    class vlog:
        @staticmethod
        def log(m): pass


def apply_vertex_colors(me, col_src):
    """Apply RGBA vertex colors to `me` from a list of uint8 tuples.

    `col_src` is a list indexed by vertex; each entry is `(r, g, b, a)`
    with channels in [0..255].

    CRITICAL: the Col attribute is created EVEN WHEN col_src is empty.
    The Avatar shaders (aaa.fx / flesh.fx) use mesh vertex color as the
    `mask` input that gates tattoo/specular/normal blend factors:

        mask = saturate(input.vertexMask);
        diffuseColor = lerp(skin, tattoo × DiffuseColor2, mask.g);

    The game's character artists paint mask.g = 0 on every vert that
    shouldn't have a tattoo.  In Blender, if the 'Col' attribute is
    MISSING, ShaderNodeVertexColor returns its default (1, 1, 1, 1) →
    mask.g = 1 → tattoo blend dominates → model renders pure white
    (because the stock empty `tattoo.xbt` is 4x4 white and DiffuseColor2
    is ~(1, 1, 1)).  Creating the attribute with (0, 0, 0, 1) defaults
    makes the chain produce the same "no tattoo" result as the game.

    `foreach_set` is the single-C-call bulk upload for performance.
    """
    n_verts = len(me.vertices)
    if n_verts == 0:
        return
    try:
        # Don't double-create if a previous import already added 'Col'.
        if "Col" in me.color_attributes:
            return
        attr = me.color_attributes.new(name="Col", type='BYTE_COLOR', domain='POINT')
        if col_src:
            # Real per-vertex data from the XBG file.
            colors_flat = [c / 255.0 for rgba in col_src for c in rgba]
        else:
            # No COLOR component in the source — fill with (0, 0, 0, 1)
            # so the shader chain reads "no mask effect" instead of
            # Blender's missing-layer (1, 1, 1, 1) default.
            colors_flat = [0.0, 0.0, 0.0, 1.0] * n_verts
            vlog.log(
                f"  Vertex color: source had no COLOR data; created "
                f"'Col' attribute with (0,0,0,1) defaults so the shader "
                f"chain doesn't pick up Blender's missing-layer white "
                f"that would force the tattoo blend to dominate.")
        attr.data.foreach_set("color", colors_flat)
    except Exception as e:
        vlog.log(f"  Vertex color import failed: {e}")


def build_vertex_color_map(tri_mesh):
    """Read the active vertex color layer as `{vert_index: (r, g, b, a)}`.

    All channels are floats in [0..1].  Multiple loops referencing the
    same vertex are averaged.  Returns `{}` when no color layer is
    present, so the caller can use truthiness to detect "no colors here".

    Supports both the modern `color_attributes` API (Blender 3.2+) and
    the legacy `vertex_colors`.
    """
    # IMPORTANT: bmesh (used by the inject split) preserves the color DATA
    # layer but NOT the "active color" flag, so `active_color` is often None on
    # a split copy even though 'Col' is present -> reading only `active_color`
    # returns {} and the host's authored mask gets wiped. Fall back to the
    # importer's 'Col' by name, then to the first color attribute.
    col_attr = None
    if hasattr(tri_mesh, 'color_attributes') and len(tri_mesh.color_attributes):
        ca = tri_mesh.color_attributes
        col_attr = ca.active_color or ca.get('Col') or ca[0]
    elif hasattr(tri_mesh, 'vertex_colors') and tri_mesh.vertex_colors:
        col_attr = tri_mesh.vertex_colors.active

    if col_attr is None:
        return {}

    data_items = getattr(col_attr, 'data', None)
    if data_items is None:
        return {}

    # CRITICAL — index by the attribute's DOMAIN, not blindly by loop.
    # 'Col' is created POINT-domain by apply_vertex_colors() (one entry per
    # VERTEX), but the legacy `vertex_colors.new()` path and user-made layers
    # are CORNER-domain (one entry per LOOP). Indexing POINT data with loop
    # indices reads the wrong element — and runs off the end (loops > verts) so
    # most reads hit the IndexError -> white default — scrambling the authored
    # aaa.fx mask into the "weird dark paint" patches across the body. This was
    # latent until active_color fell back to 'Col': before, active_color was
    # None on a bmesh-split copy, so the function returned {} and never read.
    domain = getattr(col_attr, 'domain', 'CORNER')

    if domain == 'POINT':
        # One color per vertex — read directly, no per-loop averaging.
        out = {}
        for vi in range(len(data_items)):
            c = data_items[vi].color
            a = float(c[3]) if len(c) > 3 else 1.0
            out[vi] = (float(c[0]), float(c[1]), float(c[2]), a)
        return out

    # CORNER domain — one entry per loop; average loops that share a vertex.
    accum = {}
    for poly in tri_mesh.polygons:
        for li in poly.loop_indices:
            vi = tri_mesh.loops[li].vertex_index
            try:
                c = data_items[li].color
                r, g, b = float(c[0]), float(c[1]), float(c[2])
                a = float(c[3]) if len(c) > 3 else 1.0
            except Exception:
                r, g, b, a = 1.0, 1.0, 1.0, 1.0
            accum.setdefault(vi, []).append((r, g, b, a))

    return {
        vi: tuple(sum(ch[i] for ch in cols) / len(cols) for i in range(4))
        for vi, cols in accum.items()
    }
