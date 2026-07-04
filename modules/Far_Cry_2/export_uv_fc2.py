"""UV-seam splitting for XBG mesh EXPORT (inject side).

The inject counterpart to import_uv.py. Before vertex encoding the injector must
split mesh vertices along UV-island borders (and authored hard edges) so each
vertex carries a SINGLE UV per layer — an XBG vertex stores one UV/normal, and
`export_mesh._encode_vertices` averages a Blender vertex's per-loop UVs.

BYTE-EXACT ROUND-TRIP SAFE: stock imported geometry already has its UV islands
on distinct vertices, so no shared edge has differing UVs → the seam set is
empty → the mesh is returned untouched (to_mesh is never called). Only foreign /
edited meshes with genuine shared-edge UV seams are modified.
"""

import bmesh


def _split_uv_seams(me):
    """Split mesh vertices along UV-island borders so each vertex has a
    single UV per layer.

    Why: an XBG vertex stores ONE UV/normal; `_encode_vertices`
    AVERAGES a Blender vertex's per-loop UVs. For *imported* geometry
    that's harmless (the importer already emits one separate vertex per
    XBG vertex, so every vertex's loops are uniform). For *foreign /
    edited* meshes a real UV-seam vertex carries divergent loop UVs →
    the average lands between the two islands → stretched faces along
    the seam (visible in-game and on re-import). Duplicating the seam
    vertices makes each copy single-UV, so the average becomes the
    island's own UV (identity) and the unwrap is preserved exactly.

    Also splits Sharp-marked edges (`edge.smooth == False`) so authored
    hard edges keep their normals instead of being averaged smooth.
    Still no-op on stock: the importer marks every polygon smooth.

    Limitation: a hard edge expressed ONLY via custom split normals
    (not a Sharp mark and not a UV seam) is still averaged. Mark such
    edges Sharp in Blender to preserve them.
    """
    if not me.uv_layers:
        return
    bm = bmesh.new()
    try:
        bm.from_mesh(me)
        uv_layers = list(bm.loops.layers.uv.values())
        if not uv_layers:
            return
        eps2 = 1e-10
        seam = set()
        for e in bm.edges:
            lf = e.link_faces
            if len(lf) != 2:
                continue
            # Sharp-marked edge = an authored HARD edge. The encoder
            # averages per-vertex normals, which would smooth a hard
            # edge away; split it so each side keeps its own normal.
            # No-op on stock geometry: the importer marks every polygon
            # smooth (no sharp edges), so byte-exact round-trip holds.
            if not e.smooth:
                seam.add(e)
                continue
            f0, f1 = lf[0], lf[1]
            split_this = False
            for v in e.verts:
                l0 = next((l for l in f0.loops if l.vert is v), None)
                l1 = next((l for l in f1.loops if l.vert is v), None)
                if l0 is None or l1 is None:
                    continue
                for uvl in uv_layers:
                    a = l0[uvl].uv
                    b = l1[uvl].uv
                    dx = a.x - b.x
                    dy = a.y - b.y
                    if dx * dx + dy * dy > eps2:
                        split_this = True
                        break
                if split_this:
                    break
            if split_this:
                seam.add(e)
        if not seam:
            return                      # no-op: stock geometry untouched
        bmesh.ops.split_edges(bm, edges=list(seam))
        bm.to_mesh(me)
    finally:
        bm.free()
