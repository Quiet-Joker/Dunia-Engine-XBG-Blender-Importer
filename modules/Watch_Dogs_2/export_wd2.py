"""Watch Dogs 2 .glm exporter (text GEOM source writer).

Writes edited Blender meshes back into a copy of the source .glm by
REPLACING only the geometry sub-blocks (VERTEX_LIST / NORMAL_LIST /
TVERT_LIST / FACE_LIST / BLEND_LIST) of each edited TRIMESH — every other
block (materials, skeleton, PROCEDURAL_NODES_LIST, REFLEX_DATA,
SECONDARY_MOTION_*, COLLISION_PRIMITIVE_LIST, CONNECTIVITIES_LIST, ...)
survives byte-for-byte.  Text format = no quantization and no fixed counts,
so vertex/face count changes are fully supported.

Grammar notes (from avatar01_tor_bombercoat01.glm ground truth):
  * FACE line = idx, v0,v1,v2, n0,n1,n2, NB_UV_CHANNELS*3 tvert ids
    (unused channels = -1), SMOOTHING_GROUP, MATERIAL.  (The material is
    the LAST field — its values range exactly 0..NB_MATERIAL-1; the
    second-to-last is the smoothing group, values up to 60+.)
  * NORMAL_LIST / TVERT_LIST are index pools; faces reference them
    per-corner.  The exporter writes one entry per loop (corner) — larger
    than a deduplicated pool but exactly equivalent.
  * TVERT has a third coordinate (always 0).
  * UVs are NOT V-flipped in the file (importer uses them as-is).

Downstream: the user's GLM2XBG converter (WDL_glm.exe) compiles the .glm
back into a binary .xbg.
"""

import os
import re

try:
    import bpy
except ImportError:
    bpy = None

from ..Core.debug import VerboseLogger as vlog


def _glm_block_span(txt, open_brace):
    """(start, end_exclusive) of a {...} block given the index of its '{'."""
    depth = 0
    for i in range(open_brace, len(txt)):
        c = txt[i]
        if c == '{':
            depth += 1
        elif c == '}':
            depth -= 1
            if depth == 0:
                return open_brace, i + 1
    return open_brace, len(txt)


def _find_sub_block(tri_txt, keyword):
    """(start_of_keyword, end_of_block) of `keyword\\t{...}` inside a
    TRIMESH block's text, or None."""
    m = re.search(re.escape(keyword) + r'\t\{', tri_txt)
    if not m:
        return None
    _, end = _glm_block_span(tri_txt, m.end() - 1)
    return m.start(), end


def _extract_mesh(obj):
    """Blender object -> (verts, loop_normals, loop_uvs, faces, weights).

    faces = [(v0,v1,v2, n0,n1,n2, t0,t1,t2, mat)] with n/t ids indexing the
    per-loop normal/uv pools (fan-triangulated for quads/ngons)."""
    depsgraph = bpy.context.evaluated_depsgraph_get()
    eval_obj = obj.evaluated_get(depsgraph)
    me = eval_obj.to_mesh()

    verts = [(v.co.x, v.co.y, v.co.z) for v in me.vertices]

    corner_normals = getattr(me, 'corner_normals', None)
    loop_normals = []
    for li in range(len(me.loops)):
        if corner_normals is not None:
            n = corner_normals[li].vector
        else:
            n = me.loops[li].normal
        loop_normals.append((n.x, n.y, n.z))

    uvl = me.uv_layers.active
    loop_uvs = []
    for li in range(len(me.loops)):
        if uvl:
            uv = uvl.data[li].uv
            loop_uvs.append((uv.x, uv.y))
        else:
            loop_uvs.append((0.0, 0.0))

    faces = []
    for p in me.polygons:
        ls = list(p.loop_indices)
        vs = list(p.vertices)
        mat = p.material_index
        for i in range(1, len(vs) - 1):        # fan triangulation
            faces.append((vs[0], vs[i], vs[i + 1],
                          ls[0], ls[i], ls[i + 1],
                          ls[0], ls[i], ls[i + 1],
                          mat))

    vg_names = {vg.index: vg.name for vg in obj.vertex_groups}
    weights = []
    for v in me.vertices:
        wl = [(vg_names[g.group], g.weight) for g in v.groups
              if g.group in vg_names and g.weight > 0.0]
        wl.sort(key=lambda x: -x[1])
        weights.append(wl)

    eval_obj.to_mesh_clear()
    return verts, loop_normals, loop_uvs, faces, weights


def _fmt(v):
    return "%.8f" % v


def _build_trimesh_blocks(verts, loop_normals, loop_uvs, faces, weights,
                          nb_uv_channels):
    """The replacement sub-block texts (TRIMESH-internal indentation)."""
    L = []
    a = L.append

    a("VERTEX_LIST\t{\n\t\t\tNB_VERTEX\t%d\n" % len(verts))
    for i, (x, y, z) in enumerate(verts):
        a("\t\t\tVERTEX\t%d\t%s\t%s\t%s\n" % (i, _fmt(x), _fmt(y), _fmt(z)))
    a("\t\t}")
    vertex_list = "".join(L)

    L = []
    a = L.append
    a("NORMAL_LIST\t{\n\t\t\tNB_NORMAL\t%d\n" % len(loop_normals))
    for i, (x, y, z) in enumerate(loop_normals):
        a("\t\t\tNORMAL\t%d\t%s\t%s\t%s\n" % (i, _fmt(x), _fmt(y), _fmt(z)))
    a("\t\t}")
    normal_list = "".join(L)

    L = []
    a = L.append
    a("TVERT_LIST\t{\n\t\t\tCHANNEL\t0\n\t\t\tORIGINAL_CHANNEL\t1\n"
      "\t\t\tNB_TVERT\t%d\n" % len(loop_uvs))
    for i, (u, v) in enumerate(loop_uvs):
        a("\t\t\tTVERT\t%d\t%s\t%s\t%s\n" % (i, _fmt(u), _fmt(v), _fmt(0.0)))
    a("\t\t}")
    tvert_list = "".join(L)

    L = []
    a = L.append
    a("FACE_LIST\t{\n\t\t\tNB_FACE\t%d\n\t\t\tNB_UV_CHANNELS\t%d\n"
      % (len(faces), nb_uv_channels))
    unused = "\t-1" * ((nb_uv_channels - 1) * 3)
    for i, f in enumerate(faces):
        v0, v1, v2, n0, n1, n2, t0, t1, t2, mat = f
        a("\t\t\tFACE\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d%s\t1\t%d\n"
          % (i, v0, v1, v2, n0, n1, n2, t0, t1, t2, unused, mat))
    a("\t\t}")
    face_list = "".join(L)

    blend_list = None
    if any(weights):
        L = []
        a = L.append
        a("BLEND_LIST\t{\n")
        for vi, wl in enumerate(weights):
            if not wl:
                continue
            a("\t\t\tBLEND_VERTEX\t{\n\t\t\t\tVERTEX\t%d\n"
              "\t\t\t\tNB_BONE\t%d\n" % (vi, len(wl)))
            for nm, w in wl:
                a("\t\t\t\tBONE_LINK\t\"%s\"\t%s\n" % (nm, _fmt(w)))
            a("\t\t\t}\n")
        a("\t\t}")
        blend_list = "".join(L)

    return {'VERTEX_LIST': vertex_list, 'NORMAL_LIST': normal_list,
            'TVERT_LIST': tvert_list, 'FACE_LIST': face_list,
            'BLEND_LIST': blend_list}


def export_wd2_glm(context, objects, out_path):
    """Write edited WD2 meshes back into a copy of the source .glm.

    Returns (n_meshes, n_verts, warnings)."""
    if bpy is None:
        raise RuntimeError("bpy unavailable")
    tagged = [o for o in objects
              if o.type == 'MESH' and o.get('wd2_src') is not None
              and 'wd2_mesh_index' in o.keys()]
    joined = [o for o in tagged if o.get('wd_joined')]
    if joined:
        raise RuntimeError(
            "%s was imported JOINED (one object for all submeshes) — turn "
            "Separate Primitives ON and re-import to export"
            % joined[0].name)
    if not tagged:
        raise RuntimeError("no WD2-imported meshes selected "
                           "(import a Watch Dogs 2 .glm first)")
    srcs = {o['wd2_src'] for o in tagged}
    if len(srcs) != 1:
        raise RuntimeError("selected meshes come from different .glm files")
    src = next(iter(srcs))
    if not os.path.isfile(src):
        raise RuntimeError("source .glm not found: %s" % src)

    txt = open(src, 'rb').read().decode('latin-1')
    warnings = []

    # TRIMESH spans in file order (absolute offsets into txt)
    spans = []
    for tm in re.finditer(r'\tTRIMESH\t\{', txt):
        s, e = _glm_block_span(txt, tm.end() - 1)
        spans.append((tm.start() + 1, e))   # from the 'TRIMESH' keyword

    by_index = {}
    for ob in tagged:
        mi = int(ob['wd2_mesh_index'])
        if mi in by_index:
            warnings.append("%s: TRIMESH %d already exported by another "
                            "object — skipped" % (ob.name, mi))
            continue
        if mi >= len(spans):
            warnings.append("%s: TRIMESH %d not found in source (%d present)"
                            % (ob.name, mi, len(spans)))
            continue
        by_index[mi] = ob

    n_meshes = n_verts = 0
    # replace back-to-front so earlier spans stay valid
    for mi in sorted(by_index, reverse=True):
        ob = by_index[mi]
        t_start, t_end = spans[mi]
        tri_txt = txt[t_start:t_end]

        nb_uv = 1
        m = re.search(r'NB_UV_CHANNELS\t(\d+)', tri_txt)
        if m:
            nb_uv = int(m.group(1))

        verts, l_nrm, l_uv, faces, weights = _extract_mesh(ob)
        blocks = _build_trimesh_blocks(verts, l_nrm, l_uv, faces, weights,
                                       nb_uv)

        for key in ('BLEND_LIST', 'FACE_LIST', 'TVERT_LIST',
                    'NORMAL_LIST', 'VERTEX_LIST'):    # back-to-front in-file
            new = blocks[key]
            span = _find_sub_block(tri_txt, key)
            if span is None:
                if new is not None and key != 'BLEND_LIST':
                    warnings.append("%s: source TRIMESH has no %s block — "
                                    "left unchanged" % (ob.name, key))
                continue
            if new is None:
                continue                          # keep the original block
            s, e = span
            tri_txt = tri_txt[:s] + new + tri_txt[e:]

        txt = txt[:t_start] + tri_txt + txt[t_end:]
        n_meshes += 1
        n_verts += len(verts)
        vlog.log("  [wd2 export] TRIMESH %d <- '%s' (%d verts, %d faces)"
                 % (mi, ob.name, len(verts), len(faces)))

    if not n_meshes:
        raise RuntimeError("nothing exported — " + "; ".join(warnings))

    with open(out_path, 'wb') as f:
        f.write(txt.encode('latin-1'))
    return n_meshes, n_verts, warnings
