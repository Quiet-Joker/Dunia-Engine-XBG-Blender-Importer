"""Far Cry 1 (.cgf) -> Blender import pipeline.

Textures are already real, uncompressed-header .dds files (no custom decode
needed like FCI's .xbt) -- Blender loads them natively via
`bpy.data.images.load`. Materials/UVs are per-face and per-face-corner (see
import_cgf_fc1.py docstring); geometry uses shared position/normal vertices
with a SEPARATE per-face-corner UV index (CryTexFace), which maps directly
onto Blender's per-loop UV model -- no vertex duplication needed.
"""
import os

import bpy
import mathutils

from . import import_cgf_fc1


def _get_data_root(ctx):
    try:
        prefs = bpy.context.preferences.addons["xbg-importer"].preferences
        return getattr(prefs, 'fc1_data_folder', '') or None
    except Exception:
        return None


def _walk_case_insensitive(root, parts):
    cur = root
    for i, part in enumerate(parts):
        if not os.path.isdir(cur):
            return None
        target = part.lower()
        match = None
        try:
            for name in os.listdir(cur):
                if name.lower() == target:
                    match = name
                    break
        except OSError:
            return None
        if match is None:
            return None
        cur = os.path.join(cur, match)
    return cur if os.path.isfile(cur) else None


_FILENAME_INDEX_CACHE = {}


def _filename_index(root):
    key = os.path.normcase(os.path.abspath(root))
    idx = _FILENAME_INDEX_CACHE.get(key)
    if idx is not None:
        return idx
    idx = {}
    for dirpath, _dirs, files in os.walk(root):
        for fn in files:
            idx.setdefault(fn.lower(), []).append(os.path.join(dirpath, fn))
    _FILENAME_INDEX_CACHE[key] = idx
    return idx


def resolve_texture_path(data_root, in_game_path):
    """Same 3-tier strategy as FCI's resolve_texture_path: exact join,
    case-insensitive per-segment walk, whole-tree filename index fallback."""
    if not data_root or not in_game_path:
        return None
    parts = [p for p in in_game_path.replace('/', '\\').split('\\') if p]
    if not parts:
        return None
    candidate = os.path.join(data_root, *parts)
    if os.path.isfile(candidate):
        return candidate
    hit = _walk_case_insensitive(data_root, parts)
    if hit:
        return hit
    idx = _filename_index(data_root)
    matches = idx.get(parts[-1].lower())
    if not matches:
        return None
    if len(matches) == 1:
        return matches[0]
    wanted_dirs = [p.lower() for p in parts[:-1]]

    def score(path):
        got_dirs = [p.lower() for p in os.path.normpath(path).split(os.sep)[:-1]]
        s = 0
        for a, b in zip(reversed(wanted_dirs), reversed(got_dirs)):
            if a != b:
                break
            s += 1
        return s
    return max(matches, key=score)


def _find_next_to_file(filepath, in_game_path):
    basename = os.path.basename(in_game_path.replace('/', '\\').rstrip('\\'))
    if not basename:
        return None
    d = os.path.dirname(filepath)
    target = basename.lower()
    try:
        for name in os.listdir(d):
            if name.lower() == target:
                return os.path.join(d, name)
    except OSError:
        pass
    return None


_IMAGE_CACHE = {}


def _load_image(path):
    key = os.path.normcase(os.path.abspath(path))
    img = _IMAGE_CACHE.get(key)
    if img is not None and img.name in bpy.data.images:
        return img
    try:
        img = bpy.data.images.load(path, check_existing=True)
    except Exception:
        return None
    _IMAGE_CACHE[key] = img
    return img


def _build_material_for(mat_name, diffuse_in_game_path, data_root, source_filepath):
    mat = bpy.data.materials.new(name=mat_name)
    mat.use_nodes = True
    if not diffuse_in_game_path:
        return mat
    tex_path = None
    if data_root:
        tex_path = resolve_texture_path(data_root, diffuse_in_game_path)
    if tex_path is None:
        tex_path = _find_next_to_file(source_filepath, diffuse_in_game_path)
    if tex_path is None:
        return mat
    img = _load_image(tex_path)
    if img is None:
        return mat
    nt = mat.node_tree
    bsdf = next((n for n in nt.nodes if n.type == 'BSDF_PRINCIPLED'), None)
    if bsdf is None:
        return mat
    tex_node = nt.nodes.new('ShaderNodeTexImage')
    tex_node.image = img
    nt.links.new(tex_node.outputs['Color'], bsdf.inputs['Base Color'])
    return mat


def _build_mesh_object(ctx, model, node, mesh, data_root):
    name = node.name.replace('\\', '_').replace('/', '_') or f"cgf_mesh_{mesh.chunk_id}"
    bl_mesh = bpy.data.meshes.new(name)
    bl_mesh.from_pydata(mesh.vertices, [], mesh.faces_v)
    bl_mesh.update(calc_edges=True)

    # per-vertex normals (custom split normals) straight from the file
    if mesh.normals and len(mesh.normals) == len(mesh.vertices):
        try:
            bl_mesh.normals_split_custom_set_from_vertices(mesh.normals)
        except Exception:
            pass

    # per-face-corner UVs via CryTexFace indices into the separate UV pool
    # (NOT the position-vertex indices -- classic split vert/uv topology).
    if mesh.uvs and mesh.texfaces and len(mesh.texfaces) == len(bl_mesh.polygons):
        uv_layer = bl_mesh.uv_layers.new(name="UVMap")
        for poly in bl_mesh.polygons:
            t0, t1, t2 = mesh.texfaces[poly.index]
            for loop_idx, tvi in zip(poly.loop_indices, (t0, t1, t2)):
                u, v = mesh.uvs[tvi]
                uv_layer.data[loop_idx].uv = (u, v)

    # materials: one Blender slot per resolved child material (by per-face
    # MatID -> node_face_materials[node.chunk_id][MatID]); faces whose
    # MatID is out of range (shouldn't happen on clean files) fall back to
    # slot 0.
    face_mats = model.node_face_materials.get(node.chunk_id) or []
    if face_mats:
        slot_names = []
        for i, mat_entry in enumerate(face_mats):
            mat_name = f"{name}_mat{i}_{mat_entry.name}"[:63]
            bl_mat = _build_material_for(
                mat_name, mat_entry.diffuse_path, data_root, model.filepath)
            bl_mesh.materials.append(bl_mat)
            slot_names.append(mat_name)
        indices = []
        for matid in mesh.faces_matid:
            idx = matid if 0 <= matid < len(face_mats) else 0
            indices.append(idx)
        bl_mesh.polygons.foreach_set('material_index', indices)
        bl_mesh.update()

    obj = bpy.data.objects.new(name, bl_mesh)
    ctx.collection.objects.link(obj)

    obj.location = node.pos
    obj.rotation_mode = 'QUATERNION'
    rx, ry, rz, rw = node.rot
    obj.rotation_quaternion = mathutils.Quaternion((rw, rx, ry, rz))
    obj.scale = node.scl

    obj['xbg_fc1_data'] = {
        'filepath': model.filepath,
        'node_name': node.name,
        'vertex_count': len(mesh.vertices),
        'triangle_count': len(mesh.faces_v),
        'material_count': len(face_mats),
    }
    return obj


def import_fc1_cgf(ctx, filepath):
    """Parse + build. Returns a list of created Blender mesh objects (one
    per Node chunk that has an associated Mesh -- most files have exactly
    one, but multi-node scenes like architectural set pieces have several,
    each independently positioned via its own node transform)."""
    model = import_cgf_fc1.parse_cgf(filepath)
    data_root = _get_data_root(ctx)

    objs = []
    node_by_id = {n.chunk_id: n for n in model.nodes}
    obj_by_node_id = {}
    for node in model.nodes:
        mesh = model.meshes.get(node.object_id)
        if mesh is None:
            continue
        obj = _build_mesh_object(ctx, model, node, mesh, data_root)
        objs.append(obj)
        obj_by_node_id[node.chunk_id] = obj

    # parent hierarchy (ParentID references another Node's chunk ID)
    for node in model.nodes:
        obj = obj_by_node_id.get(node.chunk_id)
        parent_obj = obj_by_node_id.get(node.parent_id)
        if obj is not None and parent_obj is not None:
            obj.parent = parent_obj

    if not objs:
        raise import_cgf_fc1.FCGFParseError(
            "No renderable Node+Mesh pairs found in this .cgf")

    for obj in objs:
        obj.select_set(True)
    ctx.view_layer.objects.active = objs[0]
    return objs
