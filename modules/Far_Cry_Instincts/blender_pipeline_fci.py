"""Far Cry Instincts -> Blender import pipeline.

Builds a mesh object with geometry, per-submesh min-max-normalized UVs, and
one textured material slot per distinct referenced .xbt (no normals or
skeleton yet). Texture lookup order: the fci_data_folder addon preference
(whole-tree search), an auto-detected extracted-dump root from the .xbg's
own path, then same-directory fallback; untextured if none resolve.
"""
import os

import bpy

from . import import_xbg_fci
from . import import_xbt_fci


def _get_data_root(ctx):
    try:
        prefs = bpy.context.preferences.addons["xbg-importer"].preferences
        return getattr(prefs, 'fci_data_folder', '') or None
    except Exception:
        return None


def _auto_data_root(filepath, texture_paths):
    """Derive the extracted-archive root from the imported .xbg's OWN path
    when the preference isn't set. A model at
    ``...\\_extracted_named\\Objects_xbox\\Pickups\\KeyCard\\KeyCard.xbg``
    references a texture ``\\Objects_xbox\\Pickups\\KeyCard\\KeyCard.xbt`` --
    both live under the same root, so find the texture path's leading folder
    (e.g. ``Objects_xbox``) in the file's own path and cut there."""
    norm = os.path.normpath(filepath)
    parts = norm.split(os.sep)
    lower = [p.lower() for p in parts]
    for tp in texture_paths or ():
        segs = [s for s in tp.replace('/', '\\').split('\\') if s]
        if not segs:
            continue
        anchor = segs[0].lower()
        if anchor not in lower:
            continue
        i = lower.index(anchor)
        if i == 0:
            continue
        root = os.path.join(parts[0] + os.sep, *parts[1:i])
        if os.path.isdir(root) and import_xbg_fci.resolve_texture_path(root, tp):
            return root
    return None


def _find_next_to_file(filepath, in_game_path):
    """Simple default (no data_root configured): look for a same-named
    texture right next to the imported .xbg itself, case-insensitively.
    Works for a model copied out on its own; won't find a texture that
    actually lives in a different branch of the tree (e.g. a shared
    "_generic_objects" folder) -- for that, set the addon's data-folder
    preference to the FULL extracted root so resolve_texture_path's
    whole-tree search can find it."""
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


def _build_material_for_texture(name, in_game_path, data_root, source_filepath):
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    if not in_game_path:
        return mat
    xbt_path = None
    if data_root:
        xbt_path = import_xbg_fci.resolve_texture_path(data_root, in_game_path)
    if xbt_path is None:
        xbt_path = _find_next_to_file(source_filepath, in_game_path)
    if xbt_path is None:
        return mat
    try:
        img = import_xbt_fci.load_xbt_as_blender_image(xbt_path)
    except Exception:
        img = None
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


def _assign_materials(mesh, model, data_root, source_filepath):
    """One material per distinct texture. Multi-material meshes (vehicles)
    author different submeshes' UVs against DIFFERENT textures (e.g. body
    vs tire vs wheel-inside) -- lumping everything under a single texture
    makes non-body submeshes' UVs look "scattered" against the wrong image,
    since they were never meant to sample it. `model.face_textures` (from
    the heuristic submesh->texture bucketing in import_xbg_fci) gives the
    per-triangle texture when known; falls back to one shared material."""
    if model.face_textures:
        distinct = list(dict.fromkeys(model.face_textures))  # stable order
        slot_of = {}
        for i, tex_path in enumerate(distinct):
            mat = _build_material_for_texture(
                f"FCI_Material_{i}", tex_path, data_root, source_filepath)
            mesh.materials.append(mat)
            slot_of[tex_path] = i
        mesh.polygons.foreach_set(
            'material_index', [slot_of[t] for t in model.face_textures])
        mesh.update()
        return
    first_tex = model.texture_paths[0] if model.texture_paths else None
    mat = _build_material_for_texture("FCI_Material", first_tex, data_root, source_filepath)
    mesh.materials.append(mat)


def import_fci_xbg(ctx, filepath):
    """Parse + build. Returns the created Blender mesh object."""
    model = import_xbg_fci.parse_xbg(filepath)

    name = os.path.splitext(os.path.basename(filepath))[0]
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(model.vertices, [], model.triangles)
    mesh.update()
    # source data is Z-up already (bbox reasoning matched a Z-up character
    # height in testing) -- no coordinate-frame rotation applied yet.

    # UV map (per-vertex UVs -> per-loop) so textures actually project.
    if model.uvs:
        uv_layer = mesh.uv_layers.new(name="UVMap")
        uvs = model.uvs
        for loop in mesh.loops:
            uv_layer.data[loop.index].uv = uvs[loop.vertex_index]

    obj = bpy.data.objects.new(name, mesh)
    ctx.collection.objects.link(obj)

    # 1) explicit preference (the intended way to get FULLY ACCURATE textures
    #    -- point it at the root of a whole fci_extract.py dump so textures
    #    that live in a totally different branch of the tree, e.g. a shared
    #    "_generic_objects" folder, are still found via resolve_texture_path's
    #    whole-tree search); 2) auto-detected root from the .xbg's own path
    #    (works if it happens to sit inside a full extracted tree); 3) if
    #    neither yields a root, _build_material_for_texture itself falls back
    #    to looking right next to the .xbg (the simple no-setup default).
    data_root = _get_data_root(ctx) or _auto_data_root(filepath, model.texture_paths)
    _assign_materials(mesh, model, data_root, filepath)

    obj['xbg_fci_data'] = {
        'filepath': filepath,
        'vertex_count': model.vertex_count,
        'triangle_count': len(model.triangles),
        'scale': model.scale,
        'texture_paths': list(model.texture_paths),
    }

    ctx.view_layer.objects.active = obj
    obj.select_set(True)
    return obj
