"""Watch Dogs 2 .glm importer (self-contained WD2-owned module).

Split out of the combined Watch_Dogs importer.  WD2 ships its model as a raw
text GEOM source (.glm, "unconverted xbg"): VERSION header, SKELETON (named
bones, parents, axis-angle rotations), GEOMETRY/TRIMESH (indexed verts, faces,
TVERT UVs, BLEND skin links).

Contains the WD2 text parser (parse_wd2_glm) and a WD2-OWNED copy of the
neutral-dict -> Blender builder (build_wd_model), so editing WD2 behaviour can
never affect Watch Dogs 1.  WD2 is import-only (no MAB / inject / skeleton /
HKX — those are WD1 binary-GEOM features).
"""

import os
import re
import math

import numpy as np

try:
    import bpy
    import mathutils
except ImportError:
    bpy = None
    mathutils = None

from ..Core.debug import VerboseLogger as vlog


def parse_wd2_glm(path):
    """Parse a Watch Dogs 2 .glm (text GEOM source) into the model dict."""
    txt = open(path, 'rb').read().decode('latin-1')
    model = {
        'source': 'wd2',
        'name': os.path.splitext(os.path.basename(path))[0],
        'bones': [],
        'meshes': [],
    }
    m = re.search(r'OBJECT_NAME\t"([^"]*)"', txt)
    if m:
        model['name'] = m.group(1)

    # ---- skeleton ----
    skel_m = re.search(r'SKELETON\t\{', txt)
    if skel_m:
        name2idx = {}
        # BONE blocks inside the skeleton are flat (no nested braces)
        for bm in re.finditer(
                r'BONE\t\{\s*'
                r'NAME\t"([^"]*)"\s*'
                r'PARENT\t"([^"]*)"\s*'
                r'POSITION\t(\S+)\t(\S+)\t(\S+)\s*'
                r'ROTATION\t(\S+)\t(\S+)\t(\S+)\t(\S+)\s*'
                r'SCALE\t\S+\t\S+\t\S+\s*\}', txt):
            name, parent = bm.group(1), bm.group(2)
            pos = tuple(float(bm.group(i)) for i in (3, 4, 5))
            ax, ay, az, ang = (float(bm.group(i)) for i in (6, 7, 8, 9))
            n = math.sqrt(ax * ax + ay * ay + az * az)
            if n > 1e-9 and abs(ang) > 1e-9:
                s = math.sin(ang / 2.0)
                quat = (math.cos(ang / 2.0), ax / n * s, ay / n * s, az / n * s)
            else:
                quat = (1.0, 0.0, 0.0, 0.0)
            if parent and parent not in name2idx:
                # referenced-but-undeclared parent (e.g. "Root") — synthesize
                name2idx[parent] = len(model['bones'])
                model['bones'].append({'name': parent, 'parent': -1,
                                       'pos': (0, 0, 0),
                                       'quat': (1.0, 0.0, 0.0, 0.0)})
            name2idx[name] = len(model['bones'])
            model['bones'].append({
                'name': name,
                'parent': name2idx.get(parent, -1),
                'pos': pos, 'quat': quat,
            })

    # ---- material slot names (per-face material ids index this list) ----
    slot_names = re.findall(r'SLOTNAME\t"([^"]*)"', txt)

    # ---- TRIMESH blocks ----
    for tm in re.finditer(r'\tTRIMESH\t\{', txt):
        block = _glm_block(txt, tm.end() - 1)
        mesh = _parse_glm_trimesh(block, slot_names)
        if mesh:
            model['meshes'].append(mesh)
    return model


def _glm_block(txt, open_brace):
    """Return the text of a {...} block given the index of its '{'."""
    depth = 0
    for i in range(open_brace, len(txt)):
        c = txt[i]
        if c == '{':
            depth += 1
        elif c == '}':
            depth -= 1
            if depth == 0:
                return txt[open_brace:i + 1]
    return txt[open_brace:]


def _parse_glm_trimesh(block, slot_names):
    name_m = re.search(r'MESH_NAME\t"([^"]*)"', block)
    name = name_m.group(1) if name_m else 'mesh'

    verts = [tuple(float(x) for x in m.groups())
             for m in re.finditer(
                 r'VERTEX\t\d+\t(\S+)\t(\S+)\t(\S+)', block)]
    if not verts:
        return None
    normals = [tuple(float(x) for x in m.groups())
               for m in re.finditer(
                   r'NORMAL\t\d+\t(\S+)\t(\S+)\t(\S+)', block)]
    # first TVERT channel only
    tverts = []
    tv_m = re.search(r'TVERT_LIST\t\{', block)
    if tv_m:
        tv_block = _glm_block(block, tv_m.end() - 1)
        tverts = [(float(m.group(1)), float(m.group(2)))
                  for m in re.finditer(
                      r'TVERT\t\d+\t(\S+)\t(\S+)', tv_block)]

    nb_uv = 1
    uv_m = re.search(r'NB_UV_CHANNELS\t(\d+)', block)
    if uv_m:
        nb_uv = int(uv_m.group(1))

    tris, loop_normals, loop_uvs, face_mats = [], [], [], []
    for fm in re.finditer(r'FACE\t([\d\t \-]+)', block):
        f = [int(x) for x in fm.group(1).split()]
        # f = idx, v0,v1,v2, n0,n1,n2, nb_uv*3 tvert ids, SMOOTHING_GROUP,
        # MATERIAL.  The LAST field is the material slot (its values range
        # exactly 0..NB_MATERIAL-1 across the file); the second-to-last is
        # the smoothing group (values up to 60+ — reading THAT as the
        # material, as this parser originally did, mis-assigned most faces
        # to slot 0 via the out-of-range fallback).
        if len(f) < 7 + nb_uv * 3 + 2:
            continue
        v = f[1:4]
        n = f[4:7]
        t = f[7:10]                       # channel 0
        mat = f[7 + nb_uv * 3 + 1]
        tris.append(tuple(v))
        for j in range(3):
            loop_normals.append(normals[n[j]] if n[j] < len(normals)
                                else (0.0, 0.0, 1.0))
            if 0 <= t[j] < len(tverts):
                u, vv = tverts[t[j]]
                loop_uvs.append((u, vv))
            else:
                loop_uvs.append((0.0, 0.0))
        face_mats.append(mat if 0 <= mat < len(slot_names) else 0)

    weights = {}
    bl_m = re.search(r'BLEND_LIST\t\{', block)
    if bl_m:
        bl_block = _glm_block(block, bl_m.end() - 1)
        for bv in re.finditer(
                r'BLEND_VERTEX\t\{\s*VERTEX\t(\d+)\s*NB_BONE\t\d+\s*'
                r'((?:BONE_LINK\t"[^"]*"\t\S+\s*)+)\}', bl_block):
            vi = int(bv.group(1))
            wl = [(m.group(1), float(m.group(2)))
                  for m in re.finditer(r'BONE_LINK\t"([^"]*)"\t(\S+)',
                                       bv.group(2))]
            if wl:
                weights[vi] = wl

    return {
        'name': name, 'verts': verts, 'tris': tris,
        'uvs': None, 'loop_uvs': loop_uvs or None,
        'normals': None, 'loop_normals': loop_normals or None,
        'weights': weights, 'material': None,
        'face_materials': face_mats, 'material_slots': slot_names,
    }


def build_wd_model(context, model, import_mesh_only=False):
    """Create armature + meshes from the neutral model dict.  Returns
    (armature_object_or_None, [created mesh objects]).

    `import_mesh_only` skips the armature and skin binding (geometry only)."""
    if bpy is None:
        raise RuntimeError("bpy unavailable")
    Mat = mathutils.Matrix
    Quat = mathutils.Quaternion
    Vec = mathutils.Vector

    mesh_objs = []
    bones = model['bones']
    arm_obj = None
    if bones and not import_mesh_only:
        ad = bpy.data.armatures.new(model['name'] + '_Armature')
        arm_obj = bpy.data.objects.new(ad.name, ad)
        context.collection.objects.link(arm_obj)
        context.view_layer.objects.active = arm_obj
        bpy.ops.object.mode_set(mode='EDIT')
        world = [None] * len(bones)
        ebs = []
        for i, b in enumerate(bones):
            local = (Mat.Translation(Vec(b['pos'])) @
                     Quat(b['quat']).to_matrix().to_4x4())
            p = b['parent']
            world[i] = (world[p] @ local
                        if 0 <= p < i and world[p] is not None else local)
            eb = ad.edit_bones.new(b['name'])
            head = world[i].to_translation()
            eb.head = head
            eb.tail = head + world[i].to_3x3() @ Vec((0.0, 0.05, 0.0))
            ebs.append(eb)
        for i, b in enumerate(bones):
            if 0 <= b['parent'] < i:
                ebs[i].parent = ebs[b['parent']]
        bpy.ops.object.mode_set(mode='OBJECT')

    for mesh in model['meshes']:
        me = bpy.data.meshes.new(mesh['name'])
        me.from_pydata(mesh['verts'], [], mesh['tris'])
        me.update()
        obj = bpy.data.objects.new(mesh['name'], me)
        context.collection.objects.link(obj)
        mesh_objs.append(obj)

        # stamp the WD1 injection layout so the mesh can be edited and
        # written back into the source .xbg (see inject_wd.py)
        inj = mesh.get('inject')
        if inj:
            obj['wd_src'] = inj['src']
            obj['wd_vb_off'] = inj['vb_off']          # offset within buffer 0
            obj['wd_stride'] = inj['stride']
            obj['wd_format'] = inj['format']
            obj['wd_scale'] = inj['scale']
            obj['wd_vcount'] = inj['vcount']
            obj['wd_mesh_index'] = inj['mesh_index']
            obj['wd_buf0_off'] = inj['buf0_off']      # file offset of buffer 0
            n_bufs = len(model['_layout']['buf_frames'])
            if n_bufs > 1:
                obj['wd_multibuffer'] = True          # in-place only; rebuild would corrupt split LOD buffers

        # UVs — bulk foreach_set (loop_uvs direct; per-vert gathered by loop).
        loop_uvs = mesh.get('loop_uvs')
        per_vert_uvs = mesh.get('uvs')
        loop_vi = None
        if loop_uvs or per_vert_uvs or mesh.get('uvs2'):
            loop_vi = np.empty(len(me.loops), dtype=np.intp)
            me.loops.foreach_get('vertex_index', loop_vi)

        def _set_uv(layer_name, loop_data, per_vert_data):
            uvl = me.uv_layers.new(name=layer_name)
            if loop_data:
                flat = np.asarray(loop_data, dtype=np.float64).ravel()
            else:
                flat = np.asarray(per_vert_data, dtype=np.float64)[loop_vi].ravel()
            uvl.data.foreach_set('uv', flat)

        if loop_uvs or per_vert_uvs:
            _set_uv('UVMap', loop_uvs, per_vert_uvs)
        per_vert_uvs2 = mesh.get('uvs2')
        if per_vert_uvs2:
            _set_uv('UVMap1', None, per_vert_uvs2)

        # Vertex colors (authored RGBA — often a shader mask, like Avatar)
        colors = mesh.get('colors')
        if colors:
            ca = me.color_attributes.new('Col', 'BYTE_COLOR', 'POINT')
            ca.data.foreach_set('color', np.asarray(colors, dtype=np.float64).ravel())

        # Normals — keep the authored vectors AND mirror them into an
        # xbg_normal attribute (Avatar-importer parity, survives edits)
        loop_normals = mesh.get('loop_normals')
        per_vert_normals = mesh.get('normals')
        for poly in me.polygons:
            poly.use_smooth = True
        try:
            if loop_normals and len(loop_normals) == len(me.loops):
                me.normals_split_custom_set(loop_normals)
            elif per_vert_normals:
                me.normals_split_custom_set_from_vertices(per_vert_normals)
            # Blender <= 4.0: custom split normals only display with
            # auto-smooth enabled (removed in 4.1+, hence the guard).
            if hasattr(me, 'use_auto_smooth'):
                me.use_auto_smooth = True
        except Exception as exc:
            vlog.log("  [wd] custom normals failed on %s: %s"
                     % (mesh['name'], exc))
        if per_vert_normals:
            na = me.attributes.new('xbg_normal', 'FLOAT_VECTOR', 'POINT')
            na.data.foreach_set(
                'vector', [c for n in per_vert_normals for c in n])

        # Tangent / binormal frames (Avatar attribute names for round-trip)
        for vec_key, w_key, vec_attr, w_attr in (
                ('tangents', 'tangents_w', 'xbg_tangent', 'xbg_tangent_w'),
                ('binormals', 'binormals_w', 'xbg_binormal', 'xbg_binormal_w')):
            vecs = mesh.get(vec_key)
            if not vecs:
                continue
            va = me.attributes.new(vec_attr, 'FLOAT_VECTOR', 'POINT')
            va.data.foreach_set('vector', [c for v in vecs for c in v])
            ws = mesh.get(w_key)
            if ws:
                wa = me.attributes.new(w_attr, 'FLOAT', 'POINT')
                wa.data.foreach_set('value', ws)

        # Materials
        slots = mesh.get('material_slots')
        if slots:
            for sn in slots:
                mat = (bpy.data.materials.get(sn)
                       or bpy.data.materials.new(sn))
                me.materials.append(mat)
            fmats = mesh.get('face_materials') or []
            for pi, poly in enumerate(me.polygons):
                if pi < len(fmats):
                    poly.material_index = fmats[pi]
        elif mesh.get('material'):
            mat = (bpy.data.materials.get(mesh['material'])
                   or bpy.data.materials.new(mesh['material']))
            me.materials.append(mat)

        # Skin weights
        if mesh['weights'] and arm_obj:
            groups = {}
            for vi, wl in mesh['weights'].items():
                for nm, w in wl:
                    g = groups.get(nm)
                    if g is None:
                        g = groups[nm] = obj.vertex_groups.new(name=nm)
                    g.add([vi], w, 'REPLACE')
            obj.parent = arm_obj
            mod = obj.modifiers.new('Armature', 'ARMATURE')
            mod.object = arm_obj

    return arm_obj, mesh_objs


def load_wd2_model(context, filepath, separate_primitives=True):
    """Parse a WD2 .glm and build it in Blender.  Returns (model, armature)."""
    head = open(filepath, 'rb').read(8)
    if head[:7] != b'VERSION':
        raise ValueError(
            "not a Watch Dogs 2 .glm text GEOM file (header %r)" % head[:7])
    model = parse_wd2_glm(filepath)
    arm, mesh_objs = (build_wd_model(context, model) if bpy else (None, []))

    # Export metadata: which source .glm and which TRIMESH (file order) each
    # object came from — export_wd2.export_wd2_glm reads these back.
    for mi, obj in enumerate(mesh_objs):
        obj['wd2_src'] = filepath
        obj['wd2_mesh_index'] = mi

    # Avatar-parity: join submeshes into one object when separate prims is OFF.
    if bpy is not None and not separate_primitives and len(mesh_objs) > 1:
        bpy.ops.object.select_all(action='DESELECT')
        for o in mesh_objs:
            o.select_set(True)
        context.view_layer.objects.active = mesh_objs[0]
        # join() leaves the absorbed objects' mesh datablocks orphaned — purge.
        victim_meshes = [o.data for o in mesh_objs[1:]]
        bpy.ops.object.join()
        joined = context.active_object
        joined.name = model['name']
        joined['wd_joined'] = True
        for m in victim_meshes:
            if m.users == 0:
                bpy.data.meshes.remove(m)
    return model, arm
