"""Far Cry 3 / Far Cry 4 → Blender import pipeline.

Extracted verbatim from the Avatar `import_xbg.py` (where it had been
co-located) into the Far Cry 3 folder so the FC3/FC4 mesh-import path is
fully self-contained and Avatar no longer depends on it.

`_load_fc3_or_fc4()` parses an FC3/FC4 .xbg via this folder's
`import_xbg_fc3.parse_xbg()` and builds the Blender meshes + armature.
The small mesh helpers (`apply_split_normals`, `store_tangent_attributes`,
`flip_face_winding`) are this folder's OWN copies (mesh_helpers_fc3) — no
imports from any other game folder.
"""

import os
import struct
import math

import bpy
import mathutils
from mathutils import Vector, Matrix, Quaternion

from .binary_primal import LE
from ..Core.debug import VerboseLogger as vlog
from . import import_xbg_primal as import_xbg_fc3  # FC4-owned parser
from .mesh_helpers_primal import (apply_split_normals, store_tangent_attributes,
                               flip_face_winding)


# Version markers in the .xbg header (u32 at offset 4, after the 'HSEM' magic).
_VERSION_FC3 = 0x00030034   # Far Cry 3 (2012)
_VERSION_FC4 = 0x00060037   # Far Cry 4 (2014)
_VERSION_PRIMAL = 0x0006003A  # Far Cry Primal (2016) — this module's target


def detect_fc_version(filepath):
    """Return the u32 version marker for an FC3/FC4 .xbg, or None.

    FC3-owned copy of the header probe (no avatar dependency).
    """
    try:
        with open(filepath, 'rb') as f:
            head = f.read(8)
    except OSError:
        return None
    if len(head) < 8 or head[:4] != b'HSEM':
        return None
    return struct.unpack_from('<I', head, 4)[0]


def _load_fc3_or_fc4(ctx, fp, version, game_name, lod, separate, lhd):
    """Import an FC3 or FC4 .xbg into Blender.

    Differences from the Avatar/FC2 path:
      * Skeleton: EDON bone hierarchy is now decoded — we build an
        Edit-Bones armature from the parent indices + translations.
        Bone rotation_raw is preserved verbatim on each Blender bone
        as a custom property since the 4-float layout doesn't always
        normalize to a unit quaternion (TBD decoding).
      * Materials are listed by reference path only (FC3+ uses
        external `.material.bin` files in Ubisoft's TAM/FCB binary
        format — different from XBM and not yet parsed).  Each
        material becomes a named Blender material slot for later
        manual wiring.
      * Only the requested LOD is imported (default LOD 0).
    """
    vlog.log(f"\n{'='*60}")
    vlog.log(f"PARSING {game_name.upper()} XBG FILE: {os.path.basename(fp)}")
    vlog.log(f"{'='*60}")
    try:
        result = import_xbg_fc3.parse_xbg(fp)
    except Exception as exc:
        raise RuntimeError(
            f"{game_name} XBG parse failed: {exc}.  See XBG_FC3_FORMAT.md "
            f"for the parser's known limits."
        ) from exc

    materials = result['materials']
    lods = result['lods']
    bones = result.get('bones', [])
    skinning = result.get('skinning')

    # ── File header summary ───────────────────────────────────────
    vlog.log(f"\nFile Header:")
    vlog.log(f"  Game:        {game_name}")
    vlog.log(f"  Version:     0x{result['version']:08x}")
    vlog.log(f"  File size:   {result['file_size']:,} bytes")
    vlog.log(f"  Chunks:      {len(result.get('chunks', []))}")

    # ── Chunk-by-chunk dump (matches Avatar log_chunk format) ─────
    for ck_name, ck_off, ck_size, ck_dsize in result.get('chunks', []):
        vlog.log_chunk(ck_name, ck_off, ck_size)

    # ── LTMR materials ────────────────────────────────────────────
    vlog.log(f"\n=== LTMR CHUNK (Materials) ===")
    vlog.log(f"Material Count: {len(materials)}")
    for i, (mat_name, mat_path) in enumerate(materials):
        vlog.log_material(i, mat_name, mat_path)

    # ── EDON bones — full skeletal log (sampled to keep size sane) ─
    if bones:
        vlog.log(f"\n=== EDON CHUNK (Bones) ===")
        vlog.log(f"Bone Count: {len(bones)}")
        # Use the same rotation-aware world-matrix calculation that the
        # armature builder uses, so the logged world positions match
        # what Blender will actually show.
        from mathutils import Vector, Quaternion, Matrix
        world_mats = [Matrix.Identity(4) for _ in bones]
        for i, b in enumerate(bones):
            t = Vector(b['translation'])
            raw = b['rotation_raw']
            q = Quaternion((raw[0], raw[1], raw[2], raw[3]))
            try:
                q.normalize()
            except Exception:
                q = Quaternion()
            local = Matrix.Translation(t) @ q.to_matrix().to_4x4()
            if b['parent'] == -1 or b['parent'] >= i:
                world_mats[i] = local
            else:
                world_mats[i] = world_mats[b['parent']] @ local
            vlog.log_bone(i, b['name'], b['parent'], b['translation'],
                          b['rotation_raw'])
            vlog.log_bone_world_transform(i, b['name'], world_mats[i].to_translation())
    else:
        vlog.log(f"\n=== EDON CHUNK ===\n  (no bone data)")

    # ── DNKS / SULC skinning structure ────────────────────────────
    if skinning:
        vlog.log(f"\n=== DNKS / SULC CHUNK (Skinning) ===")
        h = skinning.get('header') or {}
        vlog.log(f"  SULC version:    {h.get('version')}")
        vlog.log(f"  count_a:         {h.get('count_a')}")
        vlog.log(f"  count_b1/b2:     {h.get('count_b1')} / {h.get('count_b2')}")
        vlog.log(f"  count_c:         {h.get('count_c')}")
        vlog.log(f"  stride (header): {h.get('stride')}")
        vlog.log(f"  bones_used:      {h.get('bones_used')}")
        vlog.log(f"  sections:        {len(skinning['sections'])}")
        # Sample first 3 sections so the log shows the structure but
        # stays at a reasonable size for large files.
        for i, sec in enumerate(skinning['sections'][:3]):
            idx = sec['indices']
            vlog.log(f"    section[{i}] @0x{sec['offset']:x}  "
                     f"({len(idx)} u16 indices)  first 8 = {idx[:8]}")
        if len(skinning['sections']) > 3:
            vlog.log(f"    … ({len(skinning['sections']) - 3} more sections)")
    else:
        vlog.log(f"\n=== DNKS / SULC CHUNK ===\n  (no skinning data)")

    # ── SDOL LOD/mesh summary ─────────────────────────────────────
    vlog.log(f"\n=== SDOL CHUNK (Mesh LODs) ===")
    vlog.log(f"LOD Count: {len(lods)}")
    for i, l in enumerate(lods):
        v_total = sum(vb['vcount'] for vb in l['vbs'])
        f_total = sum(len(vb['faces']) for vb in l['vbs'])
        strides = sorted({vb['stride'] for vb in l['vbs']})
        vlog.log(f"  LOD[{i}]  distance={l['lod_distance']:>6.2f}  "
                 f"sections={len(l['vbs'])}  strides={strides}  "
                 f"{v_total:>6d} verts  {f_total:>6d} faces")
        for j, vb in enumerate(l['vbs']):
            vlog.log(f"    section[{j}]  flag=0x{vb['flag']:x}  "
                     f"stride={vb['stride']}  vcount={vb['vcount']}  "
                     f"offset={vb['offset']}  faces={len(vb['faces'])}")

    vlog.log(f"\n{'='*60}\nPARSING COMPLETE\n{'='*60}")

    # ── Always update file_info panel (visible without verbose logging) ─
    _lines = [
        f"File: {os.path.basename(fp)}",
        f"Game: {game_name}  (version 0x{result['version']:08x})",
        f"Size: {result['file_size']:,} bytes ({result['file_size'] / 1024:.1f} KB)",
        "",
        f"Chunks: {len(result.get('chunks', []))}",
    ]
    for ck_name, ck_off, ck_size, _ck_d in result.get('chunks', []):
        _lines.append(f"  {ck_name}  @0x{ck_off:x}  ({ck_size} bytes)")
    _lines += ["", f"Materials: {len(materials)}"]
    for i, (mn, _mp) in enumerate(materials):
        _lines.append(f"  [{i}] {mn}")
    _lines += ["", f"Bones: {len(bones)}", f"LODs: {len(lods)}"]
    for i, l in enumerate(lods):
        v = sum(vb['vcount'] for vb in l['vbs'])
        f = sum(len(vb['faces']) for vb in l['vbs'])
        _lines.append(f"  LOD[{i}]  {v} verts  {f} faces  "
                      f"dist={l['lod_distance']:.2f}")
    ctx.scene.xbg_debug_settings.file_info_data = "\n".join(_lines)

    # ── Build armature from EDON bones (with verbose log) ─────────
    armature_obj = None
    if bones:
        armature_obj = _build_fc3_armature(ctx, bones, os.path.basename(fp))
        if armature_obj:
            armature_obj["xbg_source_file"] = fp
        if skinning and armature_obj:
            # Stash raw skinning data on the armature for future
            # decoding.  Full per-vertex bone-weight wiring needs the
            # section-local-index → global-bone mapping which isn't
            # decoded yet (see XBG_FC3_FORMAT.md §6).
            armature_obj['xbg_fc3_sulc_header'] = list(
                skinning['header'].values()) if skinning['header'] else []
            armature_obj['xbg_fc3_sulc_section_count'] = len(skinning['sections'])
            vlog.log(f"  attached SULC raw data to armature "
                     f"({len(skinning['sections'])} sections)")
    for i, l in enumerate(lods):
        v = sum(vb['vcount'] for vb in l['vbs'])
        f = sum(len(vb['faces']) for vb in l['vbs'])
        strides = sorted({vb['stride'] for vb in l['vbs']})
        vlog.log(f"  LOD[{i}] dist={l['lod_distance']:.1f}  strides={strides}  "
                 f"{v} verts, {f} faces")

    if lod != -1 and lod >= len(lods):
        raise IndexError(
            f"requested LOD {lod} but only {len(lods)} LODs in file"
        )

    lods_to_import = list(range(len(lods))) if lod == -1 else [lod]
    vlog.log(f"\n=== CREATING BLENDER MESH(ES) — LODs: {lods_to_import} ===")

    all_created = []

    # Shared imports (used by every LOD).
    from collections import defaultdict
    # apply_split_normals / store_tangent_attributes come from this folder's
    # own mesh_helpers_fc3 (module-level import above) — no avatar dependency.

    # SULC sections: each carries a bone palette for one SDOL section.
    # bone_assignments[k] = global bone index for palette slot k.
    sulc_secs = (skinning.get('sections') or []) if skinning else []

    # Flat list of 48-entry bone palettes, one per SULC section.
    # palette[local_idx] = global bone index.
    _sulc_palettes = [sec['bone_assignments'] for sec in sulc_secs] if sulc_secs else []

    # Pre-compute cumulative SULC section offsets per LOD so we know
    # which SULC sections belong to each LOD.
    _sulc_lod_starts = []
    _off = 0
    for _lod in lods:
        _sulc_lod_starts.append(_off)
        _off += len(_lod.get('entries', []))
    vlog.log(f"SULC: {len(_sulc_palettes)} palettes  LOD entry counts: "
             f"{[len(l.get('entries',[])) for l in lods]}  starts: {_sulc_lod_starts}")

    def _build_vert_sulc(target, lod_idx):
        """Return list[int] mapping global vert index → SULC section index."""
        entries = target.get('entries', [])
        sulc_off = _sulc_lod_starts[lod_idx] if lod_idx < len(_sulc_lod_starts) else 0
        total_verts = sum(vb['vcount'] for vb in target['vbs'])
        result = [max(0, sulc_off)] * total_verts
        vb_vert_offs = []
        vo = 0
        for vb in target['vbs']:
            vb_vert_offs.append(vo)
            vo += vb['vcount']
        for k, e in enumerate(entries):
            vb_i = e[0]
            stride = target['vbs'][vb_i]['stride']
            if stride == 0:
                continue
            first_local = e[5] // stride
            last_local  = e[4]
            vb_off = vb_vert_offs[vb_i]
            sec_idx = sulc_off + k
            if sec_idx >= len(_sulc_palettes):
                sec_idx = len(_sulc_palettes) - 1
            for vi in range(first_local, min(last_local + 1, target['vbs'][vb_i]['vcount'])):
                gvi = vi + vb_off
                if gvi < total_verts:
                    result[gvi] = sec_idx
        return result

    def _apply_vb_weights(obj, blender_vis, raw_verts, global_vis, vert_sulc):
        """Apply skinning weights.

        blender_vis  : local (compact) vertex indices in the Blender object
        raw_verts    : matching list of vertex dicts with 'si'/'sw' keys
        global_vis   : global VB vert index for each entry (for SULC lookup)
        vert_sulc    : per-global-vert SULC section index
        """
        if not bones:
            return 0
        assigned = 0
        vg_cache = {}   # bone_name -> VertexGroup (avoid per-weight .get lookup)
        for blender_vi, vert_d, gvi in zip(blender_vis, raw_verts, global_vis):
            sw = vert_d.get('sw')
            si = vert_d.get('si')
            if sw is None or si is None or sum(sw) == 0:
                continue
            sec_idx = vert_sulc[gvi] if gvi < len(vert_sulc) else 0
            palette = _sulc_palettes[sec_idx] if _sulc_palettes and sec_idx < len(_sulc_palettes) else []
            for k in range(4):
                w_raw = sw[k]
                if w_raw == 0:
                    continue
                local_bi = si[k]
                global_bi = palette[local_bi] if palette and local_bi < len(palette) else local_bi
                if global_bi >= len(bones):
                    continue
                w = w_raw / 255.0
                bone_name = bones[global_bi]['name']
                vg = vg_cache.get(bone_name)
                if vg is None:
                    vg = (obj.vertex_groups.get(bone_name)
                          or obj.vertex_groups.new(name=bone_name))
                    vg_cache[bone_name] = vg
                vg.add([blender_vi], w, 'ADD')
                assigned += 1
        return assigned

    base_name = os.path.splitext(os.path.basename(fp))[0]

    # ── Per-LOD object creation ───────────────────────────────────────
    for lod_idx in lods_to_import:
        target = lods[lod_idx]
        vlog.log(f"\n--- LOD[{lod_idx}]  distance={target['lod_distance']:.2f}, "
                 f"{len(target['vbs'])} VBs ---")

        # Build combined vertex arrays and a flat list of raw vert dicts.
        positions   = []
        all_uvs     = []
        all_uv1s    = []
        all_normals = []
        all_tangents  = []
        all_binormals = []
        all_colors    = []
        all_verts_raw = []   # raw dicts — carry si/sw bone data
        for vb in target['vbs']:
            for v in vb['verts']:
                positions.append(v['p'])
                all_uvs.append(v['uv'])
                all_uv1s.append(v.get('uv1'))
                all_normals.append(v['n'])
                all_colors.append(v.get('color'))
                all_verts_raw.append(v)
                if v.get('t') is not None:
                    all_tangents.append(v['t'])
                if v.get('b') is not None:
                    all_binormals.append(v['b'])
        have_tangents  = len(all_tangents)  == len(positions) and bool(all_tangents)
        have_uv1    = any(u is not None for u in all_uv1s)
        have_colors = any(c is not None for c in all_colors)
        have_binormals = len(all_binormals) == len(positions) and bool(all_binormals)

        # Build per-vert SULC section index map (used by both paths).
        vert_sulc = _build_vert_sulc(target, lod_idx) if _sulc_palettes else []

        # ── SEPARATE PRIMITIVES: one Blender object per material section ──
        if separate:
            created = []
            vlog.log(f"\n=== CREATING SEPARATE SECTION OBJECTS (LOD {lod_idx}) ===")

            for vb_idx, vb in enumerate(target['vbs']):
                vb_vert_off = sum(target['vbs'][j]['vcount'] for j in range(vb_idx))
                sec_list = vb.get('sections', [])
                if not sec_list:
                    continue
                vb_idx_base = min(s[1] for s in sec_list)

                for sec_i, (mat_i, idx_s, idx_e) in enumerate(sec_list):
                    n_faces = (idx_e - idx_s) // 3
                    face_start = (idx_s - vb_idx_base) // 3
                    face_slice = vb['faces'][face_start: face_start + n_faces]
                    if not face_slice:
                        continue

                    # Compact: local vertex indices 0..N-1
                    vert_set = set()
                    for fa, fb, fc_v in face_slice:
                        vert_set.update((fa, fb, fc_v))
                    g_verts = sorted(vert_set)
                    g_to_l = {gv: lv for lv, gv in enumerate(g_verts)}
                    local_faces = [(g_to_l[fa], g_to_l[fb], g_to_l[fc_v])
                                   for fa, fb, fc_v in face_slice]

                    l_pos  = [positions[gv]     for gv in g_verts]
                    l_uv   = [all_uvs[gv]       for gv in g_verts]
                    l_uv1  = [all_uv1s[gv]      for gv in g_verts] if have_uv1    else []
                    l_col  = [all_colors[gv]     for gv in g_verts] if have_colors else []
                    l_nrm  = [all_normals[gv]   for gv in g_verts]
                    l_tan  = [all_tangents[gv]  for gv in g_verts] if have_tangents  else []
                    l_bin  = [all_binormals[gv] for gv in g_verts] if have_binormals else []

                    mat_name = materials[mat_i][0] if mat_i < len(materials) else f"mat{mat_i}"
                    mat_path = materials[mat_i][1] if mat_i < len(materials) else ""
                    sec_name = f"{base_name}_LOD{lod_idx}_{mat_name}"
                    # Deduplicate if same material appears in multiple sections
                    if bpy.data.objects.get(sec_name):
                        sec_name = f"{sec_name}_{vb_idx}_{sec_i}"

                    sec_mesh = bpy.data.meshes.new(sec_name)
                    sec_obj  = bpy.data.objects.new(sec_name, sec_mesh)
                    ctx.collection.objects.link(sec_obj)

                    sec_mesh.from_pydata(l_pos, [], local_faces)
                    sec_mesh.update(calc_edges=True)
                    # FC3/FC4 use the game's opposite front-face winding;
                    # reverse it so faces shade correctly (the normal is the
                    # outward surface normal and is NEVER negated — same rule
                    # as Avatar).  The injector reverses faces back to the
                    # original format in _extract_components.
                    flip_face_winding(sec_mesh)

                    # UV0
                    uv_layer = sec_mesh.uv_layers.new(name='UVMap')
                    flat_uv0 = []
                    for loop in sec_mesh.loops:
                        flat_uv0 += l_uv[loop.vertex_index]
                    uv_layer.data.foreach_set('uv', flat_uv0)

                    # UV1
                    if l_uv1 and any(u is not None for u in l_uv1):
                        uv1_layer = sec_mesh.uv_layers.new(name='UV1')
                        flat_uv1 = []
                        for loop in sec_mesh.loops:
                            uv1 = l_uv1[loop.vertex_index]
                            flat_uv1 += uv1 if uv1 is not None else (0.0, 0.0)
                        uv1_layer.data.foreach_set('uv', flat_uv1)

                    # Vertex colors
                    if l_col and any(c is not None for c in l_col):
                        vcol = sec_mesh.vertex_colors.new(name='Col')
                        flat_col = []
                        for loop in sec_mesh.loops:
                            c = l_col[loop.vertex_index]
                            if c is not None:
                                flat_col += [c[0]/255.0, c[1]/255.0, c[2]/255.0, c[3]/255.0]
                            else:
                                flat_col += [1.0, 1.0, 1.0, 1.0]
                        vcol.data.foreach_set('color', flat_col)

                    # Material
                    mat_bl = bpy.data.materials.get(mat_name) or bpy.data.materials.new(mat_name)
                    mat_bl['xbg_fc3_material_path'] = mat_path
                    sec_mesh.materials.append(mat_bl)
                    for poly in sec_mesh.polygons:
                        poly.material_index = 0
                        poly.use_smooth = True

                    # Custom normals + tangents/binormals for round-trip injection
                    apply_split_normals(sec_mesh, l_nrm, False)
                    if l_tan or l_bin:
                        store_tangent_attributes(sec_mesh, l_tan, l_bin)

                    # Injection metadata — the FC3 injector reads these back
                    sec_obj['xbg_fc3_data'] = {
                        'filepath':      fp,
                        'version':       f'0x{version:08x}',
                        'lod':           lod_idx,
                        'vb_index':      vb_idx,
                        'section_index': sec_i,
                        'mat_index':     mat_i,
                        'mat_name':      mat_name,
                        'idx_start':     idx_s,
                        'idx_end':       idx_e,
                        'stride':        vb['stride'],
                        'vb_vert_offset': vb_vert_off,
                    }

                    # Per-vertex bone weights
                    l_raw = [all_verts_raw[gv] for gv in g_verts]
                    n_assigned = _apply_vb_weights(
                        sec_obj, list(range(len(g_verts))), l_raw, g_verts, vert_sulc)
                    if n_assigned:
                        vlog.log(f"    {n_assigned} weight assignments")

                    if armature_obj:
                        sec_obj.parent = armature_obj
                        mod = sec_obj.modifiers.new('Armature', 'ARMATURE')
                        mod.object = armature_obj

                    created.append(sec_obj)
                    all_created.append(sec_obj)
                    vlog.log(f"  + '{sec_name}'  {len(l_pos)} verts  {len(local_faces)} faces")

            if not created:
                vlog.log(f"  !!! LOD {lod_idx}: no sections found — skipping")

        else:
            # ── COMBINED MODE: one mesh for the whole LOD ─────────────────
            mesh_name = f"{base_name}_LOD{lod_idx}"
            mesh = bpy.data.meshes.new(mesh_name)

            all_faces  = []
            face_mat_idx = []
            uvs_per_loop = []
            n_verts = len(positions)
            for vb_idx, vb in enumerate(target['vbs']):
                sections  = vb.get('sections')
                vb_faces  = vb['faces']
                if sections:
                    face_mat_list = []
                    for mat_i, idx_start, idx_end in sections:
                        face_mat_list.extend([mat_i] * ((idx_end - idx_start) // 3))
                    while len(face_mat_list) < len(vb_faces):
                        face_mat_list.append(vb_idx)
                else:
                    face_mat_list = [vb_idx] * len(vb_faces)

                for fi, (fa, fb, fc_v) in enumerate(vb_faces):
                    if 0 <= fa < n_verts and 0 <= fb < n_verts and 0 <= fc_v < n_verts:
                        all_faces.append((fa, fb, fc_v))
                        face_mat_idx.append(face_mat_list[fi] if fi < len(face_mat_list) else vb_idx)
                        uvs_per_loop.extend([all_uvs[fa], all_uvs[fb], all_uvs[fc_v]])

            if not positions or not all_faces:
                vlog.log(f"  !!! LOD {lod_idx}: no geometry — skipping")
                continue

            mesh.from_pydata(positions, [], all_faces)
            mesh.update(calc_edges=True)

            if uvs_per_loop and len(uvs_per_loop) == len(mesh.loops):
                uv_layer = mesh.uv_layers.new(name='UVMap')
                flat_uv0 = [c for uv in uvs_per_loop for c in uv]
                uv_layer.data.foreach_set('uv', flat_uv0)

            # UV1 — foreach_set for speed
            if have_uv1:
                uv1_layer = mesh.uv_layers.new(name='UV1')
                flat = []
                for loop in mesh.loops:
                    uv1 = all_uv1s[loop.vertex_index]
                    flat += uv1 if uv1 is not None else (0.0, 0.0)
                uv1_layer.data.foreach_set('uv', flat)

            # Vertex colors — foreach_set for speed
            if have_colors:
                vcol = mesh.vertex_colors.new(name='Col')
                flat = []
                for loop in mesh.loops:
                    c = all_colors[loop.vertex_index]
                    if c is not None:
                        flat += [c[0]/255.0, c[1]/255.0, c[2]/255.0, c[3]/255.0]
                    else:
                        flat += [1.0, 1.0, 1.0, 1.0]
                vcol.data.foreach_set('color', flat)

            seen = {}
            for mi in face_mat_idx:
                if mi not in seen:
                    seen[mi] = len(seen)
                    mn = materials[mi][0] if mi < len(materials) else f"{base_name}_LOD{lod_idx}_mat{mi}"
                    mp = materials[mi][1] if mi < len(materials) else ""
                    mat = bpy.data.materials.get(mn) or bpy.data.materials.new(mn)
                    mat['xbg_fc3_material_path'] = mp
                    mesh.materials.append(mat)

            for poly_idx, poly in enumerate(mesh.polygons):
                mi = face_mat_idx[poly_idx] if poly_idx < len(face_mat_idx) else 0
                poly.material_index = seen.get(mi, 0)
                poly.use_smooth = True

            # Reverse FC3/FC4's opposite front-face winding (after per-loop
            # UV/colour are set, so the bmesh round-trip reverses them with
            # the faces).  Joined mode uses geometric normals, so this alone
            # makes them shade outward.
            flip_face_winding(mesh)

            obj = bpy.data.objects.new(mesh_name, mesh)
            obj['xbg_fc3_source']     = os.path.basename(fp)
            obj['xbg_fc3_version']    = f"0x{version:08x}"
            obj['xbg_fc3_lod']        = lod_idx
            obj['xbg_fc3_vb_count']   = len(target['vbs'])
            obj['xbg_fc3_vb_strides'] = [vb['stride'] for vb in target['vbs']]
            ctx.collection.objects.link(obj)
            all_created.append(obj)

            vlog.log(f"  + '{mesh_name}'  {len(positions)} verts  {len(all_faces)} faces")
            for ltmr_mi, slot_pos in sorted(seen.items(), key=lambda x: x[1]):
                mn = materials[ltmr_mi][0] if ltmr_mi < len(materials) else '(no material)'
                fc_count = sum(1 for mi in face_mat_idx if mi == ltmr_mi)
                vlog.log(f"    mat slot[{slot_pos}] '{mn}'  faces={fc_count}")

            # Per-vertex bone weights — combined mode.
            if bones:
                vlog.log(f"\n=== APPLYING SKINNING (LOD {lod_idx}) ===")
                total_verts = len(positions)
                global_vis = list(range(total_verts))
                raw_all = [all_verts_raw[gv] for gv in global_vis]
                total_assigned = _apply_vb_weights(
                    obj, global_vis, raw_all, global_vis, vert_sulc)
                vlog.log(f"  {total_assigned} weight assignments, "
                         f"{len(obj.vertex_groups)} groups")

            # Parent to armature
            if armature_obj:
                obj.parent = armature_obj
                mod = obj.modifiers.new('Armature', 'ARMATURE')
                mod.object = armature_obj
                vlog.log(f"  {obj.name}  →  {armature_obj.name}")

    created = all_created   # alias for summary log below

    vlog.log(f"\n{'#'*60}")
    vlog.log(f"# {game_name.upper()} IMPORT COMPLETE")
    vlog.log(f"# {len(all_created)} object(s), {len(materials)} material slots, "
             f"{len(bones)} bones")
    vlog.log(f"{'#'*60}\n")
    return {'FINISHED'}

def _build_fc3_armature(ctx, bones, file_basename):
    """Build a Blender armature from FC3/FC4 EDON bones.

    Each bone's local transform is (translation, rotation) relative to
    its parent.  World transform = parent_world @ local.  EDON stores
    bones in topological order so a single left-to-right pass works.

    The 4-float `rotation_raw` field is stored in the file with a
    magnitude that is NOT always 1.0 — we normalize before use.
    Field order is (w, x, y, z) — verified against the root bone (Vaas)
    which has identity rotation stored as (1, 0, 0, 0).
    """
    vlog.log(f"\n=== CREATING FC3/FC4 ARMATURE ===")
    vlog.log(f"  bone count: {len(bones)}")
    roots = [b['name'] for b in bones if b['parent'] == -1]
    vlog.log(f"  root bone(s): {roots}")

    an = f"{os.path.splitext(file_basename)[0]}_Armature"
    ad = bpy.data.armatures.new(an)
    ao = bpy.data.objects.new(an, ad)
    ctx.collection.objects.link(ao)
    ctx.view_layer.objects.active = ao
    vlog.log(f"  armature object: {an}")

    # Compute each bone's WORLD 4x4 matrix.
    # local_matrix = translation_matrix @ rotation_matrix
    # world_matrix = parent_world @ local_matrix
    from mathutils import Vector, Quaternion, Matrix

    world_mats = [Matrix.Identity(4) for _ in bones]
    for i, b in enumerate(bones):
        t = Vector(b['translation'])
        raw = b['rotation_raw']
        # File order is (w, x, y, z) per root-bone identity check.
        # Magnitudes vary so we normalize defensively.
        q = Quaternion((raw[0], raw[1], raw[2], raw[3]))
        try:
            q.normalize()
        except Exception:
            q = Quaternion()    # fall back to identity
        local = Matrix.Translation(t) @ q.to_matrix().to_4x4()
        if b['parent'] == -1 or b['parent'] >= i:
            world_mats[i] = local
        else:
            world_mats[i] = world_mats[b['parent']] @ local

    bpy.ops.object.mode_set(mode='EDIT')
    eb = {}
    MIN_LEN = 0.05
    for i, b in enumerate(bones):
        bone = ad.edit_bones.new(b['name'])
        wm = world_mats[i]
        head = wm.to_translation()
        # Default tail = head + small +Y in bone-local space.  This
        # gets refined below to aim at the average child head.
        tail_dir = wm.to_3x3() @ Vector((0.0, MIN_LEN, 0.0))
        bone.head = head
        bone.tail = head + tail_dir
        eb[i] = bone
        bone['xbg_fc3_rotation_raw'] = list(b['rotation_raw'])
        bone['xbg_fc3_scale'] = list(b['scale'])
        bone['xbg_fc3_flag1'] = b['flag1']

    # Set parents
    parented = 0
    for i, b in enumerate(bones):
        if b['parent'] != -1 and 0 <= b['parent'] < len(bones):
            eb[i].parent = eb[b['parent']]
            parented += 1
    vlog.log(f"  parented {parented} bones (root has no parent)")

    # Aim each bone's tail at the average of its children's world
    # heads (Blender convention for sensible bone visualisation).
    children_of = {}
    for i, b in enumerate(bones):
        if b['parent'] != -1:
            children_of.setdefault(b['parent'], []).append(i)
    leaf_count = sum(1 for i in eb if not children_of.get(i))
    for i, bone in eb.items():
        children = children_of.get(i, [])
        if children:
            avg = Vector((0.0, 0.0, 0.0))
            for c in children:
                avg += world_mats[c].to_translation()
            avg /= len(children)
            if (avg - bone.head).length >= MIN_LEN:
                bone.tail = avg
    vlog.log(f"  bone tails: {len(eb) - leaf_count} aimed at children, "
             f"{leaf_count} leaf bones use default +Y direction")

    bpy.ops.object.mode_set(mode='OBJECT')
    ao['xbg_fc3_armature'] = True
    vlog.log(f"  armature created with {len(eb)} bones, "
             f"{len(children_of)} have children")
    return ao
