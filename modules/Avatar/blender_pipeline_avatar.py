"""Blender datablock CREATION for XBG import — the viewport-building pipeline.

Turns the parsed XBG data (from import_xbg's XBGParser) into actual Blender
objects: armatures, meshes, materials/textures. Split out of import_xbg.py so the
parsing (binary -> data) and the Blender building (data -> scene) live in
separate, findable places. These were methods of XBGBlenderImporter but used no
instance state (only called each other), so they're plain module functions now;
XBGBlenderImporter.load() calls them directly.

  create_armature(skel, nb, reorient_bones)         -> armature object
  create_meshes(meshes, ao, mns, ...)               -> [mesh objects]
  _compact_mesh_data(mesh)                           -> de-duplicate vertices
  setup_material_textures(m2s, df, lhd, iad)         -> wire textures onto mats
"""

import math
import os

import bpy
import mathutils

from .binary_avatar import LE
from ..Core.debug import VerboseLogger as vlog, TraceLogger
from .import_uv_avatar import apply_uv_layer, flip_face_winding
from .import_weights_avatar import apply_vertex_weights
from .normals_avatar import apply_split_normals, store_tangent_attributes
from .vertex_colors_avatar import apply_vertex_colors
from .import_materials_avatar import XBMParser
from .import_xbt_avatar import XBTConverter
from .nodes_avatar import BlenderMaterialSetup


def create_armature(skel, nb, reorient_bones=False):
    if skel.get_bone_count() == 0:
        return None
    
    vlog.log(f"\n=== CREATING ARMATURE ===")
    
    # Check if we should use MB2O bind matrices
    use_mb2o = any(bd.bind_matrix is not None for bd in skel.bones)
    if use_mb2o:
        vlog.log(f"Using MB2O inverse bind matrices for armature positioning")
    else:
        vlog.log(f"Using EDON hierarchy transforms for armature positioning")

    # MB2O ("object-to-bone") matrices are expressed in the OUTERMOST
    # root node's frame, not world space.  Models with a dedicated model
    # node at origin (npc_kendra: 'Npc_Kendra' -> 'Root') are unaffected,
    # but models whose outermost node is 'Root' itself carry its offset
    # (martyalencar/avatar_m_body: Root at z=0.996/1.694) — without this
    # correction the skeleton lands ~1 m away from the mesh.
    #
    # Two DIFFERENT root frames are needed here, for two different jobs:
    #   root_world  (translation only) - feeds mesh_xform below. The mesh's
    #     own vertex space only shares the root's ORIGIN with the bind
    #     frame, not its rotation - this was empirically pinned on
    #     avatar_m_body and is independent of MB2O, do not touch.
    #   root_world_full (translation + rotation) - feeds the MB2O bind-pose
    #     reconstruction further down. (2026-06-30: an earlier version of
    #     this function reused the translation-only frame for MB2O bones
    #     too, which happened to "look less wrong" only because the MB2O
    #     index-to-bone mapping itself was broken at the time - see
    #     skeleton_avatar.apply_bind_matrices and agents.md. Once bones are
    #     matched to the correct matrix, the ROOT'S FULL TRANSFORM is
    #     required to reconstruct the same bind position EDON already
    #     gives directly - verified by geometric cross-check, matched
    #     positions land within ~cm of EDON's own world_matrix.)
    root_world = mathutils.Matrix.Identity(4)
    root_world_full = mathutils.Matrix.Identity(4)
    root_rot_inv = mathutils.Matrix.Identity(4)
    for i, bd in enumerate(skel.bones):
        pid = bd.parent_id
        if (pid is None or pid < 0 or pid == i) and bd.world_matrix:
            root_world = mathutils.Matrix.Translation(
                bd.world_matrix.translation)
            root_world_full = bd.world_matrix.copy()
            root_rot_inv = (bd.world_matrix.to_quaternion()
                            .inverted().to_matrix().to_4x4())
            break
    if use_mb2o and root_world.translation.length > 1e-5:
        vlog.log(f"MB2O frame correction: root node offset "
                 f"{tuple(root_world.translation)}")

    an = f"{nb}_Armature"
    ad = bpy.data.armatures.new(an)
    ao = bpy.data.objects.new(an, ad)

    bpy.context.collection.objects.link(ao)
    bpy.context.view_layer.objects.active = ao
    ao.rotation_euler = (0, 0, math.radians(180))
    vlog.log(f"Armature rotation: (0, 0, 180°)")
    # Mesh lift = root TRANSLATION combined with the INVERSE of the
    # root node's rotation.  The bones are placed translation-only
    # (the O2BM frame shares only the root's origin), but the mesh
    # geometry is authored in the root node's LOCAL axes, so undoing
    # the root rotation aligns it with the bind frame.  Empirically
    # pinned on marty/avatar_m_body (root carries a 90-deg Z quat):
    # full root transform left the mesh 180 deg off, translation-only
    # left it 90 deg off — T @ R^-1 is the unique fix for both.
    mesh_xform = root_world @ root_rot_inv
    if (root_world.translation.length > 1e-5
            or abs(root_rot_inv.to_quaternion().angle) > 1e-4):
        ao['xbg_root_xform'] = [v for row in mesh_xform for v in row]
    
    bpy.ops.object.mode_set(mode='EDIT')
    eb = {}
    bind_world = {}   # bone name -> 16 floats, the world matrix actually
                      # used to place the bone (MB2O bind or EDON world).
                      # The MAB importer needs this: animation must be
                      # reconstructed against the SAME rest the bones
                      # (and the skinning) were built from.

    for i, bd in enumerate(skel.bones):
        bn = bd.name if bd.name else f"Bone_{i}"
        e = ad.edit_bones.new(bn)
        eb[i] = e

        # Use MB2O bind matrix if available, otherwise use EDON world matrix
        if use_mb2o and bd.bind_matrix is not None:
            # MB2O stores INVERSE bind matrices (in the root node's FULL
            # frame) — invert, then lift into model space via root_world_full
            try:
                bind_pose_matrix = root_world_full @ bd.bind_matrix.inverted()
                bind_world[bn] = [v for row in bind_pose_matrix for v in row]
                e.head = mathutils.Vector(bind_pose_matrix.translation)
                vlog.log(f"  Bone {i} ({bn}): Using MB2O position {e.head}")
            except:
                # If matrix is singular/non-invertible, fall back to EDON
                vlog.log(f"  WARNING: Bone {i} ({bn}): MB2O matrix non-invertible, using EDON")
                e.head = mathutils.Vector(bd.world_matrix.translation) if bd.world_matrix else mathutils.Vector((0, 0, 0))
        else:
            # Use EDON transforms
            e.head = mathutils.Vector(bd.world_matrix.translation) if bd.world_matrix else mathutils.Vector((0, 0, 0))
        
        e.tail = e.head + mathutils.Vector((0, 0.5, 0))
    
    for i, bd in enumerate(skel.bones):
        e = eb[i]
        
        if bd.parent_id is not None and bd.parent_id in eb:
            e.parent = eb[bd.parent_id]
            e.use_connect = False
        
        # Calculate tail direction based on which matrix we're using
        if use_mb2o and bd.bind_matrix is not None:
            try:
                bind_pose_matrix = root_world_full @ bd.bind_matrix.inverted()
                rot = bind_pose_matrix.to_quaternion()
                off = mathutils.Vector((0, 1, 0)) * 0.5
                off.rotate(rot)
                e.tail = e.head + off
            except:
                # Fall back to EDON rotation
                if bd.world_matrix:
                    rot = bd.world_matrix.to_quaternion()
                    off = mathutils.Vector((0, 1, 0)) * 0.5
                    off.rotate(rot)
                    e.tail = e.head + off
        else:
            # Use EDON transforms
            if bd.world_matrix:
                rot = bd.world_matrix.to_quaternion()
                off = mathutils.Vector((0, 1, 0)) * 0.5
                off.rotate(rot)
                e.tail = e.head + off
    
    # Bone reorientation: point each bone's tail toward its children's heads.
    # IMPORTANT: must run BEFORE mode_set('OBJECT') — edit bone handles are
    # only valid while the armature stays in EDIT mode. Accessing eb[] after
    # leaving and re-entering EDIT mode causes a crash (stale C pointers).
    if reorient_bones:
        # Build children map: parent_index -> [valid child indices]
        # Exclude self-references: some XBG files store parent_id == own index on
        # the root bone (e.g. 0 -> 0).  Without this guard the root appears in its
        # own children list and the tail-direction average is corrupted.
        children_map = {}
        for j, bd in enumerate(skel.bones):
            pid = bd.parent_id
            if pid is not None and pid != j and pid in eb and j in eb:
                children_map.setdefault(pid, []).append(j)

        MIN_BONE_LEN = 0.05  # prevent zero-length bones (Blender will crash)

        for i in eb:
            e  = eb[i]
            bd = skel.bones[i]
            pid = bd.parent_id
            has_real_parent = pid is not None and pid != i and pid in eb
            children = [ci for ci in children_map.get(i, []) if ci in eb]

            if has_real_parent and children:
                # Interior bone: aim tail at the average of all direct child heads.
                avg = mathutils.Vector()
                for ci in children:
                    avg += eb[ci].head
                avg /= len(children)

                # Contralateral guard: some rigs genuinely parent a chain
                # from the OPPOSITE side of the body (direhorse
                # 'Engine_Base_LF_Arms_Linker' at x=+0.28 parents the
                # left chain at x=-0.25 — verified authentic in both the
                # XBG NODE and the .skeleton).  Aiming the tail across
                # the midline draws a giant X over the chest.  Keep the
                # bone's own orientation in that case — tails are purely
                # cosmetic, deformation is unaffected.
                crosses_midline = (abs(e.head.x) > 0.05
                                   and abs(avg.x) > 0.05
                                   and (e.head.x > 0) != (avg.x > 0))

                if (avg - e.head).length >= MIN_BONE_LEN and not crosses_midline:
                    e.tail = avg
                # else: children collapsed onto this bone (or contralateral)
                # — keep world-matrix tail

            elif has_real_parent:
                # Leaf / end-of-chain bone: extend away from parent along the
                # parent->self direction (continues the visual line of the chain).
                #
                # Threshold note: MIN_BONE_LEN (0.05) is the minimum FINAL bone
                # length Blender needs to avoid a zero-length crash, but it must NOT
                # be used as the gate for whether we USE the away direction.
                # e.g. wasp wing bones sit only 0.048 units from Pelvis — valid
                # geometry, but 0.048 < 0.05 so the old >= MIN_BONE_LEN check was
                # silently discarding their direction and falling back to the
                # arbitrary world-matrix tail, producing giant off-screen bones.
                #
                # Fix: gate on > 0.001 (just avoids division-by-zero), then clamp
                # the final length UP to MIN_BONE_LEN so Blender never sees a
                # degenerate bone while still honouring the correct direction.
                away = e.head - eb[pid].head
                if away.length > 0.001:
                    e.tail = e.head + away.normalized() * max(away.length, MIN_BONE_LEN)
                # else: head truly coincides with parent — keep world-matrix tail

            # else: root bone (no real parent) — keep the compact world-matrix tail
            # set in the first pass above.  Root bones are reference/origin bones;
            # stretching them toward their children creates a misleadingly large bone.

    bpy.ops.object.mode_set(mode='OBJECT')
    if bind_world:
        ao['xbg_bind_world'] = bind_world
    vlog.log(f"Created armature: {an}")

    return ao

def _compact_mesh_data(mesh):
    """Remove unused vertices, keeping a new_to_old mapping for export correctness.

    Uses list comprehensions (faster than conditional per-loop appends) and
    builds both direction mappings in a single enumeration pass.
    """
    try:
        from ..Core.debug import TraceLogger
    except Exception:
        TraceLogger = None

    # Collect all vertex indices referenced by any face
    used_indices = set()
    for prim in mesh.primitives:
        used_indices.update(prim.indices)

    # Sort once; derive both mappings via enumerate (single pass)
    sorted_used = sorted(used_indices)
    old_to_new  = {old: new for new, old in enumerate(sorted_used)}
    new_to_old  = {new: old for new, old in enumerate(sorted_used)}

    # ── Defensive check: detect out-of-range indices BEFORE crashing.
    # An IndexError on `pl[i]` here means the SDOL index buffer wrote
    # a global vertex index that exceeds the vert section size — the
    # exact corruption the injector might silently produce in shared-
    # VB mode.  Emit a full diagnostic record + raise a clear error.
    pl = mesh.vert_pos_list
    if sorted_used and sorted_used[-1] >= len(pl):
        n_oob = sum(1 for i in sorted_used if i >= len(pl))
        # Build a per-primitive breakdown so we know which submesh has
        # the bad indices (each primitive maps to one DNKS/SDOL submesh).
        prim_rows = []
        for pi, prim in enumerate(mesh.primitives):
            bad = [i for i in prim.indices if i >= len(pl)]
            prim_rows.append({
                "primitive_idx":      pi,
                "material_index":     getattr(prim, "material_index", None),
                "material_name":      getattr(prim, "material_name", None),
                "indices_total":      len(prim.indices),
                "indices_oob":        len(bad),
                "first_bad_index":    bad[0] if bad else None,
                "max_bad_index":      max(bad) if bad else None,
                "min_index_in_prim":  min(prim.indices) if prim.indices else None,
                "max_index_in_prim":  max(prim.indices) if prim.indices else None,
            })
        err_data = {
            "mesh_lod":             mesh.lod_level,
            "mesh_part":            mesh.part_number,
            "mesh_vb_index":        mesh.vb_index,
            "vert_pos_list_len":    len(pl),
            "header_vert_count":    mesh.vert_count,
            "unique_used_indices":  len(sorted_used),
            "max_used_index":       sorted_used[-1],
            "out_of_bounds_count":  n_oob,
            "primitives":           prim_rows,
        }
        if TraceLogger is not None:
            TraceLogger.info(
                f"[import] *** OUT-OF-RANGE INDEX in mesh "
                f"LOD{mesh.lod_level}.part{mesh.part_number}: "
                f"max_idx={sorted_used[-1]} but vert_pos_list has only {len(pl)} entries "
                f"({n_oob} OOB indices across {len(prim_rows)} primitives)",
                event="import_oob_index",
                data=err_data)
        vlog.warn(f"\n*** OUT-OF-RANGE INDEX in mesh "
                  f"LOD{mesh.lod_level}.part{mesh.part_number}: "
                  f"max_idx={sorted_used[-1]} but vert_pos_list has {len(pl)} entries")
        for pr in prim_rows:
            vlog.warn(f"      prim[{pr['primitive_idx']}] mat='{pr['material_name']}'  "
                      f"indices={pr['indices_total']}  "
                      f"OOB={pr['indices_oob']}  "
                      f"range[{pr['min_index_in_prim']}..{pr['max_index_in_prim']}]")
        raise IndexError(
            f"vertex index {sorted_used[-1]} out of range "
            f"(mesh has {len(pl)} verts); LOD{mesh.lod_level} part{mesh.part_number}, "
            f"{n_oob}/{len(sorted_used)} OOB. See log/jsonl for per-primitive detail.")

    # List-comprehension slicing — significantly faster than conditional .append()
    new_verts = [pl[i] for i in sorted_used]

    ul = mesh.vert_uv_list
    new_uvs = [ul[i] for i in sorted_used] if ul else []

    # UV1 / UV2 / Color: guard against shorter lists (some verts may be unused)
    uv1l = mesh.vert_uv1_list
    new_uv1s = [uv1l[i] for i in sorted_used if i < len(uv1l)] if uv1l else []

    uv2l = mesh.vert_uv2_list
    new_uv2s = [uv2l[i] for i in sorted_used if i < len(uv2l)] if uv2l else []

    cl = mesh.vert_color_list
    new_colors = [cl[i] for i in sorted_used if i < len(cl)] if cl else []

    wl = mesh.skin_weight_list
    new_weights = [wl[i] for i in sorted_used if i < len(wl)] if wl else []

    sl = mesh.skin_indice_list
    new_skin_indices = [sl[i] for i in sorted_used if i < len(sl)] if sl else []

    # Stash compacted secondary arrays back on the mesh for create_meshes
    mesh.vert_uv1_list   = new_uv1s
    mesh.vert_uv2_list   = new_uv2s
    mesh.vert_color_list = new_colors

    nl = mesh.vert_normal_list
    new_normals = [nl[i] for i in sorted_used if i < len(nl)] if nl else []
    mesh.vert_normal_list = new_normals

    tl = mesh.vert_tangent_list
    new_tangents = [tl[i] for i in sorted_used if i < len(tl)] if tl else []
    mesh.vert_tangent_list = new_tangents

    bl = mesh.vert_binormal_list
    new_binormals = [bl[i] for i in sorted_used if i < len(bl)] if bl else []
    mesh.vert_binormal_list = new_binormals

    # Remap all primitive face indices to the compacted vertex space
    new_primitives = [
        ([old_to_new[i] for i in prim.indices], prim.material_index, prim.material_name)
        for prim in mesh.primitives
    ]

    removed_count = len(pl) - len(new_verts)
    vlog.log(f"  Vertex compaction: {len(pl)} -> {len(new_verts)} vertices ({removed_count} unused removed)")

    return new_verts, new_uvs, new_weights, new_skin_indices, new_primitives, new_to_old


import re as _re_assembly


def _assemble_rigid_part(obj, mesh_name, xm2b, ao):
    """Assemble + bind a RIGID (unskinned) weapon / vehicle part to its bone.

    Weapons and some vehicles ship "disassembled": each part is a separate
    submesh authored in its OWN BONE's local space, and the engine places
    each part at its bone's world transform (the descriptor's
    GraphicComponent maps meshName->boneName; the skeleton carries the bone
    positions). The mesh verts alone sit clustered at the origin — you have
    to move each part to its bone to reassemble the model.

    Two things, both needed (verified on dual_wasp, 2026-06-30):
      1. PLACE — move the object's origin to the matched bone's rest HEAD
         in armature space. The part verts, being bone-local, land in place.
         Translation only: the mesh-carrying weapon bones have identity
         rotation; a part whose bone carries a real rotation would also need
         that applied — noted, not yet handled (rare).
      2. BIND — weight every vert 100% to that bone. The object already
         carries an Armature modifier; at REST the modifier is identity so
         step 1 alone defines the assembled pose, and when the user POSES
         the bone the modifier applies the pose DELTA, moving the part
         rigidly with it. Without this the part assembles but does NOT
         follow the bone (the user's "the weapon doesn't move" report).

    MUST be called AFTER the mesh verts exist (needs them for the vertex
    group). The bone for a part is resolved from the XML mesh->bone map
    (`xm2b`) when present, ELSE from the submesh's own base name (many
    weapons have no sidecar XML but name their submeshes after their bones,
    e.g. wasp: WASP_FRAME_LOD0 -> bone WASP_FRAME). Gated by the caller on
    Use XML Assembly + `not mesh.has_skinning()`, so skinned character
    meshes are never touched.
    """
    if ao is None:
        return
    base = _re_assembly.sub(r'_LOD\d+$', '', mesh_name, flags=_re_assembly.IGNORECASE).upper()
    # XML map first; else fall back to the submesh's own base name.
    bone_name = (xm2b.get(base) if xm2b else None) or base
    bones = ao.data.bones
    b = bones.get(bone_name)
    if b is None:
        # case-insensitive + truncation-tolerant suffix match (armature bone
        # names may be a SUFFIX of the full name from the legacy 25-char cap).
        bn_u = bone_name.upper()
        cand = [bn for bn in bones
                if bn.name.upper() == bn_u or bn_u.endswith(bn.name.upper())]
        b = max(cand, key=lambda x: len(x.name)) if cand else None
    if b is None:
        return
    # 1. place at the bone's rest head
    obj.location = b.head_local.copy()
    # 2. bind 100% to that bone so posing the bone moves the part
    nverts = len(obj.data.vertices) if obj.data else 0
    if nverts:
        vg = obj.vertex_groups.get(b.name) or obj.vertex_groups.new(name=b.name)
        vg.add(range(nverts), 1.0, 'REPLACE')
    vlog.log(f"  [assembly] '{mesh_name}' -> bone '{b.name}' at "
             f"{tuple(round(v,4) for v in b.head_local)}  (bound {nverts} verts)")


def create_meshes(meshes, ao, mns, imo=False, df="", lt=True, lhd=True,
                  xb={}, xm2b={}, xmi2b={}, xmi2n={}, sp=False, fp="",
                  vps=1.0, uvt=0.0, uvs=1.0, iad=False, lod_names={}, compact_vertices=True,
                  lod_name_bboxes={}, pmcp_offset=0, sub_mesh_list=None, flip_normals=False,
                  endian=LE, uxa=False):
    vlog.log(f"\n=== CREATING BLENDER MESHES ===")
    try:
        from ..Core.debug import TraceLogger
    except Exception:
        TraceLogger = None

    if compact_vertices:
        vlog.log("Vertex compaction ENABLED - removing unused vertices")
    else:
        vlog.log("Vertex compaction DISABLED - keeping all vertices (ghost vertices will be visible)")

    if TraceLogger is not None:
        TraceLogger.kvblock(
            "create_meshes() entry",
            [
                ("meshes_in",         len(meshes)),
                ("imo",               imo),
                ("separate_prims",    sp),
                ("compact_vertices",  compact_vertices),
                ("pos_scale",         vps),
                ("uv_trans",          uvt),
                ("uv_scale",          uvs),
                ("materials_known",   len(mns)),
            ],
            tier="DEBUG", event="loader_create_meshes_entry")

    co = []

    for mi, mesh in enumerate(meshes):
        if not mesh.vert_pos_list:
            if TraceLogger is not None:
                TraceLogger.debug(
                    f"  [create_meshes] mesh[{mi}] LOD{mesh.lod_level} "
                    f"part{mesh.part_number} skipped — no vert_pos_list",
                    event="loader_mesh_skipped_empty",
                    data={"idx": mi, "lod": mesh.lod_level,
                          "part": mesh.part_number})
            continue

        # Apply vertex compaction if enabled
        vertex_mapping = None  # Maps new index -> old index for export
        original_vert_count = len(mesh.vert_pos_list)

        if compact_vertices:
            # Compact the mesh and get the mapping
            verts, uv_coords, weights, skin_indices, primitives, vertex_mapping = _compact_mesh_data(mesh)
        else:
            # Keep all vertices (old behavior)
            verts = mesh.vert_pos_list
            uv_coords = mesh.vert_uv_list if mesh.vert_uv_list else []
            weights = mesh.skin_weight_list if mesh.skin_weight_list else []
            skin_indices = mesh.skin_indice_list if mesh.skin_indice_list else []
            primitives = [(p.indices, p.material_index, p.material_name) for p in mesh.primitives]

        if TraceLogger is not None:
            _prim_summary = [(pi, p[1], p[2], len(p[0])//3)
                              for pi, p in enumerate(primitives)]
            TraceLogger.kvblock(
                f"create_meshes mesh[{mi}]  LOD{mesh.lod_level}.part{mesh.part_number}",
                [
                    ("verts_orig",        original_vert_count),
                    ("verts_after_compact", len(verts)),
                    ("verts_removed",     original_vert_count - len(verts)),
                    ("primitives",        len(primitives)),
                    ("primitive_sizes",   _prim_summary),
                    ("uv_layer_count",    1 if uv_coords else 0),
                    ("has_skin",          bool(weights)),
                ],
                tier="DEBUG", event="loader_mesh_prepared")
        
        if sp:
            for pi, (indices, mat_idx, mat_name) in enumerate(primitives):
                skinning_type = "Skinned" if mesh.has_skinning() else "Static"
                
                # Get actual mesh name from LOD names using the submesh index
                if mesh.lod_level in lod_names and mesh.name_index < len(lod_names[mesh.lod_level]):
                    mesh_name = lod_names[mesh.lod_level][mesh.name_index]
                    mn = mesh_name
                else:
                    # Fallback to generic name
                    lod_display = f"LOD{mesh.lod_level}"
                    if mesh.sub_part_index >= 0:
                        mn = f"Mesh_{lod_display}_P{mesh.part_number}_Sub{mesh.sub_part_index}_{skinning_type}"
                    else:
                        mn = f"Mesh_{lod_display}_P{mesh.part_number}_{skinning_type}"
                
                me = bpy.data.meshes.new(mn)
                obj = bpy.data.objects.new(mn, me)
                bpy.context.collection.objects.link(obj)
                co.append(obj)
                
                # Convert vertex mapping to string keys for Blender compatibility
                mapping_for_blender = None
                if vertex_mapping:
                    mapping_for_blender = {str(k): v for k, v in vertex_mapping.items()}

                obj["xbg_data"] = {
                    "filepath": fp,
                    "vert_offset": mesh.vert_section_offset,
                    "vert_stride": mesh.vert_stride,
                    "vert_count": original_vert_count,
                    "vert_format_flags": mesh.vert_format_flags,
                    "pos_scale": vps,
                    "uv_trans": uvt,
                    "uv_scale": uvs,
                    "lod_level": mesh.lod_level,
                    "import_mesh_only": imo,
                    "xobb_offset": mesh.xobb_chunk_offset,
                    "hpsb_offset": mesh.hpsb_chunk_offset,
                    "pmcp_offset": pmcp_offset,
                    "vertex_mapping": mapping_for_blender,
                    # Source file byte order — '<' for PC, '>' for PS3.
                    # The injector reads this to write the same endianness back.
                    "endian": endian,
                    # Flat index of this object in the SDOL submesh list for
                    # its LOD.  Stored so the injector can place this object
                    # back at the correct SDOL slot even when only a subset
                    # of primitives is selected (e.g. head-only inject).
                    "sdol_submesh_slot": mesh.name_index,
                    # Whether the import flipped face winding + negated normals
                    # to match XBG's handedness convention in the viewport.
                    # The injector reads this and reverses the flip so the
                    # round-trip is identity instead of producing inverted
                    # geometry that the game backface-culls.
                    "flipped_on_import": bool(flip_normals),
                }

                # Store the per-submesh bone palette so the injector can
                # map vertex groups back to the correct palette slots.
                # FIX: use mat_list_info[pi] (this primitive's submesh), NOT
                # always [0].  mat_list_info entries are added in the same
                # order as mesh.primitives inside _process_mesh_faces, so pi
                # maps correctly to the matching submesh/palette entry.
                #
                # DNKS-key fix (2026-05): mat_list_info[pi][2] is the
                # sub_idx VALUE (engine identifier), not the SDOL
                # positional index.  sub_mesh_list is in SDOL-position
                # order, so we key by mesh.name_index instead (see
                # _process_mesh_faces for the long explanation).
                if sub_mesh_list and mesh.mat_list_info and pi < len(mesh.mat_list_info):
                    _lod_grp = mesh.mat_list_info[pi][1]
                    _sub_pos = (mesh.name_index
                                if mesh.name_index >= 0
                                else mesh.mat_list_info[pi][2])
                    if (_lod_grp < len(sub_mesh_list) and
                            _sub_pos < len(sub_mesh_list[_lod_grp])):
                        obj["xbg_bone_palette"] = list(
                            sub_mesh_list[_lod_grp][_sub_pos].bone_data)

                imo and setattr(obj, 'rotation_euler', (0, 0, math.radians(180)))

                if ao:
                    obj.parent = ao
                    mod = obj.modifiers.new(name="Armature", type='ARMATURE')
                    mod.object = ao
                    # Lift mesh by the SAME (translation-only) root
                    # matrix that placed the bones — left-multiply so the
                    # offset lands in the armature/parent frame.
                    ro = ao.get('xbg_root_xform')
                    if ro:
                        rm = mathutils.Matrix((ro[0:4], ro[4:8],
                                               ro[8:12], ro[12:16]))
                        obj.matrix_local = rm @ obj.matrix_local

                faces = [(indices[i], indices[i+1], indices[i+2])
                         for i in range(0, len(indices), 3) if i+2 < len(indices)]

                mrn = mat_name
                mat = bpy.data.materials.get(mrn) or bpy.data.materials.new(name=mrn)
                mat['xbg_source'] = mrn   # tag: came from the game XBG
                mat.use_nodes = True
                obj.data.materials.append(mat)

                me.from_pydata(verts, [], faces)
                me.update()

                # Flip face winding for correct Blender viewport display
                # (must happen before UV / color application so the loop
                #  ordering is stable when we read it below)
                if flip_normals:
                    flip_face_winding(me)

                # UV0 uses the same module helper as UV1/UV2.  vert_uv_list
                # never contains None sentinels (only UV1/UV2 do) so the
                # sentinel branch in apply_uv_layer is a no-op here.
                apply_uv_layer(me, uv_coords, "UVMap")
                apply_uv_layer(me, mesh.vert_uv1_list, "UVMap1")
                apply_uv_layer(me, mesh.vert_uv2_list, "UVMap2")
                apply_vertex_colors(me, mesh.vert_color_list)

                # Apply bone weights via the weights module (lazy vertex-
                # group creation, uint8 -> float by /255 inside).
                if ao:
                    apply_vertex_weights(obj, ao, weights, skin_indices)

                # Reassemble + bind rigid (unskinned) weapon/vehicle parts:
                # place each at its bone AND weight it 100% so it follows
                # when the bone is posed. After the verts exist. Gated by
                # Use XML Assembly; skinned meshes are left alone.
                if uxa and ao and not mesh.has_skinning():
                    _assemble_rigid_part(obj, mn, xm2b, ao)

                # XBG normals → Blender custom split normals (viewport accuracy).
                # XBG tangent/binormal → POINT attributes (re-export round-trip).
                apply_split_normals(me, mesh.vert_normal_list, flip_normals)
                store_tangent_attributes(me, mesh.vert_tangent_list, mesh.vert_binormal_list)

                lt and df and setup_material_textures([(mat, mrn)], df, lhd, iad)
                vlog.log(f"Created mesh: {mn} ({len(verts)} verts, {len(faces)} faces)")
        else:
            # Get actual mesh name from LOD names using the submesh index
            if mesh.lod_level in lod_names and mesh.name_index < len(lod_names[mesh.lod_level]):
                # Use the actual name from the file!
                mn = lod_names[mesh.lod_level][mesh.name_index]
                vlog.log(f"  Using file name: {mn} (LOD{mesh.lod_level}, index {mesh.name_index})")
            else:
                # Fallback to generic name
                unique_parts = set(m.part_number for m in meshes)
                is_multipart = len(unique_parts) > 1
                skinning_type = "Skinned" if mesh.has_skinning() else "Static"
                lod_display = f"LOD{mesh.lod_level}"
                
                # Build mesh name
                if mesh.sub_part_index >= 0:
                    # Has sub-parts
                    if is_multipart:
                        mn = f"Mesh_{lod_display}_P{mesh.part_number}_Sub{mesh.sub_part_index}_{skinning_type}"
                    else:
                        mn = f"Mesh_{lod_display}_Sub{mesh.sub_part_index}_{skinning_type}"
                else:
                    # No sub-parts
                    if is_multipart:
                        mn = f"Mesh_{lod_display}_P{mesh.part_number}_{skinning_type}"
                    else:
                        mn = f"Mesh_{lod_display}_{skinning_type}"
            
            me = bpy.data.meshes.new(mn)
            obj = bpy.data.objects.new(mn, me)
            bpy.context.collection.objects.link(obj)
            co.append(obj)
            
            # Convert vertex mapping to string keys for Blender compatibility
            mapping_for_blender = None
            if vertex_mapping:
                mapping_for_blender = {str(k): v for k, v in vertex_mapping.items()}

            obj["xbg_data"] = {
                "filepath": fp,
                "vert_offset": mesh.vert_section_offset,
                "vert_stride": mesh.vert_stride,
                "vert_count": original_vert_count,
                "vert_format_flags": mesh.vert_format_flags,
                "pos_scale": vps,
                "uv_trans": uvt,
                "uv_scale": uvs,
                "lod_level": mesh.lod_level,
                "import_mesh_only": imo,
                "xobb_offset": mesh.xobb_chunk_offset,
                "hpsb_offset": mesh.hpsb_chunk_offset,
                "pmcp_offset": pmcp_offset,
                "vertex_mapping": mapping_for_blender,
                # Source file byte order — '<' for PC, '>' for PS3.
                # The injector reads this to write the same endianness back.
                "endian": endian,
                # Whether the import flipped face winding + negated normals
                # to match XBG's handedness convention in the viewport.
                # The injector reads this and reverses the flip so the
                # round-trip is identity instead of producing inverted
                # geometry that the game backface-culls.
                "flipped_on_import": bool(flip_normals),
            }

            # Store the per-submesh bone palette for correct weight export
            # NOTE: in joined-mesh mode all primitives are merged into one
            # object so only one palette can be stored.  We use the first
            # submesh's palette here; re-injection of joined meshes is not
            # officially supported (xbg_joined flag is set).  For reliable
            # per-primitive palette accuracy, import with Separate Primitives.
            #
            # DNKS-key fix (2026-05): see _process_mesh_faces — sub_idx is
            # the VALUE, not the position.  sub_mesh_list is keyed by
            # position, so use mesh.name_index.
            if sub_mesh_list and mesh.mat_list_info:
                _lod_grp = mesh.mat_list_info[0][1]
                _sub_pos = (mesh.name_index
                            if mesh.name_index >= 0
                            else mesh.mat_list_info[0][2])
                if (_lod_grp < len(sub_mesh_list) and
                        _sub_pos < len(sub_mesh_list[_lod_grp])):
                    obj["xbg_bone_palette"] = list(
                        sub_mesh_list[_lod_grp][_sub_pos].bone_data)

            imo and setattr(obj, 'rotation_euler', (0, 0, math.radians(180)))

            if ao:
                obj.parent = ao
                mod = obj.modifiers.new(name="Armature", type='ARMATURE')
                mod.object = ao
                # Lift mesh by the SAME (translation-only) root
                # matrix that placed the bones — left-multiply so the
                # offset lands in the armature/parent frame.
                ro = ao.get('xbg_root_xform')
                if ro:
                    rm = mathutils.Matrix((ro[0:4], ro[4:8],
                                       ro[8:12], ro[12:16]))
                    obj.matrix_local = rm @ obj.matrix_local

            faces = []
            mm = {}
            m2s = []

            for indices, mat_idx, mat_name in primitives:
                if mat_idx not in mm:
                    mrn = mat_name
                    mat = bpy.data.materials.get(mrn) or bpy.data.materials.new(name=mrn)
                    mat['xbg_source'] = mrn   # tag: came from the game XBG
                    mat.use_nodes = True
                    obj.data.materials.append(mat)
                    mm[mat_idx] = len(obj.data.materials) - 1
                    m2s.append((mat, mrn))
                [faces.append((indices[i], indices[i+1], indices[i+2]))
                 for i in range(0, len(indices), 3) if i+2 < len(indices)]

            me.from_pydata(verts, [], faces)
            me.update()

            # Flip face winding for correct Blender viewport display
            if flip_normals:
                flip_face_winding(me)

            # Assign polygon material indices via foreach_set (one C call)
            mat_index_flat = [0] * len(me.polygons)
            po = 0
            for indices, mat_idx, mat_name in primitives:
                bmi = mm.get(mat_idx, 0)
                nt  = len(indices) // 3
                for i in range(nt):
                    if po + i < len(mat_index_flat):
                        mat_index_flat[po + i] = bmi
                po += nt
            me.polygons.foreach_set("material_index", mat_index_flat)

            # UV0 / UV1 / UV2 + vertex colors all go through their modules.
            apply_uv_layer(me, uv_coords, "UVMap")
            apply_uv_layer(me, mesh.vert_uv1_list, "UVMap1")
            apply_uv_layer(me, mesh.vert_uv2_list, "UVMap2")
            apply_vertex_colors(me, mesh.vert_color_list)

            # Apply bone weights via the weights module.
            if ao:
                apply_vertex_weights(obj, ao, weights, skin_indices)

            # Reassemble + bind rigid (unskinned) weapon/vehicle parts before
            # the loader joins them: place each at its bone AND weight it 100%
            # so it follows when the bone is posed. Gated by Use XML Assembly;
            # skinned meshes untouched.
            if uxa and ao and not mesh.has_skinning():
                _assemble_rigid_part(obj, mn, xm2b, ao)

            # Custom split normals are intentionally NOT set here:
            # the subsequent join() + remove_doubles() in load() destroys them.
            # Face winding (already flipped above) gives correct auto-calc normals.

            lt and df and setup_material_textures(m2s, df, lhd, iad)
            vlog.log(f"Created mesh: {mn} ({len(verts)} verts, {len(faces)} faces)")

    return co

def setup_material_textures(m2s, df, lhd=True, iad=False):
    mf = os.path.join(df, "graphics", "_materials")
    
    for mat, mn in m2s:
        xfn = os.path.basename(mn)
        if not xfn.lower().endswith('.xbm'):
            xfn = xfn + '.xbm'
        
        xp = os.path.join(mf, xfn)
        if os.path.exists(xp):
            vlog.log(f"\nLoading XBM: {xfn}")
            xd = XBMParser.parse(xp, lhd)
            if xd:
                BlenderMaterialSetup.setup_material(mat, xd, df, lhd, iad)
