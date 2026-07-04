"""Bone weight handling for XBG meshes.

Two responsibilities:

  * `remap_skin_indices()` — convert per-vertex palette-slot indices into
    global bone IDs.  XBG vertices store indices 0..47 into a 48-entry
    bone palette (from DNKS); skinning in Blender needs the actual bone
    IDs from the armature.

  * `apply_vertex_weights()` — paint the resulting weights/IDs onto an
    object as Blender vertex groups.  Groups are created lazily so that
    only bones with at least one non-zero influence on this object are
    materialised (avoids a vertex group per bone for every submesh).

Both functions are called from `XBGBlenderImporter.create_meshes` /
`XBGParser._remap_skin_indices` in import_xbg.py.  Keep this module the
single source of truth for skinning-side import logic — do not duplicate
the loops inline in the importer.
"""


def remap_skin_indices(skin_indice_list, mat_list_info, sub_mesh_list):
    """Rewrite `skin_indice_list` in-place, palette-slot → global bone ID.

    skin_indice_list : list of (i0, i1, i2, i3) tuples, length = vertex count.
                       Mutated in place.
    mat_list_info    : ordered list of (vb_idx, lod_grp, DNKS_POS,
                       idx_offset, idx_count) tuples describing how the
                       vertex buffer is sliced across DNKS submeshes.  The
                       ordering MUST match the order vertices appear in
                       the buffer.
                       NOTE on the 3rd field: it is the POSITIONAL INDEX
                       into `sub_mesh_list[lod_grp]` (which is ordered by
                       SDOL position per `parse_dnks_chunk`).  The caller
                       in import_xbg.py._remap_skin_indices substitutes
                       each mesh's true SDOL position (mesh.name_index)
                       in place of the raw `info[2]` (sub_idx VALUE) so
                       this lookup works correctly on injected files
                       where the sub_idx value no longer equals position.
    sub_mesh_list    : list-of-lists of SubMesh (from parse_dnks_chunk).
                       Each SubMesh has `.header_data` (header_data[5] is
                       the per-submesh vertex count) and `.bone_data`
                       (48-entry palette; -1 marks an unused slot).
    """
    if not skin_indice_list:
        return

    try:
        from ..Core.debug import TraceLogger
    except Exception:
        TraceLogger = None

    vert_id_start = 0
    for slot_i, info in enumerate(mat_list_info):
        lod_grp, dnks_pos = info[1], info[2]
        if lod_grp >= len(sub_mesh_list):
            if TraceLogger is not None:
                TraceLogger.debug(
                    f"  [weights] skip slice {slot_i}: lod_grp {lod_grp} "
                    f">= sub_mesh_list len {len(sub_mesh_list)}",
                    event="weights_skip_lod",
                    data={"slot": slot_i, "lod_grp": int(lod_grp),
                          "sub_mesh_list_len": len(sub_mesh_list)})
            continue
        lod_subs = sub_mesh_list[lod_grp]
        if dnks_pos >= len(lod_subs):
            if TraceLogger is not None:
                TraceLogger.debug(
                    f"  [weights] skip slice {slot_i}: dnks_pos {dnks_pos} "
                    f">= lod_subs len {len(lod_subs)}",
                    event="weights_skip_pos",
                    data={"slot": slot_i, "dnks_pos": int(dnks_pos),
                          "lod_subs_len": len(lod_subs)})
            continue

        submesh = lod_subs[dnks_pos]
        count = submesh.header_data[5]
        palette = submesh.bone_data
        end = min(vert_id_start + count, len(skin_indice_list))

        lp = len(palette)                  # constant per submesh — hoist out of loop
        for v_idx in range(vert_id_start, end):
            skin_indice_list[v_idx] = tuple(
                (palette[r] if r < lp and palette[r] != -1 else 0)
                for r in skin_indice_list[v_idx]
            )
        vert_id_start += count


def apply_vertex_weights(obj, armature_obj, weights, skin_indices):
    """Paint per-vertex bone weights onto `obj` as Blender vertex groups.

    weights        : list of (w0, w1, w2, w3) uint8 tuples (XBG-encoded; the
                     game's runtime divides each by 255 to get the weight).
    skin_indices   : list of (b0, b1, b2, b3) GLOBAL bone IDs.  Must have
                     been remapped from palette slots via
                     `remap_skin_indices()` first.

    Vertex groups are created lazily — only bones that actually influence
    this object get a group.  Existing groups with the same name are
    reused, so calling this twice on the same object is safe.
    """
    if not armature_obj or not weights or not skin_indices:
        return

    bones = armature_obj.data.bones
    nbones = len(bones)                     # constant — hoist out of the loops

    # Pass 1: materialise the vertex groups we'll need (one new() call per
    # unique bone instead of one per vertex influence).
    vg_cache = {}
    for bone_ids in skin_indices:
        for bone_idx in bone_ids:
            if bone_idx < nbones and bone_idx not in vg_cache:
                bn = bones[bone_idx].name
                vg_cache[bone_idx] = (obj.vertex_groups.get(bn)
                                      or obj.vertex_groups.new(name=bn))

    # Pass 2: paint weights.  uint8 -> float by /255.0.
    for vert_idx, (weight_data, bone_data) in enumerate(zip(weights, skin_indices)):
        for bone_idx, weight in zip(bone_data, weight_data):
            if weight > 0 and bone_idx in vg_cache:
                vg_cache[bone_idx].add([vert_idx], weight / 255.0, 'REPLACE')
