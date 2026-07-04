"""Bone-weight + palette handling for XBG mesh EXPORT (inject side).

The inject counterpart to import_weights.py. Builds the per-vertex weight/index
maps the encoder writes back into the file: resolving each object's bone palette,
mapping Blender vertex-group weights onto the 48-bone SULC palette, re-binding
foreign/edited vertices by position, and rigid-binding leftovers. Pulled out of
inject_xbg.py so the weight logic lives in one findable place.

`_get_armature` lives here (its primary users are these functions); inject_xbg
imports it back from this module.
"""

import bmesh
import mathutils

from ..Core.debug  import VerboseLogger, TraceLogger
from .binary_avatar import LE
from .chunks_avatar import parse_dnks_for_palette
from .normals_avatar        import build_tbn_lookups


def _get_armature(obj):
    """Find and return the armature object linked to obj via modifier or parent."""
    for mod in obj.modifiers:
        if mod.type == 'ARMATURE' and mod.object:
            return mod.object
    if obj.parent and obj.parent.type == 'ARMATURE':
        return obj.parent
    return None


def _build_weight_map(obj, bone_palette):
    """
    Build a per-vertex weight map: {vertex_index: (wb_list, ib_list)}
    where wb_list = 4 uint8 bone weights, ib_list = 4 uint8 palette slots.

    bone_palette : list of 48 global bone IDs from `chunks.parse_dnks_for_palette()`,
                   or None to use the vertex group index directly as the palette slot.

    The vertex group index (obj.vertex_groups ordering) corresponds to the
    global bone index in the armature (the same order used by weights.py's
    apply_vertex_weights).  `parse_dnks_for_palette` returns a list where
    palette_slot -> global_bone_id.  We invert it so global_bone_id -> slot.

    IMPORTANT: bone_palette must be the palette for THIS specific submesh/object,
    not always submesh-0.  Use the xbg_bone_palette property stored on the object
    at import time to get the correct per-object palette.
    """
    if not obj.vertex_groups:
        TraceLogger.debug(
            f"  [_build_weight_map] '{obj.name}': no vertex groups → empty map",
            event="build_weight_map_no_groups",
            data={"object": obj.name})
        return {}

    armature = _get_armature(obj)

    if bone_palette:
        # Build bone name -> global bone ID from armature bone order
        bone_name_to_global_id = {}
        if armature:
            for i, bone in enumerate(armature.data.bones):
                bone_name_to_global_id[bone.name] = i

        # Invert: global_bone_id -> palette_slot  (skip -1 unused entries)
        reverse_palette = {bid: slot
                          for slot, bid in enumerate(bone_palette) if bid != -1}

        # vertex_group_index -> palette_slot via bone name -> global ID -> slot
        group_to_slot = {}
        for vg in obj.vertex_groups:
            global_id = bone_name_to_global_id.get(vg.name, vg.index)
            group_to_slot[vg.index] = reverse_palette.get(global_id, 0)
    else:
        # No palette available -- treat group index as palette slot directly
        group_to_slot = {vg.index: vg.index for vg in obj.vertex_groups}

    weight_map = {}
    _unweighted = 0
    _max_influences = 0
    _truncated = 0
    _influence_hist = [0, 0, 0, 0, 0]   # 0,1,2,3,4+ influences per vert
    for v in obj.data.vertices:
        influences = [(ge.weight, ge.group)
                     for ge in v.groups if ge.weight > 0.001]
        if not influences:
            _unweighted += 1
            _influence_hist[0] += 1
            continue
        n_inf = len(influences)
        if n_inf > _max_influences:
            _max_influences = n_inf
        _influence_hist[min(n_inf, 4)] += 1
        if n_inf > 3:
            _truncated += 1

        influences.sort(key=lambda x: -x[0])  # heaviest first
        # Cap at 3 — the game's vertex skinning is 3-BONE. skinning.inc.fx
        # ApplySkinning() only reads skin0.x/y/z * BlendMatrices[skin1.z/y/x];
        # skin0.w (the 4th weight byte) is NEVER referenced by the GPU. Writing
        # a 4th influence silently under-weights the vertex: the 3 weights the
        # game DOES read then sum to <255, so the vert sags toward the model
        # origin. Dropping the smallest influence and renormalising the top 3 to
        # sum 255 keeps the vertex fully weighted.
        influences = influences[:3]

        total = sum(w for w, _ in influences)
        if total < 1e-6:
            continue

        wb = [0, 0, 0, 0]   # 4th weight byte stays 0 (unused by the 3-bone path)
        ib = [0, 0, 0, 0]
        accumulated = 0
        _last = len(influences) - 1
        for i, (w, group_idx) in enumerate(influences):
            if i < _last:
                wbyte = round(w / total * 255)
                accumulated += wbyte
            else:
                # Last USED weight absorbs rounding so the (≤3) weights the game
                # reads always sum to exactly 255 = fully weighted.
                wbyte = 255 - accumulated
            wb[i] = max(0, min(255, wbyte))
            ib[i] = max(0, min(255, group_to_slot.get(group_idx, 0)))

        weight_map[v.index] = (wb, ib)

    # ── Build-weight-map stats ─────────────────────────────────────────
    TraceLogger.kvblock(
        f"_build_weight_map  '{obj.name}'",
        [
            ("vertex_groups",          len(obj.vertex_groups)),
            ("bone_palette_supplied",  bool(bone_palette)),
            ("armature",               armature.name if armature else None),
            ("source_verts",           len(obj.data.vertices)),
            ("weighted_verts",         len(weight_map)),
            ("unweighted_verts",       _unweighted),
            ("max_influences_seen",    _max_influences),
            ("verts_truncated_to_4",   _truncated),
            ("influence_hist",         {str(i): _influence_hist[i] for i in range(5)}),
            ("group_to_slot_size",     len(group_to_slot)),
        ],
        tier="DEBUG",
        event="build_weight_map_stats")
    return weight_map


def _build_slice_palette_and_weights(obj, tri_mesh, armature):
    """Per-split-submesh skinning.

    Every XBG submesh carries its OWN 48-bone palette. When we split a
    joined object by material, each slice must get a palette built from
    the bones ITS vertices actually use — inheriting one shared palette
    (the old behaviour) sends any bone outside that 48-set to slot 0,
    which pins those verts to one bone (the stretched/exploded parts).

    Returns (palette48, weight_map) where:
      palette48  : list[48] of global bone IDs (armature bone order;
                   same space _build_weight_map uses for originals),
                   padded with -1.
      weight_map : {tri_vertex_index: ([w0..3]u8, [s0..3]u8)} with slot
                   indices INTO palette48.
    Falls back to ({}, {}) if there are no vertex groups / armature.
    """
    if not obj.vertex_groups or armature is None:
        return [-1] * 48, {}

    name_to_gid = {b.name: i for i, b in enumerate(armature.data.bones)}
    # ONLY map vertex groups whose name is a REAL armature bone. A donor
    # mesh joined in often keeps vertex groups from its original rig
    # (names that aren't this skeleton's bones). Those must be dropped —
    # the old `…get(vg.name, vg.index)` fallback turned them into bogus
    # bone slots, inflating the palette to >95 "bones" and stretching
    # the mesh. Unknown groups are simply ignored and the remaining
    # real-bone weights are renormalised per vertex.
    vg_gid = {vg.index: name_to_gid[vg.name]
              for vg in obj.vertex_groups if vg.name in name_to_gid}
    dropped = len(obj.vertex_groups) - len(vg_gid)
    if dropped:
        VerboseLogger.log(f"  [inject]   dropped {dropped} non-skeleton vertex "
                          f"group(s) from '{obj.name}' (donor-rig leftovers)")

    # original-vertex -> sorted top-3 [(weight, gid), ...]  (real bones).
    # Cap 3, not 4: the game skins with 3 bones (skinning.inc.fx ApplySkinning
    # reads only skin0.x/y/z); a 4th influence is ignored by the GPU and
    # under-weights the vertex.
    src = obj.data.vertices
    infl = {}
    for v in src:
        lst = [(ge.weight, vg_gid[ge.group])
               for ge in v.groups
               if ge.weight > 0.001 and ge.group in vg_gid]
        if lst:
            lst.sort(key=lambda x: -x[0])
            infl[v.index] = lst[:3]

    # match each slice vertex to its source vertex by position
    def key(co):
        return (round(co.x, 5), round(co.y, 5), round(co.z, 5))
    pos_to_idx = {}
    for v in src:
        pos_to_idx.setdefault(key(v.co), v.index)
    kd = None

    # A joined foreign mesh (e.g. partybeach) often carries ONLY its
    # donor rig's vertex groups — all dropped above — so its verts have
    # zero real-bone influence and would collapse to the origin in-game
    # (the stretched/exploded sheets). Build a KD-tree over only the
    # source verts that DO have real Kendra-bone weights so unskinned
    # verts can borrow the closest skinned point (automatic proximity
    # skin-transfer): the foreign geometry then rigidly follows the
    # nearest body part instead of exploding.
    import mathutils
    skinned_ids = list(infl.keys())
    skin_kd = None
    if skinned_ids:
        skin_kd = mathutils.kdtree.KDTree(len(skinned_ids))
        for oi in skinned_ids:
            skin_kd.insert(src[oi].co, oi)
        skin_kd.balance()
    borrowed = 0

    tri_infl = {}
    gid_weight = {}
    native = 0          # slice verts that map to a REAL skeleton weight
    for tv in tri_mesh.vertices:
        oi = pos_to_idx.get(key(tv.co))
        if oi is None:
            if kd is None:
                kd = mathutils.kdtree.KDTree(len(src))
                for v in src:
                    kd.insert(v.co, v.index)
                kd.balance()
            _, oi, _ = kd.find(tv.co)
        lst = infl.get(oi, [])
        if lst:
            native += 1
        elif skin_kd is not None:
            _, sni, _ = skin_kd.find(tv.co)
            lst = infl.get(sni, [])
            if lst:
                borrowed += 1
        tri_infl[tv.index] = lst
        for w, g in lst:
            gid_weight[g] = gid_weight.get(g, 0.0) + w

    # RIGID-BIND fully-foreign slices.  If NOT ONE vertex of this slice
    # has a real skeleton weight (a prop/tree joined in with no Kendra-
    # bone vertex groups), the per-vertex nearest-point borrow scatters
    # it across whatever limb bones happen to be closest (hand/fingers)
    # — an animated limb then whips the rigid prop around and crushes it
    # to nothing in-game.  Instead bind the ENTIRE slice 100% to ONE
    # stable bone.
    #
    # CRITICAL: that bone MUST be a real DEFORM bone the engine actually
    # skins (i.e. one present in MB2O / used by the host's other
    # submeshes). A control bone like 'Root' has NO inverse-bind matrix
    # in MB2O (kendra MB2O = 87 entries, 'Root'=bone3 used by zero body
    # submeshes) → the engine skins the prop with a garbage matrix →
    # it's transformed off-screen / to zero → INVISIBLE in-game while
    # looking fine in Blender (which uses EDON, not MB2O). So pick the
    # host mesh's DOMINANT weighted bone: it is MB2O-valid by
    # construction (the body skins to it and renders). Override with
    # obj['xbg_rigid_bone'] = '<bone name>' (must also be a deform bone).
    if native == 0 and len(tri_infl) > 0:
        bones = armature.data.bones
        rb_name = obj.get('xbg_rigid_bone')
        rb_gid = name_to_gid.get(rb_name) if rb_name else None
        if rb_gid is None:
            # dominant bone across ALL of the host object's real weights
            from collections import Counter
            wsum = Counter()
            for _vi, _lst in infl.items():
                for _w, _g in _lst:
                    wsum[_g] += _w
            if wsum:
                rb_gid = wsum.most_common(1)[0][0]
                rb_name = (bones[rb_gid].name
                           if rb_gid < len(bones) else f'gid{rb_gid}')
        if rb_gid is None:
            for cand in ('Pelvis', 'pelvis', 'Spine', 'Spine1',
                         'Hips', 'Bip01_Pelvis', 'Bip01'):
                if cand in name_to_gid:
                    rb_gid, rb_name = name_to_gid[cand], cand
                    break
        if rb_gid is None:
            rb_gid = 0
            rb_name = bones[0].name if len(bones) else '<bone0>'
        for vidx in tri_infl:
            tri_infl[vidx] = [(1.0, rb_gid)]
        gid_weight = {rb_gid: float(len(tri_infl))}
        borrowed = 0
        VerboseLogger.log(f"  [inject]   '{obj.name}' has NO skeleton weights — "
                          f"RIGID-BIND whole slice 100% to bone '{rb_name}' "
                          f"(gid {rb_gid}); set obj['xbg_rigid_bone'] to override")
    elif borrowed:
        VerboseLogger.log(f"  [inject]   {borrowed} unskinned vert(s) in '{obj.name}' "
                          f"borrowed weights from nearest skinned point "
                          f"({native} native) — foreign mesh only partly weighted")

    # palette = bones used by this slice, most-influential first, cap 48
    ordered = sorted(gid_weight, key=lambda g: -gid_weight[g])
    if len(ordered) > 48:
        VerboseLogger.log(f"  [inject]   WARNING: slice uses {len(ordered)} bones "
                          f"(>48) — keeping the 48 most-weighted")
    palette = ordered[:48]
    gid_to_slot = {g: s for s, g in enumerate(palette)}
    palette48 = palette + [-1] * (48 - len(palette))

    weight_map = {}
    for vidx, lst in tri_infl.items():
        if not lst:
            continue
        total = sum(w for w, _ in lst) or 1.0
        wb = [0, 0, 0, 0]   # 4th weight byte stays 0 (game reads only 3 bones)
        ib = [0, 0, 0, 0]
        acc = 0
        _last = len(lst) - 1
        for i, (w, g) in enumerate(lst):
            if i < _last:
                wbyte = round(w / total * 255)
                acc += wbyte
            else:
                # last USED weight absorbs rounding -> the ≤3 weights sum to 255
                wbyte = 255 - acc
            wb[i] = max(0, min(255, wbyte))
            ib[i] = max(0, min(47, gid_to_slot.get(g, 0)))
        weight_map[vidx] = (wb, ib)

    # ── Slice palette + weights stats ──────────────────────────────────
    bones = armature.data.bones if armature else []
    palette_named = [bones[g].name if 0 <= g < len(bones) else
                     (f"-1" if g == -1 else f"gid{g}") for g in palette48]
    TraceLogger.kvblock(
        f"_build_slice_palette_and_weights  '{obj.name}'",
        [
            ("source_verts",            len(src)),
            ("slice_verts",             len(tri_mesh.vertices)),
            ("non_skel_groups_dropped", dropped),
            ("verts_natively_skinned",  native),
            ("verts_borrowed_nn",       borrowed),
            ("unique_bones_used",       len(ordered)),
            ("palette_filled",          sum(1 for g in palette48 if g >= 0)),
            ("palette_truncated_>48",   max(0, len(ordered) - 48)),
            ("palette_top8",            palette_named[:8]),
            ("output_weight_map_size",  len(weight_map)),
            ("coverage",                f"{len(weight_map)}/{len(tri_mesh.vertices)}"),
        ],
        tier="DEBUG", event="slice_palette_stats")
    return palette48, weight_map


def _remap_weights_by_position(obj, tri_mesh, weight_map):
    """Re-key an object-vertex-indexed weight_map onto a split sub-mesh.

    `_triangulate_and_split_by_material` re-indexes vertices (slice
    vertex 0 != object vertex 0), but `weight_map` is keyed by the
    ORIGINAL obj vertex index. Indexing it with the slice's v.index
    therefore reads the WRONG vertex's bone weights -> scrambled
    skinning (the in-game explosion). Split slices are exact position
    copies of the source verts, so we match by 3D position — the same
    technique build_tbn_lookups already uses for tangents.

    Returns a NEW dict keyed by tri_mesh vertex index. If nothing needs
    remapping (counts already align) the original map is returned as-is.
    """
    if not weight_map:
        TraceLogger.debug(
            f"  [_remap_weights] '{obj.name}': empty source map → returning empty",
            event="remap_weights_empty_source",
            data={"object": obj.name,
                  "tri_mesh_verts": len(tri_mesh.vertices)})
        return weight_map
    src = obj.data.vertices
    if len(tri_mesh.vertices) == len(src):
        TraceLogger.debug(
            f"  [_remap_weights] '{obj.name}': counts equal "
            f"({len(src)}={len(tri_mesh.vertices)}) → passthrough",
            event="remap_weights_passthrough",
            data={"object": obj.name, "verts": len(src)})
        return weight_map               # non-split: indices already match

    def key(co):
        return (round(co.x, 5), round(co.y, 5), round(co.z, 5))

    pos_to_w = {}
    for i, v in enumerate(src):
        w = weight_map.get(i)
        if w is not None:
            pos_to_w.setdefault(key(v.co), w)

    out = {}
    miss = 0
    nn_first_weighted    = 0   # closest hit was already weighted (typical)
    nn_fallback_weighted = 0   # closest hit was UNWEIGHTED; walked further
    nn_no_weighted_in_k  = 0   # NONE of the k nearest were weighted → drop
    kd = None

    # BUG FIX (2026-05) — "fully-foreign false positive":
    # ------------------------------------------------------------------
    # Previous code did `kd.find()` (single nearest) and then
    # `weight_map.get(idx)`.  The KD-tree contains EVERY source vertex,
    # including unweighted ones (stray verts the user never assigned to a
    # vertex group, plus runtime helpers like loose detail edges).  When
    # the nearest source vert happens to be unweighted, `.get()` returns
    # None and the slice vert is dropped from `out`.
    #
    # If enough verts drop out, the caller's coverage check (`len(out)`)
    # hits zero, `_rigid_bind_foreign_into_palette` declares the slice
    # "fully foreign", and binds the WHOLE slice (potentially Kendra's
    # own geometry) rigidly to a single Pelvis-class bone — every vert
    # on that submesh ends up at the pelvis after skinning.
    #
    # Fix: walk up to NN_K nearest source verts in distance order and
    # pick the first one with a weight in `weight_map`.  A k of 8 is
    # plenty in practice (split-by-material slices are dense position-
    # equal copies, so the nearest weighted source vert is almost always
    # one of the very first hits).  If none of k are weighted we fall
    # through to the original drop behaviour rather than guessing — that
    # legitimately means there's no nearby weighted source and rigid
    # binding is the right next step.
    NN_K = 8

    for tv in tri_mesh.vertices:
        w = pos_to_w.get(key(tv.co))
        if w is None:
            if kd is None:
                import mathutils
                kd = mathutils.kdtree.KDTree(len(src))
                for i, v in enumerate(src):
                    kd.insert(v.co, i)
                kd.balance()
            miss += 1
            picks = kd.find_n(tv.co, min(NN_K, len(src))) if len(src) else []
            for i_pick, (_co, idx, _dist) in enumerate(picks):
                cand = weight_map.get(idx)
                if cand is not None:
                    w = cand
                    if i_pick == 0:
                        nn_first_weighted += 1
                    else:
                        nn_fallback_weighted += 1
                    break
            if w is None:
                nn_no_weighted_in_k += 1
        if w is not None:
            out[tv.index] = w
    if miss:
        VerboseLogger.log(
            f"  [inject]   weight remap: {miss} vert(s) via nearest "
            f"(no exact position match); "
            f"first-hit weighted={nn_first_weighted}, "
            f"fallback-hit weighted={nn_fallback_weighted}, "
            f"all-{NN_K}-NN unweighted={nn_no_weighted_in_k}")
    TraceLogger.kvblock(
        f"_remap_weights_by_position  '{obj.name}'",
        [
            ("source_verts",          len(src)),
            ("source_weighted",       len(weight_map)),
            ("tri_mesh_verts",        len(tri_mesh.vertices)),
            ("exact_pos_matches",     len(tri_mesh.vertices) - miss),
            ("kdtree_lookups",        miss),
            ("nn_k",                  NN_K),
            ("nn_first_weighted",     nn_first_weighted),
            ("nn_fallback_weighted",  nn_fallback_weighted),
            ("nn_no_weighted_in_k",   nn_no_weighted_in_k),
            ("output_weighted",       len(out)),
            ("output_coverage",       f"{len(out)}/{len(tri_mesh.vertices)}"),
        ],
        tier="DEBUG", event="remap_weights_stats")
    return out


def _rigid_bind_foreign_into_palette(obj, tri_mesh, weight_map,
                                     bone_palette, armature):
    """Keep DNKS palettes == the ORIGINAL imported palette (so MB2O,
    which is indexed by palette order, stays valid) while still making
    a fully-foreign joined slice (a prop/tree/monkey with no Kendra
    weights) render.

    ROOT CAUSE (proven by user A/B test + structural diff): the old
    split path synthesised a NEW per-slice palette; MB2O is indexed by
    palette order and is never rebuilt, so every split submesh looked
    up the wrong inverse-bind matrix -> whole mesh invisible in-game.
    The no-split path keeps the original p48 palette and renders. So
    split must ALSO keep the original palette.

    If the slice already has weights (covered>0) it is left untouched
    (kendra body slices map cleanly into the original palette — same as
    the working no-split path). If it has NONE (foreign prop), bind the
    whole slice 100% to ONE slot of the ORIGINAL palette whose bone is
    a stable deform bone (so it's MB2O-valid by construction — that
    bone already had a working inverse-bind in the source submesh).
    Returns the (possibly augmented) weight_map.
    """
    total = len(tri_mesh.vertices)
    covered = len(weight_map) if weight_map else 0
    if covered > 0 or total == 0:
        TraceLogger.debug(
            f"  [_rigid_bind] '{obj.name}': covered={covered}/{total} → "
            f"no rigid-bind needed",
            event="rigid_bind_skipped",
            data={"object": obj.name, "covered": covered, "total": total})
        return weight_map                       # not fully foreign

    # choose a palette SLOT (index into the original 48-palette) for a
    # stable deform bone. bone_palette = list[48] global bone IDs.
    valid = [(s, g) for s, g in enumerate(bone_palette)
             if g is not None and g >= 0]
    if not valid:
        TraceLogger.info(
            f"  [_rigid_bind] '{obj.name}': NO valid palette entries — "
            f"returning unmodified weight_map ({total} verts will be UNWEIGHTED)",
            event="rigid_bind_no_palette",
            data={"object": obj.name, "total_verts": total})
        return weight_map                       # no usable palette
    slot = valid[0][0]
    name = None
    bone_resolve_trail = []
    if armature is not None:
        bones = armature.data.bones
        gid_name = {i: b.name for i, b in enumerate(bones)}
        # explicit override, else first Pelvis/Spine/Hips IN the palette
        ov = obj.get('xbg_rigid_bone')
        pref = ([ov] if ov else []) + ['Pelvis', 'pelvis', 'Spine',
                'Spine1', 'Hips', 'Bip01_Pelvis', 'Bip01']
        chosen = None
        for cand in pref:
            found_in_palette = False
            for s, g in valid:
                if gid_name.get(g) == cand:
                    chosen = (s, cand)
                    found_in_palette = True
                    break
            bone_resolve_trail.append({"candidate": cand,
                                        "found_in_palette": found_in_palette})
            if chosen:
                break
        if chosen:
            slot, name = chosen
        else:
            slot = valid[0][0]
            name = gid_name.get(valid[0][1], f'gid{valid[0][1]}')
            bone_resolve_trail.append({"candidate": "<fallback to palette[0]>",
                                        "slot": slot, "bone": name})
    wm = dict(weight_map) if weight_map else {}
    for tv in tri_mesh.vertices:
        wm[tv.index] = ([255, 0, 0, 0], [slot, 0, 0, 0])
    VerboseLogger.log(f"  [inject]   '{obj.name}' fully foreign (no skeleton "
                      f"weights) — RIGID-BIND 100% to ORIGINAL-palette slot {slot}"
                      f" (bone '{name}'); palette preserved so MB2O stays valid")
    TraceLogger.kvblock(
        f"_rigid_bind_foreign_into_palette  '{obj.name}'",
        [
            ("verts_rigid_bound",   total),
            ("chosen_slot",         slot),
            ("chosen_bone",         name),
            ("explicit_override",   obj.get('xbg_rigid_bone')),
            ("palette_valid_count", len(valid)),
            ("palette_first_4",     valid[:4]),
            ("bone_resolution",     bone_resolve_trail),
        ],
        tier="DEBUG", event="rigid_bind_done")
    return wm


def _get_object_bone_palette(obj, file_data, target_lod, endian=LE):
    """
    Return the correct 48-entry bone palette for this specific object.

    Priority:
      1. xbg_bone_palette stored on the object during import (most accurate --
         matches the exact submesh this object came from).
      2. DNKS submesh-0 fallback (old behaviour, works for single-part meshes).

    This is the fix for the weights bug: multi-part NPCs have a different
    palette per submesh.  Using submesh-0's palette for the head (submesh 2)
    maps every bone to the wrong slot.

    `endian` propagates to the DNKS scan when the stored palette is missing.
    """
    stored = obj.get("xbg_bone_palette")
    if stored and len(stored) == 48:
        palette = list(stored)
        valid   = sum(1 for b in palette if b != -1)
        VerboseLogger.log(f"    Bone palette: {valid}/48 valid slots (from object '{obj.name}')")
        return palette

    # Fallback: parse DNKS submesh 0
    palette = parse_dnks_for_palette(file_data, target_lod, 0, endian)
    if palette:
        valid = sum(1 for b in palette if b != -1)
        VerboseLogger.log(f"    Bone palette: {valid}/48 valid slots (DNKS fallback for '{obj.name}')")
    else:
        VerboseLogger.log(f"    Bone palette: not found in DNKS for '{obj.name}' -- group index used as slot")
    return palette
