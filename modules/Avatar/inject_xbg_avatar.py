"""inject_xbg.py  -  XBG Mesh Injection Module v2.1
====================================================
v2.1 changes:
  - Weights fix: each injected object now uses its own DNKS bone palette
    (stored on the object as xbg_bone_palette during import).  Previously
    the palette was always read from submesh-0, causing wrong slot mappings
    for every submesh beyond the first (head, hands, tail, etc.).
  - flip_normals removed: normals are exported exactly as they face in Blender.
  - Expand Bounds no longer creates a file copy; it sets override_game_scale.

Original v2.0 features retained:
  - Multi-object / primitives support
  - Persistent inject panel  (session metadata at scene level)
  - Multi-object injection
  - Real bone-weight painting via vertex groups
  - Vertex-color support
  - Material-based submesh splitting
  - Auto-detect single vs multi-part structure
  - Auto-scale to 16-bit bounds
  - PMCP scale override
  - XOBB / HPSB bounds patch
  - DNKS face/vert count patch
"""

import bpy
import bmesh
import struct
import math
import os
import mathutils
from collections import deque

from ..Core.debug         import VerboseLogger, TraceLogger
from .import_mesh_avatar   import VertexFlags
from .bounds_avatar import clamp_to_16bit
from .binary_avatar import detect_endian_from_bytes, LE, BE
from .chunks_avatar import (
    find_chunk, find_all_chunks,
    patch_pmcp, patch_dnks, patch_bounds,
    parse_dnks_for_palette,
    parse_ltmr_names, build_ltmr_chunk, patch_dnks_matids,
    parse_dnks_full, build_dnks_lod_block, build_dnks_chunk,
    _dnks_template_hfields,
    parse_dnks_trailing, build_dnks_trailing, resize_dnks_trailing_for_lod,
    update_dnks_trailing_lod_bbox,
)
from .sdol_avatar          import parse_sdol, build_sdol_chunk
from .normals_avatar       import (
    build_tbn_lookups, compute_tangents_from_uvs,
)
from .vertex_colors_avatar import build_vertex_color_map
from .export_weights_avatar import (
    _get_armature, _build_weight_map, _build_slice_palette_and_weights,
    _remap_weights_by_position, _rigid_bind_foreign_into_palette,
    _get_object_bone_palette,
)
from .export_mesh_avatar import (
    _encode_vertices, _null_vertex, _triangulate_and_split_by_material,
    _split_mesh_by_face_budget, _build_index_buffer,
)


# ============================================================
# Bounds / scale helpers
# ============================================================

def calculate_required_scale(obj, pos_scale, import_mesh_only=False):
    """
    Check whether the mesh's vertices fit within the signed 16-bit range
    after encoding with pos_scale.

    Returns (needs_scaling, scale_factor, info_string).
      needs_scaling  : True if any coordinate exceeds 32767
      scale_factor   : multiplier to bring the worst axis just inside the limit
      info_string    : human-readable description of the worst offender
    """
    mesh = obj.data
    obj_rotation = obj.rotation_euler.copy()

    if import_mesh_only and abs(obj_rotation.z - math.radians(180)) < 0.01:
        z_rot_inv = mathutils.Matrix.Rotation(-math.radians(180), 4, 'Z')
    elif abs(obj_rotation.z) > 0.01:
        z_rot_inv = mathutils.Matrix.Rotation(-obj_rotation.z, 4, 'Z')
    else:
        z_rot_inv = mathutils.Matrix.Identity(4)

    max_value = 32767
    inv_scale = 1.0 / pos_scale
    max_x = max_y = max_z = 0

    for vertex in mesh.vertices:
        lc = vertex.co
        rc = z_rot_inv @ mathutils.Vector((lc.x, lc.y, lc.z, 1.0))
        max_x = max(max_x, abs(int(rc.x * inv_scale)))
        max_y = max(max_y, abs(int(rc.y * inv_scale)))
        max_z = max(max_z, abs(int(rc.z * inv_scale)))

    max_coord = max(max_x, max_y, max_z)
    if max_coord > max_value:
        scale_factor = max_value / max_coord
        axis = "X" if max_coord == max_x else ("Y" if max_coord == max_y else "Z")
        return True, scale_factor, f"{axis} axis: {max_coord} (limit: {max_value})"
    return False, 1.0, "All coordinates within bounds"


# Mesh prep + vertex/index encoding live in export_mesh.py (imported above).


# ============================================================
# Main injector class
# ============================================================

class XBGMeshInjector:

    def inject(self, context, objects, output_path, target_lod=0,
               meta=None,
               override_game_scale=False, target_game_scale=1.0,
               ignore_limits=False,
               inject_vertex_colors=True,
               generate_neutral_vertex_colors=True,
               inject_bone_weights=False,
               inject_materials=False,
               force_per_submesh_vb=False):

        # -- Build/version banner (proves the reloaded code is running) ----
        VerboseLogger.log("\n" + "#" * 64)
        VerboseLogger.log("# XBG INJECT  —  build: TRACE-LOGGING-v1  (verbose)")
        VerboseLogger.log("#" * 64)
        if TraceLogger.trace_enabled() and TraceLogger.stream_path():
            VerboseLogger.log(
                f"# Live trace stream: {TraceLogger.stream_path()}\n"
                f"#   (rewritten every export; the .jsonl file gets saved "
                f"next to your Save Log target)")
        TraceLogger.debug(
            f"[trace] inject() entry: trace_enabled={TraceLogger.trace_enabled()}  "
            f"output={output_path}",
            event="inject_entry",
            data={
                "output_path": str(output_path),
                "target_lod": target_lod,
                "override_game_scale": bool(override_game_scale),
                "target_game_scale": float(target_game_scale),
                "ignore_limits": bool(ignore_limits),
                "inject_vertex_colors": bool(inject_vertex_colors),
                "generate_neutral_vertex_colors": bool(generate_neutral_vertex_colors),
                "inject_bone_weights": bool(inject_bone_weights),
                "inject_materials": bool(inject_materials),
                "force_per_submesh_vb": bool(force_per_submesh_vb),
                "n_objects_in": len(objects or []),
            })

        # -- Validate -------------------------------------------------------
        mesh_objects = [o for o in objects if o and o.type == 'MESH']
        if not mesh_objects:
            return {'CANCELLED'}, "No mesh objects in the provided list."
        VerboseLogger.log(f"[inject] mesh_objects ({len(mesh_objects)}): "
                          + ", ".join(o.name for o in mesh_objects))
        for o in mesh_objects:
            slots = []
            slot_records = []
            for sl_idx, sl in enumerate(o.material_slots):
                m = sl.material
                if not m:
                    slots.append("<empty>")
                    slot_records.append({"idx": sl_idx, "name": None, "tags": []})
                    continue
                tag = []
                if m.get('xbg_exported'):
                    tag.append("exported")
                if m.get('xbg_source'):
                    tag.append("game")
                slots.append(f"{m.name}{('['+','.join(tag)+']') if tag else ''}")
                slot_records.append({"idx": sl_idx, "name": m.name, "tags": list(tag)})
            VerboseLogger.log(f"  [inject]   '{o.name}' slots: {slots}  "
                              f"xbg_data.slot={ (o.get('xbg_data') or {}).get('sdol_submesh_slot') }")

            # ── Pre-encode object inspection (DEBUG): geometry / armature
            #    summary that lets us reproduce the input from the log alone.
            try:
                arm_obj = _get_armature(o)
                arm_info = (f"{arm_obj.name} ({len(arm_obj.data.bones)} bones)"
                            if arm_obj else "<none>")
                mod_stack = [f"{m.type}:{m.name}" for m in o.modifiers]
                mw = o.matrix_world
                me = o.data
                # World-space AABB for the object's verts (after rot inversion
                # applied by patch_bounds for consistency with stored bbox).
                # NOTE: `import_mesh_only` isn't bound yet at this point (it's
                # resolved from `meta` further down), so read it directly off
                # the object's own xbg_data — this fixes the UnboundLocalError
                # that was killing every object's pre-encode trace block.
                _imo_dbg = bool((o.get('xbg_data') or {}).get('import_mesh_only', False))
                rz = o.rotation_euler.z
                if _imo_dbg and abs(rz - math.radians(180)) < 0.01:
                    rot_inv_dbg = mathutils.Matrix.Rotation(-math.radians(180), 4, 'Z')
                elif abs(rz) > 0.01:
                    rot_inv_dbg = mathutils.Matrix.Rotation(-rz, 4, 'Z')
                else:
                    rot_inv_dbg = mathutils.Matrix.Identity(4)
                mn = [ float('inf')] * 3
                mx = [-float('inf')] * 3
                for v in me.vertices:
                    rc = rot_inv_dbg @ mathutils.Vector((v.co.x, v.co.y, v.co.z, 1.0))
                    if rc.x < mn[0]: mn[0] = rc.x
                    if rc.y < mn[1]: mn[1] = rc.y
                    if rc.z < mn[2]: mn[2] = rc.z
                    if rc.x > mx[0]: mx[0] = rc.x
                    if rc.y > mx[1]: mx[1] = rc.y
                    if rc.z > mx[2]: mx[2] = rc.z
                if mn[0] == float('inf'):
                    mn = [0.0, 0.0, 0.0]; mx = [0.0, 0.0, 0.0]
                size = (mx[0] - mn[0], mx[1] - mn[1], mx[2] - mn[2])
                TraceLogger.kvblock(
                    f"object '{o.name}' (pre-encode)",
                    [
                        ("type",            o.type),
                        ("vertex_count",    len(me.vertices)),
                        ("face_count",      len(me.polygons)),
                        ("loop_count",      len(me.loops)),
                        ("uv_layers",       [u.name for u in me.uv_layers]),
                        ("color_layers",    [c.name for c in (me.color_attributes
                                              if hasattr(me, 'color_attributes')
                                              else (me.vertex_colors or []))]),
                        ("material_slots",  slot_records),
                        ("vertex_groups",   len(o.vertex_groups)),
                        ("armature",        arm_info),
                        ("modifiers",       mod_stack),
                        ("rotation_euler",  tuple(round(a, 6) for a in o.rotation_euler)),
                        ("xbg_data.slot",   (o.get('xbg_data') or {}).get('sdol_submesh_slot')),
                        ("xbg_bone_palette_present",
                                            bool(o.get('xbg_bone_palette'))),
                        ("aabb_rotinv_min", tuple(round(v, 6) for v in mn)),
                        ("aabb_rotinv_max", tuple(round(v, 6) for v in mx)),
                        ("aabb_size",       tuple(round(v, 6) for v in size)),
                    ],
                    tier="DEBUG",
                    event="object_inspect")
            except Exception as _e:
                TraceLogger.debug(
                    f"  [trace] object inspect failed for '{o.name}': {_e}",
                    event="object_inspect_failed",
                    data={"object": o.name, "error": str(_e)[:256]})

        # -- Resolve XBG metadata ------------------------------------------
        if meta is None:
            for o in mesh_objects:
                if "xbg_data" in o:
                    meta = o["xbg_data"].to_dict()
                    break
        if meta is None:
            return {'CANCELLED'}, (
                "No XBG metadata found.  Import an XBG first, or include the "
                "original imported object in your selection.")

        original_path = meta.get("filepath", "")
        if not os.path.exists(original_path):
            return {'CANCELLED'}, f"Original XBG not found at: {original_path}"

        # -- Gather format parameters --------------------------------------
        pos_scale        = meta.get('pos_scale', 1.0)
        import_mesh_only = meta.get('import_mesh_only', False)
        uv_trans         = meta.get('uv_trans', 0.0)
        uv_scale_raw     = meta.get('uv_scale', 1.0)
        # uv_scale is stored as a single float by the importer (see
        # XBGData.uv_scale), but a list/tuple sometimes surfaces if the
        # IDProperty was edited externally — fall back to the first element.
        # (The previous code preferred index 1 which made no sense for a
        # scalar, since uv_trans is stored under its own key already.)
        if isinstance(uv_scale_raw, (list, tuple)) and uv_scale_raw:
            uv_scale = float(uv_scale_raw[0])
        else:
            uv_scale = float(uv_scale_raw)
        if abs(uv_scale) < 1e-9:
            uv_scale = 1.0

        effective_pos_scale = target_game_scale if override_game_scale else pos_scale

        VerboseLogger.log(f"\n{'='*60}")
        VerboseLogger.log(f"XBG MESH INJECTION  v2.1")
        VerboseLogger.log(f"File      : {os.path.basename(original_path)}")
        VerboseLogger.log(f"LOD       : {target_lod}")
        VerboseLogger.log(f"Objects   : {len(mesh_objects)}")
        VerboseLogger.log(f"Vertex colors   : {inject_vertex_colors}")
        VerboseLogger.log(f"Bone weights    : {inject_bone_weights}")
        VerboseLogger.log(f"Split materials : {inject_materials}")
        if override_game_scale:
            VerboseLogger.log(f"PMCP override   : {target_game_scale:.6f}")
        VerboseLogger.log(f"{'='*60}")

        with open(original_path, 'rb') as f:
            file_data = bytearray(f.read())

        # -- Import-side handedness flip -----------------------------------
        # The importer flips face winding + negates normals (default ON) so
        # the model shades correctly in the Blender viewport — XBG uses the
        # opposite handedness convention from Blender.  We must reverse both
        # on export or the round-trip produces a file with reversed winding
        # (game backface-culls every triangle that was originally visible,
        # and inward-facing mouth/eye interiors poke out as visible "stretched
        # teeth" artifacts) and negated normals (per-pixel lighting inverted).
        #
        # Default True because the import default is True.  Falls back safely
        # for older imports that didn't store this flag.
        flipped_on_import = bool(meta.get('flipped_on_import', True))

        endian = meta.get('endian')
        if endian not in (LE, BE):
            endian = detect_endian_from_bytes(file_data[:32])
        endian_label = "Big-endian (PS3)" if endian == BE else "Little-endian (PC)"
        VerboseLogger.log(f"  Byte order   : {endian_label}")

        # -- PMCP scale override ------------------------------------------
        if override_game_scale:
            patch_pmcp(file_data, target_game_scale, endian)

        # -- Locate SDOL --------------------------------------------------
        sdol_info = find_chunk(file_data, 'SDOL', endian)
        if not sdol_info:
            return {'CANCELLED'}, "No SDOL chunk found in XBG file."
        sdol_start, sdol_data_start, sdol_old_size = sdol_info

        sdol = parse_sdol(file_data, sdol_data_start, endian)
        if target_lod >= sdol.lod_count:
            return {'CANCELLED'}, (
                f"LOD {target_lod} not present "
                f"(file has {sdol.lod_count} LOD(s): 0-{sdol.lod_count - 1}).")

        lod = sdol.lods[target_lod]
        if not lod.vb_info:
            return {'CANCELLED'}, "Target LOD has no vertex buffers."

        n_orig_submeshes = len(lod.submeshes)
        n_orig_vbs       = len(lod.vb_info)
        VerboseLogger.log(f"  Original LOD {target_lod}: {n_orig_submeshes} submesh(es), "
                          f"{n_orig_vbs} vertex buffer(s)")

        # -- Read reference vertex format ---------------------------------
        vb0         = lod.vb_info[0]
        vert_fmt    = vb0['flags']
        vert_stride = vb0['stride']

        VerboseLogger.log(f"  Vertex format : 0x{vert_fmt:04X}  stride: {vert_stride}")
        VerboseLogger.log(f"  Pos scale     : {effective_pos_scale}  "
                          f"UV trans: {uv_trans}  UV scale: {uv_scale}")

        # -- Build injection slices ----------------------------------------
        slices     = []
        tmp_meshes = []

        try:
            for obj in mesh_objects:
                parts = _triangulate_and_split_by_material(
                    obj, split_by_material=inject_materials)
                for tri_mesh, sub_idx, mat_name in parts:
                    # Enforce the engine's uint16-per-submesh limit by
                    # splitting oversized slices into multiple submeshes.
                    budget = _split_mesh_by_face_budget(
                        tri_mesh, tri_mesh.name)
                    if len(budget) > 1:
                        VerboseLogger.log(f"  [inject] Slice '{obj.name}' mat="
                                         f"'{mat_name}' "
                                         f"({len(tri_mesh.polygons)}t / "
                                         f"{len(tri_mesh.vertices)}v) exceeds the "
                                         f"uint16 submesh limit -> split into "
                                         f"{len(budget)} submeshes")
                        # original tri_mesh now superseded; free it.
                        try:
                            bpy.data.meshes.remove(tri_mesh)
                        except Exception:
                            pass
                    for piece in budget:
                        slices.append((obj, piece, sub_idx, mat_name))
                        tmp_meshes.append(piece)

            if not slices:
                return {'CANCELLED'}, "No geometry could be extracted."

            # -- Bounds check warnings -------------------------------------
            for obj, _, _, _ in slices:
                ns, _, si = calculate_required_scale(
                    obj, effective_pos_scale, import_mesh_only)
                if ns:
                    VerboseLogger.log(f"  WARNING '{obj.name}': {si} -- verts may be clamped!")

            # -- Vertex count guard ----------------------------------------
            # Slices are auto-split to the uint16 budget above; this is a
            # belt-and-braces check in case a single tri references too
            # many unique verts to ever fit (degenerate input).
            for obj, tri_mesh, sub_idx, _ in slices:
                n = len(tri_mesh.vertices)
                if n > 65534:
                    return {'CANCELLED'}, (
                        f"Object '{obj.name}' (material {sub_idx}) still has "
                        f"{n} vertices after auto-split. Reduce the mesh "
                        "density and try again.")

            # -- Encode each slice -----------------------------------------
            # KEY FIX: each object now uses its OWN bone palette (stored at
            # import time as xbg_bone_palette), not always submesh-0's palette.
            all_vert_bufs = []
            all_idx_bufs  = []
            all_vert_cnts = []
            all_face_cnts = []
            all_palettes  = []        # per-slice 48-bone palette (for DNKS)
            all_bboxes    = []        # per-slice ((min_x, min_y, min_z), (max_x, max_y, max_z))
            all_obj_names = []        # per-slice source object name (for DNKS trailing names)
            total_clamped = 0

            for _slice_iter_idx, (obj, tri_mesh, sub_idx, mat_name) in enumerate(slices):
                # Per-slice checkpoint — the LAST log line on a crash tells us
                # exactly which slice was being processed.
                TraceLogger.info(
                    f"  [slice {_slice_iter_idx}/{len(slices)-1}] obj='{obj.name}' "
                    f"mat='{mat_name}' verts={len(tri_mesh.vertices)} "
                    f"faces={len(tri_mesh.polygons)}",
                    event="slice_begin",
                    data={
                        "slice_idx": _slice_iter_idx,
                        "object":    obj.name,
                        "material":  mat_name,
                        "verts":     len(tri_mesh.vertices),
                        "faces":     len(tri_mesh.polygons),
                        "tri_mesh":  getattr(tri_mesh, "name", "?"),
                    })
                apply_scale = 1.0

                # Vertex color is the aaa.fx mask (spec/normal/AO), not optional
                # cosmetic data. With "Include Vertex Colors" ON (default) we read
                # the mesh's 'Col' layer: stock verts preserve their authored mask
                # byte-exact through split-by-material, and anything the user
                # painted is injected as-is. OFF -> color_map None -> the encoder
                # writes a neutral mask for EVERY vert (deliberately drop the
                # game's shading data, e.g. a fully custom model).
                color_map = (build_vertex_color_map(tri_mesh)
                             if inject_vertex_colors else None)

                # Palette selection strategy:
                #
                # EXISTING object (has xbg_bone_palette from import):
                #   Keep the original imported palette — MB2O inverse-bind
                #   matrices are indexed by DNKS palette SLOT ORDER and are
                #   never rebuilt.  Changing the order makes the engine apply
                #   the wrong bind-pose inverse → mesh skins to wrong position
                #   (invisible / exploded in-game while looking fine in Blender).
                #
                # NEW object (no stored palette, e.g. a visor added by the user):
                #   Build a fresh palette from the bones this mesh actually uses.
                #   Without this, the fallback is submesh-0's palette (Pelvis/Spine),
                #   which does NOT contain the Head bone.  Any vertex group whose
                #   global bone ID is absent in that palette gets mapped to slot 0
                #   (= Pelvis) — the visor then follows the pelvis instead of the
                #   head even with 100% head-bone weight painting.  A new submesh
                #   has no existing MB2O row so the fresh-palette path is safe.

                is_new_object = not obj.get("xbg_bone_palette")
                bone_palette = _get_object_bone_palette(
                    obj, file_data, target_lod, endian)
                if inject_bone_weights:
                    if is_new_object:
                        arm = _get_armature(obj)
                        if arm and obj.vertex_groups:
                            bone_palette, weight_map = \
                                _build_slice_palette_and_weights(
                                    obj, tri_mesh, arm)
                            VerboseLogger.log(
                                f"  [inject] '{obj.name}' is a new object — "
                                f"built fresh palette from actual bone weights")
                        else:
                            # No armature or no vertex groups → fall back
                            weight_map = _remap_weights_by_position(
                                obj, tri_mesh,
                                _build_weight_map(obj, bone_palette))
                            weight_map = _rigid_bind_foreign_into_palette(
                                obj, tri_mesh, weight_map, bone_palette,
                                _get_armature(obj))
                    else:
                        weight_map = _remap_weights_by_position(
                            obj, tri_mesh,
                            _build_weight_map(obj, bone_palette))
                        weight_map = _rigid_bind_foreign_into_palette(
                            obj, tri_mesh, weight_map, bone_palette,
                            _get_armature(obj))
                else:
                    weight_map = None
                all_palettes.append(bone_palette)

                vert_buf, clamped = _encode_vertices(
                    tri_mesh, vert_fmt, vert_stride,
                    effective_pos_scale, uv_trans, uv_scale, obj,
                    apply_scale=apply_scale,
                    ignore_limits=ignore_limits,
                    color_map=color_map,
                    neutral_empty_colors=generate_neutral_vertex_colors,
                    weight_map=weight_map,
                    endian=endian)

                idx_buf, idx_count = _build_index_buffer(
                    tri_mesh, endian, reverse_winding=flipped_on_import)

                n_verts = len(tri_mesh.vertices)
                n_faces = len(tri_mesh.polygons)
                all_vert_bufs.append(vert_buf)
                all_idx_bufs.append(idx_buf)
                all_vert_cnts.append(n_verts)
                all_face_cnts.append(n_faces)
                total_clamped += clamped

                # Per-submesh AABB (in the same coordinate frame as XOBB —
                # apply the same Z-rotation inversion as patch_bounds so
                # the bbox we store in DNKS lines up with the engine's
                # culling).  Needed for the DNKS trailing rebuild below.
                import math as _math
                import mathutils as _mu
                _rz = obj.rotation_euler.z
                if import_mesh_only and abs(_rz - _math.radians(180)) < 0.01:
                    _rot_inv = _mu.Matrix.Rotation(-_math.radians(180), 4, 'Z')
                elif abs(_rz) > 0.01:
                    _rot_inv = _mu.Matrix.Rotation(-_rz, 4, 'Z')
                else:
                    _rot_inv = _mu.Matrix.Identity(4)
                _mnx = _mny = _mnz =  1e30
                _mxx = _mxy = _mxz = -1e30
                for _v in tri_mesh.vertices:
                    _rc = _rot_inv @ _mu.Vector((_v.co.x, _v.co.y, _v.co.z, 1.0))
                    if _rc.x < _mnx: _mnx = _rc.x
                    if _rc.y < _mny: _mny = _rc.y
                    if _rc.z < _mnz: _mnz = _rc.z
                    if _rc.x > _mxx: _mxx = _rc.x
                    if _rc.y > _mxy: _mxy = _rc.y
                    if _rc.z > _mxz: _mxz = _rc.z
                if _mnx > 1e29:
                    _mnx = _mny = _mnz = _mxx = _mxy = _mxz = 0.0
                all_bboxes.append(((_mnx, _mny, _mnz), (_mxx, _mxy, _mxz)))
                all_obj_names.append(obj.name)

                wt_info = (f"  {len(weight_map)}/{n_verts} verts weighted"
                           if weight_map else "")
                VerboseLogger.log(f"  Slice '{obj.name}' mat='{mat_name}': "
                                  f"{n_verts}v / {n_faces}t{wt_info}")

        except Exception as exc:
            # Emit a full crash-context dump BEFORE returning so the user's
            # log has all the bread-crumbs leading up to the failure.
            import traceback as _tb
            tb_text = _tb.format_exc()
            TraceLogger.info(
                f"  [trace] !!! Encoding failed: {exc.__class__.__name__}: {exc}",
                event="encoding_failed",
                data={
                    "exception_type": exc.__class__.__name__,
                    "exception_msg":  str(exc)[:1024],
                    "traceback":      tb_text,
                    "slices_done":    len(all_vert_bufs),
                    "slices_total":   len(slices) if 'slices' in dir() else None,
                    "last_obj":       getattr(obj, "name", "?") if 'obj' in dir() else None,
                    "last_mat":       mat_name if 'mat_name' in dir() else None,
                })
            # Also write the traceback as plain text lines so they appear
            # in the saved txt log (the structured record has it too).
            for tbline in tb_text.splitlines():
                VerboseLogger.log(f"    {tbline}")
            return {'CANCELLED'}, f"Encoding failed: {exc}"
        finally:
            for tm in tmp_meshes:
                try:
                    bpy.data.meshes.remove(tm)
                except Exception:
                    pass

        if total_clamped > 0:
            VerboseLogger.log(f"  WARNING: {total_clamped} vertices were clamped to int16 range")

        # ── Encode-level sanity pass ──────────────────────────────────────
        # Aggregate per-slice diagnostics into one filterable event so a
        # reader doesn't have to scroll through N encode_stats events to
        # spot a problem.  Categories:
        #   encode/clamping           clamped verts (already warned above)
        #   encode/low_headroom       any axis within 5% of int16 ceiling
        #                             (geometry one nudge away from clamping)
        #   skin/palette_unused       palette has very few valid bone slots
        #   skin/rigid_bound_slice    every vert pinned to one bone (foreign
        #                             geometry that didn't carry weights)
        #   geometry/large_submesh    a slice >55000 verts — auto-split
        #                             worked but this is fragile, advise
        #                             smaller meshes
        # Each row of the warning table corresponds to ONE slice; aggregate
        # counts also go to a structured `sanity_summary` event.
        try:
            _w_clamp = total_clamped
            _w_low_headroom = []
            _w_palette_thin = []
            _w_rigid_slice = []
            _w_large_slice = []
            _inv_pos = (1.0 / effective_pos_scale
                        if effective_pos_scale else 0.0)
            _AXIS_MAX = 32767
            _HEAD_FRAC = 0.05  # 5% — anything tighter is a near-miss
            _PAL_THIN = 6      # < 6 valid bones in a 48-slot palette
            _LARGE_SLICE = 55000  # 84% of uint16 cap
            for _si, (obj, _tm, _sub, _mn) in enumerate(slices):
                if _si >= len(all_vert_cnts):
                    break
                _n_v = all_vert_cnts[_si]
                _bb = all_bboxes[_si] if _si < len(all_bboxes) else None
                _pal = all_palettes[_si] if _si < len(all_palettes) else []
                # Headroom: largest |position| in int16 quanta.
                if _bb is not None and _inv_pos:
                    _max_axis = [max(abs(_bb[0][a]), abs(_bb[1][a])) * _inv_pos
                                 for a in range(3)]
                    _headroom = [_AXIS_MAX - mx for mx in _max_axis]
                    if min(_headroom) < _AXIS_MAX * _HEAD_FRAC:
                        _w_low_headroom.append({
                            "slice": _si, "object": obj.name,
                            "material": _mn,
                            "max_abs_int16_xyz": [round(v, 1) for v in _max_axis],
                            "headroom_xyz":      [round(v, 1) for v in _headroom],
                            "world_half_at_current_scale": round(_AXIS_MAX * effective_pos_scale, 4),
                        })
                # Palette validity.
                _valid_palette = sum(1 for p in _pal if p is not None and p >= 0)
                if _valid_palette < _PAL_THIN:
                    _w_palette_thin.append({
                        "slice": _si, "object": obj.name,
                        "valid_slots": _valid_palette,
                        "palette_size": len(_pal),
                    })
                # Large-slice flag (auto-split succeeded but it's tight).
                if _n_v > _LARGE_SLICE:
                    _w_large_slice.append({
                        "slice": _si, "object": obj.name,
                        "material": _mn,
                        "verts": _n_v,
                        "uint16_headroom": 65535 - _n_v,
                    })
            # Rigid-bind detection: cross-reference TraceLogger's previously-
            # emitted rigid_bind_done events recorded earlier in the same
            # inject call.  Cheap because we only filter records, not
            # re-walk geometry.
            for _r in VerboseLogger.get_records():
                if _r.get("event") == "rigid_bind_done":
                    d = _r.get("data") or {}
                    _w_rigid_slice.append({
                        "object":          d.get("title", "?"),
                        "verts_pinned":    d.get("verts_rigid_bound"),
                        "slot":            d.get("chosen_slot"),
                        "bone":            d.get("chosen_bone"),
                    })

            n_total = (int(bool(_w_clamp))
                       + len(_w_low_headroom)
                       + len(_w_palette_thin)
                       + len(_w_rigid_slice)
                       + len(_w_large_slice))
            VerboseLogger.log(
                f"[sanity] encode pass: {n_total} finding(s)  "
                f"clamped_verts={_w_clamp}  "
                f"low_headroom_slices={len(_w_low_headroom)}  "
                f"thin_palette_slices={len(_w_palette_thin)}  "
                f"rigid_bound_slices={len(_w_rigid_slice)}  "
                f"large_slices={len(_w_large_slice)}")
            if _w_clamp > 0:
                TraceLogger.struct("sanity_warning",
                    {"category": "encode/clamping",
                     "message":  f"{_w_clamp} vert(s) clamped to int16 "
                                 f"limit; geometry visibly truncated. "
                                 f"Reduce mesh scale or use 'Auto Scale "
                                 f"Bounding Box' to widen the encoding range.",
                     "clamped_verts": _w_clamp})
            for w in _w_low_headroom:
                TraceLogger.struct("sanity_warning",
                    {"category": "encode/low_headroom",
                     "message":  (f"slice {w['slice']} '{w['object']}' has "
                                  f"<5% int16 headroom on at least one axis. "
                                  f"Any further extrusion or scale change "
                                  f"will start clamping."),
                     **w})
            for w in _w_palette_thin:
                TraceLogger.struct("sanity_warning",
                    {"category": "skin/palette_unused",
                     "message":  (f"slice {w['slice']} '{w['object']}' has "
                                  f"only {w['valid_slots']}/48 valid palette "
                                  f"slots — most of the bone palette is "
                                  f"wasted on -1 entries."),
                     **w})
            for w in _w_rigid_slice:
                TraceLogger.struct("sanity_warning",
                    {"category": "skin/rigid_bound_slice",
                     "message":  (f"{w['object']}: {w['verts_pinned']} "
                                  f"vert(s) pinned 100% to bone "
                                  f"'{w['bone']}' (palette slot "
                                  f"{w['slot']}) — geometry was fully "
                                  f"foreign and got rigid-attached. This "
                                  f"is intentional for props/joined geometry "
                                  f"that doesn't carry skinning weights, "
                                  f"but if you EXPECTED skinning, your "
                                  f"vertex groups didn't make it through."),
                     **w})
            for w in _w_large_slice:
                TraceLogger.struct("sanity_warning",
                    {"category": "geometry/large_submesh",
                     "message":  (f"slice {w['slice']} '{w['object']}' has "
                                  f"{w['verts']} verts ({w['uint16_headroom']} "
                                  f"below uint16 cap). Future edits adding "
                                  f"verts to this slice may overflow."),
                     **w})
            TraceLogger.struct("sanity_summary", {
                "phase": "post_encode",
                "clamped_verts":         _w_clamp,
                "low_headroom_slices":   len(_w_low_headroom),
                "thin_palette_slices":   len(_w_palette_thin),
                "rigid_bound_slices":    len(_w_rigid_slice),
                "large_slices":          len(_w_large_slice),
                "total_findings":        n_total,
            })
        except Exception as _exc:
            TraceLogger.debug(
                f"  [sanity] encode-pass sanity guard raised: {_exc}",
                event="sanity_guard_failed",
                data={"err": str(_exc)[:256]})

        # -- Merge duplicate-material slices ──────────────────────────────
        # When two or more slices share the same material name (e.g. a
        # character model where "MI_Prolemuris" is assigned to both the
        # arms and the body as separate Blender material slots), we combine
        # their geometry into one submesh before building the SDOL. This
        # avoids two LTMR entries and two DNKS submeshes for the same
        # material, which can cause material binding conflicts in the engine.
        #
        # Merge rule: the first occurrence becomes the "open accumulator" for
        # its material; subsequent occurrences with the same mat_name AND the
        # same source object are appended (vert buffer concatenated, index
        # buffer offset-rebased by the accumulator's running vert count).
        #
        # TWO GUARDS keep this safe:
        #   1. SIZE  — only merge if the combined submesh stays under the
        #      uint16 limit (_MAX_SM_FACES faces / _HARD_MAX_VERT verts).
        #      This is critical: the auto-splitter intentionally divides an
        #      oversized single-material mesh into multiple submeshes, and
        #      those halves share a material name. Without this guard the
        #      merge would recombine them past 65535 indices and crash the
        #      GPU (faces*3 > 65535). Auto-split halves never fit back
        #      together (the original exceeded the limit), so they stay split.
        #   2. SAME OBJECT — only merge slices from the same source object, so
        #      slot-aware SDOL mapping (object → slot) is never broken by
        #      collapsing two different objects' geometry.
        _acc_for_mat = {}   # mat_name → open accumulator slice index
        _merged_vb   = list(all_vert_bufs)
        _merged_ib   = list(all_idx_bufs)
        _merged_vc   = list(all_vert_cnts)
        _merged_fc   = list(all_face_cnts)
        _merged_slices = list(slices)
        _merged_pal  = list(all_palettes)
        _merged_bbox = list(all_bboxes)
        _merged_onames = list(all_obj_names)
        _drop = set()
        for _mi, (_obj, _tm, _sub, _mn) in enumerate(slices):
            _key = _mn
            _acc = _acc_for_mat.get(_key)
            # No open accumulator, or it's a different source object → this
            # slice starts/replaces the accumulator for this material.
            if _acc is None or _merged_slices[_acc][0] is not _obj:
                _acc_for_mat[_key] = _mi
                continue
            _comb_fc = _merged_fc[_acc] + _merged_fc[_mi]
            _comb_vc = _merged_vc[_acc] + _merged_vc[_mi]
            if _comb_vc > _HARD_MAX_VERT:
                # The only hard constraint: vertex INDEX values must fit in
                # uint16, so vert count must be ≤ 65535. Face count alone is
                # not a blocker — the index buffer entry count does not need
                # to fit in uint16, only the individual index values do.
                _acc_for_mat[_key] = _mi
                VerboseLogger.log(
                    f"  [inject] NOT merging duplicate material '{_mn}' "
                    f"slice {_mi} into {_acc}: combined {_comb_vc} verts "
                    f"exceeds uint16 vert limit ({_HARD_MAX_VERT}) "
                    f"— kept as a separate submesh")
                continue
            # Safe + same object → merge into the accumulator.
            _base_vc = _merged_vc[_acc]
            _n_idx = len(_merged_ib[_mi]) // 2
            _local = struct.unpack(f'{endian}{_n_idx}H', _merged_ib[_mi])
            _glob  = struct.pack(f'{endian}{_n_idx}H',
                                 *(_v + _base_vc for _v in _local))
            _merged_vb[_acc]  = _merged_vb[_acc] + _merged_vb[_mi]
            _merged_ib[_acc]  = _merged_ib[_acc] + _glob
            _merged_vc[_acc] += _merged_vc[_mi]
            _merged_fc[_acc] += _merged_fc[_mi]
            _drop.add(_mi)
            VerboseLogger.log(
                f"  [inject] merged duplicate material '{_mn}' "
                f"(slice {_mi} → slice {_acc}): "
                f"combined {_merged_vc[_acc]} verts, "
                f"{_merged_fc[_acc]} faces")
        if _drop:
            _keep = [i for i in range(len(_merged_slices)) if i not in _drop]
            slices       = [_merged_slices[i] for i in _keep]
            all_vert_bufs = [_merged_vb[i]   for i in _keep]
            all_idx_bufs  = [_merged_ib[i]   for i in _keep]
            all_vert_cnts = [_merged_vc[i]   for i in _keep]
            all_face_cnts = [_merged_fc[i]   for i in _keep]
            all_palettes  = [_merged_pal[i]  for i in _keep]
            all_bboxes    = [_merged_bbox[i] for i in _keep]
            all_obj_names = [_merged_onames[i] for i in _keep]

        # -- Build combined vertex + index data ---------------------------
        #
        # SLOT-AWARE MODE: activated when every injected object carries an
        # 'sdol_submesh_slot' tag (written at import time when Separate
        # Primitives is ON).  In that mode each selected object lands at
        # its original SDOL flat index.  Slots with NO selected object are
        # written as null (invisible) geometry — 3 zeroed vertices and one
        # degenerate face — with DNKS face_count=0 / vert_count=0 so the
        # game renders nothing for them.  This lets users delete the
        # primitives they don't want in Blender and inject only the ones
        # they do want; the result is a character with those parts removed.
        #
        # SEQUENTIAL MODE (legacy): objects fill slots 0, 1, 2 … in order.
        # Used when slot tags are absent (joined import) or when all
        # original submeshes are being replaced at once.

        # Collect sdol_submesh_slot from each slice's object (if present).
        slice_slots = []
        for obj, _tm, _si, _mn in slices:
            xd = obj.get("xbg_data")
            slot = xd.get("sdol_submesh_slot") if xd else None
            # Blender IDProperties store ints as floats sometimes; normalise.
            slice_slots.append(int(slot) if slot is not None else None)

        # Every injected slice must become its OWN submesh.  Slot-aware mode
        # keys submeshes by the source object's SDOL slot, so when there are
        # MORE slices than distinct slots they collapse into one slot (slot_map
        # keeps only the first per slot — the rest are DROPPED).  This happens
        # two ways:
        #   • split-by-material: one object expands into several material slices.
        #   • a SEPARATED object inheriting the body's slot tag: a mesh split off
        #     the body keeps `xbg_data.slot=0`, colliding with the body —
        #     slot-aware would then drop the separated piece entirely.
        # In BOTH cases force sequential mode so each slice gets a distinct
        # submesh; DNKS is rebuilt below to match the new count.  (No-op for the
        # normal flow where slices == distinct slots.)
        _slots_present = [s for s in slice_slots if s is not None]
        _distinct_slots = len(set(_slots_present))
        _split_expands = (len(slices) > max(1, _distinct_slots))
        use_slot_aware = (
            all(s is not None for s in slice_slots)
            and n_orig_submeshes > 1
            and not _split_expands
        )
        if _split_expands:
            VerboseLogger.log(f"  [inject] {len(slices)} slices > {_distinct_slots} "
                              f"distinct slot(s) (split/collision) -> sequential mode, "
                              f"each slice gets its own submesh (DNKS rebuilt)")

        if use_slot_aware:
            VerboseLogger.log(f"  Slot-aware injection: {len(slices)} object(s) → "
                              f"{n_orig_submeshes} SDOL slot(s)")

            # Map slot → (vert_buf, idx_buf, n_verts, n_faces)
            slot_map = {}
            # Map slot → (bbox, name) so the DNKS trailing rebuild can
            # use the correct bbox/name for each replaced slot.
            slot_bbox_map = {}
            slot_name_map = {}
            for k, slot in enumerate(slice_slots):
                if slot not in slot_map:
                    slot_map[slot] = (
                        all_vert_bufs[k], all_idx_bufs[k],
                        all_vert_cnts[k], all_face_cnts[k])
                if slot not in slot_bbox_map:
                    slot_bbox_map[slot] = all_bboxes[k]
                if slot not in slot_name_map:
                    slot_name_map[slot] = all_obj_names[k]

            combined_vert  = bytearray()
            vb_info_new    = []
            combined_idx   = bytearray()
            submeshes_new  = []
            dnks_updates   = []
            byte_offset    = 0
            idx_u16_offset = 0

            # Same VB-layout decision as the sequential branch — the engine
            # crashes on character meshes when shared-VB LODs are silently
            # converted into per-submesh-VB.  See comment in the else branch.
            orig_vb_count_sa = len(lod.vb_info)
            use_shared_vb_sa = (orig_vb_count_sa == 1) and not force_per_submesh_vb
            if force_per_submesh_vb and orig_vb_count_sa == 1:
                VerboseLogger.log(
                    "  [inject] VB layout (slot-aware): FORCED per-slot-VB by "
                    "user toggle (source was shared-VB; untested for characters)")
            VerboseLogger.log(
                f"  [inject] VB layout (slot-aware): source LOD has "
                f"{orig_vb_count_sa} VB(s) -> "
                f"{'shared-VB (1 VB, global indices)' if use_shared_vb_sa else 'per-slot-VB (N VBs, local indices)'}")

            # In shared-VB mode we need to pre-compute every slot's vert/idx
            # buffers, then concatenate vertices into one VB and rewrite each
            # slot's indices with the global cumulative offset baked in.
            slot_buffers = []   # list of (vert_buf, idx_buf, n_verts, n_faces, is_null)
            for slot in range(n_orig_submeshes):
                if slot in slot_map:
                    vert_buf, idx_buf, n_verts, n_faces = slot_map[slot]
                    is_null = False
                else:
                    null_v   = _null_vertex(vert_fmt, vert_stride, endian)
                    vert_buf = null_v * 3
                    idx_buf  = struct.pack(f'{endian}3H', 0, 0, 0)
                    n_verts  = 3
                    n_faces  = 0
                    is_null  = True
                slot_buffers.append((vert_buf, idx_buf, n_verts, n_faces, is_null))

            if use_shared_vb_sa:
                # All slots' verts go into a single shared VB at offset 0.
                # Indices are rewritten to be GLOBAL into that shared VB.
                total_verts = sum(n_verts for _, _, n_verts, _, _ in slot_buffers)
                if total_verts > 65535:
                    return ({'CANCELLED'},
                            f"Shared-VB LOD would need {total_verts} vertices "
                            "(uint16 cap 65535). Reduce mesh density or use "
                            "per-submesh-VB by changing the source file's layout.")

                # Build the single VB info entry (preserve original flags/stride).
                orig_vb = lod.vb_info[0]
                vb_info_new.append({
                    'flags':  orig_vb.get('flags',  vert_fmt),
                    'stride': orig_vb.get('stride', vert_stride),
                    'unk':    total_verts,
                    'offset': 0,
                })

                cum_verts_count = 0
                cum_verts_bytes = 0
                for slot in range(n_orig_submeshes):
                    orig_sm = lod.submeshes[slot]
                    vert_buf, idx_buf, n_verts, n_faces, is_null = slot_buffers[slot]

                    if is_null:
                        dnks_updates.append((0, 0))
                        VerboseLogger.log(f"  Slot {slot}: compacted  (3 null verts, DNKS=0)")
                    else:
                        dnks_updates.append((n_faces, n_verts))
                        VerboseLogger.log(f"  Slot {slot}: replaced  ({n_verts}v / {n_faces}t)")

                    # Offset this slot's indices by the cumulative vert count so
                    # they become global into the shared VB.
                    n_idx = len(idx_buf) // 2
                    local = struct.unpack(f'{endian}{n_idx}H', idx_buf)
                    glob  = tuple(v + cum_verts_count for v in local)
                    glob_bytes = struct.pack(f'{endian}{n_idx}H', *glob)

                    combined_vert += vert_buf
                    combined_idx  += glob_bytes

                    submeshes_new.append({
                        'vb_idx':      0,
                        'lod_grp':     orig_sm['lod_grp'],
                        'sub_idx':     orig_sm['sub_idx'],
                        'idx_offset':  idx_u16_offset,
                        'vert_marker': cum_verts_count + n_verts - 1,
                        'unk1':        cum_verts_bytes,
                        'unk2':        orig_sm['unk2'],
                    })
                    idx_u16_offset  += n_idx
                    cum_verts_count += n_verts
                    cum_verts_bytes += n_verts * vert_stride
            else:
                # Per-slot VB layout (original behaviour).
                for slot in range(n_orig_submeshes):
                    orig_sm = lod.submeshes[slot]
                    vert_buf, idx_buf, n_verts, n_faces, is_null = slot_buffers[slot]

                    if is_null:
                        dnks_updates.append((0, 0))
                        VerboseLogger.log(f"  Slot {slot}: compacted  (3 null verts, DNKS=0)")
                    else:
                        dnks_updates.append((n_faces, n_verts))
                        VerboseLogger.log(f"  Slot {slot}: replaced  ({n_verts}v / {n_faces}t)")

                    # Capture the VB offset BEFORE incrementing — both vb_info.offset
                    # and submesh.unk1 must point here (they are the same field, stored
                    # in two places: vb_info[k]['offset'] and the submesh that USES that
                    # VB).  Preserving orig_sm['unk1'] is wrong as soon as ANY earlier
                    # slot changes size — the shrunk vertex section makes the original
                    # offset point past EOF and the GPU reads garbage, killing the
                    # whole model's render (not just the affected slot).
                    slot_vert_offset = byte_offset

                    vb_info_new.append({
                        'flags':  vert_fmt,
                        'stride': vert_stride,
                        'unk':    n_verts,
                        'offset': slot_vert_offset,
                    })
                    combined_vert  += vert_buf
                    byte_offset    += len(vert_buf)

                    submeshes_new.append({
                        'vb_idx':      slot,
                        'lod_grp':     orig_sm['lod_grp'],
                        'sub_idx':     orig_sm['sub_idx'],
                        'idx_offset':  idx_u16_offset,
                        'vert_marker': n_verts - 1,
                        'unk1':        slot_vert_offset,
                        'unk2':        orig_sm['unk2'],
                    })
                    combined_idx   += idx_buf
                    idx_u16_offset += len(idx_buf) // 2

        else:
            # Sequential (legacy) mode.
            #
            # VB-layout decision (CRITICAL — affects whether character meshes
            # render at all in the engine):
            #
            # The ORIGINAL LOD may use either layout:
            #   • shared-VB   — one VB holds every submesh's verts; submesh
            #                   indices are GLOBAL into the shared VB
            #                   (kendra LOD0 = 1 VB, 7 submeshes, all use
            #                   vb_idx=0).  This is the standard character
            #                   layout in Avatar / Dunia engine.
            #   • per-VB      — every submesh has its own VB; indices are
            #                   LOCAL to that VB (some props / FX use this).
            #
            # If we silently switch a shared-VB LOD into per-VB, the engine
            # binds the wrong vertex buffer for each submesh and the model
            # either renders as garbage or crashes on draw.  Match whatever
            # the source file used.
            orig_vb_count   = len(lod.vb_info)
            use_shared_vb   = (orig_vb_count == 1) and not force_per_submesh_vb
            # AUTO-FALLBACK on uint16 overflow: a shared VB addresses its verts
            # with uint16 indices (hard cap 65535). If the combined geometry of
            # everything injected into this LOD would exceed that, a single
            # shared VB is PHYSICALLY IMPOSSIBLE — fall back to per-submesh-VB
            # (each VB indexes <65535 LOCALLY). Only auto-switch for STATIC
            # meshes: per-VB is a real engine-supported layout for props/FX, but
            # skinned character meshes depend on the shared-VB + MB2O/palette
            # layout and can crash with per-VB, so those keep the hard error
            # (raised below) and the user must reduce the vertex count instead.
            _shared_total = sum(all_vert_cnts)
            _is_skinned   = bool(vert_fmt & 0x0010)   # BONE_WTS1 bit (0x0BDA vs 0x0BCA)
            if use_shared_vb and _shared_total > 65535 and not _is_skinned:
                use_shared_vb = False
                VerboseLogger.log(
                    f"  [inject] VB layout: shared VB would need {_shared_total} "
                    f"verts (>65535 uint16 cap) — AUTO-switched to per-submesh-VB "
                    f"(static mesh; valid props/FX layout, but verify in-game)")
            if force_per_submesh_vb and orig_vb_count == 1:
                VerboseLogger.log(
                    "  [inject] VB layout: FORCED per-submesh-VB by user toggle "
                    "(source was shared-VB; engine may not support this for "
                    "character meshes — test in-game)")
            VerboseLogger.log(
                f"  [inject] VB layout: source LOD has {orig_vb_count} VB(s) -> "
                f"{'shared-VB (1 VB, global indices)' if use_shared_vb else 'per-submesh-VB (N VBs, local indices)'}")

            combined_vert  = bytearray()
            vb_info_new    = []
            combined_idx   = bytearray()
            submeshes_new  = []
            idx_u16_offset = 0

            # -- Map each slice → which original submesh's metadata fields
            #    (lod_grp / sub_idx / unk2) it should inherit ----------------
            #
            # Split-by-material reorders slices: when an object with multiple
            # materials expands into several slices, all of them carry the
            # SAME source slot (the object's original SDOL slot), but only
            # ONE corresponds to the original submesh — the rest are new.
            #
            # The old code keyed `orig_sm` by slice INDEX (`lod.submeshes[i]`),
            # which mis-aligns lod_grp/sub_idx for every slice past an inserted
            # split point. The engine uses those fields to identify submeshes
            # for animation / cloth / ragdoll lookups, so a mismatch crashes.
            # We now key by SOURCE OBJECT SLOT and let only the first slice
            # per slot claim the original submesh's metadata.
            _orig_sm_for_slice = [None] * len(slices)
            _claimed_slots     = set()
            for _si in range(len(slices)):
                slot = slice_slots[_si] if _si < len(slice_slots) else None
                if (slot is not None
                        and slot not in _claimed_slots
                        and 0 <= slot < n_orig_submeshes):
                    _orig_sm_for_slice[_si] = lod.submeshes[slot]
                    _claimed_slots.add(slot)
            # Fallback (no slot mapping known): index-based, but ONLY when
            # slice count <= original count, so we don't mis-align splits.
            if not _claimed_slots and len(slices) <= n_orig_submeshes:
                for _si in range(len(slices)):
                    _orig_sm_for_slice[_si] = lod.submeshes[_si]

            # sub_idx assignment strategy: POSITIONAL.
            #
            # Stock XBG files have `sub_idx == SDOL position` (verified on
            # unmodified kendra: SM[0].sub_idx=0, SM[1].sub_idx=1, ... up to
            # SM[6].sub_idx=6, all matching their array index).  The engine
            # appears to rely on this invariant: when split-by-material adds
            # NEW submeshes to an existing LOD, synthesised sub_idx values
            # beyond the original max (e.g. 7,8,9 on a file with orig max 6)
            # crash the game on level-load AS SOON AS the new submesh's
            # material loads — case 2 in the user repro masks this by leaving
            # the new XBM files missing so the engine drops those submeshes
            # before reaching the buggy lookup.
            #
            # Fix: assign sub_idx = SDOL position for EVERY submesh (positional,
            # 0..N-1, no gaps, no duplicates).  Yields a submesh table that
            # is byte-identical in shape to a stock file's, so anything the
            # engine does keyed off sub_idx-as-position keeps working.
            #
            # Trade-off: original sub_idx values (3,4,5,6 for kendra) are
            # REASSIGNED when new submeshes are inserted before them.  Any
            # external file that addresses submeshes by their *original*
            # sub_idx (cloth .lks, ragdoll, runtime animation hooks) would
            # need updating in lock-step.  Plain NPC bodies without cloth
            # simulation are unaffected.
            #
            # Old strategy (kept here for archaeology — see git blame):
            # inherited orig sub_idx for first-slice-per-slot, synthesised
            # max+1, max+2, ... for the rest.  That produced the [0,1,2,7,8,
            # 9,3,4,5,6] table that crashed the game once the custom XBMs
            # actually loaded.
            _claim_lookup = {
                _si: _orig_sm_for_slice[_si]
                for _si in range(len(slices))
                if _orig_sm_for_slice[_si] is not None
            }

            def _sm_meta(i):
                """Return (lod_grp, sub_idx, unk2) for the i-th slice.

                sub_idx = i (SDOL position) for ALL slices.  lod_grp and
                unk2 still inherit from the claimed orig SM when one exists
                (those fields don't suffer the range/duplicate problem and
                preserving them keeps the per-LOD grouping intact).
                """
                src = _claim_lookup.get(i)
                if src is not None:
                    return src['lod_grp'], i, src['unk2']
                return target_lod, i, 0

            # Diagnostic dump: show every slice's (sdol_pos, old_sub_idx, new_sub_idx)
            # so any external sub_idx-keyed reference can be cross-checked.
            _subidx_rows = []
            for _si in range(len(slices)):
                src = _orig_sm_for_slice[_si]
                old = src['sub_idx'] if src is not None else None
                tag = (f"orig→{src['sub_idx']}" if src is not None
                       else "NEW (no orig SM)")
                _subidx_rows.append((_si, old, _si, tag))
            TraceLogger.table(
                "sub_idx positional reassignment",
                ("sdol_pos", "old_sub_idx", "new_sub_idx", "source"),
                _subidx_rows, tier="DEBUG", event="subidx_positional")
            VerboseLogger.log(
                "  [inject] sub_idx assignment = SDOL position "
                f"(0..{len(slices)-1}); "
                + ", ".join(f"pos{r[0]}={'orig' if r[1] is not None else 'NEW'}"
                            f"(was {r[1] if r[1] is not None else '-'})"
                            for r in _subidx_rows))

            _claim_log = []
            for _si in range(len(slices)):
                if _orig_sm_for_slice[_si] is not None:
                    _claim_log.append(
                        f"slice{_si}→origSM{slice_slots[_si]}")
                else:
                    _claim_log.append(f"slice{_si}→NEW")
            VerboseLogger.log(
                "  [inject] submesh-metadata claims: " + " ".join(_claim_log))

            if use_shared_vb:
                # All slices' vertices go into one big VB starting at offset 0.
                # Each submesh's index buffer gets offset by the cumulative
                # vert count so indices become GLOBAL into the shared VB.
                cum_verts_bytes = 0
                cum_verts_count = 0
                for vb_bytes in all_vert_bufs:
                    combined_vert += vb_bytes

                # Preserve the original VB's flags/stride; the 'unk' field of
                # the single VB carries the TOTAL vert count.
                orig_vb = lod.vb_info[0]
                vb_info_new.append({
                    'flags':  orig_vb.get('flags',  vert_fmt),
                    'stride': orig_vb.get('stride', vert_stride),
                    'unk':    sum(all_vert_cnts),
                    'offset': 0,
                })

                for i, (idx_bytes, n_verts) in enumerate(zip(all_idx_bufs, all_vert_cnts)):
                    lod_grp, sub_idx, unk2 = _sm_meta(i)

                    # Re-pack indices with the global vertex offset added.
                    # Indices arrive as uint16 in the file's byte order.
                    n_idx = len(idx_bytes) // 2
                    local = struct.unpack(f'{endian}{n_idx}H', idx_bytes)
                    glob  = tuple(v + cum_verts_count for v in local)
                    # The shared-VB engine requires uint16 indices — bail
                    # loudly if the global index would overflow rather than
                    # silently corrupting the file.
                    if cum_verts_count + n_verts > 65535:
                        return ({'CANCELLED'},
                                f"Skinned shared-VB LOD would need >65535 "
                                f"vertices (have {cum_verts_count + n_verts}). "
                                f"Character/skinned meshes must keep their "
                                f"shared-VB layout (per-VB can crash them), so "
                                f"reduce the combined vertex count — decimate, "
                                f"or inject fewer models at once. (Static meshes "
                                f"auto-split to per-VB and don't hit this.)")
                    glob_bytes = struct.pack(f'{endian}{n_idx}H', *glob)

                    submeshes_new.append({
                        'vb_idx':      0,                                # shared VB
                        'lod_grp':     lod_grp,
                        'sub_idx':     sub_idx,
                        'idx_offset':  idx_u16_offset,
                        'vert_marker': cum_verts_count + n_verts - 1,    # last global vert
                        'unk1':        cum_verts_bytes,                  # byte offset of first vert
                        'unk2':        unk2,
                    })
                    combined_idx   += glob_bytes
                    idx_u16_offset += n_idx
                    cum_verts_bytes += n_verts * vert_stride
                    cum_verts_count += n_verts
            else:
                # Per-submesh VB layout (matches the source).
                byte_offset = 0
                for i, vb_bytes in enumerate(all_vert_bufs):
                    vb_info_new.append({
                        'flags':  vert_fmt,
                        'stride': vert_stride,
                        'unk':    all_vert_cnts[i],
                        'offset': byte_offset,
                    })
                    combined_vert += vb_bytes
                    byte_offset   += len(vb_bytes)

                for i, (idx_bytes, n_verts) in enumerate(zip(all_idx_bufs, all_vert_cnts)):
                    lod_grp, sub_idx, unk2 = _sm_meta(i)

                    # unk1 = byte offset of this submesh's vertices within the
                    # per-VB layout; equals the matching vb_info entry's offset.
                    unk1 = vb_info_new[i]['offset']

                    submeshes_new.append({
                        'vb_idx':      i,
                        'lod_grp':     lod_grp,
                        'sub_idx':     sub_idx,
                        'idx_offset':  idx_u16_offset,
                        'vert_marker': n_verts - 1,
                        'unk1':        unk1,
                        'unk2':        unk2,
                    })
                    combined_idx   += idx_bytes
                    idx_u16_offset += len(idx_bytes) // 2

            dnks_updates = list(zip(all_face_cnts, all_vert_cnts))

        # -- Patch target LOD in the SDOL structure -----------------------
        lod.vb_info             = vb_info_new
        lod.submeshes           = submeshes_new
        lod.vert_data           = bytes(combined_vert)
        lod.vert_section_size   = len(combined_vert)
        lod.indice_data         = bytes(combined_idx)
        lod.indice_section_size = idx_u16_offset

        # ── Full SDOL build dump for this LOD ──────────────────────────
        TraceLogger.table(
            f"SDOL LOD{target_lod} vb_info",
            ("vb_idx", "flags", "stride", "unk(vert_count)", "offset"),
            [(i, f"0x{v['flags']:04X}", v['stride'], v['unk'], v['offset'])
             for i, v in enumerate(vb_info_new)],
            tier="DEBUG", event="sdol_vb_info")
        TraceLogger.table(
            f"SDOL LOD{target_lod} submeshes",
            ("idx","vb_idx","lod_grp","sub_idx","idx_offset","vert_marker",
             "unk1","unk2"),
            [(i, sm['vb_idx'], sm['lod_grp'], sm['sub_idx'], sm['idx_offset'],
              sm['vert_marker'], sm['unk1'], sm['unk2'])
             for i, sm in enumerate(submeshes_new)],
            tier="DEBUG", event="sdol_submeshes")
        TraceLogger.kvblock(
            f"SDOL LOD{target_lod} section sizes",
            [
                ("vb_count",            len(vb_info_new)),
                ("submesh_count",       len(submeshes_new)),
                ("vert_section_bytes",  len(combined_vert)),
                ("indice_section_u16",  idx_u16_offset),
                ("indice_section_bytes", idx_u16_offset * 2),
                ("total_face_counts",   list(all_face_cnts)),
                ("total_vert_counts",   list(all_vert_cnts)),
            ],
            tier="DEBUG", event="sdol_sizes")

        # -- LTMR (material table) rebuild --------------------------------
        # Add any NEW material .xbm names referenced by the injected
        # slices and remap each replaced submesh's DNKS mat_id to the
        # new table. GATED: if no new names appear, ltmr_delta == 0 and
        # the file is byte-identical to the old code path (no regression).
        VerboseLogger.log("  ===== LTMR REBUILD (new code path active) =====")
        ltmr_delta = 0
        new_ltmr = None
        ltmr_start = ltmr_old_size = 0
        matid_updates = []
        try:
            li = parse_ltmr_names(file_data, endian)
        except Exception as exc:
            li = None
            VerboseLogger.log(f"  [inject] LTMR parse FAILED ({exc}) -- table left as-is")
            import traceback as _tb
            TraceLogger.info(
                f"  [trace] LTMR parse raised {exc.__class__.__name__}: {exc}",
                event="ltmr_parse_failed",
                data={"exc_type": exc.__class__.__name__,
                      "exc_msg":  str(exc)[:512],
                      "traceback": _tb.format_exc()})
        VerboseLogger.log(f"  [inject] slices ({len(slices)}): "
                          + " | ".join(f"{o.name}:{mn}" for (o, _t, _s, mn) in slices))
        if li:
            ltmr_start, ltmr_old_size, l_ci, l_b, l_d, orig_names = li
            orig_set = set(orig_names)
            VerboseLogger.log(f"  [inject] original LTMR ({len(orig_names)}): {orig_names}")
            VerboseLogger.log(f"  [inject] use_slot_aware={use_slot_aware} "
                              f"slice_slots={slice_slots}")

            def _effective_mat(o, slice_mat):
                # The slice name collapses to material slot 0 when an
                # object isn't split by material. If the object carries a
                # NEW material (exported -> renamed to GRAPHICS\..\.xbm,
                # tagged 'xbg_exported', or simply a name not already in
                # the original LTMR), prefer THAT so a mesh joined into an
                # existing submesh slot adopts its own material.
                for sl in o.material_slots:
                    m = sl.material
                    if not m:
                        continue
                    why = None
                    if m.get('xbg_exported'):
                        why = "tagged xbg_exported"
                    elif (m.name not in orig_set
                          and m.name.lower().endswith('.xbm')):
                        why = "name is .xbm not in original LTMR"
                    if why:
                        VerboseLogger.log(f"  [inject]   '{o.name}': picked NEW material "
                                         f"'{m.name}' ({why}) instead of slice "
                                         f"'{slice_mat}'")
                        return m.name
                VerboseLogger.log(f"  [inject]   '{o.name}': no new material on slots; "
                                  f"keeping slice '{slice_mat}'")
                return slice_mat

            # slot -> material name for the slices we are injecting.
            #
            # In SPLIT-BY-MATERIAL mode every slice already carries its
            # OWN correct material name (one slice per material slot), so
            # we MUST use it verbatim. _effective_mat is only for the
            # non-split case where a joined object collapsed to slot 0 and
            # we have to guess its real (new) material from the slots.
            def _pick(o, mn):
                return mn if inject_materials else _effective_mat(o, mn)

            slot_to_mat = {}
            if use_slot_aware:
                for k, (o, _t, _s, mn) in enumerate(slices):
                    if slice_slots[k] is not None:
                        slot_to_mat[int(slice_slots[k])] = _pick(o, mn)
            else:
                for i, (o, _t, _s, mn) in enumerate(slices):
                    slot_to_mat[i] = _pick(o, mn)
            VerboseLogger.log(f"  [inject] slot_to_mat={slot_to_mat}")
            TraceLogger.table(
                "LTMR slot→material picks",
                ("slot", "object", "slice_mat", "effective_mat"),
                [(k,
                  slices[k][0].name if k < len(slices) else "?",
                  slices[k][3] if k < len(slices) else "?",
                  slot_to_mat.get(k))
                 for k in sorted(slot_to_mat.keys())],
                tier="DEBUG", event="ltmr_picks")
            names = list(orig_names)
            new_names_added = []
            for mn in slot_to_mat.values():
                if mn and mn not in names:
                    names.append(mn)
                    new_names_added.append(mn)
            if names != orig_names:                       # something new
                matid_updates = [None] * n_orig_submeshes
                for slot, mn in slot_to_mat.items():
                    if 0 <= slot < n_orig_submeshes and mn in names:
                        matid_updates[slot] = names.index(mn)
                new_ltmr = build_ltmr_chunk(l_ci, l_b, l_d, names, endian)
                ltmr_delta = len(new_ltmr) - ltmr_old_size
                VerboseLogger.log(f"  LTMR: {len(orig_names)} -> {len(names)} materials "
                                  f"({len(new_ltmr)}B, delta {ltmr_delta:+d}B)  "
                                  f"matid_updates={matid_updates}")
                for nm in names[len(orig_names):]:
                    VerboseLogger.log(f"    + new material: {nm}")
                TraceLogger.table(
                    "LTMR final names list",
                    ("index", "name", "origin"),
                    [(i, nm, "<orig>" if nm in orig_names else "<NEW>")
                     for i, nm in enumerate(names)],
                    tier="DEBUG", event="ltmr_final_names")
                TraceLogger.kvblock(
                    "LTMR rebuild summary",
                    [
                        ("orig_count",     len(orig_names)),
                        ("new_count",      len(names)),
                        ("added",          new_names_added),
                        ("matid_updates",  list(matid_updates)),
                        ("old_size_B",     ltmr_old_size),
                        ("new_size_B",     len(new_ltmr)),
                        ("size_delta_B",   ltmr_delta),
                    ],
                    tier="DEBUG", event="ltmr_rebuild_done")
            else:
                VerboseLogger.log("  [inject] LTMR unchanged: no NEW material names in "
                                  "the injected slices (slot_to_mat already in LTMR). "
                                  "If you expected new materials, the injected object's "
                                  "slots still carry the original game material names — "
                                  "run 'Export Custom Materials' first so they are "
                                  "renamed to GRAPHICS\\_MATERIALS\\<name>.xbm.")

        # -- Rebuild DNKS when the submesh COUNT changes -----------------
        # patch_dnks only edits existing records, so a split-by-material
        # that produces MORE submeshes than the original needs the whole
        # target-LOD DNKS section rebuilt (new count + per-submesh
        # mat_id/counts/bone-palette). GATED: only when the count differs
        # -> count unchanged keeps the proven patch_dnks path verbatim.
        new_dnks = None
        dnks_start = dnks_old_size = dnks_delta = 0
        n_new_sm = len(submeshes_new)

        # Decide whether to rebuild DNKS at all. We rebuild when:
        #   (a) the submesh count changed (add/remove submeshes), OR
        #   (b) any bbox / submesh-name differs from what's already
        #       stored in the trailing — otherwise the engine reads
        #       stale per-submesh bboxes and crashes on big / replaced
        #       geometry even though the count happens to match.
        _dnks_rebuild_reason = None
        if n_new_sm != n_orig_submeshes:
            _dnks_rebuild_reason = (
                f"count change ({n_orig_submeshes} -> {n_new_sm})")
        else:
            # Cheap pre-check: aggregate the new geometry's bbox over all
            # submeshes of this LOD and compare against the LOD's stored
            # aggregate bbox (one entry per LOD-GROUP in the trailing).
            try:
                _di_peek = parse_dnks_full(file_data, endian)
            except Exception:
                _di_peek = None
            if _di_peek and target_lod < len(_di_peek[4]):
                _tr_entries = parse_dnks_trailing(_di_peek[5], endian)
                if _tr_entries is not None:
                    _tgt_lod = next((e for e in _tr_entries
                                     if e['lod'] == target_lod), None)
                    if _tgt_lod is not None and all_bboxes:
                        nmn = [ float('inf')] * 3
                        nmx = [-float('inf')] * 3
                        for bb in all_bboxes:
                            for axis in range(3):
                                if bb[0][axis] < nmn[axis]: nmn[axis] = bb[0][axis]
                                if bb[1][axis] > nmx[axis]: nmx[axis] = bb[1][axis]
                        if nmn[0] != float('inf'):
                            old_min = _tgt_lod['bb_min']
                            old_max = _tgt_lod['bb_max']
                            # 1 cm threshold — quantisation drift on a stable
                            # round-trip is well under a mm; anything bigger
                            # means the user actually moved geometry.
                            if any(abs(a - b) > 0.01
                                   for a, b in zip(tuple(nmn) + tuple(nmx),
                                                    old_min + old_max)):
                                _dnks_rebuild_reason = (
                                    "LOD aggregate bbox drift "
                                    "(stored bound no longer covers the geometry)")

        if _dnks_rebuild_reason is not None:
            VerboseLogger.log(
                f"  [inject] DNKS rebuild requested: {_dnks_rebuild_reason}")
            try:
                di = parse_dnks_full(file_data, endian)
            except Exception as exc:
                di = None
                VerboseLogger.log(f"  [inject] DNKS parse_full failed ({exc})")
                import traceback as _tb
                TraceLogger.info(
                    f"  [trace] DNKS parse_full raised {exc.__class__.__name__}: {exc}",
                    event="dnks_parse_full_failed",
                    data={"exc_type": exc.__class__.__name__,
                          "exc_msg":  str(exc)[:512],
                          "traceback": _tb.format_exc()})
            if di and target_lod < len(di[4]):
                (dnks_start, dnks_old_size, d_ci,
                 d_pre, d_blocks, d_trail) = di
                stride_h4, fmt_h6 = _dnks_template_hfields(
                    d_blocks[target_lod], endian)
                # Per-submesh face/vert/palette MUST be keyed the same way as
                # submeshes_new (length n_new_sm).  all_face_cnts/all_vert_cnts/
                # all_palettes are keyed by REAL SLICE — in slot-aware mode the
                # injector pads unfilled slots with NULL submeshes, so those
                # lists are SHORTER than submeshes_new and indexing them by
                # submesh i overruns (IndexError on null/compacted slots).
                #   counts  -> dnks_updates  (built per-submesh in BOTH modes:
                #              zip(face,vert) sequentially, or one (face,vert)
                #              per slot incl. nulls in slot-aware mode).
                #   palette -> map each real slice's palette onto its slot
                #              (FIRST slice per slot wins, matching slot_map so
                #              geometry and palette stay paired); nulls get [].
                if use_slot_aware:
                    _slot_pal = {}
                    for _si, _slot in enumerate(slice_slots):
                        if (_slot is not None and _si < len(all_palettes)
                                and _slot not in _slot_pal):
                            _slot_pal[_slot] = all_palettes[_si]
                    _sm_pal = [_slot_pal.get(s, []) for s in range(n_new_sm)]
                else:
                    _sm_pal = [all_palettes[i] if i < len(all_palettes) else []
                               for i in range(n_new_sm)]
                sm_list = []
                for i in range(n_new_sm):
                    mn = (slot_to_mat.get(i)
                          if 'slot_to_mat' in dir() else None)
                    if mn and 'names' in dir() and mn in names:
                        mid = names.index(mn)
                    else:
                        mid = (submeshes_new[i].get('sub_idx', i)
                               if i < n_orig_submeshes else 0)
                    _fc, _vc = (dnks_updates[i] if i < len(dnks_updates)
                                else (all_face_cnts[i] if i < len(all_face_cnts)
                                      else 0,
                                      all_vert_cnts[i] if i < len(all_vert_cnts)
                                      else 0))
                    sm_list.append({
                        'mat_id':     mid,
                        'face_count': _fc,
                        'vert_count': _vc,
                        'palette':    _sm_pal[i],
                    })
                _old_block_len = len(d_blocks[target_lod])
                d_blocks[target_lod] = build_dnks_lod_block(
                    sm_list, stride_h4, fmt_h6, endian)
                _blk_delta = len(d_blocks[target_lod]) - _old_block_len

                # ---- Rebuild trailing (per-submesh bboxes + names) -------
                # The trailing carries one entry per submesh GLOBALLY (all
                # LODs interleaved). The engine uses these bboxes for
                # per-submesh culling and LOD selection, and the names for
                # runtime lookups. Leaving them stale is what crashes the
                # game when new geometry is injected, the model is scaled,
                # or a submesh is renamed.
                _tr_entries = parse_dnks_trailing(d_trail, endian)
                _trail_delta = 0
                if _tr_entries is not None:
                    # DNKS trailing reality-check (verified on stock kendra):
                    # block_count = number of LOD-GROUPS, NOT submeshes.
                    # There is ONE entry per LOD (4 for kendra) whose name
                    # ("NPC_KENDRA_BODY_LOD0", ...) is the GROUP name for
                    # every submesh under that LOD, and whose bbox is the
                    # aggregate bbox spanning ALL submeshes of that LOD.
                    #
                    # We therefore only ever rewrite the target LOD's
                    # single entry — never add or remove rows.  Adding
                    # per-submesh rows here is what was corrupting the
                    # chunk (block_count went 4 -> 10, engine read past
                    # the real trailing into garbage).  The name is left
                    # ALONE: this is the engine's per-LOD animation /
                    # cloth lookup key and Blender object names have no
                    # relationship to it.
                    mn = [ float('inf')] * 3
                    mx = [-float('inf')] * 3
                    for bb in all_bboxes:
                        for axis in range(3):
                            if bb[0][axis] < mn[axis]: mn[axis] = bb[0][axis]
                            if bb[1][axis] > mx[axis]: mx[axis] = bb[1][axis]
                    if mn[0] == float('inf'):
                        mn = (0.0, 0.0, 0.0); mx = (0.0, 0.0, 0.0)
                    _tr_entries = update_dnks_trailing_lod_bbox(
                        _tr_entries, target_lod, mn, mx)
                    _new_trail = build_dnks_trailing(_tr_entries, endian)
                    _trail_delta = len(_new_trail) - len(d_trail)
                    d_trail = _new_trail
                    _kept_name = next((e['name'] for e in _tr_entries
                                       if e['lod'] == target_lod), '?')
                    VerboseLogger.log(
                        f"  [inject] DNKS trailing updated: LOD{target_lod} "
                        f"aggregate bbox refreshed "
                        f"min({mn[0]:+.3f},{mn[1]:+.3f},{mn[2]:+.3f}) "
                        f"max({mx[0]:+.3f},{mx[1]:+.3f},{mx[2]:+.3f}); "
                        f"name '{_kept_name}' preserved; entry count unchanged "
                        f"({len(_tr_entries)} LOD groups, delta {_trail_delta:+d}B)")
                else:
                    VerboseLogger.log(
                        "  [inject] WARNING: DNKS trailing parse failed -- "
                        "per-LOD bbox left stale (engine may crash)")

                # The 28-byte DNKS preamble carries internal size fields
                # the GAME uses to size the submesh/skin array (Blender's
                # parser ignores them). Verified: u32@20 = total submesh-
                # block bytes (Σ records + per-LOD count ints); u32@16 =
                # that + a constant. Both must grow by the block delta or
                # the engine reads a truncated array and culls the mesh.
                #
                # The FIRST u32 of the preamble (offset 0) is the trailing
                # names-section size in bytes — it MUST track the trailing
                # delta or the engine reads the wrong number of name
                # entries and walks off the end of the chunk.
                d_pre = bytearray(d_pre)
                if _trail_delta and len(d_pre) >= 4:
                    _v = struct.unpack_from(f'{endian}I', d_pre, 0)[0]
                    struct.pack_into(f'{endian}I', d_pre, 0,
                                     _v + _trail_delta)
                    VerboseLogger.log(f"  [inject] DNKS preamble u32@0 (trailing size): "
                                     f"{_v} -> {_v + _trail_delta}")
                for _po in (16, 20):
                    if _po + 4 <= len(d_pre):
                        _v = struct.unpack_from(f'{endian}I', d_pre, _po)[0]
                        struct.pack_into(f'{endian}I', d_pre, _po,
                                         _v + _blk_delta)
                        VerboseLogger.log(f"  [inject] DNKS preamble u32@{_po}: "
                                         f"{_v} -> {_v + _blk_delta}")

                new_dnks = build_dnks_chunk(d_ci, bytes(d_pre), d_blocks,
                                            d_trail, endian)
                dnks_delta = len(new_dnks) - dnks_old_size
                VerboseLogger.log(f"  [inject] DNKS REBUILT LOD{target_lod}: "
                                  f"{n_orig_submeshes} -> {n_new_sm} submeshes "
                                  f"(old {dnks_old_size}B new {len(new_dnks)}B "
                                  f"delta {dnks_delta:+d})  mat_ids="
                                  f"{[s['mat_id'] for s in sm_list]}")

                # ── Detailed DNKS block-region dump ────────────────────
                TraceLogger.table(
                    f"DNKS LOD{target_lod} submesh block",
                    ("idx","mat_id","face_count","vert_count",
                     "palette_used","palette_first_4"),
                    [(i, s['mat_id'], s['face_count'], s['vert_count'],
                      sum(1 for x in (s.get('palette') or []) if x is not None and x >= 0),
                      list((s.get('palette') or [])[:4]))
                     for i, s in enumerate(sm_list)],
                    tier="DEBUG", event="dnks_block_dump")
                # Trailing entries (one per LOD)
                if _tr_entries is not None:
                    TraceLogger.table(
                        f"DNKS trailing entries (final)",
                        ("idx","lod","name","metric","bb_min","bb_max"),
                        [(i, e['lod'], e['name'], round(e.get('metric', 0), 2),
                          tuple(round(v,3) for v in e['bb_min']),
                          tuple(round(v,3) for v in e['bb_max']))
                         for i, e in enumerate(_tr_entries)],
                        tier="DEBUG", event="dnks_trailing_dump")
                TraceLogger.kvblock(
                    "DNKS rebuild summary",
                    [
                        ("submesh_count_old",   n_orig_submeshes),
                        ("submesh_count_new",   n_new_sm),
                        ("block_delta_B",       _blk_delta),
                        ("trail_delta_B",       _trail_delta),
                        ("total_dnks_old_B",    dnks_old_size),
                        ("total_dnks_new_B",    len(new_dnks)),
                        ("total_dnks_delta_B",  dnks_delta),
                    ],
                    tier="DEBUG", event="dnks_rebuild_summary")
            else:
                VerboseLogger.log("  [inject] DNKS rebuild SKIPPED (parse/LOD issue) — "
                                  "submesh count change will NOT be honoured")

        # -- Rebuild SDOL (its start shifts by LTMR + DNKS deltas) -------
        sdol_shift = ltmr_delta + dnks_delta
        ci0      = struct.unpack_from(f'{endian}i', file_data, sdol_start + 4)[0]
        new_sdol = build_sdol_chunk(sdol, sdol_start + sdol_shift, ci0, endian)
        size_delta = len(new_sdol) - sdol_old_size
        VerboseLogger.log(f"  SDOL old: {sdol_old_size}B -> new: {len(new_sdol)}B  "
                          f"(delta: {size_delta:+d}B)  sdol_shift={sdol_shift:+d}")

        # -- Generic in-order splice of every rebuilt chunk -------------
        regions = [(sdol_start, sdol_old_size, bytes(new_sdol))]
        if new_ltmr is not None:
            regions.append((ltmr_start, ltmr_old_size, bytes(new_ltmr)))
        if new_dnks is not None:
            regions.append((dnks_start, dnks_old_size, bytes(new_dnks)))
        regions.sort(key=lambda r: r[0])
        new_file = bytearray()
        cur = 0
        for (st, osz, nb) in regions:
            new_file += file_data[cur:st]
            new_file += nb
            cur = st + osz
        new_file += file_data[cur:]
        VerboseLogger.log("  [inject] spliced chunks: "
                          + ", ".join(f"@0x{s:X} {o}->{len(n)}B" for (s, o, n) in regions))

        # ── Trace: chunk-by-chunk splice table with byte ranges ────────────
        try:
            rows = []
            for (st, osz, nb) in regions:
                # Sniff the magic of the new bytes for the table label.
                mag = nb[:4]
                tag = ''.join(chr(b) if 32 <= b < 127 else '.' for b in mag)
                rows.append((tag,
                             f"0x{st:08X}",
                             osz,
                             len(nb),
                             f"{len(nb)-osz:+d}",
                             f"0x{st + len(nb):08X}"))
            TraceLogger.table(
                "spliced chunks (byte map)",
                ("magic", "orig_offset", "orig_size", "new_size", "delta", "new_end"),
                rows, tier="DEBUG", event="splice_map")
            TraceLogger.kv("orig_file_size", len(file_data), tier="DEBUG")
            TraceLogger.kv("new_file_size",  len(new_file),  tier="DEBUG")
            TraceLogger.kv("total_size_delta", len(new_file) - len(file_data),
                           tier="DEBUG", event="splice_size_delta")
        except Exception as _exc:
            TraceLogger.debug(f"  [trace] splice-map table failed: {_exc}",
                              event="splice_map_failed",
                              data={"err": str(_exc)[:256]})

        # -- Patch header file-offset words --------------------------------
        # CRITICAL: header @4 is a CONSTANT (393258 in every XBG —
        # npc_kendra/viperwolf/direhorse all identical; it isn't a
        # pointer). The ONLY genuine header pointer is the one that points
        # at/after the ORIGINAL SDOL end (e.g. @20 -> last chunk's data),
        # plus the literal file-size word. The proven original injector
        # only ever shifted values >= sdol_end_orig; shifting anything
        # below that (like the @4 constant) corrupts a field the game
        # validates and it refuses to render. So: only post-SDOL pointers
        # shift, by the FULL combined delta (LTMR+DNKS+SDOL growth).
        orig_len      = len(file_data)
        new_len       = len(new_file)
        sdol_end_orig = sdol_start + sdol_old_size
        total_delta   = sum(len(nb) - osz for (_st, osz, nb) in regions)
        if new_len != orig_len:
            patched_any = False
            for hdr_off in (4, 8, 12, 16, 20, 24):
                val = struct.unpack_from(f'{endian}I', new_file, hdr_off)[0]
                new_val = None
                if val == orig_len:
                    new_val = new_len
                elif sdol_end_orig <= val < orig_len:
                    new_val = val + total_delta      # genuine tail pointer
                # values < sdol_end_orig are constants / pre-SDOL and are
                # left ALONE (this is what made simple injects work).
                if new_val is not None and new_val != val:
                    struct.pack_into(f'{endian}I', new_file, hdr_off, new_val)
                    VerboseLogger.log(f"  [inject] Header @ {hdr_off}: {val} -> {new_val}"
                                     f"  (total_delta={total_delta:+d})")
                    patched_any = True
            if not patched_any:
                VerboseLogger.log(f"  [inject] NOTE: no header field needed shifting "
                                  f"(orig_len={orig_len}, new_len={new_len})")

        # -- DNKS face/vert + mat-id patching ----------------------------
        if new_dnks is not None:
            VerboseLogger.log("  [inject] DNKS was fully rebuilt — skipping patch_dnks/"
                              "patch_dnks_matids (counts & mat_ids already correct)")
        else:
            new_file = bytearray(
                patch_dnks(new_file, target_lod, dnks_updates, endian))
            if new_ltmr is not None and any(m is not None
                                            for m in matid_updates):
                new_file = bytearray(
                    patch_dnks_matids(new_file, target_lod,
                                      matid_updates, endian))

        # -- Patch bounding volumes ---------------------------------------
        # In slot-aware mode with null slots the injected objects cover only
        # a subset of the character (e.g. head only).  Computing XOBB/HPSB
        # from just those objects would produce a tiny bounding sphere that
        # tricks the game into showing lower LODs at normal viewing distance
        # (and may cause frustum-culling to reject the whole model).  Keep
        # the original bounds so LOD selection works correctly.
        has_null_slots = use_slot_aware and (len(slot_map) < n_orig_submeshes)
        if has_null_slots:
            VerboseLogger.log("  [inject] Bounds NOT updated (partial inject — original bounds preserved)")
        else:
            new_file = bytearray(
                patch_bounds(new_file, mesh_objects,
                             effective_pos_scale, import_mesh_only, endian))

        # -- FINAL uint16 wall (pre-write) --------------------------------
        # Re-parse the DNKS submesh headers we are about to commit and
        # refuse to write if ANY index_count / vert_count would overflow
        # the format's uint16 field. This catches not just the splitter
        # but any downstream bug, so a GPU-crashing file can never reach
        # disk. (The original crash was a silently clamped field — this
        # is the wall that makes that class of bug impossible to ship.)
        try:
            _vd = find_chunk(new_file, 'DNKS', endian)
            if _vd:
                _p = _vd[1] + 28
                for _l in range(target_lod + 1):
                    _mc = struct.unpack_from(f'{endian}i', new_file, _p)[0]
                    _p += 4
                    for _s in range(_mc):
                        _mid, _fc1, _fc2, _ic, _st, _vc, _fmt = \
                            struct.unpack_from(f'{endian}7H', new_file, _p)
                        # _ic = vert_marker (max vertex index used); _vc = vert count.
                        # These must fit in uint16. face_count * 3 = index ENTRY count,
                        # which does NOT need to fit in uint16 — the index buffer is a
                        # plain byte array; only the individual index VALUES need to be
                        # < 65536. Checking _fc1*3 > 65535 was wrong and aborted valid
                        # submeshes (e.g. 23274 faces, max vert index 22148 — perfectly
                        # representable in uint16).
                        if _ic > _HARD_MAX_IDX or _vc > _HARD_MAX_VERT:
                            return {'CANCELLED'}, (
                                f"ABORTED before write: DNKS LOD{_l} "
                                f"submesh {_s} overflows the uint16 limit "
                                f"(faces={_fc1} idx={_ic} verts={_vc}). "
                                f"This would crash the GPU in-game. The "
                                f"output file was NOT written. Please "
                                f"report this — the auto-splitter should "
                                f"have prevented it.")
                        _p += 110
        except Exception as exc:
            VerboseLogger.log(f"  [inject] pre-write DNKS guard skipped: {exc}")

        # -- Write output -------------------------------------------------
        try:
            with open(output_path, 'wb') as f:
                f.write(new_file)
        except Exception as exc:
            return {'CANCELLED'}, f"Failed to write output: {exc}"

        # -- Verify what actually landed in the written file --------------
        try:
            vi = parse_ltmr_names(bytes(new_file), endian)
            if vi:
                VerboseLogger.log(f"  [inject] VERIFY output LTMR: {len(vi[5])} "
                                  f"materials -> {vi[5]}")
            vd = find_chunk(new_file, 'DNKS', endian)
            if vd:
                _vp = vd[1] + 28
                for _l in range(target_lod + 1):
                    _mc = struct.unpack_from(f'{endian}i', new_file, _vp)[0]
                    _vp += 4
                    _ids = []
                    for _s in range(_mc):
                        _ids.append(struct.unpack_from(
                            f'{endian}H', new_file, _vp)[0])
                        _vp += 110
                    if _l == target_lod:
                        VerboseLogger.log(f"  [inject] VERIFY DNKS LOD{_l} mat_ids={_ids}")
        except Exception as exc:
            VerboseLogger.log(f"  [inject] VERIFY failed: {exc}")

        # ── Extended post-write verification (trace) ──────────────────────
        # Re-parse the bytes about to be written and confirm every key
        # field matches what we intended.  This is the wall that catches
        # silent splice corruption (length mismatch, wrong alignment,
        # truncated chunks, etc.) BEFORE the file hits disk.
        try:
            new_bytes = bytes(new_file)
            # 1) chunk walk: every chunk's magic + size should be readable
            #    until the file_size word, then we should be exactly at EOF.
            chunks_found = []
            cc_header = struct.unpack_from(f'{endian}I', new_bytes, 28)[0]
            p = 32
            for _ in range(cc_header):
                if p + 12 > len(new_bytes):
                    chunks_found.append(("<truncated>", p, 0))
                    break
                mag = new_bytes[p:p+4]
                # PS3 stores magic byte-reversed; we just dump whatever's there.
                tag = ''.join(chr(b) if 32 <= b < 127 or b == 0 else '?' for b in mag)
                _ci, _sz = struct.unpack_from(f'{endian}ii', new_bytes, p+4)
                chunks_found.append((tag, p, _sz))
                if _sz < 12 or p + _sz > len(new_bytes):
                    chunks_found.append(("<bad_size>", p, _sz))
                    break
                p += _sz
            TraceLogger.table(
                "post-write chunk walk",
                ("magic", "offset", "size"),
                [(t, f"0x{o:08X}", s) for (t, o, s) in chunks_found],
                tier="DEBUG", event="postwrite_chunk_walk")
            TraceLogger.kv("file_size",       len(new_bytes), tier="DEBUG")
            TraceLogger.kv("header_chunk_count", cc_header,   tier="DEBUG")
            TraceLogger.kv("walked_chunks",   len(chunks_found),
                           tier="DEBUG", event="postwrite_chunk_count")

            # 2) PMCP scale that actually landed
            _pi = find_chunk(new_bytes, 'PMCP', endian)
            if _pi:
                _ps_offset = _pi[1] + 16   # historical: +28 absolute → +16 from data_start
                # Defensive: scan 4-byte floats in the chunk and pick the
                # one that looks like a scale (in (0, 1)).
                _ds = _pi[1]
                _cs_payload = _pi[2] - 12
                _candidates = []
                for off in range(0, _cs_payload, 4):
                    if off + 4 > _cs_payload: break
                    f = struct.unpack_from(f'{endian}f', new_bytes, _ds + off)[0]
                    if 0.0 < f < 1.0:
                        _candidates.append((off, f))
                TraceLogger.kv("PMCP scale candidates", _candidates,
                               tier="DEBUG", event="postwrite_pmcp")

            # 3) XOBB / HPSB final values
            _xi = find_chunk(new_bytes, 'XOBB', endian)
            if _xi:
                _vals = struct.unpack_from(f'{endian}6f', new_bytes, _xi[0] + 20)
                TraceLogger.kv("XOBB min", tuple(round(v, 4) for v in _vals[:3]),
                               tier="DEBUG")
                TraceLogger.kv("XOBB max", tuple(round(v, 4) for v in _vals[3:]),
                               tier="DEBUG", event="postwrite_xobb")
            _hi = find_chunk(new_bytes, 'HPSB', endian)
            if _hi:
                _vals = struct.unpack_from(f'{endian}4f', new_bytes, _hi[0] + 20)
                TraceLogger.kv("HPSB center", tuple(round(v, 4) for v in _vals[:3]),
                               tier="DEBUG")
                TraceLogger.kv("HPSB radius", round(_vals[3], 4),
                               tier="DEBUG", event="postwrite_hpsb")

            # 4) SDOL LOD-0 sanity: parse VB count, submesh count, vert
            #    section size, index section size, and the global min/max
            #    index value.  If any of these are inconsistent, the file
            #    will fail to re-import or render garbage.
            _si = find_chunk(new_bytes, 'SDOL', endian)
            if _si and target_lod == 0:
                _sp = _si[1] + 12  # skip unk_0, unk_1, lod_count
                _lod_dist = struct.unpack_from(f'{endian}f', new_bytes, _sp)[0]; _sp += 4
                _vb_count = struct.unpack_from(f'{endian}i', new_bytes, _sp)[0]; _sp += 4
                _vbs = []
                for _ in range(_vb_count):
                    _vbs.append(struct.unpack_from(f'{endian}4i', new_bytes, _sp))
                    _sp += 16
                _sm_count = struct.unpack_from(f'{endian}i', new_bytes, _sp)[0]; _sp += 4
                _sms = []
                for _ in range(_sm_count):
                    _sms.append(struct.unpack_from(f'{endian}7i', new_bytes, _sp))
                    _sp += 28
                _vss = struct.unpack_from(f'{endian}I', new_bytes, _sp)[0]; _sp += 4
                _rem = _sp % 16
                if _rem: _sp += 16 - _rem
                _vert_start = _sp; _sp += _vss
                _iss = struct.unpack_from(f'{endian}I', new_bytes, _sp)[0]; _sp += 4
                _rem = _sp % 16
                if _rem: _sp += 16 - _rem
                _idx_start = _sp
                _all_idx = struct.unpack_from(f'{endian}{_iss}H', new_bytes, _idx_start)
                _imn = min(_all_idx) if _all_idx else 0
                _imx = max(_all_idx) if _all_idx else 0
                _stride = _vbs[0][1] if _vbs else 0
                _total_vb_verts = (_vss // _stride) if _stride else 0
                TraceLogger.kvblock(
                    "post-write SDOL LOD0",
                    [
                        ("vb_count",        _vb_count),
                        ("vb[0]",           _vbs[0] if _vbs else None),
                        ("sm_count",        _sm_count),
                        ("vert_section_B",  _vss),
                        ("total_verts_in_vb", _total_vb_verts),
                        ("index_count",     _iss),
                        ("global_idx_min",  _imn),
                        ("global_idx_max",  _imx),
                        ("idx_max_within_vb", _imx < _total_vb_verts),
                    ],
                    tier="DEBUG", event="postwrite_sdol")
                if _imx >= _total_vb_verts:
                    TraceLogger.info(
                        f"  [trace] *** WARNING: global max index ({_imx}) "
                        f">= total VB verts ({_total_vb_verts}) — file will crash on re-import",
                        event="postwrite_idx_overflow",
                        data={"max_idx": _imx, "total_verts": _total_vb_verts})
        except Exception as exc:
            TraceLogger.info(f"  [trace] post-write verify failed: {exc}",
                             event="postwrite_failed",
                             data={"err": str(exc)[:512]})

        # -- Result message -----------------------------------------------
        total_verts = sum(all_vert_cnts)
        total_faces = sum(all_face_cnts)
        msg = (f"Injected {len(slices)} submesh(es): "
               f"{total_verts}v / {total_faces}t into LOD {target_lod}")
        if override_game_scale:
            msg += f" (PMCP scale={target_game_scale:.4f})"
        if total_clamped > 0:
            msg += f" (warning: {total_clamped} verts clamped)"

        VerboseLogger.log(f"\n+ {msg}")
        VerboseLogger.log(f"  Output: {output_path}")
        VerboseLogger.log(f"{'='*60}\n")
        TraceLogger.struct("inject_finished",
                           {"output": str(output_path),
                            "n_submeshes": len(slices),
                            "total_verts": total_verts,
                            "total_faces": total_faces,
                            "msg": msg},
                           tier="INFO")
        return {'FINISHED'}, msg
