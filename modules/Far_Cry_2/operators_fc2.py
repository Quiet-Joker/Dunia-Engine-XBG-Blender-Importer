"""Avatar: The Game — operators (import / inject / MAB / LKS / materials).

Split out of the monolithic __init__.py (2026-06-09 refactor).
"""
import os
import re
import struct
import threading

import bpy

from ..Core.debug import VerboseLogger
from .binary_fc2 import detect_endian_from_bytes, encode_chunk_magic, LE, BE
from ..Core.prefs import get_prefs
from ..Core.settings import XBGMatTemplateItem, _TEMPLATE_DESCRIPTIONS
from .import_xbg_fc2 import XBGBlenderImporter
from .inject_xbg_fc2 import XBGMeshInjector, calculate_required_scale
from .import_lks_fc2 import parse_lks_file, create_lks_armature
from .import_mab_fc2 import apply_multi_bone as _mab_apply, parse_sections as _mab_parse_sections
from . import export_materials_fc2 as _export_materials


class XBG_OT_ImportFC2(bpy.types.Operator):
    bl_idname = "import_scene.xbg_model_fc2"
    bl_label = "Import XBG"
    bl_options = {'REGISTER', 'UNDO'}

    filepath: bpy.props.StringProperty(subtype="FILE_PATH")
    files: bpy.props.CollectionProperty(type=bpy.types.OperatorFileListElement)
    directory: bpy.props.StringProperty(subtype="DIR_PATH")
    filter_glob: bpy.props.StringProperty(default="*.xbg", options={'HIDDEN'})

    import_mesh_only: bpy.props.BoolProperty(
        name="Import Mesh Only",
        description="Skip skeleton import and rigging",
        default=False
    )
    reorient_bones: bpy.props.BoolProperty(
        name="Reorient Bones",
        description="Point each bone's tail toward its children, making the skeleton easier to read and pose. Leaf bones keep their original orientation.",
        default=False
    )
    import_all_lods: bpy.props.BoolProperty(
        name="Import All LODs",
        description="Import all Level of Details found in file",
        default=False
    )
    lod_level: bpy.props.IntProperty(
        name="LOD Level",
        description="Which LOD to import (0=highest detail, higher=lower detail)",
        default=0,
        min=0,
        max=10
    )

    def draw(self, context):
        layout = self.layout
        box = layout.box()
        box.label(text="LOD Selection:", icon='MOD_MULTIRES')
        box.prop(self, "import_all_lods")
        row = box.row()
        row.enabled = not self.import_all_lods
        row.prop(self, "lod_level")
        if not self.import_all_lods:
            box.label(text=f"Will import LOD {self.lod_level} only", icon='INFO')
        else:
            box.label(text="Will import ALL LODs", icon='INFO')
        box = layout.box()
        box.label(text="Other Options:", icon='PREFERENCES')
        box.prop(self, "import_mesh_only")
        row = box.row()
        row.enabled = not self.import_mesh_only
        row.prop(self, "reorient_bones")

    def invoke(self, ctx, ev):
        ctx.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}

    def execute(self, ctx):
        s, ds, p = ctx.scene.xbg_settings, ctx.scene.xbg_debug_settings, get_prefs(ctx)
        VerboseLogger.enabled = ds.verbose_logging
        VerboseLogger.clear()           # text-buf only; records persist
        from ..Core.debug import TraceLogger
        TraceLogger.set_trace(ds.verbose_logging and ds.trace_logging)
        df, lt, lhd = p.data_folder, s.load_textures, s.load_hd_textures

        VerboseLogger.session_marker(
            "import_xbg",
            file=self.filepath if not self.files else f"<batch:{len(self.files)} files>",
            lod=("ALL" if self.import_all_lods else self.lod_level),
            load_textures=lt, load_hd_textures=lhd, data_folder=df or "<unset>")
        _log_pre_import_scene_snapshot(ctx)

        if lt and not df:
            self.report({'WARNING'}, "Data folder not set - textures will not be loaded")
            lt = False

        imp = XBGBlenderImporter()
        tl = -1 if self.import_all_lods else self.lod_level

        fs = []
        if self.files:
            for f in self.files:
                if f.name.lower().endswith(".xbg"):
                    fs.append(os.path.join(self.directory, f.name))
        elif self.filepath.lower().endswith(".xbg"):
            fs.append(self.filepath)

        if not fs:
            self.report({'ERROR'}, "No valid .xbg files selected")
            return {'CANCELLED'}

        ic = 0
        if ds.import_xbt_as_dds:
            self.report({'WARNING'}, "DDS Import Mode enabled - Texture painting will be corrupted! Use PNG mode for texture painting.")

        for fp in fs:
            try:
                imp.load(
                    ctx, fp, tl, self.import_mesh_only, df, lt, lhd,
                    ds.flip_normals, ds.use_xml_assembly, ds.separate_primitives,
                    False, ds.import_xbt_as_dds,
                    ds.use_mb2o,
                    ds.compact_vertices,
                    self.reorient_bones,
                )
                ic += 1
            except Exception as e:
                import traceback as _tb
                tb_text = _tb.format_exc()
                self.report({'WARNING'}, f"Failed to import {os.path.basename(fp)}: {str(e)}")
                # Always emit a structured failure record + full traceback
                # to BOTH the text log AND the JSONL so the user can find
                # the root cause without needing the console open.
                # CRITICAL: use VerboseLogger.warn (always writes), NOT log
                # (gated by verbose flag) — failures must be visible.
                from ..Core.debug import TraceLogger as _TL
                _TL.info(
                    f"[import] !!! {os.path.basename(fp)} failed: "
                    f"{e.__class__.__name__}: {e}",
                    event="import_operator_failed",
                    data={"file": str(fp),
                          "exc_type": e.__class__.__name__,
                          "exc_msg":  str(e)[:1024],
                          "traceback": tb_text})
                # Force-write the traceback even when Verbose Logging is OFF,
                # via warn() (always writes).  Each traceback line becomes a
                # WARN-tier record in the JSONL.
                VerboseLogger.warn(f"\n=== IMPORT FAILED: {os.path.basename(fp)} ===")
                VerboseLogger.warn(f"    {e.__class__.__name__}: {e}")
                VerboseLogger.warn("    --- traceback ---")
                for tbline in tb_text.splitlines():
                    VerboseLogger.warn(f"    {tbline}")
                VerboseLogger.warn("    --- end traceback ---\n")

        if ic > 0:
            n_objs = len([o for o in bpy.context.scene.objects if o.type == 'MESH'])
            VerboseLogger.session_complete(
                "import_xbg",
                files_imported=ic,
                scene_mesh_objects=n_objs)
            # Sidecar logs next to the first imported .xbg, so users always
            # have a record even if they don't manually click "Save Log".
            if fs:
                VerboseLogger.autosave_sidecar(fs[0])
            self.report({'INFO'}, f"Imported {ic} XBG file(s)")
            return {'FINISHED'}
        else:
            self.report({'ERROR'}, "No files were imported successfully")
            return {'CANCELLED'}


class XBG_OT_RememberXBGFC2(bpy.types.Operator):
    """Store the active object's XBG metadata in the scene so the
    inject panel stays visible after you click away."""
    bl_idname = "xbg.remember_xbg_fc2"
    bl_label = "Remember This XBG for Injection"
    bl_description = (
        "Pin this XBG file to the session so the Inject panel stays "
        "visible even when you select a different object"
    )

    def execute(self, ctx):
        obj = ctx.active_object
        if not obj or "xbg_data" not in obj:
            self.report({'ERROR'}, "Active object has no XBG metadata")
            return {'CANCELLED'}

        meta = obj["xbg_data"].to_dict()
        session = ctx.scene.xbg_session_data
        session.filepath        = meta.get('filepath', '')
        session.pos_scale       = float(meta.get('pos_scale', 1.0))
        session.uv_trans        = float(meta.get('uv_trans', 0.0))
        uv_scale_raw            = meta.get('uv_scale', 1.0)
        # See inject_xbg.py for context: uv_scale is a single float.
        # The sequence branch is defensive against externally-edited
        # IDProperties — use the first element, not the second.
        if isinstance(uv_scale_raw, (list, tuple)) and uv_scale_raw:
            session.uv_scale = float(uv_scale_raw[0])
        else:
            session.uv_scale = float(uv_scale_raw)
        session.import_mesh_only = bool(meta.get('import_mesh_only', False))
        session.pmcp_offset      = int(meta.get('pmcp_offset', 0))
        session.is_loaded        = True

        self.report({'INFO'}, f"Remembered: {os.path.basename(session.filepath)}")
        return {'FINISHED'}


class XBG_OT_ClearSessionXBGFC2(bpy.types.Operator):
    """Clear the pinned session XBG."""
    bl_idname = "xbg.clear_session_xbg_fc2"
    bl_label = "Clear Session XBG"
    bl_description = "Remove the pinned XBG from the session (inject panel will hide until you select an XBG mesh again)"

    def execute(self, ctx):
        s = ctx.scene.xbg_session_data
        s.is_loaded = False
        s.filepath  = ""
        self.report({'INFO'}, "Session XBG cleared")
        return {'FINISHED'}


class XBG_OT_MergeAllMeshesFC2(bpy.types.Operator):
    bl_idname = "xbg.merge_all_meshes_fc2"
    bl_label = "Merge All Meshes"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, ctx):
        from ..Core.debug import merge_duplicate_vertices
        ds = ctx.scene.xbg_debug_settings
        objs = [o for o in ctx.scene.objects if o.type == 'MESH']
        if not objs:
            self.report({'WARNING'}, "No meshes in scene")
            return {'CANCELLED'}
        merge_duplicate_vertices(objs, ds.merge_distance)
        self.report({'INFO'}, f"Merged vertices on {len(objs)} mesh(es)")
        return {'FINISHED'}


class XBG_OT_MergeSelectedMeshFC2(bpy.types.Operator):
    bl_idname = "xbg.merge_selected_mesh_fc2"
    bl_label = "Merge Selected Mesh"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, ctx):
        from ..Core.debug import merge_duplicate_vertices
        ds = ctx.scene.xbg_debug_settings
        obj = ctx.active_object
        if not obj or obj.type != 'MESH':
            self.report({'ERROR'}, "No mesh selected")
            return {'CANCELLED'}
        merge_duplicate_vertices([obj], ds.merge_distance)
        self.report({'INFO'}, f"Merged vertices on {obj.name}")
        return {'FINISHED'}


class XBG_OT_AutoScaleBoundsFC2(bpy.types.Operator):
    """Automatically calculate and set the minimum pos_scale needed to fit all
    selected meshes within the uint16 coordinate range."""
    bl_idname = "xbg.auto_scale_bounds_fc2"
    bl_label = "Auto Scale Bounds"
    bl_description = (
        "Scan all selected meshes, find the largest vertex coordinate, "
        "and set the pos_scale so everything fits inside the uint16 range. "
        "The new scale is written into the XBG when you click Inject Mesh."
    )

    def execute(self, ctx):
        from ..Core.debug import TraceLogger
        ds  = ctx.scene.xbg_debug_settings
        VerboseLogger.enabled = ds.verbose_logging
        VerboseLogger.clear()
        TraceLogger.set_trace(ds.verbose_logging and ds.trace_logging)

        ins     = ctx.scene.xbg_inject_settings
        session = ctx.scene.xbg_session_data
        obj     = ctx.active_object

        VerboseLogger.session_marker(
            "inject_xbg_bounds_check",
            active_obj=(obj.name if obj else "<none>"),
            session_loaded=bool(session.is_loaded),
            session_pos_scale=getattr(session, 'pos_scale', None))

        # Resolve current pos_scale
        if session.is_loaded and session.pos_scale > 0:
            ps  = session.pos_scale
            imo = session.import_mesh_only
            ps_src = "session"
        elif obj and "xbg_data" in obj:
            meta = obj["xbg_data"].to_dict()
            ps   = float(meta.get("pos_scale", 1.0))
            imo  = bool(meta.get("import_mesh_only", False))
            ps_src = f"active obj '{obj.name}' xbg_data"
        else:
            self.report({'ERROR'}, "No XBG linked — import an XBG file first")
            return {'CANCELLED'}

        mesh_objects = [o for o in ctx.selected_objects if o.type == 'MESH']
        if not mesh_objects and obj and obj.type == 'MESH':
            mesh_objects = [obj]
        if not mesh_objects:
            self.report({'ERROR'}, "No mesh objects selected")
            return {'CANCELLED'}

        TraceLogger.info(
            f"[auto_scale] entry: current_pos_scale={ps:.8f} (from {ps_src}), "
            f"current_half={ps * 32767:.3f}m, import_mesh_only={imo}, "
            f"meshes={len(mesh_objects)}",
            event="autoscale_entry",
            data={"current_pos_scale": float(ps),
                  "current_half_m": float(ps * 32767),
                  "import_mesh_only": bool(imo),
                  "ps_source": ps_src,
                  "n_meshes": len(mesh_objects)})

        # Find the worst (most overflowing) scale_factor across all meshes.
        # scale_factor = 32767 / max_int16_coord, so the smallest value is worst.
        worst_sf   = 1.0
        worst_info = ""
        rows = []
        for o in mesh_objects:
            ns, sf, info = calculate_required_scale(o, ps, imo)
            rows.append((o.name, len(o.data.vertices), bool(ns),
                         f"{sf:.4f}", info))
            if ns and sf < worst_sf:
                worst_sf   = sf
                worst_info = info
        TraceLogger.table(
            "auto-scale per-mesh scan",
            ("object", "verts", "needs_scaling", "scale_factor", "limit_info"),
            rows, tier="DEBUG", event="autoscale_per_mesh")

        if worst_sf >= 1.0:
            TraceLogger.info("[auto_scale] all meshes already fit",
                              event="autoscale_no_change")
            self.report({'INFO'}, "All meshes already fit — no change needed")
            return {'FINISHED'}

        # new_ps = ps / worst_sf gives exact fit; add 2 % headroom
        new_ps   = (ps / worst_sf) * 1.02
        new_half = new_ps * 32767.0

        ins.override_game_scale = True
        ins.target_game_scale   = new_ps

        TraceLogger.kvblock(
            "auto-scale decision",
            [
                ("worst_scale_factor", worst_sf),
                ("worst_info",         worst_info),
                ("old_pos_scale",      f"{ps:.8f}"),
                ("new_pos_scale",      f"{new_ps:.8f}"),
                ("ratio_new_over_old", f"{new_ps / ps:.3f}x"),
                ("old_half_m",         f"{ps * 32767:.3f}"),
                ("new_half_m",         f"{new_half:.3f}"),
                ("override_game_scale", "set to True"),
                ("target_game_scale",   f"set to {new_ps:.8f}"),
            ],
            tier="INFO", event="autoscale_decision")

        # Surface a runtime caveat if the new pos_scale is way outside
        # the engine's character envelope (rough heuristic).  This is the
        # silent failure mode users hit — file is format-valid but the
        # engine has hardcoded character-bound assumptions.
        if new_half > 5.0:
            TraceLogger.info(
                f"[auto_scale] WARNING: new bound (±{new_half:.2f} m) is well "
                f"above typical character size. The file will encode fine but "
                f"the engine may reject / mis-cull / crash on render. Consider "
                f"scaling the geometry down in Blender instead of bumping PMCP.",
                event="autoscale_huge_warning",
                data={"new_half_m": new_half})

        self.report({'INFO'},
            f"Bounds set: pos_scale={new_ps:.8f}  (±{new_half:.3f} units)  [{worst_info}]")
        return {'FINISHED'}


def _run_pre_inject_sanity(mesh_objects):
    """Scene-level sanity pass at inject start.

    Catches the everyday gotchas that turn into in-game crashes or
    silently-broken renders.  Findings emit as SEPARATE `sanity_warning`
    structured events (one per finding) so the JSONL is filterable, plus
    a human-readable `[sanity]` summary in the text log.

    Categories flagged:
      transform/   non-applied scale or rotation on a mesh whose verts
                   we're about to int16-quantise (the unbaked transform
                   ends up baked into encoded positions on inject, so
                   it WILL change the encoded geometry — usually not
                   what the user intended).
      material/    empty material slots, duplicate slot names, materials
                   with use_nodes=False (BSDF detection fails silently),
                   missing xbg_template hint on AUTO-resolved customs.
      modifier/    deformation modifiers still on the stack at inject
                   time (Mirror / Solidify / Subdivision Surface / Array
                   / Bevel — the injector reads the EDITABLE mesh, not
                   the modifier-evaluated one, so the user is shipping
                   the pre-modifier topology).
      skin/        weighted mesh with no Armature modifier/parent (verts
                   will not animate even though weights exist).
      geometry/    zero-vert / zero-face meshes, or counts near the
                   uint16 ceiling that auto-split may not be able to
                   help with.
    """
    from ..Core.debug import TraceLogger as _TL
    import math as _math

    warnings = []

    def warn(category, obj_name, message, **detail):
        rec = {"category": category, "object": obj_name,
               "message": message, **detail}
        warnings.append(rec)
        _TL.struct("sanity_warning", rec, tier="INFO")

    for obj in mesh_objects:
        nm = obj.name
        # ----- transform -----
        scl = tuple(obj.scale)
        if any(abs(s - 1.0) > 1e-4 for s in scl):
            warn("transform/scale_not_applied", nm,
                 f"object scale={scl} != (1,1,1); apply with Ctrl+A "
                 f"or it'll be baked into the encoded int16 positions",
                 scale=scl)
        rz_deg = tuple(round(_math.degrees(a), 4) for a in obj.rotation_euler)
        # Importer applies a 180° Z to characters; that's expected.  Flag
        # rotations that are neither 0 nor 180 around Z (i.e. authored
        # rotations the user forgot to apply).
        if (abs(rz_deg[0]) > 1e-2 or abs(rz_deg[1]) > 1e-2
                or (abs(rz_deg[2]) > 1e-2 and abs(rz_deg[2] - 180) > 1e-2
                    and abs(rz_deg[2] + 180) > 1e-2)):
            warn("transform/rotation_not_applied", nm,
                 f"object rotation_euler={rz_deg}° will be baked into "
                 f"encoded positions; apply transforms before inject",
                 rotation_deg=rz_deg)

        # ----- materials -----
        names_seen = {}
        for sl_i, sl in enumerate(obj.material_slots):
            m = sl.material
            if m is None:
                warn("material/empty_slot", nm,
                     f"material slot {sl_i} is empty; any face with "
                     f"material_index={sl_i} will fall through to slot 0",
                     slot=sl_i)
                continue
            if m.name in names_seen:
                warn("material/duplicate_slot_name", nm,
                     f"material '{m.name}' appears in slots "
                     f"{names_seen[m.name]} AND {sl_i}",
                     slot=sl_i, other_slot=names_seen[m.name],
                     material=m.name)
            else:
                names_seen[m.name] = sl_i
            if not m.use_nodes:
                warn("material/no_nodes", nm,
                     f"material '{m.name}' has use_nodes=False; export "
                     f"auto-detect can't see its BSDF — template will "
                     f"default to Unlit",
                     slot=sl_i, material=m.name)

        # ----- modifiers -----
        deform_kinds = {"MIRROR", "SOLIDIFY", "SUBSURF", "ARRAY", "BEVEL",
                        "MULTIRES", "WAVE", "SHRINKWRAP", "CAST",
                        "SIMPLE_DEFORM"}
        for mod in obj.modifiers:
            if mod.type in deform_kinds:
                warn("modifier/deform_modifier_at_inject", nm,
                     f"modifier '{mod.name}' ({mod.type}) is on the stack; "
                     f"injector reads the editable mesh, NOT the modifier "
                     f"output — apply this modifier or the modifier's "
                     f"effect won't appear in the .xbg",
                     modifier=mod.name, type=mod.type)

        # ----- skinning -----
        has_weights = bool(obj.vertex_groups)
        arm_parent = (obj.parent and obj.parent.type == 'ARMATURE')
        arm_modifier = any(m.type == 'ARMATURE' and getattr(m, 'object', None)
                           for m in obj.modifiers)
        if has_weights and not (arm_parent or arm_modifier):
            warn("skin/weights_without_armature", nm,
                 f"object has {len(obj.vertex_groups)} vertex group(s) "
                 f"but no Armature parent or modifier — weights won't "
                 f"deform the mesh in Blender",
                 vertex_groups=len(obj.vertex_groups))

        # ----- geometry counts -----
        v_cnt = len(obj.data.vertices) if obj.data else 0
        f_cnt = len(obj.data.polygons) if obj.data else 0
        if v_cnt == 0:
            warn("geometry/empty_verts", nm,
                 "object has 0 vertices; will inject as a null submesh")
        if f_cnt == 0:
            warn("geometry/empty_faces", nm,
                 "object has 0 faces; will inject as a null submesh")
        # 65534 is the auto-split budget cap (one less than the format max).
        if v_cnt > 60000:
            warn("geometry/near_uint16_verts", nm,
                 f"object has {v_cnt} verts — within 10% of the 65535 "
                 f"uint16 cap. Auto-split helps for faces, but a single "
                 f"submesh's vert count is bounded too.",
                 vert_count=v_cnt)

    # Aggregate by category for the text-log summary.
    counts = {}
    for w in warnings:
        counts[w["category"]] = counts.get(w["category"], 0) + 1
    if warnings:
        VerboseLogger.log(
            f"[sanity] pre-inject pass found {len(warnings)} warning(s):  "
            + ", ".join(f"{cat}={n}" for cat, n in sorted(counts.items())))
        # Group by category in the text log so it's scannable.
        by_cat = {}
        for w in warnings:
            by_cat.setdefault(w["category"], []).append(w)
        for cat in sorted(by_cat):
            VerboseLogger.log(f"  [sanity] {cat}:")
            for w in by_cat[cat]:
                VerboseLogger.log(f"      '{w['object']}': {w['message']}")
    else:
        VerboseLogger.log("[sanity] pre-inject pass: no warnings")

    _TL.struct("sanity_summary", {
        "phase":         "pre_inject_scene",
        "total":         len(warnings),
        "by_category":   counts,
    })
    return warnings


def _log_pre_import_scene_snapshot(ctx):
    """List what's in the scene BEFORE import runs.

    Lets us tell, after the fact, whether the user is importing into an
    empty scene or merging into an existing one — and whether they had
    leftover xbg-tagged objects from a previous session that might
    interfere with later inject."""
    from ..Core.debug import TraceLogger as _TL
    rows = []
    n_xbg = 0
    for obj in (ctx.scene.objects if ctx and ctx.scene else []):
        if obj.type != 'MESH':
            continue
        v = len(obj.data.vertices) if obj.data else 0
        f = len(obj.data.polygons) if obj.data else 0
        xd = obj.get("xbg_data")
        has_xbg = xd is not None
        if has_xbg:
            n_xbg += 1
        rows.append((obj.name, v, f, len(obj.material_slots),
                     "tagged" if has_xbg else "untagged"))
    _TL.table(
        f"PRE-IMPORT SCENE SNAPSHOT  ({len(rows)} mesh object(s); "
        f"{n_xbg} already tagged from prior xbg imports)",
        ("object", "verts", "faces", "n_mat_slots", "tag"),
        rows, tier="INFO", event="pre_import_scene_snapshot",
        max_rows=50)


def _log_pre_inject_scene_diff(mesh_objects):
    """Dump a comprehensive "what did the user actually change" snapshot
    before injection runs.

    For each selected mesh, compares the state stored on `obj['xbg_data']`
    at import time (vert_count, slot, lod, pos_scale, …) against the
    object's CURRENT state in Blender.  The output table makes it
    trivial to see things like:
      - object added with no xbg_data tag (user spawned / joined)
      - vert_count drifted +N (mesh edited / extruded)
      - extra material slots vs imported (custom materials added)
      - non-identity transform applied
      - new modifiers in the stack
      - rotation/scale baked vs deferred

    Without this we'd have to guess from the verbose log what the user
    did between import and inject — which is exactly what made debugging
    the sovereigna crash slow.
    """
    from ..Core.debug import TraceLogger as _TL
    import math as _math

    n_added = n_modified = n_clean = 0
    rows = []
    for obj in mesh_objects:
        xd = obj.get("xbg_data")
        d = xd.to_dict() if (xd is not None and hasattr(xd, 'to_dict')) else (xd or {})
        imported_v = d.get("vert_count") if isinstance(d, dict) else None
        slot = d.get("sdol_submesh_slot") if isinstance(d, dict) else None
        lod = d.get("lod_level") if isinstance(d, dict) else None
        cur_v = len(obj.data.vertices) if obj.data else 0
        cur_f = len(obj.data.polygons) if obj.data else 0

        # Classify the diff at a glance.
        if imported_v is None:
            tag = "NEW"             # no xbg_data tag — user spawned this
            n_added += 1
            v_delta = "+{}".format(cur_v)
        elif cur_v != imported_v:
            tag = "MODIFIED"
            n_modified += 1
            v_delta = f"{int(imported_v)} -> {cur_v}  ({cur_v - int(imported_v):+d})"
        else:
            tag = "clean"
            n_clean += 1
            v_delta = f"={cur_v}"

        # Material slot inventory.
        slot_summary = []
        for sl_i, sl in enumerate(obj.material_slots):
            m = sl.material
            if m is None:
                slot_summary.append(f"{sl_i}:<empty>")
                continue
            origin = ('game' if m.get('xbg_source')
                      else ('exported' if m.get('xbg_exported')
                            else 'new'))
            slot_summary.append(f"{sl_i}:{origin}:{m.name}")

        # Non-identity transform check.
        loc = tuple(round(v, 6) for v in obj.location)
        rot_deg = tuple(round(_math.degrees(a), 3) for a in obj.rotation_euler)
        scl = tuple(round(v, 6) for v in obj.scale)
        transform_summary = []
        if any(abs(v) > 1e-5 for v in loc):
            transform_summary.append(f"loc={loc}")
        if any(abs(a) > 1e-3 for a in rot_deg):
            transform_summary.append(f"rot°={rot_deg}")
        if any(abs(s - 1.0) > 1e-5 for s in scl):
            transform_summary.append(f"scale={scl}")

        modifiers = [f"{m.type}:{m.name}" for m in obj.modifiers]

        # Armature parent — joined / parented to the imported skeleton?
        arm_parent = None
        if obj.parent and obj.parent.type == 'ARMATURE':
            arm_parent = obj.parent.name
        else:
            for m in obj.modifiers:
                if m.type == 'ARMATURE' and getattr(m, 'object', None):
                    arm_parent = m.object.name
                    break

        rows.append((
            tag,
            obj.name,
            v_delta,
            cur_f,
            slot,
            lod,
            len(obj.material_slots),
            " ".join(transform_summary) or "-",
            ",".join(modifiers) or "-",
            arm_parent or "-",
        ))

        # Emit per-object structured event so the JSONL is filterable.
        _TL.struct("user_edit_snapshot", {
            "obj":               obj.name,
            "tag":               tag,
            "imported_verts":    imported_v,
            "current_verts":     cur_v,
            "current_faces":     cur_f,
            "sdol_submesh_slot": slot,
            "lod_level":         lod,
            "material_slots":    slot_summary,
            "transform":         transform_summary,
            "modifiers":         modifiers,
            "armature":          arm_parent,
        })

    _TL.table(
        f"USER EDIT SNAPSHOT  (NEW={n_added}, MODIFIED={n_modified}, "
        f"clean={n_clean})  — what changed between import and this inject",
        ("tag", "object", "verts(imported->current)", "faces",
         "slot", "lod", "n_mat_slots", "transform", "modifiers", "armature"),
        rows, tier="INFO", event="user_edit_snapshot_table")

    _TL.kvblock(
        "USER EDIT SNAPSHOT summary",
        [
            ("objects_selected",   len(mesh_objects)),
            ("new_objects",        n_added),
            ("modified_objects",   n_modified),
            ("clean_objects",      n_clean),
            ("total_current_verts", sum(len(o.data.vertices) for o in mesh_objects if o.data)),
            ("total_current_faces", sum(len(o.data.polygons) for o in mesh_objects if o.data)),
        ],
        tier="INFO", event="user_edit_snapshot_summary")


class XBG_OT_InjectMeshFC2(bpy.types.Operator):
    bl_idname = "xbg.inject_mesh_fc2"
    bl_label = "Inject New Topology"
    bl_description = (
        "Inject selected Blender mesh(es) into the XBG file. "
        "Each selected object becomes a separate submesh. "
        "Vertex and face counts can change freely."
    )
    bl_options = {'REGISTER', 'UNDO'}

    filepath: bpy.props.StringProperty(subtype="FILE_PATH")

    def invoke(self, ctx, ev):
        session = ctx.scene.xbg_session_data
        if session.is_loaded and session.filepath:
            self.filepath = session.filepath
        else:
            obj = ctx.active_object
            if obj and "xbg_data" in obj:
                self.filepath = obj["xbg_data"]["filepath"]
        ctx.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}

    def execute(self, ctx):
        ds  = ctx.scene.xbg_debug_settings
        ins = ctx.scene.xbg_inject_settings
        session = ctx.scene.xbg_session_data
        VerboseLogger.enabled = ds.verbose_logging
        VerboseLogger.clear()
        # TraceLogger is gated by BOTH the verbose flag AND the trace flag.
        # When verbose is off, TraceLogger.trace_enabled() returns False.
        from ..Core.debug import TraceLogger
        TraceLogger.set_trace(ds.verbose_logging and ds.trace_logging)

        mesh_objects = [o for o in ctx.selected_objects if o.type == 'MESH']
        if not mesh_objects:
            obj = ctx.active_object
            if obj and obj.type == 'MESH':
                mesh_objects = [obj]
        if not mesh_objects:
            self.report({'ERROR'}, "No mesh objects selected")
            return {'CANCELLED'}

        VerboseLogger.session_marker(
            "inject_xbg",
            output_file=self.filepath,
            target_lod=ins.target_lod,
            n_selected=len(mesh_objects),
            selected=[o.name for o in mesh_objects],
            split_by_material=ins.inject_materials,
            inject_bone_weights=ins.inject_bone_weights,
            inject_vertex_colors=ins.inject_vertex_colors,
            override_game_scale=ins.override_game_scale,
            target_game_scale=ins.target_game_scale)
        _log_pre_inject_scene_diff(mesh_objects)
        _run_pre_inject_sanity(mesh_objects)

        meta = None
        if session.is_loaded and session.filepath:
            meta = {
                'filepath':         session.filepath,
                'pos_scale':        session.pos_scale,
                'uv_trans':         session.uv_trans,
                'uv_scale':         session.uv_scale,
                'import_mesh_only': session.import_mesh_only,
                'pmcp_offset':      session.pmcp_offset,
            }
        else:
            for o in mesh_objects:
                if "xbg_data" in o:
                    meta = o["xbg_data"].to_dict()
                    break

        if meta is None:
            self.report({'ERROR'},
                "No XBG linked. Select an imported XBG mesh, or click "
                "'Pin This File' while it is selected.")
            return {'CANCELLED'}

        injector = XBGMeshInjector()
        st, msg = injector.inject(
            ctx,
            mesh_objects,
            self.filepath,
            target_lod               = ins.target_lod,
            meta                     = meta,
            override_game_scale      = ins.override_game_scale,
            target_game_scale        = ins.target_game_scale,
            ignore_limits            = ins.ignore_format_limits,
            inject_vertex_colors     = ins.inject_vertex_colors,
            generate_neutral_vertex_colors = ins.generate_neutral_vertex_colors,
            inject_bone_weights      = ins.inject_bone_weights,
            inject_materials         = ins.inject_materials,
            force_per_submesh_vb     = ins.force_per_submesh_vb,
        )
        # Normal/tangent handling is no longer configurable: stock verts keep
        # their authored normals + tangents (byte-exact); new geometry uses its
        # viewport normals and UV-computed tangents. See inject_xbg._encode_vertices.

        if st == {'FINISHED'}:
            VerboseLogger.session_complete(
                "inject_xbg",
                output_file=self.filepath,
                target_lod=ins.target_lod,
                n_objects=len(mesh_objects),
                split_by_material=ins.inject_materials,
                msg=msg)
            VerboseLogger.autosave_sidecar(self.filepath)
            self.report({'INFO'}, msg)
        else:
            # Still autosave on cancel — the failure log is the most
            # valuable artefact a bug reporter can attach.
            VerboseLogger.session_complete(
                "inject_xbg",
                output_file=self.filepath,
                status="CANCELLED", reason=msg)
            VerboseLogger.autosave_sidecar(self.filepath)
            self.report({'ERROR'}, msg)
        return st


class XBG_OT_PeekLODsFC2(bpy.types.Operator):
    """Quickly scan a .xbg file to count its LODs without a full import."""
    bl_idname = "xbg.peek_lods_fc2"
    bl_label = "Peek LOD Count"
    bl_description = "Scan a .xbg file to show how many LODs it contains before importing"

    filepath: bpy.props.StringProperty(subtype="FILE_PATH")

    def invoke(self, ctx, ev):
        ctx.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}

    def execute(self, ctx):
        if not self.filepath or not os.path.exists(self.filepath):
            self.report({'ERROR'}, "No valid .xbg file selected")
            return {'CANCELLED'}
        if not self.filepath.lower().endswith('.xbg'):
            self.report({'ERROR'}, "Selected file is not an .xbg file")
            return {'CANCELLED'}
        try:
            lod_count = XBG_OT_PeekLODs._peek_lod_count(self.filepath)
            fn = os.path.basename(self.filepath)
            if lod_count > 0:
                result = f"{fn}: {lod_count} LOD(s)  (LOD 0 – {lod_count - 1})"
            else:
                result = f"{fn}: LOD count could not be read"
            ctx.scene.xbg_debug_settings.lod_peek_result = result
            self.report({'INFO'}, result)
        except Exception as e:
            ctx.scene.xbg_debug_settings.lod_peek_result = f"Error: {e}"
            self.report({'WARNING'}, f"Could not read file: {e}")
        return {'FINISHED'}

    @staticmethod
    def _peek_lod_count(filepath):
        fsize = os.path.getsize(filepath)
        with open(filepath, 'rb') as f:
            data = f.read(min(fsize, 4096))

        if len(data) < 32:
            return 0

        # Endian-aware: PC files are LE, PS3 files are BE.  Detect from the
        # 32-byte header, then look for the appropriate SDOL/LODS byte
        # sequence and decode all multi-byte fields in the same endian.
        en = detect_endian_from_bytes(data[:32])
        sdol_bytes = encode_chunk_magic('SDOL', en)  # b'SDOL' (LE) / b'LODS' (BE)
        cc = struct.unpack_from(f'{en}i', data, 28)[0]
        offset = 32

        for _ in range(min(cc, 64)):
            if offset + 12 > len(data):
                with open(filepath, 'rb') as f:
                    f.seek(offset)
                    hdr = f.read(12)
                if len(hdr) < 12:
                    break
                chunk_raw  = hdr[:4]
                chunk_size = struct.unpack_from(f'{en}i', hdr, 8)[0]
                if chunk_raw == sdol_bytes:
                    lod_off = offset + 20
                    with open(filepath, 'rb') as f:
                        f.seek(lod_off)
                        lc = f.read(4)
                    return max(0, struct.unpack_from(f'{en}i', lc, 0)[0]) if len(lc) == 4 else 0
                if chunk_size <= 0:
                    break
                offset += chunk_size
                continue

            chunk_raw  = data[offset:offset + 4]
            chunk_size = struct.unpack_from(f'{en}i', data, offset + 8)[0]

            if chunk_raw == sdol_bytes:
                lod_off = offset + 20
                if lod_off + 4 <= len(data):
                    return max(0, struct.unpack_from(f'{en}i', data, lod_off)[0])
                with open(filepath, 'rb') as f:
                    f.seek(lod_off)
                    lc = f.read(4)
                return max(0, struct.unpack_from(f'{en}i', lc, 0)[0]) if len(lc) == 4 else 0

            if chunk_size <= 0:
                break
            offset += chunk_size

        return 0


# ---------------------------------------------------------------------------
# Operators — updater
# ---------------------------------------------------------------------------

class XBG_OT_ExpandBoundsForInjectFC2(bpy.types.Operator):
    """Create a temp XBG copy with a larger pos_scale (bigger format bounds),
    then update the session so Inject Mesh encodes with the new scale.
    Your mesh keeps its exact vertex positions — no manual copy/paste needed."""
    bl_idname = "xbg.expand_bounds_for_inject_fc2"
    bl_label = "Expand Bounds for Inject"
    bl_description = (
        "Create a temp XBG copy with bigger format bounds (larger pos_scale). "
        "Your mesh stays in place — just click Inject Mesh after this."
    )

    new_half_size: bpy.props.FloatProperty(
        name="New Half-Size (units)",
        description="New coordinate half-size: ±N Blender units maps to the full int16 range. "
                    "Bigger = larger models fit, but less precision.",
        default=144.0,
        min=0.001,
        precision=4
    )

    def invoke(self, ctx, ev):
        session = ctx.scene.xbg_session_data
        obj = ctx.active_object
        if session.is_loaded and session.pos_scale > 0:
            ps = session.pos_scale
        elif obj and "xbg_data" in obj:
            ps = float(obj["xbg_data"].get("pos_scale", 1.0))
        else:
            ps = 1.0
        self.new_half_size = ps * 32767.0 * 2
        return ctx.window_manager.invoke_props_dialog(self, width=380)

    def draw(self, ctx):
        l = self.layout
        session = ctx.scene.xbg_session_data
        obj = ctx.active_object

        if session.is_loaded and session.pos_scale > 0:
            ps = session.pos_scale
            src = os.path.basename(session.filepath)
        elif obj and "xbg_data" in obj:
            ps = float(obj["xbg_data"].get("pos_scale", 1.0))
            src = os.path.basename(str(obj["xbg_data"].get("filepath", "")))
        else:
            ps = 1.0
            src = "(none)"

        current_half = ps * 32767.0
        new_ps = self.new_half_size / 32767.0
        factor = self.new_half_size / current_half if current_half > 0 else 1.0

        b = l.box()
        b.label(text=f"Source: {src}", icon='FILE')
        b.label(text=f"Current half-size : \u00b1{current_half:.3f} units  (pos_scale={ps:.8f})")
        l.separator()
        l.prop(self, "new_half_size")
        b2 = l.box()
        b2.label(text=f"New pos_scale : {new_ps:.8f}")
        b2.label(text=f"Range multiplier : \u00d7{factor:.3f}  "
                       f"({'larger' if factor > 1 else 'smaller'} bounds)")
        l.separator()
        i = l.box()
        i.label(text="What this does:", icon='INFO')
        i.label(text="  1. Enables 'Override Internal Scale' with the new value")
        i.label(text="  2. Scale is written into the XBG on Inject")
        i.label(text="  3. No extra file copy is created")
        i.label(text="  4. Your mesh stays exactly where it is in 3D space")
        i.label(text="Original XBG is only modified when you click Inject Mesh.")

    def execute(self, ctx):
        ins     = ctx.scene.xbg_inject_settings
        session = ctx.scene.xbg_session_data
        obj     = ctx.active_object

        # Resolve current pos_scale for reporting
        if session.is_loaded and session.pos_scale > 0:
            old_ps = session.pos_scale
        elif obj and "xbg_data" in obj:
            old_ps = float(obj["xbg_data"].get("pos_scale", 1.0))
        else:
            old_ps = 1.0

        new_ps = self.new_half_size / 32767.0
        hs     = self.new_half_size

        # No file copy needed: enable Override Scale with the new value.
        # chunks.patch_pmcp will be called during Inject Mesh to write it to the output.
        ins.override_game_scale = True
        ins.target_game_scale   = new_ps

        # Update the format-bounds lattice visualiser if it is visible
        ds = ctx.scene.xbg_debug_settings
        ds["format_bounds_x"] = hs
        ds["format_bounds_y"] = hs
        ds["format_bounds_z"] = hs
        lo = bpy.data.objects.get("XBG_Format_Bounds")
        if lo:
            lo.scale = (hs, hs, hs)

        factor = hs / (old_ps * 32767.0) if old_ps > 0 else 1.0
        self.report({'INFO'},
            f"Override Scale set x{factor:.3f}: "
            f"pos_scale={new_ps:.8f}  (+/-{hs:.3f} units)  "
            f"| Override Internal Scale is ON  -  click Inject Mesh.")
        return {'FINISHED'}


class XBG_OT_SaveFormatBoundsSizeFC2(bpy.types.Operator):
    bl_idname = "xbg.save_format_bounds_size_fc2"
    bl_label = "Apply Box Size to Session"
    bl_description = (
        "Set the bounds box size as the new pos_scale for this session. "
        "The scale is written into the XBG when you click Inject Mesh — "
        "the file on disk is not touched until then."
    )

    def execute(self, ctx):
        lo = bpy.data.objects.get("XBG_Format_Bounds")
        if not lo:
            self.report({'WARNING'}, "Format Bounds lattice not visible — enable it first")
            return {'CANCELLED'}

        ins = ctx.scene.xbg_inject_settings
        ds  = ctx.scene.xbg_debug_settings
        half          = ds.format_bounds_x
        new_pos_scale = half / 32767.0

        ins.override_game_scale = True
        ins.target_game_scale   = new_pos_scale

        lo.scale = (ds.format_bounds_x, ds.format_bounds_y, ds.format_bounds_z)
        self.report({'INFO'},
            f"Session bounds set: pos_scale={new_pos_scale:.8f}  (±{half:.4f} units)"
            f"  — click Inject Mesh to apply")
        return {'FINISHED'}


# ---------------------------------------------------------------------------
# Operators — LKS skeleton import
# ---------------------------------------------------------------------------

class XBG_OT_ImportLKSSkeletonFC2(bpy.types.Operator):
    """Import a Ubisoft LKS binary skeleton file as a Blender armature."""
    bl_idname  = "xbg.import_lks_skeleton_fc2"
    bl_label   = "Import LKS Skeleton"
    bl_description = (
        "Import a .skeleton (LKS) binary file and create a Blender armature "
        "with the full bone hierarchy, positions, and rotations."
    )
    bl_options = {'REGISTER', 'UNDO'}

    filepath: bpy.props.StringProperty(subtype="FILE_PATH")
    filter_glob: bpy.props.StringProperty(default="*.skeleton", options={'HIDDEN'})

    def invoke(self, ctx, ev):
        ctx.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}

    def execute(self, ctx):
        ds = ctx.scene.xbg_debug_settings
        VerboseLogger.enabled = ds.verbose_logging
        VerboseLogger.clear()
        VerboseLogger.session_marker("import_lks", file=self.filepath)

        if not self.filepath or not os.path.isfile(self.filepath):
            self.report({'ERROR'}, "No valid .skeleton file selected")
            return {'CANCELLED'}

        try:
            bones = parse_lks_file(self.filepath)
            name  = os.path.splitext(os.path.basename(self.filepath))[0]
            arm   = create_lks_armature(ctx, bones, armature_name=name)
            self.report({'INFO'},
                f"Imported LKS skeleton: {name}  ({len(bones)} bones)")
            return {'FINISHED'}
        except Exception as exc:
            self.report({'ERROR'}, f"Failed to import skeleton: {exc}")
            import traceback; traceback.print_exc()
            return {'CANCELLED'}


def _xbg_path_for_armature(arm, session):
    """Return the source XBG file path for *arm*, or None.

    Lookup order:
      1. xbg_source_file property stored on the armature at import time.
      2. xbg_data.filepath on any direct child mesh object.
      3. The scene session (pinned via 'Remember This XBG').
    """
    p = arm.get("xbg_source_file", "")
    if p and os.path.isfile(p):
        return p
    for child in arm.children:
        xd = child.get("xbg_data")
        if xd is None:
            continue
        d = xd.to_dict() if hasattr(xd, 'to_dict') else (xd or {})
        fp = d.get("filepath") if isinstance(d, dict) else None
        if fp and os.path.isfile(fp):
            return fp
    if session.is_loaded and session.filepath and os.path.isfile(session.filepath):
        return session.filepath
    return None


class XBG_OT_ImportMABFC2(bpy.types.Operator):
    """Import a .mab animation onto the selected armature."""
    bl_idname  = "xbg.import_mab_animation_fc2"
    bl_label   = "Import MAB Animation"
    bl_description = (
        "Select an armature first, then import a .mab animation file. "
        "Keyframes are applied to the rig's pose bones and a verbose "
        "format report is printed to the system console / log."
    )
    bl_options = {'REGISTER', 'UNDO'}

    filepath: bpy.props.StringProperty(subtype="FILE_PATH")
    filter_glob: bpy.props.StringProperty(default="*.mab", options={'HIDDEN'})

    def invoke(self, ctx, ev):
        ctx.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}

    def execute(self, ctx):
        ds = ctx.scene.xbg_debug_settings
        VerboseLogger.enabled = True  # always verbose for the decode loop

        arm = ctx.active_object
        if arm is None or arm.type != 'ARMATURE':
            arm = next((o for o in ctx.selected_objects if o.type == 'ARMATURE'), None)
        if arm is None:
            self.report({'ERROR'}, "Select an armature before importing a .mab")
            return {'CANCELLED'}
        if not self.filepath or not os.path.isfile(self.filepath):
            self.report({'ERROR'}, "No valid .mab file selected")
            return {'CANCELLED'}

        xbg_path = _xbg_path_for_armature(arm, ctx.scene.xbg_session_data)
        if not xbg_path:
            self.report({'ERROR'},
                "Cannot find source XBG — select the imported armature or use 'Remember This XBG'")
            return {'CANCELLED'}

        # Optional explicit .skeleton override (Debug panel); else also search
        # the .mab's own folder and a few parents besides the XBG's folder.
        skel_override = bpy.path.abspath(ds.mab_skeleton_path).strip() \
            if ds.mab_skeleton_path else ''
        if skel_override and not os.path.isfile(skel_override):
            self.report({'ERROR'},
                f"Animation Skeleton path does not exist: {skel_override}")
            return {'CANCELLED'}
        mab_dir = os.path.dirname(os.path.abspath(self.filepath))
        extra = [mab_dir]
        for _ in range(3):                      # walk up the animations tree
            mab_dir = os.path.dirname(mab_dir)
            extra.append(mab_dir)

        try:
            d = open(self.filepath, 'rb').read()
            sec = _mab_parse_sections(d)
            n_keyed, animated = _mab_apply(
                ctx, d, sec, arm, xbg_path=xbg_path,
                skeleton_path=skel_override or None, extra_dirs=extra,
                bone_offset=ds.mab_char_offset,
                emulate_helpers=ds.mab_emulate_helpers,
                smooth_resample=ds.mab_smooth_resample,
                resample_fps=ds.mab_resample_fps,
                twist_bake=ds.mab_twist_bake)
            self.report({'INFO'},
                f"MAB: {os.path.basename(self.filepath)}  "
                f"{n_keyed}/{len(animated)} bones / "
                f"{ctx.scene.frame_end} frames")
            return {'FINISHED'}
        except Exception as exc:
            self.report({'ERROR'}, f"Failed to import .mab: {exc}")
            import traceback; traceback.print_exc()
            return {'CANCELLED'}


class XBG_OT_PreviewJiggleFC2(bpy.types.Operator):
    """Bake a damped-spring jiggle so satchels / pouches / skirts / custom
    bust+rear bones swing with the current animation — preview before injecting.

    • A PRESET (Bust / Rear / Skirt) jiggles the bone(s) you've SELECTED in Pose
      mode — use this for the custom bones you add.
    • 'From proceduralbones.xml' reads the game's real params and jiggles any of
      those bones that are present in the rig (auto-finds the file under your
      Data Folder)."""
    bl_idname  = "xbg.preview_jiggle_fc2"
    bl_label   = "Preview Jiggle"
    bl_options = {'REGISTER', 'UNDO'}

    preset: bpy.props.EnumProperty(
        name="Source",
        items=[('BUST', "Bust preset (selected bones)", "Jiggle selected bones, bust feel"),
               ('REAR', "Rear preset (selected bones)", "Jiggle selected bones, rear feel"),
               ('SKIRT', "Skirt preset (selected bones)", "Jiggle selected bones, skirt feel"),
               ('FROM_XML', "From proceduralbones.xml", "Use the game's params for bones present in the rig")],
        default='BUST')
    pawn_type: bpy.props.StringProperty(name="Pawn Type", default="corp")
    strength: bpy.props.FloatProperty(
        name="Strength", default=3.0, min=0.0, max=20.0,
        description="Amplifies the spring lag (swing amount). 1=subtle, "
                    "3-6=natural, higher saturates smoothly at the bend limit")
    filepath: bpy.props.StringProperty(
        name="proceduralbones.xml", subtype="FILE_PATH", default="",
        description="Leave blank to auto-find under the Data Folder")

    def invoke(self, ctx, ev):
        return ctx.window_manager.invoke_props_dialog(self, width=380)

    def draw(self, ctx):
        col = self.layout.column()
        col.prop(self, "preset")
        col.prop(self, "strength")
        if self.preset == 'FROM_XML':
            col.prop(self, "pawn_type")
            col.prop(self, "filepath")
            col.label(text="Blank path = auto-find under Data Folder", icon='INFO')
        else:
            col.label(text="Jiggles the bone(s) SELECTED in Pose mode", icon='BONE_DATA')

    def execute(self, ctx):
        arm = ctx.active_object
        if arm is None or arm.type != 'ARMATURE':
            arm = next((o for o in ctx.selected_objects if o.type == 'ARMATURE'), None)
        if arm is None:
            self.report({'ERROR'}, "Select the armature first")
            return {'CANCELLED'}
        try:
            from .jiggle_fc2 import load_jiggle_defs, bake_jiggle, PRESETS, _vec3

            if self.preset == 'FROM_XML':
                path = bpy.path.abspath(self.filepath).strip() if self.filepath else ''
                if not (path and os.path.isfile(path)):
                    from ..Core.prefs import get_prefs
                    df = (get_prefs(ctx).data_folder or '').strip()
                    cand = os.path.join(bpy.path.abspath(df), 'databases',
                                        'baltazar', 'proceduralbones.xml') if df else ''
                    if cand and os.path.isfile(cand):
                        path = cand
                    else:
                        self.report({'ERROR'},
                            "proceduralbones.xml not found — set your Data Folder in "
                            "addon prefs, or pick the file in the dialog")
                        return {'CANCELLED'}
                defs = load_jiggle_defs(path, self.pawn_type)
                present = [n for n in defs if n in arm.pose.bones]
                if not present:
                    self.report({'ERROR'},
                        "None of the %s procedural bones (%s) are in this rig — "
                        "import the mesh that carries them, or use a preset on "
                        "selected bones" % (self.pawn_type, ", ".join(defs) or "none"))
                    return {'CANCELLED'}
            else:
                sel = [pb.name for pb in (ctx.selected_pose_bones or [])]
                if not sel:
                    self.report({'ERROR'},
                        "Select the bone(s) to jiggle in POSE mode, then run with a preset")
                    return {'CANCELLED'}
                p = PRESETS[self.preset]
                one = {'min_rot': _vec3(p['MinRotation']), 'max_rot': _vec3(p['MaxRotation']),
                       'axis': _vec3(p['DisplacementAxisEffect']),
                       'invert': _vec3(p['InvertDisplacementEffect']),
                       'mult': _vec3(p['MovementMultiplier']),
                       'tension': float(p['Tension']), 'friction': float(p['Friction'])}
                defs = {nm: one for nm in sel}

            n = bake_jiggle(ctx, arm, defs, ctx.scene.frame_start,
                            ctx.scene.frame_end, strength=self.strength, log=print)
            done = [nm for nm in defs if nm in arm.pose.bones]
            self.report({'INFO'}, "Jiggle baked on %d bone(s): %s"
                        % (n, ", ".join(done) or "none"))
            return {'FINISHED'}
        except Exception as exc:
            self.report({'ERROR'}, f"Jiggle preview failed: {exc}")
            import traceback; traceback.print_exc()
            return {'CANCELLED'}


class XBG_OT_ImportLFAFC2(bpy.types.Operator):
    """Import a .lfa facial pose library onto the selected armature."""
    bl_idname  = "xbg.import_lfa_poses_fc2"
    bl_label   = "Import LFA Facial Poses"
    bl_description = (
        "Select the head armature first, then pick a .lfa file. Each facial "
        "pose (jawOpen, lSneer, visemes, P_* presets...) is keyed on its own "
        "frame with a timeline marker carrying the pose name"
    )
    bl_options = {'REGISTER', 'UNDO'}

    filepath: bpy.props.StringProperty(subtype="FILE_PATH")
    filter_glob: bpy.props.StringProperty(default="*.lfa", options={'HIDDEN'})

    def invoke(self, ctx, ev):
        ctx.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}

    def execute(self, ctx):
        from .import_lfa_fc2 import parse_lfa, apply_lfa_poses
        arm = ctx.active_object
        if arm is None or arm.type != 'ARMATURE':
            arm = next((o for o in ctx.selected_objects
                        if o.type == 'ARMATURE'), None)
        if arm is None:
            self.report({'ERROR'}, "Select an armature before importing a .lfa")
            return {'CANCELLED'}
        xbg_path = _xbg_path_for_armature(arm, ctx.scene.xbg_session_data)
        if not xbg_path:
            self.report({'ERROR'},
                "Cannot find source XBG — select the imported armature or "
                "use 'Remember This XBG'")
            return {'CANCELLED'}
        try:
            lfa = parse_lfa(self.filepath)
            n_poses, n_bones = apply_lfa_poses(ctx, lfa, arm, xbg_path)
            self.report({'INFO'},
                f"LFA: {n_poses} facial poses on {n_bones} bones — "
                "one pose per frame (see timeline markers)")
            return {'FINISHED'}
        except Exception as exc:
            self.report({'ERROR'}, f"Failed to import .lfa: {exc}")
            import traceback; traceback.print_exc()
            return {'CANCELLED'}


class XBG_OT_ImportLFEFC2(bpy.types.Operator):
    """Import a .lfe facial expression clip (needs the head's .lfa first)."""
    bl_idname  = "xbg.import_lfe_expression_fc2"
    bl_label   = "Import LFE Expression"
    bl_description = (
        "Select the head armature, set the head's .lfa path in the field "
        "above, then pick a .lfe expression/emotion file to animate the "
        "facial pose channels over time"
    )
    bl_options = {'REGISTER', 'UNDO'}

    filepath: bpy.props.StringProperty(subtype="FILE_PATH")
    filter_glob: bpy.props.StringProperty(default="*.lfe", options={'HIDDEN'})

    def invoke(self, ctx, ev):
        ctx.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}

    def execute(self, ctx):
        from .import_lfa_fc2 import parse_lfa, parse_lfe, apply_lfe_expression
        ds = ctx.scene.xbg_debug_settings
        arm = ctx.active_object
        if arm is None or arm.type != 'ARMATURE':
            arm = next((o for o in ctx.selected_objects
                        if o.type == 'ARMATURE'), None)
        if arm is None:
            self.report({'ERROR'}, "Select an armature before importing a .lfe")
            return {'CANCELLED'}
        lfa_path = bpy.path.abspath(ds.lfa_path).strip() if ds.lfa_path else ''
        if not lfa_path or not os.path.isfile(lfa_path):
            self.report({'ERROR'},
                "Set the head's .lfa file first (field above the button) — "
                "the .lfe only stores curve values for the .lfa's pose names")
            return {'CANCELLED'}
        xbg_path = _xbg_path_for_armature(arm, ctx.scene.xbg_session_data)
        if not xbg_path:
            self.report({'ERROR'},
                "Cannot find source XBG — select the imported armature or "
                "use 'Remember This XBG'")
            return {'CANCELLED'}
        try:
            lfa = parse_lfa(lfa_path)
            chans = parse_lfe(self.filepath)
            n_frames, n_chans, n_skipped = apply_lfe_expression(
                ctx, lfa, chans, arm, xbg_path)
            msg = f"LFE: {n_chans} pose channels over {n_frames} frames"
            if n_skipped:
                msg += (f" ({n_skipped} channels skipped — not in this "
                        f".lfa's pose list; the quaridge .lfa covers more)")
            self.report({'WARNING' if n_skipped else 'INFO'}, msg)
            return {'FINISHED'}
        except Exception as exc:
            self.report({'ERROR'}, f"Failed to import .lfe: {exc}")
            import traceback; traceback.print_exc()
            return {'CANCELLED'}


class XBG_OT_ScanSceneMABFC2(bpy.types.Operator):
    """Inspect a .mab's scripted-scene data without importing anything."""
    bl_idname  = "xbg.scan_scene_mab_fc2"
    bl_label   = "Scan Scene MAB"
    bl_description = (
        "Pick a .mab file and list its scene elements (anchors, cameras), "
        "timed events (sound/dialog/FX/camera cues) and combined-rig info "
        "in the panel below — nothing is imported yet"
    )
    bl_options = {'REGISTER'}

    filepath: bpy.props.StringProperty(subtype="FILE_PATH")
    filter_glob: bpy.props.StringProperty(default="*.mab", options={'HIDDEN'})

    def invoke(self, ctx, ev):
        ctx.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}

    def execute(self, ctx):
        from .mab_scene_fc2 import scan_scene
        from .import_mab_fc2 import (parse_sections as _ps, _stream_counts,
                                 read_routing_masks, _MASK_SLOT)
        ds = ctx.scene.xbg_debug_settings
        if not self.filepath or not os.path.isfile(self.filepath):
            self.report({'ERROR'}, "No valid .mab file selected")
            return {'CANCELLED'}
        try:
            info = scan_scene(self.filepath)
            d = open(self.filepath, 'rb').read()
            sec = _ps(d)
            tc, rr = _stream_counts(d, sec)
            # find the combined-rig mask domain size
            domain = 0
            for nb in range(1, 8 * _MASK_SLOT + 1):
                a, c = read_routing_masks(d, sec, nb)
                if a is not None:
                    domain = nb
                    break
            lines = ["%s — %.2f s" % (os.path.basename(self.filepath),
                                      info['animlen'])]
            lines.append("Combined rig: %d bones, %d animated + %d constant"
                         % (domain, tc, rr))
            if info['elements']:
                lines.append("— Scene elements —")
                for el in info['elements']:
                    tag = 'camera' if el['kind'] == 9 else 'anchor'
                    s = "[%s] %s" % (tag, el['name'] or '<unnamed>')
                    if el['parent']:
                        s += "  on " + el['parent']
                    lines.append(s)
            cuts = info.get('fov_cuts', [])
            has_cam = any(e['kind'] == 9 for e in info['elements'])
            if cuts:
                lines.append("— Camera shots (%s) —"
                             % ("animated camera" if has_cam
                                else "live-camera FOV only"))
                for si, (t, fov) in enumerate(cuts):
                    end_t = cuts[si + 1][0] if si + 1 < len(cuts) \
                        else info['animlen']
                    lines.append("SHOT %d  %5.2fs  %d\xb0  (%.1fs)"
                                 % (si + 1, t, round(fov), max(0.0, end_t - t)))
            if info['events']:
                lines.append("— Timed events —")
                for ev in info['events']:
                    if ev['type'] == 'SetFOV':
                        continue
                    s = "%6.2fs  %s" % (ev['time'], ev['type'])
                    if ev['strings']:
                        s += "  " + ev['strings'][0]
                    lines.append(s)
            if not info['elements'] and not info['events']:
                lines.append("(no scripted-scene data in this clip)")
            ds.scene_report = "\n".join(lines)
            ds.scene_mab_path = self.filepath
            self.report({'INFO'},
                "Scene: %d elements, %d events — see panel"
                % (len(info['elements']), len(info['events'])))
            return {'FINISHED'}
        except Exception as exc:
            self.report({'ERROR'}, f"Failed to scan .mab: {exc}")
            import traceback; traceback.print_exc()
            return {'CANCELLED'}


class XBG_OT_ImportSceneMABFC2(bpy.types.Operator):
    """Create the scene's anchors/cameras and event timeline markers."""
    bl_idname  = "xbg.import_scene_mab_fc2"
    bl_label   = "Import Scene Elements"
    bl_description = (
        "Build the scanned .mab's scene in Blender: an empty per anchor, a "
        "camera object per animated camera (initial orientation from the "
        "file), and a timeline marker per timed event (sound/FX/dialog "
        "cues). Camera/anchor motion tracks are not decoded yet"
    )
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, ctx):
        from .mab_scene_fc2 import build_scene_objects
        ds = ctx.scene.xbg_debug_settings
        path = bpy.path.abspath(ds.scene_mab_path).strip() \
            if ds.scene_mab_path else ''
        if not path or not os.path.isfile(path):
            self.report({'ERROR'}, "Scan a scene .mab first")
            return {'CANCELLED'}
        try:
            r = build_scene_objects(ctx, path)
            if r['synthetic_camera']:
                msg = ("Scene: %d elements, %d event markers. This clip has "
                       "no authored camera path (live gameplay camera) — "
                       "created a camera with its %d FOV cuts at the scene "
                       "anchor; reposition it to frame the shot (Numpad 0)"
                       % (r['elements'], r['events'], r['cuts']))
            elif r['cameras']:
                msg = ("Scene: %d elements, %d cameras (%d FOV cuts), "
                       "%d event markers — active camera bound, look "
                       "through it (Numpad 0) to view the shot"
                       % (r['elements'], r['cameras'], r['cuts'], r['events']))
            else:
                msg = ("Scene: %d elements, %d event markers (no camera "
                       "or FOV data in this clip)"
                       % (r['elements'], r['events']))
            self.report({'INFO'}, msg)
            return {'FINISHED'}
        except Exception as exc:
            self.report({'ERROR'}, f"Failed to import scene: {exc}")
            import traceback; traceback.print_exc()
            return {'CANCELLED'}


# ---------------------------------------------------------------------------
# Operators — HKX collision (XML workflow)
# ---------------------------------------------------------------------------

class XBG_OT_NullSelectedVertsFC2(bpy.types.Operator):
    """Move selected vertices to (0, 0, 0) so they export as zeroed HKX verts."""
    bl_idname  = "xbg.null_selected_verts_fc2"
    bl_label   = "Null Selected Verts"
    bl_description = "Move selected vertices to world origin (0,0,0)"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, ctx):
        import bmesh
        obj = ctx.active_object
        if not obj or obj.type != 'MESH' or obj.mode != 'EDIT':
            self.report({'ERROR'}, "Must be in Edit Mode on a mesh object")
            return {'CANCELLED'}

        bm = bmesh.from_edit_mesh(obj.data)
        moved = 0
        for v in bm.verts:
            if v.select:
                v.co.x = 0.0
                v.co.y = 0.0
                v.co.z = 0.0
                moved += 1

        if moved == 0:
            self.report({'WARNING'}, "No vertices selected")
            return {'CANCELLED'}

        bmesh.update_edit_mesh(obj.data)
        self.report({'INFO'}, f"Nulled {moved} vert(s) to (0, 0, 0)")
        return {'FINISHED'}


# ---------------------------------------------------------------------------
# Panels  (redesigned for beginner-friendliness)
# ---------------------------------------------------------------------------

# ── Main container panel ────────────────────────────────────────────────────

class XBG_OT_ExportMaterialsFC2(bpy.types.Operator):
    """Bake the active object's materials and write game-ready .xbt
    textures + .xbm material files into the selected patch folder."""
    bl_idname = "xbg.export_materials_fc2"
    bl_label = "Export Custom Materials"
    bl_options = {'REGISTER'}

    # File browser — user navigates to the patch output folder.
    directory: bpy.props.StringProperty(subtype='DIR_PATH')
    filter_folder: bpy.props.BoolProperty(default=True, options={'HIDDEN'})

    size: bpy.props.EnumProperty(
        name="Texture Size",
        items=[('SOURCE', "Same as source", "Match each material's source "
                "texture resolution (falls back to 1024)"),
               ('512',   "512 × 512",   ""),
               ('1024',  "1024 × 1024", ""),
               ('2048',  "2048 × 2048", "")],
        default='SOURCE')
    tex_dir: bpy.props.StringProperty(
        name="Texture Folder",
        description="Engine-relative folder for baked .xbt files, e.g. "
        "graphics\\av_characters\\custom  (the .xbm always points here)",
        default="graphics\\av_characters\\custom")
    only_custom: bpy.props.BoolProperty(
        name="Only NEW materials",
        description="Skip original game materials — only bake new ones you added",
        default=True)
    emissive_always_on: bpy.props.BoolProperty(
        name="Emission Always On",
        description="Keep emissive materials glowing in daylight.\n"
        "ON  = visible day and night (IlluminationColor1.alpha = 0).\n"
        "OFF = night-only, scales with the game's bioluminescence system.\n"
        "Has no effect on Unlit shaders",
        default=True)
    mat_templates: bpy.props.CollectionProperty(type=XBGMatTemplateItem)

    def invoke(self, ctx, ev):
        obj = ctx.active_object
        if obj is None or obj.type != 'MESH' or not obj.material_slots:
            self.report({'ERROR'}, "Select a mesh object with materials")
            return {'CANCELLED'}
        try:
            df = get_prefs(ctx).data_folder
        except Exception:
            df = ""
        self.mat_templates.clear()
        seen = set()
        for slot in obj.material_slots:
            m = slot.material
            if not m or not m.use_nodes or m.name in seen:
                continue
            seen.add(m.name)
            item = self.mat_templates.add()
            item.mat_name  = m.name
            item.is_game   = _export_materials._is_game_material(m, df)
            # Resolution priority for the UI hint mirrors what execute()
            # actually does at export time, so the user sees the SAME
            # template the file will get written with.  Without the host
            # lookup the hint always said "Generic" even when the export
            # was about to pick Flesh / Cloth via inheritance.
            inferred = _export_materials._infer_host_template(obj, m, df)
            item.auto_type = (inferred
                              or _export_materials.resolve_template_type(m))
            item.template  = 'AUTO'
        ctx.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}

    def draw(self, ctx):
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False
        col = layout.column(align=False)
        col.prop(self, 'size')
        col.prop(self, 'tex_dir')
        col.prop(self, 'only_custom')
        col.prop(self, 'emissive_always_on')
        col.separator()
        col.label(text="Shader per material:", icon='MATERIAL')
        for item in self.mat_templates:
            skipped = item.is_game and self.only_custom
            # Display the BASE material name, not the full engine path.
            # Game-sourced or already-exported materials carry names like
            # 'GRAPHICS\\_MATERIALS\\hmf_arm_edia_mat_1a.xbm' — strip the
            # path and the .xbm extension so the user sees the actual
            # material identifier they care about.
            base = _export_materials.safe_name(item.mat_name)
            disp = base if len(base) <= 32 else base[:30] + '…'
            row = col.row(align=True)
            row.enabled = not skipped
            row.label(text=disp,
                      icon='X' if skipped else 'CHECKMARK')
            row.prop(item, 'template', text="")
            if item.template == 'AUTO' and item.auto_type:
                hint = row.row()
                hint.enabled = False
                hint.label(text=f"({item.auto_type})")
            # Description sub-line for the currently-selected template.
            # Resolves AUTO to its detected target so the user sees what
            # the auto choice actually maps to, not just "auto".
            if not skipped:
                tpl = (item.auto_type if item.template == 'AUTO'
                       else item.template)
                desc = _TEMPLATE_DESCRIPTIONS.get(tpl, '')
                if desc:
                    sub = col.row()
                    sub.enabled = False
                    sub.label(text=f"      {desc}", icon='BLANK1')
            col.separator(factor=0.3)

    def execute(self, ctx):
        obj = ctx.active_object
        df = get_prefs(ctx).data_folder
        if not df or not os.path.isdir(df):
            self.report({'ERROR'}, "Set the game Data folder in add-on prefs")
            return {'CANCELLED'}
        out = bpy.path.abspath(self.directory) if self.directory else ""
        if not out:
            out = os.path.join(os.path.dirname(df.rstrip('\\/')), "patch")
        os.makedirs(out, exist_ok=True)
        sz = self.size if self.size == 'SOURCE' else int(self.size)
        # Build per-material template map from the collection.
        tmap = {item.mat_name: (None if item.template == 'AUTO'
                                else item.template)
                for item in self.mat_templates}
        VerboseLogger.session_marker(
            "export_materials",
            active_obj=(obj.name if obj else "<none>"),
            data_folder=df, output_folder=out,
            tex_dir=self.tex_dir, size=str(self.size),
            only_custom=self.only_custom,
            emissive_always_on=self.emissive_always_on,
            template_overrides={k: v for k, v in tmap.items() if v is not None},
            n_materials=len(self.mat_templates))
        try:
            written = _export_materials.export_object_materials(
                obj, df, out, self.tex_dir, size=sz,
                only_custom=self.only_custom,
                template_overrides=tmap,
                emissive_always_on=self.emissive_always_on)
        except Exception as exc:
            import traceback
            tb = traceback.format_exc()
            VerboseLogger.session_complete(
                "export_materials",
                status="EXCEPTION",
                exception=str(exc)[:512],
                traceback=tb[:2048])
            VerboseLogger.autosave_sidecar(out)
            self.report({'ERROR'}, f"Export failed: {exc}")
            traceback.print_exc()
            return {'CANCELLED'}
        if not written:
            VerboseLogger.session_complete(
                "export_materials",
                status="EMPTY",
                reason="no materials exported (all skipped?)")
            VerboseLogger.autosave_sidecar(out)
            self.report({'WARNING'}, "No materials exported (all skipped?)")
            return {'CANCELLED'}
        VerboseLogger.session_complete(
            "export_materials",
            output_folder=out,
            n_materials_written=len(written),
            written_paths=list(written.values()))
        VerboseLogger.autosave_sidecar(out)
        self.report({'INFO'}, f"Exported {len(written)} material(s) to {out}")
        return {'FINISHED'}


