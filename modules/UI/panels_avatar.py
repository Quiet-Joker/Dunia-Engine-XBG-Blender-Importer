"""Avatar: The Game — UI panels (children of the Avatar root panel).

Split out of the monolithic __init__.py (2026-06-09 refactor).
"""
import os

import bpy

from ..Core.debug import VerboseLogger
from ..Core.prefs import get_prefs
from ..Avatar.inject_xbg_avatar import calculate_required_scale


class XBG_PT_ImportPanel(bpy.types.Panel):
    """Import an XBG file into Blender."""
    bl_label = "Import XBG"
    bl_idname = "OBJECT_PT_xbg_step1_import"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "XBG Import"
    bl_parent_id = "OBJECT_PT_xbg_avatar"

    def draw_header(self, ctx):
        self.layout.label(icon='IMPORT')

    def draw(self, ctx):
        l = self.layout
        s = ctx.scene.xbg_settings
        p = get_prefs(ctx)
        ds = ctx.scene.xbg_debug_settings

        # Game data folder
        col = l.column(align=True)
        col.label(text="Game Data Folder:", icon='FILE_FOLDER')
        col.prop(p, "data_folder", text="")
        if not p.data_folder:
            col.label(text="Set this to load textures automatically", icon='INFO')

        l.separator()

        # Texture options
        l.prop(s, "load_textures", text="Load Textures Automatically")
        if s.load_textures:
            hd_row = l.row()
            hd_row.enabled = bool(p.data_folder)
            hd_row.prop(s, "load_hd_textures", text="Use High-Quality (HD) Textures")

        if ds.advanced_mode:
            l.separator()
            l.operator("xbg.peek_lods",
                       text="Check How Many LODs a File Has", icon='VIEWZOOM')
            if ds.lod_peek_result:
                res_row = l.row()
                res_row.alignment = 'LEFT'
                res_row.label(text=ds.lod_peek_result, icon='INFO')

        l.separator()

        # Import button
        r = l.row()
        r.scale_y = 1.8
        r.operator("import_scene.xbg_model",
                   text="   Import XBG File", icon='IMPORT')

        if not ds.advanced_mode:
            return


class XBG_PT_InjectPanel(bpy.types.Panel):
    """Inject your edited mesh back into the XBG file."""
    bl_label = "Inject / Export"
    bl_idname = "OBJECT_PT_xbg_step3_inject"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "XBG Import"
    bl_parent_id = "OBJECT_PT_xbg_avatar"

    @classmethod
    def poll(cls, ctx):
        return ctx.scene.xbg_debug_settings.advanced_mode

    def draw_header(self, ctx):
        self.layout.label(icon='EXPORT')

    def draw(self, ctx):
        l = self.layout
        ins     = ctx.scene.xbg_inject_settings
        session = ctx.scene.xbg_session_data
        obj     = ctx.active_object

        # ── Joined mesh warning ─────────────────────────────────────────
        if obj and "xbg_joined" in obj:
            warn = l.box()
            warn.alert = True
            warn.label(text="Cannot inject this mesh!", icon='ERROR')
            warn.label(text="It was imported as a single joined object.")
            warn.label(text="Enable 'Separate Primitives' in Advanced")
            warn.label(text="Settings, then re-import to fix this.")
            return

        has_session  = session.is_loaded and bool(session.filepath)
        has_obj_data = (obj is not None
                        and "xbg_data" in obj
                        and "xbg_joined" not in obj)

        # ── XBG link status ─────────────────────────────────────────────
        if has_obj_data:
            info = l.box()
            info.label(
                text=f"Linked:  {os.path.basename(obj['xbg_data']['filepath'])}",
                icon='LINKED')
            info.operator("xbg.remember_xbg",
                          text="Pin This File (Keep Panel Visible)", icon='PINNED')

        if has_session:
            sess_box = l.box()
            sess_box.label(
                text=f"Pinned:  {os.path.basename(session.filepath)}",
                icon='BOOKMARKS')
            sess_box.operator("xbg.clear_session_xbg",
                              text="Unpin / Clear", icon='X')

        if not has_session and not has_obj_data:
            l.separator(factor=0.3)
            hint = l.box()
            hint.label(text="No XBG mesh selected.", icon='INFO')
            hint.label(text="Import an XBG file first,")
            hint.label(text="then select the imported mesh.")
            return

        # Resolve pos_scale for bounds check
        if has_session:
            ps  = session.pos_scale
            imo = session.import_mesh_only
        else:
            m   = obj["xbg_data"].to_dict()
            ps  = m.get("pos_scale", 1.0)
            imo = m.get("import_mesh_only", False)

        l.separator()

        # ── Mesh size status ────────────────────────────────────────────
        if obj and obj.type == 'MESH':
            effective_ps = ins.target_game_scale if ins.override_game_scale else ps
            ns, _, si_   = calculate_required_scale(obj, effective_ps, imo)
            status = l.box()
            if ns:
                status.alert = True
                status.label(text="Mesh is too large for XBG!", icon='ERROR')
                status.label(text=f"Problem axis: {si_}")
                status.separator(factor=0.3)
                r_auto = status.row()
                r_auto.scale_y = 1.3
                r_auto.operator("xbg.auto_scale_bounds",
                                text="Auto Scale Bounds", icon='FULLSCREEN_ENTER')
                status.separator(factor=0.3)
                ignore_row = status.row()
                ignore_row.alert = ins.ignore_format_limits
                ignore_row.prop(ins, "ignore_format_limits",
                                text="Ignore Limits (may corrupt!)",
                                icon='ERROR' if ins.ignore_format_limits else 'CANCEL')
                if ins.ignore_format_limits:
                    note = status.box()
                    note.label(text="Coordinates will wrap — not clamp.", icon='ERROR')
                    note.label(text="This CAN corrupt the model!")
            elif ins.override_game_scale:
                status.label(text="Bounds expanded — mesh will fit  ✓", icon='CHECKMARK')
            else:
                status.label(text="Mesh size looks good!  ✓", icon='CHECKMARK')

        l.separator()

        # ── Core export settings ────────────────────────────────────────
        col = l.column(align=True)
        col.label(text="Which LOD to replace?  (0 = best quality)")
        col.prop(ins, "target_lod", text="LOD Slot")

        l.separator()
        l.prop(ins, "inject_bone_weights")

        l.separator()

        # Selection summary
        sel_meshes = [o for o in ctx.selected_objects if o.type == 'MESH']
        if sel_meshes:
            l.label(text=f"{len(sel_meshes)} mesh object(s) ready to inject",
                    icon='OBJECT_DATA')
            if len(sel_meshes) > 1:
                l.label(text="Each object will become one submesh",
                        icon='INFO')
        else:
            l.label(text="No mesh selected — will use the active object",
                    icon='INFO')

        # Big inject button (Avatar / FC2)
        # (FC3/FC4 inject lives in its own game panel — see ui/main.py)
        r = l.row()
        r.scale_y = 1.8
        r.operator("xbg.inject_mesh",
                   text="   Inject Mesh into XBG (Avatar/FC2)", icon='EXPORT')


# ── Advanced inject options (hidden by default) ─────────────────────────────

class XBG_PT_InjectAdvancedPanel(bpy.types.Panel):
    """Advanced inject/export options — for experienced users."""
    bl_label = "Advanced Export Options"
    bl_idname = "OBJECT_PT_xbg_inject_advanced"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "XBG Import"
    bl_parent_id = "OBJECT_PT_xbg_step3_inject"
    bl_options = {'DEFAULT_CLOSED'}

    def draw_header(self, ctx):
        self.layout.label(icon='PREFERENCES')

    def draw(self, ctx):
        l = self.layout
        ins     = ctx.scene.xbg_inject_settings
        session = ctx.scene.xbg_session_data
        obj     = ctx.active_object

        # Resolve pos_scale for display
        if session.is_loaded and session.pos_scale > 0:
            ps = session.pos_scale
        elif obj and "xbg_data" in obj:
            ps = float(obj["xbg_data"].get("pos_scale", 1.0))
        else:
            ps = 1.0

        # ── Expand bounds ───────────────────────────────────────────────
        exp_box = l.box()
        exp_box.label(text="Expand Bounds:", icon='FULLSCREEN_ENTER')
        exp_box.label(text="Expand uint16 format bounds which controls")
        exp_box.label(text="the vertex accuracy for the mesh.")
        exp_box.separator(factor=0.3)
        exp_box.operator("xbg.expand_bounds_for_inject",
                         text="Expand Bounds for Inject",
                         icon='FULLSCREEN_ENTER')

        l.separator()

        # ── Optional data ───────────────────────────────────────────────
        opt_box = l.box()
        opt_box.label(text="Optional Data to Include:", icon='OPTIONS')
        opt_box.prop(ins, "inject_vertex_colors")
        # Sub-option: only meaningful when Include Vertex Colors is ON (greyed
        # out otherwise). Stops unpainted new geometry exporting a black shadow.
        sub = opt_box.column(align=True)
        sub.enabled = ins.inject_vertex_colors
        sub.prop(ins, "generate_neutral_vertex_colors")
        opt_box.prop(ins, "inject_materials")
        # (inject_bone_weights lives in the main Inject panel only - it's
        # load-bearing for every export, not an advanced-only option.)

        # (Tangent Space box removed — tangents for new geometry are always
        #  computed from UVs now; not user-configurable. Stock verts keep their
        #  stored tangents.)

        l.separator()

        # ── Weight cleanup ──────────────────────────────────────────────
        if ins.inject_bone_weights:
            w_box = l.box()
            w_box.label(text="Weight Cleanup:", icon='MOD_VERTEX_WEIGHT')
            w_box.operator("xbg.normalize_weights",
                            text="Normalize Weights to Bones", icon='MOD_VERTEX_WEIGHT')
            note = w_box.box()
            note.label(text="Optional — export already fills gaps silently.", icon='INFO')
            note.label(text="Run this to see/adjust the result before exporting.")



# ── Advanced settings — split into focused panels (2026-06-30 cleanup) ──────
# Previously two panels: "Advanced Settings" (4 import toggles) and "Debug"
# (7 unrelated sections crammed into one DEFAULT_CLOSED box, and oddly
# parented to the root game-picker panel instead of the Avatar root). Split
# into one panel per concern so each is independently collapsible/skippable
# instead of forcing a scroll through everything to find one toggle.

# (XBG_PT_ImportBehaviorPanel removed 2026-06-30 — its options
# [flip_normals, auto_smooth_normals, separate_primitives, use_xml_assembly,
# import_xbt_as_dds] moved into the Import XBG file-select popup, i.e.
# XBG_OT_Import.draw in operators_avatar.py, so every import-time choice
# lives in one place next to LOD/mesh-only/reorient.)


class XBG_PT_MeshCleanupPanel(bpy.types.Panel):
    bl_label = "Mesh Cleanup"
    bl_idname = "OBJECT_PT_xbg_mesh_cleanup"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "XBG Import"
    bl_parent_id = "OBJECT_PT_xbg_avatar"
    bl_options = {'DEFAULT_CLOSED'}

    @classmethod
    def poll(cls, ctx):
        return ctx.scene.xbg_debug_settings.advanced_mode

    def draw_header(self, ctx):
        self.layout.label(icon='AUTOMERGE_ON')

    def draw(self, ctx):
        l, ds = self.layout, ctx.scene.xbg_debug_settings

        b = l.box()
        b.label(text="Remove Unused Vertices:", icon='VERTEXSEL')
        b.prop(ds, "compact_vertices", text="Remove Unused Vertices")

        l.separator()

        b = l.box()
        b.label(text="Merge Duplicate Vertices:", icon='AUTOMERGE_ON')
        b.prop(ds, "merge_distance", text="Merge Distance")
        row = b.row(align=True)
        row.operator("xbg.merge_all_meshes",    text="All Meshes")
        row.operator("xbg.merge_selected_mesh", text="Selected Only")


class XBG_PT_SkeletonMB2OPanel(bpy.types.Panel):
    bl_label = "Skeleton / MB2O"
    bl_idname = "OBJECT_PT_xbg_skeleton_mb2o"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "XBG Import"
    bl_parent_id = "OBJECT_PT_xbg_avatar"
    bl_options = {'DEFAULT_CLOSED'}

    @classmethod
    def poll(cls, ctx):
        return ctx.scene.xbg_debug_settings.advanced_mode

    def draw_header(self, ctx):
        self.layout.label(icon='ARMATURE_DATA')

    def draw(self, ctx):
        l, ds = self.layout, ctx.scene.xbg_debug_settings
        l.prop(ds, "use_mb2o", text="Use MB2O Bone Transforms")
        if ds.use_mb2o:
            note = l.box()
            note.label(text="Cross-checks bones against MB2O.", icon='INFO')
            note.label(text="Falls back to EDON per-bone if unsure.")
        else:
            note = l.box()
            note.label(text="Using standard EDON transforms.", icon='INFO')
            note.label(text="(Already verified correct - no need to enable MB2O.)")


class XBG_PT_ExperimentalInjectPanel(bpy.types.Panel):
    bl_label = "Experimental Injection"
    bl_idname = "OBJECT_PT_xbg_experimental_inject"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "XBG Import"
    bl_parent_id = "OBJECT_PT_xbg_avatar"
    bl_options = {'DEFAULT_CLOSED'}

    @classmethod
    def poll(cls, ctx):
        return ctx.scene.xbg_debug_settings.advanced_mode

    def draw_header(self, ctx):
        self.layout.label(icon='EXPERIMENTAL')

    def draw(self, ctx):
        l = self.layout
        ins = ctx.scene.xbg_inject_settings

        row = l.row()
        row.alert = ins.force_per_submesh_vb
        row.prop(ins, "force_per_submesh_vb",
                 text="Force Per-Submesh VB (>65k verts)",
                 icon='ERROR' if ins.force_per_submesh_vb else 'NONE')
        if ins.force_per_submesh_vb:
            warn = l.box()
            warn.alert = True
            warn.label(text="EXPERIMENTAL — untested for characters!", icon='ERROR')
            warn.label(text="Writes one vertex buffer per submesh instead of")
            warn.label(text="the source's shared-VB layout. Each submesh can")
            warn.label(text="hold up to 65535 verts independently, so total")
            warn.label(text="can exceed 65535. But the engine's character")
            warn.label(text="skinning may bind VB 0 regardless of vb_idx, in")
            warn.label(text="which case only submesh 0 renders correctly.")
            warn.label(text="Use only when you NEED >65k LOD-0 verts.")
        else:
            note = l.box()
            note.scale_y = 0.85
            note.label(text="OFF: matches source VB layout (safe).", icon='INFO')
            note.label(text="LOD vertex cap = 65535 for character meshes.")


class XBG_PT_DiagnosticsPanel(bpy.types.Panel):
    bl_label = "Diagnostics & Logging"
    bl_idname = "OBJECT_PT_xbg_diagnostics"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "XBG Import"
    bl_parent_id = "OBJECT_PT_xbg_avatar"
    bl_options = {'DEFAULT_CLOSED'}

    @classmethod
    def poll(cls, ctx):
        return ctx.scene.xbg_debug_settings.advanced_mode

    def draw_header(self, ctx):
        self.layout.label(icon='CONSOLE')

    def draw(self, ctx):
        l, ds = self.layout, ctx.scene.xbg_debug_settings

        l.prop(ds, "verbose_logging", text="Verbose Console Output")
        if ds.verbose_logging:
            row = l.row()
            row.alert = ds.trace_logging
            row.prop(ds, "trace_logging",
                     text="Trace-Level Logging (per-vertex / per-byte)",
                     icon='RECORD_ON' if ds.trace_logging else 'NONE')
            if ds.trace_logging:
                warn = l.box()
                warn.scale_y = 0.85
                warn.label(text="TRACE ON: log will be 100s of KB per export.", icon='INFO')
                warn.label(text="A structured .jsonl file is written next to")
                warn.label(text="the saved log for programmatic inspection.")
            row = l.row(align=True)
            row.operator("xbg.save_log",  text="Save Log to File", icon='TEXT')
            row.operator("xbg.reset_log", text="Reset Session",    icon='TRASH')

        l.separator()
        l.prop(ds, "show_file_info",  text="Show XBG Chunk Info")
        if ds.show_file_info and ds.file_info_data:
            info_box = l.box()
            info_box.scale_y = 0.75
            for line in ds.file_info_data.split('\n'):
                if line.strip():
                    r = info_box.row()
                    r.alignment = 'LEFT'
                    r.label(text=line)


class XBG_PT_ViewportVizPanel(bpy.types.Panel):
    bl_label = "Viewport Visualizers"
    bl_idname = "OBJECT_PT_xbg_viewport_viz"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "XBG Import"
    bl_parent_id = "OBJECT_PT_xbg_avatar"
    bl_options = {'DEFAULT_CLOSED'}

    @classmethod
    def poll(cls, ctx):
        return ctx.scene.xbg_debug_settings.advanced_mode

    def draw_header(self, ctx):
        self.layout.label(icon='SHADING_BBOX')

    def draw(self, ctx):
        l, ds = self.layout, ctx.scene.xbg_debug_settings
        sc = ctx.scene

        b = l.box()
        b.label(text="Format Bounds Visualizer:", icon='SHADING_BBOX')
        b.label(text="Shows the size limit as a box in the viewport.")
        b.prop(ds, "show_format_bounds", text="Show Bounds Box")
        if ds.show_format_bounds:
            lo = bpy.data.objects.get("XBG_Format_Bounds")
            if lo:
                sub = b.box()
                sub.label(text="Box Size:", icon='DRIVER_TRANSFORM')
                sub.row(align=True).prop(ds, "format_bounds_x")
                sub.row(align=True).prop(ds, "format_bounds_y")
                sub.row(align=True).prop(ds, "format_bounds_z")
                link_row = sub.row(align=True)
                link_row.scale_y = 0.8
                for prop, label in (("link_xy", "X↔Y"), ("link_yz", "Y↔Z"), ("link_xz", "X↔Z")):
                    val = getattr(ds, prop)
                    cell = link_row.row(align=True)
                    cell.alert = val
                    cell.prop(ds, prop, toggle=True, text=label,
                              icon='LINKED' if val else 'UNLINKED')
                save_r = sub.row()
                save_r.scale_y = 1.2
                save_r.operator("xbg.save_format_bounds_size",
                                text="Save New Box Size", icon='FILE_TICK')

        l.separator()

        # ── Bounding Volume Display + Editor (unified 2026-06-30) ──────────
        # One place for the XOBB box + HPSB sphere: toggle the gizmo, edit
        # its numbers, and the viewport box/sphere updates LIVE (the fields
        # drive XBG_Bounds_Box / XBG_Bounds_Sphere via property callbacks).
        # The last-imported model auto-populates these; "Read Bounds" loads
        # any other .xbg; "Save to XBG" writes the edited values back.
        b = l.box()
        b.label(text="Bounding Volume Display + Editor:", icon='MESH_CUBE')
        b.operator("xbg.read_bounds", text="Read Bounds from XBG", icon='IMPORT')
        if sc.xbg_bounds_path:
            b.label(text=os.path.basename(sc.xbg_bounds_path), icon='FILE')

        # XOBB box: checkbox toggles the gizmo; Min/Max edit it live.
        b.prop(ds, "show_bounding_box", text="Show Bounding Boxes (XOBB)")
        if ds.show_bounding_box:
            if sc.xbg_has_xobb:
                xb = b.box()
                xb.prop(sc, "xbg_box_min", text="Min")
                xb.prop(sc, "xbg_box_max", text="Max")
            else:
                b.label(text="No XOBB in the loaded model / file.", icon='INFO')

        # HPSB sphere: checkbox toggles the gizmo; Center/Radius edit it live.
        b.prop(ds, "show_bounding_sphere", text="Show Bounding Spheres (HPSB)")
        if ds.show_bounding_sphere:
            if sc.xbg_has_hpsb:
                sph = b.box()
                sph.prop(sc, "xbg_sphere_center", text="Center")
                sph.prop(sc, "xbg_sphere_radius", text="Radius")
            else:
                b.label(text="No HPSB in the loaded model / file.", icon='INFO')

        if ds.show_bounding_box or ds.show_bounding_sphere:
            b.prop(ds, "bounds_display_type", text="Display Style")

        if sc.xbg_bounds_path:
            row = b.row(align=True)
            row.operator("xbg.fit_bounds_to_selected", text="Fit to Selected", icon='SHADING_BBOX')
            row.operator("xbg.write_bounds", text="Save to XBG", icon='EXPORT')


# ── Export Custom Materials (split out of the Inject panel) ─────────────────

class XBG_PT_ExportMaterialsPanel(bpy.types.Panel):
    """Export custom-edited materials to game-format XBM/texture files."""
    bl_label = "Export Custom Materials"
    bl_idname = "OBJECT_PT_xbg_export_materials"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "XBG Import"
    bl_parent_id = "OBJECT_PT_xbg_avatar"
    bl_options = {'DEFAULT_CLOSED'}

    @classmethod
    def poll(cls, ctx):
        return ctx.scene.xbg_debug_settings.advanced_mode

    def draw_header(self, ctx):
        self.layout.label(icon='MATERIAL')

    def draw(self, ctx):
        l = self.layout
        l.label(text="Custom materials → game files:", icon='MATERIAL')
        r = l.row()
        r.scale_y = 1.4
        r.operator("xbg.export_materials",
                   text="   Export Custom Materials", icon='EXPORT')


# ── Skeleton import (.skeleton / LKS) ───────────────────────────────────────

class XBG_PT_SkeletonPanel(bpy.types.Panel):
    """Import a .skeleton (LKS) animation rig as an armature."""
    bl_label = "Skeleton Import"
    bl_idname = "OBJECT_PT_xbg_skeleton"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "XBG Import"
    bl_parent_id = "OBJECT_PT_xbg_avatar"
    bl_options = {'DEFAULT_CLOSED'}

    @classmethod
    def poll(cls, ctx):
        return ctx.scene.xbg_debug_settings.advanced_mode

    def draw_header(self, ctx):
        self.layout.label(icon='ARMATURE_DATA')

    def draw(self, ctx):
        l = self.layout
        l.label(text="Animation rig (.skeleton / LKS):", icon='BONE_DATA')
        r = l.row()
        r.scale_y = 1.4
        r.operator("xbg.import_lks_skeleton",
                   text="Import LKS Skeleton", icon='BONE_DATA')


# ── Facial animation (.lfa / .lfe) ──────────────────────────────────────────

class XBG_PT_FacialPanel(bpy.types.Panel):
    """Facial pose libraries (.lfa) and expression clips (.lfe)."""
    bl_label = "Facial Animation"
    bl_idname = "OBJECT_PT_xbg_facial"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "XBG Import"
    bl_parent_id = "OBJECT_PT_xbg_avatar"
    bl_options = {'DEFAULT_CLOSED'}

    @classmethod
    def poll(cls, ctx):
        return ctx.scene.xbg_debug_settings.advanced_mode

    def draw_header(self, ctx):
        self.layout.label(icon='SHAPEKEY_DATA')

    def draw(self, ctx):
        l = self.layout
        ds = ctx.scene.xbg_debug_settings

        b = l.box()
        b.label(text="Pose library (.lfa) — select the head rig:",
                icon='SHAPEKEY_DATA')
        b.label(text="One pose per frame + timeline markers.")
        r = b.row()
        r.scale_y = 1.4
        r.operator("xbg.import_lfa_poses",
                   text="Import LFA Facial Poses", icon='SHAPEKEY_DATA')

        l.separator()

        b = l.box()
        b.label(text="Expression clip (.lfe):", icon='ANIM')
        b.label(text="Head .lfa (defines the pose channels):")
        b.prop(ds, "lfa_path", text="")
        r = b.row()
        r.scale_y = 1.4
        r.operator("xbg.import_lfe_expression",
                   text="Import LFE Expression", icon='ANIM')


# ── Scene Viewer (scripted scenes: cameras / anchors / events) ──────────────

class XBG_PT_SceneViewerPanel(bpy.types.Panel):
    """In-depth animation / scene viewer for scripted-scene .mab files."""
    bl_label = "Animation / Scene Viewer"
    bl_idname = "OBJECT_PT_xbg_scene_viewer"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "XBG Import"
    bl_parent_id = "OBJECT_PT_xbg_avatar"
    bl_options = {'DEFAULT_CLOSED'}

    @classmethod
    def poll(cls, ctx):
        return ctx.scene.xbg_debug_settings.advanced_mode

    def draw_header(self, ctx):
        self.layout.label(icon='VIEW_CAMERA')

    def draw(self, ctx):
        l = self.layout
        ds = ctx.scene.xbg_debug_settings

        # ── Simple: one animation onto the selected armature ────────────
        b = l.box()
        b.label(text="Simple — animate the selected rig:", icon='ANIM_DATA')
        b.label(text="Skeleton (blank = auto-detect):")
        b.prop(ds, "mab_skeleton_path", text="")
        b.prop(ds, "mab_emulate_helpers")
        if ds.mab_emulate_helpers:
            b.prop(ds, "mab_twist_bake")
        b.prop(ds, "mab_smooth_resample")
        if ds.mab_smooth_resample:
            b.prop(ds, "mab_resample_fps")
        r = b.row()
        r.scale_y = 1.4
        r.operator("xbg.import_mab_animation",
                   text="Import MAB Animation", icon='ARMATURE_DATA')
        b.operator("xbg.preview_jiggle",
                   text="Preview Jiggle (procedural bones)", icon='PHYSICS')

        l.separator()

        # ── Scripted: multi-character scenes with cameras / events ──────
        b = l.box()
        b.label(text="Scripted Scene — cameras / anchors / events:",
                icon='SEQUENCE')
        r = b.row()
        r.scale_y = 1.3
        r.operator("xbg.scan_scene_mab",
                   text="Scan Scene MAB", icon='VIEWZOOM')
        if ds.scene_mab_path:
            b.label(text=os.path.basename(ds.scene_mab_path), icon='LINKED')

        if ds.scene_report:
            rep = b.box()
            rep.scale_y = 0.75
            for line in ds.scene_report.split('\n')[:40]:
                if line.strip():
                    rr = rep.row()
                    rr.alignment = 'LEFT'
                    rr.label(text=line)

            r2 = b.row()
            r2.scale_y = 1.3
            r2.operator("xbg.import_scene_mab",
                        text="Import Scene (anchors / cameras / cues)",
                        icon='OUTLINER_OB_CAMERA')

            b.separator(factor=0.5)
            b.label(text="Which character of the combined rig?",
                    icon='COMMUNITY')
            b.prop(ds, "mab_char_offset", text="Bone Offset")
            col = b.column(align=True)
            col.scale_y = 0.8
            col.label(text="0 = first character block. To use another")
            col.label(text="NPC's track, set the offset where its")
            col.label(text="skeleton starts (try multiples of your")
            col.label(text="rig's bone count), then import the MAB")
            col.label(text="above with the armature selected.")


# ── HKX Collision panel ──────────────────────────────────────────────────────

class XBG_PT_HKXPanel(bpy.types.Panel):
    """HKX collision editing — native binary import/export."""
    bl_label       = "HKX Collision Editing"
    bl_idname      = "OBJECT_PT_xbg_hkx"
    bl_space_type  = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category    = "XBG Import"
    bl_parent_id   = "OBJECT_PT_xbg_avatar"
    bl_options     = {'DEFAULT_CLOSED'}

    @classmethod
    def poll(cls, ctx):
        return ctx.scene.xbg_debug_settings.advanced_mode

    def draw_header(self, ctx):
        self.layout.label(icon='MESH_ICOSPHERE')

    def draw(self, ctx):
        l   = self.layout
        cs  = ctx.scene.xbg_collision_settings
        obj = ctx.active_object

        # ── Native binary workflow (no Havok tools needed) ───────────────
        nat = l.box()
        nat.label(text="Native .hkx — no conversion needed:", icon='FILE_3D')
        r = nat.row()
        r.scale_y = 1.5
        r.operator("xbg.import_hkx_native",
                   text="Import .hkx Collision", icon='IMPORT')
        col = nat.column(align=True)
        col.scale_y = 0.8
        col.label(text="Edit boxes / convex hulls / mesh verts, or ADD")
        col.label(text="new collision: parent any mesh (or a duplicate")
        col.label(text="of an existing shape) under a body's empty —")
        col.label(text="it's saved as a new convex shape on export.")
        r = nat.row()
        r.scale_y = 1.5
        r.operator("xbg.export_hkx_native",
                   text="Export .hkx (patch original)", icon='EXPORT')

        l.separator()

        # ── Status bar ───────────────────────────────────────────────────
        if cs.last_status:
            sb = l.box()
            sb.alert = not cs.last_status_ok
            icon   = 'CHECKMARK' if cs.last_status_ok else 'ERROR'
            status = cs.last_status
            for i, line in enumerate(
                [status[j:j + 45] for j in range(0, len(status), 45)]
            ):
                sb.label(text=line, icon=icon if i == 0 else 'BLANK1')





