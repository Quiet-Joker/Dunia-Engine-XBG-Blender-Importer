"""Watch Dogs 1 / 2 — UI panels (children of OBJECT_PT_xbg_wd in main.py).

Mirrors the Avatar layout: one collapsible sub-panel per task, each with a
header icon.  Noob mode = Import only; Advanced Mode reveals Animation,
Edit & Inject, and Advanced Settings.
"""

import bpy

from .main import active_game


def _adv(ctx):
    return ctx.scene.xbg_debug_settings.advanced_mode


# ── Import ──────────────────────────────────────────────────────────────────

class XBG_PT_WDImport(bpy.types.Panel):
    """Import a Watch Dogs model into Blender."""
    bl_label = "Import XBG"
    bl_idname = "OBJECT_PT_xbg_wd_import"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "XBG Import"
    bl_parent_id = "OBJECT_PT_xbg_wd"

    def draw_header(self, ctx):
        self.layout.label(icon='IMPORT')

    def draw(self, ctx):
        l = self.layout
        ds = ctx.scene.xbg_debug_settings
        game = active_game(ctx)

        r = l.row()
        r.scale_y = 1.8
        if game == 'WD1':
            r.operator("xbg.import_wd_model",
                       text="   Import WD1 Model (.xbg)", icon='IMPORT')
            col = l.column(align=True)
            col.label(text="Binary GEOM 97.50", icon='FILE_CACHE')
            col.label(text="Skeleton · weights · UV1/UV2 · normals,")
            col.label(text="tangents/binormals · vertex colors")
        else:
            r.operator("xbg.import_wd2_model",
                       text="   Import WD2 Model (.glm)", icon='IMPORT')
            col = l.column(align=True)
            col.label(text="Raw text GEOM source", icon='FILE_TEXT')
            col.label(text="Skeleton · weights · UVs · normals")

        if game == 'WD1':
            l.separator()
            r2 = l.row()
            r2.scale_y = 1.3
            r2.operator("xbg.import_wd_skeleton",
                        text="Import WD1 Skeleton (.skeleton)", icon='ARMATURE_DATA')
            r2 = l.row()
            r2.scale_y = 1.3
            r2.operator("xbg.import_wd_hkx",
                        text="Import WD1 HKX Collision (.hkx)", icon='MESH_ICOSPHERE')

        if _adv(ctx) and game == 'WD1':
            l.separator()
            l.operator("xbg.wd_peek_lods",
                       text="Check How Many LODs a File Has", icon='VIEWZOOM')
            if ds.lod_peek_result:
                res = l.column(align=True)
                res.scale_y = 0.8
                for part in ds.lod_peek_result.split('; '):
                    res.label(text=part, icon='INFO')
            l.separator()
            b = l.box()
            b.label(text="Import options:", icon='TOOL_SETTINGS')
            b.prop(ds, "separate_primitives",
                   text="Separate Primitives (needed to inject)")
            if not ds.separate_primitives:
                note = b.column(align=True)
                note.scale_y = 0.8
                note.label(text="OFF: submeshes joined into one object",
                           icon='INFO')
                note.label(text="(clean view; can't be injected back).")
        elif not _adv(ctx):
            l.label(text="Enable Advanced Mode for editing / injection.",
                    icon='INFO')


# ── Animation (.mab) ────────────────────────────────────────────────────────

class XBG_PT_WDAnimation(bpy.types.Panel):
    bl_label = "Animation"
    bl_idname = "OBJECT_PT_xbg_wd_anim"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "XBG Import"
    bl_parent_id = "OBJECT_PT_xbg_wd"
    bl_options = {'DEFAULT_CLOSED'}

    @classmethod
    def poll(cls, ctx):
        return _adv(ctx) and active_game(ctx) == 'WD1'

    def draw_header(self, ctx):
        self.layout.label(icon='ARMATURE_DATA')

    def draw(self, ctx):
        l = self.layout
        l.label(text="Animation (.mab) — select armature first:",
                icon='ANIM_DATA')
        ds = ctx.scene.xbg_debug_settings
        l.prop(ds, "mab_emulate_helpers")
        if ds.mab_emulate_helpers:
            l.prop(ds, "mab_twist_bake")
        l.prop(ds, "mab_smooth_resample")
        if ds.mab_smooth_resample:
            l.prop(ds, "mab_resample_fps")
        r = l.row()
        r.scale_y = 1.4
        r.operator("xbg.import_wd_mab",
                   text="Import WD1 MAB", icon='ARMATURE_DATA')


# ── Edit & Inject ───────────────────────────────────────────────────────────

class XBG_PT_WDInject(bpy.types.Panel):
    """Inject your edited mesh back into the XBG file."""
    bl_label = "Inject / Export"
    bl_idname = "OBJECT_PT_xbg_wd_inject"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "XBG Import"
    bl_parent_id = "OBJECT_PT_xbg_wd"
    # Open by default when Advanced Mode reveals it — matches the Avatar
    # master layout (Import + Inject expanded, secondary panels closed).

    @classmethod
    def poll(cls, ctx):
        return _adv(ctx) and active_game(ctx) in ('WD1', 'WD2')

    def draw_header(self, ctx):
        self.layout.label(icon='EXPORT')

    def draw(self, ctx):
        l = self.layout
        ds = ctx.scene.xbg_debug_settings

        # ── WD2: text .glm export ────────────────────────────────────────
        if active_game(ctx) == 'WD2':
            joined2 = [o for o in ctx.selected_objects if o.get('wd_joined')]
            if joined2:
                w = l.box()
                w.alert = True
                w.label(text="This mesh was imported JOINED.", icon='ERROR')
                w.label(text="Turn on Separate Primitives and")
                w.label(text="re-import to export it.")
            col = l.column(align=True)
            col.label(text="Edit imported meshes freely — the text", icon='INFO')
            col.label(text="format supports count changes. Then:")
            l.separator()
            r = l.row()
            r.scale_y = 1.8
            r.enabled = any(o.get('wd2_src') for o in ctx.scene.objects)
            r.operator("xbg.export_wd2_model",
                       text="   Export WD2 Model (.glm)", icon='EXPORT')
            note = l.column(align=True)
            note.scale_y = 0.8
            note.label(text="Materials/skeleton/physics blocks are", icon='INFO')
            note.label(text="kept byte-for-byte. Compile the output")
            note.label(text="with GLM2XBG to get a game .xbg.")
            return

        joined = [o for o in ctx.selected_objects if o.get('wd_joined')]
        if joined:
            w = l.box()
            w.alert = True
            w.label(text="This mesh was imported JOINED.", icon='ERROR')
            w.label(text="Turn on Separate Primitives and")
            w.label(text="re-import to edit & inject it.")
        if any(o.get('wd_multibuffer') for o in ctx.selected_objects):
            mb = l.box()
            mb.label(text="Multi-buffer vehicle:", icon='AUTO')
            c = mb.column(align=True)
            c.scale_y = 0.8
            c.label(text="In-place vertex editing only (reshape /")
            c.label(text="sculpt / repaint). Count changes, drop")
            c.label(text="and re-skin aren't supported here.")
        l.label(text="Edit imported meshes (Edit Mode: add /")
        l.label(text="delete / move verts), then:")
        l.prop(ds, "wd_reskin_weights")
        l.prop(ds, "wd_recalculate_normals")
        if ds.wd_recalculate_normals:
            note = l.column(align=True)
            note.scale_y = 0.8
            note.label(text="Normal + tangent + binormal all rebuilt", icon='INFO')
            note.label(text="from geometry + UVs (MikkTSpace).")
            note.label(text="Leave OFF for round-trip fidelity.")
        l.separator()
        sync_row = l.row()
        sync_row.enabled = any(o.get('wd_src') for o in ctx.selected_objects)
        sync_row.operator("xbg.wd_sync_normals",
                          text="Sync Normals from Geometry", icon='NORMALS_FACE')
        l.separator()
        r = l.row()
        r.scale_y = 1.8
        r.enabled = any(o.get('wd_src') for o in ctx.scene.objects)
        r.operator("xbg.inject_wd_model",
                   text="   Inject WD1 Mesh (.xbg)", icon='EXPORT')
        col = l.column(align=True)
        col.scale_y = 0.8
        nsel = len([o for o in ctx.selected_objects if o.get('wd_src')])
        if nsel:
            col.label(text="SELECTION = keep-list: only selected", icon='RESTRICT_SELECT_OFF')
            col.label(text="meshes are written; unselected ones are")
            col.label(text="DROPPED (smaller file — e.g. select all")
            col.label(text="but the eyes for an eyeless head).")
        else:
            col.label(text="Nothing selected → ALL meshes written.", icon='INFO')
            col.label(text="Select a subset to drop the rest.")
        col.separator()
        col.label(text="New object (2nd head)? Join it (Ctrl+J)", icon='INFO')
        col.label(text="into an imported mesh first — a loose")
        col.label(text="object has no place in the file.")


# ── Advanced Settings / Debug ───────────────────────────────────────────────

class XBG_PT_WDDebug(bpy.types.Panel):
    bl_label = "Advanced Settings"
    bl_idname = "OBJECT_PT_xbg_wd_debug"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "XBG Import"
    bl_parent_id = "OBJECT_PT_xbg_wd"
    bl_options = {'DEFAULT_CLOSED'}

    @classmethod
    def poll(cls, ctx):
        return _adv(ctx)

    def draw_header(self, ctx):
        self.layout.label(icon='SETTINGS')

    def draw(self, ctx):
        l, ds = self.layout, ctx.scene.xbg_debug_settings
        b = l.box()
        b.label(text="Logging:", icon='CONSOLE')
        b.prop(ds, "verbose_logging", text="Verbose Console Output")
        if ds.verbose_logging:
            row = b.row()
            row.alert = ds.trace_logging
            row.prop(ds, "trace_logging",
                     text="Trace-Level Logging (per-vertex / per-byte)",
                     icon='RECORD_ON' if ds.trace_logging else 'NONE')
            row = b.row(align=True)
            row.operator("xbg.save_log",  text="Save Log to File", icon='TEXT')
            row.operator("xbg.reset_log", text="Reset Session",    icon='TRASH')


# ── Model Info ──────────────────────────────────────────────────────────────

class XBG_PT_WDModelInfo(bpy.types.Panel):
    """What the importer captured on the active mesh."""
    bl_label = "Model Info"
    bl_idname = "OBJECT_PT_xbg_wd_info"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "XBG Import"
    bl_parent_id = "OBJECT_PT_xbg_wd"
    bl_options = {'DEFAULT_CLOSED'}

    @classmethod
    def poll(cls, ctx):
        o = ctx.active_object
        return o is not None and o.type == 'MESH'

    def draw_header(self, ctx):
        self.layout.label(icon='INFO')

    def draw(self, ctx):
        l = self.layout
        o = ctx.active_object
        me = o.data

        box = l.box()
        col = box.column(align=True)
        col.label(text=o.name, icon='MESH_DATA')
        col.label(text=f"{len(me.vertices)} verts · {len(me.polygons)} tris")

        # Armature / skinning
        arm = next((m.object for m in o.modifiers
                    if m.type == 'ARMATURE' and m.object), None)
        if arm:
            col.label(text=f"Skeleton: {arm.name} "
                           f"({len(arm.data.bones)} bones)", icon='ARMATURE_DATA')
        if o.vertex_groups:
            col.label(text=f"{len(o.vertex_groups)} weighted bone groups",
                      icon='GROUP_VERTEX')

        # UVs
        if me.uv_layers:
            col.label(text="UV: " + ", ".join(uv.name for uv in me.uv_layers),
                      icon='UV')

        # Captured vertex components (importer-written attributes)
        attrs = me.attributes
        comps = []
        if 'Col' in attrs:
            comps.append("vertex colors")
        if 'xbg_normal' in attrs:
            comps.append("normals")
        if 'xbg_tangent' in attrs:
            comps.append("tangents")
        if 'xbg_binormal' in attrs:
            comps.append("binormals")
        if comps:
            col.label(text="Captured: " + ", ".join(comps), icon='CHECKMARK')
