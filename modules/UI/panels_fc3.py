"""Far Cry 3 / 4 — UI panels (children of OBJECT_PT_xbg_fc3 in main.py).

The FC3/FC4 round-trip workflow:
  1. Import the .xbg with Separate Primitives ON (one object per section —
     each carries its 'xbg_fc3_data' injection metadata).
  2. Edit the section meshes.
  3. Select them and Inject back into a copy of the source file.
"""

import os

import bpy

from .main import active_game


def _fc3_sections(ctx):
    """Selected mesh objects that carry FC3/FC4 injection metadata."""
    return [o for o in ctx.selected_objects
            if o.type == 'MESH' and o.get('xbg_fc3_data')]


class XBG_PT_FC3Import(bpy.types.Panel):
    """Import an XBG file into Blender."""
    bl_label = "Import XBG"
    bl_idname = "OBJECT_PT_xbg_fc3_import"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "XBG Import"
    bl_parent_id = "OBJECT_PT_xbg_fc3"

    def draw_header(self, ctx):
        self.layout.label(icon='IMPORT')

    def draw(self, ctx):
        l = self.layout
        ds = ctx.scene.xbg_debug_settings

        # Round-trip editing needs one object per section
        col = l.column(align=True)
        col.prop(ds, "separate_primitives", text="Separate Primitives",
                 toggle=True, icon='MOD_EXPLODE')
        if not ds.separate_primitives:
            col.label(text="Turn ON to allow injecting back", icon='ERROR')

        l.separator()

        r = l.row()
        r.scale_y = 1.8
        # FC3 / FC4 / FC5 / Primal have fully separate, self-contained
        # importers.  The UI layer dispatches by the picked game.
        g = active_game(ctx)
        if g == 'FC4':
            r.operator("import_scene.xbg_model_fc4",
                       text="   Import FC4 Model (.xbg)", icon='IMPORT')
        elif g == 'FC5':
            r.operator("import_scene.xbg_model_fc5",
                       text="   Import FC5 Model (.xbg)", icon='IMPORT')
        elif g == 'PRIMAL':
            r.operator("import_scene.xbg_model_primal",
                       text="   Import Primal Model (.xbg)", icon='IMPORT')
        else:
            r.operator("import_scene.xbg_model_fc3",
                       text="   Import FC3 Model (.xbg)", icon='IMPORT')

        # Companion files (skeleton FC3-only; HKX is 32-bit Havok, FC3+FC4).
        if g in ('FC3', 'FC4'):
            l.separator()
            if g == 'FC3':
                r2 = l.row()
                r2.scale_y = 1.3
                r2.operator("xbg.import_fc3_skeleton",
                            text="Import FC3 Skeleton (.skeleton)",
                            icon='ARMATURE_DATA')
            r2 = l.row()
            r2.scale_y = 1.3
            r2.operator("xbg.import_fc3_hkx",
                        text="Import %s HKX Collision (.hkx)" % g,
                        icon='MESH_ICOSPHERE')

        if not ds.advanced_mode:
            l.label(text="Enable Advanced Mode for editing / injection.",
                    icon='INFO')


class XBG_PT_FC4Animation(bpy.types.Panel):
    """FC3/FC4/FC5 .mab animation import."""
    bl_label = "Animation"
    bl_idname = "OBJECT_PT_xbg_fc4_anim"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "XBG Import"
    bl_parent_id = "OBJECT_PT_xbg_fc3"
    bl_options = {'DEFAULT_CLOSED'}

    @classmethod
    def poll(cls, ctx):
        return (ctx.scene.xbg_debug_settings.advanced_mode
                and active_game(ctx) in ('FC3', 'FC4', 'FC5'))

    def draw_header(self, ctx):
        self.layout.label(icon='ARMATURE_DATA')

    def draw(self, ctx):
        l = self.layout
        g = active_game(ctx)
        l.label(text="Animation (.mab) — select the %s armature:" % g,
                icon='ANIM_DATA')
        ds = ctx.scene.xbg_debug_settings
        if g == 'FC5':
            box = l.box()
            box.label(text="Skeleton override (optional):", icon='BONE_DATA')
            box.prop(ds, "mab_skeleton_override", text="")
            box.scale_y = 0.95
        l.prop(ds, "mab_emulate_helpers")
        if ds.mab_emulate_helpers:
            l.prop(ds, "mab_twist_bake")
        l.prop(ds, "mab_smooth_resample")
        if ds.mab_smooth_resample:
            l.prop(ds, "mab_resample_fps")
        r = l.row()
        r.scale_y = 1.4
        if g == 'FC5':
            r.operator("xbg.import_fc5_mab",
                       text="Import FC5 MAB", icon='ARMATURE_DATA')
        elif g == 'FC3':
            r.operator("xbg.import_fc3_mab",
                       text="Import FC3 MAB", icon='ARMATURE_DATA')
        else:
            r.operator("xbg.import_fc4_mab",
                       text="Import FC4 MAB", icon='ARMATURE_DATA')
        note = l.column(align=True)
        note.scale_y = 0.8
        if g == 'FC5':
            note.label(text="Dunia smallest-three / interpolant codec.",
                       icon='INFO')
        elif g == 'FC3':
            note.label(text="Dunia smallest-three codec (version 0x61).",
                       icon='INFO')
        else:
            note.label(text="Disrupt 'aNi' bitstream (shared with WD1).",
                       icon='INFO')
        note.label(text="Bones matched to the rig by name hash.")


class XBG_PT_FC3Sections(bpy.types.Panel):
    """Inspector for the active imported section object."""
    bl_label = "Active Section"
    bl_idname = "OBJECT_PT_xbg_fc3_sections"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "XBG Import"
    bl_parent_id = "OBJECT_PT_xbg_fc3"
    bl_options = {'DEFAULT_CLOSED'}

    @classmethod
    def poll(cls, ctx):
        o = ctx.active_object
        return o is not None and o.type == 'MESH' and o.get('xbg_fc3_data')

    def draw_header(self, ctx):
        self.layout.label(icon='INFO')

    def draw(self, ctx):
        l = self.layout
        meta = ctx.active_object['xbg_fc3_data']
        get = meta.get if hasattr(meta, 'get') else (lambda k, d=None: d)

        box = l.box()
        col = box.column(align=True)
        src = get('filepath', '')
        if src:
            col.label(text=os.path.basename(src), icon='FILE')
        col.label(text=f"LOD {get('lod', '?')}  ·  "
                       f"Section {get('section_index', '?')}  ·  "
                       f"VB {get('vb_index', '?')}")
        mat = get('mat_name', '')
        if mat:
            col.label(text=f"Material: {mat}", icon='MATERIAL')
        col.label(text=f"Stride {get('stride', '?')}  ·  "
                       f"Indices {get('idx_start', '?')}–{get('idx_end', '?')}")


class XBG_PT_FC3Inject(bpy.types.Panel):
    """Inject your edited section meshes back into the XBG file."""
    bl_label = "Inject / Export"
    bl_idname = "OBJECT_PT_xbg_fc3_inject"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "XBG Import"
    bl_parent_id = "OBJECT_PT_xbg_fc3"

    @classmethod
    def poll(cls, ctx):
        return ctx.scene.xbg_debug_settings.advanced_mode

    def draw_header(self, ctx):
        self.layout.label(icon='EXPORT')

    def draw(self, ctx):
        l = self.layout
        sections = _fc3_sections(ctx)

        if sections:
            srcs = {os.path.basename(o['xbg_fc3_data'].get('filepath', '?'))
                    if hasattr(o['xbg_fc3_data'], 'get') else '?'
                    for o in sections}
            info = l.box()
            info.label(text=f"{len(sections)} section object(s) selected",
                       icon='OBJECT_DATA')
            for s in sorted(srcs):
                info.label(text=f"Linked:  {s}", icon='LINKED')
            if len(srcs) > 1:
                warn = l.box()
                warn.alert = True
                warn.label(text="Sections come from different files!",
                           icon='ERROR')
        else:
            hint = l.box()
            hint.label(text="No section objects selected.", icon='INFO')
            hint.label(text="Import with Separate Primitives ON,")
            hint.label(text="then select the sections to write back.")

        l.separator()

        r = l.row()
        r.scale_y = 1.8
        r.enabled = bool(sections) or (
            ctx.active_object is not None
            and ctx.active_object.type == 'MESH'
            and bool(ctx.active_object.get('xbg_fc3_data')))
        g = active_game(ctx)
        if g == 'FC4':
            r.operator("xbg.inject_fc4",
                       text="   Inject into FC4 XBG", icon='EXPORT')
        elif g == 'FC5':
            r.operator("xbg.inject_fc5",
                       text="   Inject into FC5 XBG", icon='EXPORT')
        elif g == 'PRIMAL':
            r.operator("xbg.inject_primal",
                       text="   Inject into Primal XBG", icon='EXPORT')
        else:
            r.operator("xbg.inject_fc3",
                       text="   Inject into FC3 XBG", icon='EXPORT')

        note = l.column(align=True)
        note.scale_y = 0.8
        if g == 'FC5':
            note.label(text="Same count only: in-place patch", icon='INFO')
            note.label(text="(weights/handedness kept byte-for-byte).")
            note.label(text="Add/delete verts not supported for FC5 yet.")
        else:
            note.label(text="Same count: in-place (weights/tangents kept).",
                       icon='INFO')
            note.label(text="Add/delete verts: rebuild + weight re-bind")
            note.label(text="from vertex groups. Test rebuilds in-game.")
