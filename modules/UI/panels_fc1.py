"""Far Cry 1 — UI panels (children of OBJECT_PT_xbg_fc1 in main.py).

Full geometry import: position, normals, UVs, per-face materials -- the
format is documented from the leaked engine source, not reverse-engineered
(see modules/Far_Cry_1/import_cgf_fc1.py).
"""
import os

import bpy

from ..Core.prefs import get_prefs


class XBG_PT_FC1Import(bpy.types.Panel):
    """Import a CGF file into Blender."""
    bl_label = "Import CGF"
    bl_idname = "OBJECT_PT_xbg_fc1_import"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "XBG Import"
    bl_parent_id = "OBJECT_PT_xbg_fc1"

    def draw_header(self, ctx):
        self.layout.label(icon='IMPORT')

    def draw(self, ctx):
        l = self.layout
        prefs = get_prefs(ctx)

        # Game data folder (same layout as the Avatar Import panel)
        col = l.column(align=True)
        col.label(text="Game Data Folder (FCData):", icon='FILE_FOLDER')
        col.prop(prefs, "fc1_data_folder", text="")
        if not prefs.fc1_data_folder:
            col.label(text="Set this to load textures automatically",
                      icon='INFO')
        else:
            col.label(text="Whole-tree texture search enabled",
                      icon='CHECKMARK')

        l.separator()

        r = l.row()
        r.scale_y = 1.8
        r.operator("import_scene.cgf_model_fc1",
                   text="   Import Far Cry 1 Model (.cgf)", icon='IMPORT')

        note = l.column(align=True)
        note.scale_y = 0.8
        note.label(text="Geometry + UVs + per-face materials.", icon='CHECKMARK')
        note.label(text="No skeleton/animation import yet.")


class XBG_PT_FC1ModelInfo(bpy.types.Panel):
    """Inspector for the active imported FC1 model object."""
    bl_label = "Model Info"
    bl_idname = "OBJECT_PT_xbg_fc1_info"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "XBG Import"
    bl_parent_id = "OBJECT_PT_xbg_fc1"
    bl_options = {'DEFAULT_CLOSED'}

    @classmethod
    def poll(cls, ctx):
        o = ctx.active_object
        return o is not None and o.type == 'MESH' and o.get('xbg_fc1_data')

    def draw_header(self, ctx):
        self.layout.label(icon='INFO')

    def draw(self, ctx):
        l = self.layout
        meta = ctx.active_object['xbg_fc1_data']
        get = meta.get if hasattr(meta, 'get') else (lambda k, d=None: d)

        box = l.box()
        col = box.column(align=True)
        src = get('filepath', '')
        if src:
            col.label(text=os.path.basename(src), icon='FILE')
        node_name = get('node_name', '')
        if node_name:
            col.label(text=node_name, icon='OUTLINER_OB_MESH')
        col.label(text=f"{get('vertex_count', '?')} verts  ·  "
                       f"{get('triangle_count', '?')} tris")
        col.label(text=f"{get('material_count', 0)} material(s)")
