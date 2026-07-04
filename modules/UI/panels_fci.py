"""Far Cry Instincts — UI panels (children of OBJECT_PT_xbg_fci in main.py).

Read-only import: geometry + UVs + per-submesh materials with auto-decoded
.xbt textures (no normals, skeleton, or export/inject yet -- see
modules/Far_Cry_Instincts).
"""
import os

import bpy

from ..Core.prefs import get_prefs


class XBG_PT_FCIImport(bpy.types.Panel):
    """Import an Instincts XBG file into Blender."""
    bl_label = "Import XBG"
    bl_idname = "OBJECT_PT_xbg_fci_import"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "XBG Import"
    bl_parent_id = "OBJECT_PT_xbg_fci"

    def draw_header(self, ctx):
        self.layout.label(icon='IMPORT')

    def draw(self, ctx):
        l = self.layout
        prefs = get_prefs(ctx)

        # Game data folder (same layout as the Avatar Import panel)
        col = l.column(align=True)
        col.label(text="Game Data Folder (extracted dump):",
                  icon='FILE_FOLDER')
        col.prop(prefs, "fci_data_folder", text="")
        if not prefs.fci_data_folder:
            col.label(text="Set this to load textures automatically",
                      icon='INFO')
            col.label(text="(empty: looks next to the .xbg only)")
        else:
            col.label(text="Whole-tree texture search enabled",
                      icon='CHECKMARK')

        l.separator()

        r = l.row()
        r.scale_y = 1.8
        r.operator("import_scene.xbg_model_fci",
                   text="   Import FCI Model (.xbg)", icon='IMPORT')

        note = l.column(align=True)
        note.scale_y = 0.8
        note.label(text="Geometry + UVs + textured materials.", icon='CHECKMARK')
        note.label(text="No normals/skeleton/export yet.")


class XBG_PT_FCIModelInfo(bpy.types.Panel):
    """Inspector for the active imported FCI model."""
    bl_label = "Model Info"
    bl_idname = "OBJECT_PT_xbg_fci_info"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "XBG Import"
    bl_parent_id = "OBJECT_PT_xbg_fci"
    bl_options = {'DEFAULT_CLOSED'}

    @classmethod
    def poll(cls, ctx):
        o = ctx.active_object
        return o is not None and o.type == 'MESH' and o.get('xbg_fci_data')

    def draw_header(self, ctx):
        self.layout.label(icon='INFO')

    def draw(self, ctx):
        l = self.layout
        meta = ctx.active_object['xbg_fci_data']
        get = meta.get if hasattr(meta, 'get') else (lambda k, d=None: d)

        box = l.box()
        col = box.column(align=True)
        src = get('filepath', '')
        if src:
            col.label(text=os.path.basename(src), icon='FILE')
        col.label(text=f"{get('vertex_count', '?')} verts  ·  "
                       f"{get('triangle_count', '?')} tris")
        col.label(text=f"Position scale: {get('scale', 0):.6f}")
        for p in get('texture_paths', []) or []:
            col.label(text=p, icon='TEXTURE')
