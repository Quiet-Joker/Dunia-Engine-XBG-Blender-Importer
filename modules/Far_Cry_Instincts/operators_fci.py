"""Far Cry Instincts — operators."""
import bpy

from .blender_pipeline_fci import import_fci_xbg
from .import_xbg_fci import FCIParseError


class XBG_OT_ImportFCI(bpy.types.Operator):
    """Import a Far Cry Instincts (Xbox, 2005) .xbg model into Blender.

    Geometry + UVs + per-submesh materials with auto-decoded .xbt textures
    (no normals or skeleton yet). If the "Extracted Archive Folder" addon
    preference is set to an fci_extract.py output tree, textures are looked
    up across the whole tree; otherwise they're searched next to the .xbg.
    """
    bl_idname = "import_scene.xbg_model_fci"
    bl_label = "Import Far Cry Instincts Model"
    bl_description = (
        "Import a Far Cry Instincts .xbg (geometry + UVs + materials with "
        "auto-decoded .xbt textures; no normals/skeleton yet)"
    )
    bl_options = {'REGISTER', 'UNDO'}

    filepath: bpy.props.StringProperty(subtype="FILE_PATH")
    filter_glob: bpy.props.StringProperty(default="*.xbg", options={'HIDDEN'})

    def invoke(self, ctx, ev):
        ctx.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}

    def execute(self, ctx):
        try:
            obj = import_fci_xbg(ctx, self.filepath)
        except FCIParseError as exc:
            self.report({'ERROR'}, f"Far Cry Instincts import failed: {exc}")
            return {'CANCELLED'}
        except Exception as exc:
            import traceback; traceback.print_exc()
            self.report({'ERROR'}, f"Far Cry Instincts import failed: {exc}")
            return {'CANCELLED'}
        meta = obj['xbg_fci_data']
        self.report({'INFO'},
                     f"Imported {meta['vertex_count']} verts / "
                     f"{meta['triangle_count']} tris")
        return {'FINISHED'}
