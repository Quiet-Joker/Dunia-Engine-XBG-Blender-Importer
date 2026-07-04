"""Far Cry 1 — operators."""
import bpy

from .blender_pipeline_fc1 import import_fc1_cgf
from .import_cgf_fc1 import FCGFParseError


class XBG_OT_ImportFC1(bpy.types.Operator):
    """Import a Far Cry 1 (PC, CryEngine 1) .cgf model into Blender.

    Full geometry: position, normals, UVs, and per-face materials (resolved
    via the file's Node -> MTL_MULTI -> per-face-MatID chain). If the
    "FCData Folder" addon preference is set to the game's FCData directory,
    diffuse textures (.dds) are looked up and hooked up automatically.
    """
    bl_idname = "import_scene.cgf_model_fc1"
    bl_label = "Import Far Cry 1 Model"
    bl_description = "Import a Far Cry 1 .cgf (geometry + UVs + per-face materials)"
    bl_options = {'REGISTER', 'UNDO'}

    filepath: bpy.props.StringProperty(subtype="FILE_PATH")
    filter_glob: bpy.props.StringProperty(default="*.cgf", options={'HIDDEN'})

    def invoke(self, ctx, ev):
        ctx.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}

    def execute(self, ctx):
        try:
            objs = import_fc1_cgf(ctx, self.filepath)
        except FCGFParseError as exc:
            self.report({'ERROR'}, f"Far Cry 1 import failed: {exc}")
            return {'CANCELLED'}
        except Exception as exc:
            import traceback; traceback.print_exc()
            self.report({'ERROR'}, f"Far Cry 1 import failed: {exc}")
            return {'CANCELLED'}
        n_verts = sum(o['xbg_fc1_data']['vertex_count'] for o in objs)
        n_tris = sum(o['xbg_fc1_data']['triangle_count'] for o in objs)
        self.report({'INFO'},
                     f"Imported {len(objs)} object(s), {n_verts} verts / {n_tris} tris")
        return {'FINISHED'}
