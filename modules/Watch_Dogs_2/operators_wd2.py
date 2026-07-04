"""Watch Dogs 2 — operators (import + .glm export; self-contained)."""
import os

import bpy

from .import_wd2 import load_wd2_model


class XBG_OT_ImportWD2(bpy.types.Operator):
    """Import a Watch Dogs 2 .glm (raw text GEOM source) model: skeleton,
    meshes, UVs, normals, skin weights."""
    bl_idname  = "xbg.import_wd2_model"
    bl_label   = "Import Watch Dogs 2 Model"
    bl_description = (
        "Import a Watch Dogs 2 .glm (raw text GEOM source) model with skeleton "
        "and skin weights"
    )
    bl_options = {'REGISTER', 'UNDO'}

    filepath: bpy.props.StringProperty(subtype="FILE_PATH")
    files: bpy.props.CollectionProperty(type=bpy.types.OperatorFileListElement)
    directory: bpy.props.StringProperty(subtype="DIR_PATH")
    filter_glob: bpy.props.StringProperty(default="*.glm", options={'HIDDEN'})

    def invoke(self, ctx, ev):
        ctx.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}

    def execute(self, ctx):
        fs = []
        if self.files:
            for f in self.files:
                if f.name.lower().endswith('.glm'):
                    fs.append(os.path.join(self.directory, f.name))
        elif self.filepath:
            fs.append(self.filepath)
        if not fs:
            self.report({'ERROR'}, "No .glm file selected")
            return {'CANCELLED'}

        sep = bool(getattr(ctx.scene.xbg_debug_settings,
                           'separate_primitives', True))
        ok = 0
        for fp in fs:
            try:
                model, arm = load_wd2_model(ctx, fp, separate_primitives=sep)
                nv = sum(len(m['verts']) for m in model['meshes'])
                self.report({'INFO'},
                    f"{os.path.basename(fp)}: WD2 — "
                    f"{len(model['bones'])} bones, "
                    f"{len(model['meshes'])} meshes, {nv} verts")
                ok += 1
            except Exception as exc:
                self.report({'ERROR'}, f"{os.path.basename(fp)}: {exc}")
                import traceback; traceback.print_exc()
        return {'FINISHED'} if ok else {'CANCELLED'}


class XBG_OT_ExportWD2(bpy.types.Operator):
    """Export edited WD2 meshes back into a copy of the source .glm."""
    bl_idname  = "xbg.export_wd2_model"
    bl_label   = "Export Watch Dogs 2 Model"
    bl_description = (
        "Write the selected WD2-imported meshes back into a copy of the "
        "source .glm (text format - vertex/face count changes fully "
        "supported; materials/skeleton/physics blocks kept byte-for-byte). "
        "Compile the result with your GLM2XBG converter to get a game .xbg"
    )
    bl_options = {'REGISTER'}

    filepath: bpy.props.StringProperty(subtype="FILE_PATH")
    filter_glob: bpy.props.StringProperty(default="*.glm", options={'HIDDEN'})
    check_existing: bpy.props.BoolProperty(default=True, options={'HIDDEN'})

    def invoke(self, ctx, ev):
        objs = [o for o in ctx.selected_objects if o.get('wd2_src')]
        if not objs:
            objs = [o for o in ctx.scene.objects if o.get('wd2_src')]
        if objs and not self.filepath:
            src = objs[0]['wd2_src']
            base, ext = os.path.splitext(src)
            self.filepath = base + "_edited" + ext
        ctx.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}

    def execute(self, ctx):
        from .export_wd2 import export_wd2_glm
        objs = [o for o in ctx.selected_objects if o.get('wd2_src')]
        if not objs:
            objs = [o for o in ctx.scene.objects if o.get('wd2_src')]
        try:
            n_meshes, n_verts, warns = export_wd2_glm(ctx, objs, self.filepath)
        except Exception as exc:
            self.report({'ERROR'}, f"WD2 export failed: {exc}")
            import traceback; traceback.print_exc()
            return {'CANCELLED'}
        for w in warns:
            self.report({'WARNING'}, w)
        self.report({'INFO'},
            f"WD2 export: {n_meshes} meshes, {n_verts} verts -> "
            f"{os.path.basename(self.filepath)}")
        return {'FINISHED'}
