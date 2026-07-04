"""Far Cry Primal — operators (import / inject).

Self-contained: every import below resolves inside the Far_Cry_Primal
folder or shared infra — no cross-game imports.

No MAB operator yet: Primal animation is the v3 / 64-bit Dunia codec
(see agents.md Part 9 cross-game intel) — FC4's Disrupt 'aNi' decoder
does not apply.
"""
import os

import bpy

from ..Core.debug import VerboseLogger
from ..Core.prefs import get_prefs


class XBG_OT_ImportPrimal(bpy.types.Operator):
    """Import a Far Cry Primal GEOM .xbg model into Blender."""
    bl_idname  = "import_scene.xbg_model_primal"
    bl_label   = "Import Primal Model"
    bl_description = (
        "Import a Far Cry Primal GEOM .xbg (version 0x0006003A).  Builds the "
        "requested LOD's meshes + EDON-decoded armature.  Turn on Separate "
        "Primitives to enable injecting edits back into the source file."
    )
    bl_options = {'REGISTER', 'UNDO'}

    filepath: bpy.props.StringProperty(subtype="FILE_PATH")
    filter_glob: bpy.props.StringProperty(default="*.xbg", options={'HIDDEN'})

    lod_level: bpy.props.IntProperty(
        name="LOD",
        description="Which LOD to import (0 = highest detail)",
        default=0, min=0, max=10)

    def invoke(self, ctx, ev):
        ctx.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}

    def execute(self, ctx):
        from .blender_pipeline_primal import _load_fc3_or_fc4, detect_fc_version
        from .import_xbg_primal import VERSION_PRIMAL
        ds = ctx.scene.xbg_debug_settings
        VerboseLogger.enabled = ds.verbose_logging
        VerboseLogger.clear()
        VerboseLogger.session_marker("import_primal", file=self.filepath,
                                     lod=self.lod_level)

        version = detect_fc_version(self.filepath)
        if version != VERSION_PRIMAL:
            self.report({'ERROR'},
                "Not a Far Cry Primal GEOM .xbg "
                f"(version marker 0x{(version or 0):08x}).")
            return {'CANCELLED'}
        separate = getattr(ds, 'separate_primitives', False)
        lhd      = getattr(ds, 'load_hidden', True)
        try:
            _load_fc3_or_fc4(ctx, self.filepath, version, "Far Cry Primal",
                             self.lod_level, separate, lhd)
        except Exception as exc:
            import traceback; traceback.print_exc()
            self.report({'ERROR'}, f"Far Cry Primal import failed: {exc}")
            return {'CANCELLED'}
        VerboseLogger.autosave_sidecar(self.filepath)
        self.report({'INFO'},
                    f"Far Cry Primal model imported (LOD {self.lod_level})")
        return {'FINISHED'}


class XBG_OT_InjectPrimal(bpy.types.Operator):
    """Inject Primal section objects back into the source XBG file."""
    bl_idname  = "xbg.inject_primal"
    bl_label   = "Inject Primal Mesh"
    bl_description = (
        "Write selected Primal section objects back into a copy of the source "
        ".xbg.  Import with Separate Primitives ON, then reshape / sculpt, edit "
        "normals, UVs, vertex colours - or add / delete geometry. Same vertex "
        "count patches in place; a changed count rebuilds the buffers and "
        "re-binds bone weights via the SULC palette."
    )
    bl_options = {'REGISTER', 'UNDO'}

    filepath: bpy.props.StringProperty(subtype="FILE_PATH", default="output.xbg")
    filter_glob: bpy.props.StringProperty(default="*.xbg", options={'HIDDEN'})

    lod_level: bpy.props.IntProperty(
        name="Target LOD",
        description="Which LOD slot to replace (0 = highest detail)",
        default=0, min=0, max=10)

    def invoke(self, ctx, ev):
        obj = ctx.active_object
        if obj and obj.get('xbg_fc3_data'):
            meta = obj['xbg_fc3_data']
            src  = meta.get('filepath', '') if hasattr(meta, 'get') else ''
            if src:
                base, ext = os.path.splitext(src)
                self.filepath = base + "_injected" + ext
        ctx.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}

    def execute(self, ctx):
        from .inject_xbg_primal import inject_primal
        ds = ctx.scene.xbg_debug_settings
        VerboseLogger.enabled = ds.verbose_logging
        VerboseLogger.clear()
        VerboseLogger.session_marker("inject_primal",
                                     output_file=self.filepath,
                                     target_lod=self.lod_level)

        objects = [o for o in ctx.selected_objects
                   if o.type == 'MESH' and o.get('xbg_fc3_data')]
        if not objects:
            obj = ctx.active_object
            if obj and obj.type == 'MESH' and obj.get('xbg_fc3_data'):
                objects = [obj]
        if not objects:
            self.report({'ERROR'},
                "No Primal section objects selected. "
                "Import with Separate Primitives ON first.")
            return {'CANCELLED'}

        st, msg = inject_primal(ctx, objects, self.filepath, self.lod_level)
        VerboseLogger.autosave_sidecar(self.filepath)
        if st == {'FINISHED'}:
            self.report({'INFO'}, msg)
        else:
            self.report({'ERROR'}, msg)
        return st
