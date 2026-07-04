"""Far Cry 4 — operators (import / inject / animation).

Self-contained: every import below resolves inside the farcry4 folder or
shared infra — no cross-game imports.
"""
import os

import bpy

from ..Core.debug import VerboseLogger
from ..Core.prefs import get_prefs


class XBG_OT_ImportFC4(bpy.types.Operator):
    """Import a Far Cry 4 GEOM .xbg model into Blender."""
    bl_idname  = "import_scene.xbg_model_fc4"
    bl_label   = "Import FC4 Model"
    bl_description = (
        "Import a Far Cry 4 GEOM .xbg.  Builds the requested LOD's meshes + "
        "EDON-decoded armature.  Turn on Separate Primitives to enable "
        "injecting edits back into the source file."
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
        from .blender_pipeline_fc4 import (_load_fc3_or_fc4, detect_fc_version,
                                           _VERSION_FC4)
        ds = ctx.scene.xbg_debug_settings
        VerboseLogger.enabled = ds.verbose_logging
        VerboseLogger.clear()
        VerboseLogger.session_marker("import_fc4", file=self.filepath,
                                     lod=self.lod_level)

        version = detect_fc_version(self.filepath)
        if version != _VERSION_FC4:
            self.report({'ERROR'},
                "Not a Far Cry 4 GEOM .xbg "
                f"(version marker 0x{(version or 0):08x}).")
            return {'CANCELLED'}
        separate = getattr(ds, 'separate_primitives', False)
        lhd      = getattr(ds, 'load_hidden', True)
        try:
            _load_fc3_or_fc4(ctx, self.filepath, version, "Far Cry 4",
                             self.lod_level, separate, lhd)
        except Exception as exc:
            import traceback; traceback.print_exc()
            self.report({'ERROR'}, f"Far Cry 4 import failed: {exc}")
            return {'CANCELLED'}
        VerboseLogger.autosave_sidecar(self.filepath)
        self.report({'INFO'}, f"Far Cry 4 model imported (LOD {self.lod_level})")
        return {'FINISHED'}


class XBG_OT_InjectFC4(bpy.types.Operator):
    """Inject FC4 section objects back into the source XBG file."""
    bl_idname  = "xbg.inject_fc4"
    bl_label   = "Inject FC4 Mesh"
    bl_description = (
        "Write selected FC4 section objects back into a copy of the source "
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
        from .inject_xbg_fc4 import inject_fc4
        ds = ctx.scene.xbg_debug_settings
        VerboseLogger.enabled = ds.verbose_logging
        VerboseLogger.clear()
        VerboseLogger.session_marker("inject_fc4", output_file=self.filepath,
                                     target_lod=self.lod_level)

        objects = [o for o in ctx.selected_objects
                   if o.type == 'MESH' and o.get('xbg_fc3_data')]
        if not objects:
            obj = ctx.active_object
            if obj and obj.type == 'MESH' and obj.get('xbg_fc3_data'):
                objects = [obj]
        if not objects:
            self.report({'ERROR'},
                "No FC4 section objects selected. "
                "Import with Separate Primitives ON first.")
            return {'CANCELLED'}

        st, msg = inject_fc4(ctx, objects, self.filepath, self.lod_level)
        VerboseLogger.autosave_sidecar(self.filepath)
        if st == {'FINISHED'}:
            self.report({'INFO'}, msg)
        else:
            self.report({'ERROR'}, msg)
        return st


class XBG_OT_ImportFC4Mab(bpy.types.Operator):
    """Import a Far Cry 4 .mab animation onto an imported FC4 armature."""
    bl_idname  = "xbg.import_fc4_mab"
    bl_label   = "Import FC4 MAB"
    bl_description = (
        "Select an imported Far Cry 4 armature, then pick a .mab. Decodes the "
        "constant-pose bones and the compressed per-keyframe rotation bitstream "
        "(Disrupt 'aNi' codec) and keys every animated bone matched by name hash"
    )
    bl_options = {'REGISTER', 'UNDO'}

    filepath: bpy.props.StringProperty(subtype="FILE_PATH")
    filter_glob: bpy.props.StringProperty(default="*.mab", options={'HIDDEN'})

    def invoke(self, ctx, ev):
        ctx.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}

    def execute(self, ctx):
        from .import_mab_fc4 import (parse_fc4_mab, apply_fc4_mab,
                                     fc3_bones_to_model_bones)
        from .import_xbg_fc4 import parse_xbg
        arm = ctx.active_object
        if arm is None or arm.type != 'ARMATURE':
            arm = next((o for o in ctx.selected_objects
                        if o.type == 'ARMATURE'), None)
        if arm is None:
            self.report({'ERROR'}, "Select an imported FC4 armature first")
            return {'CANCELLED'}
        src = arm.get('xbg_source_file', '')
        if not src or not os.path.isfile(src):
            self.report({'ERROR'},
                "Armature has no source .xbg recorded - re-import the FC4 model "
                "(the importer stores the path for MAB matching)")
            return {'CANCELLED'}
        try:
            mab = parse_fc4_mab(self.filepath)
            model = parse_xbg(src)
            model_bones = fc3_bones_to_model_bones(model['bones'])
            _ds = ctx.scene.xbg_debug_settings
            applied, missing = apply_fc4_mab(
                ctx, mab, arm, model_bones,
                smooth_resample=getattr(_ds, 'mab_smooth_resample', True),
                resample_fps=getattr(_ds, 'mab_resample_fps', 60),
                emulate_helpers=getattr(_ds, 'mab_emulate_helpers', True),
                twist_bake=getattr(_ds, 'mab_twist_bake', True))
            msg = (f"FC4 MAB: {applied} bones keyed "
                   f"({mab['n_animated']} animated + "
                   f"{len(mab['const_rots'])} constant), "
                   f"{len(mab['key_times'])} keyframes")
            if missing:
                msg += f"  ({len(missing)} bone hashes not on this rig)"
            self.report({'WARNING' if missing else 'INFO'}, msg)
            return {'FINISHED'}
        except Exception as exc:
            self.report({'ERROR'}, f"Failed to import FC4 .mab: {exc}")
            import traceback; traceback.print_exc()
            return {'CANCELLED'}
