"""Far Cry 5 / New Dawn — operators (import).

Self-contained: every import below resolves inside the Far_Cry_5 folder or
shared infra — no cross-game imports.
"""
import os

import bpy

from ..Core.debug import VerboseLogger
from ..Core.prefs import get_prefs


class XBG_OT_ImportFC5(bpy.types.Operator):
    """Import a Far Cry 5 / New Dawn GEOM .xbg model into Blender."""
    bl_idname  = "import_scene.xbg_model_fc5"
    bl_label   = "Import FC5 Model"
    bl_description = (
        "Import a Far Cry 5 / New Dawn GEOM .xbg (version 0x000D0047).  Builds "
        "the highest-detail LOD's meshes (positions / UVs / R10G10B10A2 "
        "normals+tangents / skin weights) + EDON-decoded armature.  Turn on "
        "Separate Primitives to enable injecting edits back into the source "
        "file.  Multi-LOD stepping is still being refined (LOD 0 works)."
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
        from .blender_pipeline_fc5 import (_load_fc3_or_fc4, detect_fc_version,
                                           _VERSION_FC5)
        ds = ctx.scene.xbg_debug_settings
        VerboseLogger.enabled = ds.verbose_logging
        VerboseLogger.clear()
        VerboseLogger.session_marker("import_fc5", file=self.filepath,
                                     lod=self.lod_level)

        version = detect_fc_version(self.filepath)
        if version != _VERSION_FC5:
            self.report({'ERROR'},
                "Not a Far Cry 5 / New Dawn GEOM .xbg "
                f"(version marker 0x{(version or 0):08x}).")
            return {'CANCELLED'}
        separate = getattr(ds, 'separate_primitives', False)
        lhd      = getattr(ds, 'load_hidden', True)
        try:
            _load_fc3_or_fc4(ctx, self.filepath, version, "Far Cry 5",
                             self.lod_level, separate, lhd)
        except Exception as exc:
            import traceback; traceback.print_exc()
            self.report({'ERROR'}, f"Far Cry 5 import failed: {exc}")
            return {'CANCELLED'}
        VerboseLogger.autosave_sidecar(self.filepath)
        self.report({'INFO'}, f"Far Cry 5 model imported (LOD {self.lod_level})")
        return {'FINISHED'}


class XBG_OT_InjectFC5(bpy.types.Operator):
    """Inject FC5 section objects back into the source XBG file."""
    bl_idname  = "xbg.inject_fc5"
    bl_label   = "Inject FC5 Mesh"
    bl_description = (
        "Write selected FC5 section objects back into a copy of the source "
        ".xbg.  Import with Separate Primitives ON, then reshape / sculpt, "
        "edit normals, UVs (both channels), vertex colours.  Same vertex "
        "count only (in place — weights / handedness / unknowns kept "
        "byte-for-byte); adding or removing geometry is not supported for "
        "FC5 yet"
    )
    bl_options = {'REGISTER', 'UNDO'}

    filepath: bpy.props.StringProperty(subtype="FILE_PATH",
                                        default="output.xbg")
    filter_glob: bpy.props.StringProperty(default="*.xbg", options={'HIDDEN'})

    lod_level: bpy.props.IntProperty(
        name="Target LOD",
        description="Which LOD slot to patch (0 = highest detail)",
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
        from .inject_xbg_fc5 import inject_fc5
        ds = ctx.scene.xbg_debug_settings
        VerboseLogger.enabled = ds.verbose_logging
        VerboseLogger.clear()
        VerboseLogger.session_marker("inject_fc5", output_file=self.filepath,
                                     target_lod=self.lod_level)

        objects = [o for o in ctx.selected_objects
                   if o.type == 'MESH' and o.get('xbg_fc3_data')]
        if not objects:
            obj = ctx.active_object
            if obj and obj.type == 'MESH' and obj.get('xbg_fc3_data'):
                objects = [obj]
        if not objects:
            self.report({'ERROR'},
                "No FC5 section objects selected. "
                "Import with Separate Primitives ON first.")
            return {'CANCELLED'}

        st, msg = inject_fc5(ctx, objects, self.filepath, self.lod_level)
        VerboseLogger.autosave_sidecar(self.filepath)
        if st == {'FINISHED'}:
            self.report({'INFO'}, msg)
        else:
            self.report({'ERROR'}, msg)
        return st


class XBG_OT_ImportFC5Mab(bpy.types.Operator):
    """Import a Far Cry 5 / New Dawn .mab animation onto an imported FC5 armature."""
    bl_idname  = "xbg.import_fc5_mab"
    bl_label   = "Import FC5 MAB"
    bl_description = (
        "Select an imported Far Cry 5 armature, then pick a .mab. Decodes the "
        "constant-pose bones and the compressed per-keyframe rotation bitstream "
        "(Dunia smallest-three / interpolant codec) and keys every animated bone "
        "matched by name hash"
    )
    bl_options = {'REGISTER', 'UNDO'}

    filepath: bpy.props.StringProperty(subtype="FILE_PATH")
    filter_glob: bpy.props.StringProperty(default="*.mab", options={'HIDDEN'})

    def invoke(self, ctx, ev):
        ctx.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}

    def execute(self, ctx):
        from .import_mab_fc5 import (parse_fc5_mab, apply_fc5_mab,
                                     fc5_bones_to_model_bones,
                                     build_fc5_prop_rigs,
                                     apply_fc5_root_location)
        from .import_xbg_fc5 import parse_xbg
        import zlib
        arm = ctx.active_object
        if arm is None or arm.type != 'ARMATURE':
            arm = next((o for o in ctx.selected_objects
                        if o.type == 'ARMATURE'), None)
        if arm is None:
            self.report({'ERROR'}, "Select an imported FC5 armature first")
            return {'CANCELLED'}
        _ds = ctx.scene.xbg_debug_settings
        # Bone source: an explicit skeleton override (.xbg) wins; otherwise the
        # .xbg the active armature was imported from.
        override = (getattr(_ds, 'mab_skeleton_override', '') or '').strip()
        if override:
            override = bpy.path.abspath(override)
            if not os.path.isfile(override):
                self.report({'ERROR'},
                    "Skeleton override is set but not a file: %s" % override)
                return {'CANCELLED'}
            src = override
        else:
            src = arm.get('xbg_source_file', '')
            if not src or not os.path.isfile(src):
                self.report({'ERROR'},
                    "Armature has no source .xbg recorded - re-import the FC5 "
                    "model, or set a Skeleton (.xbg) override in the panel")
                return {'CANCELLED'}
        try:
            mab = parse_fc5_mab(self.filepath)
            model = parse_xbg(src)
            model_bones = fc5_bones_to_model_bones(model['bones'])
            applied, missing = apply_fc5_mab(
                ctx, mab, arm, model_bones,
                smooth_resample=getattr(_ds, 'mab_smooth_resample', True),
                resample_fps=getattr(_ds, 'mab_resample_fps', 60),
                emulate_helpers=getattr(_ds, 'mab_emulate_helpers', False),
                twist_bake=getattr(_ds, 'mab_twist_bake', False))
            msg = (f"FC5 MAB: {applied} bones keyed "
                   f"({mab['n_animated']} animated + "
                   f"{len(mab['const_rots'])} constant), "
                   f"{len(mab['key_times'])} keyframes")
            if mab.get('root_motion'):
                nloc = apply_fc5_root_location(
                    ctx, arm, mab, model_bones,
                    smooth_resample=getattr(_ds, 'mab_smooth_resample', True),
                    resample_fps=getattr(_ds, 'mab_resample_fps', 60))
                msg += "  +root motion" + (f" ({nloc} loc keys)" if nloc else "")
            # Secondary prop/anchor blocks (heal bandage, placed mine, …) → their
            # own placeholder armatures (rotation-faithful; bind pose needs the
            # prop's own .xbg).  Resolve bone names against the rig where possible.
            prop_objs = []
            prop_blocks = [b for b in mab.get('props', [])
                           if b['n_bones'] >= 2
                           or b.get('loc_curves', {}).get(0)
                           or b['rot_curves'].get(0)]
            if prop_blocks:
                known = {zlib.crc32(b['name'].encode('latin-1')) & 0xFFFFFFFF:
                         b['name'] for b in model_bones}
                base = os.path.splitext(os.path.basename(self.filepath))[0]
                try:
                    prop_objs = build_fc5_prop_rigs(
                        ctx, mab, base, parent_arm=arm,
                        known_names=known, log=print)
                except Exception as pe:
                    import traceback; traceback.print_exc()
                    print("[FC5 MAB] prop-rig build skipped: %s" % pe)
                if prop_objs:
                    msg += f"  +{len(prop_objs)} prop rig(s)"
            if missing:
                msg += f"  ({len(missing)} bone hashes not on this rig)"
            self.report({'WARNING' if missing else 'INFO'}, msg)
            return {'FINISHED'}
        except Exception as exc:
            self.report({'ERROR'}, f"Failed to import FC5 .mab: {exc}")
            import traceback; traceback.print_exc()
            return {'CANCELLED'}
