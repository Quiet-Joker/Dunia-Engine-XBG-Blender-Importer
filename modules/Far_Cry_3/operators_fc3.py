"""Far Cry 3 / 4 — operators.

Split out of the monolithic __init__.py (2026-06-09 refactor).
"""
import os

import bpy

from ..Core.debug import VerboseLogger
from ..Core.prefs import get_prefs


class XBG_OT_ImportFC3(bpy.types.Operator):
    """Import a Far Cry 3 / Far Cry 4 GEOM .xbg model into Blender.

    Dedicated FC3/FC4 entry point — parses + builds entirely within the
    farcry3 folder (blender_pipeline_fc3), so the Avatar importer no longer
    handles Far Cry files.
    """
    bl_idname  = "import_scene.xbg_model_fc3"
    bl_label   = "Import FC3 / FC4 Model"
    bl_description = (
        "Import a Far Cry 3 or Far Cry 4 GEOM .xbg.  Builds the requested LOD's "
        "meshes + EDON-decoded armature.  Turn on Separate Primitives to enable "
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
        from .blender_pipeline_fc3 import (_load_fc3_or_fc4, detect_fc_version,
                                           _VERSION_FC3)
        ds = ctx.scene.xbg_debug_settings
        VerboseLogger.enabled = ds.verbose_logging
        VerboseLogger.clear()
        VerboseLogger.session_marker("import_fc3", file=self.filepath,
                                     lod=self.lod_level)

        version = detect_fc_version(self.filepath)
        if version != _VERSION_FC3:
            self.report({'ERROR'},
                "Not a Far Cry 3 GEOM .xbg "
                f"(version marker 0x{(version or 0):08x}). "
                "Far Cry 4 files use the FC4 importer.")
            return {'CANCELLED'}
        separate = getattr(ds, 'separate_primitives', False)
        lhd      = getattr(ds, 'load_hidden', True)
        try:
            _load_fc3_or_fc4(ctx, self.filepath, version, "Far Cry 3",
                             self.lod_level, separate, lhd)
        except Exception as exc:
            import traceback; traceback.print_exc()
            self.report({'ERROR'}, f"Far Cry 3 import failed: {exc}")
            return {'CANCELLED'}
        VerboseLogger.autosave_sidecar(self.filepath)
        self.report({'INFO'}, f"Far Cry 3 model imported (LOD {self.lod_level})")
        return {'FINISHED'}


class XBG_OT_ImportFC3Skeleton(bpy.types.Operator):
    """Import a Far Cry 3 .skeleton (LKS) file as a standalone armature."""
    bl_idname  = "xbg.import_fc3_skeleton"
    bl_label   = "Import FC3 Skeleton"
    bl_description = (
        "Import a Far Cry 3 .skeleton (LKS) file as an armature.  Handles "
        "both character and compact prop skeleton layouts (crc-anchored "
        "parse).  FC4's name-less .skeleton variant is not supported — FC4 "
        "armatures come from the .xbg import"
    )
    bl_options = {'REGISTER', 'UNDO'}

    filepath: bpy.props.StringProperty(subtype="FILE_PATH")
    filter_glob: bpy.props.StringProperty(default="*.skeleton",
                                          options={'HIDDEN'})

    def invoke(self, ctx, ev):
        ctx.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}

    def execute(self, ctx):
        from .import_skeleton_fc3 import (parse_fc3_skeleton,
                                          build_fc3_skeleton_armature, LksError)
        try:
            bones = parse_fc3_skeleton(self.filepath)
            name = os.path.splitext(os.path.basename(self.filepath))[0]
            build_fc3_skeleton_armature(ctx, bones, name)
        except LksError as exc:
            self.report({'ERROR'}, f"Skeleton import failed: {exc}")
            return {'CANCELLED'}
        except Exception as exc:
            self.report({'ERROR'}, f"Skeleton import failed: {exc}")
            import traceback; traceback.print_exc()
            return {'CANCELLED'}
        self.report({'INFO'}, f"Skeleton imported: {len(bones)} bones")
        return {'FINISHED'}


class XBG_OT_ImportFC3Hkx(bpy.types.Operator):
    """Import a Far Cry 3 / Far Cry 4 .hkx collision file."""
    bl_idname  = "xbg.import_fc3_hkx"
    bl_label   = "Import FC3/FC4 HKX Collision"
    bl_description = (
        "Import a Far Cry 3 / Far Cry 4 .hkx (32-bit Havok 2010/2012) "
        "collision file: boxes / capsules / spheres / triangle meshes per "
        "rigid body, as wireframe objects"
    )
    bl_options = {'REGISTER', 'UNDO'}

    filepath: bpy.props.StringProperty(subtype="FILE_PATH")
    filter_glob: bpy.props.StringProperty(default="*.hkx", options={'HIDDEN'})

    def invoke(self, ctx, ev):
        ctx.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}

    def execute(self, ctx):
        from .import_hkx_fc3 import load_fc3_hkx, Hkx32Error
        try:
            objs = load_fc3_hkx(ctx, self.filepath)
        except Hkx32Error as exc:
            self.report({'ERROR'}, f"HKX import failed: {exc}")
            return {'CANCELLED'}
        except Exception as exc:
            self.report({'ERROR'}, f"HKX import failed: {exc}")
            import traceback; traceback.print_exc()
            return {'CANCELLED'}
        self.report({'INFO'}, f"HKX: {len(objs)} collision shape(s) imported")
        return {'FINISHED'}


def _fc3_xbg_path_for_armature(arm):
    """Return the source XBG file path for *arm*, or None.

    Lookup order:
      1. xbg_source_file property stored on the armature at import time
         (blender_pipeline_fc3 stamps this).
      2. xbg_fc3_data.filepath on any direct child mesh object.
    """
    p = arm.get("xbg_source_file", "")
    if p and os.path.isfile(p):
        return p
    for child in arm.children:
        xd = child.get("xbg_fc3_data")
        if xd is None:
            continue
        d = xd.to_dict() if hasattr(xd, 'to_dict') else (xd or {})
        fp = d.get("filepath") if isinstance(d, dict) else None
        if fp and os.path.isfile(fp):
            return fp
    return None


class XBG_OT_ImportFC3Mab(bpy.types.Operator):
    """Import a Far Cry 3 .mab animation onto the selected armature."""
    bl_idname  = "xbg.import_fc3_mab"
    bl_label   = "Import FC3 MAB Animation"
    bl_description = (
        "Select an FC3-imported armature first, then pick a .mab animation. "
        "FC3 mabs (version 0x61) use the Dunia smallest-three codec; "
        "keyframes are applied to the rig's pose bones"
    )
    bl_options = {'REGISTER', 'UNDO'}

    filepath: bpy.props.StringProperty(subtype="FILE_PATH")
    filter_glob: bpy.props.StringProperty(default="*.mab", options={'HIDDEN'})

    def invoke(self, ctx, ev):
        ctx.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}

    def execute(self, ctx):
        from .import_mab_fc3 import parse_sections, apply_multi_bone
        ds = ctx.scene.xbg_debug_settings
        VerboseLogger.enabled = True  # always verbose for the decode loop

        arm = ctx.active_object
        if arm is None or arm.type != 'ARMATURE':
            arm = next((o for o in ctx.selected_objects
                        if o.type == 'ARMATURE'), None)
        if arm is None:
            self.report({'ERROR'},
                        "Select an armature before importing a .mab")
            return {'CANCELLED'}
        if not self.filepath or not os.path.isfile(self.filepath):
            self.report({'ERROR'}, "No valid .mab file selected")
            return {'CANCELLED'}

        xbg_path = _fc3_xbg_path_for_armature(arm)
        if not xbg_path:
            self.report({'ERROR'},
                "Cannot find source XBG — select the armature created by "
                "the FC3 model import")
            return {'CANCELLED'}

        # Search the .mab's own folder and a few parents for a matching
        # .skeleton file besides the XBG's folder.
        mab_dir = os.path.dirname(os.path.abspath(self.filepath))
        extra = [mab_dir]
        for _ in range(3):
            mab_dir = os.path.dirname(mab_dir)
            extra.append(mab_dir)

        from .import_mab_fc3 import apply_single_bone
        try:
            d = open(self.filepath, 'rb').read()
            sec = parse_sections(d)
            try:
                n_keyed, animated = apply_multi_bone(
                    ctx, d, sec, arm, xbg_path=xbg_path,
                    skeleton_path=None, extra_dirs=extra,
                    emulate_helpers=ds.mab_emulate_helpers,
                    smooth_resample=ds.mab_smooth_resample,
                    resample_fps=ds.mab_resample_fps,
                    twist_bake=ds.mab_twist_bake)
            except Exception as multi_exc:
                # Small prop rigs (fans, doors, ...) often ship clips whose
                # routing masks don't validate against the tiny skeleton —
                # these are the SINGLE-bone case the codec was originally
                # validated on (ceiling fan). Apply the one animated track
                # to the rig's leaf bone.
                bones = arm.data.bones
                if len(bones) > 4:
                    raise multi_exc
                leaf = next((b for b in bones if not b.children), None)
                if leaf is None:
                    raise multi_exc
                print(f"[FC3 MAB] multi-bone routing failed ({multi_exc}); "
                      f"prop rig fallback: single-bone track -> '{leaf.name}'")
                apply_single_bone(
                    ctx, d, sec, arm, leaf.name, xbg_path=xbg_path,
                    smooth_resample=ds.mab_smooth_resample,
                    resample_fps=ds.mab_resample_fps)
                n_keyed, animated = 1, [leaf.name]
            self.report({'INFO'},
                f"MAB: {os.path.basename(self.filepath)}  "
                f"{n_keyed}/{len(animated)} bones / "
                f"{ctx.scene.frame_end} frames")
            return {'FINISHED'}
        except Exception as exc:
            self.report({'ERROR'}, f"Failed to import .mab: {exc}")
            import traceback; traceback.print_exc()
            return {'CANCELLED'}


class XBG_OT_InjectFC3(bpy.types.Operator):
    """Inject FC3/FC4 section objects back into the source XBG file."""
    bl_idname  = "xbg.inject_fc3"
    bl_label   = "Inject FC3 / FC4 Mesh"
    bl_description = (
        "Write selected FC3/FC4 section objects back into a copy of the source "
        ".xbg.  Import with Separate Primitives ON, then reshape / sculpt, edit "
        "normals, UVs (both channels), vertex colours — or add / delete geometry. "
        "Same vertex count patches in place (weights/tangents/binormals kept "
        "byte-for-byte); a changed count rebuilds the buffers and re-binds bone "
        "weights from the vertex groups via the SULC palette."
    )
    bl_options = {'REGISTER', 'UNDO'}

    filepath: bpy.props.StringProperty(subtype="FILE_PATH",
                                        default="output.xbg")
    filter_glob: bpy.props.StringProperty(default="*.xbg", options={'HIDDEN'})

    lod_level: bpy.props.IntProperty(
        name="Target LOD",
        description="Which LOD slot to replace (0 = highest detail)",
        default=0, min=0, max=10)

    def invoke(self, ctx, ev):
        # Pre-fill with source path of an active FC3 object
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
        from .inject_xbg_fc3 import inject_fc3
        ds = ctx.scene.xbg_debug_settings
        VerboseLogger.enabled = ds.verbose_logging
        VerboseLogger.clear()
        VerboseLogger.session_marker(
            "inject_fc3",
            output_file=self.filepath,
            target_lod=self.lod_level)

        # Collect FC3 mesh objects from selection (or active)
        objects = [o for o in ctx.selected_objects
                   if o.type == 'MESH' and o.get('xbg_fc3_data')]
        if not objects:
            obj = ctx.active_object
            if obj and obj.type == 'MESH' and obj.get('xbg_fc3_data'):
                objects = [obj]
        if not objects:
            self.report({'ERROR'},
                "No FC3/FC4 section objects selected. "
                "Import with Separate Primitives ON first.")
            return {'CANCELLED'}

        st, msg = inject_fc3(ctx, objects, self.filepath, self.lod_level)
        if st == {'FINISHED'}:
            VerboseLogger.session_complete(
                "inject_fc3",
                output_file=self.filepath,
                target_lod=self.lod_level,
                n_objects=len(objects),
                msg=msg)
            VerboseLogger.autosave_sidecar(self.filepath)
            self.report({'INFO'}, msg)
        else:
            VerboseLogger.session_complete(
                "inject_fc3",
                output_file=self.filepath,
                status="CANCELLED", reason=msg)
            VerboseLogger.autosave_sidecar(self.filepath)
            self.report({'ERROR'}, msg)
        return st
