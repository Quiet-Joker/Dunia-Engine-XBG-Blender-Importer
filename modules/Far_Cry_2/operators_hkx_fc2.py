"""Avatar: The Game — HKX collision operators (native binary workflow).

The legacy Havok-Content-Tools XML workflow (import XML / export XML /
strip header / apply fixes / transplant) was removed 2026-06-11: the
native packfile reader/patcher in hkx_native.py covers editing AND adding
collision shapes directly on the shipped .hkx files, no conversion step.
(The old XML-workflow modules collision_hkx.py / hkx_collision.py were
deleted 2026-06-23 — they were dead code.)
"""
import os

import bpy


# ---------------------------------------------------------------------------
# Native binary .hkx workflow (no Havok Content Tools / XML conversion)
# ---------------------------------------------------------------------------

class XBG_OT_ImportHKXNativeFC2(bpy.types.Operator):
    """Import an Avatar .hkx collision file directly (binary, no XML)."""
    bl_idname  = "xbg.import_hkx_native_fc2"
    bl_label   = "Import HKX Collision (Native)"
    bl_description = (
        "Read the .hkx Havok packfile directly: every rigid body becomes "
        "an empty and every collision shape (box / convex hull / capsule / "
        "triangle mesh) an editable wireframe object. No Havok Content "
        "Tools or XML conversion needed"
    )
    bl_options = {'REGISTER', 'UNDO'}

    filepath: bpy.props.StringProperty(subtype="FILE_PATH")
    filter_glob: bpy.props.StringProperty(default="*.hkx", options={'HIDDEN'})

    def invoke(self, ctx, ev):
        ctx.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}

    def execute(self, ctx):
        from .hkx_native_fc2 import import_hkx_native
        cs = ctx.scene.xbg_collision_settings
        if not self.filepath or not os.path.isfile(self.filepath):
            self.report({'ERROR'}, "No valid .hkx file selected")
            return {'CANCELLED'}
        try:
            n_bodies, n_shapes = import_hkx_native(ctx, self.filepath)
            cs.last_status = (
                f"Native import: {n_bodies} rigid bodies, "
                f"{n_shapes} shapes from {os.path.basename(self.filepath)}")
            cs.last_status_ok = True
            self.report({'INFO'}, cs.last_status)
            return {'FINISHED'}
        except Exception as exc:
            cs.last_status = f"Native import failed: {exc}"
            cs.last_status_ok = False
            self.report({'ERROR'}, str(exc))
            import traceback; traceback.print_exc()
            return {'CANCELLED'}


class XBG_OT_ExportHKXNativeFC2(bpy.types.Operator):
    """Write the edited collision back into a copy of the original .hkx."""
    bl_idname  = "xbg.export_hkx_native_fc2"
    bl_label   = "Export HKX Collision (Native)"
    bl_description = (
        "Patch all edited native-HKX objects back into a copy of the "
        "original file — box sizes, shape transforms, convex/mesh vertex "
        "positions and capsule radii. The file structure (and MOPP "
        "culling data) is preserved byte-for-byte everywhere else"
    )
    bl_options = {'REGISTER'}

    filepath: bpy.props.StringProperty(subtype="FILE_PATH")
    filter_glob: bpy.props.StringProperty(default="*.hkx", options={'HIDDEN'})
    check_existing: bpy.props.BoolProperty(default=True, options={'HIDDEN'})

    def invoke(self, ctx, ev):
        # default the save name to the source file
        objs = [o for o in ctx.scene.objects if o.get('hkx_native')]
        if objs and not self.filepath:
            src = objs[0]['hkx_path']
            base, ext = os.path.splitext(src)
            self.filepath = base + "_edited" + ext
        ctx.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}

    def execute(self, ctx):
        from .hkx_native_fc2 import export_hkx_native
        cs = ctx.scene.xbg_collision_settings
        if not self.filepath:
            self.report({'ERROR'}, "Choose an output .hkx path")
            return {'CANCELLED'}
        try:
            n, warns = export_hkx_native(ctx, self.filepath)
            for w in warns:
                self.report({'WARNING'}, w)
                print("[HKX]", w)
            cs.last_status = (
                f"Native export: {n} objects patched -> "
                f"{os.path.basename(self.filepath)}"
                + (f"  ({len(warns)} warnings — see console)" if warns else ""))
            cs.last_status_ok = True
            self.report({'INFO'}, cs.last_status)
            return {'FINISHED'}
        except Exception as exc:
            cs.last_status = f"Native export failed: {exc}"
            cs.last_status_ok = False
            self.report({'ERROR'}, str(exc))
            import traceback; traceback.print_exc()
            return {'CANCELLED'}
