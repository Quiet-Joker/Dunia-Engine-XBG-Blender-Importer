"""View / edit an XBG's bounding volumes: the XOBB box and HPSB sphere.

These are the model's single bounding box + bounding sphere (the format stores
exactly one of each — they're the whole-model bound). The engine uses them for
culling / LOD switching, and for creatures (which have no .hkx) they are the
collision proxy. Both are fixed-size float blocks, so editing them is a pure
in-place patch (nothing else in the file moves):
  XOBB: 6 floats (min xyz, max xyz)  at chunk_start + 20
  HPSB: 4 floats (center xyz, radius) at chunk_start + 20

Manual numeric editing is the primary path (creature boxes are deliberately
tucked just inside the mesh — e.g. the direhorse — so precise values matter).
"Fit to Selected" reuses the inject path's `patch_bounds` so the computed bounds
match exactly what a full inject would write, then reads them back for tweaking.
"""
import os
import struct

import bpy

from .binary_avatar import LE, BE, detect_endian_from_bytes
from .chunks_avatar import find_chunk, patch_bounds


def read_bounds(file_data, endian):
    """Return {'xobb': (min3, max3, offset)|None, 'hpsb': (center3, radius, offset)|None}."""
    en = endian
    out = {'xobb': None, 'hpsb': None}
    xi = find_chunk(file_data, 'XOBB', endian)
    if xi:
        off = xi[0] + 20
        v = struct.unpack_from(f'{en}6f', file_data, off)
        out['xobb'] = (v[:3], v[3:], off)
    hi = find_chunk(file_data, 'HPSB', endian)
    if hi:
        off = hi[0] + 20
        v = struct.unpack_from(f'{en}4f', file_data, off)
        out['hpsb'] = (v[:3], v[3], off)
    return out


def patch_bounds_floats(file_data, endian, xobb=None, hpsb=None):
    """In-place patch. xobb=(min3,max3,offset); hpsb=(center3,radius,offset)."""
    buf = bytearray(file_data)
    if xobb:
        mn, mx, off = xobb
        struct.pack_into(f'{endian}6f', buf, off, mn[0], mn[1], mn[2], mx[0], mx[1], mx[2])
    if hpsb:
        c, r, off = hpsb
        struct.pack_into(f'{endian}4f', buf, off, c[0], c[1], c[2], float(r))
    return bytes(buf)


# ---------------------------------------------------------------------------
# Blender operators + panel
# ---------------------------------------------------------------------------

def _load(sc):
    path = sc.xbg_bounds_path
    if not path or not os.path.isfile(path):
        return None, None
    data = open(path, 'rb').read()
    endian = BE if sc.xbg_bounds_endian == 'BE' else LE
    return data, endian


def _fill_from_bounds(sc, b):
    sc.xbg_has_xobb = b['xobb'] is not None
    sc.xbg_has_hpsb = b['hpsb'] is not None
    if b['xobb']:
        mn, mx, off = b['xobb']
        sc.xbg_box_min = mn; sc.xbg_box_max = mx; sc.xbg_xobb_off = off
    if b['hpsb']:
        c, r, off = b['hpsb']
        sc.xbg_sphere_center = c; sc.xbg_sphere_radius = r; sc.xbg_hpsb_off = off


class XBG_OT_ReadBounds(bpy.types.Operator):
    """Read the XOBB box + HPSB sphere from an XBG into editable fields."""
    bl_idname = "xbg.read_bounds"
    bl_label = "Read Bounding Volumes"
    bl_options = {'REGISTER'}
    filepath: bpy.props.StringProperty(subtype="FILE_PATH")
    filter_glob: bpy.props.StringProperty(default="*.xbg", options={'HIDDEN'})

    def invoke(self, ctx, ev):
        # reuse the currently-loaded/remembered XBG (no re-navigation)
        session = ctx.scene.xbg_session_data
        if not self.filepath and session.is_loaded and session.filepath:
            self.filepath = session.filepath
        ctx.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}

    def execute(self, ctx):
        sc = ctx.scene
        if not self.filepath or not os.path.isfile(self.filepath):
            self.report({'ERROR'}, "No valid .xbg selected")
            return {'CANCELLED'}
        data = open(self.filepath, 'rb').read()
        endian = detect_endian_from_bytes(data[:32])
        b = read_bounds(data, endian)
        if not b['xobb'] and not b['hpsb']:
            self.report({'ERROR'}, "No XOBB / HPSB chunk found")
            return {'CANCELLED'}
        # If this file is already imported in the scene, display the gizmo
        # through THAT mesh's matrix_world so it lines up with what's on
        # screen; otherwise there's no live reference and the gizmo shows
        # raw file-space coordinates (see _bounds_display_frame).
        norm_path = os.path.normcase(os.path.abspath(self.filepath))
        frame_obj = next(
            (o for o in ctx.scene.objects if o.type == 'MESH' and 'xbg_data' in o
             and os.path.normcase(os.path.abspath(o['xbg_data'].get('filepath', ''))) == norm_path),
            None)
        sc.xbg_bounds_frame_obj = frame_obj.name if frame_obj else ''
        sc.xbg_bounds_path = self.filepath
        sc.xbg_bounds_endian = 'BE' if endian == BE else 'LE'
        _fill_from_bounds(sc, b)
        if not frame_obj:
            self.report({'INFO'}, f"Read bounds from {os.path.basename(self.filepath)} "
                                   f"(not imported in this scene — showing raw file-space coordinates)")
        else:
            self.report({'INFO'}, f"Read bounds from {os.path.basename(self.filepath)}")
        return {'FINISHED'}


class XBG_OT_FitBoundsToSelected(bpy.types.Operator):
    """Compute the box + sphere that enclose the selected mesh objects."""
    bl_idname = "xbg.fit_bounds_to_selected"
    bl_label = "Fit to Selected"
    bl_description = ("Set the box/sphere to enclose the selected mesh objects "
                      "(uses the same computation as a full inject)")
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, ctx):
        sc = ctx.scene
        data, endian = _load(sc)
        if data is None:
            self.report({'ERROR'}, "Read an XBG first")
            return {'CANCELLED'}
        objs = [o for o in ctx.selected_objects if o.type == 'MESH']
        if not objs:
            self.report({'ERROR'}, "Select at least one mesh object")
            return {'CANCELLED'}
        ins = getattr(sc, 'xbg_inject_settings', None)
        import_mesh_only = bool(getattr(ins, 'import_mesh_only', False)) if ins else False
        patched = patch_bounds(bytearray(data), objs, 1.0, import_mesh_only, endian)
        # Display the gizmo through the selected mesh's OWN matrix_world so
        # it lines up exactly with the mesh it was fit to (2026-06-30 fix —
        # previously the gizmo always showed raw file-space coordinates, so
        # it looked like it "fit to the world origin" for any mesh whose
        # viewport transform carries rotation/translation, e.g. anything
        # parented to an armature). Must be set BEFORE _fill_from_bounds.
        sc.xbg_bounds_frame_obj = objs[0].name
        _fill_from_bounds(sc, read_bounds(patched, endian))
        self.report({'INFO'}, f"Fitted bounds to {len(objs)} object(s)")
        return {'FINISHED'}


class XBG_OT_WriteBounds(bpy.types.Operator):
    """Write the edited box + sphere back into a copy of the XBG (in place)."""
    bl_idname = "xbg.write_bounds"
    bl_label = "Save Bounding Volumes"
    bl_options = {'REGISTER'}
    filepath: bpy.props.StringProperty(subtype="FILE_PATH")
    filter_glob: bpy.props.StringProperty(default="*.xbg", options={'HIDDEN'})
    check_existing: bpy.props.BoolProperty(default=True, options={'HIDDEN'})

    def invoke(self, ctx, ev):
        src = ctx.scene.xbg_bounds_path
        if not src:
            self.report({'ERROR'}, "Read an XBG first")
            return {'CANCELLED'}
        if not self.filepath:
            base, ext = os.path.splitext(src)
            self.filepath = base + "_bounds" + ext
        ctx.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}

    def execute(self, ctx):
        sc = ctx.scene
        data, endian = _load(sc)
        if data is None:
            self.report({'ERROR'}, "Source XBG missing — Read again")
            return {'CANCELLED'}
        # re-validate chunk offsets against the live file
        live = read_bounds(data, endian)
        xobb = hpsb = None
        if sc.xbg_has_xobb and live['xobb']:
            if live['xobb'][2] != sc.xbg_xobb_off:
                self.report({'ERROR'}, "Source changed — Read again")
                return {'CANCELLED'}
            xobb = (tuple(sc.xbg_box_min), tuple(sc.xbg_box_max), sc.xbg_xobb_off)
        if sc.xbg_has_hpsb and live['hpsb']:
            if live['hpsb'][2] != sc.xbg_hpsb_off:
                self.report({'ERROR'}, "Source changed — Read again")
                return {'CANCELLED'}
            hpsb = (tuple(sc.xbg_sphere_center), sc.xbg_sphere_radius, sc.xbg_hpsb_off)
        out = patch_bounds_floats(data, endian, xobb, hpsb)
        with open(self.filepath, 'wb') as f:
            f.write(out)
        self.report({'INFO'}, f"Wrote bounds -> {os.path.basename(self.filepath)}")
        return {'FINISHED'}


# XBG_PT_BoundsEditorPanel removed (2026-06-30): its XOBB/HPSB editing was
# merged into the "Bounding Volume Display + Editor" section of
# XBG_PT_ViewportVizPanel (panels_avatar.py), so the editable fields now sit
# directly under the Show Bounding Box / Show Bounding Sphere toggles and
# edit the LIVE viewport gizmo (see Core/debug.refresh_bounds_display). The
# operators above (read_bounds / fit_bounds_to_selected / write_bounds) are
# unchanged and are driven from that merged panel.
