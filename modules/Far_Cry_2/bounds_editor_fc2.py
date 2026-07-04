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

from .binary_fc2 import LE, BE, detect_endian_from_bytes
from .chunks_fc2 import find_chunk, patch_bounds


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
    path = sc.xbg_bounds_path_fc2
    if not path or not os.path.isfile(path):
        return None, None
    data = open(path, 'rb').read()
    endian = BE if sc.xbg_bounds_endian_fc2 == 'BE' else LE
    return data, endian


def _fill_from_bounds(sc, b):
    sc.xbg_has_xobb_fc2 = b['xobb'] is not None
    sc.xbg_has_hpsb_fc2 = b['hpsb'] is not None
    if b['xobb']:
        mn, mx, off = b['xobb']
        sc.xbg_box_min_fc2 = mn; sc.xbg_box_max_fc2 = mx; sc.xbg_xobb_off_fc2 = off
    if b['hpsb']:
        c, r, off = b['hpsb']
        sc.xbg_sphere_center_fc2 = c; sc.xbg_sphere_radius_fc2 = r; sc.xbg_hpsb_off_fc2 = off


class XBG_OT_ReadBoundsFC2(bpy.types.Operator):
    """Read the XOBB box + HPSB sphere from an XBG into editable fields."""
    bl_idname = "xbg.read_bounds_fc2"
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
        sc.xbg_bounds_path_fc2 = self.filepath
        sc.xbg_bounds_endian_fc2 = 'BE' if endian == BE else 'LE'
        _fill_from_bounds(sc, b)
        self.report({'INFO'}, f"Read bounds from {os.path.basename(self.filepath)}")
        return {'FINISHED'}


class XBG_OT_FitBoundsToSelectedFC2(bpy.types.Operator):
    """Compute the box + sphere that enclose the selected mesh objects."""
    bl_idname = "xbg.fit_bounds_to_selected_fc2"
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
        _fill_from_bounds(sc, read_bounds(patched, endian))
        self.report({'INFO'}, f"Fitted bounds to {len(objs)} object(s)")
        return {'FINISHED'}


class XBG_OT_WriteBoundsFC2(bpy.types.Operator):
    """Write the edited box + sphere back into a copy of the XBG (in place)."""
    bl_idname = "xbg.write_bounds_fc2"
    bl_label = "Save Bounding Volumes"
    bl_options = {'REGISTER'}
    filepath: bpy.props.StringProperty(subtype="FILE_PATH")
    filter_glob: bpy.props.StringProperty(default="*.xbg", options={'HIDDEN'})
    check_existing: bpy.props.BoolProperty(default=True, options={'HIDDEN'})

    def invoke(self, ctx, ev):
        src = ctx.scene.xbg_bounds_path_fc2
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
        if sc.xbg_has_xobb_fc2 and live['xobb']:
            if live['xobb'][2] != sc.xbg_xobb_off_fc2:
                self.report({'ERROR'}, "Source changed — Read again")
                return {'CANCELLED'}
            xobb = (tuple(sc.xbg_box_min_fc2), tuple(sc.xbg_box_max_fc2), sc.xbg_xobb_off_fc2)
        if sc.xbg_has_hpsb_fc2 and live['hpsb']:
            if live['hpsb'][2] != sc.xbg_hpsb_off_fc2:
                self.report({'ERROR'}, "Source changed — Read again")
                return {'CANCELLED'}
            hpsb = (tuple(sc.xbg_sphere_center_fc2), sc.xbg_sphere_radius_fc2, sc.xbg_hpsb_off_fc2)
        out = patch_bounds_floats(data, endian, xobb, hpsb)
        with open(self.filepath, 'wb') as f:
            f.write(out)
        self.report({'INFO'}, f"Wrote bounds -> {os.path.basename(self.filepath)}")
        return {'FINISHED'}


class XBG_PT_BoundsEditorPanelFC2(bpy.types.Panel):
    """View / edit the XOBB box + HPSB sphere of an XBG file."""
    bl_label = "Bounding Volumes (Collision)"
    bl_idname = "OBJECT_PT_xbg_bounds_editor_fc2"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "XBG Import"
    bl_parent_id = "OBJECT_PT_xbg_fc2"
    bl_options = {'DEFAULT_CLOSED'}

    @classmethod
    def poll(cls, ctx):
        # advanced-only, same as the Avatar master panel this was cloned from
        return ctx.scene.xbg_debug_settings.advanced_mode

    def draw_header(self, ctx):
        self.layout.label(icon='MESH_CUBE')

    def draw(self, ctx):
        l = self.layout
        sc = ctx.scene
        l.operator("xbg.read_bounds_fc2", icon='IMPORT')
        if not sc.xbg_bounds_path_fc2:
            return
        l.label(text=os.path.basename(sc.xbg_bounds_path_fc2), icon='FILE')
        if sc.xbg_has_xobb_fc2:
            box = l.box()
            box.label(text="XOBB box (game space):", icon='MESH_CUBE')
            box.prop(sc, "xbg_box_min_fc2", text="Min")
            box.prop(sc, "xbg_box_max_fc2", text="Max")
        if sc.xbg_has_hpsb_fc2:
            box = l.box()
            box.label(text="HPSB sphere:", icon='MESH_UVSPHERE')
            box.prop(sc, "xbg_sphere_center_fc2", text="Center")
            box.prop(sc, "xbg_sphere_radius_fc2", text="Radius")
        l.operator("xbg.fit_bounds_to_selected_fc2", icon='SHADING_BBOX')
        l.operator("xbg.write_bounds_fc2", icon='EXPORT')
