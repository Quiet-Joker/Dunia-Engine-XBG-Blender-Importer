"""Edit the per-LOD switch distances stored in an XBG's SDOL chunk.

Each LOD record in SDOL begins with a 4-byte float `lod_dist` — the camera
distance (in metres) at which the engine switches to that LOD. Because the field
is a fixed-size float at a fixed position, editing it is a pure IN-PLACE patch:
no chunk grows, no offset moves, nothing else in the file changes. This module
provides a focused read/patch tool (no full mesh re-inject needed).

`read_lod_dists()` walks the SDOL exactly like `parse_sdol()` but records only
the absolute file offset + current value of each `lod_dist`. `patch_lod_dists()`
writes new floats straight back at those offsets.
"""
import os
import struct

import bpy

from .binary_avatar import LE, BE, detect_endian_from_bytes
from .chunks_avatar import find_chunk


def read_lod_dists(file_data, endian):
    """Return [(lod_index, abs_offset, value), ...] for the SDOL chunk, or []."""
    info = find_chunk(file_data, 'SDOL', endian)
    if not info:
        return []
    _, p, _ = info
    en = endian
    p += 8                                   # skip unk_0, unk_1
    lod_count = struct.unpack_from(f'{en}i', file_data, p)[0]; p += 4
    out = []
    for li in range(lod_count):
        dist_off = p
        dist = struct.unpack_from(f'{en}f', file_data, p)[0]; p += 4
        out.append((li, dist_off, dist))
        vb_count = struct.unpack_from(f'{en}i', file_data, p)[0]; p += 4
        p += vb_count * 16
        sm_count = struct.unpack_from(f'{en}i', file_data, p)[0]; p += 4
        p += sm_count * 28
        vsize = struct.unpack_from(f'{en}I', file_data, p)[0]; p += 4
        if p % 16:
            p += 16 - (p % 16)
        p += vsize
        isize = struct.unpack_from(f'{en}I', file_data, p)[0]; p += 4
        if p % 16:
            p += 16 - (p % 16)
        p += isize * 2
    return out


def patch_lod_dists(file_data, endian, edits):
    """edits: [(abs_offset, new_value)]. Patches each float in place."""
    buf = bytearray(file_data)
    for off, val in edits:
        struct.pack_into(f'{endian}f', buf, off, float(val))
    return bytes(buf)


# ---------------------------------------------------------------------------
# Blender property + operators + panel
# ---------------------------------------------------------------------------

class XBGLodDistItem(bpy.types.PropertyGroup):
    lod_index: bpy.props.IntProperty(name="LOD")
    distance:  bpy.props.FloatProperty(
        name="Distance", description="Camera distance (m) at which this LOD "
        "becomes active — larger = the LOD persists further away",
        min=0.0, soft_max=500.0, precision=2)
    offset:    bpy.props.IntProperty()       # absolute file offset of the float


class XBG_OT_ReadLODDistances(bpy.types.Operator):
    """Read the per-LOD switch distances from an XBG's SDOL chunk."""
    bl_idname = "xbg.read_lod_distances"
    bl_label = "Read LOD Distances"
    bl_description = ("Open an XBG and load its per-LOD switch distances into "
                      "editable fields below")
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
            self.report({'ERROR'}, "No valid .xbg file selected")
            return {'CANCELLED'}
        data = open(self.filepath, 'rb').read()
        endian = detect_endian_from_bytes(data[:32])
        dists = read_lod_dists(data, endian)
        if not dists:
            self.report({'ERROR'}, "No SDOL chunk / LOD distances found")
            return {'CANCELLED'}
        sc.xbg_lod_dists.clear()
        for li, off, val in dists:
            it = sc.xbg_lod_dists.add()
            it.lod_index = li; it.offset = off; it.distance = val
        sc.xbg_lod_dist_path = self.filepath
        sc.xbg_lod_dist_endian = 'BE' if endian == BE else 'LE'
        self.report({'INFO'}, f"Loaded {len(dists)} LOD distances from "
                              f"{os.path.basename(self.filepath)}")
        return {'FINISHED'}


class XBG_OT_WriteLODDistances(bpy.types.Operator):
    """Write the edited LOD distances back into an XBG (in-place float patch)."""
    bl_idname = "xbg.write_lod_distances"
    bl_label = "Save LOD Distances"
    bl_description = ("Patch the edited distances back into a copy of the XBG. "
                      "Only the LOD-distance floats change; the rest of the file "
                      "is byte-for-byte identical")
    bl_options = {'REGISTER'}

    filepath: bpy.props.StringProperty(subtype="FILE_PATH")
    filter_glob: bpy.props.StringProperty(default="*.xbg", options={'HIDDEN'})
    check_existing: bpy.props.BoolProperty(default=True, options={'HIDDEN'})

    def invoke(self, ctx, ev):
        src = ctx.scene.xbg_lod_dist_path
        if not src:
            self.report({'ERROR'}, "Read an XBG first")
            return {'CANCELLED'}
        if not self.filepath:
            base, ext = os.path.splitext(src)
            self.filepath = base + "_loddist" + ext
        ctx.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}

    def execute(self, ctx):
        sc = ctx.scene
        src = sc.xbg_lod_dist_path
        if not src or not os.path.isfile(src):
            self.report({'ERROR'}, "Source XBG missing — Read again")
            return {'CANCELLED'}
        endian = BE if sc.xbg_lod_dist_endian == 'BE' else LE
        data = open(src, 'rb').read()
        # re-validate the offsets against the live file before patching
        live = {li: off for li, off, _ in read_lod_dists(data, endian)}
        edits = []
        for it in sc.xbg_lod_dists:
            if live.get(it.lod_index) != it.offset:
                self.report({'ERROR'}, "Source file changed — Read it again")
                return {'CANCELLED'}
            edits.append((it.offset, it.distance))
        out = patch_lod_dists(data, endian, edits)
        with open(self.filepath, 'wb') as f:
            f.write(out)
        self.report({'INFO'}, f"Wrote {len(edits)} LOD distances -> "
                              f"{os.path.basename(self.filepath)}")
        return {'FINISHED'}


class XBG_PT_LODDistancePanel(bpy.types.Panel):
    """View / edit the per-LOD switch distances of an XBG file."""
    bl_label = "LOD Distances"
    bl_idname = "OBJECT_PT_xbg_lod_distances"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "XBG Import"
    bl_parent_id = "OBJECT_PT_xbg_avatar"
    bl_options = {'DEFAULT_CLOSED'}

    @classmethod
    def poll(cls, ctx):
        return ctx.scene.xbg_debug_settings.advanced_mode

    def draw_header(self, ctx):
        self.layout.label(icon='DRIVER_DISTANCE')

    def draw(self, ctx):
        l = self.layout
        sc = ctx.scene
        l.operator("xbg.read_lod_distances", icon='IMPORT')
        if sc.xbg_lod_dist_path:
            l.label(text=os.path.basename(sc.xbg_lod_dist_path), icon='FILE')
            box = l.box()
            if not len(sc.xbg_lod_dists):
                box.label(text="No LODs", icon='INFO')
            for it in sc.xbg_lod_dists:
                box.prop(it, "distance", text=f"LOD {it.lod_index}")
            l.operator("xbg.write_lod_distances", icon='EXPORT')
