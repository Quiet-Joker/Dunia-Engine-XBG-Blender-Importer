import struct
import threading
import urllib.request
import re
import os

bl_info = {
    "name": "XBG Importer",
    "author": "Quiet Joker, JasperZebra",
    "version": (2, 1, 2),
    "blender": (5, 0, 0),
    "location": "View3D > Sidebar > XBG Import",
    "description": "Import XBG models from James Cameron's Avatar The Game",
    "category": "Import-Export",
}

import bpy
import os
from .modules.import_xbg import XBGBlenderImporter
from .modules.export_xbg import XBGExporter
from .modules.debug import VerboseLogger


# ---------------------------------------------------------------------------
# Auto-updater
# ---------------------------------------------------------------------------

_RAW_BASE = "https://raw.githubusercontent.com/Quiet-Joker/Avatar-XBG-Blender-Importer/Dev/"

_MODULE_FILES = [
    "__init__.py",
    "modules/binary.py",
    "modules/bounds.py",
    "modules/debug.py",
    "modules/export_xbg.py",
    "modules/import_xbg.py",
    "modules/materials.py",
    "modules/mesh.py",
    "modules/nodes.py",
    "modules/skeleton.py",
    "modules/uv.py",
    "modules/weights.py",
    "modules/xbt.py",
]

_update_status = None   # None = not checked, "up_to_date", or "vX.X.X available"
_update_error  = None   # set if network fetch failed


def _fetch_remote_version():
    """Fetch remote __init__.py and return version tuple, or None on failure."""
    try:
        url = _RAW_BASE + "__init__.py"
        req = urllib.request.urlopen(url, timeout=8)
        text = req.read(4096).decode("utf-8", errors="ignore")
        m = re.search(r'"version"\s*:\s*\((\d+),\s*(\d+),\s*(\d+)\)', text)
        if m:
            return (int(m.group(1)), int(m.group(2)), int(m.group(3)))
    except Exception:
        pass
    return None


def _check_update_thread():
    global _update_status, _update_error
    remote = _fetch_remote_version()
    if remote is None:
        _update_error = "Could not reach update server."
        return
    local = bl_info["version"]
    if remote > local:
        _update_status = f"v{remote[0]}.{remote[1]}.{remote[2]} available"
    else:
        _update_status = "up_to_date"


# ---------------------------------------------------------------------------
# Addon preferences
# ---------------------------------------------------------------------------

class XBGAddonPreferences(bpy.types.AddonPreferences):
    bl_idname = __name__
    data_folder: bpy.props.StringProperty(
        name="Data Folder",
        description="Path to the game's Data folder",
        default="",
        subtype='DIR_PATH'
    )

    def draw(self, ctx):
        self.layout.prop(self, "data_folder")


# ---------------------------------------------------------------------------
# Property groups
# ---------------------------------------------------------------------------

class XBGImportSettings(bpy.types.PropertyGroup):
    load_textures: bpy.props.BoolProperty(
        name="Load Textures",
        description="Automatically load and setup textures from XBM material files",
        default=True
    )
    load_hd_textures: bpy.props.BoolProperty(
        name="Load HD Textures",
        description="Use high-resolution _mip0 texture variants when available",
        default=True
    )


class XBGExportSettings(bpy.types.PropertyGroup):
    auto_scale_to_bounds: bpy.props.BoolProperty(
        name="Auto-Scale to Fit Bounds",
        description="Automatically scale mesh to fit within XBG format limits",
        default=False
    )
    show_scale_info: bpy.props.BoolProperty(
        name="Show Scale Information",
        description="Display scaling requirements",
        default=True
    )
    ignore_format_limits: bpy.props.BoolProperty(
        name="Ignore Format Limits (DANGEROUS)",
        description="Export raw values without clamping - may corrupt model!",
        default=False
    )
    override_game_scale: bpy.props.BoolProperty(
        name="Override Game Scale",
        description="Write a new scale value to the PMCP chunk",
        default=False
    )
    target_game_scale: bpy.props.FloatProperty(
        name="New Scale Value",
        description="The new PMCP multiplier",
        default=1.0,
        precision=6,
        min=0.000001
    )


class XBGDebugSettings(bpy.types.PropertyGroup):
    verbose_logging: bpy.props.BoolProperty(
        name="Verbose Logging",
        description="Print detailed debug information to console (bones, chunks, transforms, etc.)",
        default=False
    )
    show_file_info: bpy.props.BoolProperty(
        name="Show File Info",
        description="Display XBG file chunk information in the panel",
        default=False
    )
    show_format_bounds: bpy.props.BoolProperty(
        name="Show XBG Format Bounds",
        description="Display the 16-bit coordinate limit as a lattice box",
        default=False
    )
    show_bounding_box: bpy.props.BoolProperty(
        name="Show Bounding Boxes",
        description="Visualize bounding boxes from XOBB chunks",
        default=False
    )
    show_bounding_sphere: bpy.props.BoolProperty(
        name="Show Bounding Spheres",
        description="Visualize bounding spheres from HPSB chunks",
        default=False
    )
    bounds_display_type: bpy.props.EnumProperty(
        name="Display Type",
        description="How to display bounding volumes",
        items=[
            ('WIRE', 'Wire', 'Display as wireframe'),
            ('SOLID', 'Solid', 'Display as solid with transparency'),
            ('LATTICE', 'Lattice', 'Display as lattice modifier on box')
        ],
        default='WIRE'
    )
    flip_normals: bpy.props.BoolProperty(
        name="Flip Normals",
        description="Flip all face normals after import (fixes inverted normals)",
        default=True
    )
    separate_primitives: bpy.props.BoolProperty(
        name="Separate Primitives",
        description="Create separate mesh objects for each primitive chunk instead of joining them",
        default=False
    )
    use_xml_assembly: bpy.props.BoolProperty(
        name="Use XML Assembly",
        description="Search for and use XML files to properly assemble parts using bone transforms",
        default=False
    )
    auto_smooth_normals: bpy.props.BoolProperty(
        name="Auto Smooth Normals",
        description="Automatically apply smooth shading after import",
        default=True
    )
    merge_distance: bpy.props.FloatProperty(
        name="Merge Distance",
        description="Distance threshold for merging duplicate vertices",
        default=0.0001,
        min=0.0,
        max=1.0,
        precision=4
    )
    import_xbt_as_dds: bpy.props.BoolProperty(
        name="Import XBT as DDS",
        description="Import XBT textures as DDS files instead of PNG. WARNING: DDS format will cause texture painting corruption! Use PNG (default) for texture painting",
        default=False
    )
    use_mb2o: bpy.props.BoolProperty(
        name="Use MB2O Transforms",
        description="Apply MB2O bind matrices to skeleton (if available). Disable if bones are mispositioned. Default: OFF",
        default=False
    )
    file_info_data: bpy.props.StringProperty(
        name="File Info Data",
        default=""
    )
    lod_peek_result: bpy.props.StringProperty(
        name="LOD Peek Result",
        default=""
    )
    compact_vertices: bpy.props.BoolProperty(
        name="Compact Vertices (Remove Unused)",
        description="Remove unused vertices during import. A vertex mapping is stored to ensure correct export positions. Recommended for cleaner editing.",
        default=True
    )
    # Bone reorientation toggle
    reorient_bones: bpy.props.BoolProperty(
        name="Reorient Bones",
        description="Point each bone's tail toward its children, making the skeleton easier to read and pose. Leaf bones keep their original orientation.",
        default=False
    )


# ---------------------------------------------------------------------------
# Operators — import / export
# ---------------------------------------------------------------------------

class XBG_OT_Import(bpy.types.Operator):
    bl_idname = "import_scene.xbg_model"
    bl_label = "Import XBG"
    bl_options = {'REGISTER', 'UNDO'}

    filepath: bpy.props.StringProperty(subtype="FILE_PATH")
    files: bpy.props.CollectionProperty(type=bpy.types.OperatorFileListElement)
    directory: bpy.props.StringProperty(subtype="DIR_PATH")

    import_mesh_only: bpy.props.BoolProperty(
        name="Import Mesh Only",
        description="Skip skeleton import and rigging",
        default=False
    )
    import_all_lods: bpy.props.BoolProperty(
        name="Import All LODs",
        description="Import all Level of Details found in file",
        default=False
    )
    lod_level: bpy.props.IntProperty(
        name="LOD Level",
        description="Which LOD to import (0=highest detail, higher=lower detail)",
        default=0,
        min=0,
        max=10
    )

    def draw(self, context):
        layout = self.layout
        box = layout.box()
        box.label(text="LOD Selection:", icon='MOD_MULTIRES')
        box.prop(self, "import_all_lods")
        row = box.row()
        row.enabled = not self.import_all_lods
        row.prop(self, "lod_level")
        if not self.import_all_lods:
            box.label(text=f"Will import LOD {self.lod_level} only", icon='INFO')
        else:
            box.label(text="Will import ALL LODs", icon='INFO')
        box = layout.box()
        box.label(text="Other Options:", icon='PREFERENCES')
        box.prop(self, "import_mesh_only")

    def invoke(self, ctx, ev):
        ctx.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}

    def execute(self, ctx):
        s, ds, p = ctx.scene.xbg_settings, ctx.scene.xbg_debug_settings, ctx.preferences.addons[__name__].preferences
        VerboseLogger.enabled = ds.verbose_logging
        df, lt, lhd = p.data_folder, s.load_textures, s.load_hd_textures

        if lt and not df:
            self.report({'WARNING'}, "Data folder not set - textures will not be loaded")
            lt = False

        imp = XBGBlenderImporter()
        tl = -1 if self.import_all_lods else self.lod_level

        fs = []
        if self.files:
            for f in self.files:
                if f.name.lower().endswith(".xbg"):
                    fs.append(os.path.join(self.directory, f.name))
        elif self.filepath.lower().endswith(".xbg"):
            fs.append(self.filepath)

        if not fs:
            self.report({'ERROR'}, "No valid .xbg files selected")
            return {'CANCELLED'}

        ic = 0
        if ds.import_xbt_as_dds:
            self.report({'WARNING'}, "DDS Import Mode enabled - Texture painting will be corrupted! Use PNG mode for texture painting.")

        for fp in fs:
            try:
                imp.load(
                    ctx, fp, tl, self.import_mesh_only, df, lt, lhd,
                    ds.flip_normals, ds.use_xml_assembly, ds.separate_primitives,
                    ds.show_format_bounds, ds.import_xbt_as_dds,
                    ds.use_mb2o,
                    ds.compact_vertices,  # NEW: Vertex compaction parameter
                    ds.reorient_bones,    # NEW: Bone Orientation
                )
                ic += 1
            except Exception as e:
                self.report({'WARNING'}, f"Failed to import {os.path.basename(fp)}: {str(e)}")

        if ic > 0:
            self.report({'INFO'}, f"Imported {ic} XBG file(s)")
            return {'FINISHED'}
        else:
            self.report({'ERROR'}, "No files were imported successfully")
            return {'CANCELLED'}


class XBG_OT_QuickSetScale(bpy.types.Operator):
    bl_idname = "xbg.quick_set_scale"
    bl_label = "Set Scale"
    value: bpy.props.FloatProperty()

    def execute(self, ctx):
        ctx.scene.xbg_export_settings.target_game_scale = self.value
        return {'FINISHED'}


class XBG_OT_MergeAllMeshes(bpy.types.Operator):
    bl_idname = "xbg.merge_all_meshes"
    bl_label = "Merge All Meshes"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, ctx):
        from .modules.debug import merge_duplicate_vertices
        ds = ctx.scene.xbg_debug_settings
        objs = [o for o in ctx.scene.objects if o.type == 'MESH']
        if not objs:
            self.report({'WARNING'}, "No meshes in scene")
            return {'CANCELLED'}
        merge_duplicate_vertices(objs, ds.merge_distance)
        self.report({'INFO'}, f"Merged vertices on {len(objs)} mesh(es)")
        return {'FINISHED'}


class XBG_OT_MergeSelectedMesh(bpy.types.Operator):
    bl_idname = "xbg.merge_selected_mesh"
    bl_label = "Merge Selected Mesh"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, ctx):
        from .modules.debug import merge_duplicate_vertices
        ds = ctx.scene.xbg_debug_settings
        obj = ctx.active_object
        if not obj or obj.type != 'MESH':
            self.report({'ERROR'}, "No mesh selected")
            return {'CANCELLED'}
        merge_duplicate_vertices([obj], ds.merge_distance)
        self.report({'INFO'}, f"Merged vertices on {obj.name}")
        return {'FINISHED'}


class XBG_OT_Export(bpy.types.Operator):
    bl_idname = "export_scene.xbg_inject"
    bl_label = "Export XBG (Inject)"
    bl_options = {'REGISTER', 'UNDO'}
    filepath: bpy.props.StringProperty(subtype="FILE_PATH")

    def invoke(self, ctx, ev):
        obj = ctx.active_object
        obj and "xbg_data" in obj and setattr(self, 'filepath', obj["xbg_data"]["filepath"])
        ctx.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}

    def execute(self, ctx):
        obj = ctx.active_object
        if not obj:
            self.report({'ERROR'}, "No active object selected")
            return {'CANCELLED'}

        ds, es = ctx.scene.xbg_debug_settings, ctx.scene.xbg_export_settings
        VerboseLogger.enabled = ds.verbose_logging

        if "xbg_data" in obj:
            from .modules.debug import analyze_export_scale
            m = obj["xbg_data"].to_dict()
            analyze_export_scale(obj, m.get("pos_scale", 1.0), m.get("import_mesh_only", False))

        exp = XBGExporter()
        st, msg = exp.export(ctx, obj, self.filepath, es.auto_scale_to_bounds, es.show_scale_info, es.ignore_format_limits)
        if st == {'FINISHED'}:
            self.report({'INFO'}, msg)
        else:
            self.report({'ERROR'}, msg)
        return st


class XBG_OT_PeekLODs(bpy.types.Operator):
    """Quickly scan a .xbg file to count its LODs without a full import."""
    bl_idname = "xbg.peek_lods"
    bl_label = "Peek LOD Count"
    bl_description = "Scan a .xbg file to show how many LODs it contains before importing"

    filepath: bpy.props.StringProperty(subtype="FILE_PATH")

    def invoke(self, ctx, ev):
        ctx.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}

    def execute(self, ctx):
        if not self.filepath or not os.path.exists(self.filepath):
            self.report({'ERROR'}, "No valid .xbg file selected")
            return {'CANCELLED'}
        if not self.filepath.lower().endswith('.xbg'):
            self.report({'ERROR'}, "Selected file is not an .xbg file")
            return {'CANCELLED'}
        try:
            lod_count = XBG_OT_PeekLODs._peek_lod_count(self.filepath)
            fn = os.path.basename(self.filepath)
            if lod_count > 0:
                result = f"{fn}: {lod_count} LOD(s)  (LOD 0 – {lod_count - 1})"
            else:
                result = f"{fn}: LOD count could not be read"
            ctx.scene.xbg_debug_settings.lod_peek_result = result
            self.report({'INFO'}, result)
        except Exception as e:
            ctx.scene.xbg_debug_settings.lod_peek_result = f"Error: {e}"
            self.report({'WARNING'}, f"Could not read file: {e}")
        return {'FINISHED'}

    @staticmethod
    def _peek_lod_count(filepath):
        fsize = os.path.getsize(filepath)
        with open(filepath, 'rb') as f:
            data = f.read(min(fsize, 4096))

        if len(data) < 32:
            return 0

        cc = struct.unpack_from('<i', data, 28)[0]
        offset = 32

        for _ in range(min(cc, 64)):
            if offset + 12 > len(data):
                with open(filepath, 'rb') as f:
                    f.seek(offset)
                    hdr = f.read(12)
                if len(hdr) < 12:
                    break
                chunk_sig = hdr[:4].decode('utf-8', 'ignore')
                chunk_size = struct.unpack_from('<i', hdr, 8)[0]
                if chunk_sig == 'SDOL':
                    lod_off = offset + 20
                    with open(filepath, 'rb') as f:
                        f.seek(lod_off)
                        lc = f.read(4)
                    return max(0, struct.unpack_from('<i', lc, 0)[0]) if len(lc) == 4 else 0
                if chunk_size <= 0:
                    break
                offset += chunk_size
                continue

            chunk_sig = data[offset:offset + 4].decode('utf-8', 'ignore')
            chunk_size = struct.unpack_from('<i', data, offset + 8)[0]

            if chunk_sig == 'SDOL':
                lod_off = offset + 20
                if lod_off + 4 <= len(data):
                    return max(0, struct.unpack_from('<i', data, lod_off)[0])
                with open(filepath, 'rb') as f:
                    f.seek(lod_off)
                    lc = f.read(4)
                return max(0, struct.unpack_from('<i', lc, 0)[0]) if len(lc) == 4 else 0

            if chunk_size <= 0:
                break
            offset += chunk_size

        return 0


# ---------------------------------------------------------------------------
# Operators — updater
# ---------------------------------------------------------------------------

class XBG_OT_CheckForUpdates(bpy.types.Operator):
    """Check GitHub for plugin updates"""
    bl_idname = "xbg.check_for_updates"
    bl_label = "Check for Updates"

    def execute(self, context):
        global _update_status, _update_error
        _update_status = None
        _update_error  = None
        threading.Thread(target=_check_update_thread, daemon=True).start()
        self.report({'INFO'}, "Checking for updates...")
        return {'FINISHED'}


class XBG_OT_ApplyUpdate(bpy.types.Operator):
    """Download and install the latest version from GitHub"""
    bl_idname = "xbg.apply_update"
    bl_label = "Update Now"

    def execute(self, context):
        plugin_dir = os.path.dirname(os.path.abspath(__file__))
        failed = []

        for rel_path in _MODULE_FILES:
            url = _RAW_BASE + rel_path
            dest = os.path.join(plugin_dir, rel_path.replace("/", os.sep))

            # Make sure the target directory exists (e.g. modules/)
            os.makedirs(os.path.dirname(dest), exist_ok=True)

            try:
                req = urllib.request.urlopen(url, timeout=30)
                data = req.read()
                with open(dest, 'wb') as f:
                    f.write(data)
            except Exception as e:
                failed.append(f"{rel_path}: {e}")

        global _update_status
        _update_status = None  # reset so user can re-check after restart

        if failed:
            self.report({'WARNING'},
                f"Update partially failed — {len(failed)} file(s) not downloaded. "
                f"Check console for details. Restart Blender for partial changes.")
            for msg in failed:
                print(f"[XBG Updater] FAILED: {msg}")
        else:
            self.report({'INFO'}, "Update complete! Restart Blender to apply.")

        return {'FINISHED'}


# ---------------------------------------------------------------------------
# Panels
# ---------------------------------------------------------------------------

class XBG_PT_Panel(bpy.types.Panel):
    bl_label = "XBG Import"
    bl_idname = "OBJECT_PT_xbg_import"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "XBG Import"

    def draw(self, ctx):
        l, s, p = self.layout, ctx.scene.xbg_settings, ctx.preferences.addons[__name__].preferences

        # --- Update status bar ---
        if _update_status is None and _update_error is None:
            row = l.row()
            row.operator("xbg.check_for_updates", text="Check for Updates", icon="FILE_REFRESH")
        elif _update_error:
            row = l.row()
            row.label(text="Update check failed", icon="ERROR")
            row.operator("xbg.check_for_updates", text="Retry", icon="FILE_REFRESH")
        elif _update_status == "up_to_date":
            row = l.row()
            row.label(text="Tool is up to date", icon="CHECKMARK")
            row.operator("xbg.check_for_updates", text="", icon="FILE_REFRESH")
        else:
            box = l.box()
            box.label(text=f"Update available: {_update_status}", icon="INFO")
            row = box.row()
            row.operator("xbg.apply_update", text="Update Now", icon="IMPORT")
            row.operator("xbg.check_for_updates", text="", icon="FILE_REFRESH")

        l.separator()

        b = l.box()
        b.label(text="Game Data Folder:", icon='FILE_FOLDER')
        b.prop(p, "data_folder", text="")

        b = l.box()
        b.label(text="Import Options:", icon='PREFERENCES')
        b.prop(s, "load_textures")
        r = b.row()
        r.enabled = s.load_textures
        r.prop(s, "load_hd_textures")

        l.separator()
        r = l.row()
        r.scale_y = 1.5
        r.operator("import_scene.xbg_model", icon='IMPORT')

        pb = l.box()
        pb.label(text="LOD Preview:", icon='MOD_MULTIRES')
        pb.operator("xbg.peek_lods", text="Peek LOD Count...", icon='VIEWZOOM')
        ds = ctx.scene.xbg_debug_settings
        if ds.lod_peek_result:
            pr = pb.row()
            pr.alignment = 'LEFT'
            pr.label(text=ds.lod_peek_result, icon='INFO')

        l.separator()
        b = l.box()
        b.label(text="Export (Re-Inject):", icon='EXPORT')
        obj, es = ctx.active_object, ctx.scene.xbg_export_settings

        if obj and "xbg_joined" in obj:
            wb = b.box()
            wb.alert = True
            wb.label(text="Re-inject unavailable", icon='ERROR')
            wb.label(text="Imported as joined mesh.")
            wb.label(text="Enable Separate Primitives")
            wb.label(text="and re-import to edit & inject.")
        elif obj and "xbg_data" in obj:
            b.label(text=f"Linked: {os.path.basename(obj['xbg_data']['filepath'])}", icon='LINKED')
            m = obj["xbg_data"].to_dict()
            ps, imo = m.get("pos_scale", 1.0), m.get("import_mesh_only", False)

            exp = XBGExporter()
            ns, rs, si = exp.calculate_required_scale(obj, ps, imo)

            if not es.override_game_scale:
                ib = b.box()
                if ns:
                    ib.alert = True
                    ib.label(text="⚠ MESH EXCEEDS FORMAT BOUNDS", icon='ERROR')
                    ib.label(text=f"Exceeded: {si}")
                    ib.separator()
                    es.ignore_format_limits and ib.label(text="IGNORE LIMITS ENABLED!", icon='ERROR') or (
                        es.auto_scale_to_bounds and (
                            ib.label(text="Will auto-scale to fit:", icon='INFO'),
                            ib.label(text=f"  Scale: {rs:.6f}")
                        ) or ib.label(text="Vertices will be CLAMPED!", icon='CANCEL')
                    )
                else:
                    ib.label(text="✓ Mesh fits within bounds", icon='CHECKMARK')

            sb = b.box()
            sb.label(text="Export Options:", icon='SETTINGS')
            sb.prop(es, "auto_scale_to_bounds")
            sb.prop(es, "show_scale_info")
            sb.separator()
            sb.label(text=f"Current Scale: {m['pos_scale']:.6f}", icon='LINENUMBERS_ON')
            sb.prop(es, "override_game_scale")

            if es.override_game_scale:
                r = sb.row()
                r.prop(es, "target_game_scale")
                r = sb.row(align=True)
                op = r.operator("xbg.quick_set_scale", text="x2")
                op.value = m['pos_scale'] * 2
                op = r.operator("xbg.quick_set_scale", text="x0.5")
                op.value = m['pos_scale'] * 0.5

            sb.separator()
            dr = sb.row()
            dr.alert = True
            dr.prop(es, "ignore_format_limits")

            r = b.row()
            r.scale_y = 1.3
            r.operator("export_scene.xbg_inject", text="Inject Mesh Data", icon='EXPORT')
        else:
            b.label(text="Select an imported XBG mesh", icon='INFO')
            b.enabled = False


class XBG_PT_DebugPanel(bpy.types.Panel):
    bl_label = "XBG Debug"
    bl_idname = "OBJECT_PT_xbg_debug"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "XBG Import"
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, ctx):
        l, ds = self.layout, ctx.scene.xbg_debug_settings

        b = l.box()
        b.label(text="Logging:", icon='CONSOLE')
        b.prop(ds, "verbose_logging")
        b.prop(ds, "show_file_info")

        if ds.show_file_info and ds.file_info_data:
            info_box = b.box()
            info_box.scale_y = 0.8
            for line in ds.file_info_data.split('\n'):
                if line.strip():
                    row = info_box.row()
                    row.alignment = 'LEFT'
                    row.label(text=line)

        l.separator()
        b = l.box()
        b.label(text="Advanced Skeleton:", icon='ARMATURE_DATA')
        b.prop(ds, "use_mb2o")

        if ds.use_mb2o:
            i = b.box()
            i.label(text="MB2O Mode:", icon='INFO')
            i.label(text="Uses bind matrices from MB2O chunk")
            i.label(text="May fix some bone positioning issues")
            i.label(text="Disable if bones look wrong")
        else:
            i = b.box()
            i.label(text="EDON Mode (Default):", icon='INFO')
            i.label(text="Uses skeleton from EDON chunk")
            i.label(text="Standard bone transforms")

        l.separator()
        b = l.box()
        b.label(text="Mesh Processing:", icon='MESH_DATA')
        b.prop(ds, "compact_vertices")

        if ds.compact_vertices:
            i = b.box()
            i.label(text="Vertex Compaction Enabled:", icon='INFO')
            i.label(text="Removes unused/floating vertices")
            i.label(text="Stores mapping for correct export")
            i.label(text="Cleaner editing, safe export")
        else:
            i = b.box()
            i.alert = True
            i.label(text="Compaction Disabled:", icon='ERROR')
            i.label(text="Ghost vertices will be visible")
            i.label(text="in Edit Mode")

        b.separator()
        b.prop(ds, "flip_normals")
        b.prop(ds, "separate_primitives")

        if ds.separate_primitives:
            i = b.box()
            i.label(text="Separate Primitives Mode:", icon='INFO')
            i.label(text="Each primitive = separate object")
            i.label(text="Creates individual mesh per chunk")

        b.prop(ds, "use_xml_assembly")

        if ds.use_xml_assembly:
            i = b.box()
            i.label(text="XML Assembly:", icon='INFO')
            i.label(text="Uses .xml files for bone transforms")
            i.label(text="Properly positions weapon parts")

        b.separator()
        b.prop(ds, "auto_smooth_normals")
        b.separator()
        b.label(text="Merge Vertices:", icon='AUTOMERGE_ON')
        b.prop(ds, "merge_distance")
        r = b.row(align=True)
        r.operator("xbg.merge_all_meshes", text="All Meshes")
        r.operator("xbg.merge_selected_mesh", text="Selected")

        l.separator()
        b = l.box()
        b.label(text="Texture Import:", icon='TEXTURE')
        b.prop(ds, "import_xbt_as_dds")

        if ds.import_xbt_as_dds:
            i = b.box()
            i.alert = True
            i.label(text="⚠ DDS Import Mode:", icon='ERROR')
            i.label(text="WARNING: Texture painting will be")
            i.label(text="corrupted with DDS format!")
            i.label(text="Use PNG (default) for painting.")

        l.separator()
        b = l.box()
        b.label(text="Format Bounds:", icon='SHADING_BBOX')
        b.prop(ds, "show_format_bounds")

        l.separator()
        b = l.box()
        b.label(text="Bounding Volumes:", icon='MESH_CUBE')
        b.prop(ds, "show_bounding_box")
        b.prop(ds, "show_bounding_sphere")
        if ds.show_bounding_box or ds.show_bounding_sphere:
            b.prop(ds, "bounds_display_type")


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

classes = (
    XBGAddonPreferences,
    XBGImportSettings,
    XBGExportSettings,
    XBGDebugSettings,
    XBG_OT_Import,
    XBG_OT_QuickSetScale,
    XBG_OT_MergeAllMeshes,
    XBG_OT_MergeSelectedMesh,
    XBG_OT_Export,
    XBG_OT_PeekLODs,
    XBG_OT_CheckForUpdates,
    XBG_OT_ApplyUpdate,
    XBG_PT_Panel,
    XBG_PT_DebugPanel,
)


def register():
    [bpy.utils.register_class(c) for c in classes]
    bpy.types.Scene.xbg_settings = bpy.props.PointerProperty(type=XBGImportSettings)
    bpy.types.Scene.xbg_export_settings = bpy.props.PointerProperty(type=XBGExportSettings)
    bpy.types.Scene.xbg_debug_settings = bpy.props.PointerProperty(type=XBGDebugSettings)
    # Kick off background version check on startup
    threading.Thread(target=_check_update_thread, daemon=True).start()


def unregister():
    del bpy.types.Scene.xbg_settings
    del bpy.types.Scene.xbg_export_settings
    del bpy.types.Scene.xbg_debug_settings
    [bpy.utils.unregister_class(c) for c in reversed(classes)]


if __name__ == "__main__":
    register()
