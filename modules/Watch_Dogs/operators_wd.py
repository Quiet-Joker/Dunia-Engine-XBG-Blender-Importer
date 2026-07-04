"""Watch Dogs 1 / 2 — operators.

Split out of the monolithic __init__.py (2026-06-09 refactor).
"""
import os

import bpy

from .import_wd import load_wd_model as _wd_load


class XBG_OT_ImportWDSkeleton(bpy.types.Operator):
    """Import a Watch Dogs 1 .skeleton file as a standalone Blender armature."""
    bl_idname  = "xbg.import_wd_skeleton"
    bl_label   = "Import WD1 Skeleton"
    bl_description = (
        "Parse a Watch Dogs 1 .skeleton (nbCF binary format): reads all bone "
        "names, rest-pose quaternions and translations, builds an armature with "
        "the standard WD1 character hierarchy.  Compatible with MAB animation "
        "import and weight painting"
    )
    bl_options = {'REGISTER', 'UNDO'}

    filepath: bpy.props.StringProperty(subtype="FILE_PATH")
    filter_glob: bpy.props.StringProperty(default="*.skeleton", options={'HIDDEN'})

    def invoke(self, ctx, ev):
        ctx.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}

    def execute(self, ctx):
        from .import_skeleton_wd import parse_wd1_skeleton, build_wd1_skeleton_armature
        if not self.filepath or not os.path.isfile(self.filepath):
            self.report({'ERROR'}, "No valid .skeleton file selected")
            return {'CANCELLED'}
        try:
            name   = os.path.splitext(os.path.basename(self.filepath))[0]
            bones  = parse_wd1_skeleton(self.filepath)
            arm    = build_wd1_skeleton_armature(ctx, bones, name)
            arm['wd_skeleton_src'] = self.filepath
            self.report({'INFO'},
                f"WD1 skeleton: {len(bones)} bones -> {arm.name}")
            return {'FINISHED'}
        except Exception as exc:
            self.report({'ERROR'}, f"Failed to import .skeleton: {exc}")
            import traceback; traceback.print_exc()
            return {'CANCELLED'}


class XBG_OT_ImportWDHkx(bpy.types.Operator):
    """Import a Watch Dogs 1 .hkx collision file (64-bit Havok 2012)."""
    bl_idname  = "xbg.import_wd_hkx"
    bl_label   = "Import WD1 HKX Collision"
    bl_description = (
        "Read a Watch Dogs 1 .hkx Havok packfile (64-bit Havok 2012, e.g. "
        "vehicle physics) and build a wireframe convex-hull object for each "
        "collision shape.  Compressed triangle-mesh shapes are reported but "
        "not yet decoded"
    )
    bl_options = {'REGISTER', 'UNDO'}

    filepath: bpy.props.StringProperty(subtype="FILE_PATH")
    filter_glob: bpy.props.StringProperty(default="*.hkx", options={'HIDDEN'})

    def invoke(self, ctx, ev):
        ctx.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}

    def execute(self, ctx):
        from .import_hkx_wd import import_hkx_wd
        if not self.filepath or not os.path.isfile(self.filepath):
            self.report({'ERROR'}, "No valid .hkx file selected")
            return {'CANCELLED'}
        try:
            n_hulls, n_verts, n_meshes = import_hkx_wd(ctx, self.filepath)
            msg = (f"WD1 HKX: {n_hulls} convex hulls"
                   f" from {os.path.basename(self.filepath)}")
            if n_meshes:
                msg += (f" + {n_meshes} collision mesh(es) reconstructed "
                        f"per-section from decoded vertices")
            msg += f" ({n_verts} verts)"
            self.report({'INFO'}, msg)
            return {'FINISHED'} if (n_hulls or n_meshes) else {'CANCELLED'}
        except Exception as exc:
            self.report({'ERROR'}, f"Failed to import WD1 .hkx: {exc}")
            import traceback; traceback.print_exc()
            return {'CANCELLED'}


class XBG_OT_ImportWD(bpy.types.Operator):
    """Import a Watch Dogs 1 .xbg (binary GEOM 97.50) model: skeleton,
    meshes, UVs, normals, skin weights."""
    bl_idname  = "xbg.import_wd_model"
    bl_label   = "Import Watch Dogs 1 Model"
    bl_description = (
        "Import a Watch Dogs 1 .xbg (binary GEOM 97.50) model with skeleton "
        "and skin weights"
    )
    bl_options = {'REGISTER', 'UNDO'}

    filepath: bpy.props.StringProperty(subtype="FILE_PATH")
    files: bpy.props.CollectionProperty(type=bpy.types.OperatorFileListElement)
    directory: bpy.props.StringProperty(subtype="DIR_PATH")
    filter_glob: bpy.props.StringProperty(default="*.xbg", options={'HIDDEN'})

    import_all_lods: bpy.props.BoolProperty(
        name="Import All LODs",
        description="Import every Level Of Detail present in the file "
                    "(one mesh set per LOD, named _LOD0/_LOD1/…)",
        default=False)
    lod_level: bpy.props.IntProperty(
        name="LOD Level",
        description="Which LOD to import. 0 = highest detail available; "
                    "higher = lower detail. (WD1 streams its very top LOD "
                    "externally, so level 0 is the best LOD in the file.) "
                    "Clamped to the lowest LOD present.",
        default=0, min=0, max=10)
    import_mesh_only: bpy.props.BoolProperty(
        name="Import Mesh Only",
        description="Skip the skeleton and skin binding — import geometry only",
        default=False)

    def draw(self, context):
        layout = self.layout
        box = layout.box()
        box.label(text="LOD Selection:", icon='MOD_MULTIRES')
        box.prop(self, "import_all_lods")
        row = box.row()
        row.enabled = not self.import_all_lods
        row.prop(self, "lod_level")
        if self.import_all_lods:
            box.label(text="Will import ALL LODs", icon='INFO')
        else:
            box.label(text="Will import LOD %d only" % self.lod_level,
                      icon='INFO')

        box = layout.box()
        box.label(text="Other Options:", icon='PREFERENCES')
        box.prop(self, "import_mesh_only")

    def invoke(self, ctx, ev):
        ctx.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}

    def execute(self, ctx):
        fs = []
        if self.files:
            for f in self.files:
                if f.name.lower().endswith('.xbg'):
                    fs.append(os.path.join(self.directory, f.name))
        elif self.filepath:
            fs.append(self.filepath)
        if not fs:
            self.report({'ERROR'}, "No .xbg file selected")
            return {'CANCELLED'}

        sep = bool(getattr(ctx.scene.xbg_debug_settings,
                           'separate_primitives', True))
        lod_sel = -1 if self.import_all_lods else self.lod_level
        ok = 0
        for fp in fs:
            try:
                model, arm = _wd_load(
                    ctx, fp, separate_primitives=sep,
                    lod_select=lod_sel,
                    import_mesh_only=self.import_mesh_only)
                nv = sum(len(m['verts']) for m in model['meshes'])
                navail = model.get('n_lods_available')
                lod_msg = ("LODs ALL" if self.import_all_lods
                           else "LOD %d" % min(self.lod_level,
                                               (navail or 1) - 1))
                self.report({'INFO'},
                    f"{os.path.basename(fp)}: {model['source'].upper()} "
                    f"[{lod_msg}] — {len(model['bones'])} bones, "
                    f"{len(model['meshes'])} meshes, {nv} verts")
                ok += 1
            except Exception as exc:
                self.report({'ERROR'}, f"{os.path.basename(fp)}: {exc}")
                import traceback; traceback.print_exc()
        return {'FINISHED'} if ok else {'CANCELLED'}



class XBG_OT_ImportWDMab(bpy.types.Operator):
    """Import a Watch Dogs 1 .mab animation (full rotation decode)."""
    bl_idname  = "xbg.import_wd_mab"
    bl_label   = "Import WD1 MAB"
    bl_description = (
        "Select an imported WD1 armature, then pick a .mab. Decodes the "
        "constant-pose bones AND the compressed per-keyframe rotation "
        "bitstream (DisruptEditor codec) and keys every animated bone"
    )
    bl_options = {'REGISTER', 'UNDO'}

    filepath: bpy.props.StringProperty(subtype="FILE_PATH")
    filter_glob: bpy.props.StringProperty(default="*.mab", options={'HIDDEN'})

    def invoke(self, ctx, ev):
        ctx.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}

    def execute(self, ctx):
        from .import_mab_wd import parse_wd1_mab, apply_wd1_mab
        from .import_wd import parse_wd1_xbg
        arm = ctx.active_object
        if arm is None or arm.type != 'ARMATURE':
            arm = next((o for o in ctx.selected_objects
                        if o.type == 'ARMATURE'), None)
        if arm is None:
            self.report({'ERROR'}, "Select a WD1 armature first")
            return {'CANCELLED'}
        src = arm.get('xbg_source_file', '')
        if not src or not os.path.isfile(src):
            self.report({'ERROR'},
                "Armature has no source .xbg recorded — re-import the model "
                "(newer importer stores the path)")
            return {'CANCELLED'}
        try:
            mab = parse_wd1_mab(self.filepath)
            model = parse_wd1_xbg(src)
            _ds = ctx.scene.xbg_debug_settings
            applied, missing = apply_wd1_mab(
                ctx, mab, arm, model['bones'],
                smooth_resample=getattr(_ds, 'mab_smooth_resample', True),
                resample_fps=getattr(_ds, 'mab_resample_fps', 60),
                emulate_helpers=getattr(_ds, 'mab_emulate_helpers', True),
                twist_bake=getattr(_ds, 'mab_twist_bake', True))
            msg = (f"WD1 MAB: {applied} bones keyed "
                   f"({mab['n_animated']} animated + "
                   f"{len(mab['const_rots'])} constant), "
                   f"{len(mab['key_times'])} keyframes / "
                   f"{mab['duration']:.2f}s")
            if missing:
                msg += f"  ({len(missing)} bone hashes not on this rig)"
            self.report({'WARNING' if missing else 'INFO'}, msg)
            return {'FINISHED'}
        except Exception as exc:
            self.report({'ERROR'}, f"Failed to import WD1 .mab: {exc}")
            import traceback; traceback.print_exc()
            return {'CANCELLED'}


class XBG_OT_InjectWD(bpy.types.Operator):
    """Write edited WD1 mesh vertices back into the source .xbg (in place)."""
    bl_idname  = "xbg.inject_wd_model"
    bl_label   = "Inject WD1 Mesh"
    bl_description = (
        "Patch the selected WD1-imported meshes' edited vertex positions / "
        "normals / UVs / colors back into a copy of the source .xbg. "
        "Same vertex count patches in place; changed counts trigger a full "
        "buffer rebuild (add/remove geometry, re-skin, drop unselected)"
    )
    bl_options = {'REGISTER'}

    filepath: bpy.props.StringProperty(subtype="FILE_PATH")
    filter_glob: bpy.props.StringProperty(default="*.xbg", options={'HIDDEN'})
    check_existing: bpy.props.BoolProperty(default=True, options={'HIDDEN'})

    def invoke(self, ctx, ev):
        objs = [o for o in ctx.selected_objects if o.get('wd_src')]
        if not objs:
            objs = [o for o in ctx.scene.objects if o.get('wd_src')]
        if objs and not self.filepath:
            src = objs[0]['wd_src']
            base, ext = os.path.splitext(src)
            self.filepath = base + "_edited" + ext
        ctx.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}

    def execute(self, ctx):
        from .inject_wd import inject_wd1_objects, rebuild_wd1_objects
        # The SELECTION is the keep-list: only the selected imported meshes
        # are written; unselected ones are DROPPED (their geometry is removed
        # — select all-but-the-eyes to bake a smaller eyeless head).  Select
        # nothing imported and it falls back to keeping every mesh.
        objs = [o for o in ctx.selected_objects if o.get('wd_src')]
        drop_mode = bool(objs)
        if not objs:
            objs = [o for o in ctx.scene.objects if o.get('wd_src')]
        if not objs:
            self.report({'ERROR'},
                "No WD1-imported meshes — import a Watch Dogs .xbg first")
            return {'CANCELLED'}
        src = objs[0]['wd_src']
        objs = [o for o in objs if o.get('wd_src') == src]
        if not self.filepath:
            self.report({'ERROR'}, "Choose an output .xbg path")
            return {'CANCELLED'}

        # New standalone objects (a fresh mesh you added, e.g. a second head)
        # carry no import metadata, so the injector can't place them on their
        # own.  Tell the user to JOIN them into an imported mesh first.
        new_objs = [o for o in ctx.selected_objects
                    if o.type == 'MESH' and not o.get('wd_src')
                    and not o.get('wd_joined')]
        if new_objs:
            self.report({'WARNING'},
                "%d new object(s) (%s) have no import data and will NOT be "
                "injected — join them (Ctrl+J) into an imported mesh, making "
                "the imported mesh the active object, then inject"
                % (len(new_objs), ", ".join(o.name for o in new_objs[:3])))
        reskin = bool(getattr(ctx.scene.xbg_debug_settings,
                              'wd_reskin_weights', False))
        recalc_norms = bool(getattr(ctx.scene.xbg_debug_settings,
                                    'wd_recalculate_normals', False))
        multibuffer = any(o.get('wd_multibuffer') for o in objs)
        # Streamed-LOD0 meshes (bytes in the companion .xbgmip) can only be
        # patched in place — the rebuild path repacks the .xbg's own buffers
        # and would leave the mip file inconsistent.
        has_mip = any(o.get('wd_mip_src') for o in objs)
        changed = any('wd_vcount' in o.keys()
                      and len(o.data.vertices) != int(o['wd_vcount'])
                      for o in objs if o.type == 'MESH')
        try:
            # Multi-buffer vehicles (helicopter, etc.): rebuild can't safely
            # repack their split buffers — force in-place only.
            if multibuffer or has_mip:
                if drop_mode or reskin:
                    self.report({'WARNING'},
                        "Multi-buffer vehicle: count changes, drop and "
                        "re-skin are not supported — using in-place inject")
                n_obj, n_vtx, warns = inject_wd1_objects(
                    objs, self.filepath, source_path=src,
                    recalculate_normals=recalc_norms)
                mode = ("in-place (multi-buffer)" if multibuffer
                        else "in-place (streamed LOD0)")
            elif drop_mode or changed or reskin:
                n_obj, n_vtx, warns = rebuild_wd1_objects(
                    objs, self.filepath, source_path=src, reskin=reskin,
                    drop_unselected=drop_mode,
                    recalculate_normals=recalc_norms)
                mode = "rebuild" + ("+reskin" if reskin else "")
            else:
                n_obj, n_vtx, warns = inject_wd1_objects(
                    objs, self.filepath, source_path=src,
                    recalculate_normals=recalc_norms)
                mode = "in-place"
            for w in warns:
                self.report({'WARNING'}, w)
                print("[WD1 inject]", w)
            self.report({'INFO'},
                "WD1 inject [%s]: %d meshes, %d verts -> %s%s"
                % (mode, n_obj, n_vtx, os.path.basename(self.filepath),
                   "  (%d notes — see console)" % len(warns) if warns else ""))
            return {'FINISHED'}
        except Exception as exc:
            self.report({'ERROR'}, f"Failed to inject WD1 .xbg: {exc}")
            import traceback; traceback.print_exc()
            return {'CANCELLED'}


class XBG_OT_WDPeekLODs(bpy.types.Operator):
    """Quickly scan a WD1 .xbg to count how many LODs it contains."""
    bl_idname  = "xbg.wd_peek_lods"
    bl_label   = "Check WD1 LOD Count"
    bl_description = (
        "Scan a Watch Dogs 1 .xbg to show how many LOD levels it contains "
        "and how many are stored in the file vs streamed externally"
    )

    filepath: bpy.props.StringProperty(subtype="FILE_PATH")
    filter_glob: bpy.props.StringProperty(default="*.xbg", options={'HIDDEN'})

    def invoke(self, ctx, ev):
        ctx.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}

    def execute(self, ctx):
        from .import_wd import parse_wd1_xbg
        if not self.filepath or not os.path.isfile(self.filepath):
            self.report({'ERROR'}, "No valid .xbg file selected")
            return {'CANCELLED'}
        try:
            m = parse_wd1_xbg(self.filepath)
            total = m.get('n_lods_total', 0)
            avail = m.get('n_lods_available', 0)
            skip  = m.get('lod_skip', 0)
            fn = os.path.basename(self.filepath)
            if skip:
                result = (f"{fn}: {total} LOD defs, {avail} in file "
                          f"(LOD 0-{skip-1} are external streams; "
                          f"file holds LOD {skip}-{total-1})")
            else:
                result = (f"{fn}: {total} LOD defs, all {avail} in file "
                          f"(LOD 0-{total-1})")
            ctx.scene.xbg_debug_settings.lod_peek_result = result
            self.report({'INFO'}, result)
        except Exception as e:
            result = f"Error: {e}"
            ctx.scene.xbg_debug_settings.lod_peek_result = result
            self.report({'WARNING'}, f"Could not read file: {e}")
        return {'FINISHED'}


class XBG_OT_WDSyncNormals(bpy.types.Operator):
    """Bake Blender geometry normals into xbg_normal so injection writes
    normals that match the sculpted/edited mesh shape."""
    bl_idname  = "xbg.wd_sync_normals"
    bl_label   = "Sync Normals from Geometry"
    bl_description = (
        "After sculpting or moving vertices, run this to bake Blender's "
        "computed normals into the xbg_normal attribute.  The next injection "
        "will then write normals that match the new vertex positions."
    )
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, ctx):
        return (ctx.active_object is not None
                and ctx.active_object.type == 'MESH')

    def execute(self, ctx):
        from .inject_wd import sync_normals_to_geometry
        count = 0
        for obj in ctx.selected_objects:
            if obj.type == 'MESH' and obj.get('wd_src'):
                n = sync_normals_to_geometry(obj)
                count += n
        if count:
            self.report({'INFO'},
                f"Synced normals for {count} vertices "
                f"across {len(ctx.selected_objects)} object(s)")
        else:
            self.report({'WARNING'},
                "No WD1-imported meshes selected "
                "(import a Watch Dogs .xbg first)")
        return {'FINISHED'}
