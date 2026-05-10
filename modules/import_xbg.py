import bpy
import math
import mathutils
import os
import struct

from .binary import BinaryReader
from .skeleton import Skeleton, XMLSkeletonParser, parse_skeleton_chunk, parse_mb2o_chunk
from .mesh import Mesh, parse_mesh_vertices, parse_sdol_chunk, parse_dnks_chunk
from .bounds import parse_xobb, parse_hpsb
from .uv import apply_uv_coordinates
from .weights import apply_vertex_weights, remap_skin_indices
from .materials import XBMParser
from .nodes import BlenderMaterialSetup
from .xbt import XBTConverter
from .debug import (
    VerboseLogger as vlog,
    create_format_bounds_lattice,
    create_bounding_visualizations,
    flip_normals as dbg_flip,
    auto_smooth_normals,
    display_file_info
)


class XBGData:
    def __init__(self):
        self.skeleton = Skeleton()
        self.meshes = []
        self.sub_mesh_list = []
        self.materials = []
        self.lod_count = 0
        self.lod_names = {}        # {lod_index: [name, ...]}
        self.vert_pos_scale = 1.0
        self.uv_trans = 0.0
        self.uv_scale = 1.0
        self.bounding_boxes = []
        self.bounding_spheres = []
        self.chunks = []
        self.bind_matrices = []
        # per-name bboxes {lod_index: [(bbox_min, bbox_max, metric, name)]}
        self.lod_name_bboxes = {}


class XBGParser:
    def __init__(self, fn):
        self.filename = fn
        self.data = XBGData()
    
    def parse(self, lod=0, use_mb2o=False):  # NEW: use_mb2o parameter
        vlog.log(f"\n{'='*60}\nPARSING XBG FILE: {os.path.basename(self.filename)}\n{'='*60}")
        
        with BinaryReader(self.filename) as g:
            g.word(4)
            cc = g.i(7)[6]
            vlog.log(f"\nFile Header:\n  Chunk Count: {cc}")
            
            for m in range(cc):
                back = g.tell()
                chunk = g.word(4)
                ci = g.i(2)
                cs = ci[1]
                self.data.chunks.append((chunk, back, cs))
                vlog.log_chunk(chunk, back, cs)
                
                if chunk == 'PMCP':
                    g.i(2)
                    unk, self.data.vert_pos_scale = g.f(2)
                    vlog.log_pmcp(self.data.vert_pos_scale, unk)
                
                elif chunk == 'PMCU':
                    g.i(2)
                    self.data.uv_trans, self.data.uv_scale = g.f(2)
                    vlog.log_pmcu(self.data.uv_trans, self.data.uv_scale)
                
                elif chunk == 'EDON':
                    parse_skeleton_chunk(g, self.data.skeleton)
                
                elif chunk == 'MB2O':  # NEW: Parse MB2O chunk
                    self.data.bind_matrices = parse_mb2o_chunk(g)
                
                elif chunk == 'DIKS':
                    g.i(2)
                    self.data.lod_count = g.i(1)[0]
                    vlog.log(f"\n=== DIKS CHUNK ===\nLOD Count: {self.data.lod_count}")
                    [g.H(2) or g.B(4) for _ in range(self.data.lod_count)]
                
                elif chunk == 'LTMR':
                    w = g.i(4)
                    mc = w[2]
                    vlog.log(f"\n=== LTMR CHUNK (Materials) ===\nMaterial Count: {mc}")
                    for m in range(mc):
                        nl = g.i(1)[0]
                        mf = g.word(nl)
                        sn = mf.split('/')[-1].replace('.mat', '') or f"Material_{m}"
                        self.data.materials.append(sn)
                        vlog.log_material(m, sn, mf)
                        g.b(1)
                
                elif chunk == 'SDOL':
                    parse_sdol_chunk(g, self.data.meshes, self.data.lod_names)
                
                elif chunk == 'DNKS':
                    (
                        self.data.sub_mesh_list,
                        self.data.lod_names,
                        self.data.lod_name_bboxes,
                    ) = parse_dnks_chunk(g, self.data.lod_count)
                
                elif chunk == 'XOBB':
                    bbox = parse_xobb(g, ci[1])
                    if bbox:
                        self.data.bounding_boxes.append(bbox)
                        # Bug fix: only stamp meshes not yet assigned to a bbox
                        for mesh in self.data.meshes:
                            if mesh.xobb_chunk_offset == 0:
                                mesh.xobb_chunk_offset = back
                
                elif chunk == 'HPSB':
                    sphere = parse_hpsb(g, ci[1])
                    if sphere:
                        self.data.bounding_spheres.append(sphere)
                        # Bug fix: only stamp meshes not yet assigned to a sphere
                        for mesh in self.data.meshes:
                            if mesh.hpsb_chunk_offset == 0:
                                mesh.hpsb_chunk_offset = back
                
                g.seek(back + ci[1])
            
            # NEW: Apply MB2O matrices if enabled and available
            if self.data.bind_matrices and use_mb2o:
                vlog.log(f"\n  MB2O enabled - applying bind matrices to skeleton...")
                self.data.skeleton.apply_bind_matrices(self.data.bind_matrices, self.data.sub_mesh_list)
            elif self.data.bind_matrices and not use_mb2o:
                vlog.log(f"\n  MB2O disabled - using EDON transforms only")
            elif not self.data.bind_matrices:
                vlog.log(f"\n  No MB2O data found in file")
            
            self._filter_lod(lod)
            self._process_mesh_vertices(g)
            self._remap_skin_indices(g)
            self._process_mesh_faces(g)
        
        vlog.log(f"\n{'='*60}\nPARSING COMPLETE\n{'='*60}\n")
        return self.data
    
    def _filter_lod(self, lod):
        if lod == -1:
            vlog.log("\nImporting all LODs and all Parts")
            return
        
        # Get LOD name if available (use first name from that LOD)
        lod_display = self.data.lod_names.get(lod, [f"LOD{lod}"])[0] if lod in self.data.lod_names and self.data.lod_names[lod] else f"LOD{lod}"
        vlog.log(f"\nFiltering to {lod_display} (LOD {lod}) only...")
        
        # Group meshes by (part_number, lod_level)
        groups = {}
        for mesh in self.data.meshes:
            key = (mesh.part_number, mesh.lod_level)
            if key not in groups:
                groups[key] = []
            groups[key].append(mesh)
        
        # Get all parts
        all_parts = set(m.part_number for m in self.data.meshes)
        filtered = []
        
        for part_num in sorted(all_parts):
            # Try to find meshes at the exact LOD for this part
            key = (part_num, lod)
            if key in groups:
                # Found! Add ALL meshes (including sub-parts) for this part at this LOD
                part_meshes = groups[key]
                filtered.extend(part_meshes)
                if len(part_meshes) > 1:
                    vlog.log(f"    P{part_num} at {lod_display}: {len(part_meshes)} sub-parts")
                else:
                    vlog.log(f"    P{part_num} at {lod_display}: Found")
            else:
                # Part not found at exact LOD - skip it instead of falling back
                vlog.log(f"    P{part_num}: {lod_display} unavailable, skipping")
        
        self.data.meshes = filtered
    
    def _process_mesh_vertices(self, g):
        [parse_mesh_vertices(g, mesh, self.data.vert_pos_scale, self.data.uv_trans, self.data.uv_scale) for mesh in self.data.meshes]
    
    def _remap_skin_indices(self, g):
        """Remap bone indices from palette to global bone IDs
        
        CRITICAL: When multiple meshes share the same vertex buffer,
        process all submesh palettes for that vertex buffer, then share.
        """
        # Group meshes by vertex buffer
        vb_groups = {}  # (lod, offset) -> list of meshes
        for mesh in self.data.meshes:
            vb_key = (mesh.lod_level, mesh.vert_section_offset)
            if vb_key not in vb_groups:
                vb_groups[vb_key] = []
            vb_groups[vb_key].append(mesh)
        
        # Process each vertex buffer once
        for vb_key, meshes in vb_groups.items():
            if not meshes:
                continue
            
            # Use the first mesh as reference
            ref_mesh = meshes[0]
            if not ref_mesh.skin_indice_list:
                continue
            
            # Collect all mat_list_info from all meshes sharing this VB
            # and remap using all relevant palettes
            all_mat_info = []
            for mesh in meshes:
                all_mat_info.extend(mesh.mat_list_info)
            
            # Sort by vertex range to process in order
            # Each mat_list_info has: (vb_idx, lod_grp, sub_idx, idx_offset, idx_count)
            vert_id_start = 0
            for info in all_mat_info:
                lod_grp, sub_idx = info[1], info[2]
                if lod_grp < len(self.data.sub_mesh_list):
                    submesh = self.data.sub_mesh_list[lod_grp][sub_idx] if sub_idx < len(self.data.sub_mesh_list[lod_grp]) else None
                    if submesh:
                        count = submesh.header_data[5]
                        palette = submesh.bone_data
                        end = min(vert_id_start + count, len(ref_mesh.skin_indice_list))
                        for v_idx in range(vert_id_start, end):
                            ref_mesh.skin_indice_list[v_idx] = tuple(
                                (palette[r] if r < len(palette) and palette[r] != -1 else 0) 
                                for r in ref_mesh.skin_indice_list[v_idx]
                            )
                        vert_id_start += count
            
            # Share the remapped data with all meshes using this VB
            for mesh in meshes[1:]:
                mesh.skin_indice_list = ref_mesh.skin_indice_list
                mesh.skin_weight_list = ref_mesh.skin_weight_list
    
    def _process_mesh_faces(self, g):
        """Parse face index buffers using a single bulk read per submesh.

        Previously called g.H(3) once per triangle (one read+unpack per face).
        Now reads the entire index run in one call and slices in Python.
        """
        vlog.log(f"\n=== PROCESSING MESH FACES ===")
        for mesh in self.data.meshes:
            for info in mesh.mat_list_info:
                lg, si = info[1], info[2]
                if lg < len(self.data.sub_mesh_list) and si < len(self.data.sub_mesh_list[lg]):
                    sm = self.data.sub_mesh_list[lg][si]
                    mid = sm.header_data[0]
                    mn = self.data.materials[mid] if mid < len(self.data.materials) else f"Material_{mid}"

                    if sm.face_count > 0:
                        byte_offset = mesh.indice_section_offset + info[3] * 2
                        raw_count   = sm.face_count * 3  # indices (uint16 each)
                        g.seek(byte_offset)
                        raw_buf = g.raw(raw_count * 2)   # one read instead of face_count reads
                        raw = struct.unpack_from(f'<{raw_count}H', raw_buf)

                        # Filter degenerate triangles (any index == 0xFFFF)
                        idx = []
                        append = idx.append
                        for i in range(0, raw_count, 3):
                            a, b, c = raw[i], raw[i + 1], raw[i + 2]
                            if a != 65535 and b != 65535 and c != 65535:
                                append(a); append(b); append(c)

                        if idx:
                            mesh.add_primitive(idx, mid, mn)
                        vlog.log(f"  LOD{mesh.lod_level} Material '{mn}': {len(idx)//3} triangles")


class XBGBlenderImporter:
    def load(self, ctx, fp, lod=0, imo=False, df="", lt=True, lhd=True, fn=True, uxa=True, sp=False, sfb=False, iad=False, use_mb2o=False, compact_vertices=True, reorient_bones=False):
        vlog.log(f"\n{'#'*60}\n# XBG IMPORT STARTED\n# File: {os.path.basename(fp)}\n{'#'*60}")
        
        xb = {}
        xm2b = {}
        xmi2b = {}
        xmi2n = {}
        
        if uxa:
            xp = XMLSkeletonParser.find_xml_file(fp)
            if xp:
                xb, xm2b, xmi2b, xmi2n = XMLSkeletonParser.parse_xml_skeleton(xp)
        
        sp and vlog.log(f"\n*** SEPARATE PRIMITIVES MODE ENABLED ***")
        
        parser = XBGParser(fp)
        data = parser.parse(lod, use_mb2o)  # NEW: Pass use_mb2o parameter
        
        # Always store file info, but only display if checkbox enabled
        file_info_str = display_file_info(data.chunks, os.path.basename(fp), fp)
        ctx.scene.xbg_debug_settings.file_info_data = file_info_str
        
        sfb and data.vert_pos_scale and create_format_bounds_lattice(ctx, data.vert_pos_scale)
        
        ao = None
        if not imo:
            ao = self.create_armature(data.skeleton, os.path.basename(fp), reorient_bones=reorient_bones)
        
        mos = self.create_meshes(
            data.meshes, ao, data.materials, imo, df, lt, lhd,
            xb, xm2b, xmi2b, xmi2n, sp, fp,
            data.vert_pos_scale, data.uv_trans, data.uv_scale, iad, data.lod_names,
            compact_vertices,
            data.lod_name_bboxes,
        )

        # When separate_primitives is OFF, join all created mesh objects into one
        # and weld shared boundary vertices with merge-by-distance.
        # Only produce separate objects when sp=True.
        if not sp and mos and len(mos) > 1:
            ds = ctx.scene.xbg_debug_settings
            vlog.log(f"\n=== JOINING {len(mos)} MESH OBJECTS INTO ONE ===")
            bpy.ops.object.select_all(action='DESELECT')
            for obj in mos:
                obj.select_set(True)
            bpy.context.view_layer.objects.active = mos[0]
            bpy.ops.object.join()
            joined_obj = bpy.context.active_object
            merge_dist = ds.merge_distance
            bpy.ops.object.mode_set(mode='EDIT')
            bpy.ops.mesh.select_all(action='SELECT')
            bpy.ops.mesh.remove_doubles(threshold=merge_dist)
            bpy.ops.object.mode_set(mode='OBJECT')
            joined_obj["xbg_joined"] = True  # flag: merged import, re-inject not available
            vlog.log(f"  Joined into: {joined_obj.name} (merge distance: {merge_dist})")
            mos = [joined_obj]

        fn and mos and dbg_flip(mos)
        
        ds = ctx.scene.xbg_debug_settings
        (ds.show_bounding_box or ds.show_bounding_sphere) and mos and create_bounding_visualizations(
            ctx, data, mos, ds.show_bounding_box, ds.show_bounding_sphere, ds.bounds_display_type
        )
        
        ds.auto_smooth_normals and mos and auto_smooth_normals(mos)
        
        XBTConverter.cleanup_temp_files()
        vlog.log(f"\n{'#'*60}\n# XBG IMPORT COMPLETE\n{'#'*60}\n")
        
        return {'FINISHED'}
    
    def create_armature(self, skel, nb, reorient_bones=False):
        if skel.get_bone_count() == 0:
            return None
        
        vlog.log(f"\n=== CREATING ARMATURE ===")
        
        # Check if we should use MB2O bind matrices
        use_mb2o = any(bd.bind_matrix is not None for bd in skel.bones)
        if use_mb2o:
            vlog.log(f"Using MB2O inverse bind matrices for armature positioning")
        else:
            vlog.log(f"Using EDON hierarchy transforms for armature positioning")
        
        an = f"{nb}_Armature"
        ad = bpy.data.armatures.new(an)
        ao = bpy.data.objects.new(an, ad)
        
        bpy.context.collection.objects.link(ao)
        bpy.context.view_layer.objects.active = ao
        ao.rotation_euler = (0, 0, math.radians(180))
        vlog.log(f"Armature rotation: (0, 0, 180°)")
        
        bpy.ops.object.mode_set(mode='EDIT')
        eb = {}
        
        for i, bd in enumerate(skel.bones):
            bn = bd.name if bd.name else f"Bone_{i}"
            e = ad.edit_bones.new(bn)
            eb[i] = e
            
            # Use MB2O bind matrix if available, otherwise use EDON world matrix
            if use_mb2o and bd.bind_matrix is not None:
                # MB2O stores INVERSE bind matrices, so we need to invert them
                # to get the actual bind pose position
                try:
                    bind_pose_matrix = bd.bind_matrix.inverted()
                    e.head = mathutils.Vector(bind_pose_matrix.translation)
                    vlog.log(f"  Bone {i} ({bn}): Using MB2O position {e.head}")
                except:
                    # If matrix is singular/non-invertible, fall back to EDON
                    vlog.log(f"  WARNING: Bone {i} ({bn}): MB2O matrix non-invertible, using EDON")
                    e.head = mathutils.Vector(bd.world_matrix.translation) if bd.world_matrix else mathutils.Vector((0, 0, 0))
            else:
                # Use EDON transforms
                e.head = mathutils.Vector(bd.world_matrix.translation) if bd.world_matrix else mathutils.Vector((0, 0, 0))
            
            e.tail = e.head + mathutils.Vector((0, 0.5, 0))
        
        for i, bd in enumerate(skel.bones):
            e = eb[i]
            
            if bd.parent_id is not None and bd.parent_id in eb:
                e.parent = eb[bd.parent_id]
                e.use_connect = False
            
            # Calculate tail direction based on which matrix we're using
            if use_mb2o and bd.bind_matrix is not None:
                try:
                    bind_pose_matrix = bd.bind_matrix.inverted()
                    rot = bind_pose_matrix.to_quaternion()
                    off = mathutils.Vector((0, 1, 0)) * 0.5
                    off.rotate(rot)
                    e.tail = e.head + off
                except:
                    # Fall back to EDON rotation
                    if bd.world_matrix:
                        rot = bd.world_matrix.to_quaternion()
                        off = mathutils.Vector((0, 1, 0)) * 0.5
                        off.rotate(rot)
                        e.tail = e.head + off
            else:
                # Use EDON transforms
                if bd.world_matrix:
                    rot = bd.world_matrix.to_quaternion()
                    off = mathutils.Vector((0, 1, 0)) * 0.5
                    off.rotate(rot)
                    e.tail = e.head + off
        
        # Bone reorientation: point each bone's tail toward its children's heads.
        # IMPORTANT: must run BEFORE mode_set('OBJECT') — edit bone handles are
        # only valid while the armature stays in EDIT mode. Accessing eb[] after
        # leaving and re-entering EDIT mode causes a crash (stale C pointers).
        if reorient_bones:
            # Build children map: parent_index -> [valid child indices]
            # Exclude self-references: some XBG files store parent_id == own index on
            # the root bone (e.g. 0 -> 0).  Without this guard the root appears in its
            # own children list and the tail-direction average is corrupted.
            children_map = {}
            for j, bd in enumerate(skel.bones):
                pid = bd.parent_id
                if pid is not None and pid != j and pid in eb and j in eb:
                    children_map.setdefault(pid, []).append(j)

            MIN_BONE_LEN = 0.05  # prevent zero-length bones (Blender will crash)

            for i in eb:
                e  = eb[i]
                bd = skel.bones[i]
                pid = bd.parent_id
                has_real_parent = pid is not None and pid != i and pid in eb
                children = [ci for ci in children_map.get(i, []) if ci in eb]

                if has_real_parent and children:
                    # Interior bone: aim tail at the average of all direct child heads.
                    avg = mathutils.Vector()
                    for ci in children:
                        avg += eb[ci].head
                    avg /= len(children)

                    if (avg - e.head).length >= MIN_BONE_LEN:
                        e.tail = avg
                    # else: children collapsed onto this bone — keep world-matrix tail

                elif has_real_parent:
                    # Leaf / end-of-chain bone: extend away from parent along the
                    # parent->self direction (continues the visual line of the chain).
                    #
                    # Threshold note: MIN_BONE_LEN (0.05) is the minimum FINAL bone
                    # length Blender needs to avoid a zero-length crash, but it must NOT
                    # be used as the gate for whether we USE the away direction.
                    # e.g. wasp wing bones sit only 0.048 units from Pelvis — valid
                    # geometry, but 0.048 < 0.05 so the old >= MIN_BONE_LEN check was
                    # silently discarding their direction and falling back to the
                    # arbitrary world-matrix tail, producing giant off-screen bones.
                    #
                    # Fix: gate on > 0.001 (just avoids division-by-zero), then clamp
                    # the final length UP to MIN_BONE_LEN so Blender never sees a
                    # degenerate bone while still honouring the correct direction.
                    away = e.head - eb[pid].head
                    if away.length > 0.001:
                        e.tail = e.head + away.normalized() * max(away.length, MIN_BONE_LEN)
                    # else: head truly coincides with parent — keep world-matrix tail

                # else: root bone (no real parent) — keep the compact world-matrix tail
                # set in the first pass above.  Root bones are reference/origin bones;
                # stretching them toward their children creates a misleadingly large bone.

        bpy.ops.object.mode_set(mode='OBJECT')
        vlog.log(f"Created armature: {an}")
        
        return ao
    
    def _compact_mesh_data(self, mesh):
        """Remove unused vertices, keeping a new_to_old mapping for export correctness.

        Uses list comprehensions (faster than conditional per-loop appends) and
        builds both direction mappings in a single enumeration pass.
        """
        # Collect all vertex indices referenced by any face
        used_indices = set()
        for prim in mesh.primitives:
            used_indices.update(prim.indices)

        # Sort once; derive both mappings via enumerate (single pass)
        sorted_used = sorted(used_indices)
        old_to_new  = {old: new for new, old in enumerate(sorted_used)}
        new_to_old  = {new: old for new, old in enumerate(sorted_used)}

        # List-comprehension slicing — significantly faster than conditional .append()
        pl = mesh.vert_pos_list
        new_verts = [pl[i] for i in sorted_used]

        ul = mesh.vert_uv_list
        new_uvs = [ul[i] for i in sorted_used] if ul else []

        # UV1 / UV2 / Color: guard against shorter lists (some verts may be unused)
        uv1l = mesh.vert_uv1_list
        new_uv1s = [uv1l[i] for i in sorted_used if i < len(uv1l)] if uv1l else []

        uv2l = mesh.vert_uv2_list
        new_uv2s = [uv2l[i] for i in sorted_used if i < len(uv2l)] if uv2l else []

        cl = mesh.vert_color_list
        new_colors = [cl[i] for i in sorted_used if i < len(cl)] if cl else []

        wl = mesh.skin_weight_list
        new_weights = [wl[i] for i in sorted_used if i < len(wl)] if wl else []

        sl = mesh.skin_indice_list
        new_skin_indices = [sl[i] for i in sorted_used if i < len(sl)] if sl else []

        # Stash compacted secondary arrays back on the mesh for create_meshes
        mesh.vert_uv1_list   = new_uv1s
        mesh.vert_uv2_list   = new_uv2s
        mesh.vert_color_list = new_colors

        # Remap all primitive face indices to the compacted vertex space
        new_primitives = [
            ([old_to_new[i] for i in prim.indices], prim.material_index, prim.material_name)
            for prim in mesh.primitives
        ]

        removed_count = len(pl) - len(new_verts)
        vlog.log(f"  Vertex compaction: {len(pl)} -> {len(new_verts)} vertices ({removed_count} unused removed)")

        return new_verts, new_uvs, new_weights, new_skin_indices, new_primitives, new_to_old
    
    def create_meshes(self, meshes, ao, mns, imo=False, df="", lt=True, lhd=True,
                      xb={}, xm2b={}, xmi2b={}, xmi2n={}, sp=False, fp="",
                      vps=1.0, uvt=0.0, uvs=1.0, iad=False, lod_names={}, compact_vertices=True,
                      lod_name_bboxes={}):
        vlog.log(f"\n=== CREATING BLENDER MESHES ===")
        
        if compact_vertices:
            vlog.log("Vertex compaction ENABLED - removing unused vertices")
        else:
            vlog.log("Vertex compaction DISABLED - keeping all vertices (ghost vertices will be visible)")
        
        co = []
        
        for mi, mesh in enumerate(meshes):
            if not mesh.vert_pos_list:
                continue
            
            # Apply vertex compaction if enabled
            vertex_mapping = None  # Maps new index -> old index for export
            original_vert_count = len(mesh.vert_pos_list)
            
            if compact_vertices:
                # Compact the mesh and get the mapping
                verts, uv_coords, weights, skin_indices, primitives, vertex_mapping = self._compact_mesh_data(mesh)
            else:
                # Keep all vertices (old behavior)
                verts = mesh.vert_pos_list
                uv_coords = mesh.vert_uv_list if mesh.vert_uv_list else []
                weights = mesh.skin_weight_list if mesh.skin_weight_list else []
                skin_indices = mesh.skin_indice_list if mesh.skin_indice_list else []
                primitives = [(p.indices, p.material_index, p.material_name) for p in mesh.primitives]
            
            if sp:
                for pi, (indices, mat_idx, mat_name) in enumerate(primitives):
                    skinning_type = "Skinned" if mesh.has_skinning() else "Static"
                    
                    # Get actual mesh name from LOD names using the submesh index
                    if mesh.lod_level in lod_names and mesh.name_index < len(lod_names[mesh.lod_level]):
                        mesh_name = lod_names[mesh.lod_level][mesh.name_index]
                        mn = mesh_name
                    else:
                        # Fallback to generic name
                        lod_display = f"LOD{mesh.lod_level}"
                        if mesh.sub_part_index >= 0:
                            mn = f"Mesh_{lod_display}_P{mesh.part_number}_Sub{mesh.sub_part_index}_{skinning_type}"
                        else:
                            mn = f"Mesh_{lod_display}_P{mesh.part_number}_{skinning_type}"
                    
                    me = bpy.data.meshes.new(mn)
                    obj = bpy.data.objects.new(mn, me)
                    bpy.context.collection.objects.link(obj)
                    co.append(obj)
                    
                    # Convert vertex mapping to string keys for Blender compatibility
                    mapping_for_blender = None
                    if vertex_mapping:
                        mapping_for_blender = {str(k): v for k, v in vertex_mapping.items()}

                    obj["xbg_data"] = {
                        "filepath": fp,
                        "vert_offset": mesh.vert_section_offset,
                        "vert_stride": mesh.vert_stride,
                        "vert_count": original_vert_count,
                        "vert_format_flags": mesh.vert_format_flags,
                        "pos_scale": vps,
                        "uv_trans": uvt,
                        "uv_scale": uvs,
                        "lod_level": mesh.lod_level,
                        "import_mesh_only": imo,
                        "xobb_offset": mesh.xobb_chunk_offset,
                        "hpsb_offset": mesh.hpsb_chunk_offset,
                        "vertex_mapping": mapping_for_blender,
                    }

                    imo and setattr(obj, 'rotation_euler', (0, 0, math.radians(180)))

                    if ao:
                        obj.parent = ao
                        mod = obj.modifiers.new(name="Armature", type='ARMATURE')
                        mod.object = ao

                    faces = [(indices[i], indices[i+1], indices[i+2])
                             for i in range(0, len(indices), 3) if i+2 < len(indices)]

                    mrn = mat_name
                    mat = bpy.data.materials.get(mrn) or bpy.data.materials.new(name=mrn)
                    mat.use_nodes = True
                    obj.data.materials.append(mat)

                    me.from_pydata(verts, [], faces)
                    me.update()

                    # Apply UV0 via foreach_set (one C call per mesh)
                    if uv_coords:
                        uv_layer = me.uv_layers.new(name="UVMap")
                        loop_vis = [0] * len(me.loops)
                        me.loops.foreach_get("vertex_index", loop_vis)
                        uv_flat = [coord for vi in loop_vis for coord in uv_coords[vi]]
                        uv_layer.data.foreach_set("uv", uv_flat)

                    self._apply_uv_layer(me, mesh.vert_uv1_list, "UVMap1")
                    self._apply_uv_layer(me, mesh.vert_uv2_list, "UVMap2")
                    self._apply_vertex_colors(me, mesh.vert_color_list)

                    # Apply bone weights — pre-create all needed groups, cache lookup dict
                    if ao and weights and skin_indices:
                        bones = ao.data.bones
                        vg_cache = {}
                        for bone_data in skin_indices:
                            for bone_idx in bone_data:
                                if bone_idx < len(bones) and bone_idx not in vg_cache:
                                    bn = bones[bone_idx].name
                                    vg_cache[bone_idx] = (obj.vertex_groups.get(bn)
                                                          or obj.vertex_groups.new(name=bn))
                        for vert_idx, (weight_data, bone_data) in enumerate(zip(weights, skin_indices)):
                            for bone_idx, weight in zip(bone_data, weight_data):
                                if weight > 0 and bone_idx in vg_cache:
                                    vg_cache[bone_idx].add([vert_idx], weight / 255.0, 'REPLACE')

                    lt and df and self.setup_material_textures([(mat, mrn)], df, lhd, iad)
                    vlog.log(f"Created mesh: {mn} ({len(verts)} verts, {len(faces)} faces)")
            else:
                # Get actual mesh name from LOD names using the submesh index
                if mesh.lod_level in lod_names and mesh.name_index < len(lod_names[mesh.lod_level]):
                    # Use the actual name from the file!
                    mn = lod_names[mesh.lod_level][mesh.name_index]
                    vlog.log(f"  Using file name: {mn} (LOD{mesh.lod_level}, index {mesh.name_index})")
                else:
                    # Fallback to generic name
                    unique_parts = set(m.part_number for m in meshes)
                    is_multipart = len(unique_parts) > 1
                    skinning_type = "Skinned" if mesh.has_skinning() else "Static"
                    lod_display = f"LOD{mesh.lod_level}"
                    
                    # Build mesh name
                    if mesh.sub_part_index >= 0:
                        # Has sub-parts
                        if is_multipart:
                            mn = f"Mesh_{lod_display}_P{mesh.part_number}_Sub{mesh.sub_part_index}_{skinning_type}"
                        else:
                            mn = f"Mesh_{lod_display}_Sub{mesh.sub_part_index}_{skinning_type}"
                    else:
                        # No sub-parts
                        if is_multipart:
                            mn = f"Mesh_{lod_display}_P{mesh.part_number}_{skinning_type}"
                        else:
                            mn = f"Mesh_{lod_display}_{skinning_type}"
                
                me = bpy.data.meshes.new(mn)
                obj = bpy.data.objects.new(mn, me)
                bpy.context.collection.objects.link(obj)
                co.append(obj)
                
                # Convert vertex mapping to string keys for Blender compatibility
                mapping_for_blender = None
                if vertex_mapping:
                    mapping_for_blender = {str(k): v for k, v in vertex_mapping.items()}

                obj["xbg_data"] = {
                    "filepath": fp,
                    "vert_offset": mesh.vert_section_offset,
                    "vert_stride": mesh.vert_stride,
                    "vert_count": original_vert_count,
                    "vert_format_flags": mesh.vert_format_flags,
                    "pos_scale": vps,
                    "uv_trans": uvt,
                    "uv_scale": uvs,
                    "lod_level": mesh.lod_level,
                    "import_mesh_only": imo,
                    "xobb_offset": mesh.xobb_chunk_offset,
                    "hpsb_offset": mesh.hpsb_chunk_offset,
                    "vertex_mapping": mapping_for_blender,
                }

                imo and setattr(obj, 'rotation_euler', (0, 0, math.radians(180)))

                if ao:
                    obj.parent = ao
                    mod = obj.modifiers.new(name="Armature", type='ARMATURE')
                    mod.object = ao

                faces = []
                mm = {}
                m2s = []

                for indices, mat_idx, mat_name in primitives:
                    if mat_idx not in mm:
                        mrn = mat_name
                        mat = bpy.data.materials.get(mrn) or bpy.data.materials.new(name=mrn)
                        mat.use_nodes = True
                        obj.data.materials.append(mat)
                        mm[mat_idx] = len(obj.data.materials) - 1
                        m2s.append((mat, mrn))
                    [faces.append((indices[i], indices[i+1], indices[i+2]))
                     for i in range(0, len(indices), 3) if i+2 < len(indices)]

                me.from_pydata(verts, [], faces)
                me.update()

                # Assign polygon material indices via foreach_set (one C call)
                mat_index_flat = [0] * len(me.polygons)
                po = 0
                for indices, mat_idx, mat_name in primitives:
                    bmi = mm.get(mat_idx, 0)
                    nt  = len(indices) // 3
                    for i in range(nt):
                        if po + i < len(mat_index_flat):
                            mat_index_flat[po + i] = bmi
                    po += nt
                me.polygons.foreach_set("material_index", mat_index_flat)

                # Apply UV0 via foreach_set (one C call)
                if uv_coords:
                    uv_layer = me.uv_layers.new(name="UVMap")
                    loop_vis = [0] * len(me.loops)
                    me.loops.foreach_get("vertex_index", loop_vis)
                    uv_flat = [coord for vi in loop_vis for coord in uv_coords[vi]]
                    uv_layer.data.foreach_set("uv", uv_flat)

                self._apply_uv_layer(me, mesh.vert_uv1_list, "UVMap1")
                self._apply_uv_layer(me, mesh.vert_uv2_list, "UVMap2")
                self._apply_vertex_colors(me, mesh.vert_color_list)

                # Apply bone weights — pre-create all needed groups, cache lookup dict
                if ao and weights and skin_indices:
                    bones = ao.data.bones
                    vg_cache = {}
                    for bone_data in skin_indices:
                        for bone_idx in bone_data:
                            if bone_idx < len(bones) and bone_idx not in vg_cache:
                                bn = bones[bone_idx].name
                                vg_cache[bone_idx] = (obj.vertex_groups.get(bn)
                                                      or obj.vertex_groups.new(name=bn))
                    for vert_idx, (weight_data, bone_data) in enumerate(zip(weights, skin_indices)):
                        for bone_idx, weight in zip(bone_data, weight_data):
                            if weight > 0 and bone_idx in vg_cache:
                                vg_cache[bone_idx].add([vert_idx], weight / 255.0, 'REPLACE')

                lt and df and self.setup_material_textures(m2s, df, lhd, iad)
                vlog.log(f"Created mesh: {mn} ({len(verts)} verts, {len(faces)} faces)")
        
        return co
    
    # ------------------------------------------------------------------
    # FEATURE 2: Apply vertex colors as a Blender color attribute
    # ------------------------------------------------------------------
    def _apply_vertex_colors(self, me, col_src):
        """Apply RGBA vertex colors using foreach_set (single C-level call)."""
        if not col_src:
            return
        try:
            attr = me.color_attributes.new(name="Col", type='BYTE_COLOR', domain='POINT')
            # Build flat RGBA float array then push in one call instead of per-vertex assignment
            colors_flat = [c / 255.0 for rgba in col_src for c in rgba]
            attr.data.foreach_set("color", colors_flat)
        except Exception as e:
            vlog.log(f"  Vertex color import failed: {e}")

    # ------------------------------------------------------------------
    # FEATURE 3: Apply a UV layer from a list of [u, v] pairs
    # ------------------------------------------------------------------
    def _apply_uv_layer(self, me, uv_src, name):
        """Apply a UV channel via foreach_set (single C-level call per layer).

        Sentinel values (None, from -32768/-32768 raw data) are mapped to (0, 0).
        If all values are sentinel the layer is skipped entirely.
        """
        if not uv_src:
            return
        if all(uv is None for uv in uv_src):
            vlog.log(f"  {name}: all sentinel values, skipping layer")
            return
        try:
            uv_layer = me.uv_layers.new(name=name)
            # Fetch all loop→vertex mappings in one C call
            loop_vis = [0] * len(me.loops)
            me.loops.foreach_get("vertex_index", loop_vis)
            # Build flat [u,v, u,v, ...] array; sentinel verts default to (0,0)
            default = (0.0, 0.0)
            uv_flat = [
                coord
                for vi in loop_vis
                for coord in (uv_src[vi] if vi < len(uv_src) and uv_src[vi] is not None else default)
            ]
            uv_layer.data.foreach_set("uv", uv_flat)
        except Exception as e:
            vlog.log(f"  {name} import failed: {e}")

    def setup_material_textures(self, m2s, df, lhd=True, iad=False):
        mf = os.path.join(df, "graphics", "_materials")
        
        for mat, mn in m2s:
            xfn = os.path.basename(mn)
            if not xfn.lower().endswith('.xbm'):
                xfn = xfn + '.xbm'
            
            xp = os.path.join(mf, xfn)
            if os.path.exists(xp):
                vlog.log(f"\nLoading XBM: {xfn}")
                xd = XBMParser.parse(xp, lhd)
                if xd:
                    BlenderMaterialSetup.setup_material(mat, xd, df, lhd, iad)
