import bpy
import math
import mathutils
import os
import struct

from .binary_avatar import BinaryReader, detect_endian, LE, BE
from .skeleton_avatar import Skeleton, XMLSkeletonParser, parse_skeleton_chunk, parse_mb2o_chunk
from .import_mesh_avatar import parse_mesh_vertices, parse_sdol_chunk, parse_dnks_chunk
from .bounds_avatar import parse_xobb, parse_hpsb
from .import_uv_avatar import apply_uv_layer, flip_face_winding
from .import_weights_avatar import apply_vertex_weights, remap_skin_indices
from .normals_avatar import apply_split_normals, store_tangent_attributes
from .vertex_colors_avatar import apply_vertex_colors
from .import_materials_avatar import XBMParser
from .nodes_avatar import BlenderMaterialSetup
from .import_xbt_avatar import XBTConverter
from .blender_pipeline_avatar import create_armature, create_meshes
from ..Core.debug import (
    VerboseLogger as vlog,
    create_format_bounds_lattice,
    create_bounding_visualizations,
    refresh_bounds_display,
    auto_smooth_normals,
    display_file_info
)
from .bounds_editor_avatar import read_bounds as _read_bounds, _fill_from_bounds


# --------------------------------------------------------------------------
# Version sniffer — picks which parser to use based on the file header.
# --------------------------------------------------------------------------

# Engine versions seen across the Dunia-engine family.
# All XBG files start with `HSEM` + a u32 version marker at offset 4.
_VERSION_AVATAR_FC2 = 0x0006002A   # Avatar (2009) / Far Cry 2 (2008)
_VERSION_FC3        = 0x00030034   # Far Cry 3 (2012)
_VERSION_FC4        = 0x00060037   # Far Cry 4 (2014)
_VERSION_FC5        = 0x000D0047   # Far Cry 5 (2018) / New Dawn (2019)


def _detect_xbg_version(filepath):
    """Read the first 8 bytes and return the u32 version marker, or None."""
    try:
        with open(filepath, 'rb') as f:
            head = f.read(8)
    except OSError:
        return None
    if len(head) < 8 or head[:4] != b'HSEM':
        return None
    return struct.unpack_from('<I', head, 4)[0]


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
        self.pmcp_offset = 0
        # Byte order of the source file: '<' for PC (LE), '>' for PS3 (BE).
        # Auto-detected by BinaryReader, propagated onto every imported object
        # so the injector can write the same endianness back out on export.
        self.endian = LE


def _resolve_dnks_pos(lod_grp, sub_idx, name_index, sub_mesh_list):
    """Resolve the within-block index into `sub_mesh_list[lod_grp]`.

    The verified game model is `sub_mesh_list[lod_grp][sub_idx]`:
      - `lod_grp` (SDOL field) selects the DNKS *block*.  parse_dnks_chunk
        returns a FLAT list of blocks, one per (part × damage-state × LOD)
        group.  A single-group character (kendra) has one block per LOD
        (lod_grp 0..3); a drivable vehicle (buggy: 18 blocks) has lod_grp
        running 0..17.
      - `sub_idx` (SDOL field) is the submesh's index WITHIN that block.

    This is correct for every ORIGINAL file.  The drivable-vehicle bug:
    the importer previously used `name_index` (the SDOL flat position
    within a LOD, 0..N) as the in-block index.  Vehicle blocks hold only
    1-4 submeshes, so every submesh whose flat position exceeded its
    block size failed the range check and got NO geometry — leaving only
    the first part-group's submeshes (the "only the steering wheel
    imported" symptom).

    Fallback to `name_index` only when `sub_idx` is out of the block's
    range — that covers files re-injected by the OLD inject which
    synthesised non-positional sub_idx values (since superseded by the
    positional-sub_idx inject fix; see AGENTS.md "sub_idx field").

    Returns the in-block index, or None when nothing resolves.
    """
    if lod_grp is None or lod_grp < 0 or lod_grp >= len(sub_mesh_list):
        return None
    block = sub_mesh_list[lod_grp]
    n = len(block)
    if sub_idx is not None and 0 <= sub_idx < n:
        return sub_idx
    if name_index is not None and 0 <= name_index < n:
        return name_index
    return None


class XBGParser:
    def __init__(self, fn):
        self.filename = fn
        self.data = XBGData()
    
    def parse(self, lod=0, use_mb2o=False):  # NEW: use_mb2o parameter
        vlog.log(f"\n{'='*60}\nPARSING XBG FILE: {os.path.basename(self.filename)}\n{'='*60}")

        # Auto-detect endianness BEFORE opening the reader so we can use the
        # right struct format for every subsequent int/float read.  PS3 files
        # store everything big-endian; PC files little-endian.  See
        # binary.detect_endian for the heuristic.
        self.data.endian = detect_endian(self.filename)
        endian_label = "Big-endian (PS3)" if self.data.endian == BE else "Little-endian (PC)"
        vlog.log(f"\nDetected byte order: {endian_label}")
        try:
            from ..Core.debug import TraceLogger
        except Exception:
            TraceLogger = None
        if TraceLogger is not None:
            import os as _os
            TraceLogger.info(
                f"[import] file='{self.filename}' size={_os.path.getsize(self.filename)} "
                f"endian={'BE/PS3' if self.data.endian == BE else 'LE/PC'} target_lod={lod}",
                event="import_entry",
                data={"file": str(self.filename),
                      "size": _os.path.getsize(self.filename),
                      "endian": "BE" if self.data.endian == BE else "LE",
                      "target_lod": int(lod),
                      "use_mb2o": bool(use_mb2o)})

        with BinaryReader(self.filename, endian=self.data.endian) as g:
            g.word(4)
            cc = g.i(7)[6]
            vlog.log(f"\nFile Header:\n  Chunk Count: {cc}")
            if TraceLogger is not None:
                TraceLogger.kv("header.chunk_count", cc, tier="DEBUG",
                                event="import_header")

            for m in range(cc):
                back = g.tell()
                # chunk_name() handles BE byte-reversal so the dispatch below
                # always works against the canonical LE names (SDOL, EDON, …).
                try:
                    chunk = g.chunk_name()
                    ci = g.i(2)
                    cs = ci[1]
                except Exception as exc:
                    import traceback as _tb
                    tb_text = _tb.format_exc()
                    if TraceLogger is not None:
                        TraceLogger.info(
                            f"[import] !!! chunk header[{m}] read failed at offset "
                            f"0x{back:08X}: {exc.__class__.__name__}: {exc}",
                            event="import_chunk_header_failed",
                            data={"chunk_index": m,
                                  "offset":      int(back),
                                  "chunks_read_so_far": len(self.data.chunks),
                                  "exc_type":   exc.__class__.__name__,
                                  "exc_msg":    str(exc)[:512],
                                  "traceback":  tb_text})
                    vlog.warn(f"\n*** chunk[{m}] header read failed @ 0x{back:X}: {exc}")
                    for tbline in tb_text.splitlines():
                        vlog.warn(f"    {tbline}")
                    raise
                self.data.chunks.append((chunk, back, cs))
                vlog.log_chunk(chunk, back, cs)
                if TraceLogger is not None:
                    TraceLogger.debug(
                        f"  [import] chunk[{m}] {chunk!r} @ 0x{back:08X}  size={cs}",
                        event="import_chunk",
                        data={"index": m, "magic": str(chunk),
                              "offset": int(back), "size": int(cs)})
                
                if chunk == 'PMCP':
                    self.data.pmcp_offset = back
                    g.i(2)
                    unk, self.data.vert_pos_scale = g.f(2)
                    vlog.log_pmcp(self.data.vert_pos_scale, unk)
                    if TraceLogger is not None:
                        TraceLogger.kvblock(
                            "PMCP (vertex scale)",
                            [
                                ("pos_scale",       self.data.vert_pos_scale),
                                ("unk_float",       unk),
                                ("int16_half_m",    self.data.vert_pos_scale * 32767),
                                ("chunk_offset",    f"0x{back:08X}"),
                            ],
                            tier="DEBUG", event="import_pmcp")

                elif chunk == 'PMCU':
                    g.i(2)
                    self.data.uv_trans, self.data.uv_scale = g.f(2)
                    vlog.log_pmcu(self.data.uv_trans, self.data.uv_scale)
                    if TraceLogger is not None:
                        TraceLogger.kvblock(
                            "PMCU (UV scale)",
                            [
                                ("uv_trans",        self.data.uv_trans),
                                ("uv_scale",        self.data.uv_scale),
                                ("max_uv_at_int16", 32767 * self.data.uv_scale + self.data.uv_trans),
                                ("chunk_offset",    f"0x{back:08X}"),
                            ],
                            tier="DEBUG", event="import_pmcu")

                elif chunk == 'EDON':
                    parse_skeleton_chunk(g, self.data.skeleton)
                    if TraceLogger is not None:
                        bones = self.data.skeleton.bones
                        # Sample table of the first 12 bones (and the last 3 if longer)
                        head_rows = [(i, b.name, b.parent_id,
                                       tuple(round(v,4) for v in (b.local_position or (0,0,0))))
                                      for i, b in enumerate(bones[:12])]
                        tail_rows = []
                        if len(bones) > 15:
                            tail_rows = [(i, b.name, b.parent_id,
                                          tuple(round(v,4) for v in (b.local_position or (0,0,0))))
                                         for i, b in enumerate(bones[-3:],
                                                                start=len(bones)-3)]
                        TraceLogger.kvblock(
                            "EDON (skeleton)",
                            [
                                ("bone_count",      len(bones)),
                                ("first_bone",      bones[0].name if bones else None),
                                ("last_bone",       bones[-1].name if bones else None),
                                ("chunk_offset",    f"0x{back:08X}"),
                            ],
                            tier="DEBUG", event="import_edon")
                        if head_rows:
                            TraceLogger.table(
                                "EDON bones (head)",
                                ("idx","name","parent_id","local_pos"),
                                head_rows + ([("...","...","...","...")] + tail_rows if tail_rows else []),
                                tier="DEBUG", event="import_edon_bones")

                elif chunk == 'MB2O':  # NEW: Parse MB2O chunk
                    self.data.bind_matrices = parse_mb2o_chunk(g)
                    if TraceLogger is not None:
                        TraceLogger.kvblock(
                            "MB2O (inverse-bind matrices)",
                            [
                                ("matrix_count",     len(self.data.bind_matrices)),
                                ("first_translation",
                                 (tuple(round(v,4)
                                  for v in self.data.bind_matrices[0].translation)
                                  if self.data.bind_matrices else None)),
                                ("chunk_offset",     f"0x{back:08X}"),
                            ],
                            tier="DEBUG", event="import_mb2o")

                elif chunk == 'DIKS':
                    g.i(2)
                    self.data.lod_count = g.i(1)[0]
                    vlog.log(f"\n=== DIKS CHUNK ===\nLOD Count: {self.data.lod_count}")
                    if TraceLogger is not None:
                        TraceLogger.kv("DIKS lod_count", self.data.lod_count,
                                       tier="DEBUG", event="import_diks")
                    # Skip the per-LOD metadata block (4 bytes per LOD).
                    # The outer chunk seek-back below re-aligns regardless, but
                    # we still consume the bytes here for symmetry with other
                    # readers.  Previously: [g.H(2) or g.B(4) for _ in range(N)]
                    # — the `or g.B(4)` was dead code (g.H(2) returns a
                    # non-empty tuple which is always truthy in Python).
                    for _ in range(self.data.lod_count):
                        g.H(2)

                elif chunk == 'LTMR':
                    w = g.i(4)
                    mc = w[2]
                    vlog.log(f"\n=== LTMR CHUNK (Materials) ===\nMaterial Count: {mc}")
                    mat_rows = []
                    for mat_i in range(mc):
                        nl = g.i(1)[0]
                        mf = g.word(nl)
                        sn = mf.split('/')[-1].replace('.mat', '') or f"Material_{mat_i}"
                        self.data.materials.append(sn)
                        vlog.log_material(mat_i, sn, mf)
                        g.b(1)
                        mat_rows.append((mat_i, sn, mf))
                    if TraceLogger is not None:
                        TraceLogger.table(
                            "LTMR materials",
                            ("idx","short_name","full_path"),
                            mat_rows, tier="DEBUG", event="import_ltmr")

                elif chunk == 'SDOL':
                    _meshes_before = len(self.data.meshes)
                    parse_sdol_chunk(g, self.data.meshes, self.data.lod_names)
                    if TraceLogger is not None:
                        TraceLogger.kv(
                            "SDOL meshes loaded",
                            len(self.data.meshes) - _meshes_before,
                            tier="DEBUG", event="import_sdol")
                        # Per-mesh detail
                        rows = []
                        for mi, mesh in enumerate(self.data.meshes[_meshes_before:]):
                            rows.append((mi, mesh.lod_level, mesh.part_number,
                                         mesh.vb_index, mesh.vert_count,
                                         getattr(mesh, "vert_stride", "?"),
                                         f"0x{mesh.vert_section_offset:X}"
                                         if mesh.vert_section_offset else "?"))
                        if rows:
                            TraceLogger.table(
                                "SDOL submeshes parsed",
                                ("idx","lod","part","vb_idx","verts","stride","vert_off"),
                                rows, tier="DEBUG", event="import_sdol_submeshes")

                elif chunk == 'DNKS':
                    (
                        self.data.sub_mesh_list,
                        self.data.lod_names,
                        self.data.lod_name_bboxes,
                    ) = parse_dnks_chunk(g, self.data.lod_count)
                    if TraceLogger is not None:
                        per_lod = [(li, len(blk))
                                    for li, blk in enumerate(self.data.sub_mesh_list)]
                        TraceLogger.kvblock(
                            "DNKS (skinning blocks)",
                            [
                                ("lod_blocks",       len(self.data.sub_mesh_list)),
                                ("submeshes_per_lod_block", per_lod),
                                ("lod_names",        dict(
                                    (k, v[0] if v else None)
                                    for k, v in self.data.lod_names.items())),
                                ("chunk_offset",     f"0x{back:08X}"),
                            ],
                            tier="DEBUG", event="import_dnks")
                        # Per-LOD bbox samples
                        for lod_i, entries in self.data.lod_name_bboxes.items():
                            rows = [(i, e[3], e[2], tuple(round(v,3) for v in e[0]),
                                      tuple(round(v,3) for v in e[1]))
                                     for i, e in enumerate(entries)]
                            TraceLogger.table(
                                f"DNKS trailing names (LOD {lod_i})",
                                ("idx","name","metric","bb_min","bb_max"),
                                rows, tier="DEBUG",
                                event=f"import_dnks_trailing_lod{lod_i}")

                elif chunk == 'XOBB':
                    bbox = parse_xobb(g, ci[1])
                    if bbox:
                        self.data.bounding_boxes.append(bbox)
                        # Bug fix: only stamp meshes not yet assigned to a bbox
                        for mesh in self.data.meshes:
                            if mesh.xobb_chunk_offset == 0:
                                mesh.xobb_chunk_offset = back
                        if TraceLogger is not None:
                            TraceLogger.kvblock(
                                "XOBB (bounding box)",
                                [
                                    ("min", tuple(round(v,4) for v in bbox.min)),
                                    ("max", tuple(round(v,4) for v in bbox.max)),
                                    ("size", tuple(round(bbox.max[i] - bbox.min[i], 4)
                                                    for i in range(3))),
                                    ("chunk_offset", f"0x{back:08X}"),
                                ],
                                tier="DEBUG", event="import_xobb")

                elif chunk == 'HPSB':
                    sphere = parse_hpsb(g, ci[1])
                    if sphere:
                        self.data.bounding_spheres.append(sphere)
                        # Bug fix: only stamp meshes not yet assigned to a sphere
                        for mesh in self.data.meshes:
                            if mesh.hpsb_chunk_offset == 0:
                                mesh.hpsb_chunk_offset = back
                        if TraceLogger is not None:
                            TraceLogger.kvblock(
                                "HPSB (bounding sphere)",
                                [
                                    ("center", tuple(round(v,4) for v in sphere.center)),
                                    ("radius", round(sphere.radius, 4)),
                                    ("chunk_offset", f"0x{back:08X}"),
                                ],
                                tier="DEBUG", event="import_hpsb")

                g.seek(back + ci[1])

            # NEW: Apply MB2O matrices if enabled and available
            if self.data.bind_matrices and use_mb2o:
                vlog.log(f"\n  MB2O enabled - applying bind matrices to skeleton...")
                self.data.skeleton.apply_bind_matrices(self.data.bind_matrices, self.data.sub_mesh_list)
            elif self.data.bind_matrices and not use_mb2o:
                vlog.log(f"\n  MB2O disabled - using EDON transforms only")
            elif not self.data.bind_matrices:
                vlog.log(f"\n  No MB2O data found in file")
            
            if TraceLogger is not None:
                TraceLogger.kvblock(
                    "All chunks parsed — pipeline summary",
                    [
                        ("chunk_count_in_header", cc),
                        ("chunks_walked",         len(self.data.chunks)),
                        ("materials_total",       len(self.data.materials)),
                        ("meshes_total",          len(self.data.meshes)),
                        ("skeleton_bones",        len(self.data.skeleton.bones)),
                        ("mb2o_matrices",         len(self.data.bind_matrices)),
                        ("lod_count",             self.data.lod_count),
                        ("bounding_boxes",        len(self.data.bounding_boxes)),
                        ("bounding_spheres",      len(self.data.bounding_spheres)),
                    ],
                    tier="DEBUG", event="import_all_chunks_done")

            # Each pipeline stage is wrapped individually so on a failure
            # we know EXACTLY which one died.  Without these wraps the
            # traceback just shows the inner stack, not which top-level
            # step (_filter_lod, _process_mesh_vertices, …) was running.
            def _stage(name, fn, *args, **kw):
                if TraceLogger is not None:
                    # DEBUG (gated by verbose) to match the stage-end marker
                    # below — was INFO, which spammed the console on every
                    # import regardless of settings.
                    TraceLogger.debug(f"[import] >>> stage: {name}",
                                      event="import_stage_begin",
                                      data={"stage": name})
                try:
                    return fn(*args, **kw)
                except Exception as exc:
                    import traceback as _tb
                    tb_text = _tb.format_exc()
                    if TraceLogger is not None:
                        TraceLogger.info(
                            f"[import] !!! stage '{name}' raised "
                            f"{exc.__class__.__name__}: {exc}",
                            event="import_stage_failed",
                            data={"stage": name,
                                  "exc_type": exc.__class__.__name__,
                                  "exc_msg":  str(exc)[:1024],
                                  "traceback": tb_text})
                    # Mirror to text log so it's visible without jsonl.
                    # Use warn() (always writes) so failures show up even
                    # when Verbose Logging is OFF.
                    vlog.warn(f"\n*** Import stage '{name}' FAILED: "
                              f"{exc.__class__.__name__}: {exc}")
                    for tbline in tb_text.splitlines():
                        vlog.warn(f"    {tbline}")
                    raise
                finally:
                    if TraceLogger is not None:
                        TraceLogger.debug(f"[import] <<< stage: {name} done",
                                          event="import_stage_end",
                                          data={"stage": name})

            import time
            _stage("_filter_lod",            self._filter_lod, lod)
            if TraceLogger is not None:
                TraceLogger.kv("meshes_after_lod_filter", len(self.data.meshes),
                                tier="DEBUG", event="import_after_lod_filter")
            _stage("_process_mesh_vertices", self._process_mesh_vertices, g)
            if TraceLogger is not None:
                # Per-mesh vertex/UV/skin coverage.  vert_count is the
                # SDOL header field; vert_pos_list / vert_uv_list /
                # skin_weight_list are what the binary read produced.
                rows = []
                for mi, mesh in enumerate(self.data.meshes):
                    vp = getattr(mesh, "vert_pos_list", None) or []
                    uv = getattr(mesh, "vert_uv_list",  None) or []
                    sw = getattr(mesh, "skin_weight_list", None) or []
                    rows.append((mi, mesh.lod_level, mesh.part_number,
                                 mesh.vert_count, len(vp), len(uv), len(sw)))
                TraceLogger.table(
                    "Per-mesh vertex data after parse",
                    ("idx","lod","part","vc_header","verts","uvs","skin_w"),
                    rows, tier="DEBUG", event="import_verts_parsed")
            _stage("_remap_skin_indices",    self._remap_skin_indices, g)
            _stage("_process_mesh_faces",    self._process_mesh_faces, g)
            if TraceLogger is not None:
                # Final per-mesh tally — verts AND faces.
                # MeshPrimitive doesn't support len(); use .indices/3
                # to get its triangle count.  vert source: prefer the
                # parsed vert_pos_list (mesh.vertices is a Mesh attr).
                rows = []
                _total_v = 0; _total_f = 0
                for mi, mesh in enumerate(self.data.meshes):
                    vp = getattr(mesh, "vert_pos_list", None)
                    nv = len(vp) if vp is not None else 0
                    nf = 0
                    for p in (getattr(mesh, "primitives", []) or []):
                        idx = getattr(p, "indices", None)
                        if idx is not None:
                            nf += len(idx) // 3
                    _total_v += nv; _total_f += nf
                    rows.append((mi, mesh.lod_level, mesh.part_number, nv, nf))
                TraceLogger.table(
                    "Per-mesh totals after face processing",
                    ("idx","lod","part","verts","faces"),
                    rows, tier="DEBUG", event="import_faces_done")
                TraceLogger.kvblock(
                    "Import grand totals",
                    [("meshes", len(self.data.meshes)),
                     ("total_verts", _total_v),
                     ("total_faces", _total_f)],
                    tier="DEBUG", event="import_totals")

        vlog.log(f"\n{'='*60}\nPARSING COMPLETE\n{'='*60}\n")
        if TraceLogger is not None:
            TraceLogger.struct("import_complete",
                                {"file": str(self.filename),
                                 "meshes": len(self.data.meshes),
                                 "materials": len(self.data.materials),
                                 "bones": len(self.data.skeleton.bones)},
                                tier="INFO")
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
        """Remap palette-slot indices → global bone IDs across all submeshes.

        When multiple meshes share the same vertex buffer (same lod_level +
        vert_section_offset), the remap must run ONCE per buffer using the
        combined mat_list_info of all meshes; otherwise the second mesh
        would re-remap already-remapped indices.  The actual rewrite loop
        lives in `weights.remap_skin_indices` — keep this method focused
        on the grouping + sharing logic only.
        """
        # Group meshes by vertex buffer (same LOD + same byte offset)
        vb_groups = {}  # (lod, offset) -> [meshes...]
        for mesh in self.data.meshes:
            vb_groups.setdefault(
                (mesh.lod_level, mesh.vert_section_offset), []
            ).append(mesh)

        for meshes in vb_groups.values():
            if not meshes:
                continue

            ref_mesh = meshes[0]
            if not ref_mesh.skin_indice_list:
                continue

            # Combine mat_list_info from every mesh sharing this VB so the
            # palette walk covers the whole buffer in vertex-order.
            #
            # The 3rd field of each tuple is the in-block DNKS index.  We
            # resolve it the same way _process_mesh_faces does (game model
            # sub_mesh_list[lod_grp][sub_idx], with name_index fallback for
            # old non-positional re-injects) via _resolve_dnks_pos.  Using
            # the wrong index here looks up the wrong palette AND vert_count,
            # which makes vert_id_start drift inside remap_skin_indices and
            # corrupts every subsequent slice's weights too — this was part
            # of the drivable-vehicle bug (multi-block DNKS).
            all_mat_info = []
            for m in meshes:
                for info in m.mat_list_info:
                    lg, si_val = info[1], info[2]
                    pos = _resolve_dnks_pos(
                        lg, si_val, m.name_index, self.data.sub_mesh_list)
                    if pos is None:
                        all_mat_info.append(info)
                    else:
                        # (vb_idx, lod_grp, DNKS_POS, idx_offset, idx_count)
                        all_mat_info.append(
                            (info[0], info[1], pos, info[3], info[4]))

            remap_skin_indices(
                ref_mesh.skin_indice_list,
                all_mat_info,
                self.data.sub_mesh_list,
            )

            # Share the now-remapped data with every other mesh on this VB
            for mesh in meshes[1:]:
                mesh.skin_indice_list = ref_mesh.skin_indice_list
                mesh.skin_weight_list = ref_mesh.skin_weight_list
    
    def _process_mesh_faces(self, g):
        """Parse face index buffers using a single bulk read per submesh.

        Previously called g.H(3) once per triangle (one read+unpack per face).
        Now reads the entire index run in one call and slices in Python.

        DNKS-key fix (2026-05):
        --------------------------------------------------------------------
        `info[2]` is the SDOL `sub_idx` VALUE — a 32-bit identifier the
        engine uses to address submeshes for animation / cloth / ragdoll
        lookups.  In ORIGINAL Avatar / Dunia files the sub_idx happens to
        equal the SDOL positional index (0,1,2,…), so using it as an index
        into `sub_mesh_list[lg]` (which is stored in SDOL-position order
        per the verified DNKS model — see mesh.parse_dnks_chunk) coincides
        with the correct entry.

        In RE-INJECTED files the inject side synthesises new sub_idx
        values for split-by-material slices (e.g. [0,1,2,7,8,3,4,5,6]).
        sub_idx==3 now lives at SDOL position 5, but `sub_mesh_list[lg][3]`
        still returns the DNKS entry that owns SDOL position 3 — the wrong
        face_count and the wrong bone palette.  That over-read by tens of
        thousands of bytes is what caused the "vertex index 65534 out of
        range" IndexError on re-import (we slurped indices from past the
        slice, sometimes spilling into the next LOD's vertex data).

        Fix: prefer `mesh.name_index` (the true SDOL position written by
        parse_sdol_chunk) over `info[2]`.  Fallback to `info[2]` only when
        name_index is unset (-1) so old behaviour is preserved for any
        path that doesn't populate name_index.
        """
        vlog.log(f"\n=== PROCESSING MESH FACES ===")
        try:
            from ..Core.debug import TraceLogger
        except Exception:
            TraceLogger = None
        for mesh in self.data.meshes:
            for info in mesh.mat_list_info:
                lg, si_val = info[1], info[2]
                # Game model: sub_mesh_list[lod_grp][sub_idx].  sub_idx is
                # the submesh's index WITHIN its part-group block — correct
                # for both single-group characters and multi-block drivable
                # vehicles.  name_index (SDOL flat position) is only used as
                # a fallback for old non-positional re-injects.  See
                # _resolve_dnks_pos.
                dnks_pos = _resolve_dnks_pos(
                    lg, si_val, mesh.name_index, self.data.sub_mesh_list)
                if (TraceLogger is not None and dnks_pos is not None
                        and dnks_pos != si_val):
                    TraceLogger.debug(
                        f"  [import] DNKS key: LOD{mesh.lod_level} "
                        f"lod_grp={lg} sub_idx={si_val} → DNKS pos={dnks_pos}",
                        event="import_dnks_key_fix",
                        data={"lod": mesh.lod_level, "lod_grp": int(lg),
                              "sub_idx_val": int(si_val),
                              "dnks_pos": int(dnks_pos)})
                if dnks_pos is not None:
                    sm = self.data.sub_mesh_list[lg][dnks_pos]
                    mid = sm.header_data[0]
                    mn = self.data.materials[mid] if mid < len(self.data.materials) else f"Material_{mid}"

                    if sm.face_count > 0:
                        byte_offset = mesh.indice_section_offset + info[3] * 2
                        raw_count   = sm.face_count * 3  # indices (uint16 each)
                        g.seek(byte_offset)
                        raw_buf = g.raw(raw_count * 2)   # one read instead of face_count reads
                        # Index buffer endianness follows the file's overall byte order:
                        # PC = uint16 LE, PS3 = uint16 BE.
                        raw = struct.unpack_from(f'{g.endian}{raw_count}H', raw_buf)

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
        try:
            from ..Core.debug import TraceLogger
        except Exception:
            TraceLogger = None
        if TraceLogger is not None:
            TraceLogger.kvblock(
                "XBGBlenderImporter.load() entry",
                [
                    ("filepath",             fp),
                    ("lod",                  lod),
                    ("import_mesh_only",     imo),
                    ("data_folder",          df),
                    ("load_textures",        lt),
                    ("load_hd",              lhd),
                    ("flip_normals",         fn),
                    ("use_xml_assembly",     uxa),
                    ("separate_primitives",  sp),
                    ("show_format_bounds",   sfb),
                    ("import_as_dds",        iad),
                    ("use_mb2o",             use_mb2o),
                    ("compact_vertices",     compact_vertices),
                    ("reorient_bones",       reorient_bones),
                ],
                tier="DEBUG", event="loader_entry")

        # ── Version detection / dispatch ──────────────────────────────────
        # Dunia evolved across Avatar/FC2 → FC3 → FC4 → FC5 and the XBG
        # chunk layouts diverged.  The existing parser handles the
        # Avatar/FC2 format only.  Newer files are routed to the FC3/FC4
        # parser; FC5 is detected and refused with a clear message.
        version = _detect_xbg_version(fp)
        if version in (_VERSION_FC3, _VERSION_FC4, _VERSION_FC5):
            # Far Cry GEOM files have their own self-contained importer now.
            raise ValueError(
                f"This is a Far Cry GEOM .xbg (version 0x{version:08x}). "
                f"Use the Far Cry 3/4 importer (XBG Import panel → "
                f"\"Import FC3/FC4 Model\"), not the Avatar importer.")
        if version is not None and version != _VERSION_AVATAR_FC2:
            vlog.log(f"\n*** Unknown XBG version 0x{version:08x} — "
                     f"attempting Avatar/FC2 import path anyway ***")
        # Fall through to the existing Avatar/FC2 importer below.

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
            if TraceLogger is not None:
                TraceLogger.info("[import] creating armature…",
                                  event="loader_armature_begin",
                                  data={"bones": len(data.skeleton.bones)})
            ao = create_armature(data.skeleton, os.path.basename(fp), reorient_bones=reorient_bones)
            if ao:
                ao["xbg_source_file"] = fp
            if TraceLogger is not None:
                TraceLogger.kvblock(
                    "Armature created",
                    [
                        ("name",       ao.name if ao else None),
                        ("bone_count", len(ao.data.bones) if ao else 0),
                    ],
                    tier="DEBUG", event="loader_armature_done")
        elif TraceLogger is not None:
            TraceLogger.info("[import] skipping armature (mesh-only mode)",
                              event="loader_armature_skipped")

        if TraceLogger is not None:
            TraceLogger.info(f"[import] creating {len(data.meshes)} mesh object(s)…",
                              event="loader_create_meshes_begin",
                              data={"meshes_to_create": len(data.meshes),
                                    "load_textures": lt})
        mos = create_meshes(
            data.meshes, ao, data.materials, imo, df, lt, lhd,
            xb, xm2b, xmi2b, xmi2n, sp, fp,
            data.vert_pos_scale, data.uv_trans, data.uv_scale, iad, data.lod_names,
            compact_vertices,
            data.lod_name_bboxes,
            data.pmcp_offset,
            data.sub_mesh_list,
            fn,
            data.endian,
            uxa,   # Use XML Assembly -> reassemble+bind rigid weapon/vehicle parts
        )
        if TraceLogger is not None:
            TraceLogger.kvblock(
                "Meshes created in Blender",
                [
                    ("objects_created", len(mos) if mos else 0),
                    ("object_names",    [o.name for o in (mos or [])][:12]),
                ],
                tier="DEBUG", event="loader_create_meshes_done")

        # When separate_primitives is OFF, join all created mesh objects into one
        # and weld shared boundary vertices with merge-by-distance.
        # Only produce separate objects when sp=True.
        if not sp and mos and len(mos) > 1:
            ds = ctx.scene.xbg_debug_settings
            vlog.log(f"\n=== JOINING {len(mos)} MESH OBJECTS INTO ONE ===")
            _v_before = sum(len(o.data.vertices) for o in mos)
            _f_before = sum(len(o.data.polygons) for o in mos)
            bpy.ops.object.select_all(action='DESELECT')
            for obj in mos:
                obj.select_set(True)
            bpy.context.view_layer.objects.active = mos[0]
            # join() deletes the absorbed objects but leaves their mesh
            # datablocks orphaned in bpy.data — capture and purge them.
            _victim_meshes = [o.data for o in mos[1:]]
            bpy.ops.object.join()
            joined_obj = bpy.context.active_object
            for _m in _victim_meshes:
                if _m.users == 0:
                    bpy.data.meshes.remove(_m)
            merge_dist = ds.merge_distance
            bpy.ops.object.mode_set(mode='EDIT')
            bpy.ops.mesh.select_all(action='SELECT')
            bpy.ops.mesh.remove_doubles(threshold=merge_dist)
            bpy.ops.object.mode_set(mode='OBJECT')
            joined_obj["xbg_joined"] = True  # flag: merged import, re-inject not available
            vlog.log(f"  Joined into: {joined_obj.name} (merge distance: {merge_dist})")
            if TraceLogger is not None:
                _v_after = len(joined_obj.data.vertices)
                _f_after = len(joined_obj.data.polygons)
                TraceLogger.kvblock(
                    "Merged join + remove-doubles",
                    [
                        ("inputs",            _v_before),
                        ("verts_before_join", _v_before),
                        ("faces_before_join", _f_before),
                        ("verts_after_merge", _v_after),
                        ("faces_after_merge", _f_after),
                        ("verts_welded",      _v_before - _v_after),
                        ("merge_distance",    merge_dist),
                        ("final_object",      joined_obj.name),
                    ],
                    tier="DEBUG", event="loader_join_done")
            mos = [joined_obj]

        ds = ctx.scene.xbg_debug_settings
        # Populate the editable bounds props from THIS model so the live
        # gizmo + the editor fields (now unified under the Bounding Volume
        # Display panel) reflect it. Reuse the bounds editor's own read so
        # the chunk OFFSETS are captured too (needed for Save). Setting these
        # props fires refresh_bounds_display via their update callbacks, so
        # the box/sphere appears immediately if its Show flag is on.
        try:
            sc = ctx.scene
            _b = _read_bounds(open(fp, 'rb').read(), data.endian)
            # A mesh's own matrix_world (NOT its parent armature's) is the
            # correct display frame — see _bounds_display_frame in
            # Core/debug.py for why. Must be set BEFORE _fill_from_bounds,
            # since assigning its fields fires refresh_bounds_display.
            sc.xbg_bounds_frame_obj = mos[0].name if mos else ''
            # Always fill (this RESETS xbg_has_xobb/hpsb to match THIS model,
            # so a model without bounds clears a previous model's gizmo).
            _fill_from_bounds(sc, _b)
            if _b['xobb'] or _b['hpsb']:
                sc.xbg_bounds_path = fp
                sc.xbg_bounds_endian = 'BE' if data.endian == BE else 'LE'
            refresh_bounds_display(sc)
        except Exception as _be:
            vlog.log(f"[bounds] live-display populate skipped: {_be}")

        ds.auto_smooth_normals and mos and auto_smooth_normals(mos)

        XBTConverter.cleanup_temp_files()
        vlog.log(f"\n{'#'*60}\n# XBG IMPORT COMPLETE\n{'#'*60}\n")
        if TraceLogger is not None:
            TraceLogger.struct("loader_finished",
                                {"objects_in_scene": len(mos) if mos else 0,
                                 "object_names":     [o.name for o in (mos or [])],
                                 "file":             str(fp)},
                                tier="INFO")

        return {'FINISHED'}
