"""Watch Dogs model importers.

Two formats, one entry point (`load_wd_model`):

* Watch Dogs 1 ``.xbg`` — binary 'GEOM' container, version 97.50.  Sequential
  serialized stream (NOT chunk-based like Avatar's MESH xbg).  This parser is
  a faithful Python port of DisruptEditor's xbgFile.cpp/IBinaryArchive
  (Jonathan Scott) with its PADDING_IBINARYARCHIVE alignment rules:
    - u16/u32/u64/f32 reads first align the cursor to their size
    - u8/bool reads are unaligned
    - string  = aligned u32 length + raw bytes (no terminator)
    - mat4    = pad(16) + 16 floats
    - ndVector = aligned u32 count + items  (u8 vector = count + raw block)
  The embedded ReflexSystem FCB blob is size-prefixed and skipped wholesale.
  End-of-parse is validated against the file size (same assert as the C++).

* Watch Dogs 2 ``.glm`` — the raw text GEOM source ("unconverted xbg").
  Tab-separated key/value blocks: SKELETON_LIST (bones with axis-angle
  rotations, parents by name), GEOMETRY_LIST/TRIMESH (indexed vertices,
  per-corner normals/UVs, faces with per-face material id, BLEND_LIST skin
  weights by bone name).

Both parse into one neutral model dict which `build_wd_model` turns into a
Blender armature + skinned meshes.
"""

import os
import re
import struct
import math

import numpy as np

try:
    import bpy
    import mathutils
except Exception:           # standalone (testing) mode
    bpy = None
    mathutils = None

try:
    from ..Core.debug import VerboseLogger as vlog
except Exception:
    class vlog:
        enabled = False
        @staticmethod
        def log(m): pass


# ---------------------------------------------------------------------------
# WD1 binary reader (IBinaryArchive, PADDING_IBINARYARCHIVE semantics)
# ---------------------------------------------------------------------------

class _Reader:
    __slots__ = ('d', 'p', '_dc_cap')

    def __init__(self, data):
        self.d = data
        self.p = 0
        self._dc_cap = None

    def tell(self):
        return self.p

    def size(self):
        return len(self.d)

    def pad(self, n):
        self.p += (n - self.p % n) % n

    def u8(self):
        v = self.d[self.p]; self.p += 1
        return v

    def boolean(self):
        return self.u8() != 0

    def u16(self):
        self.pad(2)
        v = struct.unpack_from('<H', self.d, self.p)[0]; self.p += 2
        return v

    def u32(self):
        self.pad(4)
        v = struct.unpack_from('<I', self.d, self.p)[0]; self.p += 4
        return v

    def f32(self):
        self.pad(4)
        v = struct.unpack_from('<f', self.d, self.p)[0]; self.p += 4
        return v

    def vec(self, n):
        self.pad(4)
        v = struct.unpack_from('<%df' % n, self.d, self.p); self.p += 4 * n
        return v

    def mat4(self):
        self.pad(16)
        v = struct.unpack_from('<16f', self.d, self.p); self.p += 64
        return v

    def string(self):
        ln = self.u32()
        if ln > 0x100000 or self.p + ln > len(self.d):
            raise ValueError("bad string length %d @0x%X" % (ln, self.p - 4))
        s = self.d[self.p:self.p + ln].decode('latin-1'); self.p += ln
        return s.rstrip('\x00')

    def name_id(self):
        """CMeshNameID = u32 CStringID hash + string."""
        self.u32()
        return self.string()

    def count(self):
        c = self.u32()
        if c > 0x200000:
            raise ValueError("implausible vector count %d @0x%X" % (c, self.p - 4))
        return c

    def u8_block(self, n):
        b = self.d[self.p:self.p + n]; self.p += n
        return b


def parse_wd1_xbg(path, lod_select=0):
    """Parse a Watch Dogs 1 .xbg (GEOM 97.50).  Returns the neutral model dict.

    `lod_select` chooses which level(s) of detail to decode:
      * 0 (default) — the best LOD actually present in the file (buffer 0)
      * N (1..)     — that LOD level, clamped to the lowest available
      * None / < 0  — ALL available LODs (one mesh set per buffer, names
                      suffixed _LOD0/_LOD1/…)
    LOD level here = SGfxBuffer index (0 = best present).  WD1 streams its
    highest-detail LOD(s) externally, so level 0 is buffer 0, not the
    absent file-LOD 0 (see the per-LOD buffer model below)."""
    data = open(path, 'rb').read()
    r = _Reader(data)

    # --- Header ---
    magic = r.u32()
    if magic != 0x47454F4D:                       # 'GEOM'
        raise ValueError("not a WD1 GEOM xbg (magic 0x%08X)" % magic)
    major, minor = r.u16(), r.u16()
    if (major, minor) != (97, 50):
        raise ValueError("unsupported GEOM version %d.%d" % (major, minor))
    r.u32(); r.u32(); r.u32()                     # header unk1..3
    r.u32(); r.u32()                              # SMemoryNeed
    r.f32()                                       # unk1 float
    r.boolean()                                   # unk2 bool

    # --- SceneGeometryParams ---
    # (byte offsets of the pos offset/scale + bounds fields are recorded so
    # the injector can REWRITE them when edited geometry exceeds the original
    # int16 quantization range — the Avatar-style bounds-expansion fix.)
    r.pad(4); gp_off1_pos = r.tell()
    gp_unk1 = r.u32()
    gp_scale_pos = r.tell()
    gp_unk2 = r.f32(); r.f32(); gp_unk4 = r.f32()
    gp_unk5 = r.f32(); r.f32()
    r.pad(4); bsphere_pos = r.tell()
    r.vec(3); r.f32()                             # bSphere
    bbox_pos = r.tell()
    r.vec(3); r.vec(3)                            # bBox
    r.u32(); r.u32(); r.u32()
    n_lods = r.count()
    lod_dists = [r.f32() for _ in range(n_lods)]
    r.f32()                                       # killDistance
    r.boolean(); r.boolean(); r.u8()

    # Decompression constants (see xbgFile::draw): pos = i16*off[1]+off[0],
    # uv = i16*off[3]+off[2].
    off = (float(gp_unk1), gp_unk2, gp_unk4, gp_unk5)

    # --- MaterialResources ---
    mr_unk0 = r.u32()
    for _ in range(n_lods - mr_unk0):
        r.f32()
    materials = []
    for _ in range(r.count()):
        r.u32()                                   # CPathID
        materials.append(r.string())

    # --- MaterialSlotToIndex ---
    for _ in range(r.count()):
        r.name_id(); r.u32()

    # --- SkinNames ---
    for _ in range(r.count()):
        r.name_id()

    # --- BonePalettes ---
    palettes = []
    for _ in range(r.count()):
        n = r.count()
        pal = [r.u16() for _ in range(n)]
        palettes.append(pal)

    # --- SkelResources ---
    bones = []
    if r.u32():
        for _ in range(r.count()):
            r.u8(); r.u8(); r.u8(); r.u8()        # boneLOD + unused[3]
            pos = r.vec(3)
            rot = r.vec(4)                        # quaternion x,y,z,w
            parent = r.u16()
            o2n = r.u16()                         # obj2NodeMatInd (node index)
            name = r.name_id()
            bones.append({
                'name': name,
                'parent': -1 if parent == 0xFFFF else parent,
                'pos': pos,
                'quat': (rot[3], rot[0], rot[1], rot[2]),   # (w,x,y,z)
                'o2n': o2n,
            })
        r.u32()                                   # unk2
        # obj2Node matrices — node-space transforms permuted by each bone's
        # obj2NodeMatInd field; NOT bind poses (verified: no convention of
        # these matches the accumulated bind positions).  Skip.
        n_mats = r.count()
        for _ in range(n_mats):
            r.mat4()

    # --- ReflexSystem: size-prefixed FCB blob — skip wholesale ---
    if r.u32():
        size = r.u32()
        r.p += size

    # --- SecondaryMotionObjects ---
    def _smo_prims():
        for _ in range(r.count()):                # spheres
            r.name_id(); r.mat4(); r.f32()
        for _ in range(r.count()):                # cylinders
            r.name_id(); r.mat4(); r.f32(); r.vec(3); r.vec(3)
        for _ in range(r.count()):                # capsules
            r.name_id(); r.mat4(); r.f32(); r.vec(3); r.vec(3)
        for _ in range(r.count()):                # planes
            r.name_id(); r.mat4(); r.vec(3); r.vec(3)

    def _smo_limits():
        for _ in range(r.count()):                # sphere limits
            r.name_id(); r.u16(); r.vec(3); r.f32()
        for _ in range(r.count()):                # box limits
            r.name_id(); r.u16(); r.vec(3); r.vec(3)
        for _ in range(r.count()):                # cylinder limits
            r.name_id(); r.u16(); r.vec(3); r.vec(3); r.f32(); r.f32()

    for _ in range(r.count()):
        # SSMSimulationParametersDesc
        r.vec(3)
        for _ in range(10):
            r.f32()
        r.u32(); r.u32(); r.boolean()
        _smo_prims()
        _smo_limits()
        for _ in range(r.count()):                # particles
            r.name_id(); r.f32(); r.boolean(); r.u16(); r.vec(2)
        for _ in range(r.count()):                # particle meshes
            r.name_id()
        for _ in range(r.count()):                # triangles
            r.u16(); r.u16(); r.u16()
        if r.count() != 0:                        # connectivities (always empty)
            raise ValueError("non-empty SMO connectivity @0x%X" % r.tell())
        for _ in range(r.count()):                # springs
            r.u16(); r.u16(); r.u16()
        r.u16(); r.boolean()

    # --- ProceduralNodes ---
    _PROC_FIELDS = {1: 'ifif', 2: 'if', 3: 'iif', 5: 'iifffffff', 6: 'iiiii'}
    for _ in range(r.count()):
        r.u16()
        t = r.u8()
        for f in _PROC_FIELDS.get(t, ''):
            (r.u32 if f == 'i' else r.f32)()

    # --- LODs ---
    lods = []
    for _li in range(n_lods):
        # capture every LOD0 drawcall (main + ranges) for the injector
        r._dc_cap = [] if _li == 0 else None
        meshes = []
        for _ in range(r.count()):
            r.vec(3); r.f32(); r.vec(3); r.vec(3)
            prim_type = r.u32()
            mat_id = r.u16()
            vformat = r.u16()
            stride = r.u8(); r.u8(); r.u16()
            bone_map = r.u32()
            dc = _basic_drawcall(r)
            n_ranges = r.u32()
            r.u32(); r.u32()
            ranges = []
            for _ in range(n_ranges):
                rdc = _basic_drawcall(r)
                r.vec(3); r.f32()                 # CSphere
                r.vec(3); r.vec(3)                # bbox
                rname = r.name_id()
                r.u16(); r.u16()
                ranges.append({'drawcall': rdc, 'name': rname})
            meshes.append({
                'prim_type': prim_type, 'mat_id': mat_id,
                'format': vformat, 'stride': stride,
                'bone_map': bone_map, 'drawcall': dc, 'ranges': ranges,
            })
        lods.append(meshes)
        if _li == 0:
            lod0_dc_cap = r._dc_cap
    r._dc_cap = None

    r.u32()                                       # unk3

    # --- SGfxBuffers ---
    buffers = []
    buf_offsets = []          # absolute file offset of each buffer's vdata
    buf_frames = []           # per-buffer framing offsets (for the injector)
    buffers_section_start = r.tell()              # the SGfxBuffers count u32
    n_buffers = r.count()
    for _ in range(n_buffers):
        r.pad(4)
        vsz_off = r.tell()
        vsz = r.count()
        vdata_off = r.tell()
        vdata = r.u8_block(vsz)
        r.pad(4)
        isz_off = r.tell()
        isz = r.count()
        idata_off = r.tell()
        idata = r.u8_block(isz)
        buffers.append((vdata, idata))
        buf_offsets.append(vdata_off)
        buf_frames.append({'vsz_off': vsz_off, 'vdata_off': vdata_off,
                           'vsz': vsz, 'isz_off': isz_off,
                           'idata_off': idata_off, 'isz': isz})
    buffers_section_end = r.tell()

    # --- GeomMips ---
    for _ in range(r.count()):
        r.u32(); r.u32(); r.u32(); r.string()

    r.u32()                                       # clothWrinkle...
    if r.tell() != r.size():
        raise ValueError("WD1 parse desync: ended at 0x%X of 0x%X"
                         % (r.tell(), r.size()))

    # ---- decode LOD0 meshes into the neutral model ----
    model = {
        'source': 'wd1',
        'name': os.path.splitext(os.path.basename(path))[0],
        'bones': bones,
        'meshes': [],
    }
    if not lods or not buffers:
        return model
    # WD1 stores each LOD's geometry in its own SGfxBuffer.  The highest-detail
    # LOD(s) are streamed externally and ABSENT from the .xbg, so there are
    # usually FEWER buffers than LOD levels.  buffer[i] holds LOD[skip + i],
    # where skip = n_lods - n_buffers; the best LOD actually present is
    # lods[skip], and it lives entirely in buffer 0 (verified byte-exact: its
    # submesh vertex extents sum to exactly len(buffer0)).  vb_offset /
    # index_start are offsets WITHIN that one buffer.
    #
    #   char01:           2 LODs, 2 buffers -> skip 0, buffer0 = LOD0.
    #   clothing/vehicles: 5-6 LODs, 4-5 buffers -> skip 1, buffer0 = LOD1.
    #
    # The old code decoded lods[0] against a CONCATENATION of every buffer.
    # That is only correct when skip == 0 (char01); for skip >= 1 it decoded
    # the absent LOD0's oversized drawcalls against mismatched bytes, which
    # produced the spiky exploded-geometry garbage on the helicopter, police
    # car, and clothing meshes.
    # ── Streamed high-detail LOD (.xbgmip companion) ─────────────────────
    # When skip >= 1 the TRUE LOD0 is streamed from "<base>.high.xbgmip"
    # next to the .xbg (its path is also embedded in the GeomMips section).
    # PIMG layout (verified byte-exact on helicopter_01.high.xbgmip):
    #   'PIMG' + u32 version(1) + u32 0 + u32 n_buffers(1) + u32 vsize
    #   + vsize vertex bytes + u32 isize + isize index bytes  == file end.
    # Prepending it to the buffer list makes buffer[0] = LOD0 again, so the
    # normal skip math decodes the real best LOD — and the injector patches
    # the mip file too (otherwise the game streams the pristine hi-res copy
    # over your edit as you approach: the "my edit reverted" bug).
    mip_path = None
    mip_vdata_off = 0
    skip0 = max(0, len(lods) - len(buffers))
    if skip0 >= 1:
        cand = os.path.splitext(path)[0] + ".high.xbgmip"
        if os.path.isfile(cand):
            try:
                mdat = open(cand, 'rb').read()
                if mdat[:4] == b'PIMG':
                    mver, mzero, mbufs, mvsz = struct.unpack_from('<4I', mdat, 4)
                    if mbufs == 1 and 20 + mvsz + 4 <= len(mdat):
                        misz = struct.unpack_from('<I', mdat, 20 + mvsz)[0]
                        if 20 + mvsz + 4 + misz == len(mdat):
                            mvd = mdat[20:20 + mvsz]
                            mid = mdat[24 + mvsz:24 + mvsz + misz]
                            buffers = [(mvd, mid)] + buffers
                            buf_offsets = [20] + buf_offsets
                            mip_path = cand
                            mip_vdata_off = 20
                            vlog.log("  [wd] streamed LOD0 loaded from %s "
                                     "(%d vtx bytes, %d idx bytes)"
                                     % (os.path.basename(cand), mvsz, misz))
            except Exception as exc:
                vlog.log("  [wd] .xbgmip read failed (%s) — falling back to "
                         "in-file LODs" % exc)

    skip = max(0, len(lods) - len(buffers))
    n_avail = len(buffers)
    model['n_lods_total'] = len(lods)
    model['n_lods_available'] = n_avail
    model['lod_skip'] = skip
    model['mip_path'] = mip_path

    # which SGfxBuffer(s) to decode (buffer b holds lods[skip + b])
    if lod_select is None or lod_select < 0:
        bufsel = list(range(n_avail))
    else:
        bufsel = [min(int(lod_select), n_avail - 1)] if n_avail else []
    multi = len(bufsel) > 1

    for b in bufsel:
        li = skip + b
        if li >= len(lods):
            continue
        vdata, idata = buffers[b]
        boff = buf_offsets[b] if b < len(buf_offsets) else 0
        indices = struct.unpack_from('<%dH' % (len(idata) // 2), idata)
        for mi, mesh in enumerate(lods[li]):
            dm = _decode_wd1_mesh(
                mesh, mi, vdata, indices, off, bones, palettes, materials)
            if multi:                       # disambiguate when importing all
                dm['name'] = '%s_LOD%d' % (dm['name'], b)
            dm['lod'] = b
            dc = mesh['drawcall']
            # injection: the decoded LOD lives wholly in its own buffer, so a
            # vertex's file position is simply buf0_off + vb_off + vi*stride —
            # no cross-buffer straddling (buf0_off = THIS LOD's buffer offset).
            from_mip = (mip_path is not None and b == 0)
            dm['inject'] = {
                'src': path,
                'vb_off': dc['vb_offset'],      # offset within the LOD buffer
                'stride': mesh['stride'],
                'format': mesh['format'],
                'scale': list(off),
                'vcount': dc['vertex_count'],
                'mesh_index': mi,
                'buf0_off': boff,               # file offset of this LOD's vdata
                # streamed LOD0: the bytes live in the companion .xbgmip, not
                # the .xbg — the injector patches (a copy of) that file.
                'mip_src': mip_path if from_mip else '',
            }
            model['meshes'].append(dm)

    # full layout for the count-changing rebuild (phase 2, inject_wd.py)
    # Per-buffer mesh tables (metadata only, no decode): buffer b holds
    # lods[skip + b]'s meshes — every (vb_off, stride, format, vcount) the
    # injector must re-encode when the global position scale changes.
    buffer_mesh_tables = []
    for b in range(len(buffers)):
        li = skip + b
        table = []
        if 0 <= li < len(lods):
            for mesh in lods[li]:
                dc = mesh['drawcall']
                table.append((dc['vb_offset'], mesh['stride'],
                              mesh['format'], dc['vertex_count']))
        buffer_mesh_tables.append(table)

    model['_layout'] = {
        'src': path,
        'scale': list(off),
        'lod_index': skip,               # best available LOD (rebuild target)
        'lod0_meshes': lods[skip] if skip < len(lods) else (lods[0] if lods else []),
        'buffers_section_start': buffers_section_start,
        'buffers_section_end': buffers_section_end,
        'buf_frames': buf_frames,
        'vdata0_off': buf_offsets[0] if buf_offsets else 0,
        'palettes': palettes,
        'bones': [b['name'] for b in bones],
        'node2bone': {b.get('o2n', i): i for i, b in enumerate(bones)},
        # for the bounds/scale expansion (inject_wd._maybe_expand_bounds):
        'header_offs': {'gp_off': gp_off1_pos, 'gp_scale': gp_scale_pos,
                        'bsphere': bsphere_pos, 'bbox': bbox_pos},
        'buf_offsets': list(buf_offsets),   # abs vdata offset per buffer
        'buffer_mesh_tables': buffer_mesh_tables,
        'mip_path': mip_path,               # buffer 0 lives here when set
        'mip_vdata_off': mip_vdata_off,
    }
    return model


def _basic_drawcall(r):
    """Read a basic drawcall, recording each field's byte offset on the
    reader (r._dc_offs) so the injector can patch counts in place."""
    offs = {}
    r.pad(4); offs['vb_offset'] = r.p; vb = r.u32()
    r.pad(4); offs['prim_count'] = r.p; pc = r.u32()
    r.pad(4); offs['index_count'] = r.p; ic = r.u32()
    r.pad(4); offs['index_start'] = r.p; isx = r.u32()
    r.pad(2); offs['vertex_count'] = r.p; vc = r.u16()
    r.pad(2); offs['min_index'] = r.p; mn = r.u16()
    r.pad(2); offs['max_index'] = r.p; mx = r.u16()
    r.pad(2); offs['group_count'] = r.p; gc = r.u16()
    dc = {'vb_offset': vb, 'prim_count': pc, 'index_count': ic,
          'index_start': isx, 'vertex_count': vc, 'min_index': mn,
          'max_index': mx, 'group_count': gc, '_offs': offs}
    cap = getattr(r, '_dc_cap', None)
    if cap is not None:
        cap.append(dc)
    return dc


def _u8n(x):
    """Compressed signed byte -> float (matches the engine/zmodeler mapping)."""
    return (x - 1) / 127.0 - 1.0


# Same mapping precomputed for all 256 byte values — avoids millions of
# per-component function calls in the per-vertex decode loop (identical result).
_U8N_LUT = [(i - 1) / 127.0 - 1.0 for i in range(256)]


def _decode_wd1_mesh(mesh, mi, vdata, indices, off, bones, palettes, materials):
    """Decode one CSceneMesh's vertex slice (port of createVertexBuffer).

    Component order (from the FVF layout): Position, UV1, UV2, UV3, Skin,
    Normal, Color, Tangent, Binormal, NormalModified.  Stride accounting is
    validated — a mismatch means an unknown flag combination.
    """
    fmt = mesh['format']
    stride = mesh['stride']
    dc = mesh['drawcall']
    start = dc['vb_offset']
    count = dc['vertex_count']

    point = fmt & 0x1
    point_comp = fmt & 0x2
    uv_full = fmt & 0x4
    uv_comp = fmt & 0x8
    skin = fmt & 0x10
    skin_extra = fmt & 0x20
    skin_rigid = fmt & 0x40
    normal_comp = fmt & 0x80
    color = fmt & 0x100
    tangent_comp = fmt & 0x200
    binormal_comp = fmt & 0x400
    uv_comp2 = fmt & 0x1000
    uv_comp3 = fmt & 0x2000
    normal_full = fmt & 0x4000
    normal_mod = fmt & 0x8000

    expected = (12 if point else 0) + (8 if point_comp else 0) \
        + (8 if uv_full else 0) + (4 if uv_comp else 0) \
        + (4 if uv_comp2 else 0) + (4 if uv_comp3 else 0) \
        + (8 if skin else 0) + (4 if skin_extra else 0) \
        + (4 if normal_comp else 0) + (12 if normal_full else 0) \
        + (4 if color else 0) + (4 if tangent_comp else 0) \
        + (4 if binormal_comp else 0) + (4 if normal_mod else 0)
    if expected != stride:
        raise ValueError("WD1 vertex format 0x%04X: computed stride %d != %d"
                         % (fmt, expected, stride))

    pal = palettes[mesh['bone_map']] if mesh['bone_map'] < len(palettes) else []

    # Palette entries are NODE indices (the obj2NodeMatInd domain), not bone
    # array indices.  Resolve through the inverse obj2NodeMatInd permutation
    # — verified on char01_head/legs (vertex-to-dominant-bone distance drops
    # to ~0.1 m with 0% outliers vs 61% mismatched under direct indexing;
    # direct indexing put left-forearm weights on the right arm).
    node2bone = {b.get('o2n', i): i for i, b in enumerate(bones)}

    # Resolve every palette slot -> bone name ONCE (was recomputed per skin
    # link — 738k+ calls).  slot_names[idx] is None for unresolved slots.
    slot_names = []
    for ni in pal:
        bi = node2bone.get(ni, ni)
        slot_names.append(bones[bi]['name'] if 0 <= bi < len(bones) else None)
    _ns = len(slot_names)

    def bone_name(idx):
        # idx is a palette SLOT; out-of-range must not crash.
        if idx is None or idx < 0 or idx >= _ns:
            return None
        return slot_names[idx]

    verts, uvs, uvs2, normals, colors = [], [], [], [], []
    tangents, tangents_w, binormals, binormals_w = [], [], [], []
    weights = {}
    for vi in range(count):
        k = start + vi * stride
        w_idx = 0.0
        if point:
            verts.append(struct.unpack_from('<3f', vdata, k))
            k += 12
        elif point_comp:
            px, py, pz, pw = struct.unpack_from('<4h', vdata, k)
            verts.append((px * off[1] + off[0], py * off[1] + off[0],
                          pz * off[1] + off[0]))
            w_idx = float(pw)
            k += 8
        else:
            verts.append((0.0, 0.0, 0.0))
        if uv_full:
            tu, tv = struct.unpack_from('<2f', vdata, k)
            uvs.append((tu, 1.0 - tv))
            k += 8
        elif uv_comp:
            tu, tv = struct.unpack_from('<2h', vdata, k)
            uvs.append((tu * off[3] + off[2], 1.0 - (tv * off[3] + off[2])))
            k += 4
        if uv_comp2:
            tu, tv = struct.unpack_from('<2h', vdata, k)
            uvs2.append((tu * off[3] + off[2], 1.0 - (tv * off[3] + off[2])))
            k += 4
        if uv_comp3:
            k += 4
        if skin:
            bw = vdata[k:k + 4]
            bi = vdata[k + 4:k + 8]
            k += 8
            if skin_extra:
                k += 4
            wl = []
            for w, ix in zip(bw, bi):
                if w:
                    nm = bone_name(ix)
                    if nm:
                        wl.append((nm, w / 255.0))
            if wl:
                weights[vi] = wl
        elif skin_rigid:
            # Rigid skinning: one bone per vertex.  The palette slot is
            # stored directly in position w (raw i16 value 0–N); confirmed
            # on helicopter_01 (pw in {0,1,3,4,5,6} for the 7-bone palette)
            # and police_01.  Earlier code divided by 256 which collapsed
            # every vertex onto slot 0 — that was wrong.
            ix = int(w_idx)
            nm = bone_name(ix)
            if nm is None and pal:
                nm = bone_name(0)
            if nm:
                weights[vi] = [(nm, 1.0)]
        if normal_comp:
            # D3DCOLOR (B,G,R,A in memory): xyz = byte2, byte1, byte0.  The old
            # in-order read scrambled the axes (geometry alignment 0.37 vs 0.97
            # for BGRA — verified on char01/police, 470k+ verts).
            normals.append((_U8N_LUT[vdata[k + 2]], _U8N_LUT[vdata[k + 1]],
                            _U8N_LUT[vdata[k]]))
            k += 4
        elif normal_full:
            normals.append(struct.unpack_from('<3f', vdata, k))
            k += 12
        if color:
            # D3DCOLOR B,G,R,A in memory -> present as R,G,B,A
            colors.append((vdata[k + 2] / 255.0, vdata[k + 1] / 255.0,
                           vdata[k] / 255.0, vdata[k + 3] / 255.0))
            k += 4
        if tangent_comp:
            tangents.append((_U8N_LUT[vdata[k + 2]], _U8N_LUT[vdata[k + 1]],
                             _U8N_LUT[vdata[k]]))
            tangents_w.append(_U8N_LUT[vdata[k + 3]])
            k += 4
        if binormal_comp:
            binormals.append((_U8N_LUT[vdata[k + 2]], _U8N_LUT[vdata[k + 1]],
                              _U8N_LUT[vdata[k]]))
            binormals_w.append(_U8N_LUT[vdata[k + 3]])
            k += 4
        if normal_mod:
            k += 4

    # Triangles: index slice, values local to the vertex slice.  WD1 winding
    # is D3D clockwise-front; Blender wants counter-clockwise, so reverse
    # (the stored vertex normals agree with the REVERSED winding — verified
    # geometrically on the char01 set).
    tris = []
    if mesh['prim_type'] == 0:
        s = dc['index_start']
        idx = indices[s:s + dc['index_count']]
        base = dc['min_index'] if (idx and max(idx) >= count) else 0
        for t in range(0, len(idx) - 2, 3):
            tris.append((idx[t] - base, idx[t + 2] - base, idx[t + 1] - base))

    name = (mesh['ranges'][0]['name'] if mesh['ranges'] and
            mesh['ranges'][0]['name'] else 'mesh%02d' % mi)
    mat = ''
    if mesh['mat_id'] < len(materials):
        mat = os.path.splitext(os.path.basename(
            materials[mesh['mat_id']].replace('\\', '/')))[0]
    return {
        'name': name, 'verts': verts, 'tris': tris,
        'uvs': uvs or None, 'uvs2': uvs2 or None,
        'normals': normals or None,
        'colors': colors or None,
        'tangents': tangents or None, 'tangents_w': tangents_w or None,
        'binormals': binormals or None, 'binormals_w': binormals_w or None,
        'weights': weights, 'material': mat,
    }


# ---------------------------------------------------------------------------
# Blender build
# ---------------------------------------------------------------------------

def build_wd_model(context, model, import_mesh_only=False):
    """Create armature + meshes from the neutral model dict.  Returns
    (armature_object_or_None, [created mesh objects]).

    `import_mesh_only` skips the armature and skin binding (geometry only)."""
    if bpy is None:
        raise RuntimeError("bpy unavailable")
    Mat = mathutils.Matrix
    Quat = mathutils.Quaternion
    Vec = mathutils.Vector

    mesh_objs = []
    bones = model['bones']
    arm_obj = None
    if bones and not import_mesh_only:
        ad = bpy.data.armatures.new(model['name'] + '_Armature')
        arm_obj = bpy.data.objects.new(ad.name, ad)
        context.collection.objects.link(arm_obj)
        context.view_layer.objects.active = arm_obj
        bpy.ops.object.mode_set(mode='EDIT')
        world = [None] * len(bones)
        ebs = []
        for i, b in enumerate(bones):
            local = (Mat.Translation(Vec(b['pos'])) @
                     Quat(b['quat']).to_matrix().to_4x4())
            p = b['parent']
            world[i] = (world[p] @ local
                        if 0 <= p < i and world[p] is not None else local)
            eb = ad.edit_bones.new(b['name'])
            head = world[i].to_translation()
            eb.head = head
            eb.tail = head + world[i].to_3x3() @ Vec((0.0, 0.05, 0.0))
            ebs.append(eb)
        for i, b in enumerate(bones):
            if 0 <= b['parent'] < i:
                ebs[i].parent = ebs[b['parent']]
        bpy.ops.object.mode_set(mode='OBJECT')

    for mesh in model['meshes']:
        me = bpy.data.meshes.new(mesh['name'])
        me.from_pydata(mesh['verts'], [], mesh['tris'])
        me.update()
        obj = bpy.data.objects.new(mesh['name'], me)
        context.collection.objects.link(obj)
        mesh_objs.append(obj)

        # stamp the WD1 injection layout so the mesh can be edited and
        # written back into the source .xbg (see inject_wd.py)
        inj = mesh.get('inject')
        if inj:
            obj['wd_src'] = inj['src']
            obj['wd_vb_off'] = inj['vb_off']          # offset within buffer 0
            obj['wd_stride'] = inj['stride']
            obj['wd_format'] = inj['format']
            obj['wd_scale'] = inj['scale']
            obj['wd_vcount'] = inj['vcount']
            obj['wd_mesh_index'] = inj['mesh_index']
            obj['wd_buf0_off'] = inj['buf0_off']      # file offset of buffer 0
            if inj.get('mip_src'):
                obj['wd_mip_src'] = inj['mip_src']    # bytes live in .xbgmip
            n_bufs = len(model['_layout']['buf_frames'])
            if n_bufs > 1:
                obj['wd_multibuffer'] = True          # in-place only; rebuild would corrupt split LOD buffers

        # UVs — bulk foreach_set instead of per-loop assignment.  loop_uvs are
        # already per-corner; per-vert UVs are gathered by loop vertex index.
        loop_uvs = mesh.get('loop_uvs')
        per_vert_uvs = mesh.get('uvs')
        loop_vi = None
        if loop_uvs or per_vert_uvs or mesh.get('uvs2'):
            loop_vi = np.empty(len(me.loops), dtype=np.intp)
            me.loops.foreach_get('vertex_index', loop_vi)

        def _set_uv(layer_name, loop_data, per_vert_data):
            uvl = me.uv_layers.new(name=layer_name)
            if loop_data:
                flat = np.asarray(loop_data, dtype=np.float64).ravel()
            else:
                flat = np.asarray(per_vert_data, dtype=np.float64)[loop_vi].ravel()
            uvl.data.foreach_set('uv', flat)

        if loop_uvs or per_vert_uvs:
            _set_uv('UVMap', loop_uvs, per_vert_uvs)
        per_vert_uvs2 = mesh.get('uvs2')
        if per_vert_uvs2:
            _set_uv('UVMap1', None, per_vert_uvs2)

        # Vertex colors (authored RGBA — often a shader mask, like Avatar).
        # FLOAT_COLOR, not BYTE_COLOR: BYTE_COLOR's Python `.color` API runs a
        # linear<->sRGB conversion through 8-bit storage, so the raw c/255
        # bytes written here came back shifted ±1 at inject time (116k drifted
        # bytes on char01's zero-edit round-trip). FLOAT_COLOR stores the
        # floats verbatim -> inject's round(c*255) recovers the exact byte.
        colors = mesh.get('colors')
        if colors:
            ca = me.color_attributes.new('Col', 'FLOAT_COLOR', 'POINT')
            ca.data.foreach_set('color', np.asarray(colors, dtype=np.float64).ravel())

        # Normals — keep the authored vectors AND mirror them into an
        # xbg_normal attribute (Avatar-importer parity, survives edits)
        loop_normals = mesh.get('loop_normals')
        per_vert_normals = mesh.get('normals')
        for poly in me.polygons:
            poly.use_smooth = True
        try:
            if loop_normals and len(loop_normals) == len(me.loops):
                me.normals_split_custom_set(loop_normals)
            elif per_vert_normals:
                me.normals_split_custom_set_from_vertices(per_vert_normals)
            # Blender <= 4.0: custom split normals only display with
            # auto-smooth enabled (removed in 4.1+, hence the guard).
            if hasattr(me, 'use_auto_smooth'):
                me.use_auto_smooth = True
        except Exception as exc:
            vlog.log("  [wd] custom normals failed on %s: %s"
                     % (mesh['name'], exc))
        if per_vert_normals:
            na = me.attributes.new('xbg_normal', 'FLOAT_VECTOR', 'POINT')
            na.data.foreach_set(
                'vector', [c for n in per_vert_normals for c in n])

        # Tangent / binormal frames (Avatar attribute names for round-trip)
        for vec_key, w_key, vec_attr, w_attr in (
                ('tangents', 'tangents_w', 'xbg_tangent', 'xbg_tangent_w'),
                ('binormals', 'binormals_w', 'xbg_binormal', 'xbg_binormal_w')):
            vecs = mesh.get(vec_key)
            if not vecs:
                continue
            va = me.attributes.new(vec_attr, 'FLOAT_VECTOR', 'POINT')
            va.data.foreach_set('vector', [c for v in vecs for c in v])
            ws = mesh.get(w_key)
            if ws:
                wa = me.attributes.new(w_attr, 'FLOAT', 'POINT')
                wa.data.foreach_set('value', ws)

        # Materials
        slots = mesh.get('material_slots')
        if slots:
            for sn in slots:
                mat = (bpy.data.materials.get(sn)
                       or bpy.data.materials.new(sn))
                me.materials.append(mat)
            fmats = mesh.get('face_materials') or []
            for pi, poly in enumerate(me.polygons):
                if pi < len(fmats):
                    poly.material_index = fmats[pi]
        elif mesh.get('material'):
            mat = (bpy.data.materials.get(mesh['material'])
                   or bpy.data.materials.new(mesh['material']))
            me.materials.append(mat)

        # Skin weights
        if mesh['weights'] and arm_obj:
            groups = {}
            for vi, wl in mesh['weights'].items():
                for nm, w in wl:
                    g = groups.get(nm)
                    if g is None:
                        g = groups[nm] = obj.vertex_groups.new(name=nm)
                    g.add([vi], w, 'REPLACE')
            obj.parent = arm_obj
            mod = obj.modifiers.new('Armature', 'ARMATURE')
            mod.object = arm_obj

    return arm_obj, mesh_objs


def load_wd_model(context, filepath, separate_primitives=True,
                  lod_select=0, import_mesh_only=False):
    """Dispatch by content: WD1 binary GEOM vs WD2 text glm.

    `separate_primitives` (Avatar-parity): True keeps one object per
    submesh (needed for injection); False joins them into a single object
    for a clean view-only import (flagged wd_joined — not injectable).
    `lod_select`: 0 = best LOD present, N = that LOD, None/<0 = all LODs.
    `import_mesh_only`: skip skeleton + skin binding (WD1 only).

    Returns (model, armature_object)."""
    head = open(filepath, 'rb').read(8)
    if head[:4] == b'MOEG':            # 'GEOM' u32 little-endian in file order
        model = parse_wd1_xbg(filepath, lod_select=lod_select)
    elif head[:7] == b'VERSION':
        raise ValueError(
            "this is a Watch Dogs 2 .glm — use the Watch Dogs 2 importer "
            "(XBG Import panel -> Watch Dogs 2 -> Import WD2 Model), not WD1")
    else:
        raise ValueError(
            "unrecognized file: %r — expected a WD1 GEOM .xbg "
            "(Avatar MESH .xbg files use the normal Import XBG)" % head[:4])
    arm, mesh_objs = (build_wd_model(context, model,
                                     import_mesh_only=import_mesh_only)
                      if bpy else (None, []))
    if arm is not None:
        arm['xbg_source_file'] = filepath   # for the WD1 MAB importer

    # Avatar-parity: when separate primitives is OFF, join the submeshes
    # into one object (clean import for viewing; re-inject not available).
    if bpy is not None and not separate_primitives and len(mesh_objs) > 1:
        bpy.ops.object.select_all(action='DESELECT')
        for o in mesh_objs:
            o.select_set(True)
        context.view_layer.objects.active = mesh_objs[0]
        # join() deletes the absorbed objects but leaves their mesh datablocks
        # orphaned in bpy.data — capture them so we can purge after.
        victim_meshes = [o.data for o in mesh_objs[1:]]
        bpy.ops.object.join()
        joined = context.active_object
        joined.name = model['name']
        joined['wd_joined'] = True          # merged — injection disabled
        for key in ('wd_vb_off', 'wd_stride', 'wd_format', 'wd_scale',
                    'wd_vcount', 'wd_mesh_index'):
            if key in joined.keys():
                del joined[key]
        for m in victim_meshes:
            if m.users == 0:
                bpy.data.meshes.remove(m)
    return model, arm
