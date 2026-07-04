"""FC4 XBG mesh importer (FC4-only fork of the FC3 importer).

Reverse-engineered from sample files (vaas.xbg, dennis.xbg, sabal.xbg)
and the Noesis community plugin (`fmt_farcry_xbg.py`).  The Noesis
plugin only handled single-VB LODs correctly — this implementation
also covers multi-VB LODs (where vb_size aggregates ALL VBs and the
LOD has ONE shared index block).

Format support:
    * Far Cry 3 (version 0x00030034)  — vaas.xbg, dennis.xbg
    * Far Cry 4 (version 0x00060037)  — sabal.xbg
    * Far Cry 5 (version 0x000D0047)  — NOT YET (different SDOL layout)

XBG file layout (FC3+):
    file header (32 B):     HSEM signature, version, hash, size, chunk count
    LTMR chunk:             external material references list
    LEKS chunk:             skeleton-present flag (4 B)
    EDON chunk:             node/bone hierarchy
    MB2O chunk:             inverse-bind-pose matrices (4x4 floats)
    DIKS chunk:             mesh-section index table
    DNKS chunk:             skin cluster data (contains SULC sub-chunk)
    [ITOM chunk (FC4+)]:    capability flag
    SDOL chunk:             ← all mesh geometry (vertex + index buffers)
    XOBB, HPSB:             bounding box, sphere
    [FIKS, KSRP, DHRM, ZNRM, SDPD (FC5+)]: capability flags
    PMCP, PMCU:             position / UV compression scale

SDOL chunk layout (FC3 / FC4):
    u32 num_lods
    per LOD:
        f32 lod_distance      (was misnamed "somefloat" in Noesis plugin)
        u32 vb_count
        per VB (16 B): u32 flag, u32 stride, u32 vcount, u32 offset
        u32 num_entries; num_entries × 7 × u32 (purpose unknown, skipped)
        u32 vb_size           (sum of vcount × stride across all VBs)
        align to 16
        per VB: vcount × stride bytes of vertex data
        u32 total_index_count
        align to 16
        total_index_count × i16  (one big buffer covering ALL VBs in this LOD)

Vertex format (stride 28..52 observed):
    i16×4   position (xyz, w)        → /16383.5
    i16×2   UV1                       → /16383.5  (+1 offset, then 2-v flip)
    [stride > 28] i32×2  UV2          → /16383.5
    u8×4    normal (xyz, w)           → /255 * 2 - 1
    [stride > 28] +8 B tangent/binormal (skipped)
    [stride > 32] +N B bone-weight / extra (skipped via stride)

⚠️ FC5 (version 0x000D0047) has a DIFFERENT SDOL layout that isn't
covered yet.  The first u32 is `n_lods` but then the structure diverges
— pillbottle's SDOL has additional header bytes before the VB metadata.
See XBG_FC3_FORMAT.md for the outstanding work.
"""

import struct


# Version markers (u32 at file offset 0x04)
VERSION_FC3 = 0x00030034
VERSION_FC4 = 0x00060037
VERSION_FC5 = 0x000D0047

SUPPORTED_VERSIONS = (VERSION_FC4,)  # FC4-only; FC3 has its own folder


def detect_version(data):
    """Return the version u32 from file offset 0x04, or None if not HSEM."""
    if len(data) < 32 or data[:4] != b'HSEM':
        return None
    return struct.unpack_from('<I', data, 4)[0]


class _Stream:
    """Minimal byte-stream reader."""
    __slots__ = ('d', 'p')

    def __init__(self, data):
        self.d = data
        self.p = 0

    def seek(self, n): self.p += n
    def setpos(self, p): self.p = p
    def pos(self): return self.p
    def u32(self):
        v = struct.unpack_from('<I', self.d, self.p)[0]; self.p += 4; return v
    def i32(self):
        v = struct.unpack_from('<i', self.d, self.p)[0]; self.p += 4; return v
    def i16(self):
        v = struct.unpack_from('<h', self.d, self.p)[0]; self.p += 2; return v
    def u8(self):
        v = self.d[self.p]; self.p += 1; return v
    def f32(self):
        v = struct.unpack_from('<f', self.d, self.p)[0]; self.p += 4; return v
    def strn(self, n):
        r = self.d[self.p:self.p + n].decode('latin-1', errors='replace')
        self.p += n
        return r
    def strz(self):
        """Read a null-terminated string."""
        a = self.p
        while self.p < len(self.d) and self.d[self.p] != 0:
            self.p += 1
        s = self.d[a:self.p].decode('latin-1', errors='replace')
        self.p += 1  # eat null
        return s
    def strex(self):
        """u32 length (ignored — advisory only), then null-terminated string."""
        self.u32()
        return self.strz()
    def align(self, n=16):
        rem = self.p % n
        if rem:
            self.p += n - rem


# D3DCOLOR unsigned-normalised byte -> signed float, precomputed once
# (identical value to b/255.0*2.0-1.0; avoids 290k+ per-call divisions).
_U2S_LUT = [i / 255.0 * 2.0 - 1.0 for i in range(256)]

# Per-vertex component offsets depend only on `flags`, not on the vertex —
# memoize so the OFF_ORDER walk runs once per unique flag set, not per vertex.
_OFFSET_CACHE = {}

# (flag, size) in serialized order — POS_* are mutually exclusive (first wins).
_OFF_ORDER = [
    (0x0001, 12), (0x0002, 8), (0x0004, 8),   # POS_FLOAT / POS_INT16 / POS_HALF
    (0x0008, 4), (0x0800, 4), (0x1000, 4),     # UV0 / UV1 / UV2
    (0x0010, 8), (0x0020, 8),                  # BONE_WTS1 / BONE_WTS2
    (0x0040, 4), (0x0080, 4),                  # NORMAL / COLOR
    (0x0100, 4), (0x0200, 4), (0x0400, 4),     # TANGENT / BINORMAL / UNK_400
]


def _vertex_offsets(flags):
    """Component byte-offsets within a vertex for the given flag set (memoized)."""
    cached = _OFFSET_CACHE.get(flags)
    if cached is not None:
        return cached
    off = 0
    offsets = {}
    pos_done = False
    for flag, size in _OFF_ORDER:
        is_pos = flag in (0x0001, 0x0002, 0x0004)
        if is_pos:
            if pos_done or not (flags & flag):
                continue
            pos_done = True
        elif not (flags & flag):
            continue
        offsets[flag] = off
        off += size
    _OFFSET_CACHE[flags] = offsets
    return offsets


def _read_vertex(s, stride, flags=0x0BDA):
    """Read one vertex using the same flag-based layout as the Avatar importer.

    Component order (identical to mesh.py VertexFlags.COMPONENT_ORDER):
      POS_INT16 (0x0002) →  8 B  i16×4  pos xyz w
      UV0       (0x0008) →  4 B  i16×2  UV channel 0
      UV1       (0x0800) →  4 B  i16×2  UV channel 1
      UV2       (0x1000) →  4 B  i16×2  UV channel 2
      BONE_WTS1 (0x0010) →  8 B  u8×4 weights + u8×4 indices
      BONE_WTS2 (0x0020) →  8 B  (second bone set, rare)
      NORMAL    (0x0040) →  4 B  int8×3 + pad
      COLOR     (0x0080) →  4 B  u8×4 RGBA
      TANGENT   (0x0100) →  4 B  int8×3 + u8 handedness
      BINORMAL  (0x0200) →  4 B  int8×3 + u8 handedness
      UNK_400   (0x0400) →  4 B  unknown

    For flags 0x0BDA (skinned, stride 40):
      pos(8) + UV0(4) + UV1(4) + bone(8) + normal(4) + color(4) + tangent(4) + binormal(4)
    """
    vstart = s.pos()
    buf = s.d
    if stride < 8:
        s.setpos(vstart + stride)
        return None

    # Component offsets depend only on flags (memoized — see _vertex_offsets).
    offsets = _vertex_offsets(flags)

    # Position
    vx = vy = vz = 0.0
    if 0x0002 in offsets:
        o = vstart + offsets[0x0002]
        vx = struct.unpack_from('<h', buf, o)[0] / 16383.5
        vy = struct.unpack_from('<h', buf, o+2)[0] / 16383.5
        vz = struct.unpack_from('<h', buf, o+4)[0] / 16383.5
    elif 0x0001 in offsets:
        o = vstart + offsets[0x0001]
        vx, vy, vz = struct.unpack_from('<3f', buf, o)

    # UV0 (primary UV)
    u = v = 0.0
    if 0x0008 in offsets:
        o = vstart + offsets[0x0008]
        ru, rv = struct.unpack_from('<2h', buf, o)
        u = ru / 16383.5 + 1.0
        v = 2.0 - rv / 16383.5

    # UV1
    uv1 = None
    if 0x0800 in offsets:
        o = vstart + offsets[0x0800]
        r1u, r1v = struct.unpack_from('<2h', buf, o)
        if r1u != -32768 or r1v != -32768:
            uv1 = (r1u / 16383.5 + 1.0, 2.0 - r1v / 16383.5)

    # Bone weights (first 4B) and palette indices (next 4B)
    skin_w = None
    skin_i = None
    if 0x0010 in offsets:
        o = vstart + offsets[0x0010]
        skin_w = struct.unpack_from('<4B', buf, o)
        skin_i = struct.unpack_from('<4B', buf, o + 4)

    # Normal / Tangent / Binormal / Color are D3DCOLOR: UNSIGNED-normalised
    # (n = byte/255*2 - 1; zero = 128, +1 = 255, -1 = 0) and stored in BGRA
    # byte order, so xyz = byte2, byte1, byte0.  The old signed `byte/127` +
    # in-order decode was wrong on BOTH counts — it only round-tripped stock
    # bytes; imported normals were non-unit and scrambled.  Verified against
    # geometry on sabal/firstperson/vaas: unsigned BGRA gives 0.94-0.96 normal
    # alignment vs ~0.37 for any other order.  Matches avatar/mesh.py.
    # Normal — kept as the outward surface normal (NEVER negated; same rule as
    # avatar/normals.py).  FC3/FC4's normals are intrinsic/outward; the apparent
    # "flip" is the game's opposite front-face winding, which import fixes by
    # REVERSING the triangle winding (in import_xbg.py), not by negating the
    # normal — negating renders black in Cycles / flat in EEVEE.
    nx = ny = nz = 0.0
    if 0x0040 in offsets:
        o = vstart + offsets[0x0040]
        b0, b1, b2 = struct.unpack_from('<3B', buf, o)
        nx, ny, nz = _U2S_LUT[b2], _U2S_LUT[b1], _U2S_LUT[b0]

    # Color (D3DCOLOR B,G,R,A in memory -> present as R,G,B,A)
    color = None
    if 0x0080 in offsets:
        o = vstart + offsets[0x0080]
        _c = struct.unpack_from('<4B', buf, o)
        color = (_c[2], _c[1], _c[0], _c[3])

    # Tangent
    tan = None
    if 0x0100 in offsets:
        o = vstart + offsets[0x0100]
        b0, b1, b2 = struct.unpack_from('<3B', buf, o)
        tw = buf[o + 3]
        tan = (_U2S_LUT[b2], _U2S_LUT[b1], _U2S_LUT[b0], tw)

    # Binormal
    bn = None
    if 0x0200 in offsets:
        o = vstart + offsets[0x0200]
        b0, b1, b2 = struct.unpack_from('<3B', buf, o)
        bw = buf[o + 3]
        bn = (_U2S_LUT[b2], _U2S_LUT[b1], _U2S_LUT[b0], bw)

    s.setpos(vstart + stride)
    return {'p': (vx, vy, vz), 'uv': (u, v), 'uv1': uv1, 'n': (nx, ny, nz),
            'color': color, 't': tan, 'b': bn, 'si': skin_i, 'sw': skin_w}


def _read_edon(s, chunk_size):
    """Parse the EDON (node/bone hierarchy) chunk.

    Returns a list of bone dicts:
        parent (i32)         -1 = root
        name (str)           bone name
        translation (3f)     bone-local translation relative to parent
        rotation (4f)        4 floats — interpretation TBD (often non-unit
                             magnitude, so not a plain quaternion).  Stored
                             verbatim for future decoding.
        scale (3f)           bone-local scale (usually 1,1,1)
        flag1 (u32)          unknown — observed to encode sibling index
        flt2 (float)         unknown — observed = 1.0 for every bone
        tag1 (4 bytes)       constant 'ff929000' across the entire corpus
        suffix (12 bytes)    post-name suffix (hash + child info?) — last
                             bone has NO suffix

    Per-bone byte layout (60 + name_len + 1 + 12 for all but the final bone):
        offset  type    field
        0       i32     parent       (-1 = root)
        4       3 × f32 quat_xyz     (quaternion x, y, z)
        16      1 × f32 quat_w       (quaternion w)
        20      3 × f32 translation  (x, y, z)
        32      3 × f32 scale        (x, y, z)
        44      u32     flag1
        48      f32     flt2
        52      4 bytes tag1
        56      u32     name_len
        60      name_len bytes + 1 null terminator
        ...     12 bytes suffix      (omitted on the LAST bone)
    """
    chunk_start = s.pos()
    count = s.u32()
    s.seek(12)  # 3 × u32 (zero, 1, -1) — purpose TBD
    bones = []
    end_offset = chunk_start + chunk_size
    for i in range(count):
        parent = struct.unpack_from('<i', s.d, s.p)[0]; s.p += 4
        qx, qy, qz = struct.unpack_from('<3f', s.d, s.p); s.p += 12
        qw = struct.unpack_from('<f', s.d, s.p)[0]; s.p += 4
        tx, ty, tz = struct.unpack_from('<3f', s.d, s.p); s.p += 12
        rot = (qw, qx, qy, qz)  # Blender (w,x,y,z) order
        trans = (tx, ty, tz)
        scale = struct.unpack_from('<3f', s.d, s.p); s.p += 12
        # flag1 is read as SIGNED because the corpus shows -1 (0xFFFFFFFF)
        # as a sentinel "no sibling/no value" — and Blender custom int
        # properties can't hold the unsigned 0xFFFFFFFF (would be > INT_MAX
        # and raise "Python int too large to convert to C int").
        flag1 = struct.unpack_from('<i', s.d, s.p)[0]; s.p += 4
        flt2 = s.f32()
        tag1 = s.d[s.p:s.p+4]; s.p += 4
        name_len = s.u32()
        if name_len > 128:
            # Bogus length — bail out of bone parse, keep what we have.
            break
        name = s.d[s.p:s.p+name_len].decode('latin-1', errors='replace')
        s.p += name_len
        if s.p < len(s.d) and s.d[s.p] == 0:
            s.p += 1   # eat null
        suffix = b''
        if i < count - 1:
            suffix = s.d[s.p:s.p+12]
            s.p += 12
        bones.append({
            'parent': parent, 'name': name,
            'translation': trans, 'rotation_raw': rot, 'scale': scale,
            'flag1': flag1, 'flt2': flt2, 'tag1': tag1, 'suffix': suffix,
        })
    return bones


def _read_dnks_sulc(data, dnks_off):
    """Parse the SULC sub-chunk inside DNKS — skinning data.

    Layout (per analysis of vaas.xbg, FC3):
        DNKS chunk:
            20 B chunk header
            SULC sub-chunk (12 510 B in vaas, contains the skinning data)
            data_size B trailing data (purpose unknown — appears unused)

        SULC payload:
            16 B header:
                u32 version            (always 1)
                u16 count_a            (varies — possibly material/cluster count)
                u16 count_b1
                u16 count_b2           (often = count_b1)
                u16 count_c            (= total verts at LOD 0?)
                u16 stride             (matches vertex buffer stride, 40 typical)
                u16 bones_used         (< total bone count, e.g. 154 of 166)
            sections (~70 of them in vaas):
                u16 = 3034              (= vb flag, section start marker)
                86 or 88 × u16          (per-vertex bone palette indices into
                                         a section-local palette; most values
                                         are < bone_count, with zeros = unused
                                         and special values like 154, 648
                                         appearing as markers within the data)

    Returns:
        dict {
            'sulc_present': bool,
            'header': dict with parsed u32+u16 values,
            'sections': list of dicts with raw u16 arrays,
        }
        OR None if no SULC found.

    ⚠️ This parser captures the STRUCTURE but does not yet decode
    per-vertex bone weights.  The section-local-index → global-bone
    mapping (probably stored in the SULC header's "count_a" entries or
    a parallel cluster table) needs to be reverse-engineered before
    Blender vertex groups can be wired up.
    """
    ck_size = struct.unpack_from('<I', data, dnks_off + 8)[0]
    dnks_end = dnks_off + ck_size
    # SULC sub-chunk starts immediately after DNKS header (20 B in).
    sulc_off = data.find(b'SULC', dnks_off, dnks_end)
    if sulc_off < 0:
        return None
    sulc_csize = struct.unpack_from('<I', data, sulc_off + 8)[0]
    sulc_dsize = struct.unpack_from('<I', data, sulc_off + 12)[0]
    sulc_pay = sulc_off + 20
    sulc_pay_end = sulc_pay + sulc_dsize

    # Read 16-byte header
    if sulc_dsize < 16:
        return {'sulc_present': True, 'header': None, 'sections': []}
    h_ver, h_a, h_b1, h_b2, h_c, h_stride, h_bones_used = struct.unpack_from(
        '<I 6H', data, sulc_pay)
    header = {
        'version': h_ver,
        'count_a': h_a,
        'count_b1': h_b1, 'count_b2': h_b2,
        'count_c': h_c,
        'stride': h_stride,
        'bones_used': h_bones_used,
    }

    # Walk sections — split by u16 = 3034 marker after the header.
    sections = []
    p = sulc_pay + 16
    section_starts = []
    while p < sulc_pay_end - 1:
        v = struct.unpack_from('<H', data, p)[0]
        if v == 3034:
            section_starts.append(p)
        p += 2

    for i, start in enumerate(section_starts):
        end = section_starts[i + 1] if i + 1 < len(section_starts) else sulc_pay_end
        # Skip the 3034 marker u16; read the rest as u16 array
        n_u16 = (end - start - 2) // 2
        if n_u16 <= 0:
            continue
        indices = struct.unpack_from(f'<{n_u16}H', data, start + 2)
        raw = list(indices)

        # Bone palette: first 48 u16s are the global bone indices for this
        # section's skin cluster.  VB stores local palette indices (0-based)
        # that index into this list.  Remaining 40 u16s are section metadata.
        PALETTE_SIZE = 48
        bone_assignments = raw[:PALETTE_SIZE]
        face_count = 0
        if len(raw) >= 5:
            face_count = raw[-5]

        sections.append({
            'offset': start,
            'indices': raw,
            'bone_assignments': bone_assignments,
            'face_count': face_count,
        })

    return {
        'sulc_present': True,
        'header': header,
        'sections': sections,
        'sulc_chunk_bytes': sulc_csize,
    }


def _read_lod(s):
    """Parse one LOD from the SDOL chunk.

    Multi-VB face indices are LOCAL (0-based per VB), not global.
    The per-entry metadata (n_entries × 7 × u32) records which slice of the
    shared index buffer belongs to each VB.  Each entry's layout:
        u32[0]  vb_index
        u32[1]  material_index  (into LTMR list)
        u32[2]  sub-entry index (multiple sections can share a material)
        u32[3]  idx_start       (start position in the flat u16 index buffer)
        u32[4]  last_vert       (last local vert index in this section, inclusive)
        u32[5]  vert_byte_off   (byte offset of this section's first vert in the VB)
        u32[6]  0               (reserved)

    We find the minimum idx_start per VB to determine the index-buffer
    boundary between VBs, then add each VB's vertex offset to convert
    local indices to global (Blender-space) indices.
    """
    lod_dist = s.f32()
    vbc = s.u32()
    vbs = []
    for _ in range(vbc):
        vbs.append({
            'flag':   s.u32(),
            'stride': s.u32(),
            'vcount': s.u32(),
            'offset': s.u32(),
            'verts':  [],
            'faces':  [],
            'sections': [],  # list of (mat_index, idx_start, idx_end) for this VB
        })
    n_entries = s.u32()
    entries = []
    for _ in range(n_entries):
        e = struct.unpack_from('<7I', s.d, s.p); s.p += 28
        entries.append(e)
    s.u32()  # vb_size = sum of vcount × stride
    s.align(16)
    # All vertex data laid out sequentially across VBs.
    for vb in vbs:
        for _ in range(vb['vcount']):
            v = _read_vertex(s, vb['stride'], vb['flag'])
            if v is not None:
                vb['verts'].append(v)
    # ONE shared index block per LOD.
    total_idx = s.u32()
    s.align(16)
    all_indices = struct.unpack_from(f'<{total_idx}H', s.d, s.p)
    s.p += total_idx * 2

    # Compute per-VB global vertex offsets (local_idx + offset = global_idx).
    vb_vert_offsets = []
    v_off = 0
    for vb in vbs:
        vb_vert_offsets.append(v_off)
        v_off += vb['vcount']

    if vbc > 1 and entries:
        # Multi-VB: find index-buffer boundary per VB via minimum idx_start.
        vb_idx_min = {}
        for e in entries:
            vb_i = e[0]
            idx_start = e[3]
            if vb_i not in vb_idx_min or idx_start < vb_idx_min[vb_i]:
                vb_idx_min[vb_i] = idx_start

        sorted_starts = sorted(vb_idx_min.items(), key=lambda x: x[1])
        vb_idx_ranges = {}
        for k, (vb_i, is_) in enumerate(sorted_starts):
            ie = sorted_starts[k + 1][1] if k + 1 < len(sorted_starts) else total_idx
            vb_idx_ranges[vb_i] = (is_, ie)

        for vb_i, vb in enumerate(vbs):
            if vb_i not in vb_idx_ranges:
                continue
            is_, ie = vb_idx_ranges[vb_i]
            offset = vb_vert_offsets[vb_i]
            n_faces = (ie - is_) // 3
            for fi in range(n_faces):
                base = is_ + fi * 3
                vb['faces'].append((
                    all_indices[base]     + offset,
                    all_indices[base + 1] + offset,
                    all_indices[base + 2] + offset,
                ))
    else:
        # Single-VB: all indices are global (offset = 0).
        total_faces = total_idx // 3
        for fi in range(total_faces):
            base = fi * 3
            vbs[0]['faces'].append((all_indices[base], all_indices[base+1], all_indices[base+2]))

    # Always extract per-material section ranges from entries (works for 1 or N VBs).
    if entries:
        for k, e in enumerate(entries):
            vb_i, mat_i = e[0], e[1]
            idx_start = e[3]
            idx_end = entries[k + 1][3] if k + 1 < len(entries) else total_idx
            vbs[vb_i]['sections'].append((mat_i, idx_start, idx_end))

    return {'lod_distance': lod_dist, 'vbs': vbs, 'entries': entries}


def parse_xbg(path_or_bytes):
    """Parse an FC3 / FC4 .xbg file.

    Returns dict:
        version       u32 from file header
        materials     list of (material_name, material_path) tuples
        lods          list of {'lod_distance': float, 'vbs': [...]}
                      where each VB has {'flag', 'stride', 'vcount',
                      'offset', 'verts', 'faces'}
    """
    if isinstance(path_or_bytes, (bytes, bytearray)):
        data = bytes(path_or_bytes)
    else:
        with open(path_or_bytes, 'rb') as f:
            data = f.read()

    version = detect_version(data)
    if version is None:
        raise ValueError("not an HSEM-signed file")
    if version not in SUPPORTED_VERSIONS:
        raise ValueError(
            f"unsupported XBG version 0x{version:08x} "
            f"(supported: FC3=0x{VERSION_FC3:08x}, FC4=0x{VERSION_FC4:08x})"
        )

    s = _Stream(data)
    s.seek(20)        # past HSEM, version, hash, two zeros
    s.u32()           # filesize - 12 (already validated by HSEM read)
    s.u32()           # zero reserved
    n_chunks = s.u32()

    materials = []
    lods = []
    bones = []
    skinning = None
    chunks_walked = []   # [(name, offset, chunk_size, data_size), …]
    pos_scale = 1.0      # PMCP position scale (default = no extra multiplier)

    for _ in range(n_chunks):
        ck_start = s.pos()
        if s.pos() + 20 > len(data):
            break
        name = s.strn(4)
        s.u32()              # chunk version (almost always 1)
        ck_size = s.u32()
        ck_dsize = s.u32()
        s.u32()              # reserved
        chunks_walked.append((name, ck_start, ck_size, ck_dsize))

        if name == 'LTMR':
            n_mats = s.u32()
            s.u32()                       # unknown
            for _ in range(n_mats):
                mp = s.strex()            # path  (e.g. "graphics\\_materials\\X.material.bin")
                mn = s.strex()            # name  (e.g. "Vaas_head2")
                materials.append((mn, mp))
            s.setpos(ck_start + ck_size)
        elif name == 'EDON':
            bones = _read_edon(s, ck_size - 20)
            s.setpos(ck_start + ck_size)
        elif name == 'DNKS':
            skinning = _read_dnks_sulc(data, ck_start)
            s.setpos(ck_start + ck_size)
        elif name == 'SDOL':
            n_lods = s.u32()
            for _ in range(n_lods):
                lods.append(_read_lod(s))
            s.setpos(ck_start + ck_size)
        elif name == 'PMCP':
            # FC3/FC4 PMCP payload is exactly 8 bytes: f32 unknown + f32 scale.
            # (Avatar/FC2 has a larger PMCP with 2 leading u32s before the
            # floats — FC3 does not; reading those extra bytes spills into the
            # next chunk and produces a near-zero multiplier that collapses all
            # vertex positions to the origin.)
            # The scale is typically 1/16384 ≈ 6.1e-5, which is already baked
            # into the /16383.5 divisor in _read_vertex.  We only apply a
            # post-multiply when the file carries a genuinely different value.
            try:
                _pmcp_unk = s.f32()
                raw_scale = s.f32()
                baseline = 1.0 / 16384.0
                if abs(raw_scale - baseline) > 1e-6 and raw_scale > 0.0:
                    pos_scale = raw_scale * 16384.0   # net extra multiplier
            except Exception:
                pass
            s.setpos(ck_start + ck_size)
        else:
            s.setpos(ck_start + ck_size)

    # Apply PMCP scale if it differed from the hardcoded baseline.
    if pos_scale != 1.0:
        for lod in lods:
            for vb in lod['vbs']:
                scaled = []
                for v in vb['verts']:
                    px, py, pz = v['p']
                    scaled.append({**v, 'p': (px * pos_scale, py * pos_scale, pz * pos_scale)})
                vb['verts'] = scaled

    return {
        'version': version,
        'materials': materials,
        'bones': bones,
        'skinning': skinning,
        'lods': lods,
        'pos_scale': pos_scale,
        'chunks': chunks_walked,
        'file_size': len(data),
    }
