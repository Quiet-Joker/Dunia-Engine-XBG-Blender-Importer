"""FC5 / New Dawn XBG mesh importer (fork of the FC3 importer).

FC5 (version 0x000D0047) shares FC3/FC4 chunk framing and the flag-driven
vertex layout, but its SDOL LOD header differs (extra u32 before lod_dist;
variable entry/palette block before the vertex data). See _read_lod_fc5 and
agents.md "Far Cry 5 MESH".

ORIGINAL FC3 NOTES BELOW ------------------------------------------------
FC3 / FC4 XBG mesh importer.

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

SUPPORTED_VERSIONS = (VERSION_FC5,)  # FC5 / New Dawn (0x000D0047)


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


def _unpack_1010102(o, buf):
    """FC5 normal/tangent/binormal: R10G10B10A2_UNORM (a single u32 — three
    10-bit UNSIGNED components mapped to [-1,1], + a 2-bit handedness/sign).
    Confirmed empirically: decoded vertex normals align with the geometric
    face normals at 0.95 |dot| (limited only by the mesh's authored hard
    edges); the DLL carries the 'R10G10B10' format string.  NOTE: it is the
    UNSIGNED 10-bit reading that's correct — the signed reading and the
    R16G16_SNORM octahedral guess both failed against geometry.
    Returns (x, y, z, w2) with xyz a unit vector and w2 the 2-bit A field."""
    u = struct.unpack_from('<I', buf, o)[0]
    x = (u & 0x3ff) / 1023.0 * 2.0 - 1.0
    y = ((u >> 10) & 0x3ff) / 1023.0 * 2.0 - 1.0
    z = ((u >> 20) & 0x3ff) / 1023.0 * 2.0 - 1.0
    length = (x * x + y * y + z * z) ** 0.5 or 1.0
    return (x / length, y / length, z / length, (u >> 30) & 0x3)

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

    # Second bone-influence set (BONE_WTS2, 0x0020).  Skinned meshes that need
    # more than 4 influences per vertex (hands/arms with overlapping finger +
    # twist + skin-proxy bones) split the weights across two sets; set1+set2
    # sum to 255.  Reading only set1 leaves those verts under-weighted, so they
    # stay partly anchored and thin geometry (arm hair) stretches.
    skin_w2 = None
    skin_i2 = None
    if 0x0020 in offsets:
        o = vstart + offsets[0x0020]
        skin_w2 = struct.unpack_from('<4B', buf, o)
        skin_i2 = struct.unpack_from('<4B', buf, o + 4)

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
    # FC5 normal/tangent/binormal are R10G10B10A2_UNORM (not D3DCOLOR).
    nx = ny = nz = 0.0
    if 0x0040 in offsets:
        nx, ny, nz, _ = _unpack_1010102(vstart + offsets[0x0040], buf)

    # Color (D3DCOLOR B,G,R,A in memory -> present as R,G,B,A)
    color = None
    if 0x0080 in offsets:
        o = vstart + offsets[0x0080]
        _c = struct.unpack_from('<4B', buf, o)
        color = (_c[2], _c[1], _c[0], _c[3])

    # Tangent — R10G10B10A2.  Flag 0x0c7a stores it in UNK_400 (0x0400) rather
    # than the TANGENT slot (0x0100); accept either.  The 2-bit A = handedness.
    tan = None
    for _tflag in (0x0100, 0x0400):
        if _tflag in offsets:
            tx, ty, tz, tw = _unpack_1010102(vstart + offsets[_tflag], buf)
            tan = (tx, ty, tz, float(tw))
            break

    # Binormal — R10G10B10A2.
    bn = None
    if 0x0200 in offsets:
        bx, by, bz, bw = _unpack_1010102(vstart + offsets[0x0200], buf)
        bn = (bx, by, bz, float(bw))

    s.setpos(vstart + stride)
    return {'p': (vx, vy, vz), 'uv': (u, v), 'uv1': uv1, 'n': (nx, ny, nz),
            'color': color, 't': tan, 'b': bn, 'si': skin_i, 'sw': skin_w,
            'si2': skin_i2, 'sw2': skin_w2}


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
        # Raw payload locator so palettes can be extracted later, once the
        # EDON bone count is known (DNKS may be read before EDON).
        'sulc_pay': sulc_pay,
        'sulc_n_u16': sulc_dsize // 2,
    }


def _extract_sulc_palettes(data, skinning, n_bones):
    """Extract every per-cluster bone palette from the SULC payload.

    A palette is a sorted-or-unsorted run of global bone indices (each value
    < n_bones) terminated by 0xFFFF padding (the payload pads every palette
    slot with 0xFFFF).  We locate each padding boundary and walk backward over
    the < n_bones values to recover the run.  RE-validated on FC5 head / new
    dawn / swat_vest / body / top meshes: every skinned VB's max palette index
    matches a palette of length exactly (max_si + 1) (see agents.md).

    Returns a list of palettes (each a list[int]); `si` indexes palette[si]
    to get the global EDON bone index.  Per-VB selection (by length) happens
    in _assign_sulc_palettes.
    """
    pay = skinning.get('sulc_pay')
    n = skinning.get('sulc_n_u16') or 0
    if pay is None or n < 9 or n_bones <= 0:
        return []
    u16 = list(struct.unpack_from('<%dH' % n, data, pay))
    FF = 0xFFFF
    pals = []
    i = 8  # skip the 8-u16 global SULC header
    while i < n:
        # Palette end boundary: a < n_bones value immediately before FFFF pad.
        if u16[i] < n_bones and i + 1 < n and u16[i + 1] == FF:
            st = i
            while st - 1 >= 8 and u16[st - 1] < n_bones:
                st -= 1
            pals.append(u16[st:i + 1])
            # Advance past this palette's last element, then skip the FFFF pad.
            # (Without the +1 we'd sit on the boundary and re-match forever.)
            i += 1
            while i < n and u16[i] == FF:
                i += 1
            continue
        i += 1
    return pals


def _assign_sulc_palettes(data, skinning, bones, lods):
    """Attach the correct bone palette to each skinned VB, PER MATERIAL SECTION.

    Each vertex's `si` indices are local indices into ONE cluster palette, but a
    single VB usually packs SEVERAL material sections (body / arm-hair / etc.),
    each skinned to its OWN cluster with its OWN palette.  The matching palette
    has length exactly max(si)+1 over that section's verts; palettes that share
    a length are identical, so a by-length lookup is unambiguous.

    Builds `vb['vert_pal']` — a per-vertex list of palettes — so sections that
    share a VB but use different clusters are weighted correctly (otherwise the
    arm-hair section gets the body palette and detaches/stretches).  Also sets
    `vb['bone_palette']` to the largest palette for any legacy single-palette
    path.
    """
    pals = _extract_sulc_palettes(data, skinning, len(bones))
    skinning['palettes'] = pals
    if not pals:
        return
    by_len = {}
    for p in pals:
        by_len.setdefault(len(p), p)
    avail = sorted(by_len)

    def _pal_for(max_si):
        if max_si < 0:
            return None
        need = max_si + 1
        pal = by_len.get(need)
        if pal is None:
            bigger = [L for L in avail if L >= need]
            pal = by_len[bigger[0]] if bigger else None
        return list(pal) if pal else None

    for lod in lods:
        for vb in lod['vbs']:
            verts = vb['verts']
            if not verts:
                continue
            # Map each vertex to the material section it belongs to, then assign
            # that section's palette (by its own max_si).
            sections = vb.get('sections') or []
            faces = vb.get('faces') or []
            vert_pal = [None] * len(verts)
            if sections and faces:
                base = min(s[1] for s in sections)
                for (_mat, idx_s, idx_e) in sections:
                    fstart = (idx_s - base) // 3
                    fend = (idx_e - base) // 3
                    sec_verts = set()
                    for f in faces[fstart:fend]:
                        sec_verts.update(f)
                    smax = -1
                    for vi in sec_verts:
                        if vi >= len(verts):
                            continue
                        v = verts[vi]
                        sw = v.get('sw')
                        si = v.get('si')
                        if not si or not sw:
                            continue
                        for k in range(4):
                            if sw[k] and si[k] > smax:
                                smax = si[k]
                    pal = _pal_for(smax)
                    for vi in sec_verts:
                        if vi < len(verts):
                            vert_pal[vi] = pal
            # Fallback for any vertex not covered by a section: VB-wide max_si.
            vb_max = -1
            for v in verts:
                sw = v.get('sw')
                si = v.get('si')
                if not si or not sw:
                    continue
                for k in range(4):
                    if sw[k] and si[k] > vb_max:
                        vb_max = si[k]
            vb_pal = _pal_for(vb_max)
            for vi in range(len(verts)):
                if vert_pal[vi] is None:
                    vert_pal[vi] = vb_pal
            vb['vert_pal'] = vert_pal
            vb['bone_palette'] = vb_pal


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



def _read_lod_fc5(s, sdol_chunk_end):
    """Read one FC5 SDOL LOD.

    FC5 differs from FC3: there is an extra u32 before lod_dist, and a variable
    entry/palette block sits between the VB metadata and the vertex data.  The
    vertex-data start is located ROBUSTLY by anchoring on the trailing index
    buffer: the vertex block (sum of stride*vcount) is immediately followed by a
    u32 index_count + that many u16 indices, ending at the SDOL chunk end.
    Per-vertex decode reuses the flag-driven _read_vertex (FC5 flag 0x0c7a maps
    to the same component layout as FC3).
    """
    buf = s.d
    lod_dist = s.f32()              # (n_lods + total-VB count read by caller)
    vbc = s.u32()
    vbs = []
    for _ in range(vbc):
        vbs.append({'flag': s.u32(), 'stride': s.u32(),
                    'vcount': s.u32(), 'offset': s.u32(),
                    'verts': [], 'faces': [], 'sections': []})
    vb_size_total = sum(vb['stride'] * vb['vcount'] for vb in vbs)
    total_vcount = sum(vb['vcount'] for vb in vbs)
    after_vb = s.pos()

    # Material-section entry table (right after the VB records): u32 count, then
    # count × 5 u32 = [vb_index, f1, f2, idx_start, last_vert].  idx_start (f3)
    # is the section's offset into the LOD's (post-padding) index buffer; the
    # sections partition the index buffer contiguously, one per material.
    _sec_entries = []   # (vb_index=f0, idx_start=f3)
    try:
        n_entries = struct.unpack_from('<I', buf, after_vb)[0]
        if 0 < n_entries < 4096:
            eo = after_vb + 4
            for _ in range(n_entries):
                _e = struct.unpack_from('<5I', buf, eo)
                _sec_entries.append(_e)   # (vb_index, f1, f2, idx_start, last_vert)
                eo += 20
    except Exception:
        _sec_entries = []

    def _lead_pad(off, max_bytes):
        """Length (bytes) of an index-buffer alignment-padding run filled with a
        descending byte sequence K, K-1, …, 1 (K = first byte).  0 if none."""
        k = buf[off]
        if 2 <= k <= max_bytes and k % 2 == 0 and \
                all(buf[off + j] == k - j for j in range(k)):
            return k
        return 0

    # Anchor the vertex block: a variable entry/palette block sits between the VB
    # metadata and the vertex data.  The vertex block (vb_size_total bytes) is
    # immediately followed by u32 idx_count + idx_count u16 indices.  Validate the
    # REAL indices (after any leading alignment padding) against the vertex count.
    vstart = None
    total_idx = 0
    pad_bytes = 0
    for cand in range(after_vb, after_vb + 768):
        ioff = cand + vb_size_total
        if ioff + 4 > len(buf):
            break
        ic = struct.unpack_from('<I', buf, ioff)[0]
        end = ioff + 4 + ic * 2
        if not (0 < ic < 0x200000 and ic % 3 == 0 and end <= sdol_chunk_end):
            continue
        pad = _lead_pad(ioff + 4, min(ic * 2, 64))
        n = min(40, ic - pad // 2)
        if n <= 0:
            continue
        sample = struct.unpack_from('<%dH' % n, buf, ioff + 4 + pad)
        if all(x < total_vcount for x in sample):
            vstart, total_idx, pad_bytes = cand, ic, pad
            break
    if vstart is None:
        raise ValueError("FC5 SDOL: could not locate vertex block (vb_size=%d)"
                         % vb_size_total)

    # Decode vertices per VB (laid out sequentially).
    voff = vstart
    for vb in vbs:
        s.setpos(voff)
        for _ in range(vb['vcount']):
            v = _read_vertex(s, vb['stride'], vb['flag'])
            if v is not None:
                vb['verts'].append(v)
        voff += vb['vcount'] * vb['stride']

    # Index buffer immediately after the vertex block (skip leading padding).
    iuoff = vstart + vb_size_total
    idx_end = iuoff + 4 + total_idx * 2
    all_indices = struct.unpack_from('<%dH' % (total_idx - pad_bytes // 2), buf,
                                     iuoff + 4 + pad_bytes)

    # Material sections: the entry table's idx_start (f3) values partition the
    # (post-padding) index buffer contiguously, one run per material.  Build
    # (material_slot, idx_start, idx_end); material_slot = section order so each
    # section gets its own Blender material slot (named from LTMR by the pipeline).
    nreal = len(all_indices)
    # Per-VB global vertex offsets: a section's triangle indices are LOCAL to its
    # own VB (0-based), so VB1+ faces must be shifted by the preceding VBs' vcount
    # or they reference VB0's vertices (the corrupted-eyeball bug).
    vb_off = []
    acc = 0
    for vb in vbs:
        vb_off.append(acc)
        acc += vb['vcount']
    start_to_vb = {}
    start_to_mat = {}   # idx_start -> LTMR material index (entry field f2)
    for (f0, f1, f2, f3, f4) in _sec_entries:
        if 0 <= f3 < nreal:
            start_to_vb.setdefault(f3, f0)
            start_to_mat.setdefault(f3, f2)
    starts = sorted(start_to_vb)
    if not starts or starts[0] != 0:
        starts = [0] + starts
        start_to_vb.setdefault(0, 0)

    # Build per-section faces in idx order, applying each section's VB offset.
    sections = []
    for i, st in enumerate(starts):
        en = starts[i + 1] if i + 1 < len(starts) else nreal
        if en <= st:
            continue
        vbi = start_to_vb.get(st, 0)
        off = vb_off[vbi] if vbi < len(vb_off) else 0
        n_faces = (en - st) // 3
        for k in range(n_faces):
            t0 = st + k * 3
            a = all_indices[t0] + off
            b = all_indices[t0 + 1] + off
            c = all_indices[t0 + 2] + off
            if a >= total_vcount or b >= total_vcount or c >= total_vcount:
                a = b = c = 0
            vbs[0]['faces'].append((a, b, c))
        # Material slot = the entry's LTMR material index (f2), NOT the section's
        # positional order — otherwise pieces get each other's names and a 4th
        # section falls back to an out-of-range "matN".
        mat_i = start_to_mat.get(st, i)
        sections.append((mat_i, st, en))
    vbs[0]['sections'] = sections

    # Advance to the next LOD.  A small variable-size trailer (≈12 bytes) follows
    # the index buffer before the next LOD header, so 16-alignment isn't reliable
    # — scan forward for the next valid header (plausible lod_dist + vbc 1..4 +
    # a VB with a known stride).  No match ⇒ this was the last LOD.
    _VALID_STRIDES = (28, 32, 36, 40, 44, 48)
    nxt = None
    for off in range(idx_end, min(idx_end + 256, sdol_chunk_end - 24)):
        dist = struct.unpack_from('<f', buf, off)[0]
        vbc2 = struct.unpack_from('<I', buf, off + 4)[0]
        if not (1 <= vbc2 <= 4 and 0.01 < dist < 1e5):
            continue
        flag0 = struct.unpack_from('<I', buf, off + 8)[0]
        strd0 = struct.unpack_from('<I', buf, off + 12)[0]
        vcnt0 = struct.unpack_from('<I', buf, off + 16)[0]
        if strd0 in _VALID_STRIDES and 0 < vcnt0 < 500000 and flag0 < 0x10000:
            nxt = off
            break
    s.setpos(nxt if nxt is not None else sdol_chunk_end)
    return {'lod_distance': lod_dist, 'vbs': vbs, 'entries': list(_sec_entries),
            # Absolute file offsets, needed by the injector to patch bytes
            # back in place: vertex block start + index-data location.
            'vstart': vstart, 'idx_data_off': iuoff + 4 + pad_bytes,
            'idx_pad_bytes': pad_bytes, 'total_idx': total_idx}


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
            f"(supported: FC5=0x{VERSION_FC5:08x})"
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

    _KNOWN = (b'LTMR', b'LEKS', b'EDON', b'MB2O', b'DIKS', b'DNKS', b'SULC',
              b'ITOM', b'SDOL', b'XOBB', b'HPSB', b'PMCP', b'PMCU', b'FIKS',
              b'KSRP', b'DHRM', b'ZNRM', b'SDPD', b'AHSB')
    for _ in range(n_chunks):
        # FC5 re-sync: some chunks (e.g. LTMR) declare a size shorter than their
        # real content, so advancing by ck_size lands mid-chunk.  If we're not on
        # a known 4CC, scan forward to the next valid chunk magic.
        if data[s.pos():s.pos()+4] not in _KNOWN:
            nxt = None
            here = s.pos()
            for off in range(here, min(len(data) - 4, here + 8192)):
                if data[off:off+4] in _KNOWN:
                    nxt = off
                    break
            if nxt is None:
                break
            s.setpos(nxt)
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
            s.u32()                      # FC5: total VB count across all LODs
            for _ in range(n_lods):
                try:
                    lods.append(_read_lod_fc5(s, ck_start + ck_size))
                except Exception:
                    # Keep the LODs decoded so far — LOD0 is the high-detail one
                    # and is all the importer needs.  (Multi-LOD stepping is not
                    # fully nailed; see agents.md FC5 TODO.)
                    break
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

    # Now that both EDON (bones) and SDOL (LODs/VBs) are decoded, resolve each
    # skinned VB's bone palette from the SULC cluster data.
    if skinning and bones and lods:
        try:
            _assign_sulc_palettes(data, skinning, bones, lods)
        except Exception:
            pass

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
