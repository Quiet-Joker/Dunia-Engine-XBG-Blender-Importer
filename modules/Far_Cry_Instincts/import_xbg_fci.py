"""Far Cry Instincts (Xbox, 2005) .xbg parser.

Reverse-engineered 2026-07-02 from real, correctly-named files recovered via
the .fat/.nfo join (see the standalone FarCryInstincts/fci_extract.py + agents.md
for the full archive-format writeup). This is the MESH format only — a
completely different, unrelated binary layout from Avatar / Far Cry 2-6's
tag-chunk .xbg (EDON/MB2O/XOBB/...); Instincts predates that scheme.

176-byte fixed header:
    0x00  magic u32 = 0x01010900 (bytes 00 09 01 01)
    0x10  u32 = 176 (constant: offset of a 48-byte-stride material/texture
                      table, not used for geometry -- not parsed here)
    0x14  u32 = table entry count
    0x18..0x6C  a handful of (offset,count) u32 pairs at fixed byte offsets
                24,32,40,48,56,64,72,80,88,96,104 -- mostly (offset,1)
                material/texture-string descriptors (one embeds the mesh's
                texture path as a null-terminated string, e.g.
                "\\objects_xbox\\Pickups\\KeyCard\\KeyCard.xbt"); exactly one
                has count > 50, which is the (offset,byte_size) geometry
                (vertex+index) block.
    0x70  6x float32 = minX,maxX,minY,maxY,minZ,maxZ (bounding box --
          NOTE interleaved min/max per axis, not grouped as (min3,max3))

Geometry block: V vertices x a per-mesh stride, followed immediately by a
uint16 triangle-STRIP index buffer (degenerate/restart triangles where two
consecutive indices repeat). The first 6-8 bytes of each vertex are always
3x signed int16 x,y,z (+ a 4th int16 w=1, matching the same POS_INT16
convention documented for Avatar/Dunia in agents.md); position = int16 *
scale, uniform scale solved per-file.

Stride is NOT fixed -- simple single-material props use a compact 16-byte
vertex (just position + 10 unidentified bytes), while richer meshes use a
wider stride carrying the same D3DCOLOR-packed normal/tangent/binormal/color
+ UV attributes documented cross-game in agents.md (confirmed 2026-07-02 by
inspecting browning_ammo_box.xbg's raw vertex bytes at stride 24: pos+w(8) +
UV(4) + three D3DCOLOR-ish blocks with the characteristic 0x80 neutral-byte
padding (12) = 24; atv.xbg matches the full 40-byte cross-game vertex
exactly: pos+w(8) + UV0(4) + UV1(4) + bone(8) + normal(4) + color(4) +
tangent(4) + binormal(4) = 40). Neither stride nor V is stored directly
anywhere found so far, so both are solved simultaneously per file: try each
candidate stride in a fixed list, and for each, the correct V is the one
that (a) makes decoded positions uniformly fill the file's own bounding box,
(b) makes the bytes immediately following the vertex array a plausible,
fully-valid index buffer once any leading per-material header noise is
skipped, and (c) has the decoded triangles actually reference (utilize)
nearly all V vertices. The stride/V pair with the smallest leftover header
(cleanest match) wins.
"""
import os
import struct

MAGIC = 0x01010900


class FCIParseError(Exception):
    pass


def _find_geometry_block(data):
    best = None
    for off_pos in range(24, 112, 8):
        if off_pos + 8 > len(data) or off_pos + 8 > 176:
            break
        off, cnt = struct.unpack_from('<2I', data, off_pos)
        if cnt > 50 and off != 0xFFFFFFFF and off + cnt <= len(data):
            if best is None or cnt > best[1]:
                best = (off, cnt)
    return best


_CANDIDATE_STRIDES = (16, 20, 24, 28, 32, 36, 40, 44, 48)

def _parse_submesh_table(data, block_off, block_size, header_start=180):
    """SOLVED 2026-07-02 (with FC1 CGF ground truth reframing the search;
    generalized same day after 4 more test files broke the first version).
    EVERY FCI mesh -- not just multi-part vehicles -- stores its geometry as
    one or more CONCATENATED submeshes inside the geometry block, each =
    [vertcount x stride vertex array][idxcount x u16 triangle strip],
    described by a fixed-layout record living just before the block:
        +0   u32 vert_off    (byte offset of this submesh's verts, block-rel)
        +4   u32 strip_off   (byte offset of this submesh's strip, block-rel)
        +8   u16 idxcount     (number of u16 strip indices)
        +10  u16 stride       (per-vertex byte stride, e.g. 16/24/40)
        +12  u16 vertcount
        +32  u16 mat_id       (per-submesh material/group index -- SOLVED
                                2026-07-03; verified against uh60.xbg, whose
                                12 records give [1,10,0,3,4,5,0,6,7,8,9,junk]
                                -- index 0 correctly shared by exactly the 2
                                submeshes that should share a material, every
                                other index unique -- and against atv.xbg's
                                10 records, which cleanly reproduce the
                                tire/rim material split found independently
                                via the vertcount heuristic. The record
                                SPACING is 36 bytes, but this field is only
                                valid when read relative to a REAL record
                                (i.e. not the file's last record, where the
                                bytes past +14ish spill into whatever follows
                                the table instead of a next record -- both
                                uh60.xbg and atv.xbg's last record read back
                                a huge garbage value there; treated as
                                "unknown" and merged into the largest
                                legitimate group, since the last/biggest
                                record is consistently the main body anyway).
    (+14 to +31: more record fields, not decoded.)

    The record's location was originally found via a trailing marker field
    that turned out to be a PER-FILE value, not a fixed constant (0x53BF
    only for atv.xbg) -- searching for a fixed marker byte-pattern silently
    found 0 records on 4 different vehicle files (glider/ah6/uh60), each of
    which DOES have real multi-submesh geometry (a helicopter is obviously
    not 14 vertices). Fixed instead with a STRUCTURAL scan requiring no
    magic constant: try every 2-byte-aligned offset for the 5 fields above
    and keep it only if the arithmetic identity `vert_off + vertcount*stride
    == strip_off` holds (true iff the vertex array and its strip really are
    back-to-back, which is only true at a genuine record) -- verified this
    finds the exact same 10 records on atv.xbg as the old marker search, AND
    correctly finds records for grenade/keycard (1 record each, matching
    what _solve_vertex_layout finds independently) -- so this table format
    is likely UNIVERSAL, not vehicle-specific; `_solve_vertex_layout` is kept
    only as a fallback for the rare file where this scan finds nothing.
    Returns a list of (vert_off, vertcount, strip_off, idxcount, stride,
    mat_id) or [] if no valid record found; mat_id is None where unreadable
    (see above).

    Vertex layout is the same POS_INT16 convention as everywhere else:
    first 8 bytes = int16 x,y,z,w (w==1), then stride-8 attribute bytes
    (D3DCOLOR normal/tangent/binormal/color + UV -- not decoded yet)."""
    recs = []
    limit = block_off - 16
    for p in range(header_start, limit, 2):
        vert_off, strip_off = struct.unpack_from('<2I', data, p)
        idxcount, stride, vertcount = struct.unpack_from('<3H', data, p + 8)
        if stride not in _CANDIDATE_STRIDES:
            continue
        if vertcount < 3 or idxcount < 3:
            continue
        if vert_off + vertcount * stride != strip_off:
            continue
        if strip_off + idxcount * 2 > block_size:
            continue
        mat_id = None
        if p + 34 <= len(data):
            candidate = struct.unpack_from('<H', data, p + 32)[0]
            if candidate < 1000:  # garbage guard (see docstring)
                mat_id = candidate
        recs.append((vert_off, vertcount, strip_off, idxcount, stride, mat_id))
    return recs


def _shape_is_plausible(data, block_off, stride, v, scale, tris, bbox_min, bbox_max):
    """The decisive correctness signal found 2026-07-02 (round 4): compare
    the candidate's own decoded-vertex axis-length RATIOS (max_axis/min_axis
    -- the object's proportions) against the file's own header-declared
    bbox proportions. A genuinely wrong (stride,v) pair -- even one that
    passes bbox-fit + trimmed_valid==1.0 + utilization>=0.85 -- tends to
    scramble int16 data into a near-PERFECT symmetric cube (axis_ratio close
    to 1.0) regardless of the object's real shape, because uncorrelated
    "noise" fit to any target scale trends toward filling all 3 axes evenly.
    A real mesh's proportions consistently match the header's, even when
    the header bbox itself is imprecise (see atv.xbg's shared-assembly-bbox
    case) -- verified: grenade/keycard/browning_ammo_box (real, elongated,
    non-cubic objects) all match within ~5%; atv.xbg/atv_wheel.xbg (the
    confirmed-wrong fits from round 3) both collapse to axis_ratio~1.0
    despite a header ratio of ~2.0. An earlier attempt at this same goal
    (edge-length-vs-bbox-diagonal) wrongly rejected browning_ammo_box's
    correct candidate -- a genuinely elongated, low-poly object can have
    long edges relative to its own size; that isn't evidence of corruption
    the way "the whole object is secretly a cube" is."""
    pos = []
    p = block_off
    for _ in range(v):
        x, y, z = struct.unpack_from('<3h', data, p)
        pos.append((x * scale, y * scale, z * scale))
        p += stride
    used = sorted(set(i for t in tris for i in t))
    xs = [pos[i][0] for i in used]
    ys = [pos[i][1] for i in used]
    zs = [pos[i][2] for i in used]
    used_rng = (max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs))
    if min(used_rng) <= 1e-9:
        return False
    used_ratio = max(used_rng) / min(used_rng)
    hdr_rng = (bbox_max[0] - bbox_min[0], bbox_max[1] - bbox_min[1], bbox_max[2] - bbox_min[2])
    if min(hdr_rng) <= 1e-9:
        return True  # degenerate header bbox -- can't compare, don't block on it
    hdr_ratio = max(hdr_rng) / min(hdr_rng)
    # The header bbox itself can be inaccurate for vehicle sub-component
    # files (a shared, inflated assembly-wide bbox -- see atv.xbg), so a
    # loose ratio-of-ratios tolerance let a used_ratio~1.0 (near-perfect
    # cube) slip through against a header_ratio~1.9 (0.52 was inside a
    # 0.5-2.0 band). Real objects are essentially never an exact cube, so
    # add a direct absolute check: reject a near-cube used_ratio outright
    # unless the header itself also expects one.
    if used_ratio < 1.1 and hdr_ratio >= 1.1:
        return False
    return 0.8 <= (used_ratio / hdr_ratio) <= 1.25


def _solve_vertex_layout(data, block_off, block_size, bbox_min, bbox_max):
    """Search (stride, vertex_count) jointly. Returns
    (stride, vcount, scale, header_skip_words) for the cleanest match, or
    None. See the module docstring for why stride isn't fixed."""
    best = None  # (header_skip, stride, v, scale)
    for stride in _CANDIDATE_STRIDES:
        max_v = block_size // stride
        if max_v < 3:
            continue
        buf = data[block_off:block_off + max_v * stride]
        pad = stride - 6
        fmt = f'<3h{pad}x'
        minx = miny = minz = 32767
        maxx = maxy = maxz = -32768
        for v, (x, y, z) in enumerate(struct.iter_unpack(fmt, buf), start=1):
            if x < minx: minx = x
            if x > maxx: maxx = x
            if y < miny: miny = y
            if y > maxy: maxy = y
            if z < minz: minz = z
            if z > maxz: maxz = z
            if v < 3:
                continue
            rx, ry, rz = maxx - minx, maxy - miny, maxz - minz
            if rx == 0 or ry == 0 or rz == 0:
                continue
            sx = (bbox_max[0] - bbox_min[0]) / rx
            sy = (bbox_max[1] - bbox_min[1]) / ry
            sz = (bbox_max[2] - bbox_min[2]) / rz
            if sx <= 0 or sy <= 0 or sz <= 0:
                continue
            ratio = max(sx, sy, sz) / min(sx, sy, sz)
            scale = (sx + sy + sz) / 3
            # NOTE (2026-07-02): a uniform-scale, single-file bounding box is
            # NOT always accurate -- e.g. atv.xbg (a vehicle sub-component)
            # has its Y axis off ~2x, most likely because vehicle assemblies
            # share one bbox across multiple part files and this file's own
            # vertices only span part of it. `ratio`/`scale` are kept only
            # as loose sanity caps.
            if not (0.0000005 < scale < 0.05) or ratio > 8.0:
                continue
            # NOTE (2026-07-02, round 2): validity/utilization MUST be
            # measured on the header-skipped strip, not the raw post-vertex
            # bytes -- multi-material files prefix the real strip with a
            # variable-length garbage/header region (length scales with
            # material count, not a fixed size) that tanks a raw validity
            # score even when the real strip after it is clean.
            idx_start = block_off + v * stride
            idx_end = block_off + block_size
            n_idx = (idx_end - idx_start) // 2
            if n_idx < 3:
                continue
            raw_indices = list(struct.unpack_from(f'<{n_idx}H', data, idx_start))
            trimmed = _skip_index_header(raw_indices, v)
            if len(trimmed) < 9 or len(trimmed) < 0.3 * n_idx:
                continue  # header ate almost everything -- not a real strip
            # a genuinely correct strip references NO out-of-range vertex,
            # so require an exact match (not just "mostly valid") -- e.g.
            # 40mmhe_grenade.xbg has a v=36 candidate that's 0.99 valid (one
            # bad index short of the true v=37, a clean 1.00).
            trimmed_valid = sum(1 for x in trimmed if x < v) / len(trimmed)
            if trimmed_valid < 1.0:
                continue
            tris = _strip_to_triangles(trimmed, v)
            if not tris:
                continue
            used = set()
            for a, b, c in tris:
                used.add(a); used.add(b); used.add(c)
            utilization = len(used) / v
            if utilization < 0.85:
                continue
            # NOTE (2026-07-02, round 3->4): bbox-fit + trimmed_valid==1.0 +
            # utilization>=0.85 can STILL all pass on genuinely wrong data --
            # caught on atv.xbg (again, now at stride 40) and atv_wheel.xbg,
            # both of which decoded to a suspiciously-perfect symmetric CUBE
            # regardless of the file's own (non-cubic) header bbox. See
            # _shape_is_plausible's docstring for the full story, including
            # why an edge-length-based check (tried first) wrongly rejected
            # a genuinely correct browning_ammo_box.xbg candidate.
            if not _shape_is_plausible(data, block_off, stride, v, scale, tris, bbox_min, bbox_max):
                continue
            header_skip = len(raw_indices) - len(trimmed)
            if best is None or header_skip < best[0]:
                best = (header_skip, stride, v, scale)
            break  # smallest v at this stride that clears every gate
        if best is not None and best[0] == 0:
            break  # a perfectly clean match can't be beaten by a wider stride
    if best is None:
        return None
    header_skip, stride, v, scale = best
    return stride, v, scale, header_skip


def _skip_index_header(indices, vcount, min_run=30):
    """Multi-material meshes (e.g. vehicles) prefix the real triangle strip
    with a small per-material header table (repeats of ~6 records, values
    unrelated to vertex indices -- not yet reverse-engineered) that shows up
    as scattered/garbage values failing the < vcount check. Single-material
    meshes have no such header (the strip starts clean at index 0). Find the
    first sufficiently long run of valid (< vcount) values and treat that as
    the real strip start, discarding anything before it."""
    run_start = None
    for i, v in enumerate(indices):
        if v < vcount:
            if run_start is None:
                run_start = i
            if i - run_start + 1 >= min_run:
                return indices[run_start:]
        else:
            run_start = None
    return indices[run_start:] if run_start is not None else indices


def _strip_to_triangles(indices, vcount=None):
    """Convert a triangle strip to a list. Multi-material meshes (e.g.
    vehicles) interleave small per-section sub-headers into the index
    buffer between strips (not yet reverse-engineered), which show up as a
    handful of out-of-range values scattered through an otherwise-valid
    buffer -- skip just the (at most 3) triangles touching each one rather
    than truncating the whole mesh at the first bad value."""
    tris = []
    for i in range(len(indices) - 2):
        a, b, c = indices[i], indices[i + 1], indices[i + 2]
        if a == b or b == c or a == c:
            continue
        if vcount is not None and (a >= vcount or b >= vcount or c >= vcount):
            continue
        tris.append((a, b, c) if i % 2 == 0 else (a, c, b))
    return tris


_TEX_RE = _re_tex = __import__('re').compile(rb'[\x20-\x7e]{5,}?\.(?:xbt|dds)')


def _is_normal_map(path):
    """Normal-map suffix convention is NOT consistent across files: seen
    both "..._NT.xbt" (underscore, e.g. atv01-00_nt.xbt) and "...00.NT.xbt"
    (dot, e.g. atvwheelinside00-00.nt.xbt) -- the latter slipping past a
    plain "'_nt' not in path" check caused a normal map to get picked as a
    submesh's "diffuse" texture (atv.xbg's wheel-inside material)."""
    stem = os.path.splitext(path)[0].lower()
    return stem.endswith('_nt') or stem.endswith('.nt')


def _read_material_strings(data, scan_limit=None):
    """Scan the header region for embedded texture paths (.xbt/.dds). Works
    for both single-material props (one path in a header (offset,count) field)
    and multi-material meshes (several paths in the material table -- e.g.
    atv.xbg has ATV01-00_DF.xbt + _NT normal + wheel textures at ~offset 436).
    Diffuse maps (`_DF`, or anything that ISN'T a `_NT`/`_nt` normal map) are
    returned FIRST so the material builder picks a base-color texture, not a
    normal map."""
    limit = scan_limit if scan_limit is not None else min(len(data), 8192)
    seen = []
    for m in _TEX_RE.finditer(data, 0, limit):
        try:
            text = m.group(0).decode('latin1')
        except Exception:
            continue
        if '\\' not in text and '/' not in text:
            continue
        if text not in seen:
            seen.append(text)
    # diffuse first (deprioritize normal maps, either naming convention)
    diffuse = [p for p in seen if not _is_normal_map(p)]
    normals = [p for p in seen if _is_normal_map(p)]
    return diffuse + normals


# UV = two int16 immediately after pos+w (byte offset 8 of every vertex when
# stride > 8). The 4 bytes at offset 8 are the ONLY ones that don't end in
# the D3DCOLOR 0x80 pad byte, which is how UV was distinguished from the
# normal/tangent/binormal/color blocks that follow it.
#
# NOT a fixed global scale (an earlier attempt at raw/32768 -> [-1,1] put
# UVs outside the [0,1] box; user-verified fix 2026-07-02: manually scaling
# the imported UVs down to fit the 1x1 box "perfectly puts the UVs where
# they're supposed to be"). Each material/submesh has its OWN SEPARATE
# texture file (confirmed: atv.xbg's 3 submesh groups reference 3 entirely
# different .xbt files, not one shared atlas), so -- exactly like position
# uses a per-object scale fit to the header bbox -- UV is quantized to use
# the FULL int16 range per texture group, and must be MIN-MAX NORMALIZED
# back to [0,1] using that group's own raw extent (there's no separately-
# stored UV bbox/scale field found, unlike position). "Group" = the whole
# mesh for the single-block fallback path, or one group per submesh for the
# submesh-table path (each submesh already maps 1:1 to one texture).
def _normalize_uvs(raw_uvs):
    """raw_uvs: list of (u_raw, v_raw) int16 pairs -> list of (u,v) in
    [0,1] via min-max fit, V flipped for Blender's bottom-left origin."""
    if not raw_uvs:
        return []
    us = [p[0] for p in raw_uvs]
    vs = [p[1] for p in raw_uvs]
    umin, umax = min(us), max(us)
    vmin, vmax = min(vs), max(vs)
    uspan = (umax - umin) or 1
    vspan = (vmax - vmin) or 1
    return [((u - umin) / uspan, 1.0 - (v - vmin) / vspan) for u, v in raw_uvs]


class FCIModel:
    def __init__(self, filepath, bbox_min, bbox_max, vertices, triangles,
                 vertex_count, scale, texture_paths, stride=None, uvs=None,
                 face_textures=None):
        self.filepath = filepath
        self.bbox_min = bbox_min
        self.bbox_max = bbox_max
        self.vertices = vertices          # list of (x,y,z) world-scale floats
        self.triangles = triangles        # list of (a,b,c) vertex indices
        self.vertex_count = vertex_count
        self.scale = scale
        self.texture_paths = texture_paths  # embedded in-game paths, e.g. "\Foo\Foo.xbt"
        self.stride = stride              # solved per-vertex byte stride (16/24/40/...)
        self.uvs = uvs                    # per-vertex (u,v) or None if stride<=8
        # per-triangle in-game texture path, parallel to `triangles` (submesh-
        # table meshes only -- a multi-material mesh's UVs are authored
        # against DIFFERENT textures per submesh, e.g. a vehicle body vs its
        # tires, so lumping everything under texture_paths[0] shows the
        # wheels' UVs "scattered" against the wrong image). None for the
        # single-material fallback path (texture_paths[0] applies to everything).
        self.face_textures = face_textures


def parse_xbg(filepath):
    """Parse a Far Cry Instincts .xbg. Raises FCIParseError on failure."""
    with open(filepath, 'rb') as f:
        data = f.read()
    if len(data) < 176:
        raise FCIParseError("File too small to be an FCI .xbg")
    magic = struct.unpack_from('<I', data, 0)[0]
    if magic != MAGIC:
        raise FCIParseError(
            f"Not a Far Cry Instincts .xbg (magic 0x{magic:08x}, "
            f"expected 0x{MAGIC:08x})")

    minx, maxx, miny, maxy, minz, maxz = struct.unpack_from('<6f', data, 112)
    bbox_min, bbox_max = (minx, miny, minz), (maxx, maxy, maxz)

    geo = _find_geometry_block(data)
    if geo is None:
        raise FCIParseError("No geometry (vertex/index) block found in header")
    block_off, block_size = geo
    tex_paths = _read_material_strings(data, block_off)

    # PRIMARY path: multi-submesh table (vehicles + multi-part props). Each
    # submesh is a self-contained [verts][strip] with exact offsets/counts
    # from the 0x53BF descriptor table, so no heuristic solving is needed.
    recs = _parse_submesh_table(data, block_off, block_size)
    if recs:
        model = _decode_submeshes(data, block_off, recs, bbox_min, bbox_max)
        if model is not None:
            verts, tris, stride, scale, uvs, face_textures, bucket_bbox, model_bbox = model
            diffuse_only = [p for p in tex_paths if not _is_normal_map(p)]
            face_textures = _assign_submesh_textures(
                face_textures, diffuse_only, bucket_bbox, model_bbox)
            return FCIModel(filepath, bbox_min, bbox_max, verts, tris,
                            len(verts), scale, tex_paths, stride, uvs,
                            face_textures)

    # FALLBACK: single-material props with no submesh table -- solve the
    # (stride, vertex_count) split heuristically.
    fit = _solve_vertex_layout(data, block_off, block_size, bbox_min, bbox_max)
    if fit is None:
        raise FCIParseError(
            "Could not solve vertex layout for this file "
            "(unusual geometry -- possibly a skinned/animated mesh, "
            "not yet supported)")
    stride, vcount, scale, _header_skip = fit

    verts = []
    raw_uvs = [] if stride > 8 else None
    p = block_off
    for _ in range(vcount):
        x, y, z = struct.unpack_from('<3h', data, p)
        verts.append((x * scale, y * scale, z * scale))
        if raw_uvs is not None:
            raw_uvs.append(struct.unpack_from('<2h', data, p + 8))
        p += stride
    uvs = _normalize_uvs(raw_uvs) if raw_uvs is not None else None

    idx_start = block_off + vcount * stride
    idx_end = block_off + block_size
    n_idx = (idx_end - idx_start) // 2
    if n_idx < 3:
        raise FCIParseError("No index buffer found after the vertex array")
    indices = list(struct.unpack_from(f'<{n_idx}H', data, idx_start))
    indices = _skip_index_header(indices, vcount)
    tris = _strip_to_triangles(indices, vcount)
    if not tris:
        raise FCIParseError("Triangle strip decoded to zero triangles")

    return FCIModel(filepath, bbox_min, bbox_max, verts, tris, vcount, scale, tex_paths, stride, uvs)


def _decode_submeshes(data, block_off, recs, bbox_min, bbox_max):
    """Decode + concatenate all submeshes from a parsed submesh table into
    one combined mesh. Uniform position scale is derived from the whole
    object's int16 range vs the header bbox (there's no separately-stored
    scale field found yet; w==1 confirms the POS_INT16 convention). Returns
    (verts, triangles, stride, scale, uvs, face_buckets, bucket_bbox,
    model_bbox) or None, where face_buckets is a per-triangle bucket key:
    `('idx', table_position, mat_id_or_None)`. Bucketed by TABLE POSITION,
    not by mat_id -- mat_id can be reused across spatially-unrelated records
    (verified on uh60.xbg: mat_id=0 is shared by a tiny duplicate window
    panel AND an unrelated full-footprint floor decal; merging them by
    mat_id would blend their bounding boxes and break the spatial signal
    _assign_submesh_textures relies on). Multiple buckets legitimately
    ending up assigned the same final texture is fine and expected."""
    # collect int16 range across every submesh for a single global scale
    lo = [32767, 32767, 32767]
    hi = [-32768, -32768, -32768]
    for vert_off, vertcount, strip_off, idxcount, stride, _mat_id in recs:
        p = block_off + vert_off
        for _ in range(vertcount):
            x, y, z, _w = struct.unpack_from('<4h', data, p)
            if x < lo[0]: lo[0] = x
            if x > hi[0]: hi[0] = x
            if y < lo[1]: lo[1] = y
            if y > hi[1]: hi[1] = y
            if z < lo[2]: lo[2] = z
            if z > hi[2]: hi[2] = z
            p += stride
    rng = [hi[i] - lo[i] for i in range(3)]
    if min(rng) <= 0:
        return None
    hdr = [bbox_max[i] - bbox_min[i] for i in range(3)]
    scale = sum(hdr[i] / rng[i] for i in range(3)) / 3.0

    verts = []
    uvs = []
    tris = []
    face_buckets = []
    bucket_bbox = {}  # bucket -> [minx,miny,minz,maxx,maxy,maxz], world-scale
    used_stride = recs[0][4]
    for table_i, (vert_off, vertcount, strip_off, idxcount, stride, mat_id) in enumerate(recs):
        base = len(verts)
        raw_uv = []
        bucket = ('idx', table_i, mat_id)
        bb = bucket_bbox.setdefault(bucket, [1e18, 1e18, 1e18, -1e18, -1e18, -1e18])
        for i in range(vertcount):
            p = block_off + vert_off + i * stride
            x, y, z, _w = struct.unpack_from('<4h', data, p)
            wx, wy, wz = x * scale, y * scale, z * scale
            verts.append((wx, wy, wz))
            raw_uv.append(struct.unpack_from('<2h', data, p + 8) if stride > 8 else (0, 0))
            if wx < bb[0]: bb[0] = wx
            if wy < bb[1]: bb[1] = wy
            if wz < bb[2]: bb[2] = wz
            if wx > bb[3]: bb[3] = wx
            if wy > bb[4]: bb[4] = wy
            if wz > bb[5]: bb[5] = wz
        # each submesh has its own separate texture and its own quantized
        # UV range -- min-max normalize THIS submesh's raw values, not a
        # fixed global scale (see _normalize_uvs docstring).
        uvs.extend(_normalize_uvs(raw_uv))

        strip = struct.unpack_from(f'<{idxcount}H', data, block_off + strip_off)
        for k in range(len(strip) - 2):
            a, b, c = strip[k], strip[k + 1], strip[k + 2]
            if a == b or b == c or a == c:
                continue
            if a >= vertcount or b >= vertcount or c >= vertcount:
                continue
            if k % 2 == 0:
                tris.append((base + a, base + b, base + c))
            else:
                tris.append((base + a, base + c, base + b))
            face_buckets.append(bucket)
    if not tris:
        return None
    model_bbox = (
        min(b[0] for b in bucket_bbox.values()), min(b[1] for b in bucket_bbox.values()),
        min(b[2] for b in bucket_bbox.values()), max(b[3] for b in bucket_bbox.values()),
        max(b[4] for b in bucket_bbox.values()), max(b[5] for b in bucket_bbox.values()))
    return verts, tris, used_stride, scale, uvs, face_buckets, bucket_bbox, model_bbox


def _assign_submesh_textures(face_buckets, diffuse_candidates, bucket_bbox=None,
                              model_bbox=None):
    """Map each triangle's bucket (`('mat', mat_id)` from the submesh
    record's real material-index field, or `('unknown',)` for the record
    where that field wasn't readable -- see _decode_submeshes) to one of the
    embedded diffuse texture paths.

    mat_id is NOT a direct index into the diffuse-texture list (tried that,
    verified wrong by user testing: it broke the previously-correct
    body/exterior assignment). It also isn't strictly 1:1 with textures --
    e.g. uh60.xbg has 2 different mat_id values (a byte-identical DUPLICATE
    submesh, same panel declared twice) that both turn out to be glass, plus
    a 3rd, unrelated mat_id that's ALSO glass (multiple window panes
    legitimately sharing one texture is normal; a ground-shadow/floor decal
    being its own single mat_id is also normal). So texture assignment is
    still heuristic, but now spatially informed (verified against uh60.xbg
    by user-driven visual testing in Blender), on top of the general
    triangle-count/keyword scheme that already worked for 2-texture vehicles
    like atv.xbg:

    1. Single largest bucket overall -> first-listed diffuse texture (body).
    2. A bucket whose XY footprint covers most of the WHOLE model's XY
       footprint (a wide, thin decal, not the body -- e.g. a rotor-downwash/
       ground-shadow blob) -> a candidate whose filename contains
       "floor"/"shadow"/"decal", consumed once.
    3. A bucket sitting in the top quarter of the model's Z range (a
       greenhouse/windshield panel) -> a candidate whose filename contains
       "window"/"glass"/"windshield" -- NOT consumed, since multiple window
       panes legitimately share one glass texture.
    4. Falls back to the old "tire"/"wheel"/"rim" keyword pass, then cycles
       whatever's left in descending triangle-count order, same as before.

    Returns a per-triangle texture path list, or None if there's nothing
    useful to assign (0-1 diffuse candidates)."""
    if not face_buckets or len(diffuse_candidates) < 2:
        return None
    from collections import Counter
    counts = Counter(face_buckets)
    all_buckets = sorted(counts, key=lambda b: -counts[b])
    body_bucket = all_buckets[0]
    body_candidate = diffuse_candidates[0]
    assign = {body_bucket: body_candidate}
    remaining_candidates = [p for p in diffuse_candidates if p != body_candidate]

    still_unassigned = all_buckets[1:]

    if bucket_bbox and model_bbox:
        mminx, mminy, mminz, mmaxx, mmaxy, mmaxz = model_bbox
        mxspan = (mmaxx - mminx) or 1.0
        myspan = (mmaxy - mminy) or 1.0
        mzspan = (mmaxz - mminz) or 1.0
        floor_candidate = next(
            (p for p in remaining_candidates
             if any(k in p.lower() for k in ('floor', 'shadow', 'decal'))), None)
        window_candidate = next(
            (p for p in remaining_candidates
             if any(k in p.lower() for k in ('window', 'glass', 'windshield'))), None)
        window_used = False
        for b in still_unassigned:
            bb = bucket_bbox.get(b)
            if bb is None:
                continue
            xfrac = (bb[3] - bb[0]) / mxspan
            yfrac = (bb[4] - bb[1]) / myspan
            if floor_candidate is not None and xfrac > 0.6 and yfrac > 0.6:
                assign[b] = floor_candidate
                remaining_candidates = [p for p in remaining_candidates if p != floor_candidate]
                floor_candidate = None
                continue
            if window_candidate is not None and bb[2] >= mminz + 0.65 * mzspan:
                assign[b] = window_candidate
                window_used = True
                continue
        if window_used:
            # multi-use (several panes can share the glass texture) but must
            # not also leak into the generic leftover-cycling pool below
            remaining_candidates = [p for p in remaining_candidates if p != window_candidate]
        still_unassigned = [b for b in still_unassigned if b not in assign]

    tire_candidate = next((p for p in remaining_candidates if 'tire' in p.lower()), None)
    wheel_candidate = next(
        (p for p in remaining_candidates
         if p != tire_candidate and ('wheel' in p.lower() or 'rim' in p.lower())),
        None)
    leftover = [p for p in remaining_candidates if p not in (tire_candidate, wheel_candidate)]
    li = 0
    for b in still_unassigned:
        if tire_candidate is not None:
            assign[b] = tire_candidate
            tire_candidate = None
            continue
        if wheel_candidate is not None:
            assign[b] = wheel_candidate
            wheel_candidate = None
            continue
        if leftover:
            assign[b] = leftover[li % len(leftover)]
            li += 1
        else:
            assign[b] = body_candidate

    return [assign.get(b, body_candidate) for b in face_buckets]


def _walk_case_insensitive(root, parts):
    """Try to resolve `parts` (path segments) under `root`, matching each
    segment case-insensitively at its own directory level (handles a tree
    whose casing differs from the embedded path without needing a full
    recursive scan). Returns a real path or None."""
    cur = root
    for i, part in enumerate(parts):
        exact = os.path.join(cur, part)
        if i == len(parts) - 1:
            if os.path.isfile(exact):
                return exact
        elif os.path.isdir(exact):
            cur = exact
            continue
        if not os.path.isdir(cur):
            return None
        target = part.lower()
        try:
            entries = os.listdir(cur)
        except OSError:
            return None
        match = next((e for e in entries if e.lower() == target), None)
        if match is None:
            return None
        cur = os.path.join(cur, match)
    return cur if os.path.isfile(cur) else None


# Per-root cache of {lowercased basename: [full paths]}, built once per
# session per root (a full extracted dump can have 100k+ files -- walking it
# per-texture-lookup would be far too slow). Keyed by os.path.normcase(root).
_FILENAME_INDEX_CACHE = {}


def _filename_index(root):
    key = os.path.normcase(os.path.abspath(root))
    idx = _FILENAME_INDEX_CACHE.get(key)
    if idx is not None:
        return idx
    idx = {}
    for dirpath, _dirnames, filenames in os.walk(root):
        for fn in filenames:
            idx.setdefault(fn.lower(), []).append(os.path.join(dirpath, fn))
    _FILENAME_INDEX_CACHE[key] = idx
    return idx


def resolve_texture_path(data_root, in_game_path):
    """Turn an embedded in-game path like "\\objects_xbox\\Pickups\\KeyCard\\KeyCard.xbt"
    into a real filesystem path, searching `data_root` (typically the root
    of an fci_extract.py dump -- NOT necessarily the .xbg's own folder, since
    a model's textures routinely live in a completely different branch of
    the tree, e.g. a shared "_generic_objects" folder). Three tiers, cheapest
    first:
      1. Exact case-sensitive join.
      2. Case-insensitive walk matching each path segment at its own level
         (handles a differently-cased tree without a full scan).
      3. A cached whole-tree filename index (handles a genuinely different
         folder structure -- e.g. the archive's stored path doesn't match
         where fci_extract.py actually wrote the file -- as long as the
         basename exists somewhere under data_root; when multiple files
         share a basename, picks the one whose parent path shares the most
         trailing segments with the embedded path).
    Returns None if not found by any tier."""
    if not data_root or not in_game_path:
        return None
    parts = [p for p in in_game_path.replace('/', '\\').split('\\') if p]
    if not parts:
        return None

    candidate = os.path.join(data_root, *parts)
    if os.path.isfile(candidate):
        return candidate

    hit = _walk_case_insensitive(data_root, parts)
    if hit:
        return hit

    idx = _filename_index(data_root)
    matches = idx.get(parts[-1].lower())
    if not matches:
        return None
    if len(matches) == 1:
        return matches[0]
    wanted_dirs = [p.lower() for p in parts[:-1]]
    def score(path):
        got_dirs = [p.lower() for p in os.path.normpath(path).split(os.sep)[:-1]]
        s = 0
        for a, b in zip(reversed(wanted_dirs), reversed(got_dirs)):
            if a != b:
                break
            s += 1
        return s
    return max(matches, key=score)
