"""Low-level XBG chunk navigation and patching helpers.

XBG layout
----------
The file header starts with a 4-byte magic + six 32-bit fields; the
chunk count lives at offset 28 (`<i` on PC, `>i` on PS3).  Each chunk
is a 12-byte header (`magic[4]`, `chunk_int[4]`, `chunk_size[4]`)
followed by `chunk_size` bytes of payload.  `chunk_size` is the TOTAL
chunk size including its own header.

Endianness
----------
PC files are little-endian; PS3 files are big-endian.  Multi-byte
ints/floats follow the file's byte order, AND the 4-byte chunk magic
strings are stored byte-reversed between the two platforms (so the
codepath that searches for an `'SDOL'` chunk in a PS3 file has to look
for `'LODS'` instead).  Every public function in this module takes an
`endian` parameter (`'<'` or `'>'`) — pass the value detected at import
time (stored on `obj["xbg_data"]["endian"]`).

API
---
All functions operate on `file_data: bytes / bytearray` — they never
read from disk.  The injector reads the whole file into a `bytearray`
once, mutates it via these helpers, then writes the result.

  * `find_chunk` / `find_all_chunks` — locate chunks by 4-byte name.
  * `patch_pmcp`                     — rewrite the PMCP pos_scale.
  * `patch_dnks`                     — rewrite face/vert counts in DNKS.
  * `patch_bounds`                   — rewrite XOBB / HPSB from a set of
                                       Blender mesh objects.
  * `parse_dnks_for_palette`         — extract one submesh's 48-entry
                                       bone palette without parsing the
                                       full DNKS chunk.
"""

import struct
import math
import mathutils

from .binary_avatar import encode_chunk_magic, LE, BE
from ..Core.debug import VerboseLogger


# ============================================================
# Chunk lookup
# ============================================================

def find_chunk(file_data, name, endian=LE):
    """Return `(chunk_start, data_start, chunk_size)` for the FIRST chunk
    whose magic matches `name`, or `None` if not found.

    `name` is the canonical (LE-byte-order) string used throughout the
    codebase ('SDOL', 'DNKS', 'PMCP', …).  On a BE file the function
    automatically searches for the reversed bytes.

    `chunk_start` is the byte offset of the magic itself; `data_start`
    is `chunk_start + 12` (i.e. the first byte after the chunk header).
    """
    magic = encode_chunk_magic(name, endian)
    if len(file_data) < 32:
        return None
    cc = struct.unpack_from(f'{endian}i', file_data, 28)[0]
    off = 32
    for _ in range(min(cc, 256)):
        if off + 12 > len(file_data):
            break
        cs = struct.unpack_from(f'{endian}i', file_data, off + 8)[0]
        if file_data[off:off + 4] == magic:
            return off, off + 12, cs
        if cs <= 0:
            break
        off += cs
    return None


def find_all_chunks(file_data, name, endian=LE):
    """Return a list of `(chunk_start, data_start, chunk_size)` tuples
    for EVERY chunk whose magic matches `name` (canonical LE form)."""
    magic = encode_chunk_magic(name, endian)
    results = []
    if len(file_data) < 32:
        return results
    cc = struct.unpack_from(f'{endian}i', file_data, 28)[0]
    off = 32
    for _ in range(min(cc, 256)):
        if off + 12 > len(file_data):
            break
        cs = struct.unpack_from(f'{endian}i', file_data, off + 8)[0]
        if file_data[off:off + 4] == magic:
            results.append((off, off + 12, cs))
        if cs <= 0:
            break
        off += cs
    return results


# ============================================================
# Chunk patching
# ============================================================

def patch_pmcp(file_data, new_scale, endian=LE):
    """Overwrite the `pos_scale` float in the PMCP chunk.

    Mutates `file_data` in place.  Logs and returns silently if no PMCP
    chunk is found.  PMCP layout: header(12) + 2x int32 + unk_float(4) +
    pos_scale(4) — so the float we want is at `chunk_start + 24`.
    """
    try:
        offset = 4
        header_ints = struct.unpack_from(f'{endian}7i', file_data, offset)
        chunk_count = header_ints[6]
        offset += 28
        pmcp_bytes = encode_chunk_magic('PMCP', endian)
        for _ in range(chunk_count):
            chunk_raw  = file_data[offset:offset + 4]
            chunk_size = struct.unpack_from(f'{endian}i', file_data, offset + 8)[0]
            if chunk_raw == pmcp_bytes:
                struct.pack_into(f'{endian}f', file_data, offset + 24, new_scale)
                VerboseLogger.log(f"  [inject] PMCP scale updated -> {new_scale:.6f}")
                return
            if chunk_size <= 0:
                break
            offset += chunk_size
        VerboseLogger.log("  [inject] WARNING: PMCP chunk not found -- scale not updated")
    except Exception as exc:
        VerboseLogger.log(f"  [inject] WARNING: Could not update PMCP scale: {exc}")


def parse_dnks_for_palette(file_data, target_lod, submesh_idx=0, endian=LE):
    """Extract the 48-entry bone palette for one DNKS submesh.

    Returns a list of 48 int16 global bone IDs, or `None` if the chunk
    / LOD / submesh is not present.  `-1` entries mark unused slots.

    DNKS data layout (mirrors `parse_dnks_chunk` in mesh.py):
      preamble = 2x int32 + 4-byte tag + 4x int32 = 28 bytes
      per-LOD  = int32(mat_count) + mat_count x (14-byte header + 96-byte
                 palette = 110 bytes)
    """
    info = find_chunk(file_data, 'DNKS', endian)
    if not info:
        return None
    _, ds, _ = info
    p = ds + 28
    for lod_idx in range(target_lod + 1):
        if p + 4 > len(file_data):
            return None
        mat_count = struct.unpack_from(f'{endian}i', file_data, p)[0]
        p += 4
        for sm_idx in range(mat_count):
            if p + 110 > len(file_data):
                return None
            if lod_idx == target_lod and sm_idx == submesh_idx:
                return list(struct.unpack_from(f'{endian}48h', file_data, p + 14))
            p += 110
    return None


def patch_dnks(file_data, target_lod, submesh_updates, endian=LE):
    """Update face/vert counts for each submesh in `target_lod`.

    DNKS submesh header layout (7 × uint16, 14 bytes):
      +0  h0  mat_id             (never touched)
      +2  h1  face_count         ← written
      +4  h2  face_count (dup.)  ← written (same value as h1)
      +6  h3  index_count        ← written (= face_count × 3)
      +8  h4  vertex stride      (never touched — fixed per file)
      +10 h5  vert_count         ← written
      +12 h6  format constant    (never touched — fixed per file)
    Then 96 bytes of bone palette (48 × int16).
    Total per-submesh block = 110 bytes.

    `submesh_updates` is a list of `(face_count, vert_count) | None`
    tuples indexed by submesh slot.  A `None` entry means "leave this
    slot's DNKS counts unchanged" — used by slot-aware injection when
    only a subset of primitives is being replaced.  Submeshes BEYOND
    the original DNKS submesh count are silently skipped.  All counts
    are clamped to uint16 max (65535) before writing.
    """
    info = find_chunk(file_data, 'DNKS', endian)
    if not info:
        VerboseLogger.log("  [inject] DNKS not found -- face/vert counts NOT updated")
        return file_data

    _, ds, _ = info
    p = ds + 28
    dnks_sm_count = 0

    for lod_idx in range(target_lod + 1):
        if p + 4 > len(file_data):
            VerboseLogger.log(f"  [inject] DNKS: ran out of data at LOD {lod_idx}")
            return file_data
        mat_count = struct.unpack_from(f'{endian}i', file_data, p)[0]
        p += 4

        if lod_idx == target_lod:
            dnks_sm_count = mat_count

        for sm_idx in range(mat_count):
            header_base = p
            if lod_idx == target_lod and sm_idx < len(submesh_updates):
                update = submesh_updates[sm_idx]
                if update is not None:
                    fc, vc = update
                    ic = fc * 3          # index count = triangle count × 3
                    struct.pack_into(f'{endian}H', file_data, header_base +  2, min(fc, 65535))
                    struct.pack_into(f'{endian}H', file_data, header_base +  4, min(fc, 65535))
                    struct.pack_into(f'{endian}H', file_data, header_base +  6, min(ic, 65535))
                    struct.pack_into(f'{endian}H', file_data, header_base + 10, min(vc, 65535))
                    VerboseLogger.log(f"  [inject] DNKS LOD{lod_idx} SM{sm_idx}: "
                                     f"face_count={fc}  index_count={ic}  vert_count={vc}")
                else:
                    VerboseLogger.log(f"  [inject] DNKS LOD{lod_idx} SM{sm_idx}: preserved (not replaced)")
            p += 110

    if len(submesh_updates) > dnks_sm_count:
        extra = len(submesh_updates) - dnks_sm_count
        VerboseLogger.log(f"  [inject] NOTE: {extra} new submesh(es) beyond DNKS capacity.")
    return file_data


def patch_bounds(file_data, objects, pos_scale, import_mesh_only, endian=LE):
    """Recompute XOBB (axis-aligned box) and HPSB (bounding sphere) from
    one or more Blender mesh objects, and write the result into the
    first XOBB and HPSB chunks in the file.

    `pos_scale` is currently unused but kept for API symmetry with the
    other patch_* functions.  The Z rotation that the importer applies
    to display the model is inverted here so the stored bounds match
    the game-space coordinate frame, not Blender's.
    """
    if not objects:
        return file_data

    rz = objects[0].rotation_euler.z
    if import_mesh_only and abs(rz - math.radians(180)) < 0.01:
        rot_inv = mathutils.Matrix.Rotation(-math.radians(180), 4, 'Z')
    elif abs(rz) > 0.01:
        rot_inv = mathutils.Matrix.Rotation(-rz, 4, 'Z')
    else:
        rot_inv = mathutils.Matrix.Identity(4)

    min_x = min_y = min_z =  1e30
    max_x = max_y = max_z = -1e30

    for obj in objects:
        if not obj.data or not obj.data.vertices:
            continue
        for v in obj.data.vertices:
            rc = rot_inv @ mathutils.Vector((v.co.x, v.co.y, v.co.z, 1.0))
            min_x = min(min_x, rc.x); max_x = max(max_x, rc.x)
            min_y = min(min_y, rc.y); max_y = max(max_y, rc.y)
            min_z = min(min_z, rc.z); max_z = max(max_z, rc.z)

    if min_x > 1e29:
        VerboseLogger.log("  [inject] No vertices found -- skipping bounds patch")
        return file_data

    # XOBB — min/max box, stored at +20 from chunk_start (6 floats).
    for xobb_start, _, _ in find_all_chunks(file_data, 'XOBB', endian):
        try:
            struct.pack_into(f'{endian}ffffff', file_data, xobb_start + 20,
                             min_x, min_y, min_z, max_x, max_y, max_z)
            VerboseLogger.log(f"  [inject] XOBB updated")
        except Exception as exc:
            VerboseLogger.log(f"  [inject] XOBB patch failed: {exc}")
        break

    # HPSB — center + radius, stored at +20 from chunk_start (4 floats).
    for hpsb_start, _, _ in find_all_chunks(file_data, 'HPSB', endian):
        cx = (min_x + max_x) * 0.5
        cy = (min_y + max_y) * 0.5
        cz = (min_z + max_z) * 0.5
        radius = 0.0
        for obj in objects:
            if not obj.data or not obj.data.vertices:
                continue
            for v in obj.data.vertices:
                rc = rot_inv @ mathutils.Vector((v.co.x, v.co.y, v.co.z, 1.0))
                d  = mathutils.Vector((rc.x - cx, rc.y - cy, rc.z - cz)).length
                radius = max(radius, d)
        try:
            struct.pack_into(f'{endian}ffff', file_data, hpsb_start + 20,
                             cx, cy, cz, radius)
            VerboseLogger.log(f"  [inject] HPSB updated: radius={radius:.3f}")
        except Exception as exc:
            VerboseLogger.log(f"  [inject] HPSB patch failed: {exc}")
        break

    return file_data


# ============================================================
# LTMR (material table) rebuild + DNKS material-id remap
# ============================================================

def parse_ltmr_names(file_data, endian=LE):
    """Return (chunk_start, chunk_size, chunk_int, f_b, f_d, [names]).

    LTMR layout (verified on real XBGs):
      12-byte chunk header: magic, chunk_int, chunk_size(total)
      payload: u32 a(=chunk_size-20), u32 b, u32 matCount, u32 d
      then matCount * (u32 nameLen, nameLen bytes, 1 NUL byte)
    """
    info = find_chunk(file_data, 'LTMR', endian)
    if not info:
        return None
    cs_start, ds, chunk_size = info
    chunk_int = struct.unpack_from(f'{endian}i', file_data, cs_start + 4)[0]
    a, f_b, mc, f_d = struct.unpack_from(f'{endian}4I', file_data, ds)
    p = ds + 16
    names = []
    for _ in range(mc):
        nl = struct.unpack_from(f'{endian}I', file_data, p)[0]
        p += 4
        names.append(file_data[p:p + nl].decode('latin-1'))
        p += nl + 1                     # name + 1 NUL terminator
    return (cs_start, chunk_size, chunk_int, f_b, f_d, names)


def build_ltmr_chunk(chunk_int, f_b, f_d, names, endian=LE):
    """Serialise an LTMR chunk (12-byte header + payload) for *names*."""
    body = bytearray()
    for nm in names:
        nb = nm.encode('latin-1')
        body += struct.pack(f'{endian}I', len(nb)) + nb + b'\x00'
    payload = bytearray(struct.pack(f'{endian}4I',
                                    0, f_b, len(names), f_d) + body)
    chunk_size = 12 + len(payload)
    struct.pack_into(f'{endian}I', payload, 0, chunk_size - 20)  # field 'a'
    magic = encode_chunk_magic('LTMR', endian)
    return magic + struct.pack(f'{endian}ii', chunk_int, chunk_size) + payload


def patch_dnks_matids(file_data, target_lod, matid_updates, endian=LE):
    """Set DNKS submesh +0 u16 mat_id for `target_lod`.

    `matid_updates`: list indexed by submesh slot; int = new mat_id,
    None = leave unchanged. Mirrors patch_dnks navigation exactly."""
    info = find_chunk(file_data, 'DNKS', endian)
    if not info:
        VerboseLogger.log("  [inject] DNKS not found -- mat_ids NOT updated")
        return file_data
    _, ds, _ = info
    p = ds + 28
    for lod_idx in range(target_lod + 1):
        if p + 4 > len(file_data):
            return file_data
        mat_count = struct.unpack_from(f'{endian}i', file_data, p)[0]
        p += 4
        for sm_idx in range(mat_count):
            if (lod_idx == target_lod and sm_idx < len(matid_updates)
                    and matid_updates[sm_idx] is not None):
                struct.pack_into(f'{endian}H', file_data, p,
                                 matid_updates[sm_idx] & 0xFFFF)
                VerboseLogger.log(f"  [inject] DNKS LOD{lod_idx} SM{sm_idx}: "
                                  f"mat_id -> {matid_updates[sm_idx]}")
            p += 110
    return file_data


# ============================================================
# DNKS full rebuild — change a LOD's submesh COUNT (split-by-material)
# ============================================================

def parse_dnks_full(file_data, endian=LE):
    """Split DNKS into (cs_start, chunk_size, chunk_int, preamble,
    lod_blocks, trailing).  lod_blocks[i] = raw bytes of LOD i's
    'i32 mat_count + mat_count*110' section.  trailing = whatever
    follows the last LOD block (names/bboxes), preserved verbatim."""
    info = find_chunk(file_data, 'DNKS', endian)
    if not info:
        return None
    cs_start, ds, chunk_size = info
    chunk_int = struct.unpack_from(f'{endian}i', file_data, cs_start + 4)[0]
    pre_len = 28
    preamble = bytes(file_data[ds:ds + pre_len])
    p = ds + pre_len
    end = cs_start + chunk_size
    lod_blocks = []
    while p + 4 <= end:
        mc = struct.unpack_from(f'{endian}i', file_data, p)[0]
        if mc < 0 or mc > 4096 or p + 4 + mc * 110 > end:
            break                       # reached the trailing section
        blk_end = p + 4 + mc * 110
        lod_blocks.append(bytes(file_data[p:blk_end]))
        p = blk_end
    trailing = bytes(file_data[p:end])
    return (cs_start, chunk_size, chunk_int, preamble, lod_blocks, trailing)


def _dnks_template_hfields(lod_block, endian=LE):
    """Return (stride_h4, fmt_h6) from a LOD block's first submesh —
    these are fixed per file and reused for newly-created submeshes."""
    if len(lod_block) >= 4 + 14:
        h4 = struct.unpack_from(f'{endian}H', lod_block, 4 + 8)[0]
        h6 = struct.unpack_from(f'{endian}H', lod_block, 4 + 12)[0]
        return h4, h6
    return 0, 0


def build_dnks_lod_block(submeshes, stride_h4, fmt_h6, endian=LE):
    """submeshes = list of dicts {mat_id, face_count, vert_count,
    palette(list of <=48 int16)}.  Returns the i32-count + records bytes."""
    out = bytearray(struct.pack(f'{endian}i', len(submeshes)))
    for sm in submeshes:
        fc = min(int(sm['face_count']), 65535)
        vc = min(int(sm['vert_count']), 65535)
        ic = min(fc * 3, 65535)
        out += struct.pack(f'{endian}7H',
                           int(sm['mat_id']) & 0xFFFF, fc, fc, ic,
                           stride_h4, vc, fmt_h6)
        # DNKS bone palette: 48 int16. Padding DEPENDS on whether the format
        # is SKINNED — getting it wrong either hides geometry or crashes:
        #   SKINNED (BONE_WTS1 bit 0x10, e.g. 0x0BDA — characters): unused
        #     slots REPEAT the first valid bone, NEVER -1. A -1 (0xFFFF) is an
        #     out-of-range bone index -> engine fetches a garbage skinning
        #     matrix -> submesh INVISIBLE. (stock-kendra split-by-material fix.)
        #   STATIC (no BONE_WTS1, e.g. 0x0BCA — vehicles/props): the game's OWN
        #     convention is ALL -1. The vertex buffer carries NO weights, so a
        #     real bone index makes the engine try to SKIN a weightless buffer
        #     -> out-of-bounds read -> CRASH ON LOAD. (Proven: stock samson
        #     static submeshes are 100% -1; rewriting them to bone 0 crashed.)
        is_skinned = bool(int(fmt_h6) & 0x0010)   # VertexFlags.BONE_WTS1
        raw = [int(x) for x in (sm.get('palette') or []) if x is not None]
        if is_skinned:
            valid = [x for x in raw if x >= 0]
            fill = valid[0] if valid else 0
            pal = [(x if x >= 0 else fill) for x in raw]
            pal = (pal + [fill] * 48)[:48]
        else:
            pal = (raw + [-1] * 48)[:48]   # static: preserve -1, never bone 0
        out += struct.pack(f'{endian}48h', *pal)
    return bytes(out)


def build_dnks_chunk(chunk_int, preamble, lod_blocks, trailing, endian=LE):
    payload = bytes(preamble) + b''.join(lod_blocks) + bytes(trailing)
    chunk_size = 12 + len(payload)
    magic = encode_chunk_magic('DNKS', endian)
    return magic + struct.pack(f'{endian}ii', chunk_int, chunk_size) + payload


# ============================================================
# DNKS trailing (per-submesh names + bboxes) parser / builder
# ============================================================
#
# The DNKS "trailing" payload that follows the submesh-block region is
# documented in mesh.py::parse_dnks_chunk.  It is laid out as:
#
#   [u32 blockCount]                              # total submeshes across all LODs
#   for k in range(blockCount):
#       meta[52]:
#         meta[ 0: 4] = f32 metric (LOD-switch distance)
#         meta[ 4:16] = 3 x f32 bbox_min
#         meta[16:28] = 3 x f32 bbox_max
#         meta[28:44] = 16 bytes (likely bounding sphere center+radius — preserved verbatim)
#         meta[44:48] = i32 LOD index
#         meta[48:52] = 4 bytes reserved
#       [u32 nameLen][nameLen bytes ascii][1 byte 0x00 terminator]
#
# The game uses the names for runtime lookups (animation bones, ragdoll
# attachment, equipment slots) and the per-submesh bbox for visibility
# culling and LOD selection.  When we add / remove / replace submeshes
# we MUST rewrite this section or the engine reads garbage and crashes.

def parse_dnks_trailing(trailing, endian=LE):
    """Parse the DNKS trailing bytes into a list of structured entries.

    Returns a list of dicts in file order:
      { 'meta':    bytes (52),       # the raw 52-byte meta record
        'metric':  float,
        'bb_min':  (x, y, z),
        'bb_max':  (x, y, z),
        'lod':     int,
        'name':    str,
        'raw_namelen':  int }        # original L value (== len(name) + 1)

    Returns None if the trailing is malformed — the caller should then
    fall back to preserving the bytes verbatim (legacy behaviour)."""
    if len(trailing) < 4:
        return None
    try:
        en = endian
        p  = 0
        block_count = struct.unpack_from(f'{en}I', trailing, p)[0]; p += 4
        entries = []
        for _ in range(block_count):
            if p + 52 > len(trailing):
                return None
            meta   = bytes(trailing[p:p + 52]); p += 52
            metric = struct.unpack_from(f'{en}f', meta,  0)[0]
            bb_min = struct.unpack_from(f'{en}3f', meta, 4)
            bb_max = struct.unpack_from(f'{en}3f', meta, 16)
            lod    = struct.unpack_from(f'{en}i', meta, 44)[0]
            if p + 4 > len(trailing):
                return None
            L = struct.unpack_from(f'{en}I', trailing, p)[0]; p += 4
            if not (1 <= L <= 256) or p + L > len(trailing):
                return None
            raw = bytes(trailing[p:p + L]); p += L
            name = raw.split(b'\x00')[0].decode('ascii', 'replace')
            # 1-byte record-separator NUL after the name bytes
            if p < len(trailing) and trailing[p] == 0:
                p += 1
            entries.append({
                'meta':   meta,
                'metric': metric,
                'bb_min': bb_min,
                'bb_max': bb_max,
                'lod':    lod,
                'name':   name,
                'raw_namelen': L,
            })
        return entries
    except Exception:
        return None


def build_dnks_trailing(entries, endian=LE):
    """Serialise structured trailing entries back to bytes.

    Each entry's `meta` 52-byte record is rewritten so that the bbox /
    metric / LOD fields match the current Python values (the 16 bytes
    at offset 28..44 are preserved verbatim from the source meta).
    """
    en  = endian
    out = bytearray()
    out += struct.pack(f'{en}I', len(entries))
    for e in entries:
        meta = bytearray(e.get('meta') or bytes(52))
        if len(meta) < 52:
            meta += b'\x00' * (52 - len(meta))
        struct.pack_into(f'{en}f',  meta,  0, float(e.get('metric', 0.0)))
        bb_min = e.get('bb_min', (0.0, 0.0, 0.0))
        bb_max = e.get('bb_max', (0.0, 0.0, 0.0))
        struct.pack_into(f'{en}3f', meta,  4, *bb_min)
        struct.pack_into(f'{en}3f', meta, 16, *bb_max)
        struct.pack_into(f'{en}i',  meta, 44, int(e.get('lod', 0)))
        out += bytes(meta)
        nm = (e.get('name') or '').encode('ascii', 'replace')
        # The original convention stores L = len(name)+1 then writes the
        # name + a separate trailing NUL.  Preserve that.
        out += struct.pack(f'{en}I', len(nm) + 1)
        out += nm + b'\x00' + b'\x00'
    return bytes(out)


def update_dnks_trailing_lod_bbox(entries, target_lod, lod_bb_min, lod_bb_max,
                                   new_name=None):
    """Update ONLY the target LOD's bbox (and optionally name) in the trailing.

    IMPORTANT — DNKS trailing structure (verified on stock kendra):
      block_count = number of LOD-GROUPS (typically == file's LOD count),
      NOT the submesh count.  Each entry names ONE LOD as a group, e.g.
        [0] 'NPC_KENDRA_BODY_LOD0'   metric=5.5
        [1] 'NPC_KENDRA_BODY_LOD1'   metric=40.0
        [2] 'NPC_KENDRA_BODY_LOD2'   metric=80.0
        [3] 'NPC_KENDRA_BODY_LOD3'   metric=200.0
      with the bbox in meta[4:28] covering EVERY submesh of that LOD as
      one big aggregate.  Adding or removing per-submesh entries here
      corrupts the chunk: the engine reads beyond block_count and either
      walks off the end or interprets garbage as a name length.

    Therefore the rebuild path may only update the bbox / name of the
    EXISTING target-LOD entry — it must NEVER resize the list to match
    the submesh count.  Callers from older code which assumed
    per-submesh entries should switch to this function.

    Returns a NEW list with the bbox/name updated for whichever entry
    has lod == target_lod, or the original list if no match.
    """
    if entries is None:
        return None
    out = []
    found = False
    for e in entries:
        if not found and e['lod'] == target_lod:
            new = dict(e)
            new['bb_min'] = tuple(lod_bb_min)
            new['bb_max'] = tuple(lod_bb_max)
            if new_name:
                new['name'] = str(new_name)
            out.append(new)
            found = True
        else:
            out.append(e)
    return out


def resize_dnks_trailing_for_lod(entries, target_lod, new_count,
                                  per_sm_bboxes=None, per_sm_names=None,
                                  default_lod_metric=None):
    """DEPRECATED: kept as a no-op wrapper for backwards-compatibility.

    Previous (incorrect) implementation expanded the trailing to one
    entry per submesh; the real DNKS trailing has one entry per LOD-GROUP.
    Use `update_dnks_trailing_lod_bbox` instead.  This shim aggregates
    the per-submesh bboxes into a single LOD-spanning bbox and forwards
    to the correct function so callers that haven't migrated still work.
    """
    if entries is None or not per_sm_bboxes:
        return entries
    mn = [ float('inf')] * 3
    mx = [-float('inf')] * 3
    for bb in per_sm_bboxes:
        if bb is None:
            continue
        for axis in range(3):
            if bb[0][axis] < mn[axis]: mn[axis] = bb[0][axis]
            if bb[1][axis] > mx[axis]: mx[axis] = bb[1][axis]
    if mn[0] == float('inf'):
        return entries
    return update_dnks_trailing_lod_bbox(entries, target_lod, mn, mx)
