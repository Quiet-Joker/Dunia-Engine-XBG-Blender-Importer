"""LKS skeleton splicer — add new bones to an existing Avatar `.skeleton`
byte-exactly, so custom jiggle bones exist at runtime.

We do NOT re-encode the whole file (too many engine-private fields).  Instead we
APPEND new bone records just before the file's trailing bone-set table, cloning
an existing sibling bone's block for all the bytes we don't fully understand
(notably ``block[0]`` — a bone-set / LOD membership bitmask) and patching only
the fields we've verified against real files:

  block[5:9]   skeleton hash (u32, constant per file = header hash)
  block[13:29] local rotation quaternion (4×f32, X Y Z W)
  block[29:41] local position (3×f32)
  block[45:47] bone_seq_idx (u16) = new bone index
  block[47:49] parent_idx (u16)
  block[49:51] first_child (u16) = 0xFFFF (jiggle bones are leaves)
  block[51:53] next_sibling (u16)
  block[55:59] crc32(name) (u32, literal name, NOT lowercased)
  block[59:63] name_len (u32)  → name bytes → 0x00 terminator

New bones are inserted as the FIRST child of their parent: we point the parent's
first_child at the new bone and set the new bone's next_sibling to the parent's
old first_child.  That keeps the child/sibling cache consistent without touching
any other existing bone (and is harmless if the engine only uses parent_idx).
bone_count (u16 @ 0x10) is bumped.  See `import_lks.py` for the read side and
agents.md "GAME FORMAT REFERENCE" for the format notes.
"""

import struct
import zlib


def _parse_layout(data):
    """Return (header_hash, bone_count, [bone records], trailing_offset).

    Each record: dict(idx, name, parent, off, block_len, ext) where `off` is the
    file offset of the bone's block start and `block_len` = bytes from `off` up
    to (but not including) the name (63 standard / 71 extended)."""
    if data[:3] != b'LKS':
        raise ValueError("not an LKS skeleton")
    header_hash = struct.unpack_from('<I', data, 4)[0]
    bone_count = struct.unpack_from('<H', data, 16)[0]
    rnl = struct.unpack_from('<I', data, 74)[0]
    recs = [{'idx': 0, 'name': data[78:78 + rnl].decode('latin-1'),
             'parent': -1, 'off': 16, 'block_len': 62, 'ext': 0}]
    off = 78 + rnl + 1
    for i in range(1, bone_count):
        nl59 = struct.unpack_from('<I', data, off + 59)[0]
        if nl59 <= 256:
            nl, noff, ext, blen = nl59, off + 63, 0, 63
        else:
            nl = struct.unpack_from('<I', data, off + 67)[0]
            noff, ext, blen = off + 71, 1, 71
        name = data[noff:noff + nl].decode('latin-1')
        parent = struct.unpack_from('<H', data, off + 47)[0]
        recs.append({'idx': i, 'name': name,
                     'parent': parent if parent != 0xFFFF else -1,
                     'off': off, 'block_len': blen, 'ext': ext})
        off = noff + nl + 1
    return header_hash, bone_count, recs, off


def _build_block(template_block, header_hash, name, quat, pos,
                 seq_idx, parent_idx, next_sibling):
    """Clone a 63-byte standard template block and patch the known fields."""
    b = bytearray(template_block[:63])           # standard block only
    struct.pack_into('<I', b, 5, header_hash & 0xFFFFFFFF)
    struct.pack_into('<4f', b, 13, *quat)        # X Y Z W
    struct.pack_into('<3f', b, 29, *pos)
    struct.pack_into('<H', b, 45, seq_idx)
    struct.pack_into('<H', b, 47, parent_idx)
    struct.pack_into('<H', b, 49, 0xFFFF)        # first_child (leaf)
    struct.pack_into('<H', b, 51, next_sibling & 0xFFFF)
    struct.pack_into('<I', b, 55, zlib.crc32(name.encode('latin-1')) & 0xFFFFFFFF)
    nb = name.encode('latin-1')
    struct.pack_into('<I', b, 59, len(nb))
    return bytes(b) + nb + b'\x00'


def splice_bones(orig_bytes, new_bones, log=None):
    """Append `new_bones` to an LKS skeleton, returning new file bytes.

    new_bones: list of dicts, each:
        name        : str (must be unique, not already in skeleton)
        parent_name : str (existing bone) OR parent_idx : int
        pos         : (x, y, z)   local position (LKS convention)
        quat        : (x, y, z, w) local rotation (LKS convention)
    Bones are added in list order; a new bone may parent to an earlier new bone.
    """
    def _log(m):
        if log:
            log(m)

    header_hash, bone_count, recs, trailing_off = _parse_layout(orig_bytes)
    name_to_idx = {r['name']: r['idx'] for r in recs}
    data = bytearray(orig_bytes)

    # Standard (non-extended, non-root) blocks usable as clone templates.
    std = [r for r in recs if r['idx'] != 0 and r['ext'] == 0]
    if not std:
        raise ValueError("no standard bone block to use as a template")

    # Children index, so we can clone a real sibling and find old first_child.
    children = {}
    for r in recs:
        if r['parent'] != -1:
            children.setdefault(r['parent'], []).append(r['idx'])

    appended = bytearray()
    new_recs = []          # (idx, name, off-in-appended) for first_child patching
    next_idx = bone_count
    # Track first_child edits to apply to `data` (existing region) at the end.
    fc_edits = {}          # parent_idx -> new first_child idx
    # Track next_sibling for the new bones (set to parent's CURRENT first child).
    parent_firstchild = {}  # parent_idx -> current first child idx (may be new)

    for nb in new_bones:
        name = nb['name']
        if name in name_to_idx:
            raise ValueError("bone %r already exists in skeleton" % name)
        if 'parent_idx' in nb:
            pidx = int(nb['parent_idx'])
        else:
            pname = nb['parent_name']
            if pname not in name_to_idx:
                raise ValueError("parent bone %r not found" % pname)
            pidx = name_to_idx[pname]

        # Clone an existing sibling's block (same depth / bone-set context);
        # fall back to any standard bone if the parent has no standard child.
        sib = None
        for ci in children.get(pidx, []):
            cr = recs[ci] if ci < len(recs) else None
            if cr and cr['ext'] == 0:
                sib = cr; break
        tmpl = sib or std[0]
        if sib is None:
            _log("[lks] no sibling template for parent #%d; cloning %r (b0/bone-set "
                 "flags may be off — verify in game)" % (pidx, std[0]['name']))
        tblock = orig_bytes[tmpl['off']:tmpl['off'] + 63]

        # Insert as first child: new.next_sibling = parent's CURRENT first child
        # (which may be a previously-added new bone under the same parent).
        if pidx in parent_firstchild:
            old_fc = parent_firstchild[pidx]
        elif pidx == 0:
            old_fc = _root_first_child(orig_bytes, recs)
        else:
            old_fc = struct.unpack_from('<H', data, recs[pidx]['off'] + 49)[0]
        block = _build_block(tblock, header_hash, name, nb['quat'], nb['pos'],
                             next_idx, pidx, old_fc)
        appended += block
        name_to_idx[name] = next_idx
        children.setdefault(pidx, []).append(next_idx)
        parent_firstchild[pidx] = next_idx
        fc_edits[pidx] = next_idx
        new_recs.append((next_idx, name, pidx))
        _log("[lks] + bone #%d %r parent #%d (clone of %r) next_sib=%s"
             % (next_idx, name, pidx, tmpl['name'],
                'none' if old_fc == 0xFFFF else old_fc))
        next_idx += 1

    # Apply parent first_child edits in the EXISTING region (root excluded —
    # root's first_child lives in its 62-byte block; patch it too if needed).
    for pidx, fc in fc_edits.items():
        if pidx == 0:
            _patch_root_first_child(data, recs, fc)
        else:
            struct.pack_into('<H', data, recs[pidx]['off'] + 49, fc & 0xFFFF)

    # Splice: [bones region] + [new bones] + [trailing table], bump count.
    out = bytearray(data[:trailing_off]) + appended + bytearray(data[trailing_off:])
    struct.pack_into('<H', out, 16, next_idx)
    _log("[lks] bone_count %d -> %d (+%d), size %d -> %d"
         % (bone_count, next_idx, len(new_bones), len(orig_bytes), len(out)))
    return bytes(out)


def _root_first_child(data, recs):
    """Root (62-byte block @16) first_child offset differs; read it."""
    # Root block layout: first_child sits at block[48:50] region per import_lks
    # (bone_seq_idx @48, parent @50). Root's first child is the first record
    # whose parent == 0; the engine derives it, so we return that.
    kids = [r['idx'] for r in recs if r['parent'] == 0]
    return min(kids) if kids else 0xFFFF


def _patch_root_first_child(data, recs, fc):
    """Root has no standard first_child u16 we rely on; leave the 62-byte block
    intact (engine builds root children from parent_idx). No-op by design."""
    return
