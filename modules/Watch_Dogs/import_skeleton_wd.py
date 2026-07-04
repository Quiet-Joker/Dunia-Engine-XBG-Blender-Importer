"""WD1 .skeleton file parser — native binary format (magic: nbCF).

Format overview (reverse-engineered from pelvis_ref.skeleton, validated
byte-for-byte against the embedded skeleton of char01.xbg):
  Header  : b'nbCF' + u32 version (=3) + two u32 counts + ~0x1b header bytes
  Per-bone: [transform block] [CONST4 marker 0d af 6f f9] [u8 name_len]
            [name + null].

KEY LAYOUT POINT — the transform block PRECEDES its CONST4+name; it does NOT
follow it.  So bone i's block starts where bone i-1's name ended
(name_end[i-1]) and runs up to bone i's CONST4 marker.  Bone 0 (Pelvis) has
its block in the file header, starting 0x5c bytes before the first marker.
Reading the block that FOLLOWS a name (the naive layout) yields the NEXT
bone's transform — an off-by-one that puts every bone on the wrong joint.

Transform block (offsets from block start = name_end[i-1]):
  +0x05..+0x08 : zlib.crc32 of the PRECEDING bone's name (u32 LE)
  +0x1a..+0x29 : quaternion (x, y, z, w) as 4×f32  → this bone's rotation
  +0x2f..+0x3a : translation (x, y, z) as 3×f32    → this bone's local pos
  ...constants (a9 30 e4 4d, f0 38 45 df, 13 b5 8f 49 Havok class hash)...

Parent hierarchy: NOT stored as usable parent pointers (the bytes near the
marker are depth-first traversal links — e.g. R Thigh links to L Toe).
Transforms are local to the ANATOMICAL parent (L Thigh and R Thigh share the
same local translation, both relative to Pelvis), so this module resolves
parents via a name table for the standard WD1 human character skeleton.

Five bones (L Wrist, R Cuff, L/R M Forearm Twist, R D Forearm Twist) use
Havok pointer-reference structures with no inline transform — runtime-computed
deform helpers.  The unit-quaternion + sane-translation guards detect them and
fall back to an identity transform on their anatomical parent.
"""

import struct
import zlib

try:
    import bpy
    import mathutils
except ImportError:
    bpy = None
    mathutils = None

_MAGIC   = b'nbCF'
_CONST4  = b'\x0d\xaf\x6f\xf9'

# Anatomical parent hierarchy for the standard WD1 human character skeleton.
# Any bone name not in this table falls back to the file-sequential predecessor.
_CHAR_HIERARCHY = {
    'Pelvis'             : None,
    # Left leg
    'L Thigh'            : 'Pelvis',
    'L Calf'             : 'L Thigh',
    'L Foot'             : 'L Calf',
    'L Toe'              : 'L Foot',
    # Right leg
    'R Thigh'            : 'Pelvis',
    'R Calf'             : 'R Thigh',
    'R Foot'             : 'R Calf',
    'R Toe'              : 'R Foot',
    # Spine
    'Spine'              : 'Pelvis',
    'Spine1'             : 'Spine',
    'Spine2'             : 'Spine1',
    # Left arm
    'L Clavicle'         : 'Spine2',
    'L UpperArm'         : 'L Clavicle',
    'L Forearm'          : 'L UpperArm',
    'L Hand'             : 'L Forearm',
    # Left fingers
    'L Index01'          : 'L Hand',
    'L Index02'          : 'L Index01',
    'L Index03'          : 'L Index02',
    'L Middle_Meta'      : 'L Hand',
    'L Middle01'         : 'L Middle_Meta',
    'L Middle02'         : 'L Middle01',
    'L Middle03'         : 'L Middle02',
    'L Ring_Meta'        : 'L Hand',
    'L Ring01'           : 'L Ring_Meta',
    'L Ring02'           : 'L Ring01',
    'L Ring03'           : 'L Ring02',
    'L Hand_Meta'        : 'L Hand',
    'L Pinky01'          : 'L Hand_Meta',
    'L Pinky02'          : 'L Pinky01',
    'L Pinky03'          : 'L Pinky02',
    'L Thumb01'          : 'L Hand',
    'L Thumb02'          : 'L Thumb01',
    'L Thumb03'          : 'L Thumb02',
    'L Wrist'            : 'L Hand',
    'L Cuff'             : 'L Forearm',
    # Right arm
    'R Clavicle'         : 'Spine2',
    'R UpperArm'         : 'R Clavicle',
    'R Forearm'          : 'R UpperArm',
    'R Hand'             : 'R Forearm',
    # Right fingers
    'R Index01'          : 'R Hand',
    'R Index02'          : 'R Index01',
    'R Index03'          : 'R Index02',
    'R Middle_Meta'      : 'R Hand',
    'R Middle01'         : 'R Middle_Meta',
    'R Middle02'         : 'R Middle01',
    'R Middle03'         : 'R Middle02',
    'R Ring_Meta'        : 'R Hand',
    'R Ring01'           : 'R Ring_Meta',
    'R Ring02'           : 'R Ring01',
    'R Ring03'           : 'R Ring02',
    'R Hand_Meta'        : 'R Hand',
    'R Pinky01'          : 'R Hand_Meta',
    'R Pinky02'          : 'R Pinky01',
    'R Pinky03'          : 'R Pinky02',
    'R Thumb01'          : 'R Hand',
    'R Thumb02'          : 'R Thumb01',
    'R Thumb03'          : 'R Thumb02',
    'R Cuff'             : 'R Forearm',
    'R Wrist'            : 'R Hand',
    # Neck / head
    'Neck'               : 'Spine2',
    'Head'               : 'Neck',
    'Mullet Root'        : 'Head',
    'Mullet Point'       : 'Mullet Root',
    'hat'                : 'Head',
    # Upper-arm twist deform helpers
    'L D UpperArm Twist' : 'L UpperArm',
    'L U UpperArm Twist' : 'L UpperArm',
    # Forearm twist deform helpers
    'L U Forearm Twist'  : 'L Forearm',
    'L M Forearm Twist'  : 'L Forearm',
    'L D Forearm Twist'  : 'L Forearm',
    'R D UpperArm Twist' : 'R UpperArm',
    'R U UpperArm Twist' : 'R UpperArm',
    'R U Forearm Twist'  : 'R Forearm',
    'R M Forearm Twist'  : 'R Forearm',
    'R D Forearm Twist'  : 'R Forearm',
}


def _is_unit_quat(q, tol=0.01):
    return abs(sum(x*x for x in q)**0.5 - 1.0) < tol


def _is_sane_trans(t, limit=10.0):
    """Check translation components are finite and within ±10 m (character scale)."""
    import math
    return all(math.isfinite(v) and abs(v) < limit for v in t)


def parse_wd1_skeleton(path):
    """Parse a WD1 .skeleton file.

    Returns a list of bone dicts:
        {'name': str, 'parent': int (-1 = root), 'pos': (x,y,z), 'quat': (w,x,y,z)}
    Positions and quaternions are ready to pass straight to the same armature
    builder used by the XBG importer (build_wd_model → edit_bone.head/tail).
    """
    data = open(path, 'rb').read()
    if data[:4] != _MAGIC:
        raise ValueError(f"not a WD1 .skeleton file (magic {data[:4]!r})")

    # --- locate all bone entries via CONST4 markers ---
    raw_bones = []
    pos = 0
    while True:
        idx = data.find(_CONST4, pos)
        if idx == -1:
            break
        n_off = idx + 4
        n_len = data[n_off]
        if 0 < n_len < 64:
            nr = data[n_off + 1: n_off + 1 + n_len]
            if (nr[-1:] == b'\x00'
                    and all(0x20 <= b < 0x7f or b == 0 for b in nr)):
                name = nr[:-1].decode('latin-1')
                name_end = n_off + 1 + n_len
                raw_bones.append({'name': name,
                                  'name_end': name_end,
                                  'const4': idx})
        pos = idx + 1

    if not raw_bones:
        raise ValueError("no bones found in .skeleton file")

    # CRITICAL — the transform block PRECEDES its [CONST4][name], not follows
    # it.  So bone i's data block starts where the PREVIOUS bone's name ended
    # (name_end[i-1]) and runs up to this bone's CONST4 marker.  Bone 0's block
    # (Pelvis) sits in the file header just before the first marker; its start
    # is a fixed 0x5c bytes back from CONST4[0].  Verified byte-exact against
    # the embedded skeleton of char01.xbg (69/69 rotations, 65/69 positions;
    # the four misses are two 4 mm thumb-tip deltas and the two pointer-ref
    # twist bones that carry no inline transform).
    #
    # Within each block (offsets from block start):
    #   +0x1a  quaternion (x, y, z, w) f32×4
    #   +0x2f  translation (x, y, z)   f32×3
    parsed = []
    for i, rb in enumerate(raw_bones):
        block_start = (raw_bones[i - 1]['name_end'] if i >= 1
                       else raw_bones[0]['const4'] - 0x5c)
        block_end = rb['const4']          # marker right after this block
        block_len = block_end - block_start

        quat = None
        trans = (0.0, 0.0, 0.0)
        if block_start >= 0 and block_len >= 0x3b:  # need +0x2f + 12 bytes
            q_raw = struct.unpack_from('<4f', data, block_start + 0x1a)  # (x,y,z,w)
            t_raw = struct.unpack_from('<3f', data, block_start + 0x2f)  # (x,y,z)
            if _is_unit_quat(q_raw):
                # Convert file (x,y,z,w) → Blender Quaternion (w,x,y,z)
                quat = (q_raw[3], q_raw[0], q_raw[1], q_raw[2])
                trans = t_raw if _is_sane_trans(t_raw) else (0.0, 0.0, 0.0)
        if quat is None:
            # Pointer-reference or short-block bone: no inline transform; use identity
            quat = (1.0, 0.0, 0.0, 0.0)
            trans = (0.0, 0.0, 0.0)
        parsed.append({'name': rb['name'], 'pos': trans, 'quat': quat})

    # --- assign anatomical parent indices ---
    name_to_idx = {b['name']: i for i, b in enumerate(parsed)}
    for i, b in enumerate(parsed):
        par_name = _CHAR_HIERARCHY.get(b['name'])
        if par_name is None and b['name'] not in _CHAR_HIERARCHY:
            # Unknown bone: fall back to file-sequential predecessor
            par_name = parsed[i - 1]['name'] if i > 0 else None
        b['parent'] = name_to_idx.get(par_name, -1) if par_name else -1

    return parsed


def build_wd1_skeleton_armature(context, bones, name):
    """Create a Blender armature from the parsed bone list.

    Uses the same approach as build_wd_model in import_wd.py so the resulting
    armature is compatible with WD1 MAB animation import and weight painting.
    Returns the created armature object.
    """
    if bpy is None:
        raise RuntimeError("bpy unavailable — run inside Blender")

    Mat  = mathutils.Matrix
    Quat = mathutils.Quaternion
    Vec  = mathutils.Vector

    ad      = bpy.data.armatures.new(name + '_Armature')
    arm_obj = bpy.data.objects.new(ad.name, ad)
    context.collection.objects.link(arm_obj)
    context.view_layer.objects.active = arm_obj

    bpy.ops.object.mode_set(mode='EDIT')
    world = [None] * len(bones)
    ebs   = []
    for i, b in enumerate(bones):
        local = (Mat.Translation(Vec(b['pos']))
                 @ Quat(b['quat']).to_matrix().to_4x4())
        p = b['parent']
        world[i] = (world[p] @ local
                    if 0 <= p < i and world[p] is not None else local)
        eb      = ad.edit_bones.new(b['name'])
        head    = world[i].to_translation()
        eb.head = head
        eb.tail = head + world[i].to_3x3() @ Vec((0.0, 0.05, 0.0))
        ebs.append(eb)

    for i, b in enumerate(bones):
        if 0 <= b['parent'] < i:
            ebs[i].parent = ebs[b['parent']]

    bpy.ops.object.mode_set(mode='OBJECT')
    return arm_obj
