"""Far Cry 3 .skeleton (LKS) importer — standalone armature builder.

FC3-era LKS files come in (at least) two block layouts (62/63-byte FC2-style
character records vs 48/44-byte compact prop records — the ceiling fan), so
fixed-offset parsing breaks across files.  This parser is VARIANT-PROOF: it
anchors on the bone-name records, which are constant across every observed
variant:

    [u32 crc32(name)] [u32 name_length] [name bytes]

and reads each bone's data BACKWARD from its crc position C:

    quat (x,y,z,w  4×f32)  @ C-36
    pos  (x,y,z    3×f32)  @ C-20
    parent index   (u16)   @ C-4   (0xFFFF = root)

Verified: fanceiling_01_ref.skeleton (3 bones — identity quats, blade offset
pos, parents 0xFFFF/0/1) and FC3 hips_ref.skeleton (70 bones — all quats
unit length, LeftLeg pos.x = 0.425 thigh length, parent chain Hips →
LeftUpLeg → LeftLeg → …).

NOT supported: FC4's hips_ref.skeleton — that is a DIFFERENT, name-less LKS
variant (bone data only, const 0x23 instead of 0x1F); FC4 armatures come
from the .xbg import instead.
"""

import struct
import zlib

try:
    import bpy
    import mathutils
except Exception:
    bpy = None
    mathutils = None

from ..Core.debug import VerboseLogger as vlog


class LksError(Exception):
    pass


def parse_fc3_skeleton(filepath):
    """Returns [{'name', 'quat'(x,y,z,w), 'pos'(x,y,z), 'parent'(int|-1)}]."""
    data = open(filepath, 'rb').read()
    if len(data) < 80 or data[:3] != b'LKS':
        raise LksError("not an LKS .skeleton file")
    bone_count = struct.unpack_from('<H', data, 16)[0]
    if not (0 < bone_count < 2048):
        raise LksError("implausible bone count %d" % bone_count)

    bones = []
    i = 20
    n = len(data)
    while i + 9 <= n and len(bones) < bone_count:
        nl = struct.unpack_from('<I', data, i + 4)[0]
        if 0 < nl < 64 and i + 8 + nl <= n:
            raw = data[i + 8:i + 8 + nl]
            if (all(0x20 <= c < 0x7F for c in raw)
                    and struct.unpack_from('<I', data, i)[0]
                        == (zlib.crc32(raw) & 0xFFFFFFFF)):
                c = i
                quat = struct.unpack_from('<4f', data, c - 36)
                pos = struct.unpack_from('<3f', data, c - 20)
                parent = struct.unpack_from('<H', data, c - 4)[0]
                bones.append({
                    'name': raw.decode('latin-1'),
                    'quat': quat,                 # x, y, z, w
                    'pos': pos,
                    'parent': -1 if parent == 0xFFFF else parent,
                })
                i += 8 + nl
                continue
        i += 1

    if len(bones) != bone_count:
        raise LksError(
            "parsed %d of %d bones — this looks like the name-less FC4 "
            ".skeleton variant, which is not supported (FC4 armatures come "
            "from the .xbg import)" % (len(bones), bone_count))
    return bones


def build_fc3_skeleton_armature(context, bones, name):
    """Build an armature from parsed LKS bones (local quat/pos + parent)."""
    if bpy is None:
        raise RuntimeError("bpy unavailable")
    Mat = mathutils.Matrix
    Quat = mathutils.Quaternion
    Vec = mathutils.Vector

    ad = bpy.data.armatures.new(name)
    arm = bpy.data.objects.new(ad.name, ad)
    context.collection.objects.link(arm)
    context.view_layer.objects.active = arm
    bpy.ops.object.mode_set(mode='EDIT')

    world = [None] * len(bones)
    ebs = []
    for i, b in enumerate(bones):
        x, y, z, w = b['quat']
        local = (Mat.Translation(Vec(b['pos'])) @
                 Quat((w, x, y, z)).to_matrix().to_4x4())
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
    vlog.log("[fc3 skeleton] built %d bones" % len(bones))
    return arm
