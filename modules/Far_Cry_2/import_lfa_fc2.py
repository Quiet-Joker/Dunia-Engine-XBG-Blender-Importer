"""Avatar facial animation: .lfa (facial rig poses) + .lfe (expressions).

Reverse-engineered 2026-06-10 from the shipped files in
animations/_common/facial (validated against npc_kendra_body.xbg —
all 25 LFA bone hashes resolve to the XBG's facial bone CRC32s).

LFA v3 ('fA', version 3) — per-head facial POSE LIBRARY
    header (0x18):
        u16 version, u16 magic 'Af'
        u32 deltas_end          (file offset just past the delta array)
        u32 header_size (0x18)
        u32 bone_count
        u32 deltas_offset
        u32 delta_count
    bone records @0x18, 64 bytes each:
        f32x3 bind position      @ 0   (LFA-local units, larger than XBG's)
        f32x4 bind quat (w,x,y,z)@12
        u32   name CRC32         @28   == zlib.crc32(bone_name) == XBG NODE crc
        f32x4 inverse bind quat  @32
        f32x2 scales             @48
        u32   delta_byte_offset  @56   (absolute file offset)
        u32   delta_count        @60
    delta records, 12 bytes each (grouped per bone, see bone records):
        u32  packed rotation: smallest-three, DLL-exact (see unpack_lfa_quat)
             0xDFF7FDFF == identity.  NOTE: quats/positions are in the LFA's
             y/z-mirrored centimetre frame (see _lfa_quat_to_xbg).
        s16x3 translation delta (scaled fixed point, see _DT_SCALE)
        u16  pose index
    pose table @deltas_end:
        16 zero bytes, then u32 subhdr {total, 0x20, nametbl_rel, pose_count,
        nametbl_rel, ...}; 40-byte pose records; then u32 name-offset table
        and the NUL-separated pose name strings.

LFE v2 ('Ef', version 2) — an EXPRESSION/emotion clip over LFA pose channels
    header (0x34):
        u16 version, u16 magic 'Ef', u32 total_size, u32 header_size (0x34),
        u32 channel_count, u32 data_offset, u32 total_keys, u32 total_size,
        u32 0, u32 hash
    channel records @0x34, 20 bytes each:
        u32 keys_offset (absolute), u32 key_count, u32 -1, u32 0,
        u32 pose-name CRC32 (zlib.crc32 of the LFA pose name, e.g. 'jawOpen')
    keys: key_count x (f32 time_seconds, f32 value)
"""

import os
import struct
import math
import zlib

try:
    import bpy
    import mathutils
except ImportError:          # standalone analysis
    bpy = None
    mathutils = None

from .import_mab_fc2 import parse_xbg_skeleton

# s16 translation fixed-point scale, in LFA-local units.  DLL-EXACT:
# const 0.000976592 (== 1/1024) at VA 0x1104F6C8, applied to the s16
# translation deltas in the same unpacker as the rotations.
_DT_SCALE = 1.0 / 1024.0


# DLL-exact constants (Dunia_Retail_1.02_decrypted.dll, LFA quat unpacker at
# VA 0x10318138; scale at 0x1104F6DC).  scale == (1/sqrt2)/511.
_LFA_QSCALE = 0.0013837706064805388
_LFA_QBIAS = 0x1FF                      # 511, integer-subtracted before scale


def unpack_lfa_quat(p):
    """Packed u32 -> (w, x, y, z).  0xDFF7FDFF decodes to identity.

    Smallest-three encoding, DLL-exact (disassembled, not guessed):
      * bits 30-31 = index of the OMITTED (largest-magnitude) component
      * bits 20-29 / 10-19 / 0-9 = the three stored components, each
        decoded as ((v & 0x3FF) - 511) * scale  (range +/- 1/sqrt2)
      * stored comps fill slots idx, idx^1, idx^2; the reconstructed
        largest (sqrt(1 - sum of squares)) goes to slot idx^3
      * slot order is (w, x, y, z) — identity packs largest at slot 0
    """
    idx = (p >> 30) & 3
    f0 = (((p >> 20) & 0x3FF) - _LFA_QBIAS) * _LFA_QSCALE
    f1 = (((p >> 10) & 0x3FF) - _LFA_QBIAS) * _LFA_QSCALE
    f2 = ((p & 0x3FF) - _LFA_QBIAS) * _LFA_QSCALE
    q = [0.0, 0.0, 0.0, 0.0]
    q[idx] = f0
    q[idx ^ 1] = f1
    q[idx ^ 2] = f2
    s = 1.0 - f0 * f0 - f1 * f1 - f2 * f2
    q[idx ^ 3] = math.sqrt(s) if s > 0.0 else 0.0
    return (q[0], q[1], q[2], q[3])


# ── LFA -> XBG coordinate-frame conversion ──────────────────────────────
# The LFA skeleton is authored in a Y/Z-SWAPPED (mirrored) frame relative to
# the XBG, in centimetres.  Verified on corp_f_head_05_tamara: every LFA
# bind position maps to the XBG one as (x, z, y)/100 and every LFA bind
# quat maps as (w, -x, -z, -y) — exact on all 25 shared bones, including
# the 9 with non-identity rests (cheeks/lips).  Conjugating a quaternion
# through the reflection S = swap(y,z) gives exactly (w, -S·v), hence the
# triple negation.  Skipping this conversion flips rotation directions
# (jawOpen closed the mouth) and bent the lip/cheek bones the wrong way.

def _lfa_quat_to_xbg(q):
    w, x, y, z = q
    return (w, -x, -z, -y)


def _lfa_vec_to_xbg(v):
    x, y, z = v
    return (x, z, y)


def parse_lfa(path):
    """Parse a .lfa -> {'bones': [...], 'poses': [names]}.

    Each bone: {'crc', 'pos', 'quat' (w,x,y,z),
                'deltas': {pose_index: ((w,x,y,z), (dx,dy,dz))}}
    """
    d = open(path, 'rb').read()
    ver, magic, deltas_end, hdr, n_bones, deltas_off, n_deltas = \
        struct.unpack_from('<HH5I', d, 0)
    if magic != 0x6641:                 # 'Af'
        raise ValueError("not a LFA file: %s" % path)

    bones = []
    for i in range(n_bones):
        off = hdr + i * 64
        px, py, pz = struct.unpack_from('<3f', d, off)
        qw, qx, qy, qz = struct.unpack_from('<4f', d, off + 12)
        crc, = struct.unpack_from('<I', d, off + 28)
        d_off, d_cnt = struct.unpack_from('<2I', d, off + 56)
        deltas = {}
        for k in range(d_cnt):
            p, dx, dy, dz, pose = struct.unpack_from('<Ihhhh', d, d_off + k * 12)
            deltas[pose] = (unpack_lfa_quat(p),
                            (dx * _DT_SCALE, dy * _DT_SCALE, dz * _DT_SCALE))
        bones.append({'crc': crc, 'pos': (px, py, pz),
                      'quat': (qw, qx, qy, qz), 'deltas': deltas})

    # ── pose names ───────────────────────────────────────────────────────
    # The pose section's zero padding and sub-table layout vary between
    # files, so self-calibrate instead: the file ends with NUL-separated
    # pose-name strings, preceded directly by their u32 offset table.
    # Find the string block, then read the table backwards from it.
    end = len(d)
    while end > 0 and d[end - 1] == 0:
        end -= 1
    s = end
    while s > deltas_end:
        c = d[s - 1]
        if c != 0 and not (0x20 <= c < 0x7F):
            break
        s -= 1
    # s..end ≈ string block (may start mid-string after binary data; align
    # to the first NUL-terminated boundary)
    first_str = s
    while first_str < end and d[first_str] == 0:
        first_str += 1
    poses = []
    # count strings and find block start by splitting
    block = d[first_str:end]
    parts = [p for p in block.split(b'\x00') if p]
    if not parts:
        return {'bones': bones, 'poses': []}
    # table of len(parts) u32 entries sits right before the first string
    n_poses = len(parts)
    tbl = first_str - 4 * n_poses
    entries = struct.unpack_from('<%dI' % n_poses, d, tbl)
    base = first_str - entries[0]
    for rel in entries:
        a = base + rel
        e = d.index(b'\x00', a)
        poses.append(d[a:e].decode('latin-1'))
    return {'bones': bones, 'poses': poses}


def parse_lfe(path):
    """Parse a .lfe -> [{'crc': pose_name_crc, 'keys': [(t_seconds, value)]}]."""
    d = open(path, 'rb').read()
    ver, magic, total, hdr, n_chan, data_off, n_keys = \
        struct.unpack_from('<HH5I', d, 0)
    if magic != 0x6645:                 # 'Ef'
        raise ValueError("not a LFE file: %s" % path)
    chans = []
    for i in range(n_chan):
        off = hdr + i * 20
        k_off, k_cnt, _neg, _z, crc = struct.unpack_from('<IIiII', d, off)
        keys = [struct.unpack_from('<2f', d, k_off + k * 8) for k in range(k_cnt)]
        chans.append({'crc': crc, 'keys': keys})
    return chans


def _match_bones(lfa, skel):
    """Map LFA bone index -> XBG bone index via the shared name CRC32."""
    crc2idx = {}
    for i, b in enumerate(skel):
        crc2idx[zlib.crc32(b['name'].encode('latin-1')) & 0xFFFFFFFF] = i
    out = {}
    for li, lb in enumerate(lfa['bones']):
        xi = crc2idx.get(lb['crc'])
        if xi is not None:
            out[li] = xi
    return out


def _unit_ratio(lfa, skel, bone_map):
    """LFA bind positions use a larger unit than the XBG — estimate the ratio
    from matched bones with non-trivial positions."""
    num = den = 0.0
    for li, xi in bone_map.items():
        lp = lfa['bones'][li]['pos']
        xp = skel[xi]['pos']
        ln = math.sqrt(sum(v * v for v in lp))
        xn = math.sqrt(sum(v * v for v in xp))
        if ln > 1.0 and xn > 1e-4:
            num += xn
            den += ln
    return (num / den) if den > 0 else 1.0


def _pose_locals(lfa, skel, bone_map, pose_idx, unit):
    """Local matrices {xbg_index: L} for one pose.

    The LFA stores each bone's ABSOLUTE local rotation for the pose, in the
    LFA's own (y/z-mirrored) frame — non-moving poses store the bone's exact
    LFA rest quat.  So: convert to the XBG frame, then REPLACE the rest
    local.  The s16 translation is a delta from the rest pivot in the same
    mirrored frame; convert and rescale with the bind-position unit ratio."""
    Mat = mathutils.Matrix
    Quat = mathutils.Quaternion
    Vec = mathutils.Vector
    out = {}
    for li, xi in bone_map.items():
        d = lfa['bones'][li]['deltas'].get(pose_idx)
        if d is None:
            continue
        dq, dt = d
        pos = Vec(skel[xi]['pos']) + Vec(_lfa_vec_to_xbg(dt)) * unit
        out[xi] = (Mat.Translation(pos) @
                   Quat(_lfa_quat_to_xbg(dq)).to_matrix().to_4x4())
    return out


def apply_lfa_poses(context, lfa, arm_obj, xbg_path):
    """Build a pose-library action: pose i keyed at frame i+1, with a
    timeline marker carrying the pose name.  Returns (n_poses, n_bones)."""
    if bpy is None:
        raise RuntimeError("bpy unavailable")
    Mat = mathutils.Matrix

    skel = parse_xbg_skeleton(xbg_path)
    if not skel:
        raise RuntimeError("could not parse XBG skeleton")
    bone_map = _match_bones(lfa, skel)
    if not bone_map:
        raise RuntimeError(
            "no LFA bone hash matches this skeleton — wrong head model?")
    unit = _unit_ratio(lfa, skel, bone_map)
    parents = [b['parent'] for b in skel]

    rest_local = [Mat.Translation(b['pos']) @
                  mathutils.Quaternion(b['quat']).to_matrix().to_4x4()
                  for b in skel]
    rest_world = []
    for i, lm in enumerate(rest_local):
        p = parents[i]
        rest_world.append(rest_world[p] @ lm
                          if p is not None and 0 <= p < i else lm.copy())

    if context.view_layer.objects.active is not arm_obj:
        context.view_layer.objects.active = arm_obj
    if arm_obj.mode != 'POSE':
        bpy.ops.object.mode_set(mode='POSE')
    if arm_obj.animation_data is None:
        arm_obj.animation_data_create()
    action = bpy.data.actions.new(name="LFA_poses")
    arm_obj.animation_data.action = action
    pbones = arm_obj.pose.bones

    scene = context.scene
    for mk in [m for m in scene.timeline_markers if m.name.startswith('LFA:')]:
        scene.timeline_markers.remove(mk)

    keyed_bones = set()
    for pi, pname in enumerate(lfa['poses']):
        frame = pi + 1
        locals_ = _pose_locals(lfa, skel, bone_map, pi, unit)
        for xi in sorted({x for x in bone_map.values()}):
            nm = skel[xi]['name']
            if nm not in pbones:
                continue
            pb = pbones[nm]
            L = locals_.get(xi, rest_local[xi])
            p = parents[xi]
            rwp = rest_world[p] if (p is not None and 0 <= p < len(skel)) \
                else Mat.Identity(4)
            ml = pb.bone.matrix_local
            # convention-independent deformation (same as the MAB importer)
            basis = (ml.inverted() @ rwp @ L @ rest_local[xi].inverted()
                     @ rwp.inverted() @ ml)
            pb.rotation_mode = 'QUATERNION'
            pb.matrix_basis = basis
            pb.keyframe_insert('rotation_quaternion', frame=frame)
            pb.keyframe_insert('location', frame=frame)
            keyed_bones.add(nm)
        scene.timeline_markers.new('LFA:' + pname, frame=frame)

    scene.frame_start = 1
    scene.frame_end = max(2, len(lfa['poses']))
    bpy.ops.object.mode_set(mode='OBJECT')
    return len(lfa['poses']), len(keyed_bones)


def apply_lfe_expression(context, lfa, lfe_chans, arm_obj, xbg_path, fps=30):
    """Animate an LFE clip: per frame, blend the active poses' bone deltas
    (weighted nlerp of rotations, weighted sum of translations)."""
    if bpy is None:
        raise RuntimeError("bpy unavailable")
    Mat = mathutils.Matrix
    Quat = mathutils.Quaternion
    Vec = mathutils.Vector

    skel = parse_xbg_skeleton(xbg_path)
    if not skel:
        raise RuntimeError("could not parse XBG skeleton")
    bone_map = _match_bones(lfa, skel)
    if not bone_map:
        raise RuntimeError("no LFA bone hash matches this skeleton")
    unit = _unit_ratio(lfa, skel, bone_map)
    parents = [b['parent'] for b in skel]

    # channel crc -> pose index
    pose_crc = {zlib.crc32(p.encode('latin-1')) & 0xFFFFFFFF: i
                for i, p in enumerate(lfa['poses'])}
    chans = [(pose_crc[c['crc']], c['keys'])
             for c in lfe_chans if c['crc'] in pose_crc and c['keys']]
    n_skipped = sum(1 for c in lfe_chans if c['keys']) - len(chans)
    if not chans:
        raise RuntimeError(
            "none of this LFE's %d channels match the LFA's pose names — "
            "this expression targets a different facial pose set (try the "
            "76-pose quaridge .lfa, which covers all expression_* channels)"
            % len(lfe_chans))
    length = max(k[-1][0] for _, k in chans)
    n_frames = max(2, int(round(length * fps)) + 1)

    def value_at(keys, t):
        if t <= keys[0][0]:
            return keys[0][1]
        for a, b in zip(keys, keys[1:]):
            if t <= b[0]:
                f = (t - a[0]) / max(1e-9, b[0] - a[0])
                return a[1] + (b[1] - a[1]) * f
        return keys[-1][1]

    rest_local = [Mat.Translation(b['pos']) @
                  Quat(b['quat']).to_matrix().to_4x4() for b in skel]
    rest_world = []
    for i, lm in enumerate(rest_local):
        p = parents[i]
        rest_world.append(rest_world[p] @ lm
                          if p is not None and 0 <= p < i else lm.copy())

    if context.view_layer.objects.active is not arm_obj:
        context.view_layer.objects.active = arm_obj
    if arm_obj.mode != 'POSE':
        bpy.ops.object.mode_set(mode='POSE')
    if arm_obj.animation_data is None:
        arm_obj.animation_data_create()
    action = bpy.data.actions.new(name="LFE_expression")
    arm_obj.animation_data.action = action
    pbones = arm_obj.pose.bones

    # Per-(bone,pose) DELTA from rest: rest_local_rot^-1 @ absolute_pose_rot.
    # The stored quat is the bone's ABSOLUTE local pose rotation in the
    # LFA's mirrored frame (see _pose_locals); convert to the XBG frame
    # first, then weight each pose's delta-from-rest and compose the result
    # back onto rest.  Translations are per-pose deltas from the rest pivot
    # (converted + unit-rescaled) and blend by weighted sum.
    rest_rot_inv = {xi: Quat(skel[xi]['quat']).inverted()
                    for xi in bone_map.values()}
    pose_delta = {}   # (pose_idx, xi) -> delta Quat
    pose_dt = {}      # (pose_idx, xi) -> delta Vector (XBG units)
    for pi, _ in chans:
        for li, xi in bone_map.items():
            dlt = lfa['bones'][li]['deltas'].get(pi)
            if dlt is not None:
                pose_delta[(pi, xi)] = (rest_rot_inv[xi] @
                                        Quat(_lfa_quat_to_xbg(dlt[0])))
                pose_dt[(pi, xi)] = Vec(_lfa_vec_to_xbg(dlt[1])) * unit

    for f in range(n_frames):
        t = f / float(fps)
        weights = [(pi, value_at(keys, t)) for pi, keys in chans]
        weights = [(pi, w) for pi, w in weights if abs(w) > 1e-4]
        # accumulate per-bone delta rotations weighted toward each pose
        acc = {}      # xbg_idx -> Quaternion accum (delta from rest)
        acc_t = {}    # xbg_idx -> Vector accum (translation delta)
        for pi, w in weights:
            for xi in bone_map.values():
                dd = pose_delta.get((pi, xi))
                if dd is None:
                    continue
                q0 = acc.get(xi, Quat((1, 0, 0, 0)))
                # negative weights drive the inverse pose; slerp factor
                # must stay within [0, 1] (Blender raises otherwise)
                dqq = dd
                ww = w
                if ww < 0.0:
                    dqq = dqq.conjugated()
                    ww = -ww
                acc[xi] = q0 @ Quat((1, 0, 0, 0)).slerp(dqq, min(1.0, ww))
                acc_t[xi] = (acc_t.get(xi, Vec((0, 0, 0))) +
                             pose_dt[(pi, xi)] * w)
        for xi in sorted(bone_map.values()):
            nm = skel[xi]['name']
            if nm not in pbones:
                continue
            pb = pbones[nm]
            dq = acc.get(xi, Quat((1, 0, 0, 0)))
            rq = Quat(skel[xi]['quat'])
            pos = Vec(skel[xi]['pos']) + acc_t.get(xi, Vec((0, 0, 0)))
            L = (Mat.Translation(pos) @
                 (rq @ dq).to_matrix().to_4x4())
            p = parents[xi]
            rwp = rest_world[p] if (p is not None and 0 <= p < len(skel)) \
                else Mat.Identity(4)
            ml = pb.bone.matrix_local
            basis = (ml.inverted() @ rwp @ L @ rest_local[xi].inverted()
                     @ rwp.inverted() @ ml)
            pb.rotation_mode = 'QUATERNION'
            pb.matrix_basis = basis
            pb.keyframe_insert('rotation_quaternion', frame=f + 1)
            pb.keyframe_insert('location', frame=f + 1)

    scene = context.scene
    scene.frame_start = 1
    scene.frame_end = n_frames
    scene.render.fps = fps
    bpy.ops.object.mode_set(mode='OBJECT')
    return n_frames, len(chans), n_skipped
