"""Far Cry 3 MAB animation importer (self-contained clone of the FC2 module,
per the one-folder-per-game architecture — FC3 mabs are version byte 0x61 in
the same Dunia codec family this parser already validates against).

Clean implementation based on buu342's validated
Dunia Animation Extractor (github.com/buu342/PY-DuniaAnimationExtractor).

WHAT WORKS (validated against the ceiling-fan ground truth):
  * Version detection (FC2 0x4C / FC3 0x61 / Blood Dragon 0x62 ...).
  * Section table parse in buu342's order.
  * Quaternion smallest-three codec (identical to our prior codec).
  * Per-time-group keyframe block decode for SINGLE-bone animations:
      block = [quat 6B][u16 bitmask][popcount(bitmask) × quat 6B]
    The fan blade decodes to a clean Z-spin (0 -> 92.9 -> 185.8 ...).
  * XBG bone list in true NODE-chunk file order.

NOT YET SOLVED (frontier — buu342's wiki: "stored contiguously, somehow..."):
  * Multi-bone contiguous quaternion packing in one block (FC2 characters).
  * track -> bone routing (engine resolves per-bone bOff at load, likely by
    name/hash). See DLL analysis in import_mab.py docstring.

This module is deliberately standalone so it can be validated outside Blender
(run `py -3 import_mab_buu.py <file.mab> <model.xbg>`), then called from the
Blender importer for the cases it handles.
"""

import os
import struct
import math

try:
    import mathutils
    import bpy
except Exception:
    mathutils = None
    bpy = None


# ---------------------------------------------------------------------------
# Version / header layout
# ---------------------------------------------------------------------------

SKIP = 16  # every raw section offset is relative to the 16-byte header

# version byte 0 -> (animlen_offset, sections_offset, name)
_VERSION_LAYOUT = {
    0x4C: (0x84, 0x88, "Far Cry 2"),
    0x61: (0xC4, 0xC8, "Far Cry 3"),
    0x62: (0xC4, 0xC8, "Far Cry 3: Blood Dragon"),
    # FC4 / Primal / FC5 use larger headers; offsets TBD, fall back to FC3.
    0x81: (0xC4, 0xC8, "Far Cry 4"),
    0x82: (0xC4, 0xC8, "Far Cry Primal"),
    0xB0: (0xC4, 0xC8, "Far Cry 5 / New Dawn"),
}

# buu342's 9-entry section order (note entries 0 and 1 are swapped in-file)
SECTION_LABELS = ['UnkSec2', 'UnkSec1', 'RootRot', 'Keyframes', 'UnkSec3',
                  'Offsets', 'Events', 'UnkSec4', 'UnkSec5']


def detect_version(d):
    """Return (animlen_off, sections_off, version_name, version_byte)."""
    vb = d[0]
    if vb in _VERSION_LAYOUT:
        a, s, name = _VERSION_LAYOUT[vb]
        return a, s, name, vb
    # Unknown — default to FC2 layout, caller may warn.
    return 0x84, 0x88, "Unknown(0x%02X)" % vb, vb


# ---------------------------------------------------------------------------
# Quaternion codec (smallest-three) — verbatim from buu342 / our prior code
# ---------------------------------------------------------------------------

_QSCALE = 4.315969e-05
_QBIAS = 0.7071068


def unpack_quaternion(FW, SW, TW):
    """(FirstWord u16, SecondWord u16, ThirdWord s16) -> (x, y, z, w) or None.

    ENGINE-EXACT (disassembled from Dunia_Retail_1.02_decrypted.dll at
    VA 0x103101D0, 2026-06-09):
      FW: movzx + `and 0x7fff` — value is low 15 bits, bit 15 routes.
      SW: movzx + `and 0x7fff` — same.
      TW: **movsx** — SIGNED 16-bit, full value (buu342's table was correct;
          the MabPlayback C++ port's unsigned read is wrong).
    Case permutations (FW.15, SW.15) -> (x,y,z,w), verbatim from the DLL:
      (1,1)->(f1,f2,f3,f4)  (0,1)->(f1,f2,f4,f3)
      (1,0)->(f1,f4,f2,f3)  (0,0)->(f4,f1,f2,f3)
    """
    f1 = float(FW & 0x7fff) * _QSCALE - _QBIAS
    f2 = float(SW & 0x7fff) * _QSCALE - _QBIAS
    f3 = float(TW) * _QSCALE - _QBIAS
    s = 1.0 - f1 * f1 - f2 * f2 - f3 * f3
    if s < 0.0:
        return None
    f4 = math.sqrt(s)
    if (FW & 0x8000) == 0:
        if (SW & 0x8000) != 0:
            return (f1, f2, f4, f3)
    elif (SW & 0x8000) != 0:
        return (f1, f2, f3, f4)
    if (FW & 0x8000) != 0:
        return (f1, f4, f2, f3)
    return (f4, f1, f2, f3)


def read_quat(d, p):
    """Read a 6-byte packed quaternion at offset p -> (x,y,z,w) or None."""
    if p + 6 > len(d):
        return None
    FW, SW = struct.unpack_from('<HH', d, p)
    TW, = struct.unpack_from('<h', d, p + 4)   # SIGNED — DLL uses movsx
    return unpack_quaternion(FW, SW, TW)


def quat_xyzw_to_blender(q):
    """(x,y,z,w) -> Blender Quaternion tuple (w,x,y,z)."""
    if q is None:
        return None
    return (q[3], q[0], q[1], q[2])


# ---------------------------------------------------------------------------
# Section table
# ---------------------------------------------------------------------------

class MabSections:
    """Holds (abs_offset, size) per named section, plus version + animlen."""

    def __init__(self):
        self.version_name = ""
        self.version_byte = 0
        self.animlen = 0.0
        self.offsets = {}   # name -> abs offset (with SKIP added)
        self.sizes = {}     # name -> size in bytes
        self.raw = {}       # name -> raw offset (before SKIP)

    def __getitem__(self, name):
        return (self.offsets.get(name, 0), self.sizes.get(name, 0))


def parse_sections(d):
    """Parse header + 9-entry section table. Returns MabSections."""
    animlen_off, sec_off, vname, vbyte = detect_version(d)
    s = MabSections()
    s.version_name = vname
    s.version_byte = vbyte
    s.animlen = struct.unpack_from('<f', d, animlen_off)[0]

    raw = struct.unpack_from('<9i', d, sec_off)
    for i, lbl in enumerate(SECTION_LABELS):
        s.raw[lbl] = raw[i]
        s.offsets[lbl] = raw[i] + SKIP if raw[i] > 0 else 0

    # Sizes: next-larger offset minus this one (buu342's algorithm).
    present = sorted([(raw[i], SECTION_LABELS[i])
                      for i in range(9) if raw[i] > 0])
    filesize_nohdr = len(d) - SKIP
    for idx, (off, lbl) in enumerate(present):
        nxt = present[idx + 1][0] if idx + 1 < len(present) else filesize_nohdr
        s.sizes[lbl] = nxt - off
    return s


# ---------------------------------------------------------------------------
# Keyframe section decode (buu342 block structure)
# ---------------------------------------------------------------------------

class KeyGroup:
    """One time-group block: a list of bone tuples.

    Each tuple = {'q': base quat (x,y,z,w), 'mask': u16, 'extra': [quats]}.
    For single-bone animations there is exactly one tuple per group and the
    decode is exact. For multi-bone FC2 the tuple split is NOT yet correct
    (frontier); callers should treat >1 tuple results as provisional.
    """

    def __init__(self, index, start, size):
        self.index = index
        self.start = start
        self.size = size
        self.tuples = []


def decode_keyframes(d, sections, expect_single_bone=False):
    """Decode the Rotation Keyframes section.

    Returns (header_info, [KeyGroup, ...]).
    header_info = {'Nq', 'fc', 'magic_or_tc', 'lastoff', 'secsize', 'offsets'}.

    Uses buu342's magic-float group-table for FC3; for FC2 the same +8 offset
    table applies but the +4 field is the u32 track count (no magic float).
    """
    kf = sections.offsets['Keyframes']
    n = len(d)
    Nq = struct.unpack_from('<H', d, kf)[0]
    fc = struct.unpack_from('<H', d, kf + 2)[0]

    # FC3 stores a magic float at +4; FC2 stores u32 track count at +4.
    is_fc3 = sections.version_byte in (0x61, 0x62, 0x81, 0x82, 0xB0)
    if is_fc3:
        magic = struct.unpack_from('<f', d, kf + 4)[0]
        lastoff = ((int(magic * sections.animlen) >> 3) * 4) + 8
        plus4 = magic
    else:
        plus4 = struct.unpack_from('<I', d, kf + 4)[0]  # track count
        # group count from frame count: ((fc-1)>>3)+1 groups (+1 terminator)
        ngroups = ((max(1, fc) - 1) >> 3) + 1
        lastoff = 8 + ngroups * 4

    # Section size without padding is the i32 right after the last offset.
    secsize = struct.unpack_from('<i', d, kf + lastoff + 4)[0] \
        if kf + lastoff + 8 <= n else 0

    # Read the offset table (block boundaries) starting at kf+8.
    offsets = []
    p = kf + 8
    cur = 0
    guard = 0
    while cur < secsize and guard < 4096:
        v = struct.unpack_from('<i', d, p)[0]
        if v <= cur or v > secsize:
            offsets.append(v)
            break
        offsets.append(v)
        p += 4
        cur = v
        guard += 1

    header = {'Nq': Nq, 'fc': fc, 'plus4': plus4,
              'lastoff': lastoff, 'secsize': secsize, 'offsets': offsets}

    groups = []
    for gi in range(len(offsets) - 1):
        bstart = kf + offsets[gi]
        bend = kf + offsets[gi + 1]
        g = KeyGroup(gi, offsets[gi], offsets[gi + 1] - offsets[gi])
        p = bstart
        # buu342 tuple walk: [quat][u16 mask][popcount quats], repeat to block end.
        while p + 8 <= bend:
            q = read_quat(d, p)
            p += 6
            mask = struct.unpack_from('<H', d, p)[0]
            p += 2
            cnt = bin(mask).count('1')
            extra = []
            for _ in range(cnt):
                if p + 6 > bend:
                    break
                extra.append(read_quat(d, p))
                p += 6
            g.tuples.append({'q': q, 'mask': mask, 'extra': extra})
            if expect_single_bone:
                break  # only the first tuple is the (single) bone
        groups.append(g)
    return header, groups


def decode_root_rotations(d, sections):
    """Section[2] RootRot: i32 count, 4 pad, count × 6-byte quats."""
    rr = sections.offsets['RootRot']
    n = len(d)
    if not (0 < rr + 8 <= n):
        return []
    count = struct.unpack_from('<i', d, rr)[0]
    if not (0 < count < 4096):
        return []
    out = []
    p = rr + 8
    for _ in range(count):
        out.append(read_quat(d, p))
        p += 6
    return out


# ---------------------------------------------------------------------------
# XBG bone reader (true NODE-chunk order) — buu342 GetMeshBones
# ---------------------------------------------------------------------------

class XbgBone:
    __slots__ = ('name', 'parent', 'index', 'iB')

    def __init__(self, name, parent, index, iB=-1):
        self.name = name
        self.parent = parent
        self.index = index
        # iB (int @ bone_record+56): the bone's sequential index within the MAB
        # rotation stream, as written by the Dunia exporter.
        #   iB == -1  → bone is completely outside the MAB rotation stream
        #               (mesh-link nodes, world Root, attachment/Holster bones,
        #                Camera bone).  Never animated, never in RootRot.
        #   0 <= iB < mask_size  → bone participates: either keyframed (animated)
        #               or constant (in RootRot), depending on the clip.
        #   iB >= mask_size  → bone is always-procedural for this skeleton type
        #               (e.g. R Thumb02/03 and R arm twist bones on Kendra).
        #               mask_size = tc + rr_count, constant per skeleton.
        self.iB = iB


def read_xbg_bones(xbg_path):
    """Return [XbgBone, ...] in NODE-chunk file order (the real bone indices).

    Each bone carries .iB — the MAB stream index field from bone_record+56.
    iB == -1 means the bone is outside the MAB rotation stream entirely.
    """
    with open(xbg_path, 'rb') as f:
        magic = struct.unpack("4s", f.read(4))[0][::-1].decode('ascii', 'replace')
        if magic != "MESH":
            return []
        majorver = struct.unpack("<H", f.read(2))[0]
        struct.unpack("<H", f.read(2))  # minorver
        f.seek(28, 0)
        chunkcount = struct.unpack("<L", f.read(4))[0]
        bones = []
        for _c in range(chunkcount):
            chunkname = struct.unpack("4s", f.read(4))[0][::-1].decode('ascii', 'replace')
            f.seek(4, 1)
            chunksize = struct.unpack("<L", f.read(4))[0]
            f.seek(8, 1)
            if chunkname != "NODE":
                f.seek(chunksize - 20, 1)
                continue
            bonecount = struct.unpack("<L", f.read(4))[0]
            for i in range(bonecount):
                f.seek(12, 1)                                    # skip CRC32+w0+w1 (+0..+11)
                parentid = struct.unpack("<l", f.read(4))[0]    # +12 parent
                f.seek(28, 1)                                    # skip quat(16)+pos(12) → at +44
                f.seek(12, 1)                                    # skip fA (3 floats)    → at +56
                iB_val = struct.unpack("<i", f.read(4))[0]      # +56 MAB stream index
                f.seek(8, 1)                                     # skip fC+iD            → at +68
                if majorver == 46:
                    f.seek(8, 1)                                 # v46 has 8 extra bytes
                namelen = struct.unpack("<L", f.read(4))[0]     # +68 (v42) / +76 (v46)
                bonename = struct.unpack("%ss" % namelen, f.read(namelen))[0].decode('ascii', 'replace')
                parent = bones[parentid].name if (0 <= parentid < len(bones)) else ""
                bones.append(XbgBone(bonename, parent, i, iB_val))
                f.seek(1, 1)
                if majorver == 46:
                    f.seek(4, 1)
            break
    return bones


# ---------------------------------------------------------------------------
# Blender applier (single-bone — the validated case)
# ---------------------------------------------------------------------------

def build_single_bone_timeline(d, sections):
    """Return [(frame_index, (x,y,z,w)), ...] for a single-bone animation.

    Frame index = group*8 + subframe.  The base quat sits at subframe 0; each
    set bit b in the group's bitmask places an extra quat at subframe **b+1**
    (bit 0 = subframe 1) — the base quat already covers subframe 0, so the
    mask only flags the frames AFTER it, same as the multi-bone decoder's
    secondary keys living at subframes 1..7.  The old `subframe = b` reading
    put every extra key one frame early, which made constant-speed clips
    (the FC3 ceiling fan: exactly 23.226°/frame when correct) jitter with a
    visible mid-clip skip/rubber-band on playback.
    """
    header, groups = decode_keyframes(d, sections, expect_single_bone=True)
    timeline = []
    for g in groups:
        if not g.tuples:
            continue
        tp = g.tuples[0]
        base_frame = g.index * 8
        if tp['q'] is not None:
            timeline.append((base_frame, tp['q']))
        # Distribute extras onto the sub-frames flagged by the bitmask.
        bits = [b for b in range(16) if (tp['mask'] >> b) & 1]
        for ex_q, sub in zip(tp['extra'], bits):
            if ex_q is not None:
                timeline.append((base_frame + sub + 1, ex_q))
    timeline.sort(key=lambda kv: kv[0])
    return header, timeline


def _xbg_rest_quat(xbg_path, bone_name):
    """Look up one bone's local rest quaternion (w,x,y,z) from the XBG."""
    try:
        x = open(xbg_path, 'rb').read()
    except Exception:
        return None
    xn, i = len(x), 0
    target = bone_name.encode('ascii', 'replace')
    while i < xn - 58:
        if x[i:i + len(target)] == target and i >= 56:
            # validate it's a real name record (u32 len precedes the string)
            nl = struct.unpack_from('<I', x, i - 4)[0]
            if nl == len(target):
                qx, qy, qz, qw = struct.unpack_from('<4f', x, i - 56)
                return (qw, qx, qy, qz)
        i += 1
    return None


def apply_single_bone(context, d, sections, arm_obj, bone_name,
                      xbg_path=None, fps=None,
                      smooth_resample=True, resample_fps=60):
    """Apply a single-bone MAB rotation stream to *bone_name* on *arm_obj*.

    MAB stores ABSOLUTE local rotations; Blender pose bones want a delta from
    rest, so we apply rest_inv @ abs when the XBG rest quat is available.
    """
    if bpy is None:
        raise RuntimeError("bpy unavailable")
    header, timeline = build_single_bone_timeline(d, sections)
    if not timeline:
        raise RuntimeError("no keyframes decoded")

    Quat = mathutils.Quaternion
    rest_inv = Quat()
    if xbg_path:
        rq = _xbg_rest_quat(xbg_path, bone_name)
        if rq:
            rest_inv = Quat(rq).inverted()

    if context.view_layer.objects.active is not arm_obj:
        context.view_layer.objects.active = arm_obj
    if arm_obj.mode != 'POSE':
        bpy.ops.object.mode_set(mode='POSE')
    if arm_obj.animation_data is None:
        arm_obj.animation_data_create()
    action = bpy.data.actions.new(name="MAB_" + bone_name)
    arm_obj.animation_data.action = action

    if bone_name not in arm_obj.pose.bones:
        raise RuntimeError("bone %r not found on armature" % bone_name)
    pb = arm_obj.pose.bones[bone_name]
    pb.rotation_mode = 'QUATERNION'

    # Decode to (frame, wxyz) keys, then optionally SQUAD-resample for smooth
    # playback of sparse / low-fps streams (see quat_resample_fc2).
    raw = []
    for frame_idx, q in timeline:
        bq = quat_xyzw_to_blender(q)
        if bq is None:
            continue
        raw.append((frame_idx, bq))

    mult = 1
    if smooth_resample and sections.animlen > 0 and len(raw) >= 2:
        max_f = max(fr for fr, _ in raw)
        if max_f > 0:
            src_fps = max_f / sections.animlen
            if src_fps > 0:
                mult = max(1, min(8, int(round(resample_fps / src_fps))))
    if mult > 1 and len(raw) >= 2:
        from .quat_resample_fc3 import resample_rotation
        raw = resample_rotation(raw, mult)

    last_frame = 0
    prev = None
    frames = []
    quats = []
    for frame_idx, bq in raw:
        qq = Quat(bq)
        if prev is not None and prev.dot(qq) < 0:
            qq = Quat((-qq.w, -qq.x, -qq.y, -qq.z))
        prev = qq
        frames.append(frame_idx + 1)
        quats.append(rest_inv @ qq)         # keyed value == old pb.rotation_quaternion
        last_frame = max(last_frame, frame_idx + 1)
    if frames:
        _bulk_key_pose(action, pb, frames, quats)   # rotation only (locs=None)

    scene = context.scene
    scene.frame_start = 1
    scene.frame_end = max(2, last_frame)
    if fps:
        scene.render.fps = fps
    elif sections.animlen > 0:
        try:
            scene.render.fps = max(1, round(last_frame / sections.animlen))
        except Exception:
            pass
    bpy.ops.object.mode_set(mode='OBJECT')
    return len(timeline)


# ---------------------------------------------------------------------------
# Multi-bone support (FC2 characters) — provisional but validated routing
# ---------------------------------------------------------------------------
#
# Routing model (see AGENTS.md MAB 2026-06-05): the MAB stores NO controller
# IDs; tracks are in XBG NODE (skeleton) order.  RootRot (sec[2]) holds the
# constant rest rotations of the NON-animated, non-identity bones (also XBG
# order).  Therefore the animated bones for a clip are the first `tc` skeleton
# bones (XBG order, excluding the leading mesh nodes) whose rest rotation is
# NOT present in RootRot.
#
# Per-frame data: the first `tc` packed quats of each keyframe group block are
# the tc bones' rotations at that group's frame (group*8).  Sub-frame detail
# (the trailer bitmask) is not yet decoded, so this yields a correct-pose,
# coarse-framerate (every 8th frame) animation — enough to validate routing.

def _qnorm(q):
    if q is None:
        return None
    m = math.sqrt(sum(v * v for v in q))
    return tuple(v / m for v in q) if m > 1e-6 else None


def parse_xbg_skeleton(xbg_path):
    """Parse the XBG NODE chunk fully.

    Returns a list (NODE/skeleton order) of dicts:
      {'name', 'parent' (int or -1), 'quat' (w,x,y,z), 'pos' (x,y,z)}
    Layout per bone record (matches skeleton.py / buu342): parent int @ +12,
    quat xyzw @ +16, pos xyz @ +32, namelen @ +68, name @ +72 (+null, +4 if v46).
    """
    with open(xbg_path, 'rb') as f:
        magic = struct.unpack("4s", f.read(4))[0][::-1].decode('ascii', 'replace')
        if magic != "MESH":
            return []
        majorver = struct.unpack("<H", f.read(2))[0]
        struct.unpack("<H", f.read(2))
        f.seek(28, 0)
        chunkcount = struct.unpack("<L", f.read(4))[0]
        out = []
        for _c in range(chunkcount):
            chunkname = struct.unpack("4s", f.read(4))[0][::-1].decode('ascii', 'replace')
            f.seek(4, 1)
            chunksize = struct.unpack("<L", f.read(4))[0]
            f.seek(8, 1)
            if chunkname != "NODE":
                f.seek(chunksize - 20, 1)
                continue
            bonecount = struct.unpack("<L", f.read(4))[0]
            for _i in range(bonecount):
                f.seek(12, 1)                                    # skip CRC32+w0+w1
                parentid = struct.unpack("<l", f.read(4))[0]    # +12
                qx, qy, qz, qw = struct.unpack("<4f", f.read(16))  # +16
                px, py, pz = struct.unpack("<3f", f.read(12))   # +32
                f.seek(12, 1)                                    # skip fA (3 floats)
                iB_val = struct.unpack("<i", f.read(4))[0]      # +56 MAB stream index
                f.seek(8, 1)                                     # skip fC+iD
                if majorver == 46:
                    f.seek(8, 1)                                 # extra 8 bytes for v46
                namelen = struct.unpack("<L", f.read(4))[0]
                bonename = struct.unpack("%ss" % namelen, f.read(namelen))[0].decode('ascii', 'replace')
                out.append({'name': bonename, 'parent': parentid,
                            'quat': (qw, qx, qy, qz), 'pos': (px, py, pz),
                            'iB': iB_val})
                f.seek(1, 1)
                if majorver == 46:
                    f.seek(4, 1)
            break
    return out




def decode_root_rot_quats(d, sections):
    """RootRot (sec[2]) -> list of normalised (w,x,y,z) constant rotations."""
    rr = sections.offsets['RootRot']
    if not rr:
        return []
    count = struct.unpack_from('<i', d, rr)[0]
    if not (0 < count < 4096):
        return []
    out = []
    p = rr + 8
    for _ in range(count):
        q = read_quat(d, p)
        p += 6
        out.append(_qnorm(quat_xyzw_to_blender(q)) if q else None)
    return out


def _stream_counts(d, sections):
    """Return (tc, rr_count) for this clip."""
    kf = sections.offsets['Keyframes']
    tc = struct.unpack_from('<H', d, kf)[0]
    rr = sections.offsets['RootRot']
    rr_count = 0
    if rr:
        rr_count = struct.unpack_from('<i', d, rr)[0]
        if not (0 <= rr_count < 4096):
            rr_count = 0
    return tc, rr_count


# FC2 MAB header: per-clip routing bitmasks over the ANIMATION SKELETON's
# bone order (the .skeleton/LKS resource — NOT the XBG NODE order).
# 20-byte mask slots (up to 160 bones), validated on the corp skeleton
# (93 bones) against walk/sprint/geton/getoff + sprint_upbody (2026-06-09):
#   0x10 : constant-rotation mask — bit k (LSB-first) set means skeleton
#          bone k takes its pose from RootRot; RootRot[j] = j-th set bit.
#   0x24 : animated-rotation mask — keyframe track t = t-th set bit.
#   0x38 : animated-translation mask (UnkSec3 vec3 entries) — not consumed yet.
# Bones in neither rotation mask (twists, procedural pouches) are untouched.
# DLL ground truth: routing loop at VA 0x1030FBD8 iterates skeleton bone
# indices, tests bit edi in two masks, consumes the quat stream sequentially
# for set bits.
_MASK_CONST_OFF = 0x10
_MASK_ANIM_OFF = 0x24
_MASK_SLOT = 0x14   # 20 bytes per mask slot


def read_lks_bone_names(skeleton_path):
    """Parse a .skeleton (LKS) file; return bone names in skeleton order.

    Minimal standalone parser (layout documented in modules/import_lks.py).
    Returns [] on failure.
    """
    try:
        data = open(skeleton_path, 'rb').read()
    except Exception:
        return []
    if len(data) < 80 or data[:3] != b'LKS':
        return []
    bone_count = struct.unpack_from('<H', data, 16)[0]
    if not (0 < bone_count < 2048):
        return []

    # Every bone name in an LKS file is stored as
    #   [u32 crc32(name)] [u32 name_length] [name bytes]
    # (verified across FC2 character skeletons AND FC3 prop skeletons — the
    # surrounding per-bone block layout VARIES by bone type/game era, but
    # this hash+len+name triple is constant, and crc32 gives an essentially
    # zero-false-positive validator).  So instead of guessing block sizes,
    # scan the whole file for CRC-validated names; file order = bone order.
    import zlib
    names = []
    i = 20
    n = len(data)
    while i + 9 <= n and len(names) < bone_count:
        nl = struct.unpack_from('<I', data, i + 4)[0]
        if 0 < nl < 64 and i + 8 + nl <= n:
            raw = data[i + 8:i + 8 + nl]
            if (all(0x20 <= c < 0x7F for c in raw)
                    and struct.unpack_from('<I', data, i)[0]
                        == (zlib.crc32(raw) & 0xFFFFFFFF)):
                names.append(raw.decode('latin-1'))
                i += 8 + nl
                continue
        i += 1
    return names if len(names) == bone_count else []


def find_skeleton_file(xbg_path, xbg_bone_names, extra_dirs=()):
    """Locate the animation .skeleton file matching this model.

    Scans the XBG's folder (and extra_dirs) for *.skeleton files and picks
    the one whose bone names overlap the XBG's the most (>= 60% of its bones
    must exist in the XBG).  Returns (path, [names]) or (None, []).
    """
    xset = set(xbg_bone_names)
    dirs = []
    if xbg_path:
        dirs.append(os.path.dirname(os.path.abspath(xbg_path)))
    dirs.extend(extra_dirs)
    best = (None, [], 0.0)
    seen = set()
    for dd in dirs:
        if not dd or dd in seen or not os.path.isdir(dd):
            continue
        seen.add(dd)
        for fn in os.listdir(dd):
            if not fn.lower().endswith('.skeleton'):
                continue
            p = os.path.join(dd, fn)
            names = read_lks_bone_names(p)
            if not names:
                continue
            score = sum(1 for n in names if n in xset) / float(len(names))
            if score > best[2]:
                best = (p, names, score)
    if best[2] >= 0.6:
        return best[0], best[1]
    return None, []


def read_routing_masks(d, sections, n_bones):
    """Read the per-clip rotation routing bitmasks (see header notes above).

    Returns (anim_bits, const_bits) — lists of 0/1 per skeleton bone — only
    if popcounts match tc / rr_count; otherwise (None, None).
    """
    tc, rr_count = _stream_counts(d, sections)
    if n_bones <= 0 or n_bones > 8 * _MASK_SLOT:
        return None, None
    if _MASK_ANIM_OFF + _MASK_SLOT > len(d):
        return None, None

    def bits(off):
        return [(d[off + i // 8] >> (i % 8)) & 1 for i in range(n_bones)]

    const_b = bits(_MASK_CONST_OFF)
    anim_b = bits(_MASK_ANIM_OFF)
    if (sum(anim_b) == tc and sum(const_b) == rr_count
            and not any(a and c for a, c in zip(anim_b, const_b))):
        return anim_b, const_b
    return None, None


def find_animated_bones(d, sections, skel_names, bone_offset=0):
    """Return (tc, [animated bone names], [constant bone names]) for this clip.

    Engine model (validated against the corp .skeleton + DLL routing loop):
      * `skel_names` = bone names of the ANIMATION SKELETON (.skeleton/LKS
        resource) in file order — this is the mask bit domain.
      * Header mask @0x24 selects the keyframed bones: track t = t-th set bit.
      * Header mask @0x10 selects the constant bones: RootRot[j] = j-th set
        bit.  Bones in neither mask are not touched by the clip.
      * `bone_offset` > 0 places this model's skeleton at that bit offset
        inside a scripted-scene COMBINED rig (several characters + anchors
        concatenated) instead of at the leading block — this is how the
        user picks WHICH character of a multi-character clip to apply.
    """
    tc, rr_count = _stream_counts(d, sections)
    anim_b = const_b = None
    if bone_offset == 0:
        anim_b, const_b = read_routing_masks(d, sections, len(skel_names))
    if anim_b is None:
        # Scripted-scene clips animate a COMBINED rig (several characters +
        # scene anchors) — the mask domain is larger than this model's
        # skeleton.  Find the domain size whose popcounts validate; if our
        # skeleton is the leading block of the combined rig, the first
        # len(skel_names) bits still route correctly and the surplus tracks
        # are simply skipped (their bones don't exist on this armature).
        min_nb = bone_offset + len(skel_names)
        start = max(min_nb, len(skel_names) + (1 if bone_offset == 0 else 0))
        for nb in range(start, 8 * _MASK_SLOT + 1):
            anim_b, const_b = read_routing_masks(d, sections, nb)
            if anim_b is not None:
                print("[MAB] scripted-scene clip: mask domain is %d bones "
                      "(model skeleton has %d at offset %d) — %d tracks "
                      "outside this character's block are skipped"
                      % (nb, len(skel_names), bone_offset,
                         sum(anim_b[:bone_offset])
                         + sum(anim_b[bone_offset + len(skel_names):])))
                break
        if anim_b is None:
            raise ValueError(
                "MAB header masks do not validate against this skeleton "
                "(%d bones at offset %d; tc=%d rr=%d) — wrong .skeleton "
                "file or character offset?"
                % (len(skel_names), bone_offset, tc, rr_count))
    # Mask bits outside [bone_offset, bone_offset+len) belong to OTHER
    # characters/anchors of the combined rig: keep them in the routing
    # (track t = t-th set bit over the FULL domain) under placeholder
    # names so track indexing stays correct, but they never match a pose
    # bone and are skipped downstream.
    names_ext = (['<pre:%d>' % i for i in range(bone_offset)]
                 + list(skel_names))
    names_ext += ['<other:%d>' % i
                  for i in range(len(anim_b) - len(names_ext))]
    animated = [nm for bit, nm in zip(anim_b, names_ext) if bit]
    constant = [nm for bit, nm in zip(const_b, names_ext) if bit]
    return tc, animated, constant


def decode_group_track_quats(d, sections):
    """Return [(frame_index, [ (x,y,z,w) per track ]), ...].

    Coarse decode: the first `tc` quats of each group block = the tc tracks'
    rotations at frame group*8.
    """
    kf = sections.offsets['Keyframes']
    n = len(d)
    tc = struct.unpack_from('<I', d, kf + 4)[0]
    fc = struct.unpack_from('<H', d, kf + 2)[0]
    ngroups = ((max(1, fc) - 1) >> 3) + 1
    # offset table at kf+8
    offsets = []
    p = kf + 8
    for _ in range(ngroups + 2):
        v = struct.unpack_from('<i', d, p)[0]
        p += 4
        if v <= 0 or v > n:
            break
        offsets.append(v)
    # The LAST offset is the section-size terminator, not a block start.
    # Iterate only real blocks (len-1), and never past the frame count.
    out = []
    nblocks = max(0, len(offsets) - 1)
    for gi in range(nblocks):
        frame = gi * 8
        if frame >= fc:
            break
        block = kf + offsets[gi]
        row = []
        for t in range(tc):
            row.append(read_quat(d, block + t * 6))
        out.append((frame, row))
    return tc, out


def decode_full_keyframes(d, sections):
    """Decode the FULL sub-frame keyframe data (DLL-accurate).

    Block layout (reverse-engineered from the Dunia sampler, VA 0x1030F430):
      block = [N primary quats (6B, one per bone = sub-frame 0)]
              [N bitmask bytes (even-rounded), one per bone]
              [secondary quats: per bone, popcount(bitmask[i]) keys]
    bitmask byte (OR 0x80): bit 7 = sub-frame 0 (the primary), bit (7-sf) = a key
    at sub-frame sf.  Secondary quats for a bone fill its set sub-frames in
    increasing order.  N = u16[base] = animated-bone count.

    Returns (N, {bone_index: [(frame, (x,y,z,w)), ...]}) across all groups.
    """
    kf = sections.offsets['Keyframes']
    n = len(d)
    N = struct.unpack_from('<H', d, kf)[0]
    fc = struct.unpack_from('<H', d, kf + 2)[0]
    ngroups = ((max(1, fc) - 1) >> 3) + 1
    offsets = []
    p = kf + 8
    for _ in range(ngroups + 2):
        v = struct.unpack_from('<i', d, p)[0]
        p += 4
        if v <= 0 or v > n:
            break
        offsets.append(v)

    result = {i: [] for i in range(N)}
    bmsize = (N + 1) & ~1
    for g in range(min(ngroups, max(0, len(offsets) - 1))):
        block = kf + offsets[g]
        bm = block + 6 * N
        sec_ptr = bm + bmsize
        for i in range(N):
            primary = read_quat(d, block + i * 6)
            if primary is not None:
                result[i].append((g * 8, primary))
            mask = d[bm + i] | 0x80 if bm + i < n else 0x80
            for sf in range(1, 8):
                if (mask >> (7 - sf)) & 1:
                    q = read_quat(d, sec_ptr)
                    sec_ptr += 6
                    frame = g * 8 + sf
                    if q is not None and frame < fc:
                        result[i].append((frame, q))
    return N, result


def _fcurve_container(obj, action):
    """Slotted-action fcurve container (Blender 4.4+/5.x) or legacy fallback —
    shared implementation lives in shared/procedural.py."""
    from .procedural_fc3 import fcurve_container
    return fcurve_container(obj, action)


def _bulk_key_pose(action, pb, frames, quats, locs=None):
    """Populate a pose bone's rotation_quaternion (4) + (optional) location (3)
    fcurves in one bulk pass via foreach_set, instead of per-frame
    keyframe_insert().

    `frames` are 1-based frame numbers; `quats` are mathutils Quaternions (wxyz);
    `locs` are mathutils Vectors, or None to key rotation only. Assumes the bone
    has no existing keys in this action (true on a fresh MAB import — each bone
    is keyed exactly once)."""
    cont = _fcurve_container(pb.id_data, action)
    bp = pb.path_from_id()                 # 'pose.bones["Name"]'
    n = len(frames)
    channels = [('rotation_quaternion', 4, quats)]
    if locs is not None:
        channels.append(('location', 3, locs))
    for path, comps, vals in channels:
        for ci in range(comps):
            fc = cont.fcurves.new(f'{bp}.{path}', index=ci)
            fc.keyframe_points.add(n)
            flat = [0.0] * (2 * n)
            for k in range(n):
                flat[2 * k] = frames[k]
                flat[2 * k + 1] = vals[k][ci]
            fc.keyframe_points.foreach_set('co', flat)
            fc.update()


def apply_multi_bone(context, d, sections, arm_obj, xbg_path,
                     skeleton_path=None, extra_dirs=(), bone_offset=0,
                     emulate_helpers=True, smooth_resample=True,
                     resample_fps=60, twist_bake=True):
    """Apply a multi-bone MAB using a convention-INDEPENDENT deformation method.

    Track/RootRot routing comes from the MAB header masks over the ANIMATION
    SKELETON's bone order (.skeleton file — auto-discovered next to the XBG
    when `skeleton_path` is not given).

    The addon's XBG importer re-orients bones (tails aimed at children, auto
    roll) and rotates the armature object, so a Blender bone's rest frame is NOT
    the XBG rest quaternion.  Applying the MAB's local rotation naively (rest⁻¹ @
    anim) tips re-oriented bones over — this is what caused the left/right
    "symmetry" breakage.

    Instead we compute, in the XBG hierarchy, each bone's rest world matrix R_i
    and its animated local matrix L_i, and set the pose bone's basis to

        basis_i = ML_i⁻¹ · (R_parent · L_i · RL_i⁻¹ · R_parent⁻¹) · ML_i

    where ML_i = bone.matrix_local (Blender's real rest) and RL_i = rest local.
    Through Blender's FK this reconstructs the exact engine world deformation
    W_i·R_i⁻¹ for every bone, regardless of how the importer oriented it.  Only
    animated bones need a basis; the rest follow at their bind pose via FK.

    Returns (num_bones_keyed, [animated bone names]).
    """
    if bpy is None:
        raise RuntimeError("bpy unavailable")
    Mat = mathutils.Matrix
    Quat = mathutils.Quaternion

    skel = parse_xbg_skeleton(xbg_path)
    if not skel:
        raise RuntimeError("could not parse XBG skeleton")
    name2idx = {b['name']: i for i, b in enumerate(skel)}
    parents = [b['parent'] for b in skel]

    # Animation-skeleton bone order = routing mask domain.
    if skeleton_path:
        skel_names = read_lks_bone_names(skeleton_path)
        if not skel_names:
            raise RuntimeError("could not parse skeleton file: %s" % skeleton_path)
    else:
        skeleton_path, skel_names = find_skeleton_file(
            xbg_path, [b['name'] for b in skel], extra_dirs=extra_dirs)

    if skel_names:
        tc, animated, constant = find_animated_bones(
            d, sections, skel_names, bone_offset=bone_offset)
    else:
        # No .skeleton file anywhere — last resort: try the routing masks
        # over the XBG's own bone order.  When the model and animation rig
        # share the same bone layout the masks validate (popcounts match)
        # and the clip decodes fine; if they don't validate, routing would
        # be garbage, so stop with a actionable message instead.
        xbg_names = [b['name'] for b in skel]
        anim_b, const_b = read_routing_masks(d, sections, len(xbg_names))
        if anim_b is None:
            raise RuntimeError(
                "no matching .skeleton file found (looked next to the XBG "
                "and the .mab) and the clip's routing masks do not line up "
                "with the XBG bone order. Set the 'Animation Skeleton' path "
                "in the Debug panel to this rig's .skeleton file")
        print("[MAB] no .skeleton file found — falling back to the XBG bone "
              "order for routing (masks validate against it)")
        tc, _ = _stream_counts(d, sections)
        animated = [nm for bit, nm in zip(anim_b, xbg_names) if bit]
        constant = [nm for bit, nm in zip(const_b, xbg_names) if bit]
    N, keys_by_idx = decode_full_keyframes(d, sections)

    def local_mat(quat_wxyz, pos):
        return Mat.Translation(pos) @ Quat(quat_wxyz).to_matrix().to_4x4()

    rest_local = [local_mat(b['quat'], b['pos']) for b in skel]

    def world_mats(local):
        wm = [None] * len(local)
        for i in range(len(local)):
            p = parents[i]
            if p is not None and 0 <= p < i and wm[p] is not None:
                wm[i] = wm[p] @ local[i]
            else:
                wm[i] = local[i]
        return wm

    rest_world = world_mats(rest_local)

    # The armature's bones may have been placed from MB2O bind matrices,
    # which can differ from the NODE rest hierarchy (direhorse: the
    # contralateral Arms_Linkers).  Animation must be reconstructed against
    # the SAME rest the bones/skinning use, or limbs shift sideways when
    # posed.  The importer stores those matrices on the armature.
    bind_world = list(rest_world)            # default: NODE rest
    stored = arm_obj.get('xbg_bind_world')
    if stored:
        try:
            sd = stored.to_dict() if hasattr(stored, 'to_dict') else dict(stored)
        except Exception:
            sd = {}
        for i, b in enumerate(skel):
            v = sd.get(b['name'])
            if v is not None and len(v) == 16:
                bind_world[i] = Mat((v[0:4], v[4:8], v[8:12], v[12:16]))

    if context.view_layer.objects.active is not arm_obj:
        context.view_layer.objects.active = arm_obj
    if arm_obj.mode != 'POSE':
        bpy.ops.object.mode_set(mode='POSE')
    if arm_obj.animation_data is None:
        arm_obj.animation_data_create()
    action = bpy.data.actions.new(name="MAB_multi")
    arm_obj.animation_data.action = action
    pbones = arm_obj.pose.bones

    # Constant (RootRot) bones: RootRot[j] = j-th constant bone in mask order
    # (see find_animated_bones).  These hold a per-clip fixed pose — usually
    # near rest for the face, posed for held hands/arms.
    rr_quats = decode_root_rot_quats(d, sections)   # (w,x,y,z), index-aligned
    const_bones = [(nm, q) for nm, q in zip(constant, rr_quats) if q is not None]

    # Engine exclusion: AVATAR XBG bones with iB == -1 are PROCEDURAL
    # (twist roots, the model root).  The MAB still carries rig tracks for
    # them (the mask routing consumes the track slot) but the engine never
    # applies them — the procedural system drives those bones instead.
    # Applying their tracks distorts the skin (direhorse 'twist Root's).
    #
    # FC2 XBGs use the SAME field with a DIFFERENT meaning: marty flags 42
    # of 107 bones -1, including Head, both Forearms and the entire facial
    # rig — bones the body MAB genuinely animates (freezing the forearms
    # left the elbow behind and stretched the arm).  Detect the FC2-style
    # semantic (core bones / large fraction flagged) and trust the routing
    # masks alone for those models.
    neg = {b['name'] for b in skel if b.get('iB', 0) == -1}
    _CORE = ('Head', 'Pelvis', 'Spine', 'L Forearm', 'R Forearm',
             'L UpperArm', 'R UpperArm', 'L Thigh', 'R Thigh')
    fc2_style = any(n in neg for n in _CORE)
    procedural = set() if fc2_style else neg
    if fc2_style:
        print("[MAB] FC2-style XBG (iB==-1 flags core bones like %s) — "
              "engine-procedural exclusion disabled, trusting the routing "
              "masks" % ', '.join(n for n in _CORE if n in neg))

    # cache Blender rest matrix_local for every bone we will pose
    ml = {}
    for nm in list(animated) + [nm for nm, _ in const_bones]:
        if nm in pbones:
            ml[nm] = pbones[nm].bone.matrix_local.copy()

    keyed = set()
    last_frame = 1
    ident = Mat.Identity(4)

    def key_bone(nm, keys):
        """Pose + keyframe one bone.  keys = [(frame_0based, bq_wxyz), ...].

        IMPORTANT: matrix_basis carries BOTH the delta rotation and a
        compensating translation — both channels must be keyed, otherwise the
        location channel silently holds the last assigned value for the whole
        timeline (bones drift into the body)."""
        nonlocal last_frame
        if nm in procedural:
            return False
        if nm not in pbones or nm not in ml or nm not in name2idx:
            return False
        i = name2idx[nm]
        p = parents[i]
        mli = ml[nm]
        mli_inv = mli.inverted()
        # Engine-side parent world and own bind world.  PW(blender) must
        # satisfy PW @ ML^-1 == W_engine @ inv(bind): with parents posed the
        # same way this reduces to basis = ML^-1 @ BW_par @ L @ BW^-1 @ ML.
        # When the binds equal the NODE rest this is identical to the old
        # rwp @ L @ RL^-1 @ rwp^-1 conjugation.
        bwp = bind_world[p] if (p is not None and 0 <= p < len(bind_world)) else ident
        bw_inv = bind_world[i].inverted()
        pos = skel[i]['pos']
        pb = pbones[nm]
        pb.rotation_mode = 'QUATERNION'
        if not keys:
            return False
        # Decompose each key's basis matrix in PURE PYTHON (matrix_basis assign
        # just does mat.decompose() internally, so we reproduce it exactly) and
        # populate the fcurves in BULK. This avoids ~2 keyframe_insert() calls
        # per frame per bone — the old hot path that made >1000-key clips crawl.
        frames = []
        quats = []
        locs = []
        prev = None
        for frame, bq in keys:
            basis = mli_inv @ (bwp @ local_mat(bq, pos) @ bw_inv) @ mli
            loc, qq, _ = basis.decompose()
            if prev is not None and prev.dot(qq) < 0.0:
                qq.negate()
            prev = qq
            frames.append(frame + 1)
            quats.append(qq)
            locs.append(loc)
        _bulk_key_pose(action, pb, frames, quats, locs)
        last_frame = max(last_frame, frames[-1])
        return True

    # SQUAD smoothing: the engine evaluates spline-compressed rotation at the
    # game framerate, so a 15 fps clip still plays smoothly.  Bake dense
    # in-between keys with spherical-cubic interpolation through the decoded
    # keys (original key poses preserved exactly; frame numbers scale by mult,
    # and the fps formula below auto-scales because last_frame scales too).
    from .quat_resample_fc3 import resample_rotation
    mult = 1
    if smooth_resample and sections.animlen > 0:
        max_f = max((fr for ks in keys_by_idx.values() for fr, _ in ks),
                    default=0)
        if max_f > 0:
            src_fps = max_f / sections.animlen
            if src_fps > 0:
                mult = max(1, min(8, int(round(resample_fps / src_fps))))

    # Animated bones: full keyframe streams (track bi belongs to animated[bi]).
    for bi in range(min(N, len(animated))):
        keys = [(fr, quat_xyzw_to_blender(q))
                for fr, q in keys_by_idx.get(bi, []) if q is not None]
        if mult > 1 and len(keys) >= 2:
            keys = resample_rotation(keys, mult)
        if key_bone(animated[bi], keys):
            keyed.add(animated[bi])

    # Constant bones: a single key at frame 1 holds the RootRot pose.
    for nm, bq in const_bones:
        if key_bone(nm, [(0, bq)]):
            keyed.add(nm)

    # ── Procedural-helper emulation (twist + elbow/knee correctives) ────
    # Bones in NEITHER routing mask (forearm/upper-arm twists, the Elbow /
    # Knee corrective joints) are driven by the engine's procedural bone
    # system at runtime — the MAB simply has no data for them.  Left at
    # rest they cause the classic artefacts: all of the hand's roll lands
    # on the wrist joint (candy-wrapper collapse) and the knee/elbow
    # corrective stops pushing outward when the joint bends (skin
    # punctures inward).  Emulate with constraints:
    #   * childless un-animated 'twist' bone with an ANIMATED sibling
    #     (e.g. the Hand under the same Forearm): COPY_ROTATION (local,
    #     long-axis only) from that sibling, influence = the twist head's
    #     fractional position along parent->sibling (data-driven, 0 at
    #     the joint, 1 at the wrist).
    #   * childless un-animated Elbow/Knee bone: COPY_ROTATION (local,
    #     all axes) at 0.5 from the animated joint child (Forearm/Calf).
    # RE 2026-06-19 (FC4 *_ref.skeleton, see agents.md "Engine internals")
    # VALIDATED this exact topology: the procedural bones are tail-of-list
    # leaves with a zero bind transform, each driven by its real-joint
    # sibling — <side>ForeArmTwistA/B←Hand, <side>ArmTwistA/B←ForeArm,
    # <side>Elbow←ForeArm, <side>Knee←Leg, <side>HandThumbHelper←Thumb2.
    # Two twist bones per segment self-distribute via the fractional
    # influence (A nearer the elbow < B nearer the wrist).  Other rigs name
    # the twists <side>ForeArmRoll / <side>ArmRoll / *RollEx → match 'roll'
    # too; finger 'Helper' bones behave like a corrective.
    n_helpers = 0
    if emulate_helpers:
        from .procedural_fc3 import emulate_procedural_helpers
        n_helpers = emulate_procedural_helpers(
            arm_obj,
            [b['name'] for b in skel],
            [b['pos'] for b in skel],
            parents,
            keyed,
            log=print,
            bake=twist_bake, context=context,
            frame_start=1, frame_end=last_frame)

    scene = context.scene
    scene.frame_start = 1
    scene.frame_end = max(2, last_frame)
    if sections.animlen > 0:
        try:
            scene.render.fps = max(1, round(last_frame / sections.animlen))
        except Exception:
            pass
    bpy.ops.object.mode_set(mode='OBJECT')
    return len(keyed), animated


# ---------------------------------------------------------------------------
# Standalone validation entry point
# ---------------------------------------------------------------------------

def _euler_from_xyzw(q):
    if q is None:
        return None
    x, y, z, w = q
    pitch = math.asin(max(-1.0, min(1.0, -2.0 * (x * z - w * y))))
    yaw = math.atan2(2.0 * (y * z + w * x), w * w - x * x - y * y + z * z)
    roll = math.atan2(2.0 * (x * y + w * z), w * w + x * x - y * y - z * z)
    return (math.degrees(pitch), math.degrees(yaw), math.degrees(roll))


def _main_cli(argv):
    if len(argv) < 2:
        print("Usage: py -3 import_mab_buu.py <file.mab> [model.xbg]")
        return
    d = open(argv[1], 'rb').read()
    sec = parse_sections(d)
    print(f"[{sec.version_name}]  animlen={sec.animlen:.4f}s  size={len(d)}")
    for lbl in SECTION_LABELS:
        if sec.offsets[lbl]:
            print(f"  {lbl:10s} @0x{sec.offsets[lbl]:X} size={sec.sizes.get(lbl,0)}")

    single = False
    if len(argv) >= 3:
        bones = read_xbg_bones(argv[2])
        anim_bones = [b for b in bones if b.parent or b.index > 0]
        print(f"\nXBG bones ({len(bones)}):")
        for b in bones[:20]:
            print(f"  [{b.index:2d}] {b.name}  parent={b.parent!r}")
        single = (len(bones) <= 3)

    header, groups = decode_keyframes(d, sec, expect_single_bone=single)
    print(f"\nKeyframes: Nq={header['Nq']} fc={header['fc']} "
          f"plus4={header['plus4']}  {len(groups)} groups")
    for g in groups[:8]:
        print(f"  group[{g.index}] off=0x{g.start:X} size={g.size} "
              f"tuples={len(g.tuples)}")
        for ti, tp in enumerate(g.tuples[:4]):
            eu = _euler_from_xyzw(tp['q'])
            eus = "(%7.2f,%7.2f,%7.2f)" % eu if eu else "None"
            print(f"    tuple[{ti}] euler={eus} mask=0x{tp['mask']:04X} "
                  f"extra={len(tp['extra'])}")


if __name__ == '__main__':
    import sys
    _main_cli(sys.argv)
