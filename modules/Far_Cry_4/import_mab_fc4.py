"""Far Cry 4 .mab animation — Disrupt 'aNi' bitstream (WD1 family).

FC4 dropped FC2/FC3's buu342 section-table MAB for the Disrupt 'aNi'
format used by Watch Dogs 1.  The per-keyframe rotation bitstream is
IDENTICAL to WD1 (verified: the WD1 decoder consumes every FC4 chunk to
the exact bit), so this module only re-implements the FC4 wrapper and
reuses watchdogs/import_mab_wd's _BitReader + _decode_bone_block.

Wrapper (reverse-engineered 2026-06-17, sample
3rdge_fulbst_short_macheteslash..., verified on 4 files):
    0x00  u32 version            (0x81 = FC4)
    0x18  char[3] 'aNi' + u8 flag
    0x1C  u16 nbBones            (e.g. 58)
    0x20  f32 duration           (== last_key / 30; clips are 30 fps)
    0x28  u32 -> descriptors     (file offset = value + 0x18)
    0x2C  u32 -> bone hash table (file offset = value + 0x18)
    0x40  u16 sectionOffsets[10] (file offset of each = value + 0x18)
          NOTE base is +0x18 (WD1 used +0x10).  Stable section indices:
            [0] JointConstantRotations (6 bytes per constant bone)
            [2] KeyTimes               (u16 timeline frame per stored key)
            [4] JointRotations         (chunked bitstream, the big one)
            [5] JointTranslations
    ...   sections, then at the END:
          bone hash table (nbBones u32 CRC32, match the .skeleton)
          descriptors     (nbBones u8, WD1 bonePathFlags: 0x10=rotation,
                           0x20=constant, low nibble = quant bit depth)
"""

import struct
import zlib

try:
    import bpy
    import mathutils
except ImportError:
    bpy = None
    mathutils = None

from .mab_codec_fc4 import (_BitReader, _decode_bone_block,
                            _unpack_const_quat)


def parse_fc4_mab(path):
    d = open(path, 'rb').read()
    if d[0] != 0x81 or d[0x18:0x1B] != b'aNi':
        raise ValueError("not an FC4 .mab (version 0x%02X, sig %r)"
                         % (d[0], d[0x18:0x1B]))
    n_bones = struct.unpack_from('<H', d, 0x1C)[0]
    sec = [struct.unpack_from('<H', d, 0x40 + i * 2)[0] + 0x18
           for i in range(10)]
    hash_off = struct.unpack_from('<I', d, 0x2C)[0] + 0x18
    desc_off = struct.unpack_from('<I', d, 0x28)[0] + 0x18

    hashes = list(struct.unpack_from('<%dI' % n_bones, d, hash_off))
    flags = d[desc_off:desc_off + n_bones]

    # KeyTimes (section 2): one u16 timeline frame per stored keyframe.
    # The key count MUST come from the section boundary (the next section's
    # start), not a monotonic-value scan: a scan over-reads by a key, which
    # bumps the final partial rotation chunk's frame count and desyncs its
    # whole bitstream (verified: an off-by-one key turns the last chunk's
    # 3 frames into 4 and corrupts every bone in it).
    kt = sec[2]
    nexts = [s for s in sec if s > kt]
    kt_end = min(nexts) if nexts else len(d)
    n_keys = max(0, (kt_end - kt) // 2)
    key_times = list(struct.unpack_from('<%dH' % n_keys, d, kt))

    # JointConstantRotations (section 0): 6-byte Dunia quat per constant bone
    const_rots = {}
    c_off = sec[0]
    ci = 0
    for bi in range(n_bones):
        if (flags[bi] & 0x20):                       # constant-rotation bone
            w0, w1, w2 = struct.unpack_from('<3H', d, c_off + ci * 6)
            const_rots[bi] = _unpack_const_quat(w0, w1, w2)
            ci += 1

    # JointRotations (section 4): chunked bitstream (same as WD1)
    rot_curves = {}
    jr = sec[4]
    table_size = struct.unpack_from('<I', d, jr)[0]
    n_chunks = table_size // 4 - 1
    if n_chunks >= 1:
        ends = struct.unpack_from('<%dI' % n_chunks, d, jr + 4)
        jr_bones = [(bi, (flags[bi] & 0x0F) + 1) for bi in range(n_bones)
                    if (flags[bi] & 0x10) and not (flags[bi] & 0x20)]
        prev = table_size
        for chunk_i in range(n_chunks):
            cstart, cend = jr + prev, jr + ends[chunk_i]
            r = _BitReader(d, cstart, cend)
            first = chunk_i * 8
            nframes = min(8, n_keys - first)
            if nframes <= 0:
                break
            for bi, nbits in jr_bones:
                quats = _decode_bone_block(r, nbits, nframes)
                cur = rot_curves.setdefault(bi, [])
                for f in range(nframes):
                    cur.append((key_times[first + f], quats[f]))
            prev = ends[chunk_i]

    last = key_times[-1] if key_times else 0
    # Duration is a f32 at 0x20 (NOT 0x04 — that field is garbage on most
    # clips).  key_times are 30 fps timeline frame indices: on every sample
    # duration == last_key / 30, so the tick rate is 30.  Derive it from the
    # duration defensively, falling back to 30 if the field is bad.
    duration = struct.unpack_from('<f', d, 0x20)[0]
    tick_rate = round(last / duration) if (0.0 < duration < 1e6 and last) else 30
    return {
        'n_bones': n_bones,
        'duration': duration if 0.0 < duration < 1e6 else (last / 30.0),
        'hashes': hashes,
        'flags': bytes(flags),
        'key_times': key_times,
        'last_frame': last,
        'tick_rate': tick_rate or 30,
        'const_rots': const_rots,
        'rot_curves': rot_curves,
        'n_animated': len(rot_curves),
    }


def fc3_bones_to_model_bones(bones):
    """Convert FC3/FC4 parse_xbg bones to the (name/parent/pos/quat) form the
    MAB applier expects.

    Mirrors _build_fc3_armature in modules/Avatar/import_xbg.py exactly so the
    rest pose matches the armature that was built: rotation_raw is (w, x, y, z)
    and normalized; translation is the parent-relative position.  The MAB
    applier rebuilds each bone's rest world from this, so the convention MUST
    match the armature or the bind-mode detection and basis go wrong.
    """
    out = []
    for b in bones:
        raw = b['rotation_raw']
        n = (raw[0] ** 2 + raw[1] ** 2 + raw[2] ** 2 + raw[3] ** 2) ** 0.5 or 1.0
        out.append({
            'name': b['name'],
            'parent': b['parent'],
            'pos': tuple(b['translation']),
            'quat': (raw[0] / n, raw[1] / n, raw[2] / n, raw[3] / n),  # (w,x,y,z)
        })
    return out


def apply_fc4_mab(context, mab, arm_obj, model_bones,
                  smooth_resample=True, resample_fps=60,
                  emulate_helpers=True, twist_bake=True):
    """Apply a decoded FC4 clip onto a Blender armature, reusing the WD1
    applier (bind-mode auto-detect + convention-independent basis).
    `model_bones` = fc3_bones_to_model_bones(parse_xbg(...)['bones']).
    Returns (n_keyed, [unresolved hashes])."""
    from .mab_codec_fc4 import apply_wd1_mab
    return apply_wd1_mab(context, mab, arm_obj, model_bones,
                         smooth_resample=smooth_resample,
                         resample_fps=resample_fps,
                         emulate_helpers=emulate_helpers,
                         twist_bake=twist_bake)
