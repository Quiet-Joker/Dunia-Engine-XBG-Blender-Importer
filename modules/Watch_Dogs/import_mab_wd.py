"""Watch Dogs 1 .mab (Disrupt 'aNi') — FULL animation support.

Wrapper layout (validated against male_uppb_mask-on_..._pda.mab and
DisruptEditor's mabFile.cpp):
    0x00  u32 magic 0x329B (WD1; 0x46B4 = WDL — bigger wrapper, see below)
    0x14  'aNi' + u8 flags
    0x18  u32 animationDataSize (file size - 16)
    0x1C  f32 duration (seconds)
    0x20  u16 numBonesInAnim
    0x22  u16 dataCounts[7]:
            [2] = constant-rotation bone count  (flag & 0x30 == 0x30)
            [4] = animated-rotation bone count  (flag & 0x10, not 0x30)
    0x30  u32 sectionOffsets[11]  (absolute = value + 0x10)
    0x5C  u16 lastFrame, u16 pad
    0x60  u32 boneNameCRC32 x numBones   (zlib.crc32 of the bone name)
    ....  u8 flag per bone:
            0x10 = participates in rotation data
            0x30 (both bits) = CONSTANT rotation (JointConstantRotations)
            low nibble + 1 = bit depth of this bone's compressed track

Sections (offset + 0x10) — decode ported from DisruptEditor's
MabPlayback/mabDecode.cpp (the bitstream is identical across WD1/FC5/WDL;
only the wrapper differs):
    [0] KeyTimes — u16 timeline-frame index per stored keyframe
    [4] JointConstantRotations — 6 bytes per constant bone, in bone order:
        three u16 words; w0/w1 top bits route which component is implicit,
        15-bit fields scale by 4.315969e-05 - 0.7071068
        (identity = FF BF FF BF FF 3F)
    [6] JointRotations — chunked bitstream:
        u32 table_size, then (table_size/4 - 1) u32 chunk END offsets
        (section-relative).  Chunk i covers stored keys [i*8, i*8+8).
        Per chunk, per animated bone (bone order), LSB-first bits:
            6  bits  cflags: b0-2 per-component CONSTANT flags,
                     b3 sign-bits present, b4-5 implicit component index
            nFrames bits  sign of the implicit component (if b3)
            per component c in 0..2:
              const:  u16 -> v = u16/32768 - 1            (all frames)
              else :  u16 -> base  = (u16 & 0xFF)/127 - 1
                             slope = (u16 >> 8)/127.5 * INTERP_SCALE[bits]
                      then nFrames x `bits`-bit raw: v = raw*slope + base
        Reconstruct smallest-three: stored components fill the slots
        skipping the implicit index; implicit = ±sqrt(1 - sum sq).
        Component order is (x, y, z, w).
"""

import math
import struct
import zlib

try:
    import bpy
    import mathutils
except ImportError:
    bpy = None
    mathutils = None


# interpolant scale per bit depth (DisruptEditor ChunkReader.cpp)
_INTERP_SCALE = [
    0, 0, 0.33333334, 0.14285715, 0.06666667, 0.032258064,
    0.015873017, 0.0078740157, 0.0039215689, 0.0019569471,
    0.00097751711, 0.00048851978, 0.00024420026, 0.00012208521,
    0.000061038882, 0.000030518509, 0.000015259022,
]


class _BitReader:
    """LSB-first bit reader (matches the engine's `num >> (bitPos & 7)`)."""

    __slots__ = ('d', 'n', 'bp')

    def __init__(self, data, start, end):
        self.d = data
        self.n = end
        self.bp = start * 8

    def read(self, nbits):
        bo = self.bp >> 3
        bi = self.bp & 7
        num = 0
        for k in range(4):
            if bo + k < self.n:
                num |= self.d[bo + k] << (8 * k)
        self.bp += nbits
        return (num >> bi) & ((1 << nbits) - 1)


def _unpack_const_quat(w0, w1, w2):
    """6-byte JointConstantRotations codec -> (x, y, z, w)."""
    SCALE, OFF = 4.315969e-05, 0.7071068
    f1 = (w0 & 0x7FFF) * SCALE - OFF
    f2 = (w1 & 0x7FFF) * SCALE - OFF
    f3 = w2 * SCALE - OFF
    s = 1.0 - f1 * f1 - f2 * f2 - f3 * f3
    f4 = s ** 0.5 if s > 0.0 else 0.0
    b0, b1 = w0 & 0x8000, w1 & 0x8000
    if not b0:
        if b1:
            return (f1, f2, f4, f3)
        return (f4, f1, f2, f3)
    if b1:
        return (f1, f2, f3, f4)
    return (f1, f4, f2, f3)


def _decode_bone_block(r, nbits, nframes):
    """One bone's chunk block -> [ (x,y,z,w) ] * nframes."""
    cflags = r.read(6)
    bsig = (cflags >> 3) & 1
    wind = (cflags >> 4) & 3
    sign_bits = r.read(nframes) if bsig else 0

    comp = [[0.0] * nframes for _ in range(3)]
    for c in range(3):
        if (cflags >> c) & 1:
            v = r.read(16) * (1.0 / 32768.0) - 1.0
            for f in range(nframes):
                comp[c][f] = v
        else:
            wv = r.read(16)
            base = (wv & 0xFF) * (1.0 / 127.0) - 1.0
            slope = (wv >> 8) * (1.0 / 127.5) * _INTERP_SCALE[nbits]
            for f in range(nframes):
                comp[c][f] = r.read(nbits) * slope + base

    out = []
    for f in range(nframes):
        q = [0.0, 0.0, 0.0, 0.0]
        s = 0.0
        for c in range(3):
            t = c if c < wind else c + 1
            q[t] = comp[c][f]
            s += q[t] * q[t]
        implicit = (1.0 - s) ** 0.5 if s < 1.0 else 0.0
        if (sign_bits >> f) & 1:
            implicit = -implicit
        q[wind] = implicit
        out.append(tuple(q))          # (x, y, z, w)
    return out


def parse_wd1_mab(path):
    d = open(path, 'rb').read()
    magic, = struct.unpack_from('<I', d, 0)
    if magic == 0x46B4:
        raise ValueError("WDL .mab (0x46B4) not supported yet — WD1 only")
    if magic != 0x329B:
        raise ValueError("not a WD1 .mab (magic 0x%X)" % magic)
    if d[0x14:0x17] != b'aNi':
        raise ValueError("bad WD1 .mab signature %r" % d[0x14:0x17])
    size, dur = struct.unpack_from('<If', d, 0x18)
    # The numBones field is a u16 with the TOP BIT (0x8000) used as a flag —
    # set on cinematic / scripted prop & object clips (e32, scriptedevents).
    # The real bone count is the low 15 bits; without masking it reads as a
    # ~32k garbage value and overruns the hash table.  Verified on all 757
    # affected files of the 12,535-clip WD1 corpus (quats decode unit-norm).
    n_bones = struct.unpack_from('<H', d, 0x20)[0] & 0x7FFF
    counts = struct.unpack_from('<7H', d, 0x22)
    offs = struct.unpack_from('<11I', d, 0x30)
    last_frame, = struct.unpack_from('<H', d, 0x5C)
    p = 0x60
    hashes = struct.unpack_from('<%dI' % n_bones, d, p) if n_bones else ()
    p += 4 * n_bones
    flags = d[p:p + n_bones]

    # KeyTimes (section 0): timeline frame per stored key
    kt_start = offs[0] + 0x10
    n_keys = max(0, (offs[1] - offs[0]) // 2)
    key_times = struct.unpack_from('<%dH' % n_keys, d, kt_start)

    # JointConstantRotations (section 4): flag & 0x30 == 0x30 bones
    const_rots = {}
    sec4 = offs[4] + 0x10
    ci = 0
    for bi in range(n_bones):
        if (flags[bi] & 0x30) != 0x30:
            continue
        if ci >= counts[2]:
            break
        w0, w1, w2 = struct.unpack_from('<3H', d, sec4 + ci * 6)
        const_rots[bi] = _unpack_const_quat(w0, w1, w2)
        ci += 1

    # JointRotations (section 6): chunked bitstream
    rot_curves = {}            # bone index -> [(timeline_frame, (x,y,z,w))]
    if counts[4]:
        jr = offs[6] + 0x10
        table_size, = struct.unpack_from('<I', d, jr)
        n_chunks = table_size // 4 - 1
        ends = struct.unpack_from('<%dI' % n_chunks, d, jr + 4)
        jr_bones = [(bi, (flags[bi] & 0x0F) + 1) for bi in range(n_bones)
                    if (flags[bi] & 0x10) and (flags[bi] & 0x30) != 0x30]
        prev = table_size
        for chunk_i in range(n_chunks):
            cstart, cend = jr + prev, jr + ends[chunk_i]
            r = _BitReader(d, cstart, cend)
            first = chunk_i * 8
            nframes = min(8, len(key_times) - first)
            if nframes <= 0:
                break
            for bi, nbits in jr_bones:
                quats = _decode_bone_block(r, nbits, nframes)
                cur = rot_curves.setdefault(bi, [])
                for f in range(nframes):
                    cur.append((key_times[first + f], quats[f]))
            prev = ends[chunk_i]

    # key_times are in fine timeline TICKS (verified 120/s on the WD1
    # sample: last key 340 over 2.8333 s).  'last_frame' is the index of
    # the last stored key, not a timeline length.
    tick_rate = (round(key_times[-1] / dur) if (dur > 0 and key_times
                                                and key_times[-1])
                 else 120)
    return {
        'duration': dur,
        'last_frame': last_frame,
        'tick_rate': tick_rate or 120,
        'n_bones': n_bones,
        'counts': counts,
        'hashes': list(hashes),
        'flags': bytes(flags),
        'key_times': list(key_times),
        'const_rots': const_rots,
        'rot_curves': rot_curves,
        'n_animated': len(rot_curves),
    }


def _bulk_key_pose(action, pb, frames, quats, locs=None):
    """Populate a pose bone's rotation_quaternion (4) + optional location (3)
    fcurves in one bulk foreach_set pass instead of ~2 keyframe_insert() calls
    per frame — the WD1 MAB import hotspot.  Mirrors the Avatar MAB importer.

    `frames` are 1-based ints; `quats`/`locs` are mathutils Quaternion/Vector.
    Assumes the bone has no existing keys in this action (true on import)."""
    from .procedural_wd import fcurve_container
    n = len(frames)
    if n == 0:
        return
    cont = fcurve_container(pb.id_data, action)
    bp = pb.path_from_id()
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


def apply_wd1_mab(context, mab, arm_obj, model_bones,
                  smooth_resample=True, resample_fps=60,
                  emulate_helpers=True, twist_bake=True):
    """Apply a decoded WD1 clip onto a WD armature.

    `model_bones` = the 'bones' list from parse_wd1_xbg (name/parent/pos/quat
    in parent-relative convention — the same data the armature was built
    from).  Uses the same convention-independent basis method as the Avatar
    MAB importer.  Returns (n_keyed, [unresolved hashes])."""
    if bpy is None:
        raise RuntimeError("bpy unavailable")
    Mat = mathutils.Matrix
    Quat = mathutils.Quaternion

    crc2idx = {zlib.crc32(b['name'].encode('latin-1')) & 0xFFFFFFFF: i
               for i, b in enumerate(model_bones)}
    parents = [b['parent'] for b in model_bones]
    rest_local = [Mat.Translation(b['pos']) @
                  Quat(b['quat']).to_matrix().to_4x4() for b in model_bones]
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
    action = bpy.data.actions.new(name="WD1_MAB")
    arm_obj.animation_data.action = action
    pbones = arm_obj.pose.bones

    keyed = 0
    keyed_names = set()          # model-bone names the MAB drives (for proc emu)
    missing = []
    ident = Mat.Identity(4)
    out_fps = 30
    tick_to_frame = out_fps / float(mab.get('tick_rate', 120))

    # SQUAD smoothing: bake dense in-between keys so sparse / low-fps clips
    # play smoothly (the engine evaluates spline-compressed rotation at the
    # game framerate).  Resampling scales the keyed frames by `mult`, so when
    # active we recompute the scene fps / range from the real max keyed frame.
    from .quat_resample_wd import resample_rotation
    mult = 1
    if smooth_resample and mab.get('duration', 0) > 0 and mab.get('last_frame', 0) > 0:
        src_fps = mab['last_frame'] / mab['duration']
        if src_fps > 0:
            mult = max(1, min(8, int(round(resample_fps / src_fps))))
    _maxf = [1]

    # ── Bind reference: ABSOLUTE vs DELTA (auto-detect per clip) ─────────
    # WD1 stores each bone's local rotation relative to its AUTHORING
    # skeleton's bind.  Two families ship in the game:
    #   * character-bind clips  — authored on the model's own rig; a static
    #     bone stores its full bind rotation  -> apply ABSOLUTE (L = T·q).
    #   * normalized-bind clips — authored on an identity-bind rig (combat /
    #     'adtv' sets); a static bone stores IDENTITY  -> apply as a DELTA
    #     onto the model's bind (L = rest_local·q), else the model's own
    #     bind (Pelvis 90°, etc.) is zeroed out and the spine breaks /
    #     character turns sideways.
    # Detect by voting on high-rest-angle bones (thumbs/clavicles/thighs):
    # is the first stored rotation nearer IDENTITY (delta) or the model's
    # REST (absolute)?  The two families vote unanimously in practice.
    def _ang(qa, qb):
        d = abs(qa.dot(qb))
        return 2.0 * math.acos(max(-1.0, min(1.0, d)))
    qident = Quat((1, 0, 0, 0))
    votes_delta = votes_abs = 0
    for mab_bi in list(mab['rot_curves']) + list(mab['const_rots']):
        xi = crc2idx.get(mab['hashes'][mab_bi])
        if xi is None:
            continue
        restq = Quat(model_bones[xi]['quat'])
        if _ang(restq, qident) < math.radians(40):
            continue                       # only high-rest bones discriminate
        src = mab['rot_curves'].get(mab_bi) or [(0, mab['const_rots'][mab_bi])]
        x, y, z, w = src[0][1]
        q0 = Quat((w, x, y, z))
        if _ang(q0, qident) < _ang(q0, restq):
            votes_delta += 1
        else:
            votes_abs += 1
    is_delta = votes_delta > votes_abs
    print("[WD1 MAB] bind mode: %s (delta votes %d / abs votes %d)"
          % ("DELTA/normalized" if is_delta else "ABSOLUTE/character",
             votes_delta, votes_abs))

    def pose_bone(mab_bi, keys):
        """keys = [(timeline_tick, (x,y,z,w)), ...]; returns True if keyed."""
        crc = mab['hashes'][mab_bi]
        xi = crc2idx.get(crc)
        if xi is None or model_bones[xi]['name'] not in pbones:
            missing.append(crc)
            return False
        nm = model_bones[xi]['name']
        keyed_names.add(nm)
        pb = pbones[nm]
        ml = pb.bone.matrix_local
        ml_inv = ml.inverted()
        p = parents[xi]
        rwp = rest_world[p] if (p is not None
                                and 0 <= p < len(model_bones)) else ident
        rwp_inv = rwp.inverted()
        rl = rest_local[xi]
        rl_inv = rl.inverted()
        pos = model_bones[xi]['pos']
        pb.rotation_mode = 'QUATERNION'
        if mult > 1 and len(keys) >= 2:
            wxyz = [(tk, (qq[3], qq[0], qq[1], qq[2])) for tk, qq in keys]
            keys = [(tk, (r[1], r[2], r[3], r[0]))
                    for tk, r in resample_rotation(wxyz, mult)]
        frames = []
        quats = []
        locs = []
        for tick, q in keys:
            x, y, z, w = q
            # ABSOLUTE clips: q is the full local rotation (replace bind).
            # DELTA clips: q is the motion relative to an identity bind, so
            # compose it onto the model's bind (rest_local @ q).  See the
            # auto-detect above.
            if is_delta:
                L = rl @ Quat((w, x, y, z)).to_matrix().to_4x4()
            else:
                L = Mat.Translation(pos) @ Quat((w, x, y, z)).to_matrix().to_4x4()
            M = ml_inv @ rwp @ L @ rl_inv @ rwp_inv @ ml
            frame = int(round(tick * tick_to_frame)) + 1
            frames.append(frame)
            quats.append(M.to_quaternion())     # == what matrix_basis would key
            locs.append(M.to_translation())
            if frame > _maxf[0]:
                _maxf[0] = frame
        # One foreach_set per channel instead of 2 keyframe_insert()/frame.
        _bulk_key_pose(action, pb, frames, quats, locs)
        return True

    for bi, q in mab['const_rots'].items():
        if pose_bone(bi, [(0, q)]):
            keyed += 1
    for bi, keys in mab['rot_curves'].items():
        if pose_bone(bi, keys):
            keyed += 1

    # Procedural twist / corrective bones (ForeArmTwistA/B, Elbow, Knee, …) are
    # NOT in the .mab — the engine drives them.  Emulate with constraints, using
    # the same data-driven logic validated against the FC4 *_ref.skeleton RE.
    if emulate_helpers:
        try:
            from .procedural_wd import emulate_procedural_helpers
            emulate_procedural_helpers(
                arm_obj,
                [b['name'] for b in model_bones],
                [b['pos'] for b in model_bones],
                parents,
                keyed_names,
                log=print,
                bake=twist_bake, context=context,
                frame_start=1, frame_end=_maxf[0])
        except Exception as e:
            print("[WD1 MAB] procedural-helper emulation skipped: %s" % e)

    scene = context.scene
    scene.frame_start = 1
    if mult > 1:
        # Resampling scaled the keyed frames by ~mult; derive range + fps from
        # the real max keyed frame so the clip plays over the same duration.
        scene.frame_end = max(2, _maxf[0])
        if mab['duration'] > 0:
            try:
                scene.render.fps = max(1, round(_maxf[0] / mab['duration']))
            except Exception:
                pass
    else:
        scene.frame_end = max(2, mab['last_frame'] + 1)
        if mab['duration'] > 0:
            try:
                scene.render.fps = max(1, round(mab['last_frame'] / mab['duration']))
            except Exception:
                pass
    bpy.ops.object.mode_set(mode='OBJECT')
    return keyed, missing
