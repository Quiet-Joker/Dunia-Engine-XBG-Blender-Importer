"""Far Cry 5 .mab skeletal-animation parser — FULLY DECODED (2026-06-25).

FC5 MAB uses the Dunia 'aNi' rotation codec (smallest-three quaternions, per-frame
interpolant deltas).  The container was reverse-engineered from the FC5 engine
(FC_m64.dll, validated via Unicorn emulation + the WD-Legion DuniaDemo decompile)
and verified purely by FC5 self-consistency: every chunk's bitstream consumes
EXACTLY to its byte boundary and the assembled curves are smooth (~0.6 deg/frame).
See agents.md "FC5 MAB ROTATION DECODE — SOLVED" for the full recipe.

Layout (ad = (file offset of b'aNi') - 0x10  ==  CAnimData base, sizeof 0x68):
    ad+0x20  u16     nbBones
    ad+0x22  u16[10] dataCounts        (dataCounts[4] = animated-rotation count)
    ad+0x38  u32[12] m_offsets
    ad+0x68          bone CRC32 hashes (nbBones u32), then per-bone FLAGS (nbBones u8)
    after flags:     KeyTimes (u16 run)
    m_offsets[5]  =  JointRotations bitstream (chunk-offset table then bitstreams)
    m_offsets[4]  =  packed-48bit const-rotation quats (n_const of them)
Per-bone flag byte: 0x10 = rotation-animated, 0x20 = const, (flag&0xF)+1 = nbits.
"""

import struct
import zlib

# ms_interpolantScaleFactors — verified against FC_m64.dll via Unicorn emulation.
_INTERP = [0.0, 0.0, 0.33333334, 0.14285715, 0.06666667, 0.032258064,
           0.015873017, 0.0078740157, 0.0039215689, 0.0019569471, 0.00097751711,
           0.00048851978, 0.00024420026, 0.00012208521, 0.000061038882,
           0.000030518509, 0.000015259022]

_FRAMES_PER_CHUNK = 8        # m_numFramesPerChunk (engine constant)
_CONST_SCALE = 0.000030518044    # 1/32767.5  (const-component decode)
_BASE_SCALE = 0.0078740157       # 1/127      (varying base byte)
_SLOPE_SCALE = 0.0078431377      # 1/127.5    (varying slope byte)


class _Bits:
    """LSB-first little-endian bit reader (engine: dword>>(bp&7))."""
    __slots__ = ('d', 'bp')

    def __init__(self, data, bit_pos):
        self.d = data
        self.bp = bit_pos

    def read(self, n):
        bo = self.bp >> 3
        bi = self.bp & 7
        num = 0
        d = self.d
        for k in range(5):
            if bo + k < len(d):
                num |= d[bo + k] << (8 * k)
        self.bp += n
        return (num >> bi) & ((1 << n) - 1)


def _decode_quat_block(r, nframes, nbits):
    """Decode one bone's quaternion block for `nframes` keyframes.

    Returns [ (x, y, z, w) ] * nframes.  Exact port of the FC5 engine's
    StartData<CQuaternion> + ExtractAnyFramePair<CQuaternion> + ReadFrameData.
    """
    cflags = r.read(6)
    sign_present = (cflags >> 3) & 1
    winding = (cflags >> 4) & 3
    sign_bits = r.read(nframes) if sign_present else 0

    comp = [[0.0] * nframes for _ in range(3)]
    for c in range(3):
        if (cflags >> c) & 1:                       # constant component
            v = r.read(16) * _CONST_SCALE - 1.0
            for f in range(nframes):
                comp[c][f] = v
        else:                                       # varying component
            wv = r.read(16)
            base = (wv & 0xFF) * _BASE_SCALE - 1.0
            slope = (wv >> 8) * _SLOPE_SCALE * _INTERP[nbits]
            for f in range(nframes):
                comp[c][f] = r.read(nbits) * slope + base

    out = []
    for f in range(nframes):
        q = [0.0, 0.0, 0.0, 0.0]
        ssum = 0.0
        for c in range(3):
            slot = c if c < winding else c + 1
            q[slot] = comp[c][f]
            ssum += q[slot] * q[slot]
        if ssum <= 1.0:
            implicit = (1.0 - ssum) ** 0.5
        else:                                       # clamp/normalise overflow
            inv = 1.0 / (ssum ** 0.5)
            for c in range(3):
                slot = c if c < winding else c + 1
                q[slot] *= inv
            implicit = 0.0
        if (sign_bits >> f) & 1:
            implicit = -implicit
        q[winding] = implicit
        out.append((q[0], q[1], q[2], q[3]))
    return out


def _unpack_const_quat(w0, w1, w2):
    """6-byte (3*u16) JointConstantRotations -> (x, y, z, w)."""
    SCALE, OFF = 4.315969e-05, 0.7071068
    f1 = (w0 & 0x7FFF) * SCALE - OFF
    f2 = (w1 & 0x7FFF) * SCALE - OFF
    f3 = w2 * SCALE - OFF
    s = 1.0 - f1 * f1 - f2 * f2 - f3 * f3
    f4 = s ** 0.5 if s > 0.0 else 0.0
    b0, b1 = w0 & 0x8000, w1 & 0x8000
    if not b0:
        return (f1, f2, f4, f3) if b1 else (f4, f1, f2, f3)
    return (f1, f2, f3, f4) if b1 else (f1, f4, f2, f3)


def _read_keytimes(d, off, end, max_frame):
    """KeyTimes are a monotonic u16 run; try LE and BE alignments, take the
    longest non-decreasing run bounded by `max_frame`."""
    best = []
    for be in (False, True):
        for start in (off, off + 1):
            run = []
            p = start
            while p + 1 < end:
                v = (d[p] << 8 | d[p + 1]) if be else (d[p] | d[p + 1] << 8)
                if run and (v < run[-1] or v > max_frame):
                    break
                run.append(v)
                p += 2
            if len(run) > len(best):
                best = run
    return best


# 0x07 = Far Cry New Dawn player clips (same chunked codec; only the m_flags
# version byte differs from FC5's 0x2f / 0x0f / 0x23).
# 0x03 = seen on FC5 melee/takedown player-skeleton blocks (e.g.
# "*_takedownfromabove*_hhmac.mab") — without it here, _parse_anim_block()
# rejected the real 60+ bone player block outright and the importer fell back
# to one of the single-bone melee_anchor/weapon prop blocks as "main",
# producing a near-static 1-bone, 2-keyframe "animation".
_KNOWN_MAB_VERSIONS = (0x03, 0x07, 0x0f, 0x23, 0x2f)


def _find_all(d, sub):
    out = []
    i = d.find(sub)
    while i >= 0:
        out.append(i)
        i = d.find(sub, i + 1)
    return out


def parse_fc5_mab(path):
    """Parse a Far Cry 5 .mab.  Returns the MAIN (player-skeleton) animation
    block; any SECONDARY blocks (prop / anchor / root-motion rigs stored after
    the main one — e.g. the heal bandage's 7-bone ANCHOR rig, the bees' Root
    tracks) are attached as `['props']` (and the full set as `['blocks']`)."""
    d = open(path, 'rb').read()
    blocks = []
    for p in _find_all(d, b'aNi'):
        blk = _parse_anim_block(d, p)
        if blk is not None:
            blocks.append(blk)
    if not blocks:
        raise ValueError("not an 'aNi' animation file")
    # The player skeleton is the richest block (most bones); coincidental 'aNi'
    # bytes inside the bitstream are rejected by _parse_anim_block's validation.
    main = max(blocks, key=lambda b: b['n_bones'])
    main['blocks'] = blocks
    main['props'] = [b for b in blocks if b is not main]
    _merge_root_motion(main)
    return main


_ROOT_CRC = zlib.crc32(b'Root') & 0xFFFFFFFF


def _merge_root_motion(main):
    """DO NOT fold single-bone `Root` blocks onto the player rig.

    HISTORICAL BUG (fixed 2026-06-30): a .mab packs the player skeleton PLUS the
    skeletons of every other entity the clip touches (props, weapons, and — for
    the bee clips — each bee).  Those extra entities use
    `singlebone_ref.skeleton`, whose lone bone is named **`Root`** — the SAME
    name (and CRC `0xb6c65665`) as the player's own root.  The old code matched
    by that hash and applied a *bee's* flight path (animated translation +
    rotation) onto the *player's* Root bone, so the first-person view spun and
    flew around the scene.

    Per the Dunia animation dev: the first-person character root is purely
    CODE-DRIVEN — it is never animated by the file.  Character displacement lives
    on the Hips (and a Props bone) INSIDE the main multi-bone block, not in a
    separate single-bone block.  So these single-`Root` blocks are always OTHER
    entities and must be built as their own objects (see `build_fc5_prop_rigs`),
    never merged onto the player.

    We leave the player's Root exactly as the main block authored it (static
    const) and tag the animated single-bone blocks so the prop builder picks
    them up."""
    main['root_motion'] = False                 # never drive the player root
    for b in main.get('props', []):
        b['is_single_entity'] = (b['n_bones'] == 1)


def _parse_anim_block(d, base):
    """Parse ONE CAnimData block whose 'aNi' signature is at byte `base`.
    Returns the block dict, or None if `base` isn't a valid block header
    (guards against the 'aNi' byte sequence appearing inside compressed data)."""
    version = d[base + 3]
    if version not in _KNOWN_MAB_VERSIONS:
        return None
    ad = base - 0x10                                # CAnimData base (sizeof 0x68)
    if ad < 0 or ad + 0x68 > len(d):
        return None
    nbBones = struct.unpack_from('<H', d, ad + 0x20)[0]
    if not (0 < nbBones < 1024):
        return None
    flags_off = ad + 0x68 + nbBones * 4
    if flags_off + nbBones > len(d):
        return None
    try:
        return _parse_anim_block_body(d, base, version, ad, nbBones, flags_off)
    except (struct.error, IndexError, ValueError):
        return None


def _parse_anim_block_body(d, base, version, ad, nbBones, flags_off):
    duration = struct.unpack_from('<f', d, ad + 0x18)[0]
    fps = struct.unpack_from('<f', d, ad + 0x1C)[0]
    data_counts = struct.unpack_from('<10H', d, ad + 0x22)
    offsets = struct.unpack_from('<12I', d, ad + 0x38)

    hashes = list(struct.unpack_from('<%dI' % nbBones, d, ad + 0x68))
    flags = list(d[flags_off:flags_off + nbBones])

    if not (0.0 < duration < 1e6) or fps <= 0:
        duration, fps = 3.0, 30.0
    max_frame = duration * fps * 1.25 + 4

    raw_keytimes = _read_keytimes(d, flags_off + nbBones,
                                  ad + offsets[0] if offsets[0] else len(d),
                                  max_frame)

    # ── Constant-rotation bones (flag & 0x30 == 0x30): packed quats @ offsets[3]
    const_rots = {}
    cq = ad + offsets[3] if offsets[3] else 0
    ci = 0
    for bi in range(nbBones):
        if (flags[bi] & 0x30) == 0x30:
            if cq:
                w0, w1, w2 = struct.unpack_from('<3H', d, cq + ci * 6)
                const_rots[bi] = _unpack_const_quat(w0, w1, w2)
            ci += 1

    # ── Animated-rotation bones (0x10 & !0x20), in bone-index order ──────────
    anim_bones = [bi for bi in range(nbBones)
                  if (flags[bi] & 0x10) and not (flags[bi] & 0x20)]
    bone_nbits = [(flags[bi] & 0xF) + 1 for bi in anim_bones]

    def _chunk_end(start_bit, nframes):
        """Decode all anim bones (consuming only) -> end byte (relative to rot)."""
        r = _Bits(d, start_bit)
        for nb in bone_nbits:
            _decode_quat_block(r, nframes, nb)
        return r.bp // 8

    rot_curves = {bi: [] for bi in anim_bones}
    chunk_frame_counts = []
    n_keys = len(raw_keytimes)
    if offsets[5] and anim_bones:
        rot = ad + offsets[5]
        table_size = struct.unpack_from('<I', d, rot)[0]
        n_chunks = max(1, table_size // 4 - 1)
        chunk_off = list(struct.unpack_from('<%dI' % (n_chunks + 1), d, rot))

        # The last chunk is partial; find its true frame count by which NF makes
        # the decode land on the chunk's byte boundary (robust across 0x23/0x2f,
        # where the keytime count can be off by a spurious leading 0).
        last_nf = _FRAMES_PER_CHUNK
        if n_chunks >= 1:
            target = rot + chunk_off[n_chunks]
            start = (rot + chunk_off[n_chunks - 1]) * 8
            best = None
            for nf in range(1, _FRAMES_PER_CHUNK + 1):
                diff = abs(_chunk_end(start, nf) - target)
                if best is None or diff < best[0]:
                    best = (diff, nf)
                if diff <= 2:
                    break
            last_nf = best[1]
        n_keys = (n_chunks - 1) * _FRAMES_PER_CHUNK + last_nf

        # Align keytimes to n_keys (drop a spurious leading entry if present).
        if len(raw_keytimes) > n_keys:
            key_times = raw_keytimes[len(raw_keytimes) - n_keys:]
        elif len(raw_keytimes) < n_keys:
            key_times = raw_keytimes + list(range(
                raw_keytimes[-1] + 1 if raw_keytimes else 0,
                raw_keytimes[-1] + 1 + n_keys - len(raw_keytimes) if raw_keytimes
                else n_keys))
        else:
            key_times = list(raw_keytimes)

        for ch in range(n_chunks):
            base_key = ch * _FRAMES_PER_CHUNK
            nframes = (last_nf if ch == n_chunks - 1
                       else min(_FRAMES_PER_CHUNK, n_keys - base_key))
            if nframes <= 0:
                break
            chunk_frame_counts.append(nframes)
            r = _Bits(d, (rot + chunk_off[ch]) * 8)
            for idx, bi in enumerate(anim_bones):
                quats = _decode_quat_block(r, nframes, bone_nbits[idx])
                cur = rot_curves[bi]
                for f in range(nframes):
                    cur.append((key_times[base_key + f], quats[f]))
    else:
        key_times = list(raw_keytimes)
    last = key_times[-1] if key_times else 0

    # Root-motion TRANSLATION (version 0x0f single-Root blocks store the player
    # tumble/slide as a CVector3 track; the 0x2f main/prop blocks don't here).
    loc_curves = {}
    if version == 0x0f and nbBones == 1 and chunk_frame_counts:
        lc = _decode_translation_0f(d, ad, offsets, key_times,
                                    chunk_frame_counts)
        if lc:
            loc_curves[0] = lc

    tick_rate = round(last / duration) if (0.0 < duration < 1e6 and last) else 30
    return {
        'version': version,
        'block_offset': base,
        'n_bones': nbBones,
        'duration': duration,
        'fps': fps,
        'hashes': hashes,
        'flags': bytes(flags),
        'key_times': key_times,
        'last_frame': last,
        'tick_rate': tick_rate or 30,
        'n_const': len(const_rots),
        'n_animated': len(anim_bones),
        'const_rots': const_rots,
        'rot_curves': rot_curves,
        'loc_curves': loc_curves,
    }


_TRANS_SCALE_UNIT = 0.00029802325        # CBoneTranslationCompressionEntry u24


def _decode_vec_block(r, nframes, nbits):
    """Decode one bone's TRANSLATION block (CVector3) for `nframes` frames.
    3-bit constFlags (one per x/y/z), each component const (read16) or varying
    (read16 base/slope + per-frame nbits delta) — the CVector3 sibling of
    _decode_quat_block, minus the quaternion reconstruction.  Returns normalized
    [(x, y, z)] (caller multiplies by the per-bone scale)."""
    cflags = r.read(3)
    comp = [[0.0] * nframes for _ in range(3)]
    for c in range(3):
        if (cflags >> c) & 1:
            v = r.read(16) * _CONST_SCALE - 1.0
            for f in range(nframes):
                comp[c][f] = v
        else:
            wv = r.read(16)
            base = (wv & 0xFF) * _BASE_SCALE - 1.0
            slope = (wv >> 8) * _SLOPE_SCALE * _INTERP[nbits]
            for f in range(nframes):
                comp[c][f] = r.read(nbits) * slope + base
    return [(comp[0][f], comp[1][f], comp[2][f]) for f in range(nframes)]


def _decode_translation_0f(d, ad, offsets, key_times, chunk_frame_counts):
    """Decode the root-motion TRANSLATION of a version-0x0f single-bone block.

    Layout (RE'd from CAnimFrameChunkStreamReader<CVector3> + the bees root
    block): per-bone compression entry at m_offsets[0] (u8 interpolantBits +
    u24 scale*0.00029802325); chunked CVector3 bitstream at m_offsets[6] (same
    chunk-offset table convention as rotation).  Returns [(keytime,(x,y,z))]."""
    tbl = ad + offsets[0]
    entry = struct.unpack_from('<I', d, tbl)[0]
    nbits = entry & 0xFF
    scale = (entry >> 8) * _TRANS_SCALE_UNIT
    if not (1 <= nbits <= 16) or offsets[6] == 0:
        return []
    trn = ad + offsets[6]
    tsz = struct.unpack_from('<I', d, trn)[0]
    n_chunks = max(1, tsz // 4 - 1)
    choff = list(struct.unpack_from('<%dI' % (n_chunks + 1), d, trn))
    curve = []
    k = 0
    for ch in range(min(n_chunks, len(chunk_frame_counts))):
        nf = chunk_frame_counts[ch]
        if nf <= 0:
            break
        r = _Bits(d, (trn + choff[ch]) * 8)
        vecs = _decode_vec_block(r, nf, nbits)
        for f in range(nf):
            if k < len(key_times):
                curve.append((key_times[k], (vecs[f][0] * scale,
                                             vecs[f][1] * scale,
                                             vecs[f][2] * scale)))
            k += 1
    return curve


def fc5_bones_to_model_bones(bones):
    """Convert FC5 parse_xbg EDON bones -> the (name/parent/pos/quat) form the
    MAB applier expects.  Mirrors _build_fc3_armature: rotation_raw is (w,x,y,z),
    translation is parent-relative."""
    out = []
    for b in bones:
        raw = b['rotation_raw']
        n = (raw[0] ** 2 + raw[1] ** 2 + raw[2] ** 2 + raw[3] ** 2) ** 0.5 or 1.0
        out.append({
            'name': b['name'],
            'parent': b['parent'],
            'pos': tuple(b['translation']),
            'quat': (raw[0] / n, raw[1] / n, raw[2] / n, raw[3] / n),
        })
    return out


def build_fc5_prop_rigs(context, mab, name_prefix, parent_arm=None,
                        known_names=None, log=None):
    """Build a small armature for each SECONDARY animation block — every entity
    the .mab packs alongside the player.  Two kinds:

      * multi-bone PROP rigs (heal bandage's 7-bone ANCHOR rig, a placed mine's
        11-bone rig) — rotation-animated.
      * single-bone ENTITY blocks (e.g. each BEE) — these use
        `singlebone_ref.skeleton` (one bone named `Root`) and carry the entity's
        flight path: an animated CVector3 TRANSLATION (m_offsets[6]) plus
        rotation.  These were previously (wrongly) folded onto the player's Root;
        they are SEPARATE objects and get built here with their location track.

    We only build single-bone blocks that are actually ANIMATED (loc or rot
    curves); static 2-key single-bone blocks are bare placement anchors and are
    skipped to avoid clutter.

    LIMITATION: the .mab stores only animation TRACKS, not bind poses (bone
    positions + hierarchy live in each entity's own model file, which we don't
    have here).  So bones use a placeholder layout; only the animated
    rotation/translation is faithful.  Returns the created armature objects."""
    import bpy
    import mathutils
    Quat = mathutils.Quaternion
    Vec = mathutils.Vector
    known_names = known_names or {}
    out_fps = 30
    created = []
    for blk in mab.get('props', []):
        single_anim = (blk['n_bones'] == 1 and
                       (blk.get('loc_curves', {}).get(0) or
                        blk['rot_curves'].get(0)))
        if blk['n_bones'] < 2 and not single_anim:
            continue
        names, seen = [], {}
        for h in blk['hashes']:
            nm = known_names.get(h) or ('bone_%08x' % h)
            if nm in seen:                              # Blender bone names unique
                seen[nm] += 1
                nm = "%s.%03d" % (nm, seen[nm])
            else:
                seen[nm] = 0
            names.append(nm)

        kind = "entity" if blk['n_bones'] == 1 else "prop"
        aname = "%s_%s_%x" % (name_prefix, kind, blk['block_offset'])
        adata = bpy.data.armatures.new(aname)
        aobj = bpy.data.objects.new(aname, adata)
        context.collection.objects.link(aobj)
        context.view_layer.objects.active = aobj
        bpy.ops.object.mode_set(mode='EDIT')
        ebs = []
        for i, nm in enumerate(names):
            eb = adata.edit_bones.new(nm)
            z = i * 0.08
            eb.head = (0.0, 0.0, z)
            eb.tail = (0.0, 0.0, z + 0.06)
            if i > 0:
                eb.parent = ebs[0]                      # ANCHOR-rooted placeholder
            ebs.append(eb)
        bpy.ops.object.mode_set(mode='POSE')
        if aobj.animation_data is None:
            aobj.animation_data_create()
        act = bpy.data.actions.new(aname)
        aobj.animation_data.action = act
        pbs = aobj.pose.bones
        tick_to_frame = out_fps / float(blk.get('tick_rate', 30) or 30)
        for i, nm in enumerate(names):
            pb = pbs[nm]
            pb.rotation_mode = 'QUATERNION'
            cur = blk['rot_curves'].get(i)
            if cur:
                for tick, (x, y, z, w) in cur:
                    pb.rotation_quaternion = Quat((w, x, y, z))
                    fr = int(round(tick * tick_to_frame)) + 1
                    pb.keyframe_insert('rotation_quaternion', frame=fr)
            elif i in blk['const_rots']:
                x, y, z, w = blk['const_rots'][i]
                pb.rotation_quaternion = Quat((w, x, y, z))
            # Single-bone entities (bees) carry a TRANSLATION flight path.
            lcur = blk.get('loc_curves', {}).get(i)
            if lcur:
                for tick, (x, y, z) in lcur:
                    pb.location = Vec((x, y, z))
                    fr = int(round(tick * tick_to_frame)) + 1
                    pb.keyframe_insert('location', frame=fr)
        bpy.ops.object.mode_set(mode='OBJECT')
        if parent_arm is not None:
            aobj.parent = parent_arm
        aobj['xbg_fc5_prop'] = True
        created.append(aobj)
        if log:
            log("[MAB] prop rig '%s': %d bones (%d animated) — placeholder layout"
                % (aname, blk['n_bones'], blk['n_animated']))
    return created


def apply_fc5_root_location(context, arm_obj, mab, model_bones,
                            smooth_resample=True, resample_fps=60):
    """Key the Root bone's LOCATION from the decoded 0x0f root-motion translation
    (rotation is applied by the main keyer via rot_curves).  Must be called AFTER
    apply_fc5_mab.  Returns the number of location keys."""
    import bpy
    import mathutils
    loc = mab.get('root_loc')
    if not loc:
        return 0
    root_bi = mab.get('root_bone_index')
    crc2name = {zlib.crc32(b['name'].encode('latin-1')) & 0xFFFFFFFF: b['name']
                for b in model_bones}
    name = crc2name.get(mab['hashes'][root_bi]) if root_bi is not None else None
    if not name or name not in arm_obj.pose.bones:
        return 0
    pb = arm_obj.pose.bones[name]
    # Translation is in the root's rest (parent) space; convert to the bone-local
    # basis frame Blender keys (location is relative to the bone's rest matrix).
    rinv = pb.bone.matrix_local.to_3x3().inverted()

    # Match the rotation keyer's frame scaling (SQUAD resample scales frames by
    # `mult`) so location and rotation stay aligned.
    mult = 1
    if smooth_resample and mab.get('duration', 0) > 0 and mab.get('last_frame', 0):
        src_fps = mab['last_frame'] / mab['duration']
        if src_fps > 0:
            mult = max(1, min(8, int(round(resample_fps / src_fps))))
    out_fps = 30
    tick_to_frame = out_fps / float(mab.get('tick_rate', 120) or 120)

    if context.view_layer.objects.active is not arm_obj:
        context.view_layer.objects.active = arm_obj
    if arm_obj.mode != 'POSE':
        bpy.ops.object.mode_set(mode='POSE')

    # CRITICAL: the rotation keyer SQUAD-resamples the Root to `mult`x density and
    # writes a location key (its basis residual, =0 for the root) at EVERY one of
    # those frames.  If we only key location at the source ticks (every `mult`th
    # frame) the rotation keyer's zero-residual keys survive on the in-between
    # frames — so the Root snaps back to the origin on every gap frame (violent
    # per-frame flicker / "jumps around").  Densely key location at EVERY frame in
    # the resampled range, linearly interpolating the decoded root-motion curve, so
    # it fully owns the channel and stays aligned with the resampled rotation.
    pts = [(tick * mult * tick_to_frame, mathutils.Vector((x, y, z)))
           for tick, (x, y, z) in loc]
    pts.sort(key=lambda p: p[0])
    f_first = int(round(pts[0][0]))
    f_last = int(round(pts[-1][0]))
    n = 0
    j = 0
    for f in range(f_first, f_last + 1):
        while j + 1 < len(pts) and pts[j + 1][0] <= f:
            j += 1
        if j + 1 < len(pts) and pts[j + 1][0] > pts[j][0]:
            t = (f - pts[j][0]) / (pts[j + 1][0] - pts[j][0])
            t = max(0.0, min(1.0, t))
            vec = pts[j][1].lerp(pts[j + 1][1], t)
        else:
            vec = pts[j][1]
        pb.location = rinv @ vec
        pb.keyframe_insert('location', frame=f + 1)
        n += 1

    # Force LINEAR interpolation so the densely-keyed motion plays as straight
    # segments between real samples (matches how the engine plays the compressed
    # track) instead of Bézier tangents overshooting at the sharp tumble keys.
    try:
        from .procedural_fc5 import fcurve_container
        cont = fcurve_container(arm_obj, arm_obj.animation_data.action)
        bp = pb.path_from_id() + '.location'
        for fc in cont.fcurves:
            if fc.data_path == bp:
                for kp in fc.keyframe_points:
                    kp.interpolation = 'LINEAR'
                fc.update()
    except Exception as e:
        print("[FC5 MAB] root-loc linear interp skip: %s" % e)
    return n


def apply_fc5_mab(context, mab, arm_obj, model_bones,
                  smooth_resample=True, resample_fps=60,
                  emulate_helpers=False, twist_bake=False):
    """Apply a decoded FC5 clip onto a Blender armature (reuses the shared
    Disrupt-'aNi' applier: bind-mode auto-detect + convention-independent basis).
    `model_bones` = fc5_bones_to_model_bones(parse_xbg(...)['bones'])."""
    from .mab_codec_fc5 import apply_wd1_mab
    return apply_wd1_mab(context, mab, arm_obj, model_bones,
                         smooth_resample=smooth_resample,
                         resample_fps=resample_fps,
                         emulate_helpers=emulate_helpers,
                         twist_bake=twist_bake)
