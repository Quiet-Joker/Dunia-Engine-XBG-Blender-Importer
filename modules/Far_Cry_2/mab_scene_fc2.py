"""MAB scripted-scene data: scene elements, anchors, cameras, timed events.

Reverse-engineered 2026-06-11 from the shipped scripted_event MABs, with the
class names confirmed in Dunia_Retail_1.02_decrypted.dll (CAnimTechAnchor,
CAnimFacialPoseEvent, AttachAnchor, m_anchorPartName ...).

Three MAB sections carry the scene data:

'Events' section — SCENE-ELEMENT TABLE (anchors / cameras):
    u32 count, then `count` records of 172 bytes:
        +0   u16  kind          (7 = anchor, 9 = animated camera, ...)
        +2   6 bytes 0xFF       (unresolved runtime handles)
        +8   u32  id1, u32 id2  (per-file ids)
        +16  u32  zero
        +20  f32  duration      (== clip length)
        +24  u32 crc32 + char[32]  element name      ('centerscene', ...)
        +60  u32 crc32 + char[32]  parent element    (attach anchor)
        +132 u32 crc32 + char[32]  reference element
    The crc32 is zlib.crc32 of the name (verified: crc32('centerscene')
    == the stored a9fb9cf2).

'UnkSec4' section — TIMED EVENT LIST (sound cues, FX, camera control):
    sequence of records:
        f32 time_seconds, u32 2, u16 fcb_size, u16 record_size,
        FCB blob (fcb_size bytes), u8 terminator 0
    where record_size == fcb_size + 13 (the 12-byte header + terminator).
    The FCB blob is the standard Dunia 'nbCF' binary-object format:
        'nbCF', u16 version (2), u16 flags, u32 objCount, u32 valueCount,
        then one object: {childCount packedU8/FF+u32, u32 nameHash,
        valueCount packed, values: {u32 nameHash, size packed, bytes},
        children...}
    Value 0x60534b32 is the event's TYPE NAME string (PlaySound,
    PlayDialog, SetFOV, CameraEvent, AttachEvent, SpawnParticleEvent,
    FacialEmotion, GearBounceEvent, ...).

'UnkSec5' section — PER-ELEMENT ANIMATION STREAMS:
    one block per scene element (same order as the element table):
        0x70-byte header (initial transform quats at +0x50/+0x60)
        'AnD\\x1a' magic, f32 duration, u64 0, offset table, packed
        position / rotation tracks (6-byte Dunia quats).
    Track decode is still partial — parse_and_blocks exposes the raw
    blocks and the initial orientation quats.
"""

import os
import re
import struct
import zlib

try:
    import bpy
    import mathutils
except ImportError:          # standalone analysis
    bpy = None
    mathutils = None

from .import_mab_fc2 import parse_sections

# FCB value hash of the event type-name string (constant across all files)
_FCB_TYPENAME_HASH = 0x60534b32


# ---------------------------------------------------------------------------
# FCB ('nbCF') binary-object reader
# ---------------------------------------------------------------------------

def _read_count(d, p):
    v = d[p]; p += 1
    if v == 0xFF:
        v, = struct.unpack_from('<I', d, p); p += 4
    return v, p


def _read_fcb_object(d, p):
    """Read one FCB object -> ({'hash', 'values': [(hash, bytes)],
    'children': [...]}, next_offset)."""
    nch, p = _read_count(d, p)
    h, = struct.unpack_from('<I', d, p); p += 4
    nval, p = _read_count(d, p)
    values = []
    for _ in range(nval):
        vh, = struct.unpack_from('<I', d, p); p += 4
        sz, p = _read_count(d, p)
        values.append((vh, d[p:p + sz])); p += sz
    children = []
    for _ in range(nch):
        c, p = _read_fcb_object(d, p)
        children.append(c)
    return {'hash': h, 'values': values, 'children': children}, p


def _read_fcb_object_at(d, off):
    """Parse one 'nbCF' blob at `off` -> (root object dict, end offset)."""
    if d[off:off + 4] != b'nbCF':
        raise ValueError("no nbCF magic at 0x%x" % off)
    ver, flags, nobj, nval = struct.unpack_from('<HHII', d, off + 4)
    return _read_fcb_object(d, off + 16)


def parse_fcb(d, off):
    """Parse one 'nbCF' blob at `off` -> root object dict."""
    return _read_fcb_object_at(d, off)[0]


def _fcb_strings(obj):
    """All NUL-terminated printable string values in an FCB tree."""
    out = []
    for vh, data in obj['values']:
        if (len(data) > 1 and data[-1:] == b'\x00'
                and all(0x20 <= c < 0x7F for c in data[:-1])):
            out.append((vh, data[:-1].decode('latin-1')))
    for c in obj['children']:
        out.extend(_fcb_strings(c))
    return out


# ---------------------------------------------------------------------------
# Section parsers
# ---------------------------------------------------------------------------

def _read_name_field(d, p):
    """u32 crc + char[32] -> (crc, name) ('' if the field is empty)."""
    crc, = struct.unpack_from('<I', d, p)
    raw = d[p + 4:p + 36]
    name = raw.split(b'\x00', 1)[0].decode('latin-1')
    return crc, name


def parse_scene_elements(d, sections):
    """Scene-element table from the 'Events' section.

    Returns [{'kind', 'id1', 'id2', 'duration', 'name', 'parent', 'ref'}].
    """
    off = sections.offsets.get('Events', 0)
    size = sections.sizes.get('Events', 0)
    if not off or size < 4:
        return []
    count, = struct.unpack_from('<I', d, off)
    out = []
    p = off + 4
    for _ in range(min(count, max(0, (size - 4) // 172))):
        kind, = struct.unpack_from('<H', d, p)
        id1, id2 = struct.unpack_from('<II', d, p + 8)
        dur, = struct.unpack_from('<f', d, p + 20)
        _, name = _read_name_field(d, p + 24)
        _, parent = _read_name_field(d, p + 60)
        _, ref = _read_name_field(d, p + 132)
        out.append({'kind': kind, 'id1': id1, 'id2': id2, 'duration': dur,
                    'name': name, 'parent': parent, 'ref': ref})
        p += 172
    return out


def parse_timed_events(d, sections):
    """Timed event list from the 'UnkSec4' section.

    Returns [{'time', 'type', 'strings': [str], 'raw': fcb_root}] sorted
    by time.  'type' is the event class name; 'strings' the remaining
    string parameters (sound/dialog/particle names, anchor names...).
    """
    off = sections.offsets.get('UnkSec4', 0)
    size = sections.sizes.get('UnkSec4', 0)
    if not off or not size:
        return []
    out = []
    p, end = off, off + size
    while p + 12 <= end:
        t, marker, fcb_size, rec_size = struct.unpack_from('<fIHH', d, p)
        if d[p + 12:p + 16] != b'nbCF':
            break
        try:
            root, fcb_end = _read_fcb_object_at(d, p + 12)
        except Exception:
            break
        strings = _fcb_strings(root)
        ev_type = next((s for vh, s in strings
                        if vh == _FCB_TYPENAME_HASH), '<unknown>')
        rest = [s for vh, s in strings if vh != _FCB_TYPENAME_HASH]
        out.append({'time': t, 'type': ev_type, 'strings': rest,
                    'raw': root})
        # rec_size == 12-byte header + fcb + 1 terminator, but the LAST
        # record stores its size as a u32 (no rec_size field) — recover
        # from the actual FCB extent in that case.
        if rec_size >= fcb_size + 13 and p + rec_size <= end:
            p += rec_size
        else:
            p = fcb_end + 1
    out.sort(key=lambda e: e['time'])
    return out


def parse_and_blocks(d, sections):
    """Per-element animation blocks from 'UnkSec5'.

    Returns [{'quat_a', 'quat_b' (w,x,y,z or None), 'duration',
              'block_off', 'block_size'}] — one per 'AnD' block, in
    element-table order.  Full track decode is TODO; the header quats
    give each element's initial orientation.
    """
    off = sections.offsets.get('UnkSec5', 0)
    size = sections.sizes.get('UnkSec5', 0)
    if not off or not size:
        return []
    blob = d[off:off + size]
    out = []
    for m in re.finditer(b'AnD\x1a', blob):
        a = m.start()
        dur, = struct.unpack_from('<f', blob, a + 4)
        qa = qb = None
        if a >= 0x20:   # header precedes the magic; quats at -0x20 / -0x10
            v = struct.unpack_from('<8f', blob, a - 0x20)
            # stored as two (x,y,z,w) quats; expose as (w,x,y,z)
            if any(abs(x) > 1e-9 for x in v[:4]):
                qa = (v[3], v[0], v[1], v[2])
            if any(abs(x) > 1e-9 for x in v[4:]):
                qb = (v[7], v[4], v[5], v[6])
        out.append({'duration': dur, 'quat_a': qa, 'quat_b': qb,
                    'block_off': off + a, 'block_size': None})
    for i in range(len(out) - 1):
        out[i]['block_size'] = out[i + 1]['block_off'] - out[i]['block_off']
    if out:
        out[-1]['block_size'] = off + size - out[-1]['block_off']
    return out


def decode_and_tracks(d, sections):
    """Decode each scene element's motion from its 'AnD' block.

    Per element (element-table order) returns:
        {'fps', 'frames',
         'rot_static': (x,y,z,w) or None,  'pos_static': (x,y,z) or None,
         'rot_keys': [(frame, (x,y,z,w))], 'pos_keys': [(frame, (x,y,z))]}

    Block layout (verified on linkunit geton/getoff):
        element = 0x70 header + 'AnD\\x1a' + f32 duration + u64 0 +
                  u32 stream_offsets[4] (element-relative) + u64 block size
        stream 0: u32 count, u32 pad, count x 6-byte Dunia quat  (static rot)
        stream 1: u16 tracks, u16 frames, u32 fps,
                  u32 group_offsets[((frames-1)>>3)+2] (stream-relative),
                  groups in the MAB keyframe encoding:
                  [quat6][u16 mask][popcount(mask) x quat6] — base quat at
                  subframe 0, extras at the masked subframes  (animated rot)
        stream 2: u32 count, u32 pad, f32x3                     (static pos)
        stream 3: u16 tracks, u16 frames, u32 fps,
                  frames x f32x3 raw positions                  (animated pos)
    """
    from .import_mab_fc2 import read_quat
    out = []
    for ab in parse_and_blocks(d, sections):
        base = ab['block_off'] - 0x70
        rec = {'fps': 30, 'frames': 0, 'rot_static': None,
               'pos_static': None, 'rot_keys': [], 'pos_keys': []}
        try:
            offs = struct.unpack_from('<4I', d, ab['block_off'] + 16)
            s0, s1, s2, s3 = (base + o for o in offs)
            c0, = struct.unpack_from('<I', d, s0)
            if c0 >= 1:
                rec['rot_static'] = read_quat(d, s0 + 8)
            c2, = struct.unpack_from('<I', d, s2)
            if c2 >= 1:
                rec['pos_static'] = struct.unpack_from('<3f', d, s2 + 8)

            t1, fc1 = struct.unpack_from('<2H', d, s1)
            fps1, = struct.unpack_from('<I', d, s1 + 4)
            if fps1:
                rec['fps'] = fps1
            rec['frames'] = fc1
            if t1 and fc1:
                ng = ((fc1 - 1) >> 3) + 1
                tbl = struct.unpack_from('<%dI' % (ng + 1), d, s1 + 8)
                for gi in range(ng):
                    p, end = s1 + tbl[gi], s1 + tbl[gi + 1]
                    q = read_quat(d, p)
                    p += 6
                    if q:
                        rec['rot_keys'].append((gi * 8, q))
                    if p + 2 <= end:
                        mask, = struct.unpack_from('<H', d, p)
                        p += 2
                        bits = [b for b in range(16) if (mask >> b) & 1]
                        for b in bits:
                            if p + 6 > end:
                                break
                            q = read_quat(d, p)
                            p += 6
                            if q:
                                rec['rot_keys'].append((gi * 8 + b, q))
                rec['rot_keys'].sort(key=lambda kv: kv[0])

            t3, fc3 = struct.unpack_from('<2H', d, s3)
            if t3 and fc3:
                rec['pos_keys'] = [
                    (i, struct.unpack_from('<3f', d, s3 + 8 + i * 12))
                    for i in range(fc3)]
        except Exception as exc:
            print("[MAB scene] AnD decode failed for block @0x%x: %s"
                  % (ab['block_off'], exc))
        out.append(rec)
    return out


# FCB value hash of a SetFOV event's field-of-view float (degrees).
# Universal across every shipped MAB (187/187 SetFOV events, 20-107 deg).
_FOV_VALUE_HASH = 0xBEF721BA


def camera_fov_cuts(events):
    """[(time_seconds, fov_degrees)] from the clip's SetFOV events, sorted.

    Each SetFOV is a CUT: the directed camera instantly takes a new
    field of view (and, in camera-element clips, teleports to a new
    shot).  Consecutive equal FOVs are kept — a cut can reuse an FOV.
    """
    out = []
    for ev in events:
        if ev['type'] != 'SetFOV':
            continue
        fov = None
        for vh, data in ev['raw']['values']:
            if vh == _FOV_VALUE_HASH and len(data) == 4:
                fov = struct.unpack('<f', data)[0]
                break
        if fov is not None:
            out.append((ev['time'], fov))
    out.sort(key=lambda kv: kv[0])
    return out


def scan_scene(path):
    """One-call summary of a .mab's scene data (no bpy needed).

    Returns {'animlen', 'elements', 'events', 'and_blocks', 'fov_cuts'}.
    """
    d = open(path, 'rb').read()
    sec = parse_sections(d)
    events = parse_timed_events(d, sec)
    return {
        'animlen': sec.animlen,
        'elements': parse_scene_elements(d, sec),
        'events': events,
        'and_blocks': parse_and_blocks(d, sec),
        'fov_cuts': camera_fov_cuts(events),
    }


# ---------------------------------------------------------------------------
# Blender builders
# ---------------------------------------------------------------------------

def build_scene_objects(context, path, fps=30):
    """Create empties/cameras for the scene elements and timeline markers
    for the timed events.  Returns (n_elements, n_events)."""
    if bpy is None:
        raise RuntimeError("bpy unavailable")
    info = scan_scene(path)
    base = os.path.splitext(os.path.basename(path))[0]

    coll = bpy.data.collections.get(base) or bpy.data.collections.new(base)
    if coll.name not in context.scene.collection.children:
        context.scene.collection.children.link(coll)

    d = open(path, 'rb').read()
    from .import_mab_fc2 import parse_sections as _ps
    tracks = decode_and_tracks(d, _ps(d))

    # Scene root carries the same 180° Z the model importer applies, so
    # cameras/anchors line up with imported characters.
    import math
    root = bpy.data.objects.new(base + ".scene", None)
    root.empty_display_type = 'ARROWS'
    root.rotation_euler = (0, 0, math.pi)
    coll.objects.link(root)

    scene = context.scene
    cut_frames = sorted({int(round(t * fps)) + 1 for t, _ in info['fov_cuts']})

    objs = {}
    cameras = []          # (obj, track) for every camera element
    n_animated = 0
    for i, el in enumerate(info['elements']):
        nm = "%s.%s" % (base, el['name'] or 'element_%d' % i)
        is_cam = 'camera' in el['name'].lower() or el['kind'] == 9
        if is_cam:
            cam = bpy.data.cameras.new(nm)
            cam.sensor_fit = 'HORIZONTAL'
            cam.sensor_width = 36.0
            ob = bpy.data.objects.new(nm, cam)
            ob.show_name = True
        else:
            ob = bpy.data.objects.new(nm, None)
            ob.empty_display_type = 'PLAIN_AXES'
            ob.empty_display_size = 0.25
        coll.objects.link(ob)
        ob['mab_scene_kind'] = el['kind']
        ob['mab_scene_name'] = el['name']
        ob['mab_scene_parent'] = el['parent']
        objs[el['name']] = ob

        tr = tracks[i] if i < len(tracks) else None
        if tr is None:
            continue
        # Dunia cameras look down their LOCAL +Y axis (verified on the
        # linkunit clips: +Y tracks the character with avg dot 0.965 over
        # the whole shot); Blender cameras look down -Z.  Rx(+90) maps
        # -Z onto +Y (and Blender's +Y up onto Dunia's +Z up).
        corr = (mathutils.Quaternion((1.0, 0.0, 0.0), math.radians(90.0))
                if is_cam else None)

        def _bq(q):
            x, y, z, w = q
            out = mathutils.Quaternion((w, x, y, z))
            return (out @ corr) if corr is not None else out

        ob.rotation_mode = 'QUATERNION'
        if tr['pos_static']:
            ob.location = tr['pos_static']
        if tr['rot_static']:
            ob.rotation_quaternion = _bq(tr['rot_static'])
        # animated tracks: AnD streams run at their own fps (often 15);
        # scale onto the scene fps used for the event markers.
        scale = fps / float(tr['fps'] or fps)
        if tr['pos_keys'] or tr['rot_keys']:
            n_animated += 1
        for fr, pos in tr['pos_keys']:
            ob.location = pos
            ob.keyframe_insert('location',
                               frame=int(round(fr * scale)) + 1)
        for fr, q in tr['rot_keys']:
            ob.rotation_quaternion = _bq(q)
            ob.keyframe_insert('rotation_quaternion',
                               frame=int(round(fr * scale)) + 1)
        # Hard cuts: a directed camera TELEPORTS at each cut, so the
        # transform key on the frame BEFORE a cut must hold (CONSTANT)
        # instead of Bezier-swooping into the next shot.
        if is_cam and cut_frames:
            _set_constant_before(ob, cut_frames)
        if is_cam:
            cameras.append((ob, tr))

    for el in info['elements']:
        ob = objs.get(el['name'])
        par = objs.get(el['parent'])
        if ob is None:
            continue
        if par is not None and par is not ob:
            ob.parent = par
        else:
            ob.parent = root

    # ── Synthesize a camera for live-camera clips ───────────────────────
    # Some scripted clips carry SetFOV CUTS but NO authored camera element
    # (the dialogue/gameplay camera position is computed at runtime, not
    # stored).  Without this the user gets shot markers but no camera
    # object.  Create one anyway: real camera, FOV cuts keyed, placed at
    # the scene's centre anchor so it sits in the action; the user can dial
    # in the framing.  It carries no authored motion (flagged on the obj).
    synth_cam = False
    if not cameras and info['fov_cuts']:
        cam = bpy.data.cameras.new(base + ".camera_fov")
        cam.sensor_fit = 'HORIZONTAL'
        cam.sensor_width = 36.0
        cam_ob = bpy.data.objects.new(base + ".camera_fov", cam)
        cam_ob.show_name = True
        cam_ob['mab_camera_synthetic'] = 1   # FOV authored, position is not
        coll.objects.link(cam_ob)
        # place + aim at the centre anchor if there is one
        anchor = next((tracks[i]['pos_static'] for i, el in
                       enumerate(info['elements'])
                       if el['kind'] == 7 and i < len(tracks)
                       and tracks[i]['pos_static']), None)
        if anchor:
            tgt = mathutils.Vector(anchor)
            cam_ob.location = tgt + mathutils.Vector((0.0, -4.0, 1.6))
            direction = (tgt - cam_ob.location)
            cam_ob.rotation_mode = 'QUATERNION'
            cam_ob.rotation_quaternion = direction.to_track_quat('-Z', 'Y')
        cam_ob.parent = root
        cameras.append((cam_ob, None))
        synth_cam = True

    # ── Clear this clip's old markers up front ──────────────────────────
    for mk in [m for m in scene.timeline_markers
               if m.name.startswith(('EV:', 'SHOT:', 'CAM:'))]:
        scene.timeline_markers.remove(mk)

    # ── Camera FOV animation (each SetFOV = a cut) ──────────────────────
    n_cuts = 0
    if cameras and info['fov_cuts']:
        cam_data = cameras[0][0].data         # one authored camera per MAB
        for t, fov in info['fov_cuts']:
            cam_data.angle = math.radians(max(1.0, min(170.0, fov)))
            cam_data.keyframe_insert('lens', frame=int(round(t * fps)) + 1)
            n_cuts += 1
        _set_fcurve_constant(cam_data)        # FOV holds per shot, jumps

    # ── Bind cameras to the timeline + set the active scene camera ──────
    # Blender switches the active camera at each bind marker, so a
    # multi-part scene (several MABs) cuts between their cameras exactly
    # like the game.  Within one MAB the single camera binds at frame 1.
    for ci, (cam_ob, _) in enumerate(cameras):
        mk = scene.timeline_markers.new(('CAM:' + cam_ob.name)[:60], frame=1)
        mk.camera = cam_ob
        if ci == 0 and scene.camera is None:
            scene.camera = cam_ob
    if cameras:
        scene.camera = cameras[0][0]

    # ── SHOT markers: one per cut, label = FOV + how long the shot lasts ─
    fov_cuts = info['fov_cuts']
    for si, (t, fov) in enumerate(fov_cuts):
        end_t = (fov_cuts[si + 1][0] if si + 1 < len(fov_cuts)
                 else info['animlen'])
        scene.timeline_markers.new(
            'SHOT:%d %d\xb0 (%.1fs)' % (si + 1, round(fov), max(0.0, end_t - t)),
            frame=int(round(t * fps)) + 1)

    # ── Event cue markers (sound / fx / dialog); SetFOV shown as SHOT: ──
    for ev in info['events']:
        if ev['type'] == 'SetFOV':
            continue
        label = ev['type']
        if ev['strings']:
            label += ' ' + ev['strings'][0]
        frame = int(round(ev['time'] * fps)) + 1
        scene.timeline_markers.new(('EV:' + label)[:60], frame=frame)

    if info['animlen'] > 0:
        scene.frame_end = max(scene.frame_end,
                              int(round(info['animlen'] * fps)) + 1)
    return {
        'elements': len(info['elements']),
        'events': len(info['events']),
        'cameras': len(cameras),
        'cuts': n_cuts,
        'synthetic_camera': synth_cam,
    }


def _iter_action_fcurves(id_owner):
    """Yield every F-Curve of an ID's active action across Blender versions.

    Blender 4.4+ replaced the flat ``Action.fcurves`` with slotted actions
    (layers -> strips -> channelbags); 5.0 removed ``Action.fcurves``
    entirely.  Try the legacy accessor first, then walk the slotted layout.
    """
    ad = getattr(id_owner, 'animation_data', None)
    if not ad or not ad.action:
        return
    action = ad.action
    legacy = getattr(action, 'fcurves', None)
    if legacy is not None:                      # Blender <= 4.3
        yield from legacy
        return
    slot = getattr(ad, 'action_slot', None)     # Blender 4.4+
    for layer in getattr(action, 'layers', ()):
        for strip in getattr(layer, 'strips', ()):
            bag = None
            if slot is not None and hasattr(strip, 'channelbag'):
                try:
                    bag = strip.channelbag(slot)
                except Exception:
                    bag = None
            if bag is not None:
                yield from bag.fcurves
            else:
                for b in getattr(strip, 'channelbags', ()):
                    yield from b.fcurves


def _set_constant_before(ob, cut_frames):
    """Set the transform keyframe on each (cut_frame - 1) to CONSTANT so a
    teleporting camera hard-cuts instead of interpolating across the jump."""
    cuts = set(cut_frames) | {f - 1 for f in cut_frames}
    for fc in _iter_action_fcurves(ob):
        if not (fc.data_path.endswith('location')
                or fc.data_path.endswith('rotation_quaternion')):
            continue
        for kp in fc.keyframe_points:
            if int(round(kp.co[0])) in cuts:
                kp.interpolation = 'CONSTANT'


def _set_fcurve_constant(cam_data):
    """All lens/FOV keys hold then jump (per-shot FOV)."""
    for fc in _iter_action_fcurves(cam_data):
        for kp in fc.keyframe_points:
            kp.interpolation = 'CONSTANT'
