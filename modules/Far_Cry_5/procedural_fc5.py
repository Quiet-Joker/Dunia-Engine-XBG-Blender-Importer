"""Procedural-bone emulation — twist / corrective constraints for MAB playback.

Ubisoft's Dunia engine drives a class of bones PROCEDURALLY at runtime; the
.mab carries no tracks for them, so left at rest they cause the classic
artefacts (candy-wrapper wrist, inward knee/elbow dent).  RE of the FC4
`*_ref.skeleton` (see agents.md "Engine internals") confirmed the topology:

  * `<side>ForeArmTwistA/B`  → leaf under `<side>ForeArm`, driven by sibling
    `<side>Hand`   (distribute WRIST roll; two bones self-distribute by their
    fractional position along the segment, so the one nearer the wrist takes
    more — purely data-driven, no name ordering assumed).
  * `<side>ArmTwistA/B`      → leaf under `<side>Arm`, driven by `<side>ForeArm`.
  * `<side>Elbow` / `<side>Knee` → corrective leaf, copies 50% of the bending
    joint (the animated sibling).
  * `<side>HandThumbHelper`  → finger corrective.
  Other rigs name the twists `<side>ForeArmRoll` / `<side>ArmRoll` / `*RollEx`.

We reproduce them with COPY_ROTATION (LOCAL/LOCAL): twist = long-axis (Blender
local Y is ALWAYS the bone's length axis, so this is convention-independent)
at the fractional influence; corrective = all axes at 0.5.

`plan_helpers` is pure Python (no bpy) so the classification + influence is
unit-testable headless; `emulate_procedural_helpers` applies the plan in Blender.
Shared by the Avatar applier (`avatar/import_mab.apply_multi_bone`) and the
Disrupt applier (`watchdogs/import_mab_wd.apply_wd1_mab`, used by FC4 + WD1).
"""

import math

_TWIST_TOKENS = ('twist', 'roll')
_CORR_TOKENS = ('elbow', 'knee', 'helper')

# Blender re-frames every bone so local +Y runs head→tail, so the roll axis of
# ANY bone is local Y regardless of the game's axis convention.
_LONG_AXIS = (0.0, 1.0, 0.0)


def extract_twist(q, axis=_LONG_AXIS, fraction=1.0):
    """Swing-twist decomposition — return the pure TWIST of quaternion `q`
    about `axis` (unit), scaled to `fraction` of its angle.  Pure Python on
    (w,x,y,z); the engine's `RollExtractionMode` does exactly this, so a twist
    bone gets only the driver's roll and NONE of its swing (bend).  COPY_ROTATION
    local-Y instead copies the Euler-Y component, which leaks swing — the reason
    the live emulation only approximates the candy-wrapper wrist.
    """
    n = math.sqrt(q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3]) or 1.0
    w, x, y, z = q[0]/n, q[1]/n, q[2]/n, q[3]/n
    d = x*axis[0] + y*axis[1] + z*axis[2]          # project onto twist axis
    tw = (w, axis[0]*d, axis[1]*d, axis[2]*d)
    m = math.sqrt(tw[0]*tw[0] + tw[1]*tw[1] + tw[2]*tw[2] + tw[3]*tw[3])
    if m < 1e-7:                                    # rotation ⟂ axis → no twist
        return (1.0, 0.0, 0.0, 0.0)
    tw = (tw[0]/m, tw[1]/m, tw[2]/m, tw[3]/m)
    if tw[0] < 0.0:                                 # canonical hemisphere
        tw = (-tw[0], -tw[1], -tw[2], -tw[3])
    if fraction < 0.999:
        half = math.acos(max(-1.0, min(1.0, tw[0]))) * fraction
        vlen = math.sqrt(tw[1]*tw[1] + tw[2]*tw[2] + tw[3]*tw[3])
        if vlen < 1e-7:
            return (1.0, 0.0, 0.0, 0.0)
        s = math.sin(half)
        tw = (math.cos(half), tw[1]/vlen*s, tw[2]/vlen*s, tw[3]/vlen*s)
    return tw


def _proj_fraction(tw, axis):
    """Fractional position of `tw` projected onto `axis` (both parent-local),
    clamped to [0,1].  0 at the joint, 1 at the sibling tip."""
    l2 = axis[0]*axis[0] + axis[1]*axis[1] + axis[2]*axis[2]
    if l2 <= 1e-12:
        return 0.5
    t = (tw[0]*axis[0] + tw[1]*axis[1] + tw[2]*axis[2]) / l2
    return 0.0 if t < 0.0 else (1.0 if t > 1.0 else t)


def plan_helpers(names, positions, parent_idx, keyed_names):
    """Decide which procedural bones to emulate and how — PURE (no bpy).

    names        : list[str]                  bone names, index-aligned
    positions    : list[(x,y,z)]              PARENT-LOCAL translation per bone
    parent_idx   : list[int]                  parent index, -1 / None for root
    keyed_names  : set[str]                   bones animated in the .mab

    Returns list of dicts: {bone, driver, kind:'twist'|'corrective', influence}.
    A procedural bone qualifies only if it is an un-keyed LEAF whose name matches
    a twist/corrective token AND it has an animated sibling to drive it.
    """
    n = len(names)
    children = {}
    for i, p in enumerate(parent_idx):
        if p is not None and 0 <= p < n:
            children.setdefault(p, []).append(i)

    out = []
    for i, nm in enumerate(names):
        if nm in keyed_names:           # animated itself — not procedural
            continue
        if children.get(i):             # helpers are leaves
            continue
        low = nm.lower()
        is_twist = any(t in low for t in _TWIST_TOKENS)
        is_corr = any(t in low for t in _CORR_TOKENS)
        if not (is_twist or is_corr):
            continue
        p = parent_idx[i]
        if p is None or not (0 <= p < n):
            continue
        # driver = the animated real-joint sibling (Hand / ForeArm / Leg / …)
        sib = next((c for c in children.get(p, ())
                    if names[c] in keyed_names and c != i), None)
        if sib is None:
            continue
        if is_twist:
            infl = _proj_fraction(positions[i], positions[sib])
            out.append({'bone': nm, 'driver': names[sib],
                        'kind': 'twist', 'influence': infl})
        else:
            out.append({'bone': nm, 'driver': names[sib],
                        'kind': 'corrective', 'influence': 0.5})
    return out


def _iter_fcurves(arm_obj):
    """All F-Curves of the object's active action (handles Blender 4.4+/5 slotted
    actions as well as legacy `Action.fcurves`)."""
    ad = getattr(arm_obj, 'animation_data', None)
    if not ad or not ad.action:
        return []
    act = ad.action
    legacy = getattr(act, 'fcurves', None)
    if legacy:
        return list(legacy)
    out = []
    try:
        slot = getattr(ad, 'action_slot', None)
        for layer in act.layers:
            for strip in layer.strips:
                cb = strip.channelbag(slot) if (slot and hasattr(strip, 'channelbag')) else None
                if cb:
                    out.extend(cb.fcurves)
    except Exception:
        pass
    return out


def fcurve_container(obj, action):
    """Container exposing `.fcurves.new(path, index)` for `obj`'s channels.

    Blender 4.4+/5.x replaced ``Action.fcurves`` with slotted actions (fcurves
    live in a channelbag under slot+layer+strip, and the slot must be bound to
    the object's animation_data for playback). Falls back to ``action.fcurves``
    on legacy (<=4.3). Shared by the MAB bone keyer and the twist baker."""
    if hasattr(action, 'fcurves'):              # legacy Blender <= 4.3
        return action
    ad = obj.animation_data
    slot = action.slots[0] if action.slots else \
        action.slots.new(id_type='OBJECT', name=obj.name[:60])
    try:
        if ad.action_slot != slot:
            ad.action_slot = slot
    except Exception:
        pass
    layer = action.layers[0] if action.layers else action.layers.new("Layer")
    strip = layer.strips[0] if layer.strips else layer.strips.new(type='KEYFRAME')
    return strip.channelbag(slot, ensure=True)


def bulk_key_quaternion(container, pb, frames, quats):
    """Populate a pose bone's 4 rotation_quaternion fcurves in one foreach_set
    pass (vs per-frame keyframe_insert). Assumes the bone has no existing keys.
    `frames` 1-based ints; `quats` mathutils Quaternions (wxyz)."""
    bp = pb.path_from_id()
    n = len(frames)
    for ci in range(4):
        fc = container.fcurves.new(f'{bp}.rotation_quaternion', index=ci)
        fc.keyframe_points.add(n)
        flat = [0.0] * (2 * n)
        for k in range(n):
            flat[2 * k] = frames[k]
            flat[2 * k + 1] = quats[k][ci]
        fc.keyframe_points.foreach_set('co', flat)
        fc.update()


def _q_angle(q):
    n = math.sqrt(q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3]) or 1.0
    return 2.0 * math.acos(min(1.0, abs(q[0]) / n))


def _bake_twist(context, arm_obj, twist_plans, frame_start, frame_end, log=None):
    """Bake pure swing-twist keys onto each twist bone (exact engine roll
    extraction).  Samples the driver's rotation F-Curves DIRECTLY (no per-frame
    `scene.frame_set` — that re-evaluates the whole depsgraph every frame and is
    the expensive part).  Keys at the driver's own keyframe frames, which are
    already dense after SQUAD, so it stays accurate without baking every integer
    frame.  Returns the number of twist bones baked.

    Diagnostic: per bone logs the driver's max rotation vs the roll extracted
    about local-Y.  driver big + roll ~0 ⇒ the roll axis ISN'T local Y in this
    rig (axis bug); roll big but mesh still static ⇒ skinning / missing bone."""
    import mathutils
    pbones = arm_obj.pose.bones
    by_dp = {}
    for fc in _iter_fcurves(arm_obj):
        by_dp.setdefault(fc.data_path, {})[fc.array_index] = fc
    n = 0
    for pl in twist_plans:
        dfc = by_dp.get('pose.bones["%s"].rotation_quaternion' % pl['driver'], {})
        if len(dfc) < 4:
            continue                                    # driver not keyed → skip
        frames = sorted({int(round(kp.co[0]))
                         for fc in dfc.values() for kp in fc.keyframe_points})
        tb = pbones[pl['bone']]
        drv = pbones[pl['driver']]
        tb.rotation_mode = 'QUATERNION'
        for c in list(tb.constraints):                  # drop any old live helper
            if c.name.startswith('MAB helper'):
                tb.constraints.remove(c)
        # Roll axis = the LIMB segment direction (the twist bone's parent — the
        # forearm/upper-arm — points down the limb), NOT the driver bone's own
        # local Y (which may not line up with the limb).  Take that segment axis
        # in (a) the driver's rest frame to read its roll, and (b) the twist
        # bone's rest frame to apply it.
        seg = tb.parent
        drv_axis = (0.0, 1.0, 0.0)
        tb_axis = mathutils.Vector((0.0, 1.0, 0.0))
        if seg is not None:
            fa = seg.bone.matrix_local.to_3x3().col[1].normalized()  # limb dir (armature space)
            da = drv.bone.matrix_local.to_3x3().inverted() @ fa
            if da.length > 1e-6:
                da.normalize(); drv_axis = (da.x, da.y, da.z)
            ta = tb.bone.matrix_local.to_3x3().inverted() @ fa
            if ta.length > 1e-6:
                ta.normalize(); tb_axis = ta
        ex, ey, ez = drv_axis
        infl = pl['influence']
        max_drv = max_roll = 0.0
        bframes = []
        bquats = []
        for f in frames:
            w = dfc[0].evaluate(f); x = dfc[1].evaluate(f)
            y = dfc[2].evaluate(f); z = dfc[3].evaluate(f)
            max_drv = max(max_drv, _q_angle((w, x, y, z)))
            ang = 2.0 * math.atan2(x*ex + y*ey + z*ez, w)   # signed roll about limb
            if ang > math.pi:
                ang -= 2.0 * math.pi
            elif ang < -math.pi:
                ang += 2.0 * math.pi
            max_roll = max(max_roll, abs(ang))
            bframes.append(f)
            bquats.append(mathutils.Quaternion(tb_axis, ang * infl))
        # Bulk-key the twist bone (was ~1 keyframe_insert per dense frame — the
        # single worst hot spot in long-animation import).
        if bframes:
            action = arm_obj.animation_data.action
            bulk_key_quaternion(fcurve_container(arm_obj, action), tb, bframes, bquats)
        if log:
            log("  twist %-22s driver max %3.0f deg -> roll-about-limb %3.0f deg "
                "(x infl %.2f)" % (pl['bone'], math.degrees(max_drv),
                                   math.degrees(max_roll), pl['influence']))
        n += 1
    return n


def add_eye_lookat(arm_obj, keyed_names, log=None):
    """Wire FC5's eye look-at IK: each eyeball bone aims at the `EyeLookAt`
    target (a child of Head the engine drives so both eyes converge / track).

    The .mab keys `EyeLookAt`; the eye bones themselves carry no rotation tracks,
    so without a constraint the eyes stare dead ahead.  We add a Damped Track per
    eye, aiming the bone's local axis that already points at the target at rest
    (so there's no pop) — computed from the ACTUAL armature rest matrices, which
    is the frame Damped Track operates in.

    Self-gating: a no-op on rigs without eye bones (e.g. FP-arms meshes).
    Returns the number of eyes wired.
    """
    import mathutils
    pbones = arm_obj.pose.bones

    def _find(suffixes):
        out = []
        for pb in pbones:
            base = pb.name.split(':')[-1]
            if base in suffixes:
                out.append(pb)
        return out

    targets = _find({'EyeLookAt'})
    if not targets:
        return 0
    target = targets[0]
    eyes = [pb for pb in pbones
            if pb.name.split(':')[-1] in ('LeftEyeRoot', 'RightEyeRoot',
                                          'LeftEye', 'RightEye')
            and pb.name not in keyed_names]
    if not eyes:
        return 0

    tpos = target.bone.matrix_local.translation
    n = 0
    for pb in eyes:
        m = pb.bone.matrix_local
        d = tpos - m.translation
        if d.length < 1e-6:
            continue
        d.normalize()
        # direction to target expressed in the eye bone's local frame
        local = m.to_3x3().inverted() @ d
        ax = max(range(3), key=lambda k: abs(local[k]))
        axis = ('X', 'Y', 'Z')[ax]
        track = ('TRACK_' if local[ax] >= 0 else 'TRACK_NEGATIVE_') + axis
        for c in list(pb.constraints):
            if c.name.startswith('MAB eye'):
                pb.constraints.remove(c)
        con = pb.constraints.new('DAMPED_TRACK')
        con.name = 'MAB eye lookat'
        con.target = arm_obj
        con.subtarget = target.name
        con.track_axis = track
        n += 1
    if log and n:
        log("[MAB] eye look-at IK: %d eyes -> %s (%s)"
            % (n, target.name, track))
    return n


def emulate_procedural_helpers(arm_obj, names, positions, parent_idx,
                               keyed_names, log=None, bake=False, context=None,
                               frame_start=1, frame_end=2):
    """Reproduce the engine's procedural bones on `arm_obj`.

    Correctives (elbow/knee/helper) are always a live COPY_ROTATION at 0.5 (a
    plain blend, already exact).  Twist bones are either BAKED as true swing-twist
    keys (`bake=True`, needs `context` + the frame range — matches the engine's
    RollExtractionMode) or a live COPY_ROTATION local-Y (the editable but
    swing-contaminated approximation).  Bake failures fall back to live, so an
    import never breaks.  Returns the count applied.
    """
    pbones = arm_obj.pose.bones
    all_plans = plan_helpers(names, positions, parent_idx, keyed_names)
    plans = [pl for pl in all_plans
             if pl['bone'] in pbones and pl['driver'] in pbones]

    if log:
        missing = [pl['bone'] for pl in all_plans if pl['bone'] not in pbones]
        nodriver = [(pl['bone'], pl['driver']) for pl in all_plans
                    if pl['bone'] in pbones and pl['driver'] not in pbones]
        if missing:
            log("[MAB] %d procedural bones NOT in the armature (can't emulate, "
                "skin won't twist): %s" % (len(missing), missing))
        if nodriver:
            log("[MAB] procedural bones whose driver is absent: %s" % nodriver)
        if plans:
            log("[MAB] procedural plan: " + ", ".join(
                "%s<-%s(%.2f,%s)" % (pl['bone'], pl['driver'], pl['influence'],
                                     pl['kind'][:4]) for pl in plans))

    baked = set()
    n = 0
    if bake and context is not None and frame_end >= frame_start:
        twist_plans = [pl for pl in plans if pl['kind'] == 'twist']
        if twist_plans:
            try:
                n += _bake_twist(context, arm_obj, twist_plans,
                                 frame_start, frame_end, log=log)
                baked = {pl['bone'] for pl in twist_plans}
            except Exception as e:
                if log:
                    log("[MAB] twist bake failed (%s); using live constraints" % e)

    for pl in plans:
        if pl['bone'] in baked:
            continue
        pb = pbones[pl['bone']]
        for c in list(pb.constraints):
            if c.name.startswith('MAB helper'):
                pb.constraints.remove(c)
        con = pb.constraints.new('COPY_ROTATION')
        con.name = 'MAB helper'
        con.target = arm_obj
        con.subtarget = pl['driver']
        con.target_space = con.owner_space = 'LOCAL'
        if pl['kind'] == 'twist':
            # FULL-FOLLOW (all axes): the twist bone follows its influence-
            # fraction of the wrist's WHOLE motion (bend + roll), so the distal
            # forearm turns *with* the wrist instead of only pronating.  (The
            # baked path stays pure roll — engine-accurate; this live path is the
            # "follow the wrist" alternative.)
            con.influence = pl['influence']
        else:
            con.influence = 0.5
        n += 1
    # Eye look-at IK (no-op on rigs without eye bones).
    try:
        add_eye_lookat(arm_obj, keyed_names, log=log)
    except Exception as e:
        if log:
            log("[MAB] eye look-at IK skipped: %s" % e)

    if log and n:
        log("[MAB] emulated %d procedural helper bones (%d twist baked)"
            % (n, len(baked)))
    return n
