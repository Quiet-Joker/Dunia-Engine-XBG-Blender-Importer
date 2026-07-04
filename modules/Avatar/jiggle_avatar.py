"""Procedural jiggle-bone simulation (Avatar `proceduralbones.xml`).

Reads each procedural bone's spring/displacement params (decoded with
`avatar.proceduralbones`) and bakes a damped-spring secondary-motion jiggle onto
the matching bones in the rig — so satchels / pouches / skirts (and your own
custom bust / rear bones) swing with the animation.  Lets you PREVIEW jiggle in
Blender before injecting anything.

PREVIEW MODEL: an inertial spring-bone.  The bone tip carries a virtual mass
that LAGS the rigid tip via a damped spring (`Tension`=stiffness,
`Friction`=damping); the lag produces overshoot + settle (real-looking bounce),
`strength` scales the lag deviation, and a tanh soft-limit caps the bend at the
`Min/MaxRotation` envelope without snapping.  This is intentionally NOT the
game's runtime integrator — the engine recomputes the real jiggle from
proceduralbones.xml at runtime (see agents.md), so the preview only has to look
physical, not be byte-exact.  `_spring_step` is pure Python (no bpy) so the
spring math stays unit-testable headless.
"""

import math
import os

import bpy


def _vec3(s, default=(0.0, 0.0, 0.0)):
    try:
        a = [float(x) for x in str(s).split(',')]
        return tuple((a + [0.0, 0.0, 0.0])[:3])
    except Exception:
        return default


def load_jiggle_defs(xml_path, pawn_type='corp'):
    """proceduralbones.xml (compiled binary OR text) -> {bone_name: params}."""
    import xml.etree.ElementTree as ET
    data = open(xml_path, 'rb').read()
    if data[:3] == b'\x00\x00\xff':
        from . import proceduralbones as pbmod
        root = pbmod.decode(data)
    else:
        root = ET.fromstring(data.decode('utf-8-sig'))
    defs = {}
    for pawn in root.findall('PawnType'):
        if pawn_type and pawn.get('type') != pawn_type:
            continue
        for b in pawn.findall('Bone'):
            defs[b.get('Name')] = {
                'min_rot': _vec3(b.get('MinRotation', '0,0,0')),
                'max_rot': _vec3(b.get('MaxRotation', '0,0,0')),
                'axis':    _vec3(b.get('DisplacementAxisEffect', '0,0,0')),
                'invert':  _vec3(b.get('InvertDisplacementEffect', '0,0,0')),
                'mult':    _vec3(b.get('MovementMultiplier', '0,0,0')),
                'tension': float(b.get('Tension', '30') or 30),
                'friction': float(b.get('Friction', '5') or 5),
            }
    return defs


def _spring_step(ang, vel, drive, tension, friction, dt):
    """One semi-implicit-Euler step of a 3-axis damped spring toward rest (0).
    Mutates `ang`/`vel` in place; `drive` = the inertial forcing per axis."""
    for ax in range(3):
        accel = drive[ax] - tension * ang[ax] - friction * vel[ax]
        vel[ax] += accel * dt
        ang[ax] += vel[ax] * dt
    return ang


def _clamp(v, lo, hi):
    return lo if v < lo else (hi if v > hi else v)


def _soft_limit(theta, cap):
    """Smooth saturation: ~linear for small theta, asymptotes to ±cap.
    Keeps the bend bounded WITHOUT the snap-to-clamp pop that a hard clamp
    gives when the drive overshoots the limit."""
    if cap <= 1e-6:
        return 0.0
    return cap * math.tanh(theta / cap)


def bake_jiggle(context, arm_obj, defs, frame_start, frame_end,
                strength=1.0, log=None):
    """Bake jiggle as an INERTIAL SPRING-BONE — the preview model.

    The game's own per-frame integrator (clamp + linear approach + linear
    decay; see agents.md "Avatar procedural-jiggle integrator") is faithful
    to runtime but, as a Blender preview, reads as a stiff side-to-side
    *follow* with no bounce — and blows up when `strength` scales its approach
    rate.  The preview doesn't need to be byte-exact (the game recomputes the
    real jiggle from proceduralbones.xml at runtime), so here we use a proper
    damped spring-bone that LOOKS physical:

      • The bone tip has a virtual mass that LAGS the rigid tip via a damped
        spring (`Tension` = stiffness, `Friction` = damping).  Lag → overshoot
        → settle = the bounce you expect from jiggle.
      • The lag *deviation* (not an approach rate) is what `strength` scales,
        so turning it up gives more swing, never per-frame snapping.
      • A tanh soft-limit caps the bend smoothly at the Min/Max envelope, so
        even high strength saturates gracefully instead of popping.
      • Sub-stepped for stability at stiff settings.

    Bakes `rotation_euler` keys on each target bone.  Returns count baked."""
    import mathutils
    from mathutils import Vector, Matrix
    scene = context.scene
    pbones = arm_obj.pose.bones
    targets = [(nm, d) for nm, d in defs.items() if nm in pbones]
    if not targets:
        if log:
            log("[jiggle] none of those bones are in this rig")
        return 0
    # Parents first, so a jiggle bone parented under another jiggle bone reads
    # its parent's already-applied pose this frame.
    targets.sort(key=lambda t: len(pbones[t[0]].parent_recursive))

    fps = scene.render.fps or 30
    dt = 1.0 / float(fps)
    SUB = 4
    hdt = dt / SUB

    st = {}
    for nm, d in targets:
        pb = pbones[nm]
        pb.rotation_mode = 'XYZ'
        # Per-axis envelope -> a single max bend angle for the soft-limit.
        lim = max(abs(d['min_rot'][i]) for i in range(3))
        lim = max(lim, max(abs(d['max_rot'][i]) for i in range(3)))
        if lim <= 1e-4:
            lim = 1.0                               # ~57°, sane default
        st[nm] = {'dyn': None, 'vel': Vector((0.0, 0.0, 0.0)), 'cap': lim}

    saved = scene.frame_current
    for f in range(int(frame_start), int(frame_end) + 1):
        scene.frame_set(f)
        for nm, d in targets:
            pb = pbones[nm]
            s = st[nm]
            base = pb.matrix.copy()                 # animated world matrix
            head_w = base.translation.copy()
            y_axis = base.col[1].xyz.normalized()   # bone points along +Y
            length = pb.length or 0.1
            tip_rest = head_w + y_axis * length

            stiff = max(1.0, d['tension'])
            damp = max(0.0, d['friction'])

            if s['dyn'] is None:                    # seed at rest, no pop
                s['dyn'] = tip_rest.copy()
                s['vel'] = Vector((0.0, 0.0, 0.0))
            dyn = s['dyn']; vel = s['vel']
            for _ in range(SUB):                    # damped spring toward rigid tip
                vel = vel + (tip_rest - dyn) * (stiff * hdt)
                vel = vel * max(0.0, 1.0 - damp * hdt)
                dyn = dyn + vel * hdt
            s['dyn'] = dyn; s['vel'] = vel

            # Amplify the lag deviation by strength, then bend the bone from its
            # rigid direction toward the (exaggerated) lagged tip.
            dev = (dyn - tip_rest) * strength
            tip_eff = tip_rest + dev
            dyn_dir = tip_eff - head_w
            if dyn_dir.length < 1e-9:
                pb.rotation_euler = (0.0, 0.0, 0.0)
                pb.keyframe_insert('rotation_euler', frame=f)
                continue
            dyn_dir.normalize()
            theta = y_axis.angle(dyn_dir, 0.0)
            axis = y_axis.cross(dyn_dir)
            if theta < 1e-6 or axis.length < 1e-9:
                pb.rotation_euler = (0.0, 0.0, 0.0)
                pb.keyframe_insert('rotation_euler', frame=f)
                continue
            axis.normalize()
            theta = _soft_limit(theta, s['cap'])    # smooth bound, no snap
            q = mathutils.Quaternion(axis, theta)   # world-space bend about head
            R = (Matrix.Translation(head_w)
                 @ q.to_matrix().to_4x4()
                 @ Matrix.Translation(-head_w))
            pb.matrix = R @ base
            context.view_layer.update()             # so child jiggle bones see it
            pb.keyframe_insert('rotation_euler', frame=f)
    scene.frame_set(saved)
    if log:
        log("[jiggle] baked %d spring-bone(s): %s"
            % (len(targets), [nm for nm, _ in targets]))
    return len(targets)


# Presets for AUTHORING custom jiggle bones (bust / rear / skirt).  Values are a
# starting feel; the user tunes.  Used by the "mark as jiggle" UI.
PRESETS = {
    'BUST':  {'Tension': '25', 'Friction': '4',  'Absorption': '200000',
              'MovementMultiplier': '0,20,30', 'MaxRotation': '0,0,0.5',
              'MinRotation': '0,0,-0.5', 'DisplacementAxisEffect': '0,2,2',
              'InvertDisplacementEffect': '0,1,0', 'UseDisplacementOnlyAtRest': '0,0,0'},
    'REAR':  {'Tension': '30', 'Friction': '5',  'Absorption': '200000',
              'MovementMultiplier': '0,15,25', 'MaxRotation': '0,0,0.4',
              'MinRotation': '0,0,-0.4', 'DisplacementAxisEffect': '0,2,2',
              'InvertDisplacementEffect': '0,1,0', 'UseDisplacementOnlyAtRest': '0,0,0'},
    'SKIRT': {'Tension': '30', 'Friction': '0.3', 'Absorption': '200000',
              'MovementMultiplier': '0,-0.2,15', 'MaxRotation': '0,0,0.8',
              'MinRotation': '0,0,-0.3', 'DisplacementAxisEffect': '0,2,2',
              'InvertDisplacementEffect': '0,2,2', 'UseDisplacementOnlyAtRest': '0,0,0'},
}

# ============================================================
# JIGGLE AUTHORING — per-bone settings, bake/clear/inject operators,
# and the UI panel (merged from the former jiggle_authoring.py).
# ============================================================

# Recursion guard for the multi-select propagation update callbacks.
_SUSPEND = False

# Fields copied bone->bone when editing with multiple bones selected.
_PROP_FIELDS = ("is_jiggle", "tension", "friction", "strength", "swing_limit",
                "mult", "axis", "invert", "absorption")


def _propagate(self, context):
    """Copy this (active) bone's jiggle params to every other selected bone."""
    global _SUSPEND
    if _SUSPEND:
        return
    apb = context.active_pose_bone
    if apb is None or apb.xbg_jiggle != self:
        return
    sel = context.selected_pose_bones or []
    if len(sel) <= 1:
        return
    _SUSPEND = True
    try:
        for pb in sel:
            if pb is apb:
                continue
            dst = pb.xbg_jiggle
            for k in _PROP_FIELDS:
                setattr(dst, k, getattr(self, k))
    finally:
        _SUSPEND = False


def _on_preset(self, context):
    """Load a preset's values into this bone's params (and propagate)."""
    if self.preset == 'CUSTOM':
        return
    p = PRESETS[self.preset]
    global _SUSPEND
    _SUSPEND = True
    try:
        self.tension = float(p['Tension'])
        self.friction = float(p['Friction'])
        self.strength = 3.0
        mn = _vec3(p['MinRotation']); mx = _vec3(p['MaxRotation'])
        lim = max(max(abs(v) for v in mn), max(abs(v) for v in mx)) or 0.5
        self.swing_limit = lim
        self.mult = _vec3(p['MovementMultiplier'])
        self.axis = _vec3(p['DisplacementAxisEffect'])
        self.invert = _vec3(p['InvertDisplacementEffect'])
        self.absorption = float(p.get('Absorption', '200000'))
    finally:
        _SUSPEND = False
    _propagate(self, context)


class XBGJiggleBoneSettings(bpy.types.PropertyGroup):
    is_jiggle: bpy.props.BoolProperty(
        name="Jiggle Bone", default=False, update=_propagate,
        description="Mark this bone as a procedural jiggle bone (bust / rear / "
                    "pouch / skirt). Marked bones are baked by Preview and "
                    "written into the patched proceduralbones.xml on inject")
    preset: bpy.props.EnumProperty(
        name="Preset",
        items=[('BUST', "Bust", "Soft, medium swing"),
               ('REAR', "Rear", "Heavier, shorter swing"),
               ('SKIRT', "Skirt", "Light, floaty, low friction"),
               ('CUSTOM', "Custom", "Hand-tuned values")],
        default='CUSTOM', update=_on_preset)
    # --- physical feel (sliders) ---
    tension: bpy.props.FloatProperty(
        name="Tension", default=25.0, min=0.1, max=200.0, update=_propagate,
        description="Spring stiffness — higher = snappier, follows the body more")
    friction: bpy.props.FloatProperty(
        name="Friction", default=5.0, min=0.0, max=60.0, update=_propagate,
        description="Damping — higher = settles faster, lower = floppier/bouncier")
    strength: bpy.props.FloatProperty(
        name="Strength", default=3.0, min=0.0, max=20.0, update=_propagate,
        description="Swing amount — amplifies the spring lag (saturates smoothly)")
    swing_limit: bpy.props.FloatProperty(
        name="Swing Limit", subtype='ANGLE',
        default=math.radians(28.0), min=0.0, max=math.radians(120.0),
        update=_propagate,
        description="Maximum bend the jiggle can reach (soft-limited, no snap)")
    # --- engine params for proceduralbones.xml (preset-driven, not slidered) ---
    mult: bpy.props.FloatVectorProperty(name="MovementMultiplier", size=3,
                                         default=(0.0, 20.0, 30.0))
    axis: bpy.props.FloatVectorProperty(name="DisplacementAxisEffect", size=3,
                                        default=(0.0, 2.0, 2.0))
    invert: bpy.props.FloatVectorProperty(name="InvertDisplacementEffect", size=3,
                                          default=(0.0, 1.0, 0.0))
    absorption: bpy.props.FloatProperty(name="Absorption", default=200000.0)


def _bone_defs(pbones):
    """Build the bake `defs` dict from each bone's xbg_jiggle settings."""
    defs = {}
    for pb in pbones:
        j = pb.xbg_jiggle
        lim = j.swing_limit
        defs[pb.name] = {
            'tension': j.tension, 'friction': j.friction,
            'min_rot': (0.0, 0.0, -lim), 'max_rot': (0.0, 0.0, lim),
            'axis': tuple(j.axis), 'invert': tuple(j.invert), 'mult': tuple(j.mult),
        }
    return defs


def _jiggle_targets(context):
    """Selected pose bones marked as jiggle; fall back to all selected."""
    sel = context.selected_pose_bones or []
    marked = [pb for pb in sel if pb.xbg_jiggle.is_jiggle]
    return marked or sel


class XBG_OT_BakeJiggle(bpy.types.Operator):
    """Bake the spring-bone jiggle preview onto the selected (jiggle) bones
    using their per-bone slider settings."""
    bl_idname = "xbg.bake_jiggle"
    bl_label = "Bake Jiggle Preview"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, ctx):
        arm = ctx.active_object
        if arm is None or arm.type != 'ARMATURE' or ctx.mode != 'POSE':
            self.report({'ERROR'}, "Enter Pose mode on the armature first")
            return {'CANCELLED'}
        targets = _jiggle_targets(ctx)
        if not targets:
            self.report({'ERROR'}, "Select the bone(s) to jiggle in Pose mode")
            return {'CANCELLED'}
        # Strength is per-bone; bake_jiggle takes a single strength, so bake in
        # groups keyed by strength to honour each bone's slider.
        from collections import defaultdict
        groups = defaultdict(list)
        for pb in targets:
            groups[round(pb.xbg_jiggle.strength, 4)].append(pb)
        total = 0
        for strength, pbs in groups.items():
            defs = _bone_defs(pbs)
            total += bake_jiggle(ctx, arm, defs, ctx.scene.frame_start,
                                 ctx.scene.frame_end, strength=strength, log=print)
        self.report({'INFO'}, f"Baked jiggle on {total} bone(s)")
        return {'FINISHED'}


class XBG_OT_ClearJiggle(bpy.types.Operator):
    """Remove baked jiggle keyframes from the selected bones and reset them."""
    bl_idname = "xbg.clear_jiggle"
    bl_label = "Clear Jiggle Keys"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, ctx):
        arm = ctx.active_object
        if arm is None or arm.type != 'ARMATURE':
            self.report({'ERROR'}, "Select the armature first")
            return {'CANCELLED'}
        sel = ctx.selected_pose_bones or []
        if not sel:
            self.report({'ERROR'}, "Select the bone(s) in Pose mode")
            return {'CANCELLED'}
        act = arm.animation_data.action if arm.animation_data else None
        removed = 0
        for pb in sel:
            dp = 'pose.bones["%s"].rotation_euler' % pb.name
            if act:
                for fc in [f for f in act.fcurves if f.data_path == dp]:
                    act.fcurves.remove(fc)
                    removed += 1
            pb.rotation_euler = (0.0, 0.0, 0.0)
        self.report({'INFO'}, f"Cleared {removed} jiggle curve(s)")
        return {'FINISHED'}


# ---------------------------------------------------------------------------
# Patch inject — skeleton splice + proceduralbones.xml, into the patch folder
# ---------------------------------------------------------------------------

def _fmt_num(v):
    """'30' for whole numbers, else a trimmed float — matching the game's style."""
    v = float(v)
    return str(int(round(v))) if abs(v - round(v)) < 1e-6 else ("%g" % v)


def _fmt_vec(v):
    return ",".join(_fmt_num(x) for x in v)


def _world_mats_from_bones(bones):
    """Replicate the importer's local->world chain (object space, pre FLIP_Z)."""
    import mathutils
    wm = [None] * len(bones)
    for i, b in enumerate(bones):
        x, y, z, w = b['quat']                       # LKS stores XYZW
        q = mathutils.Quaternion((w, x, y, z))       # Blender wants WXYZ
        local = mathutils.Matrix.Translation(b['pos']) @ q.to_matrix().to_4x4()
        p = b['parent']
        wm[i] = (wm[p] @ local) if (p != -1 and wm[p] is not None) else local
    return wm


def _patch_proceduralbones(src_path, dst_path, pawn_type, bone_settings, log=None):
    """Copy/decode src proceduralbones, upsert a <Bone> per jiggle bone, write
    binary to dst.  `bone_settings` = {bone_name: PoseBone.xbg_jiggle}."""
    import xml.etree.ElementTree as ET
    from . import proceduralbones as pbmod

    data = open(src_path, 'rb').read()
    if data[:3] == b'\x00\x00\xff':
        root = pbmod.decode(data)
    else:
        root = ET.fromstring(data.decode('utf-8-sig'))

    pawn = None
    for p in root.findall('PawnType'):
        if p.get('type') == pawn_type:
            pawn = p; break
    if pawn is None:
        pawn = ET.SubElement(root, 'PawnType'); pawn.set('type', pawn_type)

    existing = {b.get('Name'): b for b in pawn.findall('Bone')}
    for name, j in bone_settings.items():
        lim = j.swing_limit
        mn = [0.0, 0.0, 0.0]; mx = [0.0, 0.0, 0.0]
        for a in range(3):
            if abs(j.axis[a]) > 1e-6:
                mn[a] = -lim; mx[a] = lim
        attrs = {
            'Name': name,
            'MaxRotation': _fmt_vec(mx), 'MinRotation': _fmt_vec(mn),
            'DisplacementAxisEffect': _fmt_vec(j.axis),
            'InvertDisplacementEffect': _fmt_vec(j.invert),
            'UseDisplacementOnlyAtRest': '0,0,0',
            'Tension': _fmt_num(j.tension), 'Friction': _fmt_num(j.friction),
            'Absorption': _fmt_num(j.absorption),
            'MovementMultiplier': _fmt_vec(j.mult),
            'MaxRotBoneName': '', 'MinRotBoneName': '',
        }
        el = existing.get(name) or ET.SubElement(pawn, 'Bone')
        for k, v in attrs.items():
            el.set(k, v)
        if log:
            log("[pbones] %s %r" % ("update" if name in existing else "add", name))

    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    pbmod.encode_to_file(root, dst_path)


class XBG_OT_InjectJigglePatch(bpy.types.Operator):
    """Write a patched skeleton (with the new jiggle bones spliced in) and a
    patched proceduralbones.xml into the Patch Folder, mirroring their relative
    path under the Data Folder.  Non-destructive: originals are never touched."""
    bl_idname = "xbg.inject_jiggle_patch"
    bl_label = "Inject Jiggle Patch"
    bl_options = {'REGISTER'}

    skeleton_path: bpy.props.StringProperty(
        name="Master .skeleton", subtype='FILE_PATH', default="",
        description="The character's skeleton INSIDE the Data Folder (its "
                    "relative path is mirrored into the Patch Folder)")
    pbones_path: bpy.props.StringProperty(
        name="proceduralbones.xml", subtype='FILE_PATH', default="",
        description="Blank = auto-find databases/baltazar/proceduralbones.xml "
                    "under the Data Folder")
    pawn_type: bpy.props.StringProperty(name="Pawn Type", default="corp")

    def invoke(self, ctx, ev):
        from ..Core.prefs import get_prefs
        ds = ctx.scene.xbg_debug_settings
        if not self.skeleton_path and getattr(ds, 'mab_skeleton_path', ''):
            self.skeleton_path = ds.mab_skeleton_path
        df = (get_prefs(ctx).data_folder or '').strip()
        if not self.pbones_path and df:
            cand = os.path.join(bpy.path.abspath(df), 'databases', 'baltazar',
                                'proceduralbones.xml')
            if os.path.isfile(cand):
                self.pbones_path = cand
        return ctx.window_manager.invoke_props_dialog(self, width=460)

    def draw(self, ctx):
        from ..Core.prefs import get_prefs
        l = self.layout
        p = get_prefs(ctx)
        if not p.patch_folder:
            l.label(text="Set a Patch Folder in addon prefs first!", icon='ERROR')
        l.prop(self, "skeleton_path")
        l.prop(self, "pbones_path")
        l.prop(self, "pawn_type")
        l.label(text="Originals are copied + patched into the Patch Folder.",
                icon='INFO')

    def execute(self, ctx):
        from ..Core.prefs import get_prefs
        from .import_lks_avatar import parse_lks_file
        from . import export_lks

        arm = ctx.active_object
        if arm is None or arm.type != 'ARMATURE':
            self.report({'ERROR'}, "Select the armature first")
            return {'CANCELLED'}
        jbones = [pb for pb in arm.pose.bones if pb.xbg_jiggle.is_jiggle]
        if not jbones:
            self.report({'ERROR'}, "No bones marked as jiggle (mark them first)")
            return {'CANCELLED'}

        p = get_prefs(ctx)
        df = bpy.path.abspath((p.data_folder or '').strip())
        patch = bpy.path.abspath((p.patch_folder or '').strip())
        if not patch:
            self.report({'ERROR'}, "Set a Patch Folder in addon preferences")
            return {'CANCELLED'}
        skel = bpy.path.abspath(self.skeleton_path.strip()) if self.skeleton_path else ''
        if not (skel and os.path.isfile(skel)):
            self.report({'ERROR'}, "Pick the master .skeleton (inside the Data Folder)")
            return {'CANCELLED'}
        pbones_src = bpy.path.abspath(self.pbones_path.strip()) if self.pbones_path else ''
        if not (pbones_src and os.path.isfile(pbones_src)):
            self.report({'ERROR'}, "Pick proceduralbones.xml, or set the Data Folder")
            return {'CANCELLED'}

        # Relative paths (so the patch mirrors the in-game layout).
        def _rel(path, default):
            if df and os.path.normcase(path).startswith(os.path.normcase(df)):
                return os.path.relpath(path, df)
            return default
        skel_rel = _rel(skel, os.path.basename(skel))
        pbones_rel = _rel(pbones_src,
                          os.path.join('databases', 'baltazar', 'proceduralbones.xml'))

        # Parse original skeleton; build name->world matrix and the existing set.
        bones = parse_lks_file(skel)
        name_to_idx = {b['name']: b['idx'] for b in bones}
        wm = _world_mats_from_bones(bones)
        name_to_wm = {b['name']: wm[b['idx']] for b in bones}

        new_specs = []
        bone_settings = {}
        skipped = []
        for pb in jbones:
            bone_settings[pb.name] = pb.xbg_jiggle
            if pb.name in name_to_idx:
                continue                                  # existing bone — pbones only
            parent = pb.parent
            if parent is None or parent.name not in name_to_wm:
                skipped.append(pb.name)
                continue
            local = name_to_wm[parent.name].inverted() @ pb.bone.matrix_local
            q = local.to_quaternion()
            new_specs.append({
                'name': pb.name, 'parent_name': parent.name,
                'pos': tuple(local.translation),
                'quat': (q.x, q.y, q.z, q.w),             # LKS XYZW
            })

        if skipped:
            self.report({'WARNING'}, "Skipped (parent not in skeleton): %s"
                        % ", ".join(skipped))

        # 1) Skeleton: copy + splice into the patch folder.
        skel_dst = os.path.join(patch, skel_rel)
        os.makedirs(os.path.dirname(skel_dst), exist_ok=True)
        orig = open(skel, 'rb').read()
        out = export_lks.splice_bones(orig, new_specs, log=print) if new_specs else orig
        with open(skel_dst, 'wb') as f:
            f.write(out)

        # 2) proceduralbones.xml: copy + upsert entries for ALL marked bones.
        pbones_dst = os.path.join(patch, pbones_rel)
        _patch_proceduralbones(pbones_src, pbones_dst, self.pawn_type,
                               bone_settings, log=print)

        self.report({'INFO'},
            "Patched: +%d new bone(s) -> %s  |  %d proceduralbones entr(y/ies) -> %s"
            % (len(new_specs), os.path.relpath(skel_dst, patch),
               len(bone_settings), os.path.relpath(pbones_dst, patch)))
        return {'FINISHED'}


class XBG_PT_JigglePanel(bpy.types.Panel):
    """Author + preview procedural jiggle bones."""
    bl_label = "Jiggle Bones"
    bl_idname = "OBJECT_PT_xbg_jiggle"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "XBG Import"
    bl_parent_id = "OBJECT_PT_xbg_avatar"
    bl_options = {'DEFAULT_CLOSED'}

    @classmethod
    def poll(cls, ctx):
        return ctx.scene.xbg_debug_settings.advanced_mode

    def draw_header(self, ctx):
        self.layout.label(icon='PHYSICS')

    def draw(self, ctx):
        l = self.layout
        arm = ctx.active_object
        if arm is None or arm.type != 'ARMATURE':
            l.label(text="Select an armature", icon='INFO')
            return
        if ctx.mode != 'POSE':
            l.label(text="Enter Pose mode to edit bones", icon='INFO')
            l.operator("xbg.preview_jiggle", icon='PHYSICS',
                       text="Preview from proceduralbones.xml")
            return
        apb = ctx.active_pose_bone
        if apb is None:
            l.label(text="Select a bone", icon='BONE_DATA')
            return

        j = apb.xbg_jiggle
        sel = ctx.selected_pose_bones or []
        n = len(sel)

        l.prop(j, "is_jiggle", text="Mark as Jiggle Bone", icon='CHECKMARK')
        if n > 1:
            l.label(text="Editing applies to %d selected bones" % n, icon='GROUP_BONE')

        col = l.column()
        col.enabled = j.is_jiggle
        col.prop(j, "preset")
        box = col.box()
        box.label(text="Physical feel:", icon='FORCE_HARMONIC')
        box.prop(j, "tension", slider=True)
        box.prop(j, "friction", slider=True)
        box.prop(j, "strength", slider=True)
        box.prop(j, "swing_limit", slider=True)

        l.separator()
        row = l.row(align=True)
        row.operator("xbg.bake_jiggle", icon='PLAY', text="Bake Preview")
        row.operator("xbg.clear_jiggle", icon='TRASH', text="Clear")
        l.label(text="Play the timeline to see the jiggle", icon='INFO')

        l.separator()
        box = l.box()
        n_marked = sum(1 for pb in arm.pose.bones if pb.xbg_jiggle.is_jiggle)
        box.label(text="Inject to game (%d jiggle bone(s)):" % n_marked,
                  icon='EXPORT')
        col = box.column()
        col.enabled = n_marked > 0
        col.operator("xbg.inject_jiggle_patch", icon='MODIFIER',
                     text="Inject Jiggle Patch")
        box.label(text="Writes patched skeleton + proceduralbones.xml",
                  icon='INFO')
