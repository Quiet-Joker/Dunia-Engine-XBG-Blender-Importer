"""LKS binary skeleton file parser and Blender armature builder.

LKS is Ubisoft's binary skeleton format used in Avatar: The Game.

File layout
-----------
  Offset  0 : magic 'LKS\\x00' (4 bytes)
  Offset  4 : format constants — three u32 (version=18, hash, unk=7)
  Offset 16 : root bone block (62 bytes)
  Offset 78 : root bone name (length-prefixed u32 LE) + 0x01 terminator
  After that : one record per non-root bone (indices 1 .. bone_count-1)
               Each record = bone block + name (length-prefixed u32 LE) + 0x00

Root bone block (62 bytes, file offset 16):
  block[0:2]   bone_count (u16)
  block[2:4]   format constant 0x1F
  block[4:8]   hash
  block[8:12]  unknown (u32, observed value 7)
  block[12:28] local rotation quaternion (4 × float32, X Y Z W)
  block[28:40] local position (3 × float32)
  block[48:50] bone_seq_idx = 1
  block[50:52] parent_idx = 0xFFFF  (no parent)
  block[58:62] name_length (u32)

Non-root bone block — standard (63 bytes):
  block[0]     0x80 constant
  block[5:9]   hash (u32)
  block[13:29] local rotation quaternion (4 × float32, X Y Z W)
  block[29:41] local position (3 × float32)
  block[45:47] bone_seq_idx (u16, sequential 1..N-1)
  block[47:49] parent_idx (u16, 0-based, 0xFFFF = no parent)
  block[49:51] first_child_idx (u16)
  block[51:53] next_sibling_idx (u16)
  block[55:59] per-bone hash (u32)
  block[59:63] name_length (u32)          ← STANDARD bones only

Non-root bone block — extended (71 bytes, certain IK / twist bones):
  Same as standard through block[58], then:
  block[59:67] 8-byte extended field
  block[67:71] name_length (u32)

Extended block detection: if block[59:63] read as u32 > 256, the block is
extended.  All positional/rotational fields (quat, pos, parent_idx) remain
at the same offsets in both variants.
"""

import struct
import math
import mathutils
import bpy

try:
    from ..Core.debug import VerboseLogger as vlog
except Exception:
    class vlog:
        enabled = False
        @staticmethod
        def log(m): pass


# ---------------------------------------------------------------------------
# LKS parser
# ---------------------------------------------------------------------------

def parse_lks_file(filepath):
    """Parse an LKS binary skeleton file.

    Returns a list of bone dicts (one per bone, index = position in list):
      {
        'idx'    : int,          # 0-based bone index
        'name'   : str,          # bone name
        'parent' : int,          # parent index, or -1 for root
        'pos'    : (x, y, z),   # local position (float32 × 3)
        'quat'   : (x, y, z, w) # local rotation (float32 × 4, XBG/LKS order)
      }

    Raises ValueError if the file is not a valid LKS skeleton.
    """
    with open(filepath, 'rb') as f:
        data = f.read()

    if len(data) < 20 or data[:3] != b'LKS':
        raise ValueError(f"Not a valid LKS file: {filepath}")

    # ── Root bone (file offset 16, block = 62 bytes) ────────────────────
    bone_count    = struct.unpack_from('<H', data, 16)[0]
    root_quat     = struct.unpack_from('<4f', data, 28)    # block[12:28]
    root_pos      = struct.unpack_from('<3f', data, 44)    # block[28:40]
    root_name_len = struct.unpack_from('<I', data, 74)[0]  # block[58:62]
    root_name     = data[78 : 78 + root_name_len].decode('latin-1')

    vlog.log(f"\n=== LKS SKELETON PARSE ===")
    vlog.log(f"  bone_count = {bone_count}")
    vlog.log(f"  root = {root_name!r}  pos={root_pos}  quat={root_quat}")

    bones = [{
        'idx': 0, 'name': root_name, 'parent': -1,
        'pos': root_pos, 'quat': root_quat,
    }]

    off = 78 + root_name_len + 1  # skip root name + 0x01 terminator

    # ── Non-root bones ───────────────────────────────────────────────────
    for i in range(1, bone_count):
        if off + 63 > len(data):
            vlog.log(f"  [LKS] WARNING: ran out of data at bone {i} (off={off})")
            break

        # Detect extended block: name_len at [59:63] is invalid (> 256)
        name_len_at_59 = struct.unpack_from('<I', data, off + 59)[0]
        if name_len_at_59 <= 256:
            # Standard 63-byte block
            name_len = name_len_at_59
            name_off = off + 63
        else:
            # Extended 71-byte block: name_len at off+67
            if off + 75 > len(data):
                vlog.log(f"  [LKS] WARNING: extended bone {i} out of bounds (off={off})")
                break
            name_len = struct.unpack_from('<I', data, off + 67)[0]
            name_off = off + 71

        if name_len > 256 or name_off + name_len > len(data):
            vlog.log(f"  [LKS] WARNING: invalid name_len={name_len} at bone {i} (off={off})")
            break

        name      = data[name_off : name_off + name_len].decode('latin-1')
        quat      = struct.unpack_from('<4f', data, off + 13)
        pos       = struct.unpack_from('<3f', data, off + 29)
        parent_id = struct.unpack_from('<H',  data, off + 47)[0]
        parent    = parent_id if parent_id != 0xFFFF else -1

        vlog.log(f"  Bone {i:3d}: {name!r:<40} parent={parent:3d}  pos={pos}  quat={quat}")

        bones.append({
            'idx': i, 'name': name, 'parent': parent,
            'pos': pos, 'quat': quat,
        })

        off = name_off + name_len + 1  # advance past name + terminator

    vlog.log(f"  Parsed {len(bones)} bones total.")
    return bones


# ---------------------------------------------------------------------------
# Blender armature builder
# ---------------------------------------------------------------------------

def _lks_quat_to_blender(quat_xyzw):
    """Convert LKS quaternion (X, Y, Z, W) → Blender Quaternion (W, X, Y, Z)."""
    x, y, z, w = quat_xyzw
    return mathutils.Quaternion((w, x, y, z))


def _is_grid_quaternion(q, tol=0.02):
    """Return True if *q* is a rotation by 0/90/180/270° around a principal axis.

    Grid quaternions have at most two non-zero components (W and at most one of
    X/Y/Z).  They are recognisable as coordinate-system flips baked into an
    asset's root bone (e.g. a plant rig whose root encodes 180° Z so the mesh
    faces the right direction in engine space), NOT as genuine bind-pose
    rotations which always have several non-zero components.

    Examples that return True
    -------------------------
      (W=1, 0, 0, 0)                 → identity  (0°)
      (W=0, 0, 0, Z=±1)             → 180° around Z  ← scorpion thistle
      (W=0, X=±1, 0, 0)             → 180° around X
      (W=±√½, 0, 0, Z=±√½)         → ±90° around Z

    Examples that return False
    --------------------------
      (W=0.95, X=0.10, Y=0.30, Z=0.02)  → pelvis bind pose (multiple axes)
    """
    nz      = sum(1 for v in (q.w, q.x, q.y, q.z) if abs(v) > tol)
    xyz_nz  = sum(1 for v in (q.x, q.y, q.z)      if abs(v) > tol)
    # A grid quat has at most 2 non-zero components, with at most 1 in the XYZ part.
    return nz <= 2 and xyz_nz <= 1


def create_lks_armature(context, bones, armature_name="LKS_Skeleton"):
    """Build a Blender armature object from a parsed LKS bone list.

    Matches the coordinate convention of the XBG importer:
      • The armature object is rotated 180° around Z so the model faces
        the same direction as an imported XBG mesh.
      • Bone tails are set by rotating the Y-axis by the bone's world
        rotation quaternion (same as XBG's create_armature).

    Parameters
    ----------
    context       : bpy.types.Context
    bones         : list returned by :func:`parse_lks_file`
    armature_name : str — name for the new armature and its object

    Returns the created armature Object.
    """
    n = len(bones)

    # ── 1. Compute world matrices (local → world via parent chain) ───────
    world_mats = [None] * n
    for i, bone in enumerate(bones):
        q         = _lks_quat_to_blender(bone['quat'])
        loc       = mathutils.Matrix.Translation(bone['pos'])
        rot       = q.to_matrix().to_4x4()
        local_mat = loc @ rot

        p = bone['parent']
        world_mats[i] = (
            world_mats[p] @ local_mat
            if (p != -1 and world_mats[p] is not None)
            else local_mat
        )

    # ── 2. Build child index lists ────────────────────────────────────────
    children = {i: [] for i in range(n)}
    for i, bone in enumerate(bones):
        if bone['parent'] != -1:
            children[bone['parent']].append(i)

    # ── 3. Create armature object ─────────────────────────────────────────
    armature = bpy.data.armatures.new(armature_name)
    arm_obj  = bpy.data.objects.new(armature_name, armature)
    context.collection.objects.link(arm_obj)

    # Compute armature object rotation.
    #
    # The standard XBG-importer convention is a 180° Z flip on the armature
    # object.  This works for most LKS files because their root bone has the
    # same bind-pose rotation as the companion XBG skeleton's root (so both
    # get 180° Z and the world positions line up).
    #
    # Exception — "grid" root rotations (0°/90°/180°/270° around a principal
    # axis): some prop/plant rigs bake a 180° Z coordinate-system flip into
    # the LKS root bone even though the XBG root has identity.  A hardcoded
    # 180° Z on the armature would then double-flip (360° = identity), leaving
    # every child bone mirrored in X/Y.  For these we compensate with
    # arm_rot = 180°Z × root_rot⁻¹, which cancels the baked rotation and
    # restores the correct orientation.
    #
    # For genuine bind-pose roots (pelvis, spine, etc.) that have complex
    # non-axis-aligned rotations, the standard 180° Z is kept unchanged so
    # the LKS skeleton continues to match the companion XBG skeleton.
    _FLIP_Z = mathutils.Quaternion((0.0, 0.0, 0.0, 1.0))   # 180° around +Z
    root_q  = _lks_quat_to_blender(bones[0]['quat'])
    if _is_grid_quaternion(root_q):
        # Root encodes an axis-aligned coordinate-system rotation (not a real
        # bind-pose).  Compensate so the net armature transform is 180° Z.
        arm_rot_q = _FLIP_Z @ root_q.inverted()
        vlog.log(f"  Root is a grid quaternion — compensating armature rotation.")
    else:
        # Root has a genuine bind-pose rotation shared with the XBG skeleton.
        arm_rot_q = _FLIP_Z
    arm_obj.rotation_euler = arm_rot_q.to_euler()
    vlog.log(f"  Armature rotation: {tuple(round(math.degrees(v),2) for v in arm_obj.rotation_euler)} deg"
             f"  (root_q={tuple(round(v,4) for v in root_q)})")

    for obj in context.selected_objects:
        obj.select_set(False)
    arm_obj.select_set(True)
    context.view_layer.objects.active = arm_obj

    bpy.ops.object.mode_set(mode='EDIT')
    edit_bones = armature.edit_bones

    # ── 4. Create all edit bones ──────────────────────────────────────────

    def _find_tail_pos(start_idx, head):
        """Return the best tail position for a bone with children.

        Algorithm — BFS level by level:
          1. At each generation, collect every non-coincident descendant.
          2. If any found at this level, return the NEAREST one (Euclidean).
          3. If all at this level are coincident, descend another generation.

        This means:
          • A bone whose first child is coincident (e.g. root 'Banshee' →
            'Banshee_Chassis', both at origin) skips the coincident child and
            points toward the nearest non-coincident grandchild ('Pelvis' at
            ≈1.27 units) instead of some far-away sibling ('Base_Right_Hand01'
            at ≈2.76 units that just happened to come first in index order).
          • A bone whose direct children are all non-coincident simply picks
            the nearest direct child — same as before.
        """
        queue = list(children[start_idx])
        seen  = {start_idx}
        while queue:
            next_level = []
            non_coinc  = []
            for ci in queue:
                if ci in seen:
                    continue
                seen.add(ci)
                p = world_mats[ci].translation
                d = (p - head).length
                if d > 0.001:
                    non_coinc.append((d, p.copy()))
                else:
                    next_level.extend(children[ci])
            if non_coinc:
                # Sort by distance ONLY — never fall through to comparing
                # the mathutils.Vector on a distance tie (unreliable).
                non_coinc.sort(key=lambda t: t[0])
                return non_coinc[0][1]
            queue = next_level            # all coincident at this level → go deeper
        return None

    eb_map = {}
    for i, bone in enumerate(bones):
        eb      = edit_bones.new(bone['name'])
        head    = world_mats[i].translation.copy()
        eb.head = head

        # Tail assignment:
        #  • Bones with children  → BFS-level nearest (see _find_tail_pos).
        #  • Leaf bones           → extend in the parent→self direction.
        child_list = children[i]
        if child_list:
            tail_pos = _find_tail_pos(i, head)
            eb.tail  = tail_pos if tail_pos is not None \
                       else head + mathutils.Vector((0.0, 0.05, 0.0))
        else:
            p = bone['parent']
            if p != -1:
                parent_head = world_mats[p].translation
                diff = head - parent_head
                if diff.length > 0.001:
                    eb.tail = head + diff.normalized() * 0.05
                else:
                    eb.tail = head + mathutils.Vector((0.0, 0.05, 0.0))
            else:
                eb.tail = head + mathutils.Vector((0.0, 0.05, 0.0))

        eb_map[i] = eb
        vlog.log(f"  EditBone {i:3d} {bone['name']!r:<40} head={tuple(round(v,4) for v in head)}")

    # ── 5. Set parent relationships ───────────────────────────────────────
    for i, bone in enumerate(bones):
        p = bone['parent']
        if p != -1 and p in eb_map:
            eb_map[i].parent       = eb_map[p]
            eb_map[i].use_connect  = False

    bpy.ops.object.mode_set(mode='OBJECT')
    vlog.log(f"  Armature {armature_name!r} created with {n} bones.")
    return arm_obj
