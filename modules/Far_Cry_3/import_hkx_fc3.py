"""Far Cry 3 / Far Cry 4 .hkx collision importer (32-bit Havok packfiles).

FC3 ships hk_2010.2.0-r1, FC4 hk_2012.2.0-r1 — BOTH 32-bit little-endian
binary packfiles (bytesInPointer=4) behind a 16/20-byte Dunia wrapper (the
packfile magic 57E0E057 10C0C010 is found by scan, same trick as the WD1
reader — which is 64-bit-only and explicitly rejects these files).

Object graph (fanceiling_01.hkx + bear_ragdoll.hkx ground truth):
  hkpRigidBody → shape tree of:
    hkpBoxShape              halfExtents vec3 @ +32   (radius @ +16)
    hkpCapsuleShape          vertexA @ +32, vertexB @ +48, radius @ +16
    hkpSphereShape           radius @ +16
    hkpConvexTranslateShape  child ptr @ +24 (global fixup), translation @ +32
    hkpConvexTransformShape  child ptr @ +24, rotation 3×vec4 cols @ +32/48/64,
                             translation @ +80
    hkpListShape             child-info array ptr @ +24 (local fixup),
                             16-byte entries, shape ptr per entry (global fixup)
    hkpMoppBvTreeShape       wrapped child shape via global fixup (pass through)
    hkpStorageExtendedMeshShape / …MeshSubpartStorage:
                             vertices  hkArray @ +8  (vec4 each)
                             indices16 hkArray @ +32 (u16, FOUR per triangle —
                             Havok "degenerate strip" storage; 4th is padding)
  Rigid-body world transform: a 4×vec4 block (3 orthonormal rotation columns
  + translation) located by SCAN inside the body object (its fixed offset
  differs between hk2010/2012 — the orthonormality test is version-proof).

Primitive shapes import as real Blender geometry (boxes/capsules/spheres as
mesh primitives); storage meshes import verbatim.  MOPP bitstreams are never
decoded — unnecessary, the underlying storage mesh has plain triangles.
"""

import math
import os
import struct

try:
    import bpy
    import mathutils
except Exception:
    bpy = None
    mathutils = None

from ..Core.debug import VerboseLogger as vlog

HAVOK_MAGIC = bytes.fromhex('57e0e05710c0c010')


class Hkx32Error(Exception):
    pass


class Hkx32:
    """Minimal 32-bit Havok 2010/2012 packfile reader (fixup tables only)."""

    def __init__(self, path):
        raw = open(path, 'rb').read()
        idx = raw.find(HAVOK_MAGIC)
        if idx < 0:
            raise Hkx32Error("not a Havok packfile: %s" % path)
        d = raw[idx:]
        self.d = d
        if d[16] != 4:
            raise Hkx32Error(
                "not a 32-bit Havok packfile (bytesInPointer=%d) — use the "
                "Watch Dogs (64-bit) or Avatar (Havok 5.5) reader" % d[16])
        self.version = d[40:56].split(b'\0')[0].decode('latin-1')

        num_sections = struct.unpack_from('<i', d, 20)[0]
        self.sections = {}
        for i in range(num_sections):
            off = 64 + i * 48
            tag = d[off:off + 19].split(b'\0')[0].decode('latin-1')
            vals = struct.unpack_from('<7i', d, off + 20)
            self.sections[tag] = dict(zip(
                ('abs', 'local', 'global', 'virtual', 'exports',
                 'imports', 'end'), vals))

        cn = self.sections['__classnames__']
        dat = self.sections['__data__']
        self.base = dat['abs']
        self.data_size = dat['local']          # object data ends here

        def classname(off):
            e = d.index(b'\0', cn['abs'] + off)
            return d[cn['abs'] + off:e].decode('latin-1')

        self.objects = {}                      # data-rel offset -> class name
        p = self.base + dat['virtual']
        while p + 12 <= self.base + dat['exports']:
            o, s, c = struct.unpack_from('<3i', d, p); p += 12
            if o != -1:
                self.objects[o] = classname(c)

        self.local = {}                        # from -> to (data-relative)
        p = self.base + dat['local']
        while p + 8 <= self.base + dat['global']:
            f, t = struct.unpack_from('<2i', d, p); p += 8
            if f != -1:
                self.local[f] = t

        self.glob = {}                         # from -> target object offset
        p = self.base + dat['global']
        while p + 12 <= self.base + dat['virtual']:
            f, s, t = struct.unpack_from('<3i', d, p); p += 12
            if f != -1:
                self.glob[f] = t

        self._obj_ends = {}
        starts = sorted(self.objects)
        for i, o in enumerate(starts):
            self._obj_ends[o] = (starts[i + 1] if i + 1 < len(starts)
                                 else self.data_size)

    # typed reads (object-relative) -----------------------------------------
    def f32(self, obj, rel):
        return struct.unpack_from('<f', self.d, self.base + obj + rel)[0]

    def u16(self, obj, rel):
        return struct.unpack_from('<H', self.d, self.base + obj + rel)[0]

    def u32(self, obj, rel):
        return struct.unpack_from('<I', self.d, self.base + obj + rel)[0]

    def vec3(self, obj, rel):
        return struct.unpack_from('<3f', self.d, self.base + obj + rel)

    def vec4(self, obj, rel):
        return struct.unpack_from('<4f', self.d, self.base + obj + rel)

    def child_obj(self, obj, rel):
        """Follow a global fixup at obj+rel -> child object offset (or None)."""
        return self.glob.get(obj + rel)

    def local_ptr(self, obj, rel):
        """Follow a local fixup at obj+rel -> data-relative offset (or None)."""
        return self.local.get(obj + rel)


# ── shape tree decoding ─────────────────────────────────────────────────────

def _decode_shape(hk, off, xform, out, depth=0):
    """Recursively decode the shape at data-rel `off`, accumulating
    (kind, params, world_matrix) entries into `out`.

    `xform` is a mathutils.Matrix world transform for this shape."""
    if depth > 16 or off is None:
        return
    cls = hk.objects.get(off)
    if cls is None:
        return

    if cls == 'hkpMoppBvTreeShape':
        # wrapped child shape: the only global fixup inside the object
        for rel in range(0, hk._obj_ends[off] - off, 4):
            child = hk.child_obj(off, rel)
            if child is not None and hk.objects.get(child, '') != 'hkpMoppCode':
                _decode_shape(hk, child, xform, out, depth + 1)
                return

    elif cls == 'hkpListShape':
        arr = hk.local_ptr(off, 24)
        n = hk.u32(off, 28)
        if arr is not None and 0 < n < 4096:
            for i in range(n):
                child = hk.glob.get(arr + i * 16)
                if child is not None:
                    _decode_shape(hk, child, xform, out, depth + 1)

    elif cls == 'hkpConvexTranslateShape':
        child = hk.child_obj(off, 24)
        t = hk.vec3(off, 32)
        m = xform @ mathutils.Matrix.Translation(t)
        _decode_shape(hk, child, m, out, depth + 1)

    elif cls == 'hkpConvexTransformShape':
        child = hk.child_obj(off, 24)
        c0 = hk.vec3(off, 32)
        c1 = hk.vec3(off, 48)
        c2 = hk.vec3(off, 64)
        t = hk.vec3(off, 80)
        m = mathutils.Matrix((
            (c0[0], c1[0], c2[0], t[0]),
            (c0[1], c1[1], c2[1], t[1]),
            (c0[2], c1[2], c2[2], t[2]),
            (0.0, 0.0, 0.0, 1.0)))
        _decode_shape(hk, child, xform @ m, out, depth + 1)

    elif cls == 'hkpBoxShape':
        out.append(('box', {'half': hk.vec3(off, 32)}, xform.copy()))

    elif cls == 'hkpCapsuleShape':
        out.append(('capsule',
                    {'a': hk.vec3(off, 32), 'b': hk.vec3(off, 48),
                     'r': hk.f32(off, 16)}, xform.copy()))

    elif cls == 'hkpSphereShape':
        out.append(('sphere', {'r': hk.f32(off, 16)}, xform.copy()))

    elif cls == 'hkpStorageExtendedMeshShape':
        # geometry lives in the SubpartStorage object(s) referenced from it;
        # simplest robust route: take every global fixup inside this object
        # that targets a SubpartStorage.
        seen = set()
        for rel in range(0, hk._obj_ends[off] - off, 4):
            child = hk.glob.get(off + rel)
            if (child is not None and child not in seen and
                    hk.objects.get(child, '').endswith('MeshSubpartStorage')):
                seen.add(child)
                _decode_storage(hk, child, xform, out)

    elif cls.endswith('MeshSubpartStorage'):
        _decode_storage(hk, off, xform, out)


def _decode_storage(hk, off, xform, out):
    """hkpStorageExtendedMeshShapeMeshSubpartStorage: vertices hkArray @ +8
    (vec4), indices16 hkArray @ +32 (u16 × 4 per triangle)."""
    vptr = hk.local_ptr(off, 8)
    vcount = hk.u32(off, 12)
    iptr = hk.local_ptr(off, 32)
    icount = hk.u32(off, 36)
    if vptr is None or iptr is None or not (0 < vcount < 2_000_000):
        return
    verts = [hk.vec3(vptr, i * 16) for i in range(vcount)]
    tris = []
    n_tri = icount // 4
    for t in range(n_tri):
        a = hk.u16(iptr, t * 8 + 0)
        b = hk.u16(iptr, t * 8 + 2)
        c = hk.u16(iptr, t * 8 + 4)
        if a < vcount and b < vcount and c < vcount and \
                a != b and b != c and a != c:
            tris.append((a, b, c))
    if verts and tris:
        out.append(('mesh', {'verts': verts, 'tris': tris}, xform.copy()))


def _find_body_transform(hk, off):
    """Locate the rigid body's world transform: scan the object for a
    16-aligned run of 4 vec4s whose first three are orthonormal columns
    (version-proof — the member offset differs between hk2010/2012)."""
    end = hk._obj_ends[off]
    Vec = mathutils.Vector
    for rel in range(0, end - off - 63, 16):
        cols = [Vec(hk.vec3(off, rel + i * 16)) for i in range(3)]
        try:
            if any(abs(c.length - 1.0) > 1e-3 for c in cols):
                continue
            if (abs(cols[0].dot(cols[1])) > 1e-3 or
                    abs(cols[0].dot(cols[2])) > 1e-3 or
                    abs(cols[1].dot(cols[2])) > 1e-3):
                continue
            if cols[0].cross(cols[1]).dot(cols[2]) < 0.5:
                continue
        except Exception:
            continue
        t = hk.vec3(off, rel + 48)
        c0, c1, c2 = cols
        return mathutils.Matrix((
            (c0[0], c1[0], c2[0], t[0]),
            (c0[1], c1[1], c2[1], t[1]),
            (c0[2], c1[2], c2[2], t[2]),
            (0.0, 0.0, 0.0, 1.0)))
    return mathutils.Matrix.Identity(4)


def _body_name(hk, off):
    """m_name: a local fixup inside the body pointing at an ASCII string."""
    end = hk._obj_ends[off]
    for rel in range(0, end - off, 4):
        tgt = hk.local.get(off + rel)
        if tgt is None:
            continue
        p = hk.base + tgt
        e = hk.d.find(b'\0', p, p + 96)
        if e > p:
            s = hk.d[p:e]
            if s and all(0x20 <= c < 0x7F for c in s):
                return s.decode('latin-1')
    return None


# ── Blender build ────────────────────────────────────────────────────────────

def _capsule_mesh(a, b, r, segments=12, rings=6):
    """Simple capsule triangle mesh between points a-b with radius r."""
    Vec = mathutils.Vector
    a = Vec(a); b = Vec(b)
    axis = b - a
    L = axis.length
    if L < 1e-9:
        axis = Vec((0, 0, 1)); L = 0.0
    z = axis.normalized() if L > 0 else Vec((0, 0, 1))
    x = z.orthogonal().normalized()
    y = z.cross(x)
    verts, faces = [], []

    def ring(center, radius, zoff):
        base = len(verts)
        for s in range(segments):
            th = 2.0 * math.pi * s / segments
            verts.append(center + (x * math.cos(th) + y * math.sin(th)) * radius
                         + z * zoff)
        return base

    rings_list = []
    for k in range(rings + 1):                      # bottom hemisphere
        phi = -math.pi / 2 + (math.pi / 2) * k / rings
        rings_list.append(ring(a, r * math.cos(phi), r * math.sin(phi)))
    for k in range(rings + 1):                      # top hemisphere
        phi = (math.pi / 2) * k / rings
        rings_list.append(ring(b, r * math.cos(phi), r * math.sin(phi)))
    for i in range(len(rings_list) - 1):
        r0, r1 = rings_list[i], rings_list[i + 1]
        for s in range(segments):
            s2 = (s + 1) % segments
            faces.append((r0 + s, r0 + s2, r1 + s2, r1 + s))
    return [tuple(v) for v in verts], faces


def load_fc3_hkx(context, filepath):
    """Import every rigid body's collision shapes. Returns created objects."""
    if bpy is None:
        raise RuntimeError("bpy unavailable")
    hk = Hkx32(filepath)
    vlog.log("[fc3 hkx] %s  version=%s  %d objects"
             % (os.path.basename(filepath), hk.version, len(hk.objects)))

    base_name = os.path.splitext(os.path.basename(filepath))[0]
    created = []
    body_i = 0
    for off, cls in sorted(hk.objects.items()):
        if cls != 'hkpRigidBody':
            continue
        shape = hk.child_obj(off, 16)
        if shape is None:
            continue
        xform = _find_body_transform(hk, off)
        shapes = []
        _decode_shape(hk, shape, xform, shapes)
        if not shapes:
            continue
        nm = _body_name(hk, off) or ("body%d" % body_i)
        body_i += 1

        for si, (kind, prm, m) in enumerate(shapes):
            name = f"{base_name}_{nm}_{kind}{si}"
            if kind == 'mesh':
                verts, faces = prm['verts'], prm['tris']
            elif kind == 'box':
                hx, hy, hz = prm['half']
                verts = [(sx * hx, sy * hy, sz * hz)
                         for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]
                faces = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1),
                         (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
            elif kind == 'capsule':
                verts, faces = _capsule_mesh(prm['a'], prm['b'], prm['r'])
            elif kind == 'sphere':
                verts, faces = _capsule_mesh((0, 0, 0), (0, 0, 0), prm['r'])
            else:
                continue
            me = bpy.data.meshes.new(name)
            me.from_pydata(verts, [], faces)
            me.update(calc_edges=True)
            ob = bpy.data.objects.new(name, me)
            ob.matrix_world = m
            ob.display_type = 'WIRE'
            context.collection.objects.link(ob)
            created.append(ob)
    if not created:
        raise Hkx32Error("no decodable collision shapes found")
    return created
