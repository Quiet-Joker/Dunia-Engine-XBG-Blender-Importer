"""Native Avatar .hkx collision import / patch-export — no Havok tools.

Avatar's .hkx files are Havok 5.5.0-r1 BINARY PACKFILES (32-bit LE,
layout 04 01 00 01) behind the game's 16-byte wrapper header.  The
packfile's __types__ section is empty but __data__ carries full fixup
tables, which is everything needed to read it:

    virtual fixups : object offset -> class name   (object enumeration)
    global fixups  : pointer field -> target object (shape graph edges)
    local fixups   : array field   -> array data    (verts, indices, names)

Reverse-engineered 2026-06-11 on the shipped atv/samson files; member
offsets below were confirmed byte-by-byte against those files.

EDIT MODEL — patch, don't re-serialize:
Collision edits are float changes (box half-extents, wrapper transforms,
convex/mesh vertex positions).  Export therefore copies the ORIGINAL file
and overwrites only those floats at their recorded offsets — structure,
MOPP bytecode and unknown fields survive untouched.  (The hkpMoppCode
bytecode is geometry-dependent: large geometry changes may make culling
stale, same caveat as the old XML workflow.)

Confirmed Havok 5.5 32-bit member offsets (object-relative):
    hkpShape base       : +0x10 f32 radius (convex shapes)
    hkpBoxShape         : +0x20 hkVector4 halfExtents          (size 0x30)
    hkpConvexTranslateShape : +0x20 hkVector4 translation      (size 0x30)
    hkpConvexTransformShape : +0x20/0x30/0x40 rotation columns,
                              +0x50 translation                (size 0x60)
    hkpConvexVerticesShape  : +0x20 aabbHalfExtents, +0x30 aabbCenter,
                              +0x40 m_rotatedVertices array (FourVectors),
                              +0x4c u32 numVertices
    hkpListShape        : +0x18 childInfo array (16-byte entries,
                          shape ptr at entry+0 via global fixup)
    hkpRigidBody        : +0x10 shape ptr (collidable.cdBody),
                          +0xE0 hkTransform (3 rotation column vec4s +
                          translation vec4 at +0x110),
                          +0x120/+0x130 sweptTransform COM positions,
                          +0x140/+0x150 sweptTransform quats (xyzw),
                          +0x160 centerOfMassLocal
    ...MeshSubpartStorage : +0x08 m_vertices array (hkVector4 each),
                            +0x14 m_indices16 array (4 u16 per triangle:
                            a, b, c, pad)
"""

import os
import struct

try:
    import bpy
    import bmesh
    import mathutils
except ImportError:          # standalone analysis
    bpy = None
    bmesh = None
    mathutils = None

HAVOK_MAGIC = b'\x57\xe0\xe0\x57\x10\xc0\xc0\x10'

_SHAPE_CLASSES = {
    'hkpBoxShape', 'hkpConvexTranslateShape', 'hkpConvexTransformShape',
    'hkpConvexVerticesShape', 'hkpListShape', 'hkpMoppBvTreeShape',
    'hkpStorageExtendedMeshShape', 'hkpExtendedMeshShape', 'hkpSphereShape',
    'hkpCapsuleShape', 'hkpCylinderShape', 'hkpTriangleShape',
}


class HkxFile:
    """Parsed Havok 5.5 packfile (read-only view + patch helpers)."""

    def __init__(self, path, raw=None):
        self.path = path
        self.raw = raw if raw is not None else open(path, 'rb').read()
        # The game wrapper is 16 bytes before the Havok magic; some tools
        # strip it, so detect rather than assume.
        if self.raw[:8] == HAVOK_MAGIC:
            self.wrapper = 0
        elif self.raw[16:24] == HAVOK_MAGIC:
            self.wrapper = 16
        else:
            raise ValueError("not a Havok 5.5 packfile: %s" % path)
        d = self.raw[self.wrapper:]
        self.d = d

        num_sections, = struct.unpack_from('<i', d, 20)
        ver = d[40:56].split(b'\0')[0].decode('latin-1')
        if not ver.startswith('Havok-5.'):
            print("[HKX] warning: untested Havok version %r" % ver)
        self.version = ver

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
        self.data_abs = dat['abs']
        self._cn_abs = cn['abs']

        def classname(off):
            e = d.index(b'\0', self._cn_abs + off)
            return d[self._cn_abs + off:e].decode('latin-1')

        # objects: data-section offset -> class name (virtual fixups)
        self.objects = {}
        p = self.data_abs + dat['virtual']
        while p + 12 <= self.data_abs + dat['exports']:
            o, s, c = struct.unpack_from('<3i', d, p); p += 12
            if o != -1:
                self.objects[o] = classname(c)
        self._sorted = sorted(self.objects)
        self.obj_size = {}
        for i, o in enumerate(self._sorted):
            nxt = (self._sorted[i + 1] if i + 1 < len(self._sorted)
                   else dat['local'])
            self.obj_size[o] = nxt - o

        # local fixups: from-offset -> to-offset (both data-relative)
        self.local = {}
        p = self.data_abs + dat['local']
        while p + 8 <= self.data_abs + dat['global']:
            f, t = struct.unpack_from('<2i', d, p); p += 8
            if f != -1:
                self.local[f] = t
        # global fixups: from-offset -> target object offset
        # (global_entries keeps the target-section index for re-emission)
        self.globals = {}
        self.global_entries = {}
        p = self.data_abs + dat['global']
        while p + 12 <= self.data_abs + dat['virtual']:
            f, s, t = struct.unpack_from('<3i', d, p); p += 12
            if f != -1:
                self.globals[f] = t
                self.global_entries[f] = (s, t)
        # virtual fixup entries verbatim (for table re-emission)
        self.virtual_entries = []
        p = self.data_abs + dat['virtual']
        while p + 12 <= self.data_abs + dat['exports']:
            o, s, c = struct.unpack_from('<3i', d, p); p += 12
            if o != -1:
                self.virtual_entries.append((o, s, c))
        # classnames map: class name -> offset of the NAME within the
        # classnames section (the offset virtual fixups use)
        self.classnames = {}
        p = self._cn_abs
        cn_end = self._cn_abs + cn['end']
        while p + 5 < cn_end and d[p + 4:p + 5] == b'\x09':
            e = d.index(b'\0', p + 5)
            self.classnames[d[p + 5:e].decode('latin-1')] = \
                p + 5 - self._cn_abs
            p = e + 1
        self._cn_used = p - self._cn_abs   # actual bytes used (<= cn end)

    # ── primitives ──────────────────────────────────────────────────────
    def file_off(self, data_off):
        """Data-section offset -> absolute offset in self.raw."""
        return self.wrapper + self.data_abs + data_off

    def f32s(self, data_off, n):
        return struct.unpack_from('<%df' % n,
                                  self.d, self.data_abs + data_off)

    def u32(self, data_off):
        return struct.unpack_from('<I', self.d, self.data_abs + data_off)[0]

    def u16s(self, data_off, n):
        return struct.unpack_from('<%dH' % n,
                                  self.d, self.data_abs + data_off)

    def obj_globals(self, obj_off):
        """Global fixups whose pointer field lies inside this object,
        sorted by field offset -> [(field_off, target_obj_off)]."""
        end = obj_off + self.obj_size[obj_off]
        return sorted((f, t) for f, t in self.globals.items()
                      if obj_off <= f < end)

    def obj_name(self, obj_off):
        """hkpWorldObject m_name: a local fixup inside the object header
        whose target is a printable NUL-terminated string."""
        end = obj_off + min(self.obj_size[obj_off], 0x80)
        for f, t in self.local.items():
            if obj_off <= f < end:
                a = self.data_abs + t
                e = self.d.index(b'\0', a)
                s = self.d[a:e]
                if 0 < len(s) < 96 and all(0x20 <= c < 0x7F for c in s):
                    return s.decode('latin-1')
        return None

    # ── shape graph ─────────────────────────────────────────────────────
    def shape_children(self, obj_off):
        """Child SHAPE objects pointed to from inside this object."""
        return [(f, t) for f, t in self.obj_globals(obj_off)
                if self.objects.get(t) in _SHAPE_CLASSES]

    def walk_shape(self, obj_off, xform=None, out=None):
        """Flatten the shape graph into leaf records:
        [{'class', 'off', 'xform' (4x4 rows tuple or None), ...geometry,
          'patch': {member: file_offset}}]."""
        if out is None:
            out = []
        if mathutils is not None:
            ident = mathutils.Matrix.Identity(4)
            xf = xform if xform is not None else ident
        else:
            xf = xform
        cls = self.objects.get(obj_off)

        if cls == 'hkpBoxShape':
            he = self.f32s(obj_off + 0x20, 3)
            out.append({'class': cls, 'off': obj_off, 'xform': xf,
                        'half_extents': he,
                        'radius': self.f32s(obj_off + 0x10, 1)[0],
                        'patch': {'half_extents':
                                  self.file_off(obj_off + 0x20)}})

        elif cls == 'hkpConvexVerticesShape':
            n = self.u32(obj_off + 0x4c)
            arr = self.local.get(obj_off + 0x40)
            verts = []
            if arr is not None:
                nchunks = (n + 3) // 4
                for c in range(nchunks):
                    v = self.f32s(arr + c * 48, 12)
                    for j in range(min(4, n - c * 4)):
                        verts.append((v[j], v[4 + j], v[8 + j]))
            out.append({'class': cls, 'off': obj_off, 'xform': xf,
                        'verts': verts, 'verts_off': arr,
                        'patch': {'rotated_vertices':
                                  None if arr is None
                                  else self.file_off(arr),
                                  'aabb_half': self.file_off(obj_off + 0x20),
                                  'aabb_center':
                                  self.file_off(obj_off + 0x30)}})

        elif cls in ('hkpStorageExtendedMeshShape', 'hkpExtendedMeshShape'):
            sto = next((t for f, t in self.obj_globals(obj_off)
                        if self.objects.get(t, '').endswith(
                            'MeshSubpartStorage')), None)
            verts, tris, voff = [], [], None
            if sto is not None:
                voff = self.local.get(sto + 0x08)
                nv = self.u32(sto + 0x0c)
                ioff = self.local.get(sto + 0x14)
                ni = self.u32(sto + 0x18)
                if voff is not None:
                    for i in range(nv):
                        x, y, z, _ = self.f32s(voff + i * 16, 4)
                        verts.append((x, y, z))
                if ioff is not None:
                    idx = self.u16s(ioff, ni)
                    for i in range(0, ni - 3, 4):
                        a, b, c = idx[i], idx[i + 1], idx[i + 2]
                        if a != b and b != c and a != c:
                            tris.append((a, b, c))
            out.append({'class': cls, 'off': obj_off, 'xform': xf,
                        'verts': verts, 'tris': tris, 'verts_off': voff,
                        'patch': {'vertices':
                                  None if voff is None
                                  else self.file_off(voff)}})

        elif cls in ('hkpCapsuleShape', 'hkpCylinderShape'):
            va = self.f32s(obj_off + 0x20, 3)
            vb = self.f32s(obj_off + 0x30, 3)
            out.append({'class': cls, 'off': obj_off, 'xform': xf,
                        'verts': [va, vb],
                        'radius': self.f32s(obj_off + 0x10, 1)[0],
                        'patch': {'capsule_a': self.file_off(obj_off + 0x20),
                                  'capsule_b': self.file_off(obj_off + 0x30),
                                  'radius': self.file_off(obj_off + 0x10)}})

        elif cls == 'hkpSphereShape':
            out.append({'class': cls, 'off': obj_off, 'xform': xf,
                        'radius': self.f32s(obj_off + 0x10, 1)[0],
                        'patch': {'radius': self.file_off(obj_off + 0x10)}})

        elif cls == 'hkpConvexTranslateShape':
            t = self.f32s(obj_off + 0x20, 3)
            m = (mathutils.Matrix.Translation(t)
                 if mathutils is not None else None)
            for f, child in self.shape_children(obj_off):
                self.walk_shape(child,
                                (xf @ m) if m is not None else None, out)
                if out:
                    out[-1]['wrap_trans_off'] = self.file_off(obj_off + 0x20)
                    out[-1]['wrap_class'] = cls

        elif cls == 'hkpConvexTransformShape':
            c0 = self.f32s(obj_off + 0x20, 3)
            c1 = self.f32s(obj_off + 0x30, 3)
            c2 = self.f32s(obj_off + 0x40, 3)
            tr = self.f32s(obj_off + 0x50, 3)
            if mathutils is not None:
                m = mathutils.Matrix((
                    (c0[0], c1[0], c2[0], tr[0]),
                    (c0[1], c1[1], c2[1], tr[1]),
                    (c0[2], c1[2], c2[2], tr[2]),
                    (0, 0, 0, 1)))
            else:
                m = None
            for f, child in self.shape_children(obj_off):
                self.walk_shape(child,
                                (xf @ m) if m is not None else None, out)
                if out:
                    out[-1]['wrap_trans_off'] = self.file_off(obj_off + 0x50)
                    out[-1]['wrap_rot_off'] = self.file_off(obj_off + 0x20)
                    out[-1]['wrap_class'] = cls

        elif cls == 'hkpListShape':
            # children live in the childInfo ARRAY (local fixup at +0x18,
            # count at +0x1c, shape ptr at entry+0) — the array may sit
            # anywhere in the data section, not inside the object's range
            arr = self.local.get(obj_off + 0x18)
            cnt = self.u32(obj_off + 0x1c)
            if arr is not None:
                for i in range(cnt):
                    t = self.globals.get(arr + 16 * i)
                    if t is not None and \
                            self.objects.get(t) in _SHAPE_CLASSES:
                        self.walk_shape(t, xf, out)
            else:
                for f, child in self.shape_children(obj_off):
                    self.walk_shape(child, xf, out)

        elif cls == 'hkpMoppBvTreeShape':
            for f, child in self.shape_children(obj_off):
                self.walk_shape(child, xf, out)

        elif cls is not None:
            # unknown shape wrapper — descend through its shape pointers
            for f, child in self.shape_children(obj_off):
                self.walk_shape(child, xf, out)
        return out

    def rigid_bodies(self):
        """[{'off', 'name', 'xform', 'shapes': [leaf records]}]"""
        out = []
        for o, cls in sorted(self.objects.items()):
            if cls != 'hkpRigidBody':
                continue
            c0 = self.f32s(o + 0xE0, 3)
            c1 = self.f32s(o + 0xF0, 3)
            c2 = self.f32s(o + 0x100, 3)
            tr = self.f32s(o + 0x110, 3)
            if mathutils is not None:
                m = mathutils.Matrix((
                    (c0[0], c1[0], c2[0], tr[0]),
                    (c0[1], c1[1], c2[1], tr[1]),
                    (c0[2], c1[2], c2[2], tr[2]),
                    (0, 0, 0, 1)))
            else:
                m = None
            shape = self.globals.get(o + 0x10)
            shapes = (self.walk_shape(shape) if shape is not None else [])
            out.append({'off': o, 'name': self.obj_name(o), 'xform': m,
                        'shapes': shapes,
                        'patch': {'transform': self.file_off(o + 0xE0),
                                  'swept_pos0': self.file_off(o + 0x120),
                                  'swept_pos1': self.file_off(o + 0x130),
                                  'swept_q0': self.file_off(o + 0x140),
                                  'swept_q1': self.file_off(o + 0x150),
                                  'com_local': self.file_off(o + 0x160)}})
        return out


# ---------------------------------------------------------------------------
# Additive rebuild — append NEW shapes to the packfile
# ---------------------------------------------------------------------------
# New shapes are serialized as hkpConvexVerticesShape (handles boxes and any
# convex geometry; vertices are baked in rigid-body space so no transform
# wrapper class is needed) and hooked into the body's hkpListShape childInfo
# array.  Arrays in a packfile are located via local fixups, so growing one
# just means writing a bigger copy at the end of the data section and
# retargeting the fixup.  The fixup tables live AFTER the data, so they are
# re-emitted; everything before the data section is byte-preserved (except
# an optional classname append into the section's padding gap).
#
# MOPP caveat: hkpMoppBvTreeShape culls children by precompiled bytecode, so
# shapes added to a MOPP-wrapped list would never be hit-tested.  When that
# wrapper is present, the rigid body's shape pointer is retargeted to the
# list directly (bypassing the MOPP) — slightly slower broad-phase, but all
# children collide.

# class signatures for appending a missing __classnames__ entry
# (extracted from the shipped atv.hkx)
_CLASS_SIGS = {
    'hkpConvexVerticesShape': 0xea645297,
    'hkpBoxShape': 0x9adfa090,
    'hkpConvexTranslateShape': 0xf510b071,
    'hkpListShape': 0xa797a56b,
}
# common 16-byte object header of every shape in these files
# (vtable placeholder, memSize/refCount, userData, type)
_SHAPE_HEAD = bytes.fromhex('00000000 00000000 0b000000 00000000'
                            .replace(' ', ''))


def _align16(n):
    return (n + 15) & ~15


def find_list_for_rb(hk, rb_off):
    """(list_off, mopp_off or None) for a rigid body, or (None, None)."""
    t = hk.globals.get(rb_off + 0x10)
    if t is None:
        return None, None
    cls = hk.objects.get(t)
    if cls == 'hkpListShape':
        return t, None
    if cls == 'hkpMoppBvTreeShape':
        for f, c in hk.shape_children(t):
            if hk.objects.get(c) == 'hkpListShape':
                return c, t
    return None, None


def rebuild_with_additions(buf, additions):
    """Append new convex shapes to a (possibly already patched) packfile.

    `buf`      : bytes of the whole file (wrapper included).
    `additions`: [{'rb_off', 'verts': [(x,y,z) in rigid-body space],
                   'planes': [(nx,ny,nz,d)], 'radius'}]
    Returns (new_file_bytes, warnings).
    """
    hk = HkxFile('<buffer>', raw=bytes(buf))
    dat = hk.sections['__data__']
    cn = hk.sections['__classnames__']
    w = hk.wrapper
    warnings = []

    head = bytearray(buf[:w + hk.data_abs])
    data = bytearray(buf[w + hk.data_abs:
                         w + hk.data_abs + dat['local']])
    local = dict(hk.local)
    gentries = dict(hk.global_entries)
    ventries = list(hk.virtual_entries)

    # classname offset for the new shapes (append entry if missing)
    cname = 'hkpConvexVerticesShape'
    if cname in hk.classnames:
        cname_off = hk.classnames[cname]
    else:
        entry = (struct.pack('<I', _CLASS_SIGS[cname]) + b'\x09'
                 + cname.encode() + b'\0')
        gap = hk.data_abs - (cn['abs'] + hk._cn_used)
        if len(entry) > gap:
            raise RuntimeError(
                "no room to register class %r in this file's "
                "__classnames__ section" % cname)
        pos = w + cn['abs'] + hk._cn_used
        head[pos:pos + len(entry)] = entry
        cname_off = hk._cn_used + 5
        new_end = hk._cn_used + len(entry)
        # classnames section header: keep 'end' fields past the new entry
        if new_end > cn['end']:
            for i, k in enumerate(('local', 'global', 'virtual',
                                   'exports', 'imports', 'end')):
                struct.pack_into('<i', head,
                                 w + 64 + 0 * 48 + 20 + 4 + 4 * i, new_end)

    # group additions per rigid body, resolve each body's list shape
    by_rb = {}
    for a in additions:
        by_rb.setdefault(a['rb_off'], []).append(a)

    n_added = 0
    for rb_off, adds in by_rb.items():
        lst, mopp = find_list_for_rb(hk, rb_off)
        if lst is None:
            warnings.append(
                "rigid body @0x%x has no hkpListShape — new shapes for it "
                "were skipped (only list-based bodies can take extra "
                "shapes)" % rb_off)
            continue
        if mopp is not None:
            gentries[rb_off + 0x10] = (gentries[rb_off + 0x10][0], lst)
            warnings.append(
                "MOPP culling bypassed on body @0x%x so the new shapes "
                "actually collide (slightly slower, fully functional)"
                % rb_off)

        new_shape_offs = []
        for a in adds:
            verts = a['verts']
            planes = a['planes']
            n = len(verts)
            nchunks = (n + 3) // 4
            obj = _align16(len(data))
            data.extend(b'\0' * (obj - len(data)))
            rec = bytearray(0x60)
            rec[0:16] = _SHAPE_HEAD
            struct.pack_into('<f', rec, 0x10, a.get('radius', 0.025))
            mn = [min(v[i] for v in verts) for i in range(3)]
            mx = [max(v[i] for v in verts) for i in range(3)]
            struct.pack_into('<4f', rec, 0x20,
                             *[(mx[i] - mn[i]) / 2 for i in range(3)], 0.0)
            struct.pack_into('<4f', rec, 0x30,
                             *[(mx[i] + mn[i]) / 2 for i in range(3)], 0.0)
            struct.pack_into('<3I', rec, 0x40,
                             0, nchunks, nchunks | 0xC0000000)
            struct.pack_into('<I', rec, 0x4c, n)
            struct.pack_into('<3I', rec, 0x50,
                             0, len(planes), len(planes) | 0xC0000000)
            data.extend(rec)
            # vertex chunks (FourVectors: x[4], y[4], z[4])
            varr = _align16(len(data))
            data.extend(b'\0' * (varr - len(data)))
            for c in range(nchunks):
                chunk = list(verts[c * 4:c * 4 + 4])
                while len(chunk) < 4:
                    chunk.append(chunk[-1])
                for axis in range(3):
                    data.extend(struct.pack(
                        '<4f', *[v[axis] for v in chunk]))
            parr = _align16(len(data))
            data.extend(b'\0' * (parr - len(data)))
            for pl in planes:
                data.extend(struct.pack('<4f', *pl))
            local[obj + 0x40] = varr
            local[obj + 0x50] = parr
            ventries.append((obj, 0, cname_off))
            new_shape_offs.append(obj)
            n_added += 1

        # grow the list's childInfo array (relocate to the end of data)
        old_arr = hk.local[lst + 0x18]
        old_n = hk.u32(lst + 0x1c)
        new_arr = _align16(len(data))
        data.extend(b'\0' * (new_arr - len(data)))
        old_bytes = buf[w + hk.data_abs + old_arr:
                        w + hk.data_abs + old_arr + old_n * 16]
        data.extend(old_bytes)
        template = old_bytes[-16:] if old_bytes else b'\0' * 16
        for so in new_shape_offs:
            data.extend(b'\0' * 4 + template[4:16])
        total = old_n + len(new_shape_offs)
        # entry field +0xc mirrors the list's child count in shipped files —
        # keep it consistent across all (old + new) entries
        olds = {struct.unpack_from('<I', old_bytes, 16 * i + 12)[0]
                for i in range(old_n)}
        if olds <= {old_n}:
            for i in range(total):
                struct.pack_into('<I', data, new_arr + 16 * i + 12, total)
        # move the old entries' shape-pointer fixups to the new location
        for i in range(old_n):
            ge = gentries.pop(old_arr + 16 * i, None)
            if ge is not None:
                gentries[new_arr + 16 * i] = ge
        for i, so in enumerate(new_shape_offs):
            gentries[new_arr + 16 * (old_n + i)] = (2, so)
        local[lst + 0x18] = new_arr
        struct.pack_into('<2I', data, lst + 0x1c,
                         total, total | 0xC0000000)
        # expand the list's cached AABB to cover the new shapes
        lh = list(struct.unpack_from('<3f', data, lst + 0x30))
        lc = list(struct.unpack_from('<3f', data, lst + 0x40))
        amn = [lc[i] - lh[i] for i in range(3)]
        amx = [lc[i] + lh[i] for i in range(3)]
        for a in adds:
            r = a.get('radius', 0.025)
            for v in a['verts']:
                for i in range(3):
                    amn[i] = min(amn[i], v[i] - r)
                    amx[i] = max(amx[i], v[i] + r)
        struct.pack_into('<3f', data, lst + 0x30,
                         *[(amx[i] - amn[i]) / 2 for i in range(3)])
        struct.pack_into('<3f', data, lst + 0x40,
                         *[(amx[i] + amn[i]) / 2 for i in range(3)])

    # ── re-emit the fixup tables after the (grown) data ─────────────────
    pad = _align16(len(data)) - len(data)
    data.extend(b'\0' * pad)
    local_off = len(data)
    tab = bytearray()
    for f in sorted(local):
        tab += struct.pack('<2i', f, local[f])
    while len(tab) % 16:
        tab += b'\xff'
    global_off = local_off + len(tab)
    gtab = bytearray()
    for f in sorted(gentries):
        s, t = gentries[f]
        gtab += struct.pack('<3i', f, s, t)
    while len(gtab) % 16:
        gtab += b'\xff'
    virtual_off = global_off + len(gtab)
    vtab = bytearray()
    for o, s, c in sorted(ventries):
        vtab += struct.pack('<3i', o, s, c)
    while len(vtab) % 16:
        vtab += b'\xff'
    end_off = virtual_off + len(vtab)

    # __data__ section header (section index 2): local..end fields
    base = w + 64 + 2 * 48 + 20 + 4
    for i, v in enumerate((local_off, global_off, virtual_off,
                           end_off, end_off, end_off)):
        struct.pack_into('<i', head, base + 4 * i, v)

    out = bytes(head) + bytes(data) + bytes(tab) + bytes(gtab) + bytes(vtab)
    return out, warnings, n_added


def _emit_packfile(head, data, local, gentries, ventries, w):
    """Re-emit local/global/virtual fixup tables after the (grown) data region
    and patch the __data__ section header (index 2).  Shared tail for every
    rebuild.  Returns the full file bytes."""
    pad = _align16(len(data)) - len(data)
    data = bytearray(data) + b'\0' * pad
    local_off = len(data)
    tab = bytearray()
    for f in sorted(local):
        tab += struct.pack('<2i', f, local[f])
    while len(tab) % 16:
        tab += b'\xff'
    global_off = local_off + len(tab)
    gtab = bytearray()
    for f in sorted(gentries):
        s, t = gentries[f]
        gtab += struct.pack('<3i', f, s, t)
    while len(gtab) % 16:
        gtab += b'\xff'
    virtual_off = global_off + len(gtab)
    vtab = bytearray()
    for o, s, c in sorted(ventries):
        vtab += struct.pack('<3i', o, s, c)
    while len(vtab) % 16:
        vtab += b'\xff'
    end_off = virtual_off + len(vtab)
    base = w + 64 + 2 * 48 + 20 + 4
    for i, v in enumerate((local_off, global_off, virtual_off,
                           end_off, end_off, end_off)):
        struct.pack_into('<i', head, base + 4 * i, v)
    return bytes(head) + bytes(data) + bytes(tab) + bytes(gtab) + bytes(vtab)


def rebuild_with_mesh_resizes(buf, edits):
    """Change a triangle-mesh collision shape's vertex AND/OR triangle COUNT.

    `edits`: [{'ext_off': hkpStorageExtendedMeshShape data-offset (= a mesh
                          object's 'hkx_obj_off'),
               'verts':  [(x,y,z), ...]   (shape-local),
               'tris':   [(a,b,c), ...],
               'mats':   [int per tri] | None}]

    Relocates the MeshSubpartStorage m_vertices (float4, w=0) and m_indices16
    (uint16 stride-4: [v0,v1,v2,material]) arrays to the end of the data
    section, retargets their local fixups, and rewrites EVERY dependent count:
      MeshSubpartStorage m_vertices/m_indices16 size+capacity,
      the inline TrianglesSubpart m_numTriangleShapes(+0x10)/m_numVertices(+0x1c),
      hkpStorageExtendedMeshShape tri-count(+0x80)/vert-count(+0x8c) + AABB.
    The MOPP indexes the OLD triangles, so any rigid body reaching this mesh
    THROUGH a hkpMoppBvTreeShape is retargeted straight at the mesh (MOPP
    bypassed) — slower broad-phase, but the new geometry actually collides.
    Returns (new_file_bytes, warnings)."""
    hk = HkxFile('<buffer>', raw=bytes(buf))
    dat = hk.sections['__data__']
    w = hk.wrapper
    head = bytearray(buf[:w + hk.data_abs])
    data = bytearray(buf[w + hk.data_abs:w + hk.data_abs + dat['local']])
    local = dict(hk.local)
    gentries = dict(hk.global_entries)
    ventries = list(hk.virtual_entries)
    warnings = []

    for e in edits:
        ext = e['ext_off']
        verts = e['verts']
        tris = e['tris']
        mats = e.get('mats') or [0] * len(tris)
        nv, nt = len(verts), len(tris)
        if nv > 0xFFFF:
            warnings.append("ext@0x%x: %d verts exceeds uint16 index range — "
                            "skipped" % (ext, nv))
            continue
        sto = next((t for f, t in hk.obj_globals(ext)
                    if hk.objects.get(t, '').endswith('MeshSubpartStorage')),
                   None)
        if sto is None:
            warnings.append("ext@0x%x: no MeshSubpartStorage — skipped" % ext)
            continue

        # new vertex array (hkVector4, w=0)
        voff = _align16(len(data))
        data.extend(b'\0' * (voff - len(data)))
        for x, y, z in verts:
            data.extend(struct.pack('<4f', x, y, z, 0.0))
        # new index array (uint16, STRIDE 4 per triangle: v0 v1 v2 + 1 pad u16;
        # the real per-triangle material lives in m_materialIndices below, so the
        # 4th slot is stride padding — 0 is safe, the game reads 3 per stride-8).
        ioff = _align16(len(data))
        data.extend(b'\0' * (ioff - len(data)))
        for a, b, c in tris:
            data.extend(struct.pack('<4H', a & 0xFFFF, b & 0xFFFF,
                                    c & 0xFFFF, 0))
        # new per-triangle material-index array (uint16, one per triangle).
        # MeshSubpartStorage.m_materialIndices @+0x2c is per-triangle (size ==
        # tri count) — leaving it stale would make the game index a wrong-sized
        # array. New tris default to material 0 (m_materials always has ≥1).
        moff = _align16(len(data))
        data.extend(b'\0' * (moff - len(data)))
        for mt in mats:
            data.extend(struct.pack('<H', int(mt) & 0xFFFF))

        # MeshSubpartStorage: retarget + counts (cap flag 0xC0000000 = owned)
        local[sto + 0x08] = voff
        struct.pack_into('<2I', data, sto + 0x0c, nv, nv | 0xC0000000)
        local[sto + 0x14] = ioff
        struct.pack_into('<2I', data, sto + 0x18, nt * 4,
                         (nt * 4) | 0xC0000000)
        local[sto + 0x2c] = moff
        struct.pack_into('<2I', data, sto + 0x30, nt, nt | 0xC0000000)
        # hkpStorageExtendedMeshShape inline counts
        struct.pack_into('<I', data, ext + 0x80, nt)
        struct.pack_into('<I', data, ext + 0x8c, nv)
        # inline TrianglesSubpart (m_subparts array @ local[ext+0x50])
        tsub = hk.local.get(ext + 0x50)
        if tsub is not None:
            struct.pack_into('<I', data, tsub + 0x10, nt)   # numTriangleShapes
            struct.pack_into('<I', data, tsub + 0x1c, nv)   # numVertices
        # AABB
        if verts:
            mn = [min(v[i] for v in verts) for i in range(3)]
            mx = [max(v[i] for v in verts) for i in range(3)]
            struct.pack_into('<3f', data, ext + 0x30,
                             *[(mx[i] - mn[i]) / 2 for i in range(3)])
            struct.pack_into('<3f', data, ext + 0x40,
                             *[(mx[i] + mn[i]) / 2 for i in range(3)])

        # MOPP bypass — retarget any rigid body that reaches this mesh through a
        # MoppBvTreeShape straight at the mesh (the MOPP's triangle indices are
        # now invalid).
        for rb, cls in hk.objects.items():
            if cls != 'hkpRigidBody':
                continue
            shp = hk.globals.get(rb + 0x10)
            if shp is not None and hk.objects.get(shp) == 'hkpMoppBvTreeShape':
                if any(c == ext for _f, c in hk.shape_children(shp)):
                    sect = gentries[rb + 0x10][0]
                    gentries[rb + 0x10] = (sect, ext)
                    warnings.append("MOPP bypassed on body @0x%x so the resized "
                                    "mesh collides (slower broad-phase)" % rb)

    return _emit_packfile(head, data, local, gentries, ventries, w), warnings


# ---------------------------------------------------------------------------
# Blender import
# ---------------------------------------------------------------------------

def _tag(ob, path, patch, extra=None):
    ob['hkx_native'] = True
    ob['hkx_path'] = path
    for k, v in patch.items():
        if v is not None:
            ob['hkx_off_' + k] = v
    for k, v in (extra or {}).items():
        ob[k] = v


def import_hkx_native(context, path):
    """Build Blender objects from a native .hkx. Returns (n_bodies, n_shapes)."""
    if bpy is None:
        raise RuntimeError("bpy unavailable")
    hk = HkxFile(path)
    base = os.path.splitext(os.path.basename(path))[0]
    coll = bpy.data.collections.get(base) or bpy.data.collections.new(base)
    if coll.name not in context.scene.collection.children:
        context.scene.collection.children.link(coll)

    n_shapes = 0
    bodies = hk.rigid_bodies()
    for bi, rb in enumerate(bodies):
        rb_name = rb['name'] or ('body_%d' % bi)
        root = bpy.data.objects.new("%s.%s" % (base, rb_name), None)
        root.empty_display_type = 'ARROWS'
        root.empty_display_size = 0.3
        root.matrix_world = rb['xform']
        coll.objects.link(root)
        com = hk.f32s(rb['off'] + 0x160, 3)
        _tag(root, path, rb['patch'],
             {'hkx_kind': 'rigid_body', 'hkx_obj_off': rb['off'],
              'hkx_com_local': list(com)})

        for si, sh in enumerate(rb['shapes']):
            nm = "%s.%s.%s_%d" % (base, rb_name,
                                  sh['class'].replace('hkp', ''), si)
            me = bpy.data.meshes.new(nm)
            if sh['class'] == 'hkpBoxShape':
                hx, hy, hz = sh['half_extents']
                vs = [(sx * hx, sy * hy, sz * hz)
                      for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]
                fs = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1),
                      (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
                me.from_pydata(vs, [], fs)
                kind = 'box'
            elif sh['class'] == 'hkpConvexVerticesShape':
                me.from_pydata(sh['verts'], [], [])
                # display faces: hull the verts WITHOUT reordering them
                # (export patches verts by index, so order must survive)
                if len(sh['verts']) >= 4:
                    try:
                        bm = bmesh.new()
                        bm.from_mesh(me)
                        bmesh.ops.convex_hull(bm, input=bm.verts[:])
                        bm.to_mesh(me)
                        bm.free()
                    except Exception:
                        pass    # leave as a point cloud if hulling fails
                kind = 'convex'
            elif sh['class'] in ('hkpCapsuleShape', 'hkpCylinderShape'):
                me.from_pydata(sh['verts'], [(0, 1)], [])
                kind = 'capsule'
            elif sh['class'] == 'hkpSphereShape':
                me.from_pydata([(0.0, 0.0, 0.0)], [], [])
                kind = 'sphere'
            else:                                   # triangle mesh
                me.from_pydata(sh['verts'], [], sh.get('tris', []))
                kind = 'mesh'
            me.update()
            ob = bpy.data.objects.new(nm, me)
            ob.display_type = 'WIRE'
            ob.show_all_edges = True
            coll.objects.link(ob)
            ob.parent = root
            ob.matrix_parent_inverse = mathutils.Matrix.Identity(4)
            if sh['xform'] is not None:
                ob.matrix_local = sh['xform']
            patch = {k: v for k, v in sh['patch'].items()}
            if 'wrap_trans_off' in sh:
                patch['wrap_trans'] = sh['wrap_trans_off']
            if 'wrap_rot_off' in sh:
                patch['wrap_rot'] = sh['wrap_rot_off']
            extra = {'hkx_kind': kind, 'hkx_obj_off': sh['off'],
                     'hkx_nverts': len(sh.get('verts', [])),
                     'hkx_ntris': len(sh.get('tris', []))}
            if 'radius' in sh:
                extra['hkx_radius'] = sh['radius']
            _tag(ob, path, patch, extra)
            n_shapes += 1
    return len(bodies), n_shapes


# ---------------------------------------------------------------------------
# Blender export (patch the original file)
# ---------------------------------------------------------------------------

def _w_vec(buf, off, vec, w=0.0):
    struct.pack_into('<4f', buf, off, vec[0], vec[1], vec[2], w)


def _hull_planes(verts):
    """Convex-hull face planes [(nx,ny,nz,d)] with n·v + d <= 0 inside
    (the file's verified convention).  Coplanar triangles are merged."""
    bm = bmesh.new()
    vs = [bm.verts.new(v) for v in verts]
    bmesh.ops.convex_hull(bm, input=vs)
    planes = {}
    for f in bm.faces:
        n = f.normal.normalized()
        dd = -n.dot(f.verts[0].co)
        key = (round(n.x, 4), round(n.y, 4), round(n.z, 4), round(dd, 4))
        planes[key] = (n.x, n.y, n.z, dd)
    bm.free()
    return list(planes.values())


def _collect_new_objects(context, src, tagged):
    """Untagged (or duplicated) mesh objects under this file's rigid-body
    empties -> [(ob, rb_off)].  A Blender duplicate copies the custom
    props, so several objects claiming the same file offset means the
    extras are NEW shapes."""
    rb_empties = {o: o['hkx_obj_off'] for o in tagged
                  if o.get('hkx_kind') == 'rigid_body'}
    seen_off = {}
    new = []
    for ob in tagged:
        if ob.get('hkx_kind') == 'rigid_body' or ob.type != 'MESH':
            continue
        off = ob.get('hkx_obj_off')
        prev = seen_off.get(off)
        if prev is None:
            seen_off[off] = ob
        else:
            # keep the original (shorter name wins: 'x' beats 'x.001')
            keep, extra = ((prev, ob) if len(prev.name) <= len(ob.name)
                           else (ob, prev))
            seen_off[off] = keep
            par = _rb_parent(extra, rb_empties)
            if par is not None:
                new.append((extra, rb_empties[par]))
    for ob in context.scene.objects:
        if ob.type != 'MESH' or ob.get('hkx_native'):
            continue
        par = _rb_parent(ob, rb_empties)
        if par is not None:
            new.append((ob, rb_empties[par]))
    return new, {o for o, _ in new}


def _rb_parent(ob, rb_empties):
    p = ob.parent
    while p is not None:
        if p in rb_empties:
            return p
        p = p.parent
    return None


def export_hkx_native(context, out_path, source_path=None):
    """Patch every edited hkx-native object's data back into a copy of the
    source file; objects ADDED under a rigid-body empty are serialized as
    new convex collision shapes.  Returns (n_patched, warnings:list)."""
    all_tagged = [o for o in context.scene.objects if o.get('hkx_native')]
    if not all_tagged:
        raise RuntimeError("no native-HKX objects in the scene "
                           "(import a .hkx first)")
    src = source_path or all_tagged[0]['hkx_path']
    tagged = [o for o in all_tagged if o['hkx_path'] == src]
    new_objs, new_set = _collect_new_objects(context, src, tagged)
    objs = [o for o in tagged if o not in new_set]
    buf = bytearray(open(src, 'rb').read())
    warnings = []
    resize_edits = []
    n = 0

    for ob in objs:
        kind = ob.get('hkx_kind')
        if kind == 'rigid_body':
            m = ob.matrix_world
            off = ob.get('hkx_off_transform')
            if off is not None:
                for col in range(3):
                    _w_vec(buf, off + col * 16,
                           (m[0][col], m[1][col], m[2][col]))
                _w_vec(buf, off + 48, m.translation, 1.0)
            com = mathutils.Vector(ob.get('hkx_com_local',
                                          (0.0, 0.0, 0.0))[:3])
            swept = (m @ com)
            q = m.to_quaternion()          # (w,x,y,z) -> file is xyzw
            for k in ('swept_pos0', 'swept_pos1'):
                o2 = ob.get('hkx_off_' + k)
                if o2 is not None:
                    _w_vec(buf, o2, swept)
            for k in ('swept_q0', 'swept_q1'):
                o2 = ob.get('hkx_off_' + k)
                if o2 is not None:
                    struct.pack_into('<4f', buf, o2, q.x, q.y, q.z, q.w)
            n += 1
            continue

        ml = ob.matrix_local
        loc, rot, scale = ml.decompose()
        # wrapper transform (how the shape sits under its rigid body)
        if ob.get('hkx_off_wrap_rot') is not None:
            ro = ob['hkx_off_wrap_rot']
            for col in range(3):
                _w_vec(buf, ro + col * 16,
                       (ml[0][col] / scale[col] if scale[col] else 0,
                        ml[1][col] / scale[col] if scale[col] else 0,
                        ml[2][col] / scale[col] if scale[col] else 0))
            _w_vec(buf, ob['hkx_off_wrap_trans'], loc)
        elif ob.get('hkx_off_wrap_trans') is not None:
            _w_vec(buf, ob['hkx_off_wrap_trans'], loc)
            if abs(rot.angle) > 1e-4:
                warnings.append(
                    "%s: rotation ignored — its file wrapper is "
                    "translate-only (hkpConvexTranslateShape)" % ob.name)
        elif (loc.length > 1e-5 or abs(rot.angle) > 1e-4) \
                and kind != 'mesh':
            warnings.append(
                "%s: moved/rotated but the file has no transform wrapper "
                "for it — change is NOT saved (edit verts/size instead)"
                % ob.name)

        if kind == 'box':
            off = ob.get('hkx_off_half_extents')
            if off is not None:
                # half extents from the (possibly scaled) bound box
                bb = ob.bound_box
                he = [(max(c[i] for c in bb) - min(c[i] for c in bb)) / 2
                      * abs(scale[i]) for i in range(3)]
                struct.pack_into('<3f', buf, off, *he)
                n += 1

        elif kind == 'convex':
            off = ob.get('hkx_off_rotated_vertices')
            nv = ob.get('hkx_nverts', 0)
            if off is None:
                continue
            if len(ob.data.vertices) != nv:
                warnings.append(
                    "%s: vertex count changed (%d -> %d) — convex shapes "
                    "must keep their count; skipped"
                    % (ob.name, nv, len(ob.data.vertices)))
                continue
            vs = [v.co for v in ob.data.vertices]
            for c in range((nv + 3) // 4):
                chunk = vs[c * 4:c * 4 + 4]
                while len(chunk) < 4:
                    chunk.append(chunk[-1])
                for axis in range(3):
                    struct.pack_into('<4f', buf, off + c * 48 + axis * 16,
                                     *[v[axis] for v in chunk])
            # keep the cached AABB in sync
            if ob.get('hkx_off_aabb_half') is not None:
                mn = [min(v[i] for v in vs) for i in range(3)]
                mx = [max(v[i] for v in vs) for i in range(3)]
                _w_vec(buf, ob['hkx_off_aabb_half'],
                       [(mx[i] - mn[i]) / 2 for i in range(3)])
                _w_vec(buf, ob['hkx_off_aabb_center'],
                       [(mx[i] + mn[i]) / 2 for i in range(3)])
            n += 1

        elif kind == 'capsule':
            if len(ob.data.vertices) >= 2:
                r = float(ob.get('hkx_radius', 0.0))
                a = ob.data.vertices[0].co
                b = ob.data.vertices[1].co
                if ob.get('hkx_off_capsule_a') is not None:
                    _w_vec(buf, ob['hkx_off_capsule_a'], a, r)
                    _w_vec(buf, ob['hkx_off_capsule_b'], b, r)
                if ob.get('hkx_off_radius') is not None:
                    struct.pack_into('<f', buf, ob['hkx_off_radius'], r)
                n += 1

        elif kind == 'sphere':
            if ob.get('hkx_off_radius') is not None:
                struct.pack_into('<f', buf, ob['hkx_off_radius'],
                                 float(ob.get('hkx_radius', 0.0)))
                n += 1

        elif kind == 'mesh':
            off = ob.get('hkx_off_vertices')
            nv = ob.get('hkx_nverts', 0)
            if off is None:
                continue
            me = ob.data
            me.calc_loop_triangles()
            cur_tris = [tuple(t.vertices) for t in me.loop_triangles]
            ntris0 = ob.get('hkx_ntris', len(cur_tris))
            if len(me.vertices) == nv and len(cur_tris) == ntris0:
                # counts unchanged -> patch vertex positions in place (fast,
                # MOPP stays valid for small deformations)
                for i, v in enumerate(me.vertices):
                    struct.pack_into('<3f', buf, off + i * 16, *v.co)
                n += 1
            else:
                # vertex and/or triangle count changed -> full resize (handled
                # after the loop; relocates arrays, fixes counts, bypasses MOPP)
                resize_edits.append({
                    'ext_off': ob['hkx_obj_off'],
                    'verts': [tuple(v.co) for v in me.vertices],
                    'tris': cur_tris})
                n += 1

    out_bytes = bytes(buf)

    # ── mesh resizes (vertex/triangle COUNT changed) ────────────────────
    if resize_edits:
        out_bytes, rw = rebuild_with_mesh_resizes(out_bytes, resize_edits)
        warnings.extend(rw)
        warnings.append(
            "%d mesh(es) resized (count changed) — re-import the exported file "
            "to keep editing them" % len(resize_edits))

    # ── new collision shapes (added or duplicated objects) ──────────────
    if new_objs:
        rb_world = {o['hkx_obj_off']: o.matrix_world
                    for o in tagged if o.get('hkx_kind') == 'rigid_body'}
        additions = []
        for ob, rb_off in new_objs:
            inv = rb_world[rb_off].inverted()
            verts = [tuple(inv @ (ob.matrix_world @ v.co))
                     for v in ob.data.vertices]
            if len(verts) < 4:
                warnings.append("%s: needs at least 4 vertices to form a "
                                "convex shape — skipped" % ob.name)
                continue
            additions.append({'rb_off': rb_off, 'verts': verts,
                              'planes': _hull_planes(verts),
                              'radius': float(ob.get('hkx_radius', 0.025))})
        if additions:
            out_bytes, rw, n_added = rebuild_with_additions(out_bytes,
                                                            additions)
            warnings.extend(rw)
            n += n_added
            if n_added:
                warnings.append(
                    "%d new convex shape(s) added — re-import the exported "
                    "file to keep editing them" % n_added)

    with open(out_path, 'wb') as f:
        f.write(out_bytes)
    return n, warnings
