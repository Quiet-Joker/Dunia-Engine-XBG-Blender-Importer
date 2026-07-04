"""WD1 .hkx collision import — native 64-bit Havok 2012 packfile reader.

Watch Dogs 1 .hkx files are Havok 2012.2.0-r1 BINARY PACKFILES with 8-byte
pointers (64-bit), behind the game's 16-byte wrapper header.  This is a
DIFFERENT layout from Avatar's 32-bit Havok 5.5 packfiles (see avatar/
hkx_native.py), so it needs its own reader: the class member offsets shift
because every pointer/vtable slot is 8 bytes instead of 4.

What we extract
---------------
The vehicle collision is a hkpRigidBody → hkpListShape (or
hkpStaticCompoundShape) holding a set of hkpConvexVerticesShape convex hulls
plus, for the detailed mesh, hkpBvCompressedMeshShape triangle soups.  This
reader pulls out the convex hulls (the simplified physics proxy, which is what
modders actually edit) and builds one wireframe hull object per shape.  The
compressed mesh shapes use Havok's BVH-compressed quantised format and are
reported but not yet decoded.

Confirmed hkpConvexVerticesShape layout (Havok 2012, 64-bit, object-relative):
    +0x20 f32  m_radius
    +0x30 hkVector4 m_aabbHalfExtents
    +0x40 hkVector4 m_aabbCenter
    +0x50 hkArray<hkFourVectors> m_rotatedVertices
          (ptr via local fixup, +0x58 u32 size = number of hkFourVectors)
    +0x60 u32  m_numVertices
    +0x68 hkArray m_planeEquations
    +0x78 ptr  m_connectivity (global fixup → hkpConvexVerticesConnectivity)

hkFourVectors is SOA: [x0 x1 x2 x3][y0 y1 y2 y3][z0 z1 z2 z3] (48 bytes),
holding 4 vertices each; the last block is padded to a multiple of 4.
"""

import os
import struct

try:
    import bpy
    import bmesh
    import mathutils
except ImportError:
    bpy = None
    bmesh = None
    mathutils = None

HAVOK_MAGIC = b'\x57\xe0\xe0\x57\x10\xc0\xc0\x10'


class WdHkxFile:
    """Parsed WD1 (64-bit Havok 2012) collision packfile — read-only view."""

    def __init__(self, path, raw=None):
        self.path = path
        self.raw = raw if raw is not None else open(path, 'rb').read()

        if self.raw[:8] == HAVOK_MAGIC:
            self.wrapper = 0
        elif self.raw[16:24] == HAVOK_MAGIC:
            self.wrapper = 16
        else:
            idx = self.raw.find(HAVOK_MAGIC)
            if idx < 0:
                raise ValueError("not a Havok packfile: %s" % path)
            self.wrapper = idx
        d = self.raw[self.wrapper:]
        self.d = d

        ptr_size = d[16]
        if ptr_size != 8:
            raise ValueError(
                "not a 64-bit Havok packfile (bytesInPointer=%d) — use the "
                "Avatar 32-bit reader instead" % ptr_size)

        num_sections, = struct.unpack_from('<i', d, 20)
        self.version = d[40:56].split(b'\0')[0].decode('latin-1')

        # Section headers (48 bytes each)
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
        self._cn_abs = cn['abs']

        def classname(off):
            e = d.index(b'\0', self._cn_abs + off)
            return d[self._cn_abs + off:e].decode('latin-1')

        # objects: data-relative offset -> class name (virtual fixups,
        # 12-byte entries: [u32 from][u32 sectionIdx][u32 nameOffset])
        self.objects = {}
        p = self.base + dat['virtual']
        while p + 12 <= self.base + dat['exports']:
            o, s, c = struct.unpack_from('<3i', d, p); p += 12
            if o != -1:
                self.objects[o] = classname(c)

        # local fixups: from-offset -> to-offset (both data-relative, 4-byte)
        self.local = {}
        p = self.base + dat['local']
        while p + 8 <= self.base + dat['global']:
            f, t = struct.unpack_from('<2i', d, p); p += 8
            if f != -1:
                self.local[f] = t

        # global fixups: from-offset -> target object offset
        self.glob = {}
        p = self.base + dat['global']
        while p + 12 <= self.base + dat['virtual']:
            f, s, t = struct.unpack_from('<3i', d, p); p += 12
            if f != -1:
                self.glob[f] = t

    # -- typed reads (object-relative) --
    def f32(self, obj, rel):
        return struct.unpack_from('<f', self.d, self.base + obj + rel)[0]

    def u32(self, obj, rel):
        return struct.unpack_from('<I', self.d, self.base + obj + rel)[0]

    def vec3(self, obj, rel):
        return struct.unpack_from('<3f', self.d, self.base + obj + rel)

    def convex_shapes(self):
        """Yield dicts for every hkpConvexVerticesShape: offset, radius, verts."""
        for o in sorted(self.objects):
            if self.objects[o] != 'hkpConvexVerticesShape':
                continue
            num = self.u32(o, 0x60)
            arr = self.local.get(o + 0x50)
            verts = self._read_fourvectors(arr, num) if arr is not None else []
            yield {
                'offset': o,
                'radius': self.f32(o, 0x20),
                'num': num,
                'verts': verts,
            }

    def _read_fourvectors(self, data_off, num):
        """Read num vertices from an SOA hkFourVectors array at data_off."""
        verts = []
        p = self.base + data_off
        for _ in range((num + 3) // 4):
            xs = struct.unpack_from('<4f', self.d, p)
            ys = struct.unpack_from('<4f', self.d, p + 16)
            zs = struct.unpack_from('<4f', self.d, p + 32)
            p += 48
            for k in range(4):
                verts.append((xs[k], ys[k], zs[k]))
        return verts[:num]

    def compressed_mesh_count(self):
        return sum(1 for c in self.objects.values()
                   if c == 'hkpBvCompressedMeshShape')

    def compressed_meshes(self):
        """Yield each hkpBvCompressedMeshShape's domain + decoded per-section
        vertex clusters.

        The detailed collision is an hkcdStaticMeshTree: a BVH of `sections`,
        each carrying a tight local AABB and a block of 11/11/10-bit quantised
        vertices (dequantised against that AABB — the Watch Dogs / Havok 2012
        convention, scale = extent/(2047,2047,1023)).  Decoding gives back the
        exact collision-surface vertices (validated: section vertex minima land
        on their AABB minima, and the union of all sections equals the mesh
        domain).

        Layout (object-relative, 64-bit):
            +0x80 hkVector4 domain AABB min
            +0x90 hkVector4 domain AABB max
            +0xb0 hkArray<Section> (ptr via local fixup, +0xb8 u32 count)
        Section (0x60 bytes):
            +0x00 hkArray<packedVertex u32> (ptr via local fixup, +0x08 count)
            +0x10 AABB min   +0x20 AABB max
        Packed vertex (u32): x = bits 0-10, y = bits 11-21, z = bits 22-31.

        NOTE the exact triangle connectivity lives in Havok's compressed BVH
        primitive bitstream (per-section +0x50/+0x54 bit offsets) and is not
        decoded; callers reconstruct a surface per section from the vertices.
        """
        SEC = 0x60
        for o in sorted(self.objects):
            if self.objects[o] != 'hkpBvCompressedMeshShape':
                continue
            dmin = self.vec3(o, 0x80)
            dmax = self.vec3(o, 0x90)
            sec_ptr = self.local.get(o + 0xb0)
            n = self.u32(o, 0xb8)
            sections = []
            if sec_ptr is not None:
                for s in range(n):
                    so = sec_ptr + s * SEC
                    amin = struct.unpack_from('<3f', self.d, self.base + so + 0x10)
                    amax = struct.unpack_from('<3f', self.d, self.base + so + 0x20)
                    vptr = self.local.get(so + 0x00)
                    nv = self.u32(so, 0x08)
                    verts = self._decode_section_verts(vptr, nv, amin, amax)
                    sections.append({'aabb': (amin, amax), 'verts': verts})
            yield {
                'offset': o,
                'domain': (dmin, dmax),
                'sections': sections,
            }

    def _decode_section_verts(self, vptr, nv, amin, amax):
        """Dequantise nv 11/11/10-bit packed vertices against the section AABB."""
        if vptr is None or nv == 0:
            return []
        sx = (amax[0] - amin[0]) / 2047.0
        sy = (amax[1] - amin[1]) / 2047.0
        sz = (amax[2] - amin[2]) / 1023.0
        out = []
        p = self.base + vptr
        for i in range(nv):
            pk = struct.unpack_from('<I', self.d, p + i * 4)[0]
            xq = pk & 0x7FF
            yq = (pk >> 11) & 0x7FF
            zq = (pk >> 22) & 0x3FF
            out.append((amin[0] + xq * sx,
                        amin[1] + yq * sy,
                        amin[2] + zq * sz))
        return out


def _add_box(bm, amin, amax):
    """Add a wireframe box (8 verts, 12 edges) spanning amin..amax to bm."""
    x0, y0, z0 = amin
    x1, y1, z1 = amax
    corners = [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),
               (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)]
    vs = [bm.verts.new(c) for c in corners]
    edges = [(0, 1), (1, 2), (2, 3), (3, 0),   # bottom
             (4, 5), (5, 6), (6, 7), (7, 4),   # top
             (0, 4), (1, 5), (2, 6), (3, 7)]   # verticals
    for a, b in edges:
        try:
            bm.edges.new((vs[a], vs[b]))
        except ValueError:
            pass  # duplicate edge (degenerate box)


def import_hkx_wd(context, path):
    """Build wireframe collision objects for a WD1 .hkx collision file.

    Creates a closed wireframe hull per hkpConvexVerticesShape, plus a
    section-bounds proxy per hkpBvCompressedMeshShape (the detailed collision
    mesh, represented by its tile of section AABBs until the quantised triangle
    stream is decoded).

    Returns (n_hulls, n_total_hull_verts, n_compressed_meshes).
    """
    if bpy is None:
        raise RuntimeError("bpy unavailable — run inside Blender")

    f = WdHkxFile(path)
    base_name = os.path.splitext(os.path.basename(path))[0]

    # Parent empty so all shapes group under one node (mirrors Avatar's
    # rigid-body empties).
    root = bpy.data.objects.new(base_name + "_collision", None)
    root.empty_display_type = 'CUBE'
    root.empty_display_size = 0.25
    context.collection.objects.link(root)

    n_hulls = 0
    n_verts = 0
    for shape in f.convex_shapes():
        verts = shape['verts']
        if not verts:
            continue
        me = bpy.data.meshes.new("%s_hull%d" % (base_name, n_hulls))
        bm = bmesh.new()
        for v in verts:
            bm.verts.new(v)
        bm.verts.ensure_lookup_table()
        # Build the closed hull surface from the point cloud.
        try:
            bmesh.ops.convex_hull(bm, input=bm.verts)
        except Exception:
            pass  # fall back to a loose point cloud if hull fails
        bm.to_mesh(me)
        bm.free()
        obj = bpy.data.objects.new(me.name, me)
        obj.display_type = 'WIRE'
        obj.show_wire = True
        obj['wd_hkx_src'] = path
        obj['wd_hkx_shape_off'] = shape['offset']
        context.collection.objects.link(obj)
        obj.parent = root
        n_hulls += 1
        n_verts += len(verts)

    # Detailed collision meshes (hkcdStaticMeshTree): one solid object per
    # shape, reconstructed as the per-section convex hull of the real decoded
    # vertices.  The exact BVH-packed triangulation isn't decoded, but each
    # section is a small local cluster so its hull faithfully approximates that
    # patch of the collision surface; the union of section hulls is a usable,
    # editable collision mesh built entirely from genuine vertex data.
    n_meshes = 0
    n_mesh_verts = 0
    for cm in f.compressed_meshes():
        me = bpy.data.meshes.new("%s_collmesh%d" % (base_name, n_meshes))
        bm = bmesh.new()
        any_geo = False
        for sec in cm['sections']:
            verts = sec['verts']
            n_mesh_verts += len(verts)
            if len(verts) < 4:
                continue
            tmp = bmesh.new()
            for v in verts:
                tmp.verts.new(v)
            tmp.verts.ensure_lookup_table()
            try:
                res = bmesh.ops.convex_hull(tmp, input=tmp.verts)
                # Drop the interior / unused points so only the hull surface
                # (verts on faces) is kept.
                discard = set(res.get('geom_interior', [])) | set(res.get('geom_unused', []))
                if discard:
                    bmesh.ops.delete(tmp, geom=list(discard), context='VERTS')
                tmp_me = bpy.data.meshes.new("_tmp")
                tmp.to_mesh(tmp_me)
                bm.from_mesh(tmp_me)
                bpy.data.meshes.remove(tmp_me)
                any_geo = True
            except Exception:
                pass
            tmp.free()
        if not any_geo:
            _add_box(bm, *cm['domain'])
        bm.to_mesh(me)
        bm.free()
        obj = bpy.data.objects.new(me.name, me)
        obj.display_type = 'WIRE'
        obj.show_wire = True
        obj['wd_hkx_src'] = path
        obj['wd_hkx_shape_off'] = cm['offset']
        obj['wd_hkx_is_hull_reconstruction'] = True
        context.collection.objects.link(obj)
        obj.parent = root
        n_meshes += 1

    return n_hulls, n_verts + n_mesh_verts, n_meshes
