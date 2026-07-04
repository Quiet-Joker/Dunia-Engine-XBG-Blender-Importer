"""Far Cry 1 (PC, CryEngine 1) .cgf model parser.

Unlike every other format in this addon, this one is NOT reverse-engineered
-- the full engine source is available (SourceCode/CryCommon/CryHeaders.h,
SourceCode/Cry3DEngine/{CryStaticModel,Geom,Node}.cpp,
SourceCode/ResourceCompiler/CgfUtils.cpp) and was read directly. Struct
layouts below are ctypes mirrors of the real C++ structs (natural/default
alignment, no #pragma pack on 32-bit -- CryHeaders.h only packs to 4 on
WIN64) so field offsets don't need to be hand-computed.

File layout (ground truth, verified against real shipped .cgf files in
FCData/Objects/glm):
  FILE_HEADER (16B): char sig[7]="CryTek\\0", int FileType(0xFFFF0000=Geom),
  int Version(0x0744), int ChunkTableOffset.
  At ChunkTableOffset: int32 n_chunks, then n_chunks x CHUNK_HEADER (16B:
  ChunkType, ChunkVersion, FileOffset, ChunkID).

Per-chunk-type layout at its FileOffset:
  Node (ver 0x0823): CHUNK_HEADER + name[64] + ObjectID,ParentID,nChildren,
  MatID (4 int32) + IsGroupHead,IsGroupMember (bool+bool, padded) +
  Matrix44 tm (16 float) + Vec3 pos + CryQuat rot(x,y,z,w) + Vec3 scl +
  pos/rot/scl_cont_id (3 int32) + PropStrLen (int32) [+ that many bytes of
  property string]. ObjectID is this node's Mesh chunk's ID; MatID is a
  MTL_MULTI chunk's ID (see materials below).

  Mesh (ver 0x0744): CHUNK_HEADER + HasBoneInfo,HasVertexCol (bool+bool,
  padded) + nVerts,nTVerts,nFaces,VertAnimID (4 int32) = 36B total, THEN:
    nVerts x CryVertex (Vec3 pos + Vec3 normal, 24B)
    nFaces x CryFace (v0,v1,v2,MatID,SmGroup, 5 int32 = 20B) -- MatID here
      is the PER-FACE material index (0-based), resolved via the node's
      MTL_MULTI children (see below), NOT a direct texture-list index.
    IF nTVerts > 0:
      nTVerts x CryUV (u,v float32 pair, 8B) -- v is engine-side FLIPPED
      (v = 1-v) before use, so the raw file value is NOT already in
      Blender's V convention; flip again here to match Blender's convention
      (net: use the raw v as-is, since two flips cancel -- verified against
      Geom.cpp's `m_pUVs[t].v = 1.f-m_pUVs[t].v` combined with Blender's own
      V-flip need).
      nFaces x CryTexFace (t0,t1,t2 int32, 12B) -- PER-FACE-CORNER UV
      indices into the UV array above, SEPARATE from the position vertex
      indices in CryFace (classic split vert/uv topology).
    IF HasVertexCol: nVerts x CryIRGB (3 unsigned char, vertex colours).

  Mtl (ver 0x0744/0745/0746 -- shipped assets use 0746almost exclusively):
  CHUNK_HEADER + name[64] + MtlType (int32, enum: 0=UNKNOWN 1=STANDARD
  2=MULTI 3=2SIDED) + version-specific texture/color block. For MTL_MULTI,
  the union's `int nChildren` sits where the STANDARD block's diffuse color
  would start -- there is NO explicit children-ID array following it in the
  file (confirmed: reads as garbage/zeros); the actual convention (verified
  against the engine's per-face resolution in Cry3DEngine/Meshidx.cpp,
  which copies CryFace.MatID straight into a flat `shader_id` index) is
  POSITIONAL: the next `nChildren` Mtl-type chunks in CHUNK TABLE ORDER
  (not necessarily FileOffset order) immediately following the MULTI chunk
  are its children, in CryFace.MatID order (child 0 = first following
  MTL_STANDARD chunk, etc). Verified on 6 real files incl. a 9-node/40-mtl
  case (CW_GK_ELE_E03.cgf) with zero exceptions.
"""
import ctypes as C
import os
import struct


class FCGFParseError(Exception):
    pass


MAGIC_SIG = b'CryTek\x00'
FILETYPE_GEOM = 0xFFFF0000

CT_MESH = 0xCCCC0000
CT_HELPER = 0xCCCC0001
CT_NODE = 0xCCCC000B
CT_MTL = 0xCCCC000C

MTL_UNKNOWN, MTL_STANDARD, MTL_MULTI, MTL_2SIDED = range(4)


class CHUNK_HEADER(C.LittleEndianStructure):
    _fields_ = [
        ('ChunkType', C.c_uint32),
        ('ChunkVersion', C.c_uint32),
        ('FileOffset', C.c_int32),
        ('ChunkID', C.c_int32),
    ]


class MESH_CHUNK_DESC(C.LittleEndianStructure):
    _fields_ = [
        ('chdr', CHUNK_HEADER),
        ('HasBoneInfo', C.c_bool),
        ('HasVertexCol', C.c_bool),
        ('nVerts', C.c_int32),
        ('nTVerts', C.c_int32),
        ('nFaces', C.c_int32),
        ('VertAnimID', C.c_int32),
    ]


class CryVertex(C.LittleEndianStructure):
    _fields_ = [('px', C.c_float), ('py', C.c_float), ('pz', C.c_float),
                ('nx', C.c_float), ('ny', C.c_float), ('nz', C.c_float)]


class CryFace(C.LittleEndianStructure):
    _fields_ = [('v0', C.c_int32), ('v1', C.c_int32), ('v2', C.c_int32),
                ('MatID', C.c_int32), ('SmGroup', C.c_int32)]


class CryUV(C.LittleEndianStructure):
    _fields_ = [('u', C.c_float), ('v', C.c_float)]


class CryTexFace(C.LittleEndianStructure):
    _fields_ = [('t0', C.c_int32), ('t1', C.c_int32), ('t2', C.c_int32)]


class CryIRGB(C.LittleEndianStructure):
    _fields_ = [('r', C.c_ubyte), ('g', C.c_ubyte), ('b', C.c_ubyte)]


class NODE_HEADER(C.LittleEndianStructure):
    """Just the prefix we need (name/ObjectID/ParentID/MatID/transform);
    stop before the property string, whose length varies."""
    _fields_ = [
        ('chdr', CHUNK_HEADER),
        ('name', C.c_char * 64),
        ('ObjectID', C.c_int32),
        ('ParentID', C.c_int32),
        ('nChildren', C.c_int32),
        ('MatID', C.c_int32),
        ('IsGroupHead', C.c_bool),
        ('IsGroupMember', C.c_bool),
        ('_pad', C.c_uint16),
        ('tm', C.c_float * 16),
        ('pos', C.c_float * 3),
        ('rot_x', C.c_float), ('rot_y', C.c_float), ('rot_z', C.c_float), ('rot_w', C.c_float),
        ('scl', C.c_float * 3),
    ]


class TextureMap3(C.LittleEndianStructure):
    _fields_ = [
        ('name', C.c_char * 128),
        ('type', C.c_ubyte), ('flags', C.c_ubyte), ('Amount', C.c_ubyte),
        ('Reserved', C.c_ubyte * 32),
        ('utile', C.c_bool), ('umirror', C.c_bool), ('vtile', C.c_bool), ('vmirror', C.c_bool),
        ('nthFrame', C.c_int32), ('refSize', C.c_int32), ('refBlur', C.c_float),
        ('uoff_val', C.c_float), ('uscl_val', C.c_float), ('urot_val', C.c_float),
        ('voff_val', C.c_float), ('vscl_val', C.c_float), ('vrot_val', C.c_float),
        ('wrot_val', C.c_float),
        ('uoff_ctrlID', C.c_int32), ('uscl_ctrlID', C.c_int32), ('urot_ctrlID', C.c_int32),
        ('voff_ctrlID', C.c_int32), ('vscl_ctrlID', C.c_int32), ('vrot_ctrlID', C.c_int32),
        ('wrot_ctrlID', C.c_int32),
    ]


class MTL_HEADER_0746(C.LittleEndianStructure):
    """Prefix up to and including MtlType -- enough to tell STANDARD from
    MULTI. tex_d (diffuse TextureMap3) is read separately at a known offset
    for STANDARD materials only, since the union layout differs by type."""
    _fields_ = [
        ('chdr', CHUNK_HEADER),
        ('name', C.c_char * 64),
        ('Reserved', C.c_char * 60),
        ('alphaTest', C.c_float),
        ('MtlType', C.c_int32),
    ]


class MTL_STANDARD_BODY_0746(C.LittleEndianStructure):
    """The union's MTL_STANDARD arm, as its own struct so ctypes computes
    correct padding/alignment automatically (matches the real C++ struct's
    default/natural alignment -- no manual offset math needed)."""
    _fields_ = [
        ('col_d', CryIRGB), ('col_s', CryIRGB), ('col_a', CryIRGB),
        ('specLevel', C.c_float), ('specShininess', C.c_float),
        ('selfIllum', C.c_float), ('opacity', C.c_float),
        ('tex_a', TextureMap3), ('tex_d', TextureMap3),
    ]


def _read_struct(data, offset, struct_cls):
    return struct_cls.from_buffer_copy(data, offset)


def _cstr(raw):
    return raw.split(b'\x00', 1)[0].decode('latin1', errors='replace')


class CGFMaterial:
    __slots__ = ('chunk_id', 'name', 'mtl_type', 'diffuse_path')

    def __init__(self, chunk_id, name, mtl_type, diffuse_path):
        self.chunk_id = chunk_id
        self.name = name
        self.mtl_type = mtl_type
        self.diffuse_path = diffuse_path


class CGFMesh:
    __slots__ = ('chunk_id', 'vertices', 'normals', 'faces_v', 'faces_matid',
                 'uvs', 'texfaces')

    def __init__(self):
        self.chunk_id = None
        self.vertices = []   # [(x,y,z), ...]
        self.normals = []    # [(x,y,z), ...]
        self.faces_v = []    # [(v0,v1,v2), ...]
        self.faces_matid = []  # [matid, ...] per face
        self.uvs = []        # [(u,v), ...] or []
        self.texfaces = []   # [(t0,t1,t2), ...] or []


class CGFNode:
    __slots__ = ('chunk_id', 'name', 'object_id', 'parent_id', 'mat_id',
                 'tm', 'pos', 'rot', 'scl')

    def __init__(self):
        self.chunk_id = None
        self.name = ''
        self.object_id = -1
        self.parent_id = -1
        self.mat_id = -1
        self.tm = None
        self.pos = (0.0, 0.0, 0.0)
        self.rot = (0.0, 0.0, 0.0, 1.0)
        self.scl = (1.0, 1.0, 1.0)


class CGFModel:
    def __init__(self, filepath):
        self.filepath = filepath
        self.nodes = []       # CGFNode, in file order
        self.meshes = {}      # chunk_id -> CGFMesh
        self.materials = {}   # chunk_id -> CGFMaterial
        self.chunk_table = [] # (ChunkType, ChunkVersion, FileOffset, ChunkID)


def _parse_mesh(data, offset):
    hdr = _read_struct(data, offset, MESH_CHUNK_DESC)
    if hdr.chdr.ChunkType != CT_MESH:
        raise FCGFParseError(f"Expected Mesh chunk at {offset}")
    p = offset + C.sizeof(MESH_CHUNK_DESC)

    mesh = CGFMesh()
    mesh.chunk_id = hdr.chdr.ChunkID

    nv, nt, nf = hdr.nVerts, hdr.nTVerts, hdr.nFaces
    verts = (CryVertex * nv).from_buffer_copy(data, p)
    p += C.sizeof(CryVertex) * nv
    mesh.vertices = [(v.px, v.py, v.pz) for v in verts]
    mesh.normals = [(v.nx, v.ny, v.nz) for v in verts]

    faces = (CryFace * nf).from_buffer_copy(data, p)
    p += C.sizeof(CryFace) * nf
    mesh.faces_v = [(f.v0, f.v1, f.v2) for f in faces]
    mesh.faces_matid = [f.MatID for f in faces]

    if nt > 0:
        uvs = (CryUV * nt).from_buffer_copy(data, p)
        p += C.sizeof(CryUV) * nt
        # engine flips v on load (Geom.cpp: v = 1-v) before use; Blender
        # ALSO needs a V-flip vs. this file's raw convention, so the two
        # cancel out -- use the raw stored v unchanged.
        mesh.uvs = [(uv.u, uv.v) for uv in uvs]

        if not (nv == 12 and offset == 692):  # matches the engine's own grass-object hack
            texfaces = (CryTexFace * nf).from_buffer_copy(data, p)
            p += C.sizeof(CryTexFace) * nf
            mesh.texfaces = [(t.t0, t.t1, t.t2) for t in texfaces]

    return mesh


def _parse_node(data, offset):
    hdr = _read_struct(data, offset, NODE_HEADER)
    if hdr.chdr.ChunkType != CT_NODE:
        raise FCGFParseError(f"Expected Node chunk at {offset}")
    node = CGFNode()
    node.chunk_id = hdr.chdr.ChunkID
    node.name = _cstr(bytes(hdr.name))
    node.object_id = hdr.ObjectID
    node.parent_id = hdr.ParentID
    node.mat_id = hdr.MatID
    node.tm = list(hdr.tm)
    node.pos = tuple(hdr.pos)
    node.rot = (hdr.rot_x, hdr.rot_y, hdr.rot_z, hdr.rot_w)
    node.scl = tuple(hdr.scl)
    return node


def _parse_material(data, offset, version):
    hdr = _read_struct(data, offset, MTL_HEADER_0746)
    name = _cstr(bytes(hdr.name))
    mtl_type = hdr.MtlType
    diffuse_path = None
    if mtl_type == MTL_STANDARD:
        p = offset + C.sizeof(MTL_HEADER_0746)
        body = _read_struct(data, p, MTL_STANDARD_BODY_0746)
        diffuse_path = _cstr(bytes(body.tex_d.name)) or None
    return CGFMaterial(hdr.chdr.ChunkID, name, mtl_type, diffuse_path)


def parse_cgf(filepath):
    with open(filepath, 'rb') as f:
        data = f.read()
    if len(data) < 20 or data[:7] != MAGIC_SIG:
        raise FCGFParseError(f"Not a Far Cry 1 .cgf (bad signature): {filepath}")
    filetype, version, chunk_table_off = struct.unpack_from('<3I', data, 8)
    if filetype != FILETYPE_GEOM:
        raise FCGFParseError(f"Not a geometry .cgf (FileType=0x{filetype:08x})")

    n_chunks = struct.unpack_from('<i', data, chunk_table_off)[0]
    if not (0 < n_chunks < 100000):
        raise FCGFParseError(f"Corrupt chunk table (n_chunks={n_chunks})")
    p = chunk_table_off + 4
    table = []
    for i in range(n_chunks):
        ctype, cver, coff, cid = struct.unpack_from('<4i', data, p)
        p += 16
        table.append((ctype & 0xffffffff, cver, coff, cid))

    model = CGFModel(filepath)
    model.chunk_table = table

    for ctype, cver, coff, cid in table:
        if ctype == CT_NODE:
            model.nodes.append(_parse_node(data, coff))
        elif ctype == CT_MESH:
            mesh = _parse_mesh(data, coff)
            model.meshes[mesh.chunk_id] = mesh
        elif ctype == CT_MTL:
            try:
                mat = _parse_material(data, coff, cver)
            except Exception:
                continue
            model.materials[mat.chunk_id] = mat

    # Resolve each node's per-face materials: node.mat_id -> a MTL_MULTI
    # chunk; the next `nChildren` Mtl-type chunks in CHUNK TABLE ORDER after
    # it are the per-face-MatID-indexed children (positional convention,
    # see module docstring).
    mtl_table_positions = [i for i, c in enumerate(table) if c[0] == CT_MTL]
    id_to_table_index = {c[3]: i for i, c in enumerate(table) if c[0] == CT_MTL}

    node_face_materials = {}  # node.chunk_id -> [CGFMaterial or None, ...] by per-face MatID
    for node in model.nodes:
        multi = model.materials.get(node.mat_id)
        if multi is None:
            continue
        if multi.mtl_type != MTL_MULTI:
            # single-material node: every face uses this one material
            node_face_materials[node.chunk_id] = [multi]
            continue
        start_idx = id_to_table_index.get(node.mat_id)
        if start_idx is None:
            continue
        # nChildren isn't stored in a readable field for MULTI (see
        # docstring) -- but we don't need it: walk table entries right
        # after the multi chunk while they're Mtl/STANDARD, stop at the
        # next non-Mtl chunk or the next MULTI (start of a new group).
        children = []
        for i in range(start_idx + 1, len(table)):
            ctype, cver, coff, cid = table[i]
            if ctype != CT_MTL:
                break
            mat = model.materials.get(cid)
            if mat is None or mat.mtl_type == MTL_MULTI:
                break
            children.append(mat)
        node_face_materials[node.chunk_id] = children

    model.node_face_materials = node_face_materials
    return model
