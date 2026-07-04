"""SDOL chunk parser and serializer (injection side).

SDOL is the geometry payload — for each LOD it holds:
  * the list of vertex buffers (`vb_info`) — format flags, stride,
    byte offset within the LOD's vertex section, etc.
  * the list of submeshes (`submeshes`) — vertex-buffer index, LOD
    group, sub-index, starting index offset, etc.
  * the packed vertex section bytes (`vert_data`) — 16-byte aligned.
  * the packed index section bytes (`indice_data`) — uint16 indices,
    also 16-byte aligned.

`parse_sdol()` reads a chunk into an `SDOL` container; `build_sdol_chunk()`
goes the other way and produces a complete chunk (12-byte header + data)
ready to be spliced into the file at any offset.

Note on the import side: `parse_sdol_chunk` in mesh.py also reads SDOL
data, but it uses the streaming `BinaryReader` and parses straight into
`Mesh` instances for Blender geometry construction.  The two functions
do NOT share code intentionally — they have different output shapes and
the inject side needs random access to raw bytes for splicing.

Endianness
----------
All multi-byte ints/floats follow the file's byte order (`'<'` for PC,
`'>'` for PS3).  The vertex data (`vert_data`) and index data
(`indice_data`) bytes themselves are passed through unmodified — they
were either produced by `_encode_vertices()` in the right byte order
(inject path) or read straight from the source file (round-trip path).
The chunk magic `'SDOL'` is also written endian-aware: on PS3 it lands
in the file as the reversed bytes `'LODS'`.
"""

import struct

from .binary_fc2 import encode_chunk_magic, LE


class LOD:
    """One LOD level inside the SDOL chunk."""
    __slots__ = ('lod_dist', 'vb_info', 'submeshes',
                 'vert_section_size', 'vert_data',
                 'indice_section_size', 'indice_data')

    def __init__(self):
        self.lod_dist            = 0.0
        self.vb_info             = []   # list of dicts: flags, stride, unk, offset
        self.submeshes           = []   # list of dicts: vb_idx, lod_grp, sub_idx,
                                        #                idx_offset, vert_marker,
                                        #                unk1, unk2
        self.vert_section_size   = 0
        self.vert_data           = b''
        self.indice_section_size = 0    # count of uint16 index values (NOT bytes)
        self.indice_data         = b''


class SDOL:
    """Parsed representation of the whole SDOL chunk content."""
    __slots__ = ('unk_0', 'unk_1', 'lod_count', 'lods')

    def __init__(self):
        self.unk_0     = 0
        self.unk_1     = 0
        self.lod_count = 0
        self.lods      = []


def parse_sdol(file_data, data_start, endian=LE):
    """Parse an SDOL chunk from raw bytes.

    `data_start` is the absolute file offset of the FIRST byte after
    the 12-byte chunk header (i.e. `chunk_start + 12`).  `endian` is
    `'<'` (PC) or `'>'` (PS3).  Returns an `SDOL` instance.
    """
    p  = data_start
    sd = SDOL()
    en = endian

    sd.unk_0     = struct.unpack_from(f'{en}i', file_data, p)[0]; p += 4
    sd.unk_1     = struct.unpack_from(f'{en}i', file_data, p)[0]; p += 4
    sd.lod_count = struct.unpack_from(f'{en}i', file_data, p)[0]; p += 4

    for _ in range(sd.lod_count):
        lod          = LOD()
        lod.lod_dist = struct.unpack_from(f'{en}f', file_data, p)[0]; p += 4
        vb_count     = struct.unpack_from(f'{en}i', file_data, p)[0]; p += 4

        for _ in range(vb_count):
            lod.vb_info.append({
                'flags':  struct.unpack_from(f'{en}i', file_data, p     )[0],
                'stride': struct.unpack_from(f'{en}i', file_data, p +  4)[0],
                'unk':    struct.unpack_from(f'{en}i', file_data, p +  8)[0],
                'offset': struct.unpack_from(f'{en}i', file_data, p + 12)[0],
            }); p += 16

        sm_count = struct.unpack_from(f'{en}i', file_data, p)[0]; p += 4
        for _ in range(sm_count):
            lod.submeshes.append({
                'vb_idx':      struct.unpack_from(f'{en}i', file_data, p     )[0],
                'lod_grp':     struct.unpack_from(f'{en}i', file_data, p +  4)[0],
                'sub_idx':     struct.unpack_from(f'{en}i', file_data, p +  8)[0],
                'idx_offset':  struct.unpack_from(f'{en}i', file_data, p + 12)[0],
                'vert_marker': struct.unpack_from(f'{en}i', file_data, p + 16)[0],
                'unk1':        struct.unpack_from(f'{en}i', file_data, p + 20)[0],
                'unk2':        struct.unpack_from(f'{en}i', file_data, p + 24)[0],
            }); p += 28

        # -- Vertex section (16-byte aligned start) -----------------------
        lod.vert_section_size = struct.unpack_from(f'{en}I', file_data, p)[0]; p += 4
        rem = p % 16
        if rem:
            p += 16 - rem
        lod.vert_data = bytes(file_data[p : p + lod.vert_section_size])
        p += lod.vert_section_size

        # -- Index section (16-byte aligned start; size is uint16 COUNT) --
        lod.indice_section_size = struct.unpack_from(f'{en}I', file_data, p)[0]; p += 4
        rem = p % 16
        if rem:
            p += 16 - rem
        lod.indice_data = bytes(file_data[p : p + lod.indice_section_size * 2])
        p += lod.indice_section_size * 2

        sd.lods.append(lod)

    return sd


def build_sdol_chunk(sdol, chunk_start, ci0, endian=LE):
    """Serialise an `SDOL` to a complete chunk (header + data).

    `chunk_start` : absolute file offset where the magic `SDOL` will
                    land (used to compute the 16-byte alignment padding,
                    which is offset-relative).
    `ci0`         : original value of the second 32-bit field in the
                    chunk header — preserved verbatim because we don't
                    know what the game uses it for.
    `endian`      : `'<'` for PC or `'>'` for PS3.  Affects every
                    int/float field AND the byte order of the 4-byte
                    `SDOL` magic (PS3 writes it as `'LODS'`).

    Returns the chunk as immutable `bytes`.
    """
    en = endian
    data_file_base = chunk_start + 12
    data = bytearray()

    data += struct.pack(f'{en}ii', sdol.unk_0, sdol.unk_1)
    data += struct.pack(f'{en}i',  sdol.lod_count)

    for lod in sdol.lods:
        data += struct.pack(f'{en}f', lod.lod_dist)
        data += struct.pack(f'{en}i', len(lod.vb_info))
        for vb in lod.vb_info:
            data += struct.pack(f'{en}iiii',
                vb['flags'], vb['stride'], vb['unk'], vb['offset'])

        data += struct.pack(f'{en}i', len(lod.submeshes))
        for sm in lod.submeshes:
            data += struct.pack(f'{en}iiiiiii',
                sm['vb_idx'], sm['lod_grp'], sm['sub_idx'],
                sm['idx_offset'], sm['vert_marker'], sm['unk1'], sm['unk2'])

        # -- Vertex section -----------------------------------------------
        data += struct.pack(f'{en}I', lod.vert_section_size)
        abs_p = data_file_base + len(data)
        rem   = abs_p % 16
        if rem:
            data += b'\x00' * (16 - rem)
        data += lod.vert_data

        # -- Index section ------------------------------------------------
        data += struct.pack(f'{en}I', lod.indice_section_size)
        abs_p = data_file_base + len(data)
        rem   = abs_p % 16
        if rem:
            data += b'\x00' * (16 - rem)
        data += lod.indice_data

    # unk_0 stores the total data payload size minus its own 8 bytes
    struct.pack_into(f'{en}i', data, 0, len(data) - 8)

    chunk  = bytearray()
    chunk += encode_chunk_magic('SDOL', en)   # 'SDOL' (LE) or 'LODS' (BE)
    chunk += struct.pack(f'{en}i', ci0)
    chunk += struct.pack(f'{en}i', 12 + len(data))
    chunk += data
    return bytes(chunk)
