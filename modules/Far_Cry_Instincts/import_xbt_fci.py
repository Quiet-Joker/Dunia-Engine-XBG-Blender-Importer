"""Far Cry Instincts (Xbox) .xbt texture -> Blender image.

Reverse-engineered 2026-07-01 (see FarCryInstincts/fci_texture_export.py for
the standalone/bulk version this is ported from). 32-byte header: magic
0x03040100, width, height, payload_size, format_id, then raw DXT-compressed
(or, for a minority of formats, Xbox-swizzled uncompressed) pixel data.

Only the DXT1 / DXT3-5-like formats (the large majority) decode correctly
here -- they're stored linearly on Xbox, matching PC DDS layout byte-for-
byte. The uncompressed 16/32bpp formats are Xbox-tile-swizzled and are not
de-swizzled (not yet cracked); importing one of those produces a garbled
image rather than failing outright.
"""
import struct
import os
import tempfile

MAGIC = 0x03040100


def _block_dims(w, h, block_px=4):
    return (max(1, (w + block_px - 1) // block_px),
            max(1, (h + block_px - 1) // block_px))


def _base_level_bytes(w, h, format_id):
    if format_id == 16:
        bx, by = _block_dims(w, h)
        return bx * by * 8, 'DXT1'
    if format_id in (14, 17, 18):
        bx, by = _block_dims(w, h)
        return bx * by * 16, 'DXT5'
    if format_id == 12:
        return w * h * 4, 'RGBA32'
    if format_id == 8:
        return w * h * 2, 'RGB565'
    return None, None


def build_dds_bytes(xbt_data):
    """Return (dds_bytes, width, height, kind) or None if unrecognized."""
    if len(xbt_data) < 32:
        return None
    magic, w, h, payload_size, fmt = struct.unpack_from('<5I', xbt_data, 0)
    if magic != MAGIC or 32 + payload_size != len(xbt_data):
        return None
    nbytes, kind = _base_level_bytes(w, h, fmt)
    if kind is None or nbytes > payload_size:
        return None
    pixels = xbt_data[32:32 + nbytes]

    flags = 0x1 | 0x2 | 0x4 | 0x1000
    pitch_or_linsize = nbytes
    if kind in ('DXT1', 'DXT5'):
        flags |= 0x80000
        pf_flags = 0x4
        fourcc = kind.encode('ascii')
        rgb_bitcount = 0
        masks = (0, 0, 0, 0)
    else:
        flags |= 0x8
        pitch_or_linsize = w * (4 if kind == 'RGBA32' else 2)
        pf_flags = 0x40 | (0x1 if kind == 'RGBA32' else 0)
        fourcc = b'\x00\x00\x00\x00'
        if kind == 'RGBA32':
            rgb_bitcount = 32
            masks = (0x00FF0000, 0x0000FF00, 0x000000FF, 0xFF000000)
        else:
            rgb_bitcount = 16
            masks = (0xF800, 0x07E0, 0x001F, 0)

    header = b'DDS '
    header += struct.pack('<7I', 124, flags, h, w, pitch_or_linsize, 0, 0)
    header += b'\x00' * 44
    header += struct.pack('<2I', 32, pf_flags)
    header += fourcc
    header += struct.pack('<5I', rgb_bitcount, *masks)
    header += struct.pack('<4I', 0x1000, 0, 0, 0)
    header += struct.pack('<I', 0)

    return header + pixels, w, h, kind


def load_xbt_as_blender_image(xbt_path, image_name=None):
    """Decode a .xbt file and load it as a bpy.data.images entry.

    Blender has no native DDS loader either, so this writes a temp .dds and
    lets Blender's own image loader (which DOES understand DDS) read it --
    same trick the Avatar importer uses for its .xbt textures.
    """
    import bpy
    data = open(xbt_path, 'rb').read()
    result = build_dds_bytes(data)
    if result is None:
        return None
    dds_bytes, w, h, kind = result
    name = image_name or os.path.splitext(os.path.basename(xbt_path))[0]
    tmp_dir = tempfile.gettempdir()
    tmp_path = os.path.join(tmp_dir, f"fci_tex_{name}_{os.getpid()}.dds")
    with open(tmp_path, 'wb') as f:
        f.write(dds_bytes)
    try:
        img = bpy.data.images.load(tmp_path, check_existing=False)
        img.name = name
        img.pack()  # embed so the temp file can be removed safely
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
    return img
