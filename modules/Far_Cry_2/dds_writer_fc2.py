"""DDS writer with uncompressed RGBA *and* DXT1 / DXT5 (BC1 / BC3) output.

Game .xbt files store textures as DXT1 (no alpha, 4 bpp) or DXT5 (alpha, 8 bpp).
We auto-pick the right format based on whether the source RGBA actually has
any alpha < 255.  The uncompressed A8R8G8B8 path is kept as a fallback for
environments without numpy and as a debug-friendly option.

DXT encoder is a numpy-vectorised "range fit": for each 4x4 block we pick the
min/max RGB as endpoints, build a 4-color palette by linear interpolation in
RGB565 space, and assign each pixel to the nearest entry.  Quality is on par
with nvcompress fast mode and *much* better than the typical paint.net export
because we operate on the full float palette before quantising the indices.

No external dependencies beyond numpy (which Blender ships).  Pure-Python
fallback writes uncompressed A8R8G8B8 (same as before) so unit tests without
numpy still pass.
"""

import struct

try:
    import numpy as _np
    _HAS_NP = True
except ImportError:
    _HAS_NP = False


# DDS flags
_DDSD_CAPS = 0x1
_DDSD_HEIGHT = 0x2
_DDSD_WIDTH = 0x4
_DDSD_PITCH = 0x8
_DDSD_PIXELFORMAT = 0x1000
_DDSD_LINEARSIZE = 0x80000
_DDPF_RGB = 0x40
_DDPF_ALPHAPIXELS = 0x1
_DDPF_FOURCC = 0x4
_DDSCAPS_TEXTURE = 0x1000


def _dds_header(width, height, pitch_or_linear_size, *,
                pixel_flags, fourcc=b'\x00\x00\x00\x00',
                rgb_bit_count=0, r_mask=0, g_mask=0, b_mask=0, a_mask=0,
                linear=False):
    """Build the 124-byte DDS header (DX9 form) for a top-level mip only."""
    flags = (_DDSD_CAPS | _DDSD_HEIGHT | _DDSD_WIDTH | _DDSD_PIXELFORMAT
             | (_DDSD_LINEARSIZE if linear else _DDSD_PITCH))
    header = bytearray(124)
    struct.pack_into('<I', header, 0, 124)                      # dwSize
    struct.pack_into('<I', header, 4, flags)
    struct.pack_into('<I', header, 8, height)
    struct.pack_into('<I', header, 12, width)
    struct.pack_into('<I', header, 16, pitch_or_linear_size)
    struct.pack_into('<I', header, 20, 0)                       # depth
    struct.pack_into('<I', header, 24, 0)                       # mipCount
    # pixel format @ offset 72, 32 bytes
    struct.pack_into('<I', header, 72, 32)                      # pfSize
    struct.pack_into('<I', header, 76, pixel_flags)
    header[80:84] = fourcc
    struct.pack_into('<I', header, 84, rgb_bit_count)
    struct.pack_into('<I', header, 88, r_mask)
    struct.pack_into('<I', header, 92, g_mask)
    struct.pack_into('<I', header, 96, b_mask)
    struct.pack_into('<I', header, 100, a_mask)
    struct.pack_into('<I', header, 104, _DDSCAPS_TEXTURE)       # caps1
    return bytes(header)


# ---------------------------------------------------------------------------
# Uncompressed A8R8G8B8 (legacy / fallback)
# ---------------------------------------------------------------------------

def build_dds_rgba(width, height, rgba_bytes):
    """Uncompressed 32-bpp BGRA.  Largest output; always works."""
    if len(rgba_bytes) != width * height * 4:
        raise ValueError("rgba_bytes must be width*height*4")
    src = memoryview(rgba_bytes)
    out = bytearray(len(rgba_bytes))
    out[0::4] = src[2::4]    # B
    out[1::4] = src[1::4]    # G
    out[2::4] = src[0::4]    # R
    out[3::4] = src[3::4]    # A
    header = _dds_header(width, height, width * 4,
                         pixel_flags=_DDPF_RGB | _DDPF_ALPHAPIXELS,
                         rgb_bit_count=32,
                         r_mask=0x00FF0000, g_mask=0x0000FF00,
                         b_mask=0x000000FF, a_mask=0xFF000000)
    return b'DDS ' + header + bytes(out)


# ---------------------------------------------------------------------------
# DXT encoder helpers (numpy vectorised)
# ---------------------------------------------------------------------------

def _to_block_view(arr_uint8):
    """Reshape (h, w, c) into (bh, bw, 4, 4, c) 4x4 blocks.  Pads h/w up to
    the next multiple of 4 by repeating the bottom/right edge (DXT requires
    block alignment; this keeps the visual edge color and is what most game
    tools do)."""
    h, w, c = arr_uint8.shape
    ph = (4 - h % 4) % 4
    pw = (4 - w % 4) % 4
    if ph or pw:
        arr_uint8 = _np.pad(arr_uint8, ((0, ph), (0, pw), (0, 0)), mode='edge')
        h, w = arr_uint8.shape[:2]
    bh, bw = h // 4, w // 4
    return (arr_uint8.reshape(bh, 4, bw, 4, c)
                     .swapaxes(1, 2)
                     .reshape(bh, bw, 16, c))


def _rgb_to_565(rgb_uint8):
    """uint8 RGB -> uint16 R5G6B5.  Standard bit-truncation quantisation."""
    r = (rgb_uint8[..., 0].astype(_np.uint16) >> 3) & 0x1F
    g = (rgb_uint8[..., 1].astype(_np.uint16) >> 2) & 0x3F
    b = (rgb_uint8[..., 2].astype(_np.uint16) >> 3) & 0x1F
    return (r << 11) | (g << 5) | b


def _565_to_rgb_float(c565):
    """uint16 R5G6B5 -> float (r, g, b) in 0..255 range, matching the GPU
    hardware decoder's expansion (replicate high bits into low bits).  Using
    the exact GPU-recovered colors lets the index search reflect what the
    block will actually render to."""
    r5 = (c565 >> 11) & 0x1F
    g6 = (c565 >> 5)  & 0x3F
    b5 = c565        & 0x1F
    r = ((r5 << 3) | (r5 >> 2)).astype(_np.float32)
    g = ((g6 << 2) | (g6 >> 4)).astype(_np.float32)
    b = ((b5 << 3) | (b5 >> 2)).astype(_np.float32)
    return _np.stack([r, g, b], axis=-1)


def _pack_dxt_color_block(blocks_rgb, c0_565, c1_565):
    """Given per-block (4,3) endpoints already chosen, build the 8-byte color
    portion of a DXT1/DXT5 block: color0, color1, 16x 2-bit indices.

    `blocks_rgb` : (bh, bw, 16, 3) uint8 pixels.
    `c0_565`, `c1_565` : (bh, bw) uint16 endpoint colors in R5G6B5.
    Returns (bh, bw, 8) uint8 ready to flatten into a DDS payload.
    """
    # GPU-recovered endpoint colors (what the hardware decoder will produce).
    c0_rgb = _565_to_rgb_float(c0_565)
    c1_rgb = _565_to_rgb_float(c1_565)
    # 4-color palette: c0, c1, (2c0+c1)/3, (c0+2c1)/3
    palette = _np.stack([
        c0_rgb,
        c1_rgb,
        (2.0 * c0_rgb + c1_rgb) / 3.0,
        (c0_rgb + 2.0 * c1_rgb) / 3.0,
    ], axis=2)  # (bh, bw, 4, 3)

    # Nearest-palette index per pixel. Done as 4 small passes (one per palette
    # entry) instead of one (bh,bw,16,4,3) broadcast — that broadcast allocates
    # ~800 MB for a 4096^2 texture; this keeps peak memory ~4x smaller and runs
    # faster. argmin picks the FIRST minimum on ties, so a STRICT `<` keeps the
    # earlier index and matches argmin's result exactly.
    pix = blocks_rgb.astype(_np.float32)             # (bh, bw, 16, 3)
    idx = _np.zeros(pix.shape[:3], dtype=_np.uint8)  # (bh, bw, 16)
    best_dist = None
    for k in range(4):
        d = pix - palette[..., k, :][..., None, :]   # (bh, bw, 16, 3)
        dk = (d * d).sum(axis=-1)                     # (bh, bw, 16)
        if best_dist is None:
            best_dist = dk
        else:
            better = dk < best_dist
            idx[better] = k
            best_dist = _np.minimum(best_dist, dk)

    # Pack 16 x 2-bit indices into 4 bytes (vectorised: each byte j holds
    # idx[4j+0..3] at shifts 0,2,4,6 — non-overlapping, so OR them per byte).
    idx4 = idx.reshape(idx.shape[:2] + (4, 4))            # (bh, bw, 4 bytes, 4 idx)
    idx_bytes = (idx4[..., 0] | (idx4[..., 1] << 2)
                 | (idx4[..., 2] << 4) | (idx4[..., 3] << 6)).astype(_np.uint8)

    out = _np.zeros(idx.shape[:2] + (8,), dtype=_np.uint8)
    out[..., 0] = (c0_565 & 0xFF).astype(_np.uint8)
    out[..., 1] = ((c0_565 >> 8) & 0xFF).astype(_np.uint8)
    out[..., 2] = (c1_565 & 0xFF).astype(_np.uint8)
    out[..., 3] = ((c1_565 >> 8) & 0xFF).astype(_np.uint8)
    out[..., 4:8] = idx_bytes
    return out


def _encode_dxt_color(blocks_rgba):
    """Color portion shared by DXT1 and DXT5: pick min/max RGB per block
    as endpoints, build 4-color palette, assign indices.

    For DXT1 we keep color0 > color1 (4-color opaque mode); the DXT1
    encoder we use produces no 1-bit-alpha blocks (DXT5 handles alpha)."""
    bh, bw = blocks_rgba.shape[:2]
    rgb = blocks_rgba[..., :3]                  # (bh, bw, 16, 3) uint8
    mn = rgb.min(axis=2)                         # (bh, bw, 3)
    mx = rgb.max(axis=2)                         # (bh, bw, 3)

    c0_565 = _rgb_to_565(mx)                     # color0 = max
    c1_565 = _rgb_to_565(mn)                     # color1 = min

    # Flat / single-color blocks may quantise to c0_565 == c1_565.  When that
    # happens the 4-color palette degenerates to one entry, which is exactly
    # what we want, but DXT1 reads a `c0 == c1` block as a special mode that
    # forces transparent index-3.  Nudge c1 down by one quantum so we stay
    # in normal opaque mode.
    same = c0_565 == c1_565
    if _np.any(same):
        c1_565 = _np.where(same & (c1_565 > 0), c1_565 - 1, c1_565).astype(_np.uint16)
        # If color0 was 0 too, bump color0 up to keep c0 > c1.
        c0_565 = _np.where(same & (c1_565 == 0) & (c0_565 == 0),
                           1, c0_565).astype(_np.uint16)

    # If quantisation flipped the order (c1 > c0), swap so c0 stays the
    # "max" endpoint.  DXT1 4-color mode requires c0 > c1; if c0 <= c1 the
    # hardware decoder switches to 3-color + 1-bit-alpha mode and pixels
    # mapped to index 3 become transparent black.
    flip = c0_565 < c1_565
    if _np.any(flip):
        c0_565, c1_565 = (_np.where(flip, c1_565, c0_565).astype(_np.uint16),
                          _np.where(flip, c0_565, c1_565).astype(_np.uint16))

    return _pack_dxt_color_block(rgb, c0_565, c1_565)


def _encode_dxt5_alpha(blocks_rgba):
    """8-byte alpha block per 4x4 region: alpha0 (max), alpha1 (min) followed
    by 16 x 3-bit indices into an 8-level palette.

    8-level mode (alpha0 > alpha1) palette per the BC3 spec:
        pal[0] = alpha0
        pal[1] = alpha1
        pal[k] = ((8-k) * alpha0 + (k-1) * alpha1) / 7   for k = 2..7

    Note: pal[1] is alpha1 directly, NOT a linear interpolation step.  An
    earlier version of this encoder used `pal[k] = a0 - k*span/7` which
    placed pal[1] at (6*a0+a1)/7 — off by one slot — and introduced up to
    18-unit alpha error on gradients.  The weight tables below match what
    the GPU hardware decoder actually does.
    """
    a = blocks_rgba[..., 3]                       # (bh, bw, 16) uint8
    a0 = a.max(axis=2).astype(_np.uint8)          # (bh, bw)
    a1 = a.min(axis=2).astype(_np.uint8)

    # Per-index weights for the 8-level (a0 > a1) palette.  Multiplying by
    # 7.0 in the denominator (vs integer //7) keeps the float palette aligned
    # with what the GPU produces — the GPU's truncating divide rounds to
    # zero, but using float minimises encode-side mismatch since we pick the
    # nearest entry by absolute difference.
    w0 = _np.array([7, 0, 6, 5, 4, 3, 2, 1],
                   dtype=_np.float32).reshape(1, 1, 8)
    w1 = _np.array([0, 7, 1, 2, 3, 4, 5, 6],
                   dtype=_np.float32).reshape(1, 1, 8)
    a0f = a0.astype(_np.float32)[..., None]       # (bh, bw, 1)
    a1f = a1.astype(_np.float32)[..., None]
    pal = (a0f * w0 + a1f * w1) / 7.0             # (bh, bw, 8)

    # Nearest-level index per pixel.
    diff = a.astype(_np.float32)[..., None] - pal[..., None, :]  # (bh, bw, 16, 8)
    idx = _np.abs(diff).argmin(axis=-1).astype(_np.uint8)        # (bh, bw, 16)

    # Pack 16 x 3-bit indices into 6 bytes (48 bits total).  Layout: indices
    # of pixels 0..15 in little-endian, with pixel 0 in the low 3 bits.
    # (vectorised: 3-bit fields at shifts 0,3,...,45 are non-overlapping, so a
    # shifted sum == OR-accumulate over the 16 indices.)
    bits = (idx.astype(_np.uint64) & 0x7)                        # (bh, bw, 16)
    _shifts = (3 * _np.arange(16, dtype=_np.uint64))             # 0,3,...,45
    packed = (bits << _shifts).sum(axis=-1)                      # (bh, bw) uint64
    # Split 48-bit packed value into 6 bytes (little-endian).
    six = _np.stack([
        ((packed >> (8 * k)) & 0xFF).astype(_np.uint8) for k in range(6)
    ], axis=-1)                                                  # (bh, bw, 6)

    out = _np.zeros(idx.shape[:2] + (8,), dtype=_np.uint8)
    out[..., 0] = a0
    out[..., 1] = a1
    out[..., 2:8] = six
    return out


def _encode_dxt1(rgba_arr):
    """RGBA uint8 (h, w, 4) -> DXT1 payload bytes.  Alpha is dropped."""
    h, w = rgba_arr.shape[:2]
    blocks = _to_block_view(rgba_arr)             # (bh, bw, 16, 4)
    color = _encode_dxt_color(blocks)             # (bh, bw, 8)
    return color.tobytes()


def _encode_dxt5(rgba_arr):
    """RGBA uint8 (h, w, 4) -> DXT5 payload bytes."""
    blocks = _to_block_view(rgba_arr)             # (bh, bw, 16, 4)
    alpha = _encode_dxt5_alpha(blocks)            # (bh, bw, 8)
    color = _encode_dxt_color(blocks)             # (bh, bw, 8)
    # DXT5 block layout = alpha(8) + color(8)
    out = _np.concatenate([alpha, color], axis=-1)
    return out.tobytes()


def _has_alpha(rgba_arr):
    """Any pixel with alpha < 255 ?"""
    return bool((rgba_arr[..., 3] < 255).any())


# ---------------------------------------------------------------------------
# Public DXT entry point
# ---------------------------------------------------------------------------

def build_dds_dxt(width, height, rgba_bytes):
    """Compress to DXT1 (no alpha) or DXT5 (has alpha) and wrap as DDS.

    Auto-picks BC1 when every pixel has alpha == 255, BC3 otherwise.
    Pads to a 4-pixel boundary by repeating the edge if width/height aren't
    multiples of 4 (the engine's texture loader doesn't care about the pad).

    Falls back to `build_dds_rgba` when numpy is unavailable.
    """
    if not _HAS_NP:
        return build_dds_rgba(width, height, rgba_bytes)
    if len(rgba_bytes) != width * height * 4:
        raise ValueError("rgba_bytes must be width*height*4")

    arr = _np.frombuffer(rgba_bytes, dtype=_np.uint8).reshape(height, width, 4)
    has_a = _has_alpha(arr)

    if has_a:
        payload = _encode_dxt5(arr)
        fourcc = b'DXT5'
        bytes_per_block = 16
    else:
        payload = _encode_dxt1(arr)
        fourcc = b'DXT1'
        bytes_per_block = 8

    # Pitch field for compressed formats is `linearSize` = bytes for the
    # top-level surface.  Compute from the (possibly padded) block grid.
    pad_w = (width + 3) & ~3
    pad_h = (height + 3) & ~3
    linear_size = (pad_w // 4) * (pad_h // 4) * bytes_per_block

    header = _dds_header(width, height, linear_size,
                         pixel_flags=_DDPF_FOURCC,
                         fourcc=fourcc,
                         linear=True)
    return b'DDS ' + header + payload


def build_dds_auto(width, height, rgba_bytes, prefer='dxt'):
    """Auto-pick compression:
       prefer='dxt' (default): DXT1/DXT5 when numpy is available, else RGBA.
       prefer='rgba'         : uncompressed 32-bpp BGRA (debug / fallback).
    """
    if prefer == 'rgba' or not _HAS_NP:
        return build_dds_rgba(width, height, rgba_bytes)
    return build_dds_dxt(width, height, rgba_bytes)
