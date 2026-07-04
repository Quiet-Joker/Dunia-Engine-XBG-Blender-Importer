"""XBT texture writer for EXPORT — wraps DDS bytes in the Avalanche 'TBX'
container. The export counterpart to import_xbt.py (XBTConverter / read_dds).

Verified header (from real game .xbt files):
  0x00  'TBX\\0'                     magic
  0x04  u32 0x0000000B               format/version constant
  0x08  u32 header_size              (32 for the minimal form)
  0x0C  u32 0x00000001               constant
  0x10  u32 hash                     per-asset (CRC of source; 0 works)
  0x14  8 bytes  3E70368F04B91B47    constant CTextureResource type-id
  0x1C  u32 0                        (minimal-header tail)
  0x20  DDS ...                      raw DirectDraw Surface

The 32-byte header is exactly what the game's *_mip0.xbt files use — the safe
minimal form (larger headers only append the source path string the engine
doesn't need to load the texture).
"""

import struct

_TYPE_ID = bytes.fromhex('3e70368f04b91b47')
_MAGIC = b'TBX\x00'


def build_xbt(dds_bytes, asset_hash=0):
    """Wrap raw DDS bytes into a minimal, game-loadable .xbt."""
    if dds_bytes[:4] != b'DDS ':
        raise ValueError("not DDS data (missing 'DDS ' magic)")
    header = (_MAGIC
              + struct.pack('<I', 0x0B)
              + struct.pack('<I', 32)              # header_size
              + struct.pack('<I', 1)
              + struct.pack('<I', asset_hash & 0xFFFFFFFF)
              + _TYPE_ID
              + struct.pack('<I', 0))
    assert len(header) == 32, len(header)
    return header + dds_bytes
