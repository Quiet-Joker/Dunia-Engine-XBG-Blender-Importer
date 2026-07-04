import struct


# ---------------------------------------------------------------------------
# Endianness
# ---------------------------------------------------------------------------
# XBG files exist in two byte-orders:
#   - Little-endian ("LE", '<'): PC, Xbox 360 retail PC ports
#   - Big-endian    ("BE", '>'): PS3
#
# Empirically (confirmed by diffing the viperwolf PC vs PS3 .xbg side-by-side):
#   * The 16-byte file preamble (offset 0..15) is byte-identical between
#     platforms.  Only the "HSEM" magic and a couple of opaque metadata words
#     live there — they are not endian-flipped.
#   * Everything from offset 16 onward (file-size word, chunk count, every
#     chunk header, every chunk payload) is stored in the native byte order
#     of the target platform.
#   * Chunk magic strings (SDOL, EDON, PMCP, LTMR, DNKS, XOBB, HPSB, DIKS,
#     PMCU, MB2O) are stored as 32-bit FOURCC values.  On LE the bytes read
#     as the reverse of the logical name; on BE they read forward.  Because
#     the existing codebase compares against the LE byte order ("SDOL"
#     etc.), reading on BE must reverse the 4-byte name back to the
#     canonical form before comparison.  Likewise, writing a magic on BE
#     means writing the reversed bytes ('LODS' instead of 'SDOL').
#
# Detection is automatic — see `detect_endian()`.

LE = '<'
BE = '>'


def detect_endian(file_path):
    """Inspect the first 32 bytes of an XBG file and return '<' or '>'.

    Strategy: the chunk count lives at byte offset 28 as a 32-bit int.
    Real XBG files always have a small chunk count (< 256).  Reading it
    as both endians yields one sensible value and one garbage value;
    pick whichever falls inside [1, 255].  Falls back to little-endian
    if neither interpretation is sensible.
    """
    try:
        with open(file_path, 'rb') as f:
            head = f.read(32)
    except Exception:
        return LE
    return detect_endian_from_bytes(head)


def detect_endian_from_bytes(head):
    """Same logic as `detect_endian` but operates on already-read bytes."""
    if len(head) < 32:
        return LE
    le_cc = struct.unpack_from('<I', head, 28)[0]
    be_cc = struct.unpack_from('>I', head, 28)[0]
    le_ok = 1 <= le_cc < 256
    be_ok = 1 <= be_cc < 256
    if le_ok and not be_ok:
        return LE
    if be_ok and not le_ok:
        return BE
    # Both ambiguous (shouldn't happen for real files): prefer LE.
    return LE


def encode_chunk_magic(name, endian=LE):
    """Convert a canonical (LE-byte-order) chunk name to the raw 4 bytes
    that appear in a file of the requested endianness.  Used when WRITING
    chunk headers or scanning a file with `bytes.find` / equality checks.

    Example:
        encode_chunk_magic('SDOL', '<')  -> b'SDOL'
        encode_chunk_magic('SDOL', '>')  -> b'LODS'
    """
    b = name.encode('ascii') if isinstance(name, str) else bytes(name)
    if endian == BE:
        b = b[::-1]
    return b


def decode_chunk_magic(raw_bytes, endian=LE):
    """Inverse of `encode_chunk_magic`: convert raw 4-byte file bytes to
    the canonical (LE-byte-order) chunk name string used throughout the
    codebase.  On BE the bytes are reversed back to PC convention."""
    b = raw_bytes[::-1] if endian == BE else bytes(raw_bytes)
    return b.split(b'\x00', 1)[0].decode('utf-8', errors='ignore')


class BinaryReader:
    """Endian-aware binary reader used by the XBG parser.

    Always use this as a context manager — the underlying file handle is
    opened in __init__ and only closed in __exit__:

        with BinaryReader(path) as g:
            ...

    `endian` is '<' (little, default) or '>' (big).  If omitted, it is
    auto-detected from the file header.  Method names match the struct
    format characters (`i`/`I`/`h`/`H`/`f`/`B`/`b`) and return a tuple of
    `n` values, except `raw(n)` which returns the raw bytes and
    `word(n)` which returns a NUL-terminated string of at most `n` bytes.

    For 4-byte chunk magic identifiers use `chunk_name()` instead of
    `word(4)` — it reverses the bytes on BE so the returned string always
    matches the canonical LE name (SDOL, EDON, PMCP, …) used in the
    parser's if/elif chain.
    """

    __slots__ = ('file', '_read', '_unpack', '_seek', '_tell', 'endian', 'big_endian')

    def __init__(self, file_path: str, endian=None):
        if endian is None:
            endian = detect_endian(file_path)
        if endian not in (LE, BE):
            raise ValueError(f"endian must be '<' or '>', got {endian!r}")
        self.file = open(file_path, 'rb')
        self._read = self.file.read
        self._unpack = struct.unpack
        self._seek = self.file.seek
        self._tell = self.file.tell
        self.endian = endian
        self.big_endian = (endian == BE)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.file.close()

    def tell(self):
        return self._tell()

    def seek(self, offset, whence=0):
        self._seek(offset, whence)

    def seekpad(self, pad, type=0):
        size = self._tell()
        seek = (pad - (size % pad)) % pad
        if type == 1 and seek == 0:
            seek += pad
        if seek:
            self._seek(seek, 1)

    def i(self, n): return self._unpack(f'{self.endian}{n}i', self._read(n * 4))
    def I(self, n): return self._unpack(f'{self.endian}{n}I', self._read(n * 4))
    def h(self, n): return self._unpack(f'{self.endian}{n}h', self._read(n * 2))
    def H(self, n): return self._unpack(f'{self.endian}{n}H', self._read(n * 2))
    def f(self, n): return self._unpack(f'{self.endian}{n}f', self._read(n * 4))
    def B(self, n): return self._unpack(f'<{n}B', self._read(n))   # single-byte: endian-independent
    def b(self, n): return self._unpack(f'<{n}b', self._read(n))   # single-byte: endian-independent

    def raw(self, n):
        """Return n raw bytes without unpacking — used for bulk vertex reads."""
        return self._read(n)

    def word(self, length):
        """Read `length` bytes and return the UTF-8 string up to the first NUL."""
        return self._read(length).split(b'\x00', 1)[0].decode('utf-8', errors='ignore')

    def chunk_name(self):
        """Read a 4-byte chunk magic and return its canonical (LE byte-order)
        string form.  On a BE reader the bytes are reversed before decoding
        so the parser's chunk dispatch never has to care about endianness."""
        raw = self._read(4)
        if self.big_endian:
            raw = raw[::-1]
        return raw.split(b'\x00', 1)[0].decode('utf-8', errors='ignore')
