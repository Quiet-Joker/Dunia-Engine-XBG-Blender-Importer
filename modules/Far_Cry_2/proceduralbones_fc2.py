"""Avatar (Dunia-1) binary-XML codec — read AND write.

Many Avatar "`.xml`" files (e.g. `databases/baltazar/proceduralbones.xml`) are
NOT text — they're a compiled Dunia-1 binary object.  It is NOT the FC2/Dunia-2
FCB, so Gibbed's tools can't read it (`BinaryResourceFile.Deserialize` throws).
This module fully implements the format (decoder verified by a byte-for-byte
semantic match against a known-good text export), so we can edit these files
ourselves — the basis for authoring custom jiggle / procedural bones.

Format (see agents.md "Avatar binary-XML codec"):
  Header (9 B): `00 00 FF` magic, u32 stringPoolSize, u8 nodeCount, u8 attrCount.
  Root node starts at offset 0x09.  String pool at EOF: null-separated; an empty
  string lives at index 1 (for empty values).  Refs = 0-based byte offsets into
  the pool, 1 byte if <0xFF else `0xFF` + u32-LE.
  Node = nameRef, 0x10, attrCount:u8, childCount:u8,
         [ if attrCount>0: 0x00, then attrCount × (nameRef, valueRef) with a
           single 0x00 terminator BETWEEN attrs (none after the last) ],
         then childCount × node.
"""

import struct
import xml.etree.ElementTree as ET

_MAGIC = b'\x00\x00\xff'
_MARK = 0x10


# ── decode ─────────────────────────────────────────────────────────────────
def decode(data):
    """Binary blob → ElementTree root Element."""
    if data[:3] != _MAGIC:
        raise ValueError("not an Avatar binary-XML object (bad magic)")
    pool_size = struct.unpack_from('<I', data, 3)[0]
    pool_start = len(data) - pool_size
    pool = data[pool_start:]
    strings = {}
    off = 0
    for chunk in pool.split(b'\x00'):
        strings[off] = chunk.decode('latin-1')
        off += len(chunk) + 1

    def S(r):
        if r not in strings:
            raise ValueError("dangling string ref 0x%x" % r)
        return strings[r]

    def ref(i):
        if data[i] == 0xFF:
            return struct.unpack_from('<I', data, i + 1)[0], i + 5
        return data[i], i + 1

    def node(i, parent):
        name, i = ref(i)
        if data[i] != _MARK:
            raise ValueError("bad node marker 0x%x at 0x%x" % (data[i], i))
        i += 1
        ac = data[i]; i += 1
        cc = data[i]; i += 1
        el = ET.Element(S(name)) if parent is None else ET.SubElement(parent, S(name))
        if ac > 0:
            i += 1                                  # leading 0x00
            for k in range(ac):
                an, i = ref(i)
                av, i = ref(i)
                el.set(S(an), S(av))
                if k < ac - 1:
                    i += 1                          # 0x00 terminator
        for _ in range(cc):
            _, i = node(i, el)
        return el, i

    root, _ = node(0x09, None)
    return root


# ── encode ─────────────────────────────────────────────────────────────────
def encode(root):
    """ElementTree root Element → binary blob (structurally valid; the game
    reads by ref so our DFS pool order need not match Ubisoft's byte order)."""
    # 1. ordered, de-duplicated string pool — root name, then "" (index 1), then
    #    DFS: node name, each attr (name, value), children.
    order = []
    seen = set()

    def add(s):
        if s not in seen:
            seen.add(s); order.append(s)

    add(root.tag)
    add("")                                         # empty string at index 1

    def collect(el):
        add(el.tag)
        for k, v in el.attrib.items():
            add(k); add(v)
        for c in el:
            collect(c)
    for c in root:
        collect(c)

    pool = bytearray()
    offset = {}
    for s in order:
        offset[s] = len(pool)
        pool += s.encode('latin-1') + b'\x00'

    def ref(s):
        o = offset[s]
        if o < 0xFF:
            return bytes((o,))
        return b'\xff' + struct.pack('<I', o)

    n_nodes = [0]
    n_attrs = [0]

    def enc(el):
        n_nodes[0] += 1
        out = bytearray(ref(el.tag))
        out.append(_MARK)
        attrs = list(el.attrib.items())
        n_attrs[0] += len(attrs)
        out.append(len(attrs))
        out.append(len(el))                         # child count
        if attrs:
            out.append(0x00)
            for k, (an, av) in enumerate(attrs):
                out += ref(an); out += ref(av)
                if k < len(attrs) - 1:
                    out.append(0x00)
        for c in el:
            out += enc(c)
        return out

    body = enc(root)
    header = _MAGIC + struct.pack('<I', len(pool)) + bytes((n_nodes[0] & 0xFF,
                                                            n_attrs[0] & 0xFF))
    return bytes(header + body + pool)


# ── convenience ──────────────────────────────────────────────────────────────
def decode_file(path):
    with open(path, 'rb') as f:
        return decode(f.read())


def encode_to_file(root, path):
    with open(path, 'wb') as f:
        f.write(encode(root))


def to_xml(root):
    try:
        ET.indent(root, '  ')                       # py3.9+
    except Exception:
        pass
    return ET.tostring(root, encoding='unicode')
