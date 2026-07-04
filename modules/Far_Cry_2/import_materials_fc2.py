"""XBM material parser (theHunter / Avalanche "MESH" container).

XBM binary layout
==================
All integers are little-endian. FourCC tags are stored byte-reversed
(e.g. the bytes ``HSEM`` mean ``MESH``, ``LTMD`` means ``DMTL``).

File header (0x20 bytes)
    0x00  char[4]  "MESH"   (stored "HSEM")
    0x04  u16      0x2A     version major
    0x06  u16      0x06     version minor
    0x08  u32      unique id / hash (per-asset, ignore)
    0x0C  u32      0
    0x10  u32      0
    0x14  u32      payload size = filesize - 12
    0x18  u32      0
    0x1C  u32      tag count (10)

Chunks follow as a flat list. Each chunk:
    char[4]  FourCC (reversed)
    u32      version (1)
    u32      chunkSize  (whole chunk incl. this 16-byte header)
    u32      dataSize   (payload bytes)
    u8[...]  payload
Known tags: DMTL(material), NODE, SKID, SKND, CLUS, LODS, BBOX, BSPH,
LOD, PCMP, UCMP. Only DMTL is relevant for Blender material recreation.

DMTL payload
    9 reserved bytes (zero)
    string  material name
    string  shader template ("Generic", "Skin", ...)
    then SIX typed property groups, identified by position (no type tag):
        group 0  textures  : u32 count, then count * (string value, string key)
        group 1  float1    : u32 count, then count * (string key, 1 float)
        group 2  float2    : u32 count, then count * (string key, 2 floats)  -> UV tiling (u,v)
        group 3  float3    : u32 count, then count * (string key, 3 floats)  -> RGB colors
        group 4  float4    : u32 count, then count * (string key, 4 floats)  -> RGBA / color+intensity
        group 5  int       : u32 count, then count * (string key, 1 u32)     -> bools / ids / priorities

A "string" is: u32 length, <length> ASCII bytes, 1 NUL terminator.

The property dict this produces is the full material definition needed to
rebuild the shader as Blender nodes (textures, base colors, specular,
emission, tiling, two-sided, alpha, etc.).
"""

import struct
import os
import re


class XBMMaterialData:
    def __init__(self):
        # --- legacy fields kept for nodes.py compatibility ---
        self.textures = {}            # category -> path  (diffuse/normal/specular/bio/...)
        self.illumination_color = None
        self.diffuse_tiling = 1.0
        self.specular_tiling = 1.0
        self.normal_tiling = 1.0
        # --- full parsed material (for faithful node recreation) ---
        self.name = None              # material name, e.g. "HEXAPEDE_BODY"
        self.template = None          # shader template, e.g. "Generic" / "Skin"
        self.properties = {}          # every key -> value (float tuples, ints, strings)
        self.texture_slots = {}       # original texture key -> path (e.g. "NormalTexture2")


# Texture key (exact, lower-case) -> Blender category.
# Categories mirror the AAA.fx / Skin.fx sampler roles so nodes.py can
# rebuild the real shader graph. DiffuseTexture1 vs DiffuseTexture2 etc.
# are kept DISTINCT (collapsing them was the npc "no skin" bug).
_TEX_CATEGORY_EXACT = {
    'diffusetexture1': 'diffuse',
    'diffusetexture2': 'diffuse2',
    'diffusetexture': 'diffuse',          # Water
    'skintexture': 'diffuse',             # Skin shader base map
    'masktexture0': 'mask0',              # FC2 Vehicle — distinct from MaskTexture1
    'masktexture1': 'mask',
    'diffusemasktexture': 'diffuse_mask',  # Water
    'speculartexture1': 'specular',
    'speculartexture': 'specular',
    'normaltexture1': 'normal',
    'normaltexture': 'normal',
    'normaltexture2': 'normal2',
    'illuminationtexture': 'emission',
    'tattootexture': 'tattoo',
    'bloodtexture': 'blood',
    'rimlighttexture': 'rim',
    'reflectioncubetexture': 'reflection',
    'reflectiontexture': 'reflection',
    'realreflectiontexture': 'reflection',
    # Verified against every shader parameters.inc.fx across all 1583 xbm:
    'lighttexture': 'light',              # aaa.fx: globalOcclusion = LightTexture.g
    'glowtexture': 'glow',                # additive glow -> emission
    'burntdiffusetexture': 'diffuse_burnt',  # RealtreeTrunk burn blend
    'specularid': 'specular_id',          # aaaleaf specular id/mask
    'masktexturebroken': 'mask_broken',   # Weapon damaged-state mask
    # Decal alpha
    'alphatexture1': 'alpha',
    'alphatexture1wrap': 'alpha',
    # FC2-specific texture slots (not used by any Avatar shader):
    'printtexture': 'print',              # FC2 Cloth — pattern/print overlay
    'fabrictexture': 'fabric',            # FC2 Cloth — alternate fabric base
}


def _read_string(data, p):
    n = struct.unpack('<I', data[p:p + 4])[0]
    p += 4
    if n > 4096 or p + n > len(data):
        raise ValueError("string length out of range")
    s = data[p:p + n].decode('ascii', errors='replace')
    p += n
    # skip NUL terminator if present
    if p < len(data) and data[p] == 0:
        p += 1
    return s, p


def _categorize(tex_key):
    k = tex_key.lower()
    if k in _TEX_CATEGORY_EXACT:
        return _TEX_CATEGORY_EXACT[k]
    # loose fallback for unseen variants
    if 'normal' in k:
        return 'normal2' if k.endswith('2') else 'normal'
    if 'specular' in k:
        return 'specular'
    if 'mask' in k:
        return 'mask'
    if 'illumination' in k or 'glow' in k:
        return 'emission'
    if 'reflection' in k:
        return 'reflection'
    if 'light' in k:
        return 'light'
    if 'burnt' in k:
        return 'diffuse_burnt'
    if 'diffuse' in k or 'skin' in k:
        return 'diffuse2' if k.endswith('2') else 'diffuse'
    return None


class XBMParser:
    @staticmethod
    def parse(fp, lhd=True):
        try:
            with open(fp, 'rb') as f:
                data = f.read()
            result = XBMMaterialData()
            if not XBMParser._parse_structured(data, result):
                # Structured parse failed: fall back to the old heuristic
                # scraper so a malformed/variant file still imports.
                XBMParser._legacy_extract_textures(data, result)
                XBMParser._legacy_extract_illumination_color(data, result)
                XBMParser._legacy_extract_tiling(data, result)
            XBMParser._find_missing_textures(result, fp, lhd)
            return result
        except Exception:
            return None

    # ---------------- structured parser ----------------

    @staticmethod
    def _parse_structured(data, result):
        idx = data.find(b'LTMD')           # "DMTL" reversed
        if idx == -1:
            return False
        try:
            # 16-byte chunk header + 9-byte reserved material preamble
            p = idx + 16 + 9
            result.name, p = _read_string(data, p)
            result.template, p = _read_string(data, p)

            # group 0: textures (value, key) pairs
            count = struct.unpack('<I', data[p:p + 4])[0]
            p += 4
            if count > 256:
                return False
            for _ in range(count):
                value, p = _read_string(data, p)
                key, p = _read_string(data, p)
                result.properties[key] = value
                result.texture_slots[key] = value
                cat = _categorize(key)
                if cat and cat not in result.textures:
                    result.textures[cat] = value

            # groups 1..4: float properties (1,2,3,4 components)
            for ncomp in (1, 2, 3, 4):
                count = struct.unpack('<I', data[p:p + 4])[0]
                p += 4
                if count > 1024:
                    return False
                for _ in range(count):
                    key, p = _read_string(data, p)
                    vals = struct.unpack('<%df' % ncomp,
                                         data[p:p + 4 * ncomp])
                    p += 4 * ncomp
                    result.properties[key] = vals if ncomp > 1 else vals[0]

            # group 5: integer / boolean properties
            count = struct.unpack('<I', data[p:p + 4])[0]
            p += 4
            if count > 1024:
                return False
            for _ in range(count):
                key, p = _read_string(data, p)
                result.properties[key] = struct.unpack('<I', data[p:p + 4])[0]
                p += 4

            XBMParser._map_legacy_fields(result)
            return bool(result.name)
        except Exception:
            return False

    @staticmethod
    def _map_legacy_fields(result):
        """Populate the legacy scalar fields nodes.py still reads."""
        props = result.properties

        # 'bio' kept as an alias of 'emission' for _find_missing_textures
        # and any external callers that still look for it.
        if 'emission' in result.textures and 'bio' not in result.textures:
            result.textures['bio'] = result.textures['emission']

        # Tiling: file stores a (u, v) pair; legacy code expects a scalar.
        for prop, attr in (('DiffuseTiling1', 'diffuse_tiling'),
                           ('SpecularTiling1', 'specular_tiling'),
                           ('NormalTiling1', 'normal_tiling')):
            v = props.get(prop)
            if isinstance(v, (tuple, list)) and v:
                setattr(result, attr, v[0])
            elif isinstance(v, (int, float)):
                setattr(result, attr, float(v))

        # Illumination color: RGBA in file; nodes.py wants normalized RGB.
        ic = props.get('IlluminationColor1')
        if isinstance(ic, (tuple, list)) and len(ic) >= 3:
            r, g, b = ic[0], ic[1], ic[2]
            m = max(r, g, b, 1.0)
            result.illumination_color = (r / m, g / m, b / m)

    # ---------------- legacy fallback ----------------

    @staticmethod
    def _legacy_extract_textures(data, result):
        found_textures = {}
        base_textures = []
        for match in re.finditer(rb'graphics[/\\][^\x00]{10,200}\.xbt', data):
            try:
                path = match.group().decode('ascii', errors='ignore')
                basename = os.path.basename(path).lower()
                is_mip0 = '_mip0.xbt' in basename
                tex_type = None
                if '_d.xbt' in basename or '_d_mip0.xbt' in basename:
                    tex_type = 'diffuse'
                elif '_n.xbt' in basename or '_n_mip0.xbt' in basename:
                    tex_type = 'normal'
                elif '_s.xbt' in basename or '_s_mip0.xbt' in basename:
                    tex_type = 'specular'
                elif '_m.xbt' in basename or '_m_mip0.xbt' in basename:
                    tex_type = 'bio'
                else:
                    base_textures.append((path, is_mip0))
                    continue
                if tex_type not in found_textures:
                    found_textures[tex_type] = {'mip0': None, 'regular': None}
                found_textures[tex_type]['mip0' if is_mip0 else 'regular'] = path
            except Exception:
                continue
        if 'diffuse' not in found_textures and base_textures:
            mip0_base = [t for t in base_textures if t[1]]
            regular_base = [t for t in base_textures if not t[1]]
            if mip0_base:
                found_textures['diffuse'] = {'mip0': mip0_base[0][0], 'regular': None}
            elif regular_base:
                found_textures['diffuse'] = {'mip0': None, 'regular': regular_base[0][0]}
        for tex_type, versions in found_textures.items():
            result.textures[tex_type] = versions['mip0'] or versions['regular']

    @staticmethod
    def _legacy_extract_illumination_color(data, result):
        for term in [b'IlluminationColor1', b'illuminationcolor1']:
            pos = data.find(term)
            if pos != -1:
                val_pos = pos + len(term)
                while val_pos < len(data) and data[val_pos] == 0:
                    val_pos += 1
                if val_pos + 12 <= len(data):
                    try:
                        r = struct.unpack('<f', data[val_pos:val_pos + 4])[0]
                        g = struct.unpack('<f', data[val_pos + 4:val_pos + 8])[0]
                        b = struct.unpack('<f', data[val_pos + 8:val_pos + 12])[0]
                        max_val = max(r, g, b, 1.0)
                        result.illumination_color = (r / max_val, g / max_val, b / max_val)
                        return
                    except Exception:
                        pass

    @staticmethod
    def _legacy_extract_tiling(data, result):
        for search_term, attr_name in [
            (b'DiffuseTiling1', 'diffuse_tiling'),
            (b'SpecularTiling1', 'specular_tiling'),
            (b'NormalTiling1', 'normal_tiling')
        ]:
            pos = data.find(search_term)
            if pos != -1:
                val_pos = pos + len(search_term)
                while val_pos < len(data) and data[val_pos] == 0:
                    val_pos += 1
                if val_pos + 4 <= len(data):
                    try:
                        value = struct.unpack('<f', data[val_pos:val_pos + 4])[0]
                        if 0.001 < abs(value) < 1000:
                            setattr(result, attr_name, value)
                    except Exception:
                        pass

    # ---------------- texture resolution (unchanged) ----------------

    @staticmethod
    def _find_missing_textures(result, xbm_filepath, lhd=True):
        xbm_dir = os.path.dirname(xbm_filepath)
        data_folder = xbm_dir

        while data_folder and os.path.basename(data_folder).lower() != 'data':
            parent = os.path.dirname(data_folder)
            if parent == data_folder:
                break
            data_folder = parent

        if not data_folder or not os.path.exists(data_folder):
            return

        reference_texture = None
        for tex_type in ['diffuse', 'normal', 'specular', 'bio']:
            if tex_type in result.textures:
                reference_texture = result.textures[tex_type]
                break

        if not reference_texture:
            return

        basename = os.path.basename(reference_texture).lower().replace('.xbt', '')
        for suffix in ['_d', '_n', '_s', '_m', '_mip0']:
            if basename.endswith(suffix):
                basename = basename[:-len(suffix)]
                break

        texture_dir = os.path.dirname(reference_texture)

        texture_types = [
            ('diffuse', '.xbt', '_mip0.xbt', '_d.xbt', '_d_mip0.xbt'),
            ('normal', '_n.xbt', '_n_mip0.xbt'),
            ('specular', '_s.xbt', '_s_mip0.xbt'),
            ('bio', '_m.xbt', '_m_mip0.xbt')
        ]

        for tex_type_info in texture_types:
            tex_type = tex_type_info[0]
            suffixes = tex_type_info[1:]

            if tex_type in result.textures:
                current_path = result.textures[tex_type]
                if lhd and '_mip0.xbt' not in current_path.lower():
                    for suffix in suffixes:
                        if '_mip0' in suffix:
                            potential_path = texture_dir + '/' + basename + suffix
                            full_path = os.path.join(data_folder, potential_path.replace('\\', os.sep).replace('/', os.sep))
                            if os.path.exists(full_path):
                                result.textures[tex_type] = potential_path
                                break
            else:
                for suffix in suffixes:
                    if lhd and '_mip0' in suffix:
                        potential_path = texture_dir + '/' + basename + suffix
                        full_path = os.path.join(data_folder, potential_path.replace('\\', os.sep).replace('/', os.sep))
                        if os.path.exists(full_path):
                            result.textures[tex_type] = potential_path
                            break
                    else:
                        potential_path = texture_dir + '/' + basename + suffix
                        full_path = os.path.join(data_folder, potential_path.replace('\\', os.sep).replace('/', os.sep))
                        if os.path.exists(full_path):
                            result.textures[tex_type] = potential_path
                            break
