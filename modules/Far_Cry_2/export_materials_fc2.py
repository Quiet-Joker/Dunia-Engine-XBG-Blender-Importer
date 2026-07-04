"""Custom-material exporter.

For Blender materials that have no game origin (e.g. a mesh joined into an
imported XBG), bakes the node tree to the flat maps the engine supports
and writes them out as game files:

  <Data>\\<texdir>\\<MATNAME>_d.xbt   (diffuse + alpha)
  <Data>\\<texdir>\\<MATNAME>_n.xbt   (normal)
  <Data>\\<texdir>\\<MATNAME>_s.xbt   (specular)
  <Data>\\<texdir>\\<MATNAME>_m.xbt   (illumination/emission, if emissive)
  <Data>\\graphics\\_materials\\<MATNAME>.xbm

The .xbm is a valid 'Generic' material (built from a real template via
xbm_writer, round-trip proven on all 1583 game files) that points at the
texture paths chosen in the export popup (relative to the add-on's Data
folder).

Pure file I/O lives here and is unit-testable without Blender; the bake
helpers require bpy.
"""

import os
import re
import struct

from . import xbm_builder_fc2 as xbm_builder
from .export_xbt_fc2 import build_xbt
from . import dds_writer_fc2 as dds_writer
from ..Core.debug import VerboseLogger

try:
    import numpy as _np
    _HAS_NP = True
except ImportError:
    _HAS_NP = False


def safe_name(mat_name):
    """Turn a Blender material name into a clean file/asset base name.

    Imported game materials are named with their full engine path, e.g.
    'GRAPHICS\\_MATERIALS\\GMOLLE-M-2009052838127568.xbm'. We want just
    'GMOLLE-M-2009052838127568' (no path, no extension, no separators)."""
    n = str(mat_name).replace('/', '\\').split('\\')[-1]
    if n.lower().endswith('.xbm'):
        n = n[:-4]
    n = re.sub(r'[^A-Za-z0-9_\-.]', '_', n).strip('._') or 'material'
    # MUST be lowercase: the Dunia engine lowercases asset paths before
    # its VFS hash (proven — stock UPPERCASE GMOLLE-M-*.xbm resolve
    # because the base pack was built that way), but the repacker hashes
    # the literal on-disk filename. An uppercase custom name → repacker
    # stores `Material.xbm`, engine looks up `material.xbm` → hash miss
    # → material not found → submesh silently skipped (invisible in-game,
    # fine in Blender/re-import). Lowercasing here propagates to the
    # .xbm filename, .xbt filenames, in-.xbm texture paths AND the LTMR
    # reference (all derived from this one base), keeping them mutually
    # consistent. Verified across 14 in-game-tested materials: every
    # all-lowercase name rendered, every name with an uppercase letter
    # was invisible (byte-identical otherwise, e.g. denim_test vs F_test).
    return n.lower()

try:
    import bpy
except Exception:
    bpy = None

_CHANNELS = (
    ('diffuse', '_d', 'DiffuseTexture1'),
    ('normal', '_n', 'NormalTexture1'),
    ('specular', '_s', 'SpecularTexture1'),
    ('emission', '_m', 'IlluminationTexture'),
)


# --------------------------------------------------------------------------
# File writing (no bpy)
# --------------------------------------------------------------------------

def _write_lc(out_dir, filename, data):
    """Write `data` to out_dir/filename, FORCING the on-disk name
    lowercase.

    Windows/NTFS is case-INSENSITIVE but case-PRESERVING: writing
    'foo.xbt' when 'FOO.xbt' already exists overwrites the bytes but
    keeps the OLD uppercase name on disk. The repacker then hashes the
    literal uppercase name while the engine looks it up lowercased ->
    VFS miss -> texture/material not found -> black or invisible
    (perfect in Blender / re-import, which never touch the VFS hash).
    Deleting any existing case-variant first makes the new lowercase
    name actually stick. Returns the final lowercase filename."""
    os.makedirs(out_dir, exist_ok=True)
    fn = filename.lower()
    try:
        for ex in os.listdir(out_dir):
            if ex.lower() == fn and ex != fn:
                try:
                    os.remove(os.path.join(out_dir, ex))
                except OSError:
                    pass
    except OSError:
        pass
    with open(os.path.join(out_dir, fn), 'wb') as f:
        f.write(data)
    return fn


def write_texture(data_folder, rel_dir, base_name, suffix, w, h, rgba,
                  compress='dxt'):
    """RGBA8 (top-down) -> DDS -> XBT, written under data_folder/rel_dir.

    compress :
        'dxt'  (default) - DXT1 when the source has no alpha, DXT5 when it
                           does.  Matches what real game .xbt files store
                           (BC1 4 bpp / BC3 8 bpp) and gives ~4-8x size
                           reduction vs uncompressed.
        'rgba'           - uncompressed A8R8G8B8 (legacy debug mode).
                           Larger files but bit-exact.

    Filename, on-disk directory and the returned engine path are all
    forced lowercase — the engine lowercases asset paths before its VFS
    hash but the repacker hashes the literal on-disk name (see
    `safe_name` / AGENTS.md "filename CASE" root cause)."""
    rel = (rel_dir.strip('\\/').replace('/', os.sep)
           .replace('\\', os.sep).lower())
    out_dir = os.path.join(data_folder, rel)
    dds = dds_writer.build_dds_auto(w, h, rgba, prefer=compress)
    fourcc = dds[0x54:0x58]
    fmt_label = (fourcc.decode('ascii', 'replace').strip('\x00')
                 if fourcc != b'\x00\x00\x00\x00' else 'RGBA')
    xbt = build_xbt(dds)
    safe_base = str(base_name).replace('/', '_').replace('\\', '_').lower()
    fn = _write_lc(out_dir, safe_base + suffix + '.xbt', xbt)
    VerboseLogger.log(
        f"[xbm-export] write_texture {fn} {w}x{h} format={fmt_label} "
        f"xbt_size={len(xbt)} ({len(xbt) / (w * h):.2f} B/px)")
    # engine path form: backslashed, relative to Data, lowercase
    clean = rel_dir.strip('\\/').replace('/', '\\').lower()
    return clean + '\\' + fn


def write_material_xbm(data_folder, template_type, mat_name,
                       texture_paths, diffuse_color=(0.996, 0.996, 0.996),
                       specular_color=None, specular_power=20.0,
                       emissive_power=0.0, emissive_color=None,
                       emissive_always_on=True, alpha_mode='KEEP',
                       two_sided=None, vertex_color=None,
                       unlit_blending=None):
    """Write graphics\\_materials\\<MAT>.xbm referencing texture_paths.

    Builds the XBM via xbm_builder, which clones a real game template of
    the matching (template_type, with/without illumination) variant and
    overwrites only the user-controlled fields.  Defaults that read
    'KEEP' / None preserve the template's natural value so each shader
    family gets sensible defaults out of the box (Cloth → alpha-blend +
    two-sided for hair, Leaf → alpha-test + two-sided for foliage, etc.).

      diffuse_color      (r,g,b)  tints DiffuseColor1/DiffuseColorBase
      specular_color     (r,g,b)  tints SpecularColor1/Base; None=keep template
      specular_power     float    Blinn-Phong shininess
      emissive_power     float    >0 enables glow (HDRMul for Unlit, else IlluminationColor1)
      emissive_color     (r,g,b)  IlluminationColor1.rgb; None=white power
      emissive_always_on bool     True=alpha 0 (constant), False=alpha 1 (night-only bio)
      alpha_mode         str      'KEEP' (default) / 'NONE' / 'TEST' / 'BLEND'
      two_sided          bool|None  None=keep template; bool overrides
      vertex_color       bool|None  None=keep template; bool overrides
      unlit_blending     str|None   None=keep template; 'OPAQUE'/'ADDITIVE'/'MULTIPLY' overrides (Unlit only)
    """
    base = str(mat_name).lower()
    xbm = xbm_builder.build_xbm(
        template_type, base, texture_paths,
        diffuse_color=diffuse_color, specular_color=specular_color,
        specular_power=specular_power,
        emissive_power=emissive_power, emissive_color=emissive_color,
        emissive_always_on=emissive_always_on,
        alpha_mode=alpha_mode, two_sided=two_sided,
        vertex_color=vertex_color, unlit_blending=unlit_blending)
    mat_dir = os.path.join(data_folder, 'graphics', '_materials')
    fn = _write_lc(mat_dir, f"{base}.xbm", xbm)
    return f"graphics\\_materials\\{fn}"


# Public template type list (kept as a module attribute so the UI can
# populate its dropdown from a single source of truth).
TEMPLATE_TYPES = tuple(sorted(xbm_builder.TEMPLATE_SCHEMAS.keys()))

# Feature → supporting template, for auto-detect when the user hasn't
# explicitly tagged the material.  Resolution is NODE-based, not name-
# based — the recommended template must support the textures and inputs
# that the material's node graph actually contains.  The user can always
# override via the per-material dropdown (which sets mat['xbg_template']).
#
# Supported-feature matrix per template (only including features the
# auto-detect needs to discriminate on):
#
#     Generic       : BSDF + diffuse + normal + specular + (illum) + alpha
#     Flesh         : Generic + DiffuseTexture2 (tattoo) + skinning-only
#                     features.  Manually selected when the mesh is a
#                     character body.
#     Cloth         : Generic + RimLightingColor + skinning.  Manually
#                     selected for hair / fabric.
#     Leaf          : SSS foliage (no normal map, uses SpecularID).
#                     Manually selected.
#     BigLeaf       : SSS + translucency.  Manually selected.
#     Grass         : Billboard + alpha-test.  Manually selected.
#     Weapon        : Damage-state material (Clean/Broken triples).  Manually.
#     RealtreeTrunk : Bark with burnt state.  Manually selected.
#     Unlit         : NO BSDF — pure Image Texture → Emission → Output, or
#                     emission-only material.  Auto-detected when no
#                     Principled BSDF is present.
#
# The auto rule:
#   - No BSDF in the node graph     → Unlit (flat textured)
#   - BSDF present                  → Generic (universal lit; supports
#                                              diffuse + normal + specular +
#                                              illumination + alpha; also
#                                              works on skinned meshes per
#                                              corpus data)
# Specialized templates require an explicit mat['xbg_template'] override.


def _detect_features(mat):
    """Return a set of feature tags describing the material's node graph.

    Tags (only the ones used by the auto-template resolver):
        'bsdf'     : a Principled BSDF feeds the material output
        'diffuse_tex'  : Base Color is driven by an upstream texture
        'normal_tex'   : Normal is driven by an upstream texture
        'specular_tex' : any specular-family socket is driven by a texture
        'emission'     : Emission Strength > 0 or Emission Color textured
        'alpha'        : BSDF Alpha < 1 or textured

    Doing this purely from node connectivity (not material name) is the
    correct way to pick a shader template — names are arbitrary and were
    causing misclassifications.
    """
    f = set()
    if not mat or not mat.use_nodes:
        return f
    bsdf = next((n for n in mat.node_tree.nodes
                 if n.type == 'BSDF_PRINCIPLED'), None)
    if bsdf is None:
        return f
    f.add('bsdf')

    def _linked(name):
        s = bsdf.inputs.get(name)
        return s is not None and bool(s.links)

    def _scalar_above(name, threshold):
        s = bsdf.inputs.get(name)
        if s is None or s.links:
            return False
        try:
            return float(s.default_value) > threshold
        except Exception:
            return False

    def _below(name, threshold):
        s = bsdf.inputs.get(name)
        if s is None or s.links:
            return False
        try:
            return float(s.default_value) < threshold
        except Exception:
            return False

    if _linked('Base Color'):
        f.add('diffuse_tex')
    if _linked('Normal'):
        f.add('normal_tex')
    # Specular-family inputs (Blender 3.x and 4.x names both checked).
    for n in ('Roughness', 'Coat Roughness', 'Coat Weight',
              'Specular IOR Level', 'Specular', 'Specular Tint', 'Metallic'):
        if _linked(n):
            f.add('specular_tex')
            break
    # Emission: textured input, or Emission Strength > 0 (constant), or
    # Emission Color non-black constant.
    if _linked('Emission Color') or _linked('Emission') \
            or _linked('Emission Strength'):
        f.add('emission')
    elif _scalar_above('Emission Strength', 0.0):
        f.add('emission')
    # Alpha: textured input OR constant < 1.
    if _linked('Alpha') or _below('Alpha', 0.999):
        f.add('alpha')
    return f


def resolve_template_type(mat):
    """Recommend the engine template for a Blender material — node-based.

    Priority:
      1. explicit `mat['xbg_template']` override (user-set via UI dropdown)
      2. node-graph feature detection:
           no Principled BSDF in the tree → 'Unlit' (flat textured)
           BSDF present                   → 'Generic' (universal lit)

    Specialized templates (Flesh, Cloth, Leaf, BigLeaf, Grass, Weapon,
    RealtreeTrunk) need to be selected explicitly via the per-material
    dropdown — their distinguishing features (SSS, character skinning,
    damage states, etc.) can't be reliably inferred from a Blender node
    graph alone.

    This function is INTENTIONALLY context-free.  For the export pipeline,
    `_infer_host_template(obj, mat, data_folder)` is called FIRST so that a
    custom material joined into a Flesh-templated character body inherits
    the host's template (Flesh) instead of defaulting to Generic — mixing a
    creature-class material onto a skin-class vertex buffer crashes the
    game at level-load (different GPU shader, mismatched sampler layout).
    """
    ov = (mat.get('xbg_template') or '').strip()
    for t in TEMPLATE_TYPES:
        if ov.lower() == t.lower():
            return t
    f = _detect_features(mat)
    if 'bsdf' not in f:
        return 'Unlit'
    return 'Generic'


def _read_xbm_template(path):
    """Parse just the LTMD template-name field from an .xbm on disk.

    Cheap lookup that avoids importing the full XBMParser dependency
    chain.  Returns the template string (e.g. 'Flesh', 'Cloth') or None.
    """
    try:
        with open(path, 'rb') as f:
            data = f.read()
    except OSError:
        return None
    idx = data.find(b'LTMD')
    if idx < 0:
        return None
    # 16-byte chunk header + 9-byte LTMD preamble, then two length-prefixed
    # ASCII strings: material name, then template name.  Each string is
    # `u32 len | len bytes | optional NUL`.
    p = idx + 16 + 9
    try:
        # Skip material name string.
        n = struct.unpack_from('<I', data, p)[0]
        if n > 256:
            return None
        p += 4 + n
        if p < len(data) and data[p] == 0:
            p += 1
        # Read template name string.
        n2 = struct.unpack_from('<I', data, p)[0]
        p += 4
        if n2 == 0 or n2 > 63 or p + n2 > len(data):
            return None
        return data[p:p + n2].decode('ascii', 'replace')
    except struct.error:
        return None


# Cache of recursive data_folder scans (data_folder -> {lowercase basename:
# full path}).  Keyed by absolute data_folder so multiple objects in one
# export pass share a single walk.  Reset by clearing the dict.
_XBM_DISK_INDEX_CACHE = {}


def _build_xbm_index(data_folder):
    """Walk data_folder once, indexing every .xbm by lowercase basename.

    Skipped folders: anything starting with '.' (hidden / VCS) and any path
    component named 'patch' under the data folder — patches contain our own
    re-exported XBMs which we should NOT use as ground-truth templates."""
    if not data_folder:
        return {}
    key = os.path.abspath(data_folder)
    cached = _XBM_DISK_INDEX_CACHE.get(key)
    if cached is not None:
        return cached
    index = {}
    if not os.path.isdir(key):
        VerboseLogger.log(
            f"[export_materials] data_folder not a directory: {key!r} "
            f"— host-template auto-detect disabled (set the addon's data "
            f"folder to your extracted game root, e.g. .../dist)")
        _XBM_DISK_INDEX_CACHE[key] = index
        return index
    skipped_dirs = 0
    for root, dirs, files in os.walk(key):
        # Prune hidden dirs and freshly-exported patch outputs so we don't
        # match our own re-exports as the source template.
        dirs[:] = [d for d in dirs
                   if not d.startswith('.')
                   and d.lower() != 'patch']
        skipped_dirs += sum(1 for d in dirs if d.startswith('.'))
        for fn in files:
            if fn.lower().endswith('.xbm'):
                index.setdefault(fn.lower(), os.path.join(root, fn))
    _XBM_DISK_INDEX_CACHE[key] = index
    VerboseLogger.log(
        f"[export_materials] indexed {len(index)} .xbm file(s) under "
        f"{key!r} for host-template lookup")
    return index


def _resolve_xbm_path(mat_name, data_folder, disk_index):
    """Find the on-disk path of a game .xbm referenced by mat_name.

    Tries (in order):
      1. <data_folder>/<mat_name as relative path>            (e.g. GRAPHICS\\_MATERIALS\\…)
      2. lowercased version of (1)
      3. <data_folder>/graphics/_materials/<basename>          (importer's convention)
      4. recursive walk index by lowercase basename            (handles any layout)

    Returns the absolute file path or None.
    """
    nm = str(mat_name).replace('/', '\\').strip('\\')
    if not nm.lower().endswith('.xbm'):
        nm += '.xbm'
    candidates = []
    if data_folder:
        rel = nm.replace('\\', os.sep)
        candidates.append(os.path.join(data_folder, rel))
        candidates.append(os.path.join(data_folder, rel.lower()))
        # Importer convention.
        bn = os.path.basename(rel).lower()
        candidates.append(os.path.join(data_folder, 'graphics', '_materials', bn))
    for c in candidates:
        if os.path.isfile(c):
            return c
    # Final fallback: recursive index lookup by lowercase basename.
    bn = os.path.basename(nm).lower()
    return disk_index.get(bn)


def _infer_host_template(obj, exclude_mat, data_folder):
    """Pick a template by looking at the OBJECT's other (game) material slots.

    A custom material added to a multi-material object (e.g. sovereigna
    geometry joined into a kendra body submesh that already references
    GMOLLE-…-Flesh.xbm) should inherit the host's shader family.  Otherwise
    the engine binds a mismatched GPU shader against a vertex buffer that
    was built for the host's format → level-load crash.

    Lookup order per sibling material:
      1. `mat['xbg_source_template']` — stored by nodes.setup_material at
         import time (cheap, no disk hit).  Set only on imports that ran
         AFTER the source-template-storage fix; older scenes will miss
         this and fall through to step 2.
      2. Resolve the .xbm file via `_resolve_xbm_path` (tries the engine-
         relative path under data_folder, then `graphics/_materials/`,
         then a recursive index walk).  Parse just the template string.

    Returns the most common template among siblings, or None when nothing
    usable is found (caller falls back to `resolve_template_type`).
    """
    if obj is None:
        VerboseLogger.log("[export_materials] _infer_host_template: obj=None")
        return None
    try:
        from collections import Counter
    except ImportError:
        return None
    valid = {t.lower(): t for t in TEMPLATE_TYPES}
    counts = Counter()
    sources = []
    skipped = []
    disk_index = _build_xbm_index(data_folder) if data_folder else {}
    for slot_idx, slot in enumerate(obj.material_slots):
        m = slot.material
        if m is None:
            skipped.append((slot_idx, '<empty>', 'no material'))
            continue
        if m == exclude_mat:
            skipped.append((slot_idx, m.name, 'this is the mat we are exporting'))
            continue
        if m.get('xbg_exported'):
            # Another freshly-baked custom material — useless as a signal
            # (it inherits FROM somewhere, doesn't provide a ground truth).
            skipped.append((slot_idx, m.name, 'tagged xbg_exported'))
            continue
        # Prefer the tag stored at import time.
        t = (m.get('xbg_source_template') or '').strip()
        if t and t.lower() in valid:
            counts[valid[t.lower()]] += 1
            sources.append((slot_idx, m.name, valid[t.lower()], 'tag'))
            continue
        # Fall back to disk parsing.
        path = _resolve_xbm_path(m.name, data_folder, disk_index)
        if not path:
            skipped.append((slot_idx, m.name,
                            'no xbg_source_template tag AND .xbm not found '
                            'on disk (try re-importing to refresh tags, or '
                            'point data_folder at your extracted game root)'))
            continue
        t = (_read_xbm_template(path) or '').strip()
        if not t or t.lower() not in valid:
            skipped.append((slot_idx, m.name,
                            f'parsed {path!r}: template={t!r} not in known set'))
            continue
        counts[valid[t.lower()]] += 1
        sources.append((slot_idx, m.name, valid[t.lower()], f'parsed:{path}'))

    if sources:
        VerboseLogger.log(
            f"[export_materials] '{obj.name}' host-template candidates:")
        for slot_idx, nm, tp, src in sources:
            VerboseLogger.log(f"    slot{slot_idx} {nm!r:60s} -> {tp:8s} ({src})")
    if skipped:
        VerboseLogger.log(
            f"[export_materials] '{obj.name}' host-template skips:")
        for slot_idx, nm, reason in skipped:
            VerboseLogger.log(f"    slot{slot_idx} {nm!r:60s} -- {reason}")
    if not counts:
        VerboseLogger.log(
            f"[export_materials] '{obj.name}': no host template inferred "
            f"-> falling through to BSDF feature detection (likely 'Generic'). "
            f"To override: set mat['xbg_template'] = 'Flesh' (or 'Cloth' etc.) "
            f"on the custom material in Blender's IDProperties panel.")
        return None
    winner, _ = counts.most_common(1)[0]
    VerboseLogger.log(
        f"[export_materials] '{obj.name}': inferred host template "
        f"'{winner}' (winning vote among {dict(counts)})")
    return winner


# --------------------------------------------------------------------------
# Baking (needs bpy)
# --------------------------------------------------------------------------

def _enable_gpu_or_cpu(scene):
    """Set Cycles to GPU when a compute device is available, else CPU.

    Tries each GPU backend in preference order (OPTIX > CUDA > HIP > oneAPI >
    Metal), enables every non-CPU device for that backend, and switches the
    scene to GPU. Falls back to CPU if nothing usable is found. Fully guarded —
    never raises (a bake on CPU is always correct, just slower)."""
    try:
        prefs = bpy.context.preferences.addons['cycles'].preferences
    except Exception:
        scene.cycles.device = 'CPU'
        return
    for backend in ('OPTIX', 'CUDA', 'HIP', 'ONEAPI', 'METAL'):
        try:
            prefs.compute_device_type = backend
        except (TypeError, Exception):
            continue                      # backend not supported by this build
        try:
            prefs.refresh_devices()
        except Exception:
            pass
        gpu = False
        for d in prefs.devices:
            is_gpu = (getattr(d, 'type', 'CPU') != 'CPU')
            try:
                d.use = is_gpu
            except Exception:
                pass
            gpu = gpu or is_gpu
        if gpu:
            scene.cycles.device = 'GPU'
            return
    scene.cycles.device = 'CPU'


def _img_to_rgba8(img):
    """Convert a Blender image to top-down RGBA8 bytes.

    Fast path  — numpy + foreach_get (Blender's C-level bulk read):
      foreach_get copies the float pixel buffer directly into a pre-allocated
      numpy array without creating any Python float objects.  A single
      vectorised clip+cast then replaces the double Python for-loop.
      ~1000× faster than the pure-Python path for a 2048² texture.

    Fallback — pure Python (numpy unavailable, which should never happen in
      Blender, but kept for unit-test environments without bpy).
    """
    w, h = img.size
    if _HAS_NP:
        buf = _np.empty(w * h * 4, dtype=_np.float32)
        img.pixels.foreach_get(buf)              # C-level copy, no Python objects
        buf = buf.reshape(h, w, 4)[::-1]        # flip rows: bottom-up → top-down
        return bytes(_np.clip(buf * 255.0 + 0.5, 0, 255)
                     .astype(_np.uint8).tobytes())
    # Pure-Python fallback (slow for large textures)
    px = list(img.pixels)
    out = bytearray(w * h * 4)
    for y in range(h):
        sy = h - 1 - y
        srow = sy * w * 4
        drow = y * w * 4
        for i in range(w * 4):
            v = px[srow + i]
            out[drow + i] = 0 if v <= 0 else (255 if v >= 1 else int(v * 255 + 0.5))
    return bytes(out)


def _encode_game_normal(rgba):
    """Repack a Blender-baked tangent normal into the engine's DXT5-GA form.

    Blender's NORMAL bake always outputs OpenGL convention (Y-up):
        RGBA = (X, Y, Z, 1)

    Pack for the engine (normalmap.inc.fx NORMALMAP_COMPRESSED_DXT5_GA):
        n.xy = tex2D(sampler, uv).ag   → X from ALPHA, Y from GREEN
        n.z  = sqrt(1 - dot(n.xy,n.xy))  Z is reconstructed, not stored

    No channel flip — pass X and Y through as-is:
        R = X      (unused by engine but kept for debug visibility)
        G = Y      (engine reads Y from green)
        B = 0      (Z dropped — engine reconstructs it)
        A = X      (engine reads X from alpha)
    """
    if _HAS_NP:
        arr = _np.frombuffer(rgba, dtype=_np.uint8).reshape(-1, 4).copy()
        arr[:, 3] = arr[:, 0]        # A = X  (engine reads X from alpha)
        arr[:, 2] = 0                # B = 0  (Z dropped; engine reconstructs)
        return bytes(arr.tobytes())
    out = bytearray(rgba)
    for i in range(len(out) // 4):
        out[i*4+2] = 0               # B: drop Z
        out[i*4+3] = rgba[i*4]       # A = X
    return bytes(out)


def _bsdf_and_output(nt):
    out = next((n for n in nt.nodes if n.type == 'OUTPUT_MATERIAL'
                and n.is_active_output), None) \
        or next((n for n in nt.nodes if n.type == 'OUTPUT_MATERIAL'), None)
    bsdf = None
    if out and out.inputs['Surface'].links:
        bsdf = out.inputs['Surface'].links[0].from_node
    return bsdf, out


# Ordered list of Principled BSDF input names that carry specular / surface
# roughness info.  Blender 4.x renamed several of these (e.g. 'Specular' →
# 'Specular IOR Level', 'Clearcoat' → 'Coat Weight') so we try all variants.
_SPEC_BSDF_INPUTS = (
    'Roughness', 'Coat Roughness', 'Coat Weight',
    'Specular IOR Level', 'Specular', 'Metallic', 'Specular Tint',
)

# Maps each export channel to the BSDF inputs to walk upstream from when
# looking for a source Image Texture node.
_CHANNEL_BSDF_INPUTS = {
    'diffuse':  ('Base Color',),
    'normal':   ('Normal',),
    'specular': _SPEC_BSDF_INPUTS,
    'emission': ('Emission Color', 'Emission'),
}


def _find_upstream_image(socket):
    """Walk the node tree upstream from *socket*.

    Returns the first Image Texture node's .image found, or None.
    Transparently crosses intermediate nodes such as Separate Color,
    Normal Map, RGB Curves, Math, Multiply, Overlay, etc.

    Skips TEX_IMAGE nodes whose .image was removed from bpy.data —
    accessing those raises "StructRNA of type Image has been removed".
    """
    if not socket.links:
        return None
    visited = set()
    stack = [socket.links[0].from_node]
    while stack:
        node = stack.pop()
        nid = id(node)
        if nid in visited:
            continue
        visited.add(nid)
        if node.type == 'TEX_IMAGE':
            img = getattr(node, 'image', None)
            if _img_is_alive(img):
                return img
        for inp in node.inputs:
            for lnk in inp.links:
                stack.append(lnk.from_node)
    return None


def _try_direct_copy(bsdf, ch, target_size=None):
    """Return (w, h, rgba8_bytes) by copying pixels straight from an upstream
    Image Texture node, bypassing Cycles baking entirely.

    Walks backwards from each BSDF input listed in _CHANNEL_BSDF_INPUTS[ch]
    through any intermediate nodes (Separate Color, Normal Map, RGB Curves,
    Multiply, Overlay …) until a TEX_IMAGE is found.  This correctly handles
    the common game-asset pattern where a spec texture feeds a Separate Color
    node whose channels drive Roughness / Coat Roughness / etc.

    target_size : if not None and the source image is a different resolution,
                  a temporary scaled copy is created and discarded.  Pass None
                  (SOURCE mode) to always use the image's natural resolution.

    Returns None if no upstream image is reachable (e.g. procedural chain),
    so the caller can fall back to Cycles baking.
    """
    if bsdf is None:
        return None
    for name in _CHANNEL_BSDF_INPUTS.get(ch, ()):
        inp = bsdf.inputs.get(name)
        if inp is None or not inp.links:
            continue
        img = _find_upstream_image(inp)
        if not _img_is_alive(img):
            continue
        try:
            iw, ih = img.size[0], img.size[1]
        except Exception:
            continue
        if iw == 0 or ih == 0:
            continue

        src = img
        tmp = None
        if (target_size is not None
                and (iw != target_size or ih != target_size)
                and bpy is not None):
            try:
                tmp = img.copy()
                tmp.scale(target_size, target_size)
                src = tmp
            except Exception:
                if tmp is not None:
                    try:
                        bpy.data.images.remove(tmp)
                    except Exception:
                        pass
                tmp = None
                src = img      # fall back to natural size

        try:
            rgba = _img_to_rgba8(src)
        except Exception:
            if tmp is not None:
                try:
                    bpy.data.images.remove(tmp)
                except Exception:
                    pass
            continue

        # Capture every attribute we'll need (size, name) BEFORE removing
        # the temp image. Reading `src.size` after `bpy.data.images.remove(tmp)`
        # raises "StructRNA of type Image has been removed" when `src is tmp`;
        # the original `img` reference itself is also untouched here but we
        # cached its dimensions earlier (`iw`, `ih`) and read its name now
        # while it's still guaranteed-alive.
        try:
            w, h = src.size[0], src.size[1]
        except Exception:
            # src vanished between rgba copy and now — fall back to cached
            # dims from the chosen path.
            w, h = (target_size, target_size) if tmp is not None else (iw, ih)
        try:
            src_img_nm = img.name
        except Exception:
            src_img_nm = '<removed>'

        if tmp is not None:
            try:
                bpy.data.images.remove(tmp)
            except Exception:
                pass

        if ch == 'normal':
            rgba = _encode_game_normal(rgba)
        scaled_note = (f" → {w}x{h}" if (w != iw or h != ih) else "")
        VerboseLogger.log(
            f"[xbm-export] {ch}: direct copy from '{src_img_nm}' "
            f"({iw}x{ih}{scaled_note})"
        )
        return (w, h, rgba)
    return None


def _isolate_copy(obj, mat):
    """Make a temporary single-material copy of *obj* containing ONLY the
    faces that use *mat*. Returns the temp object (caller must delete it).

    This is the fix for black/garbled bakes: Blender's bake processes the
    WHOLE object and needs the target image node in EVERY material slot,
    so a joined multi-material mesh bakes wrong. Isolating one material
    onto its own object makes every bake behave like the single-material
    case the user's manual method works on.
    """
    import bmesh
    midx = next((i for i, s in enumerate(obj.material_slots)
                 if s.material == mat), None)
    if midx is None:
        return None
    dup = obj.copy()
    dup.data = obj.data.copy()
    dup.name = "_xbgbake_" + safe_name(mat.name)
    dup.animation_data_clear()
    bpy.context.scene.collection.objects.link(dup)

    bm = bmesh.new()
    bm.from_mesh(dup.data)
    dead = [f for f in bm.faces if f.material_index != midx]
    if len(dead) == len(bm.faces):
        bm.free()
        bpy.data.objects.remove(dup, do_unlink=True)
        return None
    bmesh.ops.delete(bm, geom=dead, context='FACES')
    bm.to_mesh(dup.data)
    bm.free()
    dup.data.materials.clear()
    dup.data.materials.append(mat)
    for poly in dup.data.polygons:
        poly.material_index = 0
    return dup


def bake_material(obj, mat, size, channels, source_mode=False):
    """Bake selected channels of *mat* on *obj*. Returns
    {channel: (w, h, rgba8_bytes)}.

    Fast path — direct pixel copy
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    For each channel we first walk the node tree backwards from the relevant
    Principled BSDF input(s) looking for a source Image Texture node.  If one
    is found the pixels are copied directly — no Cycles render required.

    This correctly handles the typical game-asset pattern where a spec texture
    feeds a Separate Color node whose R/G/B outputs drive Roughness, Coat
    Roughness, etc.  Without it the old 'Specular Tint' emit-swap produced a
    flat grey because the spec texture was never connected to that input.

    Slow path — Cycles bake
    ~~~~~~~~~~~~~~~~~~~~~~~~
    Any channel whose source could not be determined (procedural textures,
    complex multi-input chains, emission with wave animation, etc.) falls back
    to a Cycles render on a temporary isolated copy of the mesh.  The specular
    fallback now iterates _SPEC_BSDF_INPUTS to find whatever connected input
    exists in the Blender 3.x / 4.x BSDF (the input names changed between
    major versions).
    """
    if bpy is None:
        raise RuntimeError("bpy unavailable")
    ctx = bpy.context
    scene = ctx.scene

    nt = mat.node_tree
    nodes, links = nt.nodes, nt.links
    bsdf, out_node = _bsdf_and_output(nt)
    results = {}

    # ── Fast path: direct pixel copy ─────────────────────────────────────
    # SOURCE mode: each texture at its own natural resolution.
    # Explicit size (512/1024/2048): scale to match the requested size.
    target_size = None if source_mode else size
    needs_bake = []
    # These channels must ALWAYS be Cycles-baked — never raw-copied from an
    # upstream image via _try_direct_copy. Base Color (diffuse), specular and
    # emission are all driven by PACKED multi-channel textures (RMO / RMOA /
    # spec maps) through Separate Color / Invert / Mix chains. _try_direct_copy
    # walks upstream and grabs the WHOLE packed texture (e.g. the RMO) and dumps
    # it straight into the slot — that's how the spider diffuse was purple
    # (RMO raw-copied as diffuse), specular was black (baked wrong channel), and
    # emission was wrong. Baking EVALUATES the full node tree result per-channel:
    #   diffuse  -> DIFFUSE pass = real albedo (mix of BC + tattoo + warpaint)
    #   specular -> emit-swap(Roughness) = actual specular response
    #   emission -> EMIT pass = real bio (inverted-alpha dots)
    # 'normal' CAN still direct-copy (it's always a plain _N texture -> Normal Map
    # node, no packed channels).
    # ALL four channels always bake — never raw-copy from upstream images.
    # Normal is included because many materials blend multiple normal maps
    # together (base + skin detail + wrinkles via Mix Normal Map groups).
    # Direct-copy would grab only the first image found upstream and miss
    # the full blend. The NORMAL bake pass evaluates the complete combined
    # result. The NORMAL bake is isolated to only the Normal socket
    # (via _KEEP['normal']) so nothing else interferes.
    _BAKE_ONLY = {'diffuse', 'normal', 'specular', 'emission'}
    for ch in channels:
        direct = (None if ch in _BAKE_ONLY
                  else _try_direct_copy(bsdf, ch, target_size=target_size))
        if direct is not None:
            results[ch] = direct
        else:
            needs_bake.append(ch)

    if not needs_bake:
        return results

    # ── Slow path: Cycles bake ────────────────────────────────────────────
    prev_engine = scene.render.engine
    scene.render.engine = 'CYCLES'
    # These passes (NORMAL / EMIT / DIFFUSE-COLOR) are DETERMINISTIC — no light
    # integration — so 1 sample gives the exact value (only sub-texel AA differs)
    # and bakes ~4x faster than 4 samples.
    scene.cycles.samples = 1
    prev_device = getattr(scene.cycles, 'device', 'CPU')
    _enable_gpu_or_cpu(scene)              # GPU if available, else CPU (safe)
    try:
        bk = scene.render.bake
        bk.margin = 4
        bk.use_clear = True
        bk.use_selected_to_active = False
        bk.use_pass_direct = False
        bk.use_pass_indirect = False
        bk.use_pass_color = True
    except Exception:
        pass

    if obj.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')

    dup = _isolate_copy(obj, mat)
    if dup is None:
        VerboseLogger.log(f"[xbm-export] '{mat.name}': no faces on object, skipped")
        scene.render.engine = prev_engine
        scene.cycles.device = prev_device
        return results

    for o in ctx.view_layer.objects:
        o.select_set(False)
    dup.select_set(True)
    ctx.view_layer.objects.active = dup

    bake_node = nodes.new('ShaderNodeTexImage')
    bake_node.select = True
    nodes.active = bake_node

    surf_src = (out_node.inputs['Surface'].links[0].from_socket
                if out_node and out_node.inputs['Surface'].links else None)
    emit = None

    # ── Per-channel BSDF input isolation ─────────────────────────────────
    # When baking one channel we DISCONNECT all other BSDF inputs so they
    # can't interfere. E.g. metallic=1 makes diffuse go dark; roughness
    # affects the normal bake; emission bleeds into diffuse, etc.
    # For each channel we define the inputs that should STAY connected;
    # everything else on the BSDF is temporarily unlinked and restored
    # after the bake.
    _KEEP = {
        # Alpha is ALWAYS kept connected across every channel — never
        # disconnected. It controls transparency cutouts (eyelashes, fur,
        # shadows) and must remain intact so the material's blend/test
        # state is preserved throughout the bake session.
        # diffuse: keep Base Color + Alpha. Everything else disconnected
        #          so the BSDF can't darken through metallic/roughness/etc.
        'diffuse':  {'Base Color', 'Alpha'},
        # normal: Normal + Alpha.
        'normal':   {'Normal', 'Alpha'},
        # specular (GLOSSY): inputs that define the specular response + Alpha.
        'specular': {'Base Color', 'Metallic', 'Roughness', 'Coat Roughness',
                     'Coat Weight', 'Specular IOR Level', 'Specular',
                     'Specular Tint', 'IOR', 'Alpha'},
        # emission: emission sockets + Alpha.
        'emission': {'Emission Color', 'Emission', 'Emission Strength', 'Alpha'},
    }

    # Safe bake defaults: the Principled BSDF's Blender defaults for every
    # socket that could interfere with baking. When we disconnect a socket
    # we ALSO reset its value here so a user-changed constant (e.g. Coat
    # Weight = 0.8, or Emission Strength = 5.0) doesn't bleed into the bake
    # even after the link is removed.
    _BSDF_DEFAULTS = {
        'Metallic':           0.0,
        'Roughness':          0.5,
        'IOR':                1.5,
        # Alpha is intentionally NOT in this dict — it is always kept
        # connected (present in every _KEEP set) and its value is NEVER
        # reset, regardless of what the user set it to. Whether it's a
        # constant 0.5 or linked to an eyelash cutout texture, it passes
        # through the bake untouched.
        'Specular IOR Level': 0.5,
        'Specular':           0.5,          # older Blender name
        'Coat Weight':        0.0,          # clearcoat off
        'Coat Roughness':     0.03,
        'Coat IOR':           1.5,
        'Sheen Weight':       0.0,
        'Sheen Roughness':    0.5,
        'Subsurface Weight':  0.0,
        'Transmission Weight':0.0,
        'Anisotropic':        0.0,
        'Anisotropic Rotation':0.0,
        'Diffuse Roughness':  0.0,
        'Emission Strength':  0.0,
        # colour sockets — use tuples; Blender assigns RGBA from a sequence
        'Emission Color':     (0.0, 0.0, 0.0, 1.0),
        'Emission':           (0.0, 0.0, 0.0, 1.0),
        'Specular Tint':      (1.0, 1.0, 1.0, 1.0),
        'Coat Tint':          (1.0, 1.0, 1.0, 1.0),
        'Sheen Tint':         (1.0, 1.0, 1.0, 1.0),
        # vector sockets (Normal, Clearcoat Normal, Tangent) — zero vector
        # means "use geometry normal / no tangent", which is the safe default.
        'Normal':             (0.0, 0.0, 0.0),
        'Clearcoat Normal':   (0.0, 0.0, 0.0),
        'Coat Normal':        (0.0, 0.0, 0.0),
        'Tangent':            (0.0, 0.0, 0.0),
    }

    def _disconnect_others(keep_set):
        """Unlink all BSDF inputs NOT in keep_set AND reset their values to
        safe bake defaults so user-modified constants don't interfere.

        Returns a list of (socket, saved_links, saved_value) tuples so
        _reconnect() can restore everything afterward.
        """
        if bsdf is None:
            return []
        saved = []
        for inp in bsdf.inputs:
            if inp.name in keep_set:
                continue
            # ── save + remove links ──────────────────────────────────────
            inp_links = []
            for lk in list(inp.links):
                inp_links.append(lk.from_socket)
                links.remove(lk)
            # ── save + reset default value ───────────────────────────────
            saved_val = None
            safe_val  = _BSDF_DEFAULTS.get(inp.name)
            try:
                dv = inp.default_value
                # save: colour/vector → list, scalar → plain value
                try:
                    saved_val = list(dv)
                except TypeError:
                    saved_val = dv
                # reset to safe default if we know one
                if safe_val is not None:
                    try:
                        inp.default_value = safe_val
                    except Exception:
                        pass
            except Exception:
                pass   # socket type has no default_value (e.g. geometry)
            saved.append((inp, inp_links, saved_val))
        return saved

    def _reconnect(saved):
        """Restore all links AND default values saved by _disconnect_others."""
        for inp, inp_links, saved_val in saved:
            # restore default value first (so any new link lands on clean state)
            if saved_val is not None:
                try:
                    inp.default_value = saved_val
                except Exception:
                    pass
            # restore links
            for from_sock in inp_links:
                try:
                    links.new(from_sock, inp)
                except Exception:
                    pass

    def emit_swap(input_name, fallback_rgba):
        nonlocal emit
        emit = nodes.new('ShaderNodeEmission')
        sock = None
        val = fallback_rgba
        if bsdf and input_name in bsdf.inputs:
            bi = bsdf.inputs[input_name]
            if bi.links:
                sock = bi.links[0].from_socket
            elif hasattr(bi, 'default_value'):
                dv = bi.default_value
                try:
                    val = (dv[0], dv[1], dv[2], 1.0)
                except TypeError:
                    val = (dv, dv, dv, 1.0)
        if sock is not None:
            links.new(sock, emit.inputs['Color'])
        else:
            emit.inputs['Color'].default_value = val
        if out_node:
            links.new(emit.outputs['Emission'], out_node.inputs['Surface'])

    def restore():
        nonlocal emit
        if emit is not None:
            nodes.remove(emit)
            emit = None
        if out_node and surf_src is not None:
            links.new(surf_src, out_node.inputs['Surface'])

    try:
        for ch in needs_bake:
            img = bpy.data.images.new(f"_bake_{safe_name(mat.name)}_{ch}",
                                      size, size, alpha=True,
                                      float_buffer=(ch == 'normal'))
            bake_node.image = img
            # Disconnect all BSDF inputs that shouldn't influence this bake.
            saved_links = _disconnect_others(_KEEP.get(ch, set()))
            try:
                if ch == 'normal':
                    # Blender ALWAYS outputs OpenGL convention from a NORMAL
                    # bake regardless of the source texture convention — the
                    # Normal Map node's OpenGL/DirectX toggle converts the
                    # source to OpenGL internally before baking. So the user
                    # sets the toggle correctly for their texture and we
                    # always get clean OpenGL output. No convention detection
                    # needed here — _encode_game_normal handles the DXT5-GA
                    # repacking (X → Alpha channel).
                    bpy.ops.object.bake(type='NORMAL')
                elif ch == 'emission':
                    # Only Emission sockets connected → pure bio/glow.
                    bpy.ops.object.bake(type='EMIT')
                elif ch == 'specular':
                    # Bake the roughness value via emit_swap, then INVERT it
                    # to produce a specular INTENSITY map:
                    #   low roughness (smooth) → bright → high specular ✓
                    #   high roughness (rough)  → dark  → low specular  ✓
                    # Avatar's SpecularTexture1 is a per-pixel intensity mask
                    # (lerped with SpecularColorBase/Color1). GLOSSY COLOR bake
                    # was wrong here — it outputs near-white for non-metallic
                    # surfaces (achromatic PBR highlights), making everything
                    # maximally glossy in-game regardless of actual roughness.
                    spec_input = next(
                        (n for n in _SPEC_BSDF_INPUTS
                         if bsdf and bsdf.inputs.get(n)
                         and bsdf.inputs[n].links),
                        None)
                    emit_swap(spec_input or 'Roughness',
                              (0.5, 0.5, 0.5, 1.0))
                    bpy.ops.object.bake(type='EMIT')
                    restore()
                else:  # diffuse
                    # Only Base Color is connected (_disconnect_others above
                    # already removed everything else). The DIFFUSE COLOR pass
                    # now gets the pure base color with zero BSDF interference —
                    # no metallic darkening, no roughness Oren-Nayar shift, no
                    # IOR energy-conservation dimming, no normal-map tilt.
                    # All the Mix/ColorRamp/group/war-paint nodes still evaluate
                    # correctly because they feed into Base Color, not the
                    # sockets we disconnected.
                    bpy.ops.object.bake(type='DIFFUSE', pass_filter={'COLOR'})
                rgba = _img_to_rgba8(img)
                if ch == 'normal':
                    rgba = _encode_game_normal(rgba)
                elif ch == 'specular':
                    # Invert the baked roughness: bright rough areas become dark
                    # (low specular), dark smooth areas become bright (high
                    # specular). Invert RGB only — leave alpha untouched.
                    ba = bytearray(rgba)
                    for pi in range(len(ba) // 4):
                        ba[pi*4]   = 255 - ba[pi*4]
                        ba[pi*4+1] = 255 - ba[pi*4+1]
                        ba[pi*4+2] = 255 - ba[pi*4+2]
                    rgba = bytes(ba)

                results[ch] = (size, size, rgba)
            except Exception as e:
                VerboseLogger.log(f"[xbm-export] bake {ch} failed: {e}")
                restore()
            finally:
                # Always restore disconnected links before next channel.
                _reconnect(saved_links)
                bpy.data.images.remove(img)
    finally:
        restore()
        try:
            nodes.remove(bake_node)
        except Exception:
            pass
        try:
            mesh_data = dup.data
            bpy.data.objects.remove(dup, do_unlink=True)
            if mesh_data.users == 0:
                bpy.data.meshes.remove(mesh_data)
        except Exception:
            pass
        scene.render.engine = prev_engine
        scene.cycles.device = prev_device
    return results


def _img_is_alive(img):
    """True if `img` is a valid (non-removed) bpy.types.Image.

    Image-Texture nodes can keep a Python reference to a bpy.types.Image
    whose underlying datablock has already been removed; touching any
    attribute on it raises "StructRNA of type Image has been removed".
    We guard every attribute access by reading `.size` inside try/except.
    """
    if img is None:
        return False
    try:
        _ = img.size[0]
        return True
    except Exception:
        return False


def _source_size(mat, default=1024):
    """'Same as source' size: largest image dimension used by the
    material's nodes (square, power-of-two-ish), else *default*."""
    best = 0
    if mat.use_nodes:
        for n in mat.node_tree.nodes:
            img = getattr(n, 'image', None)
            if not _img_is_alive(img):
                continue
            try:
                if img.size[0] and img.size[1]:
                    best = max(best, img.size[0], img.size[1])
            except Exception:
                # Image got removed between the alive-check and now.
                continue
    return best if best else default


def export_object_materials(obj, data_folder, output_folder, tex_dir,
                            size='SOURCE',
                            channels=('diffuse', 'normal', 'specular',
                                      'emission'),
                            only_custom=True,
                            force_type=None,
                            template_overrides=None,
                            emissive_always_on=True):
    """Bake + write every (custom) material on *obj*.

    data_folder    : extracted source (read-only) — used only to find a
                     template .xbm; never written to.
    output_folder  : where files are physically written, mirroring the
                     engine-relative structure (this becomes patch.pak).
    tex_dir        : ONE engine-relative folder all baked .xbt go into
                     (e.g. 'graphics\\av_characters\\corp\\npc_kendra').
    size           : int, or 'SOURCE' to match each material's source
                     texture resolution (falls back to 1024).
    Returns {mat_name: xbm_engine_path}.  Engine path strings written
    into the .xbm/.xbt are identical regardless of output_folder.
    """
    # Build fingerprint: bumps with every behavioural change so the user
    # can grep the log to confirm the freshest code is the one Blender
    # actually loaded.  If you don't see this line in your verbose log,
    # Blender is running a cached older version — restart it or use
    # "Reload Scripts" / disable+re-enable the addon.
    VerboseLogger.log(
        f"[export_materials] build: HOST-TEMPLATE-INHERITANCE + DXT-AUTO v1  "
        f"obj={obj.name!r} data_folder={data_folder!r}")

    # Texture folder is user-controlled. The engine resolves texture
    # paths anywhere in the VFS (proven: stock gmolle…hair.xbm sits in
    # graphics\_materials but references textures in av_characters\corp\
    # _textures and baltazar\ and renders fine). Co-location was NOT the
    # invisible-material cause (that was the non-conformant Generic
    # schema, since fixed). Honour whatever folder the caller passed.
    tex_dir = (tex_dir or 'graphics\\_materials').strip().strip('\\/')
    # Clear the per-export-pass disk index cache so an in-place edit of
    # the data folder gets picked up between successive export calls.
    _XBM_DISK_INDEX_CACHE.clear()
    written = {}
    for slot in obj.material_slots:
        mat = slot.material
        if not mat or not mat.use_nodes:
            continue
        if only_custom and _is_game_material(mat, data_folder):
            # Original game material — leave it untouched; the injector
            # will keep referencing its existing .xbm.
            continue
        # Skip duplicate material slots — when two slots share the exact
        # same material name (e.g. "MI_Prolemuris" for both arms and body),
        # the second slot would re-bake identical textures and overwrite
        # the first export, wasting time and potentially causing material
        # binding issues in the injector. The first slot's baked files and
        # .xbm are already on disk; the second slot can reuse them.
        _slot_base = safe_name(mat.name)
        if _slot_base in written:
            VerboseLogger.log(
                f"[export_materials] '{mat.name}' already exported as "
                f"'{_slot_base}' — skipping duplicate slot (reusing existing)")
            continue
        # Per-material engine template resolution order:
        #   1. explicit per-material override from `template_overrides`
        #      (set by the UI dropdown for THIS material specifically)
        #   2. `force_type` — global setting applied to every custom mat
        #      in this export pass
        #   3. NEW: host-template inheritance — when the custom material
        #      lives alongside a game material on the SAME object (the
        #      "joined into an existing submesh" case), copy that material's
        #      template family.  Mixing a Generic creature material onto a
        #      Flesh-skinned host vertex buffer crashes the level-load.
        #   4. `mat['xbg_template']` IDProperty override
        #   5. node-graph feature detection (Generic/Unlit fallback)
        overrides = template_overrides or {}
        ui_override = overrides.get(mat.name)
        host_inferred = _infer_host_template(obj, mat, data_folder)
        node_detected = resolve_template_type(mat)
        ttype = (ui_override
                 or force_type
                 or host_inferred
                 or node_detected)
        # Decision rationale: record WHICH rule won so a noob's log shows
        # exactly why a template ended up being what it is.  Critical for
        # debugging "I picked Flesh in the dropdown, why did my .xbm say
        # Generic?" — usually because they didn't trigger UI invoke()
        # which is what populates template_overrides.
        why = ('ui_dropdown_override' if ui_override
               else 'force_type_arg' if force_type
               else 'host_template_inheritance' if host_inferred
               else 'node_feature_detection')
        if ttype not in xbm_builder.TEMPLATE_SCHEMAS:
            raise RuntimeError(
                f"unknown template type {ttype!r}; "
                f"known: {sorted(xbm_builder.TEMPLATE_SCHEMAS)}")
        VerboseLogger.log(
            f"[export_materials] '{mat.name}' -> template '{ttype}' "
            f"(reason: {why}; "
            f"ui_override={ui_override!r}, force_type={force_type!r}, "
            f"host_inferred={host_inferred!r}, node_detected={node_detected!r})")
        try:
            from ..Core.debug import TraceLogger as _TL
            _TL.struct("template_resolution", {
                "material":          mat.name,
                "resolved_template": ttype,
                "decision":          why,
                "ui_override":       ui_override,
                "force_type":        force_type,
                "host_inferred":     host_inferred,
                "node_detected":     node_detected,
            })
        except Exception:
            pass
        base = safe_name(mat.name)
        src_mode = (size == 'SOURCE')
        msize = _source_size(mat) if src_mode else int(size)

        # Trim the channel list to only what this material actually uses.
        # detect_features() already reads the node tree, so we skip any bake
        # pass the material can't produce — saves a full Cycles render per
        # skipped channel. Rules:
        #   diffuse  : always bake (every material has a surface colour).
        #   normal   : only if a Normal socket is linked (has a normal map).
        #   specular : only if any roughness/spec/metallic socket is linked
        #              OR any spec-family socket has a non-trivial constant.
        #   emission : only if emission is detected (linked strength/color,
        #              or Strength constant > 0).
        feats = _detect_features(mat)
        active_channels = []
        for ch in channels:
            if ch == 'diffuse':
                active_channels.append(ch)          # always
            elif ch == 'normal':
                if 'normal_tex' in feats:
                    active_channels.append(ch)
                else:
                    VerboseLogger.log(f"  [{mat.name}] skipping normal bake (no normal map linked)")
            elif ch == 'specular':
                if 'specular_tex' in feats:
                    active_channels.append(ch)
                else:
                    VerboseLogger.log(f"  [{mat.name}] skipping specular bake (no specular/roughness/metallic linked)")
            elif ch == 'emission':
                if 'emission' in feats:
                    active_channels.append(ch)
                else:
                    VerboseLogger.log(f"  [{mat.name}] skipping emission bake (Emission Strength = 0 / unlinked)")
            else:
                active_channels.append(ch)          # unknown channel: keep

        baked = bake_material(obj, mat, msize, active_channels, source_mode=src_mode)
        if not baked:
            continue
        tex_paths = {}
        for ch, suffix, key in _CHANNELS:
            if ch not in baked:
                continue
            w, h, rgba = baked[ch]
            if ch == 'emission' and max(rgba) < 3:
                # Baked emission is all-black — the Blender material has no
                # Emission node or it outputs nothing.  Skip writing a black
                # xbt; _resolve_slot will then fall back to the diffuse
                # texture for the IlluminationTexture slot, so the whole
                # surface glows when IlluminationColor1 is applied.
                continue
            tex_paths[key] = write_texture(
                output_folder, tex_dir, base, suffix, w, h, rgba)
        # ------------------------------------------------------------------
        # Read Blender material settings -> xbm_builder kwargs.
        #
        # Game shader inputs we extract:
        #   DiffuseColor1/Base : BSDF 'Base Color' (constant only — when a
        #                        texture is plugged the per-pixel result is
        #                        already in the baked diffuse xbt).
        #   SpecularColor1     : BSDF 'Specular Tint' blends white<->base;
        #                        BSDF 'Metallic' >=0.5 also tints by base.
        #   SpecularPower      : Blinn-Phong shininess derived from
        #                        Roughness (or Coat Roughness if connected
        #                        to Roughness) via power = (1-r)^2 * 128,
        #                        with Coat Weight adding a bonus.
        #   IlluminationColor1 : 'Emission Color' (constant) combined with
        #                        'Emission Strength' as a power multiplier.
        #   AlphaTestEnabled   : BSDF 'Alpha' < 1.0 with mat.blend_method
        #                        == 'CLIP'.
        #   AlphaBlendEnabled  : Alpha < 1.0 with mat.blend_method
        #                        in {'HASHED', 'BLEND'}.
        #   TwoSided           : not mat.use_backface_culling.
        #
        # Sockets that ARE textured (have links) are skipped for the
        # constant read — the texture itself already encodes the value.
        # ------------------------------------------------------------------
        bsdf = next((n for n in mat.node_tree.nodes
                     if n.type == 'BSDF_PRINCIPLED'), None)

        dcol = (0.996, 0.996, 0.996)
        scol = None                       # None -> keep schema default
        spow = 20.0
        epow = 0.0
        ecol = None
        amode = 'NONE'
        two_sided = not getattr(mat, 'use_backface_culling', True)
        vcol = False

        def _socket_val(name):
            """Return the BSDF socket's default_value if unconnected, else None."""
            inp = bsdf.inputs.get(name) if bsdf else None
            if inp is None or inp.links:
                return None
            try:
                return inp.default_value
            except Exception:
                return None

        if bsdf:
            # Base Color (constant fallback when no diffuse texture).
            c = _socket_val('Base Color')
            if c is not None:
                dcol = (float(c[0]), float(c[1]), float(c[2]))

            # Roughness -> SpecularPower (Blinn-Phong approximation).
            # Avatar's SpecularPower is a scalar — no per-pixel roughness slot.
            # We compute it from the AVERAGE roughness of the material.
            # If the roughness socket is a plain constant, read it directly.
            # If it's texture-driven (e.g. RMO.R → Separate Color → Roughness),
            # walk upstream to find the source image and sample its average
            # pixel value for the relevant channel — this gives a meaningful
            # SpecularPower instead of the wrong socket default (which was
            # always ~0.03, making every material near-mirror at power=120).
            def _roughness_from_texture(sock_name):
                """Return average roughness [0,1] by sampling the upstream
                texture for a linked Roughness socket, or None if unreachable."""
                inp = bsdf.inputs.get(sock_name) if bsdf else None
                if inp is None or not inp.links:
                    return None
                node = inp.links[0].from_node
                src_socket = inp.links[0].from_socket.name  # e.g. 'Red','Green','Blue'
                # Walk back through Separate Color to the source image
                if node.type in ('SEPARATE_COLOR', 'SEPRGB'):
                    col_inp = node.inputs.get('Color') or node.inputs.get(0)
                    if col_inp and col_inp.links:
                        img_node = col_inp.links[0].from_node
                        if img_node.type == 'TEX_IMAGE' and _img_is_alive(img_node.image):
                            img = img_node.image
                            try:
                                px = list(img.pixels)   # flat RGBA
                                n = len(px) // 4
                                if n == 0:
                                    return None
                                # map socket name to RGBA channel index
                                ch_idx = {'Red':0,'Green':1,'Blue':2,'Alpha':3,
                                          'R':0,'G':1,'B':2,'A':3}.get(src_socket, 0)
                                avg = sum(px[i*4 + ch_idx] for i in range(n)) / n
                                return float(avg)
                            except Exception:
                                return None
                # Direct image link
                if node.type == 'TEX_IMAGE' and _img_is_alive(node.image):
                    try:
                        px = list(node.image.pixels)
                        n = len(px) // 4
                        avg = sum(px[i*4] for i in range(n)) / n   # use R channel
                        return float(avg)
                    except Exception:
                        return None
                return None

            for rname in ('Roughness', 'Coat Roughness'):
                # Try constant value first (unlinked socket)
                r = _socket_val(rname)
                if r is not None:
                    rf = float(r)
                    if 0.0 <= rf <= 1.0:
                        spow = max(1.0, (1.0 - rf) ** 2 * 128.0)
                        break
                # Try texture-driven: sample the upstream image average
                rf = _roughness_from_texture(rname)
                if rf is not None and 0.0 <= rf <= 1.0:
                    spow = max(1.0, (1.0 - rf) ** 2 * 128.0)
                    VerboseLogger.log(
                        f"  SpecularPower: sampled '{rname}' texture avg "
                        f"roughness={rf:.3f} → power={spow:.1f}")
                    break

            # Coat Weight bumps SpecularPower up to mimic a clearcoat sheen
            # (no dedicated coat slot in the game shader so we fold it in).
            cw = _socket_val('Coat Weight')
            if cw is not None and float(cw) > 0.01:
                spow = min(128.0, spow + float(cw) * 32.0)

            # Specular Tint + Metallic -> SpecularColor1.
            # Specular Tint blends white -> base; Metallic forces base-tinted spec.
            # NOTE: In Blender 4.x+ Specular Tint is a Color (bpy_prop_array),
            # not a plain float. _scalar() safely reduces any value to a float.
            def _scalar(raw):
                """Return a single float from a socket value that may be a
                bpy_prop_array (colour), a plain float/int, or None."""
                if raw is None:
                    return 0.0
                try:
                    return float(raw)
                except TypeError:
                    try:
                        return float(raw[0])
                    except (TypeError, IndexError):
                        return 0.0

            metal = _scalar(_socket_val('Metallic'))
            # Specular Tint is RGB in Blender 4+ — use luminance (perceptual avg)
            _st_raw = _socket_val('Specular Tint')
            if _st_raw is None:
                stint = 0.0
            else:
                try:
                    # Scalar (old Blender): single float 0-1
                    stint = float(_st_raw)
                except TypeError:
                    try:
                        # Color (Blender 4+): use perceptual luminance
                        stint = 0.2126 * float(_st_raw[0]) + 0.7152 * float(_st_raw[1]) + 0.0722 * float(_st_raw[2])
                    except (TypeError, IndexError):
                        stint = 0.0
            tint_factor = max(metal, stint)
            if tint_factor > 0.01:
                white = (1.0, 1.0, 1.0)
                t = min(1.0, tint_factor)
                scol = tuple(white[i] * (1 - t) + dcol[i] * t for i in range(3))

            # Emission Strength + Emission Color -> emissive_power / color.
            es = _socket_val('Emission Strength')
            epow_reason = "default(0)"
            if es is not None:
                epow = max(0.0, float(es))
                if epow > 0:
                    epow_reason = f"BSDF Emission Strength={epow:.3f}"
            ec = _socket_val('Emission Color') or _socket_val('Emission')
            if ec is not None:
                try:
                    ecol = (float(ec[0]), float(ec[1]), float(ec[2]))
                except (TypeError, IndexError):
                    pass

            # If a non-black emission was baked into _m.xbt, force epow=1.0
            # so the IlluminationColor1 multiplier kicks in even when the
            # BSDF Emission Strength constant is 0 (texture-driven emission).
            if epow == 0.0 and 'IlluminationTexture' in tex_paths:
                epow = 1.0
                epow_reason = "texture-driven (non-black _m.xbt baked)"

            # Alpha -> AlphaTestEnabled / AlphaBlendEnabled.
            #   blend_method 'OPAQUE'        -> NONE
            #   blend_method 'CLIP'          -> TEST  (hard cutout, 1-bit)
            #   blend_method 'HASHED'/'BLEND'-> BLEND (translucent)
            bm = getattr(mat, 'blend_method', 'OPAQUE')
            alpha = _socket_val('Alpha')
            # When Alpha is linked, verify the source image actually has
            # real alpha data before enabling alpha blend/test. A common
            # setup is connecting a diffuse texture's Alpha output "just in
            # case" when the diffuse has no real alpha channel (all pixels
            # at 1.0). Treating that as transparent would incorrectly enable
            # AlphaBlend in the XBM, causing inverted-normal / sorting
            # artefacts in-game. We sample the upstream image's alpha
            # channel average; if it's all-white (avg > 0.98, i.e. no real
            # cutout data), we treat the Alpha socket as effectively = 1.0.
            alpha_socket_textured = False
            _alpha_inp = bsdf.inputs.get('Alpha') if bsdf else None
            if _alpha_inp and _alpha_inp.links:
                _alpha_img = _find_upstream_image(_alpha_inp)
                if _img_is_alive(_alpha_img):
                    try:
                        _apx = list(_alpha_img.pixels)
                        _an  = len(_apx) // 4
                        if _an:
                            # check which channel feeds Alpha
                            _a_from = _alpha_inp.links[0].from_socket.name
                            _ach = {'Alpha':3,'A':3,'Red':0,'Green':1,
                                    'Blue':2,'Color':0}.get(_a_from, 3)
                            _avg_a = sum(_apx[i*4+_ach]
                                         for i in range(min(_an, 4096))) \
                                     / min(_an, 4096)
                            if _avg_a < 0.98:
                                alpha_socket_textured = True
                            else:
                                VerboseLogger.log(
                                    f"  [{mat.name}] Alpha socket linked but "
                                    f"source '{_alpha_img.name}' ch={_a_from} "
                                    f"avg={_avg_a:.3f} (≥0.98 = no real alpha) "
                                    f"→ treating as opaque")
                    except Exception:
                        alpha_socket_textured = True  # can't check → assume real
                else:
                    alpha_socket_textured = True  # no image found → keep linked
            has_alpha = (alpha_socket_textured
                         or (alpha is not None and float(alpha) < 0.999))
            amode_reason = "no alpha detected"
            if has_alpha:
                if bm == 'CLIP':
                    amode = 'TEST'
                    amode_reason = f"BSDF alpha < 1 AND blend_method='CLIP'"
                elif bm in ('HASHED', 'BLEND'):
                    amode = 'BLEND'
                    amode_reason = f"BSDF alpha < 1 AND blend_method='{bm}'"
                else:
                    # User has alpha but didn't set Blender's blend mode —
                    # pick TEST (cutout) which is cheaper and works for
                    # foliage/fence/wings; BLEND would require sorted draws.
                    amode = 'TEST'
                    amode_reason = ("BSDF alpha<1 OR alpha-socket textured, "
                                    "blend_method=OPAQUE → defaulted to TEST")

            try:
                from ..Core.debug import TraceLogger as _TL
                _TL.struct("material_decisions", {
                    "material":         mat.name,
                    "emissive_power":   epow,
                    "emissive_reason":  epow_reason,
                    "alpha_mode":       amode,
                    "alpha_reason":     amode_reason,
                    "specular_tint":    round(float(stint), 4),
                    "metallic":         round(float(metal), 4),
                    "two_sided":        two_sided,
                })
            except Exception:
                pass

        VerboseLogger.log(
            f"[export_materials] '{mat.name}' -> '{base}.xbm'\n"
            f"  ttype={ttype}  diffuse_color={dcol}  specular_color={scol}\n"
            f"  specular_power={spow:.1f}  emissive_power={epow:.3f}  emissive_color={ecol}\n"
            f"  alpha_mode={amode}  two_sided={two_sided}  emissive_always_on={emissive_always_on}\n"
            f"  tex_paths={tex_paths}"
        )
        try:
            from ..Core.debug import TraceLogger as _TL
        except Exception:
            _TL = None
        if _TL is not None:
            _TL.kvblock(
                f"export_object_materials material '{mat.name}'",
                [
                    ("blender_name",       mat.name),
                    ("safe_base",          base),
                    ("template_type",      ttype),
                    ("xbm_will_be_at",
                     f"{output_folder}/graphics/_materials/{base}.xbm"),
                    ("texture_dir",        tex_dir),
                    ("texture_paths",      dict(tex_paths)),
                    ("diffuse_color",      dcol),
                    ("specular_color",     scol),
                    ("specular_power",     spow),
                    ("emissive_power",     epow),
                    ("emissive_color",     ecol),
                    ("alpha_mode",         amode),
                    ("two_sided",          two_sided),
                    ("vertex_color",       vcol),
                    ("only_custom",        only_custom),
                ],
                tier="DEBUG", event="export_material_record")
        # Only pass alpha/two-sided/vertex-color overrides when the user
        # explicitly set them in Blender — otherwise let the cloned
        # template keep its built-in defaults (e.g. Cloth/hair templates
        # already ship with AlphaBlend=1 + TwoSided=1; Leaf with
        # AlphaTest=1 + TwoSided=1; etc.).  Forcing these to NONE/False
        # when the Blender material is in its default state would
        # clobber those baked-in defaults and break hair / foliage / etc.
        kwargs = dict(diffuse_color=dcol, specular_color=scol,
                      specular_power=spow,
                      emissive_power=epow, emissive_color=ecol,
                      emissive_always_on=emissive_always_on)
        if amode != 'NONE':           # user has alpha → explicit override
            kwargs['alpha_mode'] = amode
        if two_sided:                 # user disabled backface culling → explicit
            kwargs['two_sided'] = True
        if vcol:                      # user requested vertex colors → explicit
            kwargs['vertex_color'] = True
        write_material_xbm(output_folder, ttype, base, tex_paths, **kwargs)
        # Rename the Blender material to its new engine .xbm path and tag
        # it, so the existing injector (which builds the XBG LTMR table
        # from mat.name) automatically points the XBG at the new .xbm.
        engine_path = 'GRAPHICS\\_MATERIALS\\' + base + '.xbm'
        try:
            mat.name = engine_path
        except Exception:
            pass
        # NOT 'xbg_source' — that would make a later re-export skip this
        # material. This tag only records what we exported it as.
        mat['xbg_exported'] = engine_path
        written[base] = engine_path
    return written


def _is_game_material(mat, data_folder):
    """True = unmodified game material; skip it when only_custom.

    A material we just exported is tagged `xbg_exported` and renamed to
    an engine `.xbm` path — that is NOT a game material (re-baking it is
    fine). An original imported game material is tagged `xbg_source`
    and/or carries an engine `.xbm` name with no `xbg_exported` tag.
    The old disk-existence test failed when Data lives in unextracted
    .pak (so stock GMOLLE-….xbm leaked through as "custom"); detect by
    tag + name instead, with the disk test only as an extra positive.
    """
    if mat.get('xbg_exported'):
        return False                       # one of ours — bake it
    if mat.get('xbg_source'):
        return True                        # imported game material
    nm = str(mat.name).replace('/', '\\')
    low = nm.lower()
    if low.endswith('.xbm') or '\\_materials\\' in low:
        # Engine .xbm-path name, not tagged as ours → treat as a stock
        # game material (don't re-bake under only_custom).
        if data_folder:
            rel = nm.strip('\\').replace('\\', os.sep)
            if os.path.isfile(os.path.join(data_folder, rel)):
                return True
        return True
    return False
