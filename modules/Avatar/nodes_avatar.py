"""Blender node graphs rebuilt from the real Avalanche engine shaders.

Logic transcribed from shaders/meta/aaa.fx (Generic + Flesh templates),
Skin.parameters, Water/Road/FX. Key recipe (AAA.fx GetDiffuseColor /
GetSpecularColor / GetNormalVectorTS, LightingPS):

    mask          = vertexMask * MaskTexture1   (defaults: r=b=1, g=0)
    base.rgb      = DiffuseTexture1.rgb * lerp(DiffuseColorBase, DiffuseColor1, mask.b)
    base.rgb      = lerp(base.rgb, DiffuseTexture2.rgb*DiffuseColor2, mask.g)   [if Diffuse2]
    base.a        = DiffuseTexture1.a
    spec          = lerp(SpecularColorBase, SpecularTexture1*SpecularColor1, mask.r)
    normalTS      = lerp((0,0,1), Uncompress(NormalTexture1), mask.b)
    normalTS      = lerp(normalTS, Uncompress(NormalTexture2), mask.g)          [if Normal2]
    emission     += IlluminationColor1.rgb * IlluminationTexture.rgb            (strength = .a)

Normal maps are DXT5 GA packed (normalmap.inc.fx): X=Alpha, Y=Green,
Z=sqrt(1-X^2-Y^2), each *2-1.
"""

import bpy
import os
import math
from bpy_extras.image_utils import load_image
from .import_materials_avatar import XBMMaterialData
from .import_xbt_avatar import XBTConverter
from ..Core.debug import VerboseLogger


_TILING_PROP = {
    'diffuse': 'DiffuseTiling1',
    'diffuse2': 'DiffuseTiling2',
    'specular': 'SpecularTiling1',
    'normal': 'NormalTiling1',
    'normal2': 'NormalTiling2',
    'emission': 'IlluminationTiling1',
    'mask': 'MaskTiling1',
    'blood': 'BloodTiling',
    'tattoo': 'TattooTiling',
}

# Per-texture UV-channel routing.  aaa.fx's SWITCH_GROUP_AND_TILING macro:
#     #define SWITCH_GROUP_AND_TILING(t) ((uv0_uv1*t).xy + (uv0_uv1*t).zw)
# where uv0_uv1 = (uv0.x, uv0.y, uv1.x, uv1.y).  So the derived
# <Slot>TilingAndGroup float4 encodes UV0 scale in .xy and UV1 scale in
# .zw — a slot reads UV0 with (tile,tile,0,0) or UV1 with (0,0,tile,tile).
#
# CRITICAL (verified against providerdescriptors/descriptors.xml): the
# Generic descriptor declares only FOUR inputs — UVGroupMapChannel0..3 —
# and the per-slot TilingAndGroup float4s are built from them in compiled
# C++ "post-loading code" we can't read.  An earlier mapping wrongly
# assumed one channel per CB slot (UVGroupMapChannel0..6) and read
# channels 4/5/6 that DON'T EXIST for this shader — which mis-routed the
# mask onto UV1 and produced the "metal mixed with mask" artifact on the
# dumptruck.
#
# Evidence-grounded model (matches the descriptor's "1" vs "2" texture-
# suffix convention AND the dumptruck's appearance + the user's feedback):
#   - BASE set   (Diffuse1, Normal1, Specular1, Mask1, Illumination) share
#                UVGroupMapChannel0 — the unique unwrap layer.
#   - DETAIL set (Diffuse2, Normal2) share UVGroupMapChannel1 — the tiled
#                wear/dirt layer (dumptruck: 6x6 on UV1).
# Channel value 0 -> UV0 ('UVMap'), 1 -> UV1 ('UVMap1').
# Result on DUMPTRUCK_METAL: metal+normal+spec+mask on UV0 (aligned base),
# dirt diffuse+normal on UV1 6x6.  Kendra: all channels 0 -> all UV0
# (byte-identical fast path, no regression).
_UVGROUP_PROP = {
    # base set -> UVGroupMapChannel0
    'diffuse':  'UVGroupMapChannel0',
    'normal':   'UVGroupMapChannel0',
    'specular': 'UVGroupMapChannel0',
    'mask':     'UVGroupMapChannel0',
    'emission': 'UVGroupMapChannel0',
    # detail set -> UVGroupMapChannel1
    'diffuse2': 'UVGroupMapChannel1',
    'normal2':  'UVGroupMapChannel1',
}
# Blender UV-layer names the importer creates (apply_uv_layer in import_xbg).
_UV_LAYER_FOR_CHANNEL = {0: 'UVMap', 1: 'UVMap1', 2: 'UVMap2'}


def _vec2(v, default=(1.0, 1.0)):
    if isinstance(v, (tuple, list)) and len(v) >= 2:
        return (float(v[0]), float(v[1]))
    if isinstance(v, (int, float)):
        return (float(v), float(v))
    return default


def _rgb(v, default=(1.0, 1.0, 1.0)):
    if isinstance(v, (tuple, list)) and len(v) >= 3:
        return (float(v[0]), float(v[1]), float(v[2]))
    return default


def _image_colored_fraction(img, max_samples=16384,
                            sat_tol=0.12, value_floor=0.05):
    """Fraction (0..1) of an image's pixels that carry genuine colour.

    A pixel is "coloured" when its HSV-style saturation exceeds `sat_tol`
    AND it's bright enough (max channel > `value_floor`, so black/near-black
    noise doesn't count).  Saturation = (max-min)/max is used instead of a
    raw channel difference so a dim-but-clearly-yellow/red region still
    registers, while a bright neutral grey does not.

    Returns -1.0 when the pixels can't be read (caller should treat as
    grayscale / unknown)."""
    try:
        if img is None or not img.size[0] or not img.size[1]:
            return 0.0
        import numpy as np
        n = img.size[0] * img.size[1]
        px = np.empty(n * 4, dtype=np.float32)
        img.pixels.foreach_get(px)            # raw stored buffer
        px = px.reshape(-1, 4)
        if n > max_samples:                   # even stride subsample
            idx = np.linspace(0, n - 1, max_samples).astype(np.int64)
            px = px[idx]
        rgb = px[:, :3]
        mx = rgb.max(axis=1)
        mn = rgb.min(axis=1)
        sat = np.where(mx > 1e-6, (mx - mn) / mx, 0.0)
        coloured = (sat > sat_tol) & (mx > value_floor)
        return float(coloured.mean())
    except Exception:
        return -1.0


def _image_is_grayscale(img, frac_tol=0.08, **kw):
    """True when an image is predominantly black-and-white.

    Used to decide where a specular map belongs on the Principled BSDF:
    a B&W map is an INTENSITY map (-> 'Specular IOR Level'); a map where a
    SIGNIFICANT SHARE of pixels are genuinely coloured carries a highlight
    tint (-> 'Specular Tint').  We test the FRACTION of coloured pixels (not
    the single most-colourful pixel) so a mostly-grey map with a few stray
    coloured/noisy texels still counts as grayscale.  Errs toward grayscale
    when pixels can't be read."""
    frac = _image_colored_fraction(img, **kw)
    if frac < 0.0:                            # unreadable -> assume grayscale
        return True
    return frac < frac_tol


def _input(node, *names):
    """Look up a socket by name, preferring the ENABLED one.

    Blender 4+/5+ `ShaderNodeMix` exposes multiple sockets with the same
    `name` — one per data type (FLOAT / VECTOR / RGBA / ROTATION).  Only
    the sockets matching the node's current `data_type` are .enabled.
    A naive `node.inputs['A']` returns the FIRST socket with that name,
    which is the FLOAT 'A' — connecting an Image Texture's Color output
    to the FLOAT input leaves the COLOR 'A' input at its default white,
    and the node evaluates `white × tint` regardless of the texture.
    That bug rendered every Mix-using material as pure white in 4.x/5.x
    while logs falsely reported "Base Color connected".

    Pass: try enabled-only by name first, then fall back to first-by-name
    so older Blenders that don't have multi-type Mix nodes still work.
    """
    for n in names:
        # Enabled-only pass — picks the right socket on multi-type nodes.
        for s in node.inputs:
            if s.name == n and getattr(s, 'enabled', True):
                return s
        # Fallback for older Blender where there's only one socket per name.
        if n in node.inputs:
            return node.inputs[n]
    return None


def _output(node, *names):
    """Same as _input but for output sockets — needed because the Mix
    node also has multiple outputs named 'Result' (one per data type)."""
    for n in names:
        for s in node.outputs:
            if s.name == n and getattr(s, 'enabled', True):
                return s
        if n in node.outputs:
            return node.outputs[n]
    return None


def _spec_roughness(power):
    try:
        return max(0.02, min(1.0, math.sqrt(2.0 / (float(power) + 2.0))))
    except Exception:
        return 0.5


def _glass_tint(props):
    """Pick the glass tint from the most-saturated of SpecularColor1 /
    DiffuseColor1.

    The engine stores the glass colour in different slots: orange BUGGY_WINDOW
    carries it in SpecularColor1 (DiffuseColor1 is black), the bluish corp
    windows carry it in DiffuseColor1 (SpecularColor1 is white/absent). Black is
    "no tint", not a colour. Returns (rgb_in_0_1, is_tinted, source_key); the
    rgb is hue-preserved (divided by its brightest channel) so a COLOR blend
    keeps the hue, and `is_tinted` is False for white/grey/absent so clear glass
    isn't desaturated.
    """
    best = None  # (saturation, rgb, key)
    for key in ('SpecularColor1', 'DiffuseColor1'):
        v = props.get(key)
        if not isinstance(v, (tuple, list)) or len(v) < 3:
            continue
        c = (float(v[0]), float(v[1]), float(v[2]))
        mx = max(c)
        if mx <= 0.0:                      # black → not a colour
            continue
        sat = mx - min(c)
        if best is None or sat > best[0]:
            best = (sat, c, key)
    if best is None:
        return (1.0, 1.0, 1.0), False, 'none'
    _, c, key = best
    m = max(c[0], c[1], c[2], 1.0)
    tint = (c[0] / m, c[1] / m, c[2] / m)
    tinted = (max(tint) - min(tint)) > 0.05
    return tint, tinted, (key if tinted else 'none')


class BlenderMaterialSetup:
    @staticmethod
    def setup_material(mat, xbm_data, data_folder, load_hd_textures=True, import_as_dds=False):
        prev_iad = mat.get('xbg_setup_iad')
        if prev_iad is not None and bool(prev_iad) == bool(import_as_dds):
            return
        if prev_iad is not None or mat.get('xbg_material_setup_complete'):
            mat.node_tree.nodes.clear()
        if not mat.use_nodes:
            mat.use_nodes = True

        nt = mat.node_tree
        nodes, links = nt.nodes, nt.links
        props = getattr(xbm_data, 'properties', {}) or {}
        tex = xbm_data.textures

        bsdf = next((n for n in nodes if n.type == 'BSDF_PRINCIPLED'), None)
        if not bsdf:
            bsdf = nodes.new('ShaderNodeBsdfPrincipled')
            bsdf.location = (300, 0)
        out = next((n for n in nodes if n.type == 'OUTPUT_MATERIAL'), None)
        if not out:
            out = nodes.new('ShaderNodeOutputMaterial')
            out.location = (650, 0)
        if not bsdf.outputs['BSDF'].links:
            links.new(bsdf.outputs['BSDF'], out.inputs['Surface'])

        tc = nodes.new('ShaderNodeTexCoord')
        tc.location = (-1700, 0)

        ctx = _Ctx(nodes, links, props, tex, tc, data_folder,
                   load_hd_textures, import_as_dds, bsdf, mat)

        # Persist the engine template name on the Blender material so the
        # exporter can later default new sibling materials to the same
        # template (e.g. a custom mesh joined into a Flesh-templated body
        # submesh must export as Flesh, not Generic — otherwise the game's
        # shader mismatch crashes the level on load).
        if xbm_data.template:
            mat['xbg_source_template'] = xbm_data.template

        template = (xbm_data.template or 'Generic').lower()

        if template == 'skin':
            BlenderMaterialSetup._build_skin(ctx)
        elif template in ('water', 'waterriver'):
            BlenderMaterialSetup._build_water(ctx)
        elif template == 'road':
            BlenderMaterialSetup._build_road(ctx)
        elif template in ('unlit', 'meshunlit'):
            BlenderMaterialSetup._build_unlit(ctx)
        elif template in ('hair', 'vehicle', 'thincloth',
                          'aaaleaf', 'leaf', 'realtreeleafpure',
                          'bigleaf', 'realtreeleafhybrid', 'realtreetrunk',
                          'weapon', 'faketerrain', 'decal', 'staticdecal'):
            BlenderMaterialSetup._build_aaa(ctx)
        elif template in ('glow', 'fx', 'particle', 'meshfx'):
            BlenderMaterialSetup._build_fx(ctx)
        elif BlenderMaterialSetup._is_glass(props, tex):
            BlenderMaterialSetup._build_glass(ctx)
        else:                                   # Generic, Flesh, Cloth, fallback
            BlenderMaterialSetup._build_aaa(ctx)

        BlenderMaterialSetup._apply_flags(mat, props, template)
        mat['xbg_setup_iad'] = int(bool(import_as_dds))
        mat['xbg_material_setup_complete'] = 1

        # ── Per-material setup summary ──────────────────────────────────
        # Log which BSDF inputs ended up connected and which textures got
        # categorised.  A material that renders white in the viewport is
        # almost always one where 'Base Color' is in the unconnected_inputs
        # list (because the diffuse texture failed to load above — see the
        # texture_load events with status != "ok" to find which one).
        try:
            from ..Core.debug import TraceLogger as _TL
            connected = []
            unconnected = []
            for sock_name in ('Base Color', 'Normal', 'Alpha', 'Specular Tint',
                              'Specular IOR Level', 'Specular', 'Roughness',
                              'Emission', 'Emission Color'):
                s = bsdf.inputs.get(sock_name)
                if s is None:
                    continue
                if s.is_linked:
                    connected.append(sock_name)
                else:
                    unconnected.append(sock_name)
            # Emission-driven materials (Unlit) deliberately leave the
            # Principled BSDF disconnected and feed the Material Output from
            # an Emission/Add-Shader chain instead — so "Base Color not
            # connected" is NOT a white-render risk for them.  Only warn when
            # the Principled is actually what drives the output.
            try:
                bsdf_drives_output = bool(bsdf.outputs['BSDF'].links)
            except Exception:
                bsdf_drives_output = True
            white_risk = ('Base Color' not in connected) and bsdf_drives_output
            tex_categories_found = sorted([k for k in tex.keys()]) if tex else []
            _TL.struct("material_node_setup", {
                "material":              mat.name,
                "template":              xbm_data.template,
                "resolved_to_builder":   template,
                "textures_categorised":  tex_categories_found,
                "bsdf_connected":        connected,
                "bsdf_unconnected":      unconnected,
                "base_color_connected":  ('Base Color' in connected),
                "bsdf_drives_output":    bsdf_drives_output,
                "warning_white_render":  white_risk,
            })
            if white_risk:
                VerboseLogger.warn(
                    f"[material_setup] '{mat.name}' Base Color NOT connected "
                    f"— material will render as the BSDF default (white-ish). "
                    f"Check the texture_load events above for which diffuse "
                    f"file failed to resolve. Texture categories found: "
                    f"{tex_categories_found}")
        except Exception as _exc:
            pass

        # ── Remove an orphaned Principled BSDF ──────────────────────────────
        # setup_material always creates a Principled BSDF, but emission-driven
        # builders (Unlit) drive the Material Output from their own shader chain
        # and leave it disconnected. Delete the dead node so (a) the node tree
        # isn't cluttered, and (b) export_materials' template detector — which
        # keys off "is there a Principled BSDF?" ("none → Unlit, present →
        # Generic") — classifies the material correctly instead of seeing a
        # leftover stub. Builders that USE the Principled keep it (its BSDF
        # output is still linked to the Material Output).
        try:
            if bsdf is not None and not bsdf.outputs['BSDF'].links:
                nodes.remove(bsdf)
        except Exception:
            pass

    # ---------------------------------------------------------------- AAA
    @staticmethod
    def _is_glass(props, tex):
        """Detect glass that should render transparent + reflective.

        Authoritative signals mined from the game data (see AGENTS.md "Glass
        material"):
          * `Glass` engine flag == 1 → the material uses the GLASS shader
            permutation (only CORP_SAMSON_ALPHA / _HELLSGATE set it, but it is
            definitive when present).
          * `LogicalMaterialId == 24` ("Glass" surface, per
            nomadlogicmaterialdefinition.xml) AND `AlphaBlendEnabled` → the
            artist tagged it glass AND it's actually alpha-blended (24 windows;
            the old heuristic only caught 8 of them — the rest rendered opaque).

        Plus the original heuristic (AlphaBlend + black DiffuseColor1 +
        reflection) as a defensive fallback for files outside the known corpus.

        Not mis-routed here: Unlit "glass" (screens/HUD, also LogicalMaterialId
        24) is dispatched to the emissive path BEFORE this check; opaque
        logical-glass (bottles, crystal, lights) has no AlphaBlend.
        """
        if props.get('Glass'):
            return True
        blend = props.get('AlphaBlendEnabled')
        if not blend:
            return False
        if props.get('LogicalMaterialId') == 24:
            return True
        # legacy fallback (subset of the above for the stock corpus, but keeps
        # working for non-corpus files lacking a LogicalMaterialId tag)
        dc1 = _rgb(props.get('DiffuseColor1'), (1.0, 1.0, 1.0))
        return dc1 == (0.0, 0.0, 0.0) and bool(tex.get('reflection'))

    @staticmethod
    def _set_transparent_blend(mat):
        """Set material to alpha-blend in both old (pre-4.2) and new (4.2+) Blender APIs."""
        if mat is None:
            return
        for attr, val in (('blend_method', 'BLEND'),
                          ('surface_render_method', 'BLENDED')):
            if hasattr(mat, attr):
                try:
                    setattr(mat, attr, val)
                except Exception:
                    pass
        for attr, val in (('shadow_method', 'HASHED'),):
            if hasattr(mat, attr):
                try:
                    setattr(mat, attr, val)
                except Exception:
                    pass

    @staticmethod
    def _wire_transparent_mix(ctx, opacity_sock):
        """Insert Mix Shader + Transparent BSDF between Principled BSDF and Material Output.

        opacity_sock: a float socket where 0=fully transparent, 1=fully opaque.
        Works with every render engine and Blender version without relying on
        blend_method or Alpha socket behaviour.
        """
        n, l, bsdf = ctx.nodes, ctx.links, ctx.bsdf

        # Find the Material Output the BSDF is already wired to
        out_node = next((nd for nd in n if nd.type == 'OUTPUT_MATERIAL'), None)
        if out_node is None:
            return

        # Transparent BSDF
        trans = n.new('ShaderNodeBsdfTransparent')
        trans.location = (bsdf.location[0], bsdf.location[1] + 200)

        # Mix Shader between transparent and principled
        mix = n.new('ShaderNodeMixShader')
        mix.location = (bsdf.location[0] + 280, bsdf.location[1] + 100)

        # Disconnect existing BSDF → Output link and rewire through Mix
        for lnk in list(bsdf.outputs['BSDF'].links):
            l.remove(lnk)

        l.new(opacity_sock,          mix.inputs[0])    # Factor (index-safe)
        l.new(trans.outputs['BSDF'], mix.inputs[1])   # fac=0 → transparent
        l.new(bsdf.outputs['BSDF'],  mix.inputs[2])   # fac=1 → principled
        l.new(mix.outputs['Shader'], out_node.inputs['Surface'])

    @staticmethod
    def _build_glass(ctx):
        """aaa.fx AlphaBlend + REFLECTION glass (e.g. BUGGY_WINDOW).

        Detected by _is_glass (AlphaBlend + black DiffuseColor1 + reflection
        cubemap). These are NOT necessarily the engine's compiled GLASS variant
        (that permutation is decided in runtime C++ we can't read); empirically
        the buggy window's transparency is the **diffuse texture's own alpha
        channel** — the standard non-GLASS AlphaBlend path, aaa.fx:377/1067
        `finalColor.a = diffuseColor.a`. So we honour the authored diffuse alpha
        instead of synthesising a fresnel curve (which is what the earlier build
        did, and which ignored the diffuse texture entirely).

        Build (Principled BSDF):
          Base Color    = DiffuseTexture1, Color-blended with SpecularColor1's
                          hue (orange for BUGGY_WINDOW; a near-white tint is a
                          no-op, so clear corp/samson glass stays clear).
          Alpha         = DiffuseTexture1 alpha (authored see-through; falls back
                          to a fresnel curve ONLY when there is no diffuse).
          Specular Tint = SpecularColor1 hue (orange highlight + env reflection).
          Roughness     = capped _spec_roughness (sharp — cubemaps are *_sharp_*).
          IOR           = 1.45 (glassy fresnel reflection of the environment).

        The previous build wired the orange into **Emission**, so the glass
        *glowed* orange in every lighting condition (the "weird patchwork").
        Glass reflects/transmits — it does not emit. Transmission is avoided on
        purpose (EEVEE needs raytrace/SSR flags to show it); alpha-blend is
        robust on every engine/version. The orange-tinted reflection comes from
        Principled's glossy lobe + Specular Tint.

        Credit: the diffuse-alpha + Color-tint approach is the user's, validated
        against the in-game look; it is more faithful than the fresnel-glossy
        version it replaces.
        """
        n, l, p, bsdf = ctx.nodes, ctx.links, ctx.props, ctx.bsdf
        y = 300

        # Tint = whichever of SpecularColor1 / DiffuseColor1 is actually
        # coloured (orange BUGGY_WINDOW → SpecularColor1; bluish corp windows →
        # DiffuseColor1; clear glass → neither). `tinted` gates the COLOR blend
        # so a white/grey tint doesn't desaturate the diffuse to greyscale.
        tint, tinted, tint_src = _glass_tint(p)

        d1 = ctx.tex('diffuse', y)
        alpha_source = "none"
        if d1:
            if tinted:
                # COLOR blend: diffuse luminance + SpecularColor1 hue/sat — the
                # user-validated way to make the whole pane read orange.
                cmix = ctx.mix('RGBA', 'COLOR', (-450, y),
                               a_sock=d1.outputs['Color'], b=tint, fac_default=1.0)
                base = ctx.out(cmix) if cmix is not None else d1.outputs['Color']
            else:
                base = d1.outputs['Color']
            l.new(base, bsdf.inputs['Base Color'])
            if 'Alpha' in d1.outputs:
                l.new(d1.outputs['Alpha'], bsdf.inputs['Alpha'])
                alpha_source = "diffuse_alpha"
        else:
            bc = bsdf.inputs.get('Base Color')
            if bc:
                bc.default_value = (*tint, 1.0)
            # No diffuse texture → no authored alpha. Synthesise fresnel opacity
            # (0.1 face-on → 1.0 grazing) so the pane is still see-through.
            a_in = bsdf.inputs.get('Alpha')
            if a_in:
                lw = n.new('ShaderNodeLayerWeight'); lw.location = (-560, y + 160)
                mul = n.new('ShaderNodeMath'); mul.operation = 'MULTIPLY'
                mul.location = (-380, y + 160); mul.inputs[1].default_value = 0.9
                l.new(lw.outputs['Facing'], mul.inputs[0])
                add = n.new('ShaderNodeMath'); add.operation = 'ADD'
                add.location = (-220, y + 160); add.inputs[1].default_value = 0.1
                l.new(mul.outputs[0], add.inputs[0])
                l.new(add.outputs[0], a_in)
                alpha_source = "fresnel"

        # Specular highlight.  Prefer the authored SpecularTexture1 when present
        # (the old glass path ignored it entirely — vehicle canopies like the
        # corp scorpion ship a richly COLOURED spec map that carries the
        # highlight colour/pattern, and dropping it lost the whole effect):
        #   * coloured spec map  -> drive Specular Tint (highlight colour).
        #   * grayscale spec map -> drive Specular IOR Level (intensity), keep
        #                           the constant glass tint on Specular Tint.
        #   * no spec map        -> constant tint from _glass_tint (old path).
        st = _input(bsdf, 'Specular Tint')
        lvl = _input(bsdf, 'Specular IOR Level', 'Specular')
        s1 = ctx.tex('specular', y - 120)
        if s1:
            sc1 = _rgb(p.get('SpecularColor1'), (1, 1, 1))
            spec_sock = s1.outputs['Color']
            if any(abs(c - 1.0) > 0.02 for c in sc1):   # modulate by SpecularColor1
                sm = ctx.mix('RGBA', 'MULTIPLY', (-250, y - 120),
                             a_sock=spec_sock, b=sc1, fac_default=1.0)
                spec_sock = ctx.out(sm) if sm is not None else spec_sock
            if not _image_is_grayscale(getattr(s1, 'image', None)):
                if st:
                    l.new(spec_sock, st)                # coloured -> Tint
            elif lvl:
                bw = n.new('ShaderNodeRGBToBW'); bw.location = (-80, y - 120)
                l.new(spec_sock, bw.inputs['Color'])
                l.new(bw.outputs['Val'], lvl)           # grayscale -> IOR Level
                if st and hasattr(st, 'default_value') and len(st.default_value) >= 3:
                    st.default_value = (*tint, 1.0)
        elif st and hasattr(st, 'default_value') and len(st.default_value) >= 3:
            st.default_value = (*tint, 1.0)

        sp = p.get('SpecularPower')
        rough = min(_spec_roughness(sp) if isinstance(sp, (int, float)) else 0.1, 0.3)
        if 'Roughness' in bsdf.inputs:
            bsdf.inputs['Roughness'].default_value = rough
        if 'IOR' in bsdf.inputs:
            bsdf.inputs['IOR'].default_value = 1.45

        BlenderMaterialSetup._set_transparent_blend(ctx.mat)

        try:
            from ..Core.debug import TraceLogger as _TL
            _TL.struct("glass_material_setup", {
                "material":     ctx.mat.name if ctx.mat else None,
                "tint":         tint,
                "tinted":       tinted,
                "tint_source":  tint_src,
                "alpha_source": alpha_source,
                "roughness":    rough,
                "has_diffuse":  bool(d1),
                "logical_glass": p.get('LogicalMaterialId') == 24,
                "glass_flag":   bool(p.get('Glass')),
            })
        except Exception:
            pass

    @staticmethod
    def _build_aaa(ctx):
        n, l, p, bsdf = ctx.nodes, ctx.links, ctx.props, ctx.bsdf
        y = 600

        mask = ctx.mask_channels(y)            # (r, g, b) sockets
        y -= 300

        # ---- Base color ----
        d1 = ctx.tex('diffuse', y)
        if d1:
            # lerp(DiffuseColorBase, DiffuseColor1, mask.b)  — per aaa.fx GetDiffuseColor
            tint = ctx.mix('RGBA', 'MIX', (-650, y + 120),
                           a=_rgb(p.get('DiffuseColorBase')),
                           b=_rgb(p.get('DiffuseColor1')),
                           fac=mask[2], fac_default=1.0)
            base = ctx.mix('RGBA', 'MULTIPLY', (-450, y),
                           a_sock=d1.outputs['Color'],
                           b_sock=ctx.out(tint), fac_default=1.0)
            base_out = ctx.out(base)

            d2 = ctx.tex('diffuse2', y - 300)
            if d2:
                # Stock-empty-tattoo detection.  Avatar ships a 4x4 white
                # `tattoo.xbt` that gets referenced as DiffuseTexture2 by
                # every character material, even when the character has
                # no actual tattoo.  In-game this is harmless because the
                # character's mesh vertex-color green is 0 on every vert
                # without a tattoo decal, so the lerp factor is 0 and the
                # blend silently does nothing.  In Blender, if the mesh
                # doesn't have a 'Col' attribute, ShaderNodeVertexColor
                # returns (1, 1, 1, 1) instead of (0, 0, 0, 1), the lerp
                # picks the tattoo branch fully, multiplies the 4x4 white
                # by DiffuseColor2 = (1.004,1.004,1.004) → model renders
                # solid white.  We harden the chain by FORCING the lerp
                # factor to a literal 0 when the diffuse2 slot is the
                # known stock empty texture — matches in-game appearance
                # whether the mesh has a Col attribute or not.
                d2_path = (ctx.tex_map.get('diffuse2') or '').lower()
                d2_base = d2_path.rsplit('\\', 1)[-1].rsplit('/', 1)[-1]
                is_stock_tattoo = d2_base.startswith('tattoo')
                d2_col = _rgb(p.get('DiffuseColor2'), (1.0, 1.0, 1.0))
                d2_col_near_white = all(abs(c - 1.0) < 0.05 for c in d2_col)

                det = ctx.mix('RGBA', 'MULTIPLY', (-450, y - 300),
                              a_sock=d2.outputs['Color'],
                              b=d2_col, fac_default=1.0)
                if is_stock_tattoo and d2_col_near_white:
                    # No tattoo → leave base_out unchanged.  Skip the
                    # Mix entirely AND log it so a user editing this
                    # material later knows why no tattoo blend is wired.
                    VerboseLogger.log(
                        f"[material_setup] '{ctx.mat.name if ctx.mat else '?'}' "
                        f"diffuse2 slot is the stock empty tattoo "
                        f"({d2_base}); DiffuseColor2={d2_col} ≈ white. "
                        f"Skipping the tattoo blend so the result matches "
                        f"the in-game appearance regardless of whether "
                        f"the mesh has a 'Col' vertex-color attribute.")
                    try:
                        from ..Core.debug import TraceLogger as _TL
                        _TL.struct("tattoo_blend_skipped", {
                            "material": ctx.mat.name if ctx.mat else None,
                            "diffuse2": d2_path,
                            "diffuse_color2": d2_col,
                            "reason": "stock_empty_tattoo",
                        })
                    except Exception:
                        pass
                else:
                    mixd = ctx.mix('RGBA', 'MIX', (-250, y - 120),
                                   a_sock=base_out, b_sock=ctx.out(det),
                                   fac=mask[1], fac_default=0.0)
                    base_out = ctx.out(mixd)

            l.new(base_out, bsdf.inputs['Base Color'])
            if 'Alpha' in d1.outputs and (p.get('AlphaTestEnabled') or p.get('AlphaBlendEnabled')):
                l.new(d1.outputs['Alpha'], bsdf.inputs['Alpha'])
            y -= 600

        # ---- Specular ----
        # The game's specular map is an INTENSITY mask (almost always
        # grayscale): it belongs on 'Specular IOR Level' (the strength
        # input), NOT on 'Specular Tint' (which colours the highlight and
        # leaves the strength at the unmodulated default).  Only the rare
        # genuinely coloured spec maps additionally tint the highlight.
        scb = _rgb(p.get('SpecularColorBase'), (0, 0, 0))
        s1 = ctx.tex('specular', y)
        st = _input(bsdf, 'Specular Tint')
        lvl = _input(bsdf, 'Specular IOR Level', 'Specular')
        spec_color_sock = None
        if s1:
            sm = ctx.mix('RGBA', 'MULTIPLY', (-450, y),
                         a_sock=s1.outputs['Color'],
                         b=_rgb(p.get('SpecularColor1')), fac_default=1.0)
            sx = ctx.mix('RGBA', 'MIX', (-250, y),
                         a=scb, b_sock=ctx.out(sm),
                         fac=mask[0], fac_default=1.0)
            spec_color_sock = ctx.out(sx)
            if lvl:
                bw = n.new('ShaderNodeRGBToBW')
                bw.location = (-80, y - 40)
                l.new(spec_color_sock, bw.inputs['Color'])
                l.new(bw.outputs['Val'], lvl)
            if st and not _image_is_grayscale(getattr(s1, 'image', None)):
                # coloured spec map -> also tint the highlight
                l.new(spec_color_sock, st)
            y -= 300
        else:
            if st and hasattr(st, 'default_value') and len(st.default_value) >= 3:
                st.default_value = (*scb, 1.0)
            if lvl:
                lvl.default_value = max(
                    0.0, min(1.0, 0.299 * scb[0] + 0.587 * scb[1] + 0.114 * scb[2]))
        sp = p.get('SpecularPower')
        spec_roughness_val = (_spec_roughness(sp)
                              if isinstance(sp, (int, float)) else 0.5)
        if isinstance(sp, (int, float)) and 'Roughness' in bsdf.inputs:
            bsdf.inputs['Roughness'].default_value = spec_roughness_val

        # ---- Normal (DXT5 GA decode + mask blend) ----
        nrm = None
        # Some materials ship only NormalTexture2 (no NormalTexture1).
        primary_normal = 'normal' if ctx.tex_map.get('normal') else 'normal2'
        nn1 = ctx.tex(primary_normal, y, non_color=True)
        if nn1:
            nrm = ctx.normal_map(nn1, (-250, y), strength=mask[2], strength_default=1.0)
            y -= 350
            nn2 = ctx.tex('normal2', y, non_color=True) if primary_normal == 'normal' else None
            if nn2:
                nrm2 = ctx.normal_map(nn2, (-250, y))
                blend = ctx.mix('VECTOR', 'MIX', (-50, y + 175),
                                a_sock=nrm, b_sock=nrm2,
                                fac=mask[1], fac_default=0.0)
                nrm = ctx.out(blend)
                y -= 350
        if nrm:
            l.new(nrm, bsdf.inputs['Normal'])

        # ---- Emission ----
        ctx.emission(y)

        # NOTE on the (removed) parallel Glossy BSDF attempt:
        #   Tried wiring a separate ShaderNodeBsdfAnisotropic in parallel
        #   to Principled (combined via Add Shader) so the game's
        #   Blinn-Phong specular got a "true" colored highlight.  Worked
        #   well for matte materials (dark SpecularColorBase) but blew
        #   out high-spec materials — EYE_STANDARD_BROWN has
        #   SpecularColorBase ≈ (0.95, 0.95, 0.95) and SpecularPower=128,
        #   so the Glossy produced a near-mirror highlight that combined
        #   with Blender's Material Preview HDRI (which lights from every
        #   direction at once) turned the corneas into white discs.
        #   Principled's Fresnel-modulated spec naturally tames this
        #   class of artist-tuned shiny materials, so we stay with that.
        #   Specular Tint already carries the colour from the spec-lerp
        #   chain; Specular IOR Level is set above from the luminance of
        #   SpecularColorBase; Roughness is the Blinn-Phong → microfacet
        #   conversion.  Approximation, but it doesn't blow out under
        #   HDRI preview.

        # ---- Auxiliary textures (FC2 Cloth/Vehicle/Generic extras) ----
        # FC2 templates ship with additional texture slots that Avatar's
        # AAA pipeline doesn't natively use: PrintTexture / FabricTexture
        # (FC2 Cloth pattern/fabric overlay), MaskTexture0 (FC2 Vehicle's
        # second dirt mask), BloodTexture / RimLightTexture (FC2 wear/sheen).
        # We surface them as auxiliary image nodes in a side frame so the
        # user can see they exist and wire them manually if desired —
        # auto-wiring would require FC2-specific shader logic that
        # Avatar's render path can't replicate anyway.
        ctx.aux_frame(['mask0', 'blood', 'rim', 'print', 'fabric'],
                      y - 350,
                      'FC2 extras (mask0 / blood / rim / print / fabric)')

    # --------------------------------------------------------------- Skin
    @staticmethod
    def _build_skin(ctx):
        n, l, p, bsdf = ctx.nodes, ctx.links, ctx.props, ctx.bsdf
        y = 500
        sk = ctx.tex('diffuse', y)              # SkinTexture
        if sk:
            base_out = sk.outputs['Color']
            skc = _rgb(p.get('SkinColor'))
            if skc != (1.0, 1.0, 1.0):
                m = ctx.mix('RGBA', 'MULTIPLY', (-450, y),
                            a_sock=base_out, b=skc, fac_default=1.0)
                base_out = ctx.out(m)
            # Tattoo decal over skin (alpha-blended).
            tt = ctx.tex('tattoo', y - 300)
            if tt and 'Alpha' in tt.outputs:
                mt = ctx.mix('RGBA', 'MIX', (-250, y - 120),
                             a_sock=base_out, b_sock=tt.outputs['Color'],
                             fac_sock=tt.outputs['Alpha'])
                base_out = ctx.out(mt)
            l.new(base_out, bsdf.inputs['Base Color'])
            y -= 600

        # Subsurface (SubsurfaceColor)
        ss = _rgb(p.get('SubsurfaceColor'), (0, 0, 0))
        if ss != (0, 0, 0):
            sc = _input(bsdf, 'Subsurface Color')
            if sc and hasattr(sc, 'default_value'):
                sc.default_value = (*ss, 1.0)
            sw = _input(bsdf, 'Subsurface Weight', 'Subsurface')
            if sw:
                sw.default_value = 1.0

        # Specular / roughness
        scb = _rgb(p.get('SpecularColorBase'), (0, 0, 0))
        st = _input(bsdf, 'Specular Tint')
        if st and hasattr(st, 'default_value') and len(st.default_value) >= 3:
            st.default_value = (*scb, 1.0)
        sp = p.get('SpecularPower')
        if isinstance(sp, (int, float)) and 'Roughness' in bsdf.inputs:
            bsdf.inputs['Roughness'].default_value = _spec_roughness(sp)

        # Normals: NormalTexture1 blended with NormalTexture2 by NormalBlendFactor
        nrm = None
        nn1 = ctx.tex('normal', y, non_color=True)
        if nn1:
            nrm = ctx.normal_map(nn1, (-250, y))
            y -= 350
            nn2 = ctx.tex('normal2', y, non_color=True)
            if nn2:
                nrm2 = ctx.normal_map(nn2, (-250, y))
                nbf = p.get('NormalBlendFactor')
                blend = ctx.mix('VECTOR', 'MIX', (-50, y + 175),
                                a_sock=nrm, b_sock=nrm2,
                                fac_default=float(nbf) if isinstance(nbf, (int, float)) else 0.5)
                nrm = ctx.out(blend)
                y -= 350
        if nrm:
            l.new(nrm, bsdf.inputs['Normal'])

        ctx.aux_frame(['mask', 'blood', 'rim'], y,
                      'Skin extra maps (mask / blood / rim)')

    # -------------------------------------------------------------- Water
    @staticmethod
    def _build_water(ctx):
        n, l, p, bsdf = ctx.nodes, ctx.links, ctx.props, ctx.bsdf
        wc = _rgb(p.get('WaterColor'), (0.07, 0.11, 0.11))
        bsdf.inputs['Base Color'].default_value = (*wc, 1.0)
        if 'Roughness' in bsdf.inputs:
            bsdf.inputs['Roughness'].default_value = 0.03
        tr = _input(bsdf, 'Transmission Weight', 'Transmission')
        if tr:
            tr.default_value = 1.0
        if 'IOR' in bsdf.inputs:
            bsdf.inputs['IOR'].default_value = 1.33
        nn = ctx.tex('normal', 300, non_color=True)
        if nn:
            l.new(ctx.normal_map(nn, (-250, 300)), bsdf.inputs['Normal'])
        ctx.aux_frame(['diffuse', 'diffuse_mask', 'reflection'], -100,
                       'Water extra maps (foam / reflection cube)')

    # --------------------------------------------------------------- Road
    @staticmethod
    def _build_road(ctx):
        n, l, p, bsdf = ctx.nodes, ctx.links, ctx.props, ctx.bsdf
        y = 400
        d1 = ctx.tex('diffuse', y)
        base_out = d1.outputs['Color'] if d1 else None
        d2 = ctx.tex('diffuse2', y - 300)
        if d2 and base_out:
            det = ctx.mix('RGBA', 'MULTIPLY', (-450, y - 300),
                          a_sock=d2.outputs['Color'],
                          b=_rgb(p.get('DiffuseColor2')), fac_default=1.0)
            # roads blend the overlay by the overlay texture's own alpha
            fac = d2.outputs['Alpha'] if 'Alpha' in d2.outputs else None
            mixd = ctx.mix('RGBA', 'MIX', (-250, y - 120),
                           a_sock=base_out, b_sock=ctx.out(det),
                           fac_sock=fac, fac_default=0.5)
            base_out = ctx.out(mixd)
        if base_out:
            l.new(base_out, bsdf.inputs['Base Color'])
        sp = p.get('SpecularPower')
        if isinstance(sp, (int, float)) and 'Roughness' in bsdf.inputs:
            bsdf.inputs['Roughness'].default_value = _spec_roughness(sp)
        nn = ctx.tex('normal', y - 600, non_color=True)
        if nn:
            l.new(ctx.normal_map(nn, (-250, y - 600)), bsdf.inputs['Normal'])

    # ----------------------------------------------------------------- FX
    @staticmethod
    def _build_fx(ctx):
        n, l, p, bsdf = ctx.nodes, ctx.links, ctx.props, ctx.bsdf
        # FX shares the Unlit UV-anim mechanism (mesh_fx.fx:419
        # `scanlineUV.y += Time * SCROLLING_SPEED_V`) — the energy shields
        # (Z_TEMP_CORPSHIELD / Z_TEMP_NAVISHIELD) scroll their UVs. Use the
        # animated diffuse when anim params are set; otherwise the plain tex()
        # path, so glow + static fx are byte-for-byte unchanged.
        _anim = int(p.get('AnimType') or 0)
        _us = float(p.get('USpeed') or 0.0)
        _vs = float(p.get('VSpeed') or 0.0)
        _ang = float(p.get('AngularSpeed') or 0.0)
        if _anim or _us or _vs or _ang:
            d1 = BlenderMaterialSetup._build_animated_diffuse(ctx, _anim, _us, _vs, _ang)
        else:
            d1 = ctx.tex('diffuse', 300)
        col = _rgb(p.get('DiffuseColor1'))
        if d1:
            tint = ctx.mix('RGBA', 'MULTIPLY', (-450, 300),
                           a_sock=d1.outputs['Color'], b=col, fac_default=1.0)
            cout = ctx.out(tint)
            l.new(cout, bsdf.inputs['Base Color'])
            ec = _input(bsdf, 'Emission Color', 'Emission')
            if ec:
                l.new(cout, ec)
            if 'Alpha' in d1.outputs:
                l.new(d1.outputs['Alpha'], bsdf.inputs['Alpha'])
        else:
            bc = bsdf.inputs.get('Base Color')
            if bc:
                bc.default_value = (*col, 1.0)
        es = _input(bsdf, 'Emission Strength')
        if es:
            hm = p.get('HDRMul')
            es.default_value = float(hm) if isinstance(hm, (int, float)) and hm > 0 else 1.0

    # ----------------------------------------------------------- Unlit
    @staticmethod
    def _build_unlit(ctx):
        """mesh_unlit.fx — pure-emissive material with optional additive
        blend + frame-driven UV animation.

        LightingPS (mesh_unlit.fx:275-298):
            finalColor      = tex2D(DiffuseTexture1, anim_uv * DiffuseTiling1)
            finalColor     *= input.vertexColor          # rgba
            finalColor.rgb *= DiffuseColor1              # HDR tint (>1 ok)
            finalColor.rgb *= HDRMul                     # HDR intensity
          #ifdef ATTENUATION
            finalColor     *= pow(saturate(dot(N,V)), NormalAttenuationPower)

        MainVS (mesh_unlit.fx:140-167) scrolls / ping-pongs / rotates the UVs
        by UVAnimControlParams (USpeed / VSpeed / AngularSpeed), selected by
        AnimType.  The "black turns transparent, bright veins glow, pattern
        scrolls" look that the diffuse XBT (no alpha) shows in-game comes from
        the additive blend state (BlendingType != 0) + the UV scroll — NOT a
        texture alpha channel.  We rebuild all of it: emission + Add-Shader
        (Transparent) for the additive blend, and Mapping-node drivers for the
        motion.
        """
        n, l, p, bsdf = ctx.nodes, ctx.links, ctx.props, ctx.bsdf

        col1       = _rgb(p.get('DiffuseColor1'))
        hdrmul     = p.get('HDRMul')
        hdrmul     = float(hdrmul) if isinstance(hdrmul, (int, float)) and hdrmul > 0 else 1.0
        anim       = int(p.get('AnimType') or 0)
        uspeed     = float(p.get('USpeed') or 0.0)
        vspeed     = float(p.get('VSpeed') or 0.0)
        angspeed   = float(p.get('AngularSpeed') or 0.0)
        attn_pow   = p.get('NormalAttenuationPower')
        blend_t    = int(p.get('BlendingType') or 0)
        use_vcol   = bool(p.get('VertexColorEnabled'))
        use_valpha = bool(p.get('VertexAlphaEnabled'))

        # ---- animated, tiled diffuse ----
        d1 = BlenderMaterialSetup._build_animated_diffuse(
            ctx, anim, uspeed, vspeed, angspeed)

        # ---- emission colour = diffuse * DiffuseColor1 [* vertexColor] ----
        # DiffuseColor1 routed through an RGB node (ctx._color_sock) so HDR
        # values >1 (e.g. the alien pod's (2.0, 1.21, 0.0)) survive — a colour
        # socket default_value would clamp them.
        col1_sock = ctx._color_sock(col1, (-650, 360))
        if d1:
            cmix = ctx.mix('RGBA', 'MULTIPLY', (-450, 300),
                           a_sock=d1.outputs['Color'], b_sock=col1_sock,
                           fac_default=1.0)
            emis_col = ctx.out(cmix) if cmix is not None else col1_sock
        else:
            emis_col = col1_sock

        # Vertex colour / alpha are gated by their Enabled flags.  This matters:
        # vertex_colors.apply_vertex_colors defaults the 'Col' attribute to
        # BLACK (0,0,0,1) when the mesh had no COLOR data — multiplying emission
        # by that unconditionally would make every such Unlit mesh invisible.
        # The flags say whether the engine actually reads vertex colour here;
        # every energy-vegetation Unlit material in the corpus has them off.
        vc = None
        if use_vcol or use_valpha:
            vc = n.new('ShaderNodeVertexColor')
            vc.layer_name = 'Col'
            vc.location = (-650, 60)
        if use_vcol and vc is not None:
            vcmix = ctx.mix('RGBA', 'MULTIPLY', (-250, 240),
                            a_sock=emis_col, b_sock=vc.outputs['Color'],
                            fac_default=1.0)
            if vcmix is not None:
                emis_col = ctx.out(vcmix)

        # ---- emission strength = HDRMul [* facing-attenuation] [* vtx alpha] ----
        strength_sock = None
        if isinstance(attn_pow, (int, float)) and float(attn_pow) != 0.0:
            atten = BlenderMaterialSetup._facing_attenuation(ctx, float(attn_pow))
            mul = n.new('ShaderNodeMath')
            mul.operation = 'MULTIPLY'
            mul.location = (-150, -120)
            mul.inputs[1].default_value = hdrmul
            l.new(atten, mul.inputs[0])
            strength_sock = mul.outputs[0]
        if use_valpha and vc is not None:
            mulv = n.new('ShaderNodeMath')
            mulv.operation = 'MULTIPLY'
            mulv.location = (0, -200)
            if strength_sock is not None:
                l.new(strength_sock, mulv.inputs[0])
            else:
                mulv.inputs[0].default_value = hdrmul
            l.new(vc.outputs['Alpha'], mulv.inputs[1])
            strength_sock = mulv.outputs[0]

        # ---- emission node ----
        em = n.new('ShaderNodeEmission')
        em.location = (150, 200)
        l.new(emis_col, em.inputs['Color'])
        if strength_sock is not None:
            l.new(strength_sock, em.inputs['Strength'])
        else:
            em.inputs['Strength'].default_value = hdrmul

        # ---- output: additive (BlendingType != 0) or opaque emission ----
        # setup_material wired Principled BSDF -> Output; Unlit is emission-only
        # so detach the BSDF and drive the output ourselves. The now-orphaned
        # Principled is deleted by setup_material's end-of-build cleanup (its
        # BSDF output has no links), so the final tree has no dead node.
        out_node = next((nd for nd in n if nd.type == 'OUTPUT_MATERIAL'), None)
        try:
            for lnk in list(bsdf.outputs['BSDF'].links):
                l.remove(lnk)
        except Exception:
            pass

        additive = blend_t != 0
        if additive:
            # Add-Shader(Emission, Transparent) == background + emission, i.e.
            # additive: a black emission pixel adds nothing (looks transparent),
            # a bright one glows.  Reproduces the energy-glow blend without any
            # texture alpha.
            trans = n.new('ShaderNodeBsdfTransparent')
            trans.location = (320, 360)
            add = n.new('ShaderNodeAddShader')
            add.location = (470, 280)
            l.new(em.outputs['Emission'], add.inputs[0])
            l.new(trans.outputs['BSDF'], add.inputs[1])
            if out_node is not None:
                l.new(add.outputs['Shader'], out_node.inputs['Surface'])
            BlenderMaterialSetup._set_transparent_blend(ctx.mat)
        elif out_node is not None:
            l.new(em.outputs['Emission'], out_node.inputs['Surface'])

        try:
            from ..Core.debug import TraceLogger as _TL
            _TL.struct("unlit_material_setup", {
                "material":          ctx.mat.name if ctx.mat else None,
                "blending_type":     blend_t,
                "additive":          additive,
                "anim_type":         anim,
                "u_speed":           uspeed,
                "v_speed":           vspeed,
                "angular_speed":     angspeed,
                "hdr_mul":           hdrmul,
                "diffuse_color1":    col1,
                "attenuation_power": attn_pow,
                "vertex_color":      use_vcol,
                "vertex_alpha":      use_valpha,
                "has_diffuse":       bool(d1),
            })
        except Exception:
            pass

    @staticmethod
    def _build_animated_diffuse(ctx, anim_type, uspeed, vspeed, angspeed):
        """Diffuse Image node with tiling + frame-driven UV animation.

        Shared by the Unlit and FX builders (both use the same UVAnimControl
        scroll/rotate mechanism). Unlike ctx.tex() this ALWAYS inserts a
        Mapping node so the scroll / rotation drivers have somewhere to live.
        These materials reference a single texture on UV0 (UVGroupMapChannel0 ==
        0 across the corpus), so there is no UV1 routing to worry about here.
        """
        path = ctx.tex_map.get('diffuse')
        if not path:
            return None
        tn = BlenderMaterialSetup._load_texture_node(
            ctx.nodes, path, ctx.data_folder, (-1100, 300),
            non_color=False, load_hd_textures=ctx.lhd, import_as_dds=ctx.iad)
        if not tn:
            return None
        uv = _vec2(ctx.props.get('DiffuseTiling1'))
        m = ctx.nodes.new('ShaderNodeMapping')
        m.location = (-1400, 300)
        m.inputs['Scale'].default_value = (uv[0], uv[1], 1.0)
        ctx.links.new(ctx.tc.outputs['UV'], m.inputs['Vector'])
        ctx.links.new(m.outputs['Vector'], tn.inputs['Vector'])
        BlenderMaterialSetup._add_uv_anim_drivers(
            m, anim_type, uspeed, vspeed, angspeed, uv)
        return tn

    @staticmethod
    def _add_uv_anim_drivers(mapping_node, anim_type, uspeed, vspeed,
                             angspeed, tiling):
        """Frame-driven UV animation matching mesh_unlit.fx MainVS [140-167].

        The engine drives motion from `Time` (seconds); we drive from the
        current frame / scene fps.  Expressions use only `frame`, arithmetic
        and sin/cos so Blender treats them as *simple expressions* — they
        evaluate WITHOUT "Auto Run Python Scripts" being enabled (a full Python
        driver would silently stay frozen for most users).

        AnimType -> UVAnimControlFlags (corpus-confirmed: 0 none, 1 scroll,
        2 offset-scroll, 3 ping-pong; 4 would be rotation).  Scroll offset is
        pre-multiplied by tiling because the engine computes (uv+offset)*tiling
        while a Mapping node computes uv*tiling + location.
        """
        try:
            scene = bpy.context.scene
            fps = scene.render.fps / max(1e-6, float(scene.render.fps_base))
        except Exception:
            fps = 24.0
        if fps <= 0:
            fps = 24.0
        tU = tiling[0] or 1.0
        tV = tiling[1] or 1.0

        def _drv(sock, index, expr):
            try:
                fc = sock.driver_add('default_value', index)
                d = fc.driver
                d.type = 'SCRIPTED'
                d.expression = expr
            except Exception:
                pass

        loc = mapping_node.inputs['Location']
        rot = mapping_node.inputs['Rotation']
        t = "(frame/%.6f)" % fps

        # The importer flips V on decode (`Blender_v = 1 - game_v`, see
        # mesh.parse_mesh_vertices:205) but leaves U unflipped. A positive game
        # V-scroll therefore has to be NEGATED in Blender or the texture crawls
        # the wrong way — the energy bark (VSpeed=0.07) scrolled DOWN instead of
        # up. The V reflection also reverses rotation handedness, so negate the
        # angular speed too. U is untouched.
        vspeed = -vspeed
        angspeed = -angspeed

        if anim_type == 3:                       # ping-pong (cos/sin)
            if uspeed:
                _drv(loc, 0, "cos(%s)*%.8f" % (t, uspeed * tU))
            if vspeed:
                _drv(loc, 1, "sin(%s)*%.8f" % (t, vspeed * tV))
        elif anim_type >= 4:                     # rotation
            if angspeed:
                _drv(rot, 2, "%s*%.8f" % (t, angspeed))
        else:                                    # 1/2 (and any speed) -> scroll
            if uspeed:
                _drv(loc, 0, "%s*%.8f" % (t, uspeed * tU))
            if vspeed:
                _drv(loc, 1, "%s*%.8f" % (t, vspeed * tV))

    @staticmethod
    def _facing_attenuation(ctx, power):
        """pow(saturate(dot(N, V)), power) via a Layer Weight 'Facing' output.

        Layer Weight 'Facing' is ~0 head-on and ~1 at grazing angles, i.e.
        (1 - NdotV), so NdotV = 1 - Facing.  Returns a float socket carrying
        the attenuation factor.
        """
        n, l = ctx.nodes, ctx.links
        lw = n.new('ShaderNodeLayerWeight')
        lw.location = (-700, -160)
        sub = n.new('ShaderNodeMath')
        sub.operation = 'SUBTRACT'
        sub.location = (-500, -160)
        sub.inputs[0].default_value = 1.0
        l.new(lw.outputs['Facing'], sub.inputs[1])
        pw = n.new('ShaderNodeMath')
        pw.operation = 'POWER'
        pw.location = (-330, -160)
        l.new(sub.outputs[0], pw.inputs[0])
        pw.inputs[1].default_value = float(power)
        return pw.outputs[0]

    # ----------------------------------------------------------- material
    @staticmethod
    def _apply_flags(mat, props, template):
        _always_two_sided = ('fx', 'particle', 'meshfx', 'aaaleaf', 'leaf',
                              'realtreeleafpure', 'bigleaf', 'realtreeleafhybrid',
                              'thincloth', 'hair')
        two_sided = bool(props.get('TwoSided', 0)) or template in _always_two_sided
        mat.use_backface_culling = not two_sided
        if hasattr(mat, 'show_transparent_back'):
            mat.show_transparent_back = two_sided

        _wants_clip  = bool(props.get('AlphaTestEnabled'))
        _wants_blend = bool(props.get('AlphaBlendEnabled')) or template in (
                'water', 'fx', 'particle', 'meshfx', 'glow', 'decal', 'staticdecal',
                'aaaleaf', 'leaf', 'realtreeleafpure', 'bigleaf', 'realtreeleafhybrid')

        if _wants_clip:
            # Pre-4.2 API
            if hasattr(mat, 'blend_method'):
                try:
                    mat.blend_method = 'CLIP'
                except Exception:
                    pass
            if hasattr(mat, 'shadow_method'):
                try:
                    mat.shadow_method = 'CLIP'
                except Exception:
                    pass
            # 4.2+ EEVEE Next
            if hasattr(mat, 'surface_render_method'):
                try:
                    mat.surface_render_method = 'DITHERED'
                except Exception:
                    pass
        elif _wants_blend:
            # Pre-4.2 API
            if hasattr(mat, 'blend_method'):
                try:
                    mat.blend_method = 'BLEND'
                except Exception:
                    pass
            if hasattr(mat, 'shadow_method'):
                try:
                    mat.shadow_method = 'HASHED'
                except Exception:
                    pass
            # 4.2+ EEVEE Next
            if hasattr(mat, 'surface_render_method'):
                try:
                    mat.surface_render_method = 'BLENDED'
                except Exception:
                    pass


class _Ctx:
    """Carries node-tree state and the reusable graph builders."""

    def __init__(self, nodes, links, props, tex, tc, data_folder,
                 lhd, iad, bsdf, mat=None):
        self.nodes = nodes
        self.links = links
        self.props = props
        self.tex_map = tex
        self.tc = tc
        self.data_folder = data_folder
        self.lhd = lhd
        self.iad = iad
        self.bsdf = bsdf
        self.mat = mat

    # ---- texture image node with per-map tiling + UV-channel routing ----
    def tex(self, category, y, non_color=False):
        path = self.tex_map.get(category)
        if not path:
            return None
        tn = BlenderMaterialSetup._load_texture_node(
            self.nodes, path, self.data_folder, (-1100, y),
            non_color=non_color, load_hd_textures=self.lhd,
            import_as_dds=self.iad)
        if not tn:
            return None

        uv = _vec2(self.props.get(_TILING_PROP.get(category, '')))

        # UV channel: 0 → UVMap (UV0), 1 → UVMap1 (UV1).  See _UVGROUP_PROP.
        # Defaults to 0 (UV0) when the prop is absent — matches the common
        # character-material case where everything is on a single UV layer.
        ch_raw = self.props.get(_UVGROUP_PROP.get(category, ''))
        try:
            uv_channel = int(ch_raw) if ch_raw is not None else 0
        except (TypeError, ValueError):
            uv_channel = 0

        # Fast path: UV0 + no tiling → no extra nodes, image samples the
        # mesh's active UV (UVMap).  Identical bytes to the old behaviour
        # for every material that doesn't use UV1 or non-unit tiling.
        if uv_channel == 0 and uv == (1.0, 1.0):
            return tn

        # We need an explicit UV source whenever the channel isn't 0 OR the
        # tiling isn't identity.  A UV Map node nails the exact layer (so a
        # non-zero channel reaches UVMap1 instead of the active layer that
        # ShaderNodeTexCoord.UV would give); a Mapping node applies tiling.
        vec_out = None
        if uv_channel != 0:
            uvm = self.nodes.new('ShaderNodeUVMap')
            uvm.location = (-1600, y)
            layer = _UV_LAYER_FOR_CHANNEL.get(uv_channel, 'UVMap')
            try:
                uvm.uv_map = layer
            except Exception:
                pass
            vec_out = uvm.outputs['UV']
        else:
            # UV0 but non-identity tiling — feed from the shared Texture
            # Coordinate node's UV output (the active layer = UVMap = UV0).
            vec_out = self.tc.outputs['UV']

        if uv != (1.0, 1.0):
            m = self.nodes.new('ShaderNodeMapping')
            m.location = (-1400, y)
            m.inputs['Scale'].default_value = (uv[0], uv[1], 1.0)
            self.links.new(vec_out, m.inputs['Vector'])
            self.links.new(m.outputs['Vector'], tn.inputs['Vector'])
        else:
            # Non-zero channel, identity tiling — UV Map straight into image.
            self.links.new(vec_out, tn.inputs['Vector'])
        return tn

    # ---- mask -> per-channel sockets ----
    # aaa.fx: FINALMASKSRC = vertexMask * MaskTexture1  (or just vertexMask without mask map)
    # vertexMask IS the mesh vertex color — always read, always used as mask base.
    def mask_channels(self, y):
        vc = self.nodes.new('ShaderNodeVertexColor')
        vc.layer_name = 'Col'
        vc.location = (-1350, y)

        mt = self.tex('mask', y)
        if mt:
            # Reuse the robust ctx.mix() constructor — it handles modern-
            # vs-legacy node fallback AND the multi-socket A/B problem on
            # Blender 4+/5+ with its data_type verification.
            blend = self.mix('RGBA', 'MULTIPLY', (-1050, y),
                              a_sock=vc.outputs['Color'],
                              b_sock=mt.outputs['Color'],
                              fac_default=1.0)
            if blend is None:
                # Mix node creation failed entirely — fall back to plain
                # vertex color as the mask source.
                mask_out = vc.outputs['Color']
            else:
                mask_out = _output(blend, 'Result', 'Color') or blend.outputs[0]
        else:
            mask_out = vc.outputs['Color']

        try:
            sep = self.nodes.new('ShaderNodeSeparateColor')
            self.links.new(mask_out, sep.inputs['Color'])
            sep.location = (-850, y)
            return (sep.outputs[0], sep.outputs[1], sep.outputs[2])
        except Exception:
            sep = self.nodes.new('ShaderNodeSeparateRGB')
            self.links.new(mask_out, sep.inputs['Image'])
            sep.location = (-850, y)
            return (sep.outputs[0], sep.outputs[1], sep.outputs[2])

    # ---- generic Mix node (RGBA / VECTOR / FLOAT) with fallbacks ----
    def mix(self, dtype, blend, loc, a=None, b=None, a_sock=None,
            b_sock=None, fac=None, fac_sock=None, fac_default=0.5):
        # For color/RGB material chains the modern ShaderNodeMix in
        # Blender 4+/5+ has THREE failure modes that have all bitten us:
        #   1. inputs['A'] returns the FLOAT 'A' socket (same name as the
        #      RGBA one).  Fixed in _input/_output by filtering enabled.
        #   2. The enum value for the color data type might be 'RGBA' OR
        #      'COLOR' depending on Blender version.  Setting it to the
        #      wrong name silently leaves the node as FLOAT — the assignment
        #      just doesn't take.  After assignment we VERIFY by reading
        #      back; if it's still FLOAT we try the other name; if both
        #      fail we drop to the legacy ShaderNodeMixRGB which has fixed
        #      socket names.
        #   3. blend_type is only meaningful for the COLOR data_type;
        #      attempting to set it while the node is still FLOAT raises
        #      on some Blenders.
        #
        # Goal: get a node whose ENABLED color sockets are reachable via
        # _input(node, 'A' / 'B' / 'Factor'), or fall back to the legacy
        # node which doesn't have the multi-socket problem.
        mx = None
        # Log which path we took once per material so a user reporting
        # "still white" can confirm what's actually being built.
        try:
            from ..Core.debug import TraceLogger as _TL
        except Exception:
            _TL = None

        if dtype == 'RGBA':
            # Try the modern Mix node, then fall back to ShaderNodeMixRGB.
            try:
                mx = self.nodes.new('ShaderNodeMix')
                # Try both possible enum spellings — Blender 5.x may have
                # renamed 'RGBA' → 'COLOR'.
                for candidate in ('RGBA', 'COLOR'):
                    try:
                        mx.data_type = candidate
                    except Exception:
                        continue
                    if mx.data_type == candidate:
                        break
                if mx.data_type not in ('RGBA', 'COLOR'):
                    # Modern node refused the color enum on this Blender.
                    self.nodes.remove(mx)
                    raise RuntimeError(
                        f"ShaderNodeMix.data_type would not accept "
                        f"'RGBA' or 'COLOR' (stayed {mx.data_type!r})")
                try:
                    mx.blend_type = blend
                except Exception:
                    pass
                f = _input(mx, 'Factor')
                ai = _input(mx, 'A')
                bi = _input(mx, 'B')
                # Verify all three sockets resolved AND are reachable.
                # If any of them came back as a FLOAT socket (because
                # enabled-filtering still missed) we'd silently route a
                # color to a float input — defeat in detail.  Use socket
                # .type to confirm — must be 'RGBA' for A and B (Factor
                # is always FLOAT and that's correct).
                a_type = getattr(ai, 'type', None) if ai else None
                b_type = getattr(bi, 'type', None) if bi else None
                if a_type != 'RGBA' or b_type != 'RGBA':
                    self.nodes.remove(mx)
                    raise RuntimeError(
                        f"ShaderNodeMix A/B sockets came back as "
                        f"({a_type}, {b_type}) — expected ('RGBA', 'RGBA')")
                if _TL is not None:
                    _TL.struct("mix_node_path", {
                        "blend": blend, "data_type": mx.data_type,
                        "path": "modern_mix",
                        "a_socket_type": a_type, "b_socket_type": b_type,
                    })
            except Exception as _exc:
                if mx is not None and mx.name in {n.name for n in self.nodes}:
                    try: self.nodes.remove(mx)
                    except Exception: pass
                # Legacy path — ShaderNodeMixRGB has named sockets that
                # don't collide and works on every Blender that still
                # ships it (4.x deprecates it but still creates it; 5.x
                # may or may not — handled with another try).
                try:
                    mx = self.nodes.new('ShaderNodeMixRGB')
                    mx.blend_type = blend
                    f = mx.inputs['Fac']
                    ai = mx.inputs['Color1']
                    bi = mx.inputs['Color2']
                    if _TL is not None:
                        _TL.struct("mix_node_path", {
                            "blend": blend, "data_type": dtype,
                            "path": "legacy_mixrgb",
                            "fallback_reason": str(_exc)[:200],
                        })
                except Exception:
                    # Neither node available — give up and return None
                    # so the caller can detect and skip the chain step.
                    VerboseLogger.warn(
                        f"[mix] could not create EITHER ShaderNodeMix "
                        f"(modern) OR ShaderNodeMixRGB (legacy) for "
                        f"blend={blend!r} dtype={dtype!r}; chain step "
                        f"skipped (material will look wrong)")
                    if _TL is not None:
                        _TL.struct("mix_node_path", {
                            "blend": blend, "data_type": dtype,
                            "path": "TOTAL_FAILURE",
                            "exception": str(_exc)[:200],
                        })
                    return None
        else:
            # VECTOR / FLOAT / etc — only the modern node supports these.
            mx = self.nodes.new('ShaderNodeMix')
            try:
                mx.data_type = dtype
            except Exception:
                pass
            f = _input(mx, 'Factor')
            ai = _input(mx, 'A')
            bi = _input(mx, 'B')
        mx.location = loc
        fs = fac if fac is not None else fac_sock
        if fs is not None:
            self.links.new(fs, f)
        elif f is not None and hasattr(f, 'default_value'):
            f.default_value = fac_default
        if a_sock is not None:
            self.links.new(a_sock, ai)
        elif a is not None and ai is not None:
            self.links.new(self._color_sock(a, (loc[0] - 180, loc[1] + 80)), ai)
        if b_sock is not None:
            self.links.new(b_sock, bi)
        elif b is not None and bi is not None:
            self.links.new(self._color_sock(b, (loc[0] - 180, loc[1] - 80)), bi)
        return mx

    @staticmethod
    def out(mix_node):
        # Same multi-socket problem as _input: Blender 4+/5+ Mix node has
        # multiple 'Result' outputs (FLOAT / VECTOR / RGBA / ROTATION) and
        # only the one matching data_type is .enabled.  Picking the first
        # by name returns the FLOAT 'Result' which produces a single greyscale
        # value instead of the colour we want — the downstream BSDF then
        # got greyscale or white-on-default.  Prefer the enabled socket.
        for name in ('Result', 'Color'):
            for s in mix_node.outputs:
                if s.name == name and getattr(s, 'enabled', True):
                    return s
            if name in mix_node.outputs:
                return mix_node.outputs[name]
        return mix_node.outputs[0]

    # ---- RGB node — supports HDR values (>1.0) that color socket default_value clips ----
    def _color_sock(self, rgb, loc):
        rn = self.nodes.new('ShaderNodeRGB')
        rn.location = loc
        rn.outputs[0].default_value = (float(rgb[0]), float(rgb[1]), float(rgb[2]), 1.0)
        return rn.outputs[0]

    # ---- DXT5-GA normal decode (X=A, Y=G, Z reconstructed) + Normal Map ----
    def normal_map(self, tex_node, loc, strength=None, strength_default=1.0):
        n, l = self.nodes, self.links
        try:
            sep = n.new('ShaderNodeSeparateColor')
            l.new(tex_node.outputs['Color'], sep.inputs['Color'])
            g_sock = sep.outputs[1]
        except Exception:
            sep = n.new('ShaderNodeSeparateRGB')
            l.new(tex_node.outputs['Color'], sep.inputs['Image'])
            g_sock = sep.outputs[1]
        sep.location = (loc[0] - 400, loc[1])
        x_sock = tex_node.outputs['Alpha'] if 'Alpha' in tex_node.outputs else sep.outputs[0]

        def m(op, i0, i1=None, v1=0.5, x=0, yy=0):
            nd = n.new('ShaderNodeMath')
            nd.operation = op
            nd.location = (loc[0] - 300 + x, loc[1] + yy)
            if hasattr(i0, 'name'):
                l.new(i0, nd.inputs[0])
            else:
                nd.inputs[0].default_value = i0
            if i1 is not None:
                if hasattr(i1, 'name'):
                    l.new(i1, nd.inputs[1])
                else:
                    nd.inputs[1].default_value = i1
            else:
                nd.inputs[1].default_value = v1
            return nd.outputs[0]

        # Decode signed normal: x2 = x*2-1 ; y2 = y*2-1
        x2 = m('SUBTRACT', m('MULTIPLY', x_sock, 2.0, x=-150, yy=120), 1.0, x=0, yy=120)
        y2 = m('SUBTRACT', m('MULTIPLY', g_sock, 2.0, x=-150, yy=40), 1.0, x=0, yy=40)
        # z = sqrt(max(1 - x2^2 - y2^2, 0))
        sx = m('MULTIPLY', x2, x2, x=120, yy=120)
        sy = m('MULTIPLY', y2, y2, x=120, yy=40)
        ssum = m('ADD', sx, sy, x=240, yy=80)
        zc = m('SQRT', m('MAXIMUM', m('SUBTRACT', 1.0, ssum, x=360, yy=80),
                         0.0, x=480, yy=80), x=600, yy=80)
        # Re-encode to 0..1 (the Normal Map node maps color*2-1 internally)
        xe = m('ADD', m('MULTIPLY', x2, 0.5, x=700, yy=160), 0.5, x=820, yy=160)
        ye = m('ADD', m('MULTIPLY', y2, 0.5, x=700, yy=90), 0.5, x=820, yy=90)
        ze = m('ADD', m('MULTIPLY', zc, 0.5, x=700, yy=20), 0.5, x=820, yy=20)
        try:
            comb = n.new('ShaderNodeCombineColor')
            ci = (comb.inputs[0], comb.inputs[1], comb.inputs[2])
        except Exception:
            comb = n.new('ShaderNodeCombineRGB')
            ci = (comb.inputs[0], comb.inputs[1], comb.inputs[2])
        comb.location = (loc[0] - 20, loc[1])
        l.new(xe, ci[0])
        l.new(ye, ci[1])
        l.new(ze, ci[2])
        nm = n.new('ShaderNodeNormalMap')
        nm.location = (loc[0] + 160, loc[1])
        s = _input(nm, 'Strength')
        if strength is not None and s is not None:
            l.new(strength, s)
        elif s is not None:
            s.default_value = strength_default
        l.new(comb.outputs[0], nm.inputs['Color'])
        return nm.outputs['Normal']

    # ---- emission (IlluminationTexture * IlluminationColor1) ----
    def emission(self, y):
        et = self.tex('emission', y)
        if not et:
            return
        ic = self.props.get('IlluminationColor1')
        rgb = _rgb(ic) if isinstance(ic, (tuple, list)) else (1, 1, 1)
        emix = self.mix('RGBA', 'MULTIPLY', (-250, y),
                        a_sock=et.outputs['Color'], b=rgb, fac_default=1.0)
        ec = _input(self.bsdf, 'Emission Color', 'Emission')
        if ec:
            self.links.new(self.out(emix), ec)
        es = _input(self.bsdf, 'Emission Strength')
        if es:
            es.default_value = float(ic[3]) if isinstance(ic, (tuple, list)) and len(ic) >= 4 else 1.0

    # ---- aux maps: load + label inside a frame, don't mis-wire ----
    def aux_frame(self, categories, y, label):
        present = [c for c in categories if self.tex_map.get(c)]
        if not present:
            return
        fr = self.nodes.new('NodeFrame')
        fr.label = label
        for c in present:
            t = self.tex(c, y, non_color=True)
            if t:
                t.label = c
                t.parent = fr
                y -= 320

    # ---- image loader (unchanged behaviour) ----
    pass


def _load_texture_node(nodes, texture_path, data_folder, location,
                       non_color=False, load_hd_textures=True, import_as_dds=False):
    """Load an XBT/DDS/PNG texture into a Blender Image Texture node.

    Returns the texture node, or None on any failure (caller chains to
    `if tn:` and bypasses connecting that channel).  EVERY failure path
    now emits a structured `texture_load` event with the engine path,
    the resolved disk path, whether the file existed, and why we bailed
    — silent None-returns were exactly why the user couldn't tell which
    texture was missing when a material rendered white.
    """
    try:
        from ..Core.debug import TraceLogger as _TL
    except Exception:
        _TL = None

    def _emit(status, **fields):
        """Drop one structured record per attempt (success or failure)."""
        if _TL is None:
            return
        try:
            _TL.struct("texture_load", {
                "engine_path":   texture_path,
                "data_folder":   data_folder,
                "load_hd":       bool(load_hd_textures),
                "import_as_dds": bool(import_as_dds),
                "non_color":     bool(non_color),
                "status":        status,
                **fields,
            })
        except Exception:
            pass

    if not texture_path:
        _emit("missing/no_engine_path")
        return None
    if not data_folder:
        VerboseLogger.log(
            f"[texture_load] '{texture_path}' SKIPPED — no data_folder set "
            f"in addon preferences (textures won't load)")
        _emit("missing/no_data_folder")
        return None

    mip0 = XBTConverter.find_mip0_variant(texture_path, data_folder) if load_hd_textures else None
    actual_path = mip0 or texture_path
    full_path = os.path.join(data_folder,
                              actual_path.replace('\\', os.sep).replace('/', os.sep))
    if not os.path.exists(full_path):
        # Try a lowercased fallback — extracted packs sometimes flatten
        # case, and the engine paths in .xbm are mixed-case.
        full_path_lc = os.path.join(data_folder,
                                     actual_path.lower().replace('\\', os.sep).replace('/', os.sep))
        if os.path.isfile(full_path_lc):
            full_path = full_path_lc
        else:
            VerboseLogger.log(
                f"[texture_load] '{texture_path}' NOT FOUND on disk\n"
                f"    tried: {full_path}\n"
                f"    tried: {full_path_lc}\n"
                f"    (this is why a material may render white — the diffuse "
                f"texture isn't being connected to BSDF Base Color)")
            _emit("missing/file_not_found",
                  resolved_path=full_path,
                  lowercase_tried=full_path_lc,
                  mip0_variant_used=bool(mip0))
            return None

    texture_file_path = XBTConverter.get_temp_texture_path(full_path, import_as_dds)
    if not texture_file_path:
        VerboseLogger.log(
            f"[texture_load] '{texture_path}' XBT->image conversion FAILED "
            f"(no DDS payload found inside the XBT — file may be corrupt)")
        _emit("failed/xbt_conversion",
              resolved_path=full_path, xbt_size=os.path.getsize(full_path))
        return None

    actual_ext = os.path.splitext(texture_file_path)[1].lower()
    base_name = os.path.splitext(os.path.basename(texture_path))[0]
    img_name = f"{base_name}{actual_ext}"
    img = bpy.data.images.get(img_name)
    cached = img is not None
    if not img:
        try:
            img = load_image(texture_file_path, check_existing=False)
            if img:
                img.name = img_name
                img.pack()
                non_color and setattr(img.colorspace_settings, 'name', 'Non-Color')
            else:
                VerboseLogger.log(
                    f"[texture_load] '{texture_path}' load_image returned None "
                    f"(temp file: {texture_file_path})")
                _emit("failed/load_image_none",
                      resolved_path=full_path, temp_file=texture_file_path)
                return None
        except Exception as e:
            VerboseLogger.log(f"[texture_load] '{texture_path}' raised: {e}")
            _emit("failed/load_image_exception",
                  resolved_path=full_path, temp_file=texture_file_path,
                  error=str(e)[:256])
            return None

    tex_node = nodes.new('ShaderNodeTexImage')
    tex_node.location = location
    tex_node.image = img
    # Read back the actual loaded dimensions so a 1x1-pixel "broken" image
    # surfaces in the log.  Blender's DDS loader sometimes silently produces
    # 1x1 black when the BCn format isn't supported on this hardware.
    try:
        w, h = img.size[0], img.size[1]
    except Exception:
        w, h = 0, 0
    # Several game-default textures are 4x4 BY DESIGN — `flat_normal.xbt`
    # is a constant (0.5, 0.5, 1.0) up-normal, `white_texture.xbt` is a
    # solid white, `tattoo.xbt` for a non-tattooed character is empty.
    # Don't bark at these — only warn for full-resolution textures that
    # collapsed (e.g. NEWBODY_d coming back 4x4 IS a real decode failure).
    _GAME_DEFAULT_TINY = ('flat_normal', 'white_texture', 'tattoo',
                          'black_texture', 'flatnormal', 'whitetexture')
    base_lc = base_name.lower()
    is_known_tiny = any(d in base_lc for d in _GAME_DEFAULT_TINY)
    if w <= 4 and h <= 4 and not is_known_tiny:
        VerboseLogger.warn(
            f"[texture_load] '{texture_path}' loaded but is only {w}x{h} "
            f"pixels — Blender's DDS decoder may have failed silently. "
            f"Material will render almost entirely as the BSDF default.")
    _emit("ok",
          resolved_path=full_path, temp_file=texture_file_path,
          mip0_variant_used=bool(mip0),
          image_name=img_name,
          size=[w, h], cached=cached)
    return tex_node


BlenderMaterialSetup._load_texture_node = staticmethod(_load_texture_node)
