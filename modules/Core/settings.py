"""Scene-level settings PropertyGroups (shared by all game UIs).

Split out of the monolithic __init__.py (2026-06-09 refactor).
"""
import bpy

class XBGImportSettings(bpy.types.PropertyGroup):
    load_textures: bpy.props.BoolProperty(
        name="Load Textures",
        description="Automatically load and setup textures from XBM material files",
        default=True
    )
    load_hd_textures: bpy.props.BoolProperty(
        name="Load HD Textures",
        description="Use high-resolution _mip0 texture variants when available",
        default=True
    )


class XBGInjectSettings(bpy.types.PropertyGroup):
    target_lod: bpy.props.IntProperty(
        name="Target LOD",
        description="Which LOD slot to replace (0 = highest detail)",
        default=0,
        min=0,
        max=10
    )
    ignore_format_limits: bpy.props.BoolProperty(
        name="Ignore Format Limits",
        description="Write coordinates even if they exceed the uint16 range — wraps instead of clamping. May corrupt the model.",
        default=False
    )
    override_game_scale: bpy.props.BoolProperty(
        name="Override Internal Scale",
        description="Write a new scale value to the PMCP chunk before injecting",
        default=False
    )
    target_game_scale: bpy.props.FloatProperty(
        name="New Scale Value",
        description="The new PMCP pos_scale multiplier",
        default=1.0,
        precision=6,
        min=0.000001
    )
    # "Keep Custom Normals" toggle removed (2026-06-02). It's now redundant:
    # stock verts are ALWAYS preserved via the xbg_normal POINT attribute
    # (read first in _encode_vertices), and new geometry always exports its
    # viewport normals. There is no recompute path anymore — the param was
    # dropped from inject()/_encode_vertices entirely. To reset normals, do it
    # in Blender (Mesh ▸ Normals) before export.
    inject_vertex_colors: bpy.props.BoolProperty(
        name="Include Vertex Colors",
        description="ON (default): write the mesh's vertex colors into the file. Original colors are preserved exactly when you didn't touch them, and anything you painted is injected as-is (vertex color is the game's aaa.fx shading mask: specular / normal-strength / AO). OFF: ignore vertex colors and write a neutral mask everywhere — e.g. for a fully custom model that shouldn't carry the original's shading data.",
        default=True
    )
    generate_neutral_vertex_colors: bpy.props.BoolProperty(
        name="Generate Neutral Vertex Colors",
        description="ON (default): geometry that has NO vertex colors of its own — new meshes you never painted — gets a neutral mask instead of Blender's all-zero default, which the game reads as a black shadow (the 'shadow on the body' issue). Geometry you DID paint is injected unchanged. OFF: write the raw values as-is (can produce dark shadows on unpainted new geometry). Only applies when Include Vertex Colors is ON.",
        default=True
    )
    inject_bone_weights: bpy.props.BoolProperty(
        name="Include Bone Weights",
        description="Write vertex group weights into the XBG bone weight buffer. Must be ON for any rigged/skinned mesh — turning this off replaces every vertex's skinning with a single bone (slot 0), destroying the rig.",
        default=True
    )
    inject_materials: bpy.props.BoolProperty(
        name="Split by Material Slots",
        description="Split each object by material slot — each material becomes a separate XBG submesh/primitive (like Samson's body parts)",
        default=False
    )
    # Tangent-space options removed from the UI (2026-06-02). New geometry now
    # ALWAYS computes tangents from UVs (the single, principled method); see the
    # hard-coded values in the inject() call. Stock verts re-export their stored
    # xbg_tangent unchanged. The orthogonal floor is an unconditional internal
    # safety net (inject_xbg never emits a zero tangent) — never a user choice.
    force_per_submesh_vb: bpy.props.BoolProperty(
        name="Force Per-Submesh VB (>65k verts, UNTESTED)",
        description=(
            "EXPERIMENTAL: write a separate vertex buffer per submesh "
            "instead of matching the source file's shared-VB layout. "
            "Allows each submesh to hold up to 65535 vertices independently "
            "(total can exceed 65535) — useful for very-high-poly imports. "
            "WARNING: character meshes (Kendra, viperwolf, …) ship with "
            "shared-VB and the engine's skinning path likely binds VB 0 "
            "unconditionally. If so, only submesh 0 renders and the rest "
            "show garbage or crash. Leave OFF unless you specifically need "
            ">65535 verts and are willing to test in-game."
        ),
        default=False
    )


# ---------------------------------------------------------------------------
# Session data — persists the linked XBG across object selection changes
# ---------------------------------------------------------------------------

class XBGSessionData(bpy.types.PropertyGroup):
    """Scene-level storage so the inject panel stays visible regardless of
    which object is currently selected.  Populated by 'Remember This XBG'."""
    is_loaded: bpy.props.BoolProperty(
        name="Session XBG Loaded",
        default=False
    )
    filepath: bpy.props.StringProperty(
        name="Session XBG Path",
        default=""
    )
    pos_scale: bpy.props.FloatProperty(
        name="Pos Scale",
        default=1.0
    )
    uv_trans: bpy.props.FloatProperty(
        name="UV Trans",
        default=0.0
    )
    uv_scale: bpy.props.FloatProperty(
        name="UV Scale",
        default=1.0
    )
    import_mesh_only: bpy.props.BoolProperty(
        name="Import Mesh Only",
        default=False
    )
    pmcp_offset: bpy.props.IntProperty(
        name="PMCP Offset",
        default=0
    )


def _update_format_bounds(self, ctx):
    """Called whenever show_format_bounds checkbox is toggled."""
    existing = bpy.data.objects.get("XBG_Format_Bounds")
    if self.show_format_bounds:
        if existing:
            return
        ps = None
        for obj in ctx.scene.objects:
            if obj.type == 'MESH' and 'xbg_data' in obj:
                ps = obj['xbg_data'].get('pos_scale', None)
                if ps:
                    break
        if ps is None:
            self.show_format_bounds = False
            return
        from .debug import create_format_bounds_lattice
        lo = create_format_bounds_lattice(ctx, ps)
        half = lo.scale[0]
        self.format_bounds_x = half
        self.format_bounds_y = half
        self.format_bounds_z = half
    else:
        if existing:
            bpy.data.objects.remove(existing, do_unlink=True)


def _update_bounds_display(self, ctx):
    """Toggling Show Bounding Box/Sphere or the display style rebuilds the
    live gizmo from the editable bounds props."""
    from .debug import refresh_bounds_display
    refresh_bounds_display(ctx.scene)


class XBGDebugSettings(bpy.types.PropertyGroup):
    advanced_mode: bpy.props.BoolProperty(
        name="Advanced Mode",
        description="Show inject, export, and all advanced options",
        default=False
    )
    verbose_logging: bpy.props.BoolProperty(
        name="Verbose Logging",
        description="Print detailed debug information to console (bones, chunks, transforms, etc.)",
        default=False
    )
    show_file_info: bpy.props.BoolProperty(
        name="Show File Info",
        description="Display XBG file chunk information in the panel",
        default=False
    )
    show_format_bounds: bpy.props.BoolProperty(
        name="Show XBG Format Bounds",
        description="Display the 16-bit coordinate limit as a lattice box",
        default=False,
        update=_update_format_bounds
    )
    show_bounding_box: bpy.props.BoolProperty(
        name="Show Bounding Boxes",
        description="Show the model's XOBB bounding box as a live gizmo you "
                    "can edit below (and see update in the viewport)",
        default=False,
        update=_update_bounds_display
    )
    show_bounding_sphere: bpy.props.BoolProperty(
        name="Show Bounding Spheres",
        description="Show the model's HPSB bounding sphere as a live gizmo "
                    "you can edit below (and see update in the viewport)",
        default=False,
        update=_update_bounds_display
    )
    bounds_display_type: bpy.props.EnumProperty(
        name="Display Type",
        description="How to display bounding volumes",
        items=[
            ('WIRE', 'Wire', 'Display as wireframe'),
            ('SOLID', 'Solid', 'Display as solid with transparency'),
            ('LATTICE', 'Lattice', 'Display as lattice modifier on box')
        ],
        default='WIRE',
        update=_update_bounds_display
    )
    flip_normals: bpy.props.BoolProperty(
        name="Flip Normals on Import",
        description="Flip all face normals after import (fixes inverted normals)",
        default=True
    )
    separate_primitives: bpy.props.BoolProperty(
        name="Separate Primitives",
        description="Create separate mesh objects for each primitive chunk instead of joining them",
        default=False
    )
    use_xml_assembly: bpy.props.BoolProperty(
        name="Use XML Assembly",
        description="Search for and use XML files to properly assemble parts using bone transforms",
        default=False
    )
    auto_smooth_normals: bpy.props.BoolProperty(
        name="Auto Smooth Normals",
        description="Automatically apply smooth shading after import",
        default=True
    )
    merge_distance: bpy.props.FloatProperty(
        name="Merge Distance",
        description="Distance threshold for merging duplicate vertices",
        default=0.0001,
        min=0.0,
        max=1.0,
        precision=4
    )
    import_xbt_as_dds: bpy.props.BoolProperty(
        name="Import XBT as DDS",
        description="Import XBT textures as DDS files instead of PNG. WARNING: DDS format will cause texture painting corruption! Use PNG (default) for texture painting",
        default=False
    )
    use_mb2o: bpy.props.BoolProperty(
        name="Use MB2O Transforms",
        description="Cross-check skeleton bones against MB2O bind matrices (if available), falling back to EDON per-bone when unsure. EDON alone is already verified correct, so this rarely changes anything - it's a safety net, not a fix. Default: OFF",
        default=False
    )
    trace_logging: bpy.props.BoolProperty(
        name="Trace-Level Logging",
        description=(
            "EXPENSIVE — only enable when chasing a specific bug. Adds per-vertex / per-face "
            "/ per-byte detail to the verbose log: full hex dumps of encoded vertices, "
            "before/after Blender→world→int16 position transforms, chunk-splice byte maps, "
            "and a structured .jsonl record file written next to the saved log. Generates "
            "hundreds of KB to a few MB of log data per export. Requires Verbose Logging to "
            "be ON; has no effect on its own"
        ),
        default=False
    )
    file_info_data: bpy.props.StringProperty(
        name="File Info Data",
        default=""
    )
    lod_peek_result: bpy.props.StringProperty(
        name="LOD Peek Result",
        default=""
    )
    lfa_path: bpy.props.StringProperty(
        name="Facial LFA",
        description="This head's .lfa facial pose library — required to "
                    "decode .lfe expressions (they reference its pose names)",
        default="",
        subtype='FILE_PATH'
    )
    mab_skeleton_path: bpy.props.StringProperty(
        name="Animation Skeleton",
        description="The .skeleton file that defines this rig's animation "
                    "bone order (MAB routing masks are indexed by it). "
                    "Leave empty to auto-detect — the importer scans the "
                    "XBG's folder and the .mab's folder (and its parents)",
        default="",
        subtype='FILE_PATH'
    )
    scene_mab_path: bpy.props.StringProperty(
        name="Scene MAB",
        description="The scripted-scene .mab to inspect — its anchors, "
                    "cameras and timed events (sound/dialog/FX cues) are "
                    "listed in the Scene Viewer",
        default="",
        subtype='FILE_PATH'
    )
    scene_report: bpy.props.StringProperty(
        name="Scene Report",
        default=""
    )
    mab_emulate_helpers: bpy.props.BoolProperty(
        name="Emulate Twist / Corrective Bones",
        description="The engine drives twist and elbow/knee corrective "
                    "bones procedurally (the .mab has no data for them). "
                    "ON: add constraints that reproduce that behaviour — "
                    "fixes the collapsing wrist and the knee denting "
                    "inward. OFF: leave helper bones at rest",
        default=True
    )
    mab_twist_bake: bpy.props.BoolProperty(
        name="Exact Twist (bake swing-twist)",
        description="How the forearm/upper-arm TWIST bones are emulated. "
                    "ON (recommended): bake true swing-twist keys per frame — "
                    "the bone gets only the hand's ROLL and none of its bend, "
                    "matching the engine's RollExtractionMode (properly fixes "
                    "the candy-wrapper wrist). OFF: a live Copy-Rotation "
                    "constraint (editable, but copies the Euler-Y component so "
                    "it leaks the bend and only approximates)",
        default=True
    )
    mab_smooth_resample: bpy.props.BoolProperty(
        name="Smooth Playback (SQUAD resample)",
        description="The engine stores spline-compressed rotation and "
                    "evaluates a smooth curve at the game framerate, so a "
                    "15 fps clip still plays smoothly. ON: bake dense in-"
                    "between keys with SQUAD (spherical cubic) interpolation "
                    "through the decoded keys, reproducing that smoothing. "
                    "OFF: key only the decoded frames (sparse / choppy, but "
                    "easier to hand-edit)",
        default=True
    )
    mab_resample_fps: bpy.props.IntProperty(
        name="Smooth Target FPS",
        description="Resample target framerate for Smooth Playback. The clip "
                    "is upsampled toward this rate (e.g. a 15 fps clip → 60). "
                    "Original keyframe poses are preserved exactly",
        default=60, min=24, max=120
    )
    mab_char_offset: bpy.props.IntProperty(
        name="Character Bone Offset",
        description="Scripted-scene clips animate a COMBINED rig (several "
                    "characters + anchors concatenated). 0 = first "
                    "character block. To apply a different NPC's animation "
                    "to your mesh, set the bit offset where that "
                    "character's skeleton starts inside the combined rig "
                    "(try multiples of your skeleton's bone count)",
        default=0,
        min=0,
        max=4096
    )
    mab_skeleton_override: bpy.props.StringProperty(
        name="Skeleton (.xbg)",
        description="Override which rig the .mab bones are matched against. "
                    "Leave blank to use the .xbg the selected armature was "
                    "imported from. Set this to a different model .xbg (e.g. the "
                    "first-person arms / full-body skeleton that matches the "
                    "animation) when the active armature is the wrong or a "
                    "partial rig — its bones (names + parents + rest pose) are "
                    "decoded and used for hash matching",
        subtype='FILE_PATH',
        default=""
    )
    wd_reskin_weights: bpy.props.BoolProperty(
        name="Include Bone Weights",
        description="Same as the Avatar inject option: write vertex-group "
                    "weights into the mesh. ON: re-derive EVERY vertex's "
                    "bone weights from its vertex groups (for weight "
                    "painting / re-rigging). OFF: existing vertices keep "
                    "their original weights byte-exact and only newly added "
                    "geometry is skinned from its groups",
        default=False
    )
    wd_recalculate_normals: bpy.props.BoolProperty(
        name="Recalculate Normals + Tangents",
        description="ON: recompute the full TBN frame from geometry — normal "
                    "from face angles, tangent + binormal from UV layout "
                    "(MikkTSpace). Use this after sculpting so normals and "
                    "the normal-map tangent space match the new vertex "
                    "positions. OFF (default): preserve the original authored "
                    "values for unchanged vertices; only newly added geometry "
                    "is recomputed regardless",
        default=False
    )
    compact_vertices: bpy.props.BoolProperty(
        name="Compact Vertices (Remove Unused)",
        description="Remove unused vertices during import. A vertex mapping is stored to ensure correct export positions. Recommended for cleaner editing.",
        default=True
    )

    link_xy: bpy.props.BoolProperty(name="Link X↔Y", default=False)
    link_yz: bpy.props.BoolProperty(name="Link Y↔Z", default=False)
    link_xz: bpy.props.BoolProperty(name="Link X↔Z", default=False)

    def _apply_bounds(self):
        lo = bpy.data.objects.get("XBG_Format_Bounds")
        if lo:
            lo.scale = (self.format_bounds_x, self.format_bounds_y, self.format_bounds_z)

    def _update_x(self, ctx):
        if self.link_xy and self.link_xz:
            self['format_bounds_y'] = self.format_bounds_x
            self['format_bounds_z'] = self.format_bounds_x
        elif self.link_xy:
            self['format_bounds_y'] = self.format_bounds_x
        elif self.link_xz:
            self['format_bounds_z'] = self.format_bounds_x
        self._apply_bounds()

    def _update_y(self, ctx):
        if self.link_xy and self.link_yz:
            self['format_bounds_x'] = self.format_bounds_y
            self['format_bounds_z'] = self.format_bounds_y
        elif self.link_xy:
            self['format_bounds_x'] = self.format_bounds_y
        elif self.link_yz:
            self['format_bounds_z'] = self.format_bounds_y
        self._apply_bounds()

    def _update_z(self, ctx):
        if self.link_xz and self.link_yz:
            self['format_bounds_x'] = self.format_bounds_z
            self['format_bounds_y'] = self.format_bounds_z
        elif self.link_xz:
            self['format_bounds_x'] = self.format_bounds_z
        elif self.link_yz:
            self['format_bounds_y'] = self.format_bounds_z
        self._apply_bounds()

    format_bounds_x: bpy.props.FloatProperty(
        name="X",
        description="Half-size of the format bounds lattice on X axis",
        default=72.0,
        min=0.001,
        precision=4,
        update=_update_x
    )
    format_bounds_y: bpy.props.FloatProperty(
        name="Y",
        description="Half-size of the format bounds lattice on Y axis",
        default=72.0,
        min=0.001,
        precision=4,
        update=_update_y
    )
    format_bounds_z: bpy.props.FloatProperty(
        name="Z",
        description="Half-size of the format bounds lattice on Z axis",
        default=72.0,
        min=0.001,
        precision=4,
        update=_update_z
    )

    reorient_bones: bpy.props.BoolProperty(
        name="Reorient Bones",
        description="Point each bone's tail toward its children, making the skeleton easier to read and pose. Leaf bones keep their original orientation.",
        default=False
    )


class XBGCollisionSettings(bpy.types.PropertyGroup):
    last_import_path: bpy.props.StringProperty(
        name="Last HKX Import Path",
        default="",
        subtype='FILE_PATH'
    )
    export_output_path: bpy.props.StringProperty(
        name="Export Output Path",
        default="",
        subtype='FILE_PATH'
    )
    last_status: bpy.props.StringProperty(
        name="Last Status",
        default=""
    )
    last_status_ok: bpy.props.BoolProperty(
        name="Last Status OK",
        default=True
    )
    # (manual_header_hex / original_hkx_path removed 2026-06-11 — they
    #  belonged to the legacy XML workflow; the native .hkx editor keeps
    #  the 16-byte game header intact automatically.)


# ---------------------------------------------------------------------------
# Operators — import
# ---------------------------------------------------------------------------

# Short, factual description of each template — shown both as the
# EnumProperty tooltip (third tuple element) and as a sub-line in the
# export popup under the per-material picker.  Phrased as "what this
# template supports / when to use it" so non-experts can pick correctly.
_TEMPLATE_DESCRIPTIONS = {
    'AUTO':          "Pick automatically from the material's node graph "
                     "(BSDF present → Generic; otherwise → Unlit).",
    # --- Avatar (in-game tested) ---
    'Generic':       "Universal lit shader. Diffuse + Normal + Specular + "
                     "Alpha + optional Emission. Works on static and skinned meshes.",
    'Flesh':         "Character skin / body. Like Generic + DiffuseTexture2 "
                     "tattoo overlay. Built for skinned character meshes.",
    'Cloth':         "Hair / fabric. Like Flesh + RimLighting. Templates "
                     "ship with AlphaBlend + TwoSided enabled.",
    'Leaf':          "SSS foliage. Alpha-test + TwoSided. No normal map "
                     "(uses SpecularID). SSS colour / strength / highlight controls.",
    'BigLeaf':       "Translucent leaves / cork. SSS + Illumination/Bio glow + "
                     "TranslucencyEnabled.",
    'Grass':         "Billboard grass. Diffuse-only + alpha-test + BillboardEnabled. "
                     "Minimal schema, no normal/specular.",
    'RealtreeTrunk': "Tree bark. Burnt-state diffuse colour, no illumination, "
                     "no character skinning.",
    'Unlit':         "Flat textured, no lighting. HDRMul controls brightness. "
                     "Use for emission-only panels / UI / pure-texture decals.",
    'Weapon':        "Weapon shader. Clean/Broken damage colour triples + "
                     "Reflection cubemap. Reliability scalar.",
    # --- Far Cry 2 only (Dunia engine variants Avatar does NOT use) ---
    # Pick these ONLY when exporting for an FC2 mod.  Avatar's compiled
    # shaders likely have no permutation for FC2_Skin/Hair/Vehicle or the
    # FC2-flavoured Generic/Cloth schemas — using them on an Avatar mesh
    # will most likely crash on save reload.
    'FC2_Generic':   "FC2 only. Generic + MaskTexture1 (dirt/wear). "
                     "AVATAR-INCOMPATIBLE — use only for Far Cry 2 mods.",
    'FC2_Cloth':     "FC2 only. Cloth + BloodTexture + PrintTexture + "
                     "FabricTexture + RimLightTexture. FC2 mods only.",
    'FC2_Skin':      "FC2 only. Character skin shader (FC2 equivalent of "
                     "Avatar's Flesh). Has SkinTexture + BloodTexture. FC2 mods only.",
    'FC2_Hair':      "FC2 only. Dedicated hair shader with Fresnel "
                     "colour/power controls. FC2 mods only.",
    'FC2_Vehicle':   "FC2 only. Vehicle bodywork with Clean/Broken damage "
                     "states and dirt mask. FC2 mods only.",
    'FC2_Water':     "FC2 only. Water surface shader with caustics, fresnel, "
                     "reflection cubemap and animated normal maps. FC2 mods only.",
}

_TEMPLATE_ITEMS = [
    (key, key if key != 'AUTO' else 'Auto', _TEMPLATE_DESCRIPTIONS[key])
    for key in (
        # Avatar
        'AUTO', 'Generic', 'Flesh', 'Cloth', 'Leaf', 'BigLeaf',
        'Grass', 'RealtreeTrunk', 'Unlit', 'Weapon',
        # Far Cry 2
        'FC2_Generic', 'FC2_Cloth', 'FC2_Skin', 'FC2_Hair', 'FC2_Vehicle',
        'FC2_Water',
    )
]


class XBGMatTemplateItem(bpy.types.PropertyGroup):
    """One entry in the per-material template list."""
    mat_name:  bpy.props.StringProperty()
    is_game:   bpy.props.BoolProperty()
    auto_type: bpy.props.StringProperty()   # detected type when template == 'AUTO'
    template:  bpy.props.EnumProperty(
        name="Template",
        items=_TEMPLATE_ITEMS,
        default='AUTO')


