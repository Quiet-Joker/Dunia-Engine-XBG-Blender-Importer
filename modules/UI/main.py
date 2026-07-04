"""Main UI — game picker.

The sidebar opens on a "Select the game you wish to modify" screen with one
button per game.  Picking a game swaps the panel content to that game's
tools; the back arrow returns to the picker.  The selection lives in
``Scene.xbg_active_game`` so it survives undo / file boundaries gracefully.

Per-game tool panels are ordinary sub-panels parented to that game's root
container panel (e.g. ``OBJECT_PT_xbg_avatar``), so a game's whole UI hides
with one poll check.
"""

import bpy

from ..Core import prefs as _prefs


# (identifier, button label, supported)
GAMES = [
    ('AVATAR', "Avatar: The Game", True),
    ('FC1',    "Far Cry 1",        True),    # CryEngine 1 .cgf — leaked source, not reverse-engineered
    ('FCI',    "Far Cry Instincts", True),   # Xbox 2005 — unrelated earlier .xbg format
    ('FC2',    "Far Cry 2",        True),    # Dunia — shares the Avatar tools
    ('FC3',    "Far Cry 3",        True),
    ('FC4',    "Far Cry 4",        True),    # same GEOM path as FC3
    ('FC5',    "Far Cry 5",        True),     # mesh RE in progress
    ('PRIMAL', "Far Cry Primal",   True),     # FC4-family GEOM (0x0006003A)
    ('FC6',    "Far Cry 6",        False),
    ('WD1',    "Watch Dogs 1",     True),
    ('WD2',    "Watch Dogs 2",     True),
]

GAME_LABELS = {gid: label for gid, label, _ in GAMES}
SUPPORTED = {gid for gid, _, ok in GAMES if ok}

GAME_ENUM_ITEMS = [('NONE', "None", "No game selected")] + [
    (gid, label, label) for gid, label, _ in GAMES
]


def active_game(ctx):
    return getattr(ctx.scene, 'xbg_active_game', 'NONE')


class XBG_OT_SelectGame(bpy.types.Operator):
    """Switch the sidebar to this game's tools (or back to the picker)."""
    bl_idname = "xbg.select_game"
    bl_label = "Select Game"
    bl_options = {'INTERNAL'}

    game: bpy.props.StringProperty(default='NONE')

    def execute(self, ctx):
        ctx.scene.xbg_active_game = self.game
        return {'FINISHED'}


class XBG_PT_Panel(bpy.types.Panel):
    """Root panel: update bar + game picker / per-game header."""
    bl_label = "XBG Importer"
    bl_idname = "OBJECT_PT_xbg_import"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "XBG Import"

    def draw(self, ctx):
        l = self.layout

        game = active_game(ctx)

        if game == 'NONE':
            # ── Update bar (only on the home screen) ────────────────────
            if _prefs._update_status is None and _prefs._update_error is None:
                l.operator("xbg.check_for_updates",
                           text="Check for Updates", icon="FILE_REFRESH")
            elif _prefs._update_error:
                row = l.row()
                row.label(text="Update check failed", icon="ERROR")
                row.operator("xbg.check_for_updates", text="Retry",
                             icon="FILE_REFRESH")
            elif _prefs._update_status == "up_to_date":
                row = l.row()
                row.label(text="Plugin is up to date", icon="CHECKMARK")
                row.operator("xbg.check_for_updates", text="",
                             icon="FILE_REFRESH")
            else:
                box = l.box()
                box.label(text="New changes are available", icon="INFO")
                row = box.row()
                row.operator("xbg.apply_update", text="Update Now",
                             icon="IMPORT")
                row.operator("xbg.check_for_updates", text="",
                             icon="FILE_REFRESH")
            l.separator()

            # ── Game picker ─────────────────────────────────────────────
            l.label(text="Select the game you wish to modify:",
                    icon='RESTRICT_SELECT_OFF')
            col = l.column(align=True)
            col.scale_y = 1.4
            for gid, label, supported in GAMES:
                if supported:
                    op = col.operator("xbg.select_game", text=label)
                    op.game = gid
            l.separator()
            sub = l.column(align=True)
            sub.scale_y = 1.1
            sub.enabled = False
            sub.label(text="Coming soon:")
            for gid, label, supported in GAMES:
                if not supported:
                    sub.operator("xbg.select_game", text=label, icon='LOCKED')
            return

        # ── A game is selected: back arrow + title ──────────────────────
        row = l.row(align=True)
        op = row.operator("xbg.select_game", text="", icon='BACK')
        op.game = 'NONE'
        row.label(text=GAME_LABELS.get(game, game))

        if game not in SUPPORTED:
            box = l.box()
            box.label(text="No tools for this game yet.", icon='INFO')
            box.label(text="Support is planned — check for updates.")


class XBG_PT_AvatarRoot(bpy.types.Panel):
    """Container for the Avatar / Far Cry 2 (Dunia) toolset."""
    bl_label = "Tools"
    bl_idname = "OBJECT_PT_xbg_avatar"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "XBG Import"
    bl_parent_id = "OBJECT_PT_xbg_import"
    bl_options = {'HIDE_HEADER'}

    @classmethod
    def poll(cls, ctx):
        return active_game(ctx) == 'AVATAR'

    def draw(self, ctx):
        l = self.layout
        ds = ctx.scene.xbg_debug_settings
        row = l.row()
        row.scale_y = 1.3
        icon = 'SETTINGS' if ds.advanced_mode else 'PREFERENCES'
        row.prop(ds, "advanced_mode", text="Advanced Mode", icon=icon,
                 toggle=True)


class XBG_PT_FC1Root(bpy.types.Panel):
    """Container for the Far Cry 1 toolset (panels in panels_fc1.py).

    CryEngine 1 .cgf -- documented from the leaked engine source, not
    reverse-engineered. Same root layout as Avatar (the master UI template):
    an Advanced Mode toggle, even while FC1 has few advanced-only panels.
    """
    bl_label = "Tools"
    bl_idname = "OBJECT_PT_xbg_fc1"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "XBG Import"
    bl_parent_id = "OBJECT_PT_xbg_import"
    bl_options = {'HIDE_HEADER'}

    @classmethod
    def poll(cls, ctx):
        return active_game(ctx) == 'FC1'

    def draw(self, ctx):
        l = self.layout
        ds = ctx.scene.xbg_debug_settings
        row = l.row()
        row.scale_y = 1.3
        icon = 'SETTINGS' if ds.advanced_mode else 'PREFERENCES'
        row.prop(ds, "advanced_mode", text="Advanced Mode", icon=icon,
                 toggle=True)


class XBG_PT_FCIRoot(bpy.types.Panel):
    """Container for the Far Cry Instincts toolset (panels in panels_fci.py).

    Instincts is an unrelated, earlier .xbg format (Xbox, 2005). Same root
    layout as Avatar (the master UI template): an Advanced Mode toggle, even
    while FCI has few advanced-only panels.
    """
    bl_label = "Tools"
    bl_idname = "OBJECT_PT_xbg_fci"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "XBG Import"
    bl_parent_id = "OBJECT_PT_xbg_import"
    bl_options = {'HIDE_HEADER'}

    @classmethod
    def poll(cls, ctx):
        return active_game(ctx) == 'FCI'

    def draw(self, ctx):
        l = self.layout
        ds = ctx.scene.xbg_debug_settings
        row = l.row()
        row.scale_y = 1.3
        icon = 'SETTINGS' if ds.advanced_mode else 'PREFERENCES'
        row.prop(ds, "advanced_mode", text="Advanced Mode", icon=icon,
                 toggle=True)


class XBG_PT_FC2Root(bpy.types.Panel):
    """Container for the Far Cry 2 toolset (independent Dunia-1 clone;
    panels in panels_fc2.py)."""
    bl_label = "Tools"
    bl_idname = "OBJECT_PT_xbg_fc2"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "XBG Import"
    bl_parent_id = "OBJECT_PT_xbg_import"
    bl_options = {'HIDE_HEADER'}

    @classmethod
    def poll(cls, ctx):
        return active_game(ctx) == 'FC2'

    def draw(self, ctx):
        l = self.layout
        ds = ctx.scene.xbg_debug_settings
        row = l.row()
        row.scale_y = 1.3
        icon = 'SETTINGS' if ds.advanced_mode else 'PREFERENCES'
        row.prop(ds, "advanced_mode", text="Advanced Mode", icon=icon,
                 toggle=True)


class XBG_PT_FC3Root(bpy.types.Panel):
    """Container for the Far Cry 3 / 4 / 5 / Primal toolset
    (panels in panels_fc3.py)."""
    bl_label = "Tools"
    bl_idname = "OBJECT_PT_xbg_fc3"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "XBG Import"
    bl_parent_id = "OBJECT_PT_xbg_import"
    bl_options = {'HIDE_HEADER'}

    @classmethod
    def poll(cls, ctx):
        return active_game(ctx) in ('FC3', 'FC4', 'FC5', 'PRIMAL')

    def draw(self, ctx):
        # Must draw SOMETHING: an empty HIDE_HEADER container can fail to
        # render its child sub-panels, which left the FC3 tab blank (FC4 only
        # worked because its Animation sub-panel anchored the group).  The
        # Advanced Mode toggle doubles as that anchor, matching the Avatar
        # master layout.
        l = self.layout
        ds = ctx.scene.xbg_debug_settings
        row = l.row()
        row.scale_y = 1.3
        icon = 'SETTINGS' if ds.advanced_mode else 'PREFERENCES'
        row.prop(ds, "advanced_mode", text="Advanced Mode", icon=icon,
                 toggle=True)


class XBG_PT_WDRoot(bpy.types.Panel):
    """Container for the Watch Dogs 1 / 2 toolset (panels in panels_wd.py)."""
    bl_label = "Watch Dogs Tools"
    bl_idname = "OBJECT_PT_xbg_wd"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "XBG Import"
    bl_parent_id = "OBJECT_PT_xbg_import"
    bl_options = {'HIDE_HEADER'}

    @classmethod
    def poll(cls, ctx):
        return active_game(ctx) in ('WD1', 'WD2')

    def draw(self, ctx):
        l = self.layout
        ds = ctx.scene.xbg_debug_settings
        row = l.row()
        row.scale_y = 1.3
        icon = 'SETTINGS' if ds.advanced_mode else 'PREFERENCES'
        row.prop(ds, "advanced_mode", text="Advanced Mode", icon=icon,
                 toggle=True)
