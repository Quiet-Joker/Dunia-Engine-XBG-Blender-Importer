"""XBG Importer — multi-game Ubisoft model/animation tools for Blender.

Layout (2026-06-09 reorganisation):
    modules/Core/       addon prefs, updater, settings PropertyGroups, log ops
    modules/Shared/     game-agnostic binary/debug/chunk helpers
    modules/Avatar/     Avatar: The Game / Far Cry 2 (Dunia) tools
    modules/Far_Cry_3/    Far Cry 3 / 4 tools
    modules/Watch_Dogs/  Watch Dogs 1 / 2 tools
    modules/UI/         game-picker root panel + per-game panels

This file only assembles the pieces and registers them.
"""

bl_info = {
    "name": "Dunia Engine XBG Blender Importer",
    "author": "Quiet Joker, Jasper_Zebra",
    "version": (3, 0, 0),
    "blender": (5, 0, 0),
    "location": "View3D > Sidebar > XBG Import",
    "description": "Import/edit/re-export models from Avatar: The Game, "
                   "Far Cry 1/2/3/4/5/Primal/Instincts and Watch Dogs 1/2",
    "category": "Import-Export",
}

import bpy

from .modules.Core.debug import VerboseLogger

from .modules.Core.prefs import (
    XBGAddonPreferences,
    XBG_OT_CheckForUpdates,
    XBG_OT_ApplyUpdate,
    startup_update_check,
)
from .modules.Core.settings import (
    XBGImportSettings,
    XBGInjectSettings,
    XBGSessionData,
    XBGDebugSettings,
    XBGCollisionSettings,
    XBGMatTemplateItem,
)
from .modules.Core.ops_log import (
    XBG_OT_ResetLog,
    XBG_OT_SaveLog,
)

from .modules.Avatar.operators_avatar import (
    XBG_OT_Import,
    XBG_OT_RememberXBG,
    XBG_OT_ClearSessionXBG,
    XBG_OT_MergeAllMeshes,
    XBG_OT_MergeSelectedMesh,
    XBG_OT_AutoScaleBounds,
    XBG_OT_InjectMesh,
    XBG_OT_PeekLODs,
    XBG_OT_ExpandBoundsForInject,
    XBG_OT_SaveFormatBoundsSize,
    XBG_OT_ImportLKSSkeleton,
    XBG_OT_ImportMAB,
    XBG_OT_PreviewJiggle,
    XBG_OT_ImportLFA,
    XBG_OT_ImportLFE,
    XBG_OT_ScanSceneMAB,
    XBG_OT_ImportSceneMAB,
    XBG_OT_NullSelectedVerts,
    XBG_OT_ExportMaterials,
)
from .modules.Avatar.operators_hkx_avatar import (
    XBG_OT_ImportHKXNative,
    XBG_OT_ExportHKXNative,
)
from .modules.Avatar.jiggle_avatar import (
    XBGJiggleBoneSettings,
    XBG_OT_BakeJiggle,
    XBG_OT_ClearJiggle,
    XBG_OT_InjectJigglePatch,
    XBG_PT_JigglePanel,
)
from .modules.Avatar.lod_distance_avatar import (
    XBGLodDistItem,
    XBG_OT_ReadLODDistances,
    XBG_OT_WriteLODDistances,
    XBG_PT_LODDistancePanel,
)
from .modules.Avatar.bounds_editor_avatar import (
    XBG_OT_ReadBounds,
    XBG_OT_FitBoundsToSelected,
    XBG_OT_WriteBounds,
)
from .modules.Avatar.normalize_weights_avatar import (
    XBG_OT_NormalizeWeights,
)

from .modules.Far_Cry_2.operators_fc2 import (
    XBG_OT_ImportFC2,
    XBG_OT_RememberXBGFC2,
    XBG_OT_ClearSessionXBGFC2,
    XBG_OT_MergeAllMeshesFC2,
    XBG_OT_MergeSelectedMeshFC2,
    XBG_OT_AutoScaleBoundsFC2,
    XBG_OT_InjectMeshFC2,
    XBG_OT_PeekLODsFC2,
    XBG_OT_ExpandBoundsForInjectFC2,
    XBG_OT_SaveFormatBoundsSizeFC2,
    XBG_OT_ImportLKSSkeletonFC2,
    XBG_OT_ImportMABFC2,
    XBG_OT_PreviewJiggleFC2,
    XBG_OT_ImportLFAFC2,
    XBG_OT_ImportLFEFC2,
    XBG_OT_ScanSceneMABFC2,
    XBG_OT_ImportSceneMABFC2,
    XBG_OT_NullSelectedVertsFC2,
    XBG_OT_ExportMaterialsFC2,
)
from .modules.Far_Cry_2.operators_hkx_fc2 import (
    XBG_OT_ImportHKXNativeFC2,
    XBG_OT_ExportHKXNativeFC2,
)
from .modules.Far_Cry_2.jiggle_fc2 import (
    XBGJiggleBoneSettingsFC2,
    XBG_OT_BakeJiggleFC2,
    XBG_OT_ClearJiggleFC2,
    XBG_OT_InjectJigglePatchFC2,
    XBG_PT_JigglePanelFC2,
)
from .modules.Far_Cry_2.lod_distance_fc2 import (
    XBGLodDistItemFC2,
    XBG_OT_ReadLODDistancesFC2,
    XBG_OT_WriteLODDistancesFC2,
    XBG_PT_LODDistancePanelFC2,
)
from .modules.Far_Cry_2.bounds_editor_fc2 import (
    XBG_OT_ReadBoundsFC2,
    XBG_OT_FitBoundsToSelectedFC2,
    XBG_OT_WriteBoundsFC2,
    XBG_PT_BoundsEditorPanelFC2,
)

from .modules.Far_Cry_1.operators_fc1 import XBG_OT_ImportFC1
from .modules.Far_Cry_Instincts.operators_fci import XBG_OT_ImportFCI
from .modules.Far_Cry_3.operators_fc3 import (XBG_OT_ImportFC3, XBG_OT_InjectFC3,
                                        XBG_OT_ImportFC3Mab,
                                        XBG_OT_ImportFC3Skeleton,
                                        XBG_OT_ImportFC3Hkx)
from .modules.Far_Cry_4.operators_fc4 import (XBG_OT_ImportFC4, XBG_OT_InjectFC4,
                                        XBG_OT_ImportFC4Mab)
from .modules.Far_Cry_5.operators_fc5 import (XBG_OT_ImportFC5, XBG_OT_ImportFC5Mab,
                                        XBG_OT_InjectFC5)
from .modules.Far_Cry_Primal.operators_primal import (XBG_OT_ImportPrimal,
                                        XBG_OT_InjectPrimal)
from .modules.Watch_Dogs.operators_wd import (
    XBG_OT_ImportWD, XBG_OT_ImportWDMab, XBG_OT_InjectWD,
    XBG_OT_WDPeekLODs, XBG_OT_WDSyncNormals,
    XBG_OT_ImportWDSkeleton, XBG_OT_ImportWDHkx)
from .modules.Watch_Dogs_2.operators_wd2 import XBG_OT_ImportWD2, XBG_OT_ExportWD2

from .modules.UI.main import (
    XBG_OT_SelectGame,
    XBG_PT_Panel,
    XBG_PT_AvatarRoot,
    XBG_PT_FC1Root,
    XBG_PT_FCIRoot,
    XBG_PT_FC2Root,
    XBG_PT_FC3Root,
    XBG_PT_WDRoot,
)
from .modules.UI.panels_fc1 import (
    XBG_PT_FC1Import,
    XBG_PT_FC1ModelInfo,
)
from .modules.UI.panels_fci import (
    XBG_PT_FCIImport,
    XBG_PT_FCIModelInfo,
)
from .modules.UI.panels_fc2 import (
    XBG_PT_ImportPanelFC2,
    XBG_PT_InjectPanelFC2,
    XBG_PT_InjectAdvancedPanelFC2,
    XBG_PT_DebugPanelFC2,
    XBG_PT_DebugMenuPanelFC2,
    XBG_PT_SkeletonPanelFC2,
    XBG_PT_FacialPanelFC2,
    XBG_PT_SceneViewerPanelFC2,
    XBG_PT_HKXPanelFC2,
)
from .modules.UI.panels_avatar import (
    XBG_PT_ImportPanel,
    XBG_PT_InjectPanel,
    XBG_PT_InjectAdvancedPanel,
    XBG_PT_MeshCleanupPanel,
    XBG_PT_SkeletonMB2OPanel,
    XBG_PT_ExperimentalInjectPanel,
    XBG_PT_DiagnosticsPanel,
    XBG_PT_ViewportVizPanel,
    XBG_PT_ExportMaterialsPanel,
    XBG_PT_SkeletonPanel,
    XBG_PT_FacialPanel,
    XBG_PT_SceneViewerPanel,
    XBG_PT_HKXPanel,
)
from .modules.UI.panels_fc3 import (
    XBG_PT_FC3Import,
    XBG_PT_FC4Animation,
    XBG_PT_FC3Sections,
    XBG_PT_FC3Inject,
)
from .modules.UI.panels_wd import (
    XBG_PT_WDImport,
    XBG_PT_WDAnimation,
    XBG_PT_WDInject,
    XBG_PT_WDDebug,
    XBG_PT_WDModelInfo,
)


classes = (
    # preferences + settings (PropertyGroups first)
    XBGAddonPreferences,
    XBGImportSettings,
    XBGInjectSettings,
    XBGSessionData,
    XBGDebugSettings,
    XBGCollisionSettings,
    XBGMatTemplateItem,
    XBGJiggleBoneSettings,
    XBGLodDistItem,
    # core ops
    XBG_OT_CheckForUpdates,
    XBG_OT_ApplyUpdate,
    XBG_OT_ResetLog,
    XBG_OT_SaveLog,
    # Avatar / FC2
    XBG_OT_Import,
    XBG_OT_AutoScaleBounds,
    XBG_OT_RememberXBG,
    XBG_OT_ClearSessionXBG,
    XBG_OT_MergeAllMeshes,
    XBG_OT_MergeSelectedMesh,
    XBG_OT_InjectMesh,
    XBG_OT_PeekLODs,
    XBG_OT_ExpandBoundsForInject,
    XBG_OT_SaveFormatBoundsSize,
    XBG_OT_ImportLKSSkeleton,
    XBG_OT_ImportMAB,
    XBG_OT_PreviewJiggle,
    XBG_OT_ImportLFA,
    XBG_OT_ImportLFE,
    XBG_OT_ScanSceneMAB,
    XBG_OT_ImportSceneMAB,
    XBG_OT_NullSelectedVerts,
    XBG_OT_ExportMaterials,
    # Avatar HKX collision (native binary)
    XBG_OT_ImportHKXNative,
    XBG_OT_ExportHKXNative,
    # Avatar jiggle authoring
    XBG_OT_BakeJiggle,
    XBG_OT_ClearJiggle,
    XBG_OT_InjectJigglePatch,
    # Avatar LOD-distance editor
    XBG_OT_ReadLODDistances,
    XBG_OT_WriteLODDistances,
    # Avatar bounding-volume editor
    XBG_OT_ReadBounds,
    XBG_OT_FitBoundsToSelected,
    XBG_OT_WriteBounds,
    # Far Cry 2 (independent clone of the Avatar/Dunia-1 toolset)
    XBGJiggleBoneSettingsFC2,
    XBGLodDistItemFC2,
    XBG_OT_ImportFC2,
    XBG_OT_AutoScaleBoundsFC2,
    XBG_OT_RememberXBGFC2,
    XBG_OT_ClearSessionXBGFC2,
    XBG_OT_MergeAllMeshesFC2,
    XBG_OT_MergeSelectedMeshFC2,
    XBG_OT_InjectMeshFC2,
    XBG_OT_PeekLODsFC2,
    XBG_OT_ExpandBoundsForInjectFC2,
    XBG_OT_SaveFormatBoundsSizeFC2,
    XBG_OT_ImportLKSSkeletonFC2,
    XBG_OT_ImportMABFC2,
    XBG_OT_PreviewJiggleFC2,
    XBG_OT_ImportLFAFC2,
    XBG_OT_ImportLFEFC2,
    XBG_OT_ScanSceneMABFC2,
    XBG_OT_ImportSceneMABFC2,
    XBG_OT_NullSelectedVertsFC2,
    XBG_OT_ExportMaterialsFC2,
    XBG_OT_ImportHKXNativeFC2,
    XBG_OT_ExportHKXNativeFC2,
    XBG_OT_BakeJiggleFC2,
    XBG_OT_ClearJiggleFC2,
    XBG_OT_InjectJigglePatchFC2,
    XBG_OT_ReadLODDistancesFC2,
    XBG_OT_WriteLODDistancesFC2,
    XBG_OT_ReadBoundsFC2,
    XBG_OT_FitBoundsToSelectedFC2,
    XBG_OT_WriteBoundsFC2,
    # Far Cry 1 (CryEngine 1 .cgf — leaked source, not reverse-engineered)
    XBG_OT_ImportFC1,
    # Far Cry Instincts (Xbox 2005 — unrelated earlier .xbg format)
    XBG_OT_ImportFCI,
    # Far Cry 3
    XBG_OT_ImportFC3,
    XBG_OT_InjectFC3,
    XBG_OT_ImportFC3Mab,
    XBG_OT_ImportFC3Skeleton,
    XBG_OT_ImportFC3Hkx,
    # Far Cry 4
    XBG_OT_ImportFC4,
    XBG_OT_InjectFC4,
    XBG_OT_ImportFC4Mab,
    # Far Cry 5 (RE in progress)
    XBG_OT_ImportFC5,
    XBG_OT_ImportFC5Mab,
    XBG_OT_InjectFC5,
    # Far Cry Primal (FC4-family GEOM, version 0x0006003A)
    XBG_OT_ImportPrimal,
    XBG_OT_InjectPrimal,
    # Watch Dogs 1 / 2
    XBG_OT_ImportWD,
    XBG_OT_ImportWDMab,
    XBG_OT_InjectWD,
    XBG_OT_WDPeekLODs,
    XBG_OT_WDSyncNormals,
    XBG_OT_ImportWDSkeleton,
    XBG_OT_ImportWDHkx,
    # Watch Dogs 2 (import-only, own folder)
    XBG_OT_ImportWD2,
    XBG_OT_ExportWD2,
    # UI (order matters: parents before children)
    XBG_OT_SelectGame,
    XBG_PT_Panel,
    XBG_PT_AvatarRoot,
    XBG_PT_ImportPanel,
    XBG_PT_InjectPanel,
    XBG_PT_InjectAdvancedPanel,
    XBG_PT_MeshCleanupPanel,
    XBG_PT_SkeletonMB2OPanel,
    XBG_PT_ExperimentalInjectPanel,
    XBG_PT_DiagnosticsPanel,
    XBG_PT_ViewportVizPanel,
    XBG_PT_ExportMaterialsPanel,
    XBG_PT_SkeletonPanel,
    XBG_PT_FacialPanel,
    XBG_PT_SceneViewerPanel,
    XBG_PT_HKXPanel,
    XBG_PT_JigglePanel,
    XBG_PT_LODDistancePanel,
    XBG_OT_NormalizeWeights,
    # Far Cry 2 UI (independent clone)
    XBG_PT_FC2Root,
    XBG_PT_ImportPanelFC2,
    XBG_PT_InjectPanelFC2,
    XBG_PT_InjectAdvancedPanelFC2,
    XBG_PT_DebugPanelFC2,
    XBG_PT_DebugMenuPanelFC2,
    XBG_PT_SkeletonPanelFC2,
    XBG_PT_FacialPanelFC2,
    XBG_PT_SceneViewerPanelFC2,
    XBG_PT_HKXPanelFC2,
    XBG_PT_JigglePanelFC2,
    XBG_PT_LODDistancePanelFC2,
    XBG_PT_BoundsEditorPanelFC2,
    XBG_PT_FC3Root,
    XBG_PT_FC3Import,
    XBG_PT_FC4Animation,
    XBG_PT_FC3Sections,
    XBG_PT_FC3Inject,
    XBG_PT_FC1Root,
    XBG_PT_FC1Import,
    XBG_PT_FC1ModelInfo,
    XBG_PT_FCIRoot,
    XBG_PT_FCIImport,
    XBG_PT_FCIModelInfo,
    XBG_PT_WDRoot,
    XBG_PT_WDImport,
    XBG_PT_WDAnimation,
    XBG_PT_WDInject,
    XBG_PT_WDDebug,
    XBG_PT_WDModelInfo,
)


def register():
    for c in classes:
        bpy.utils.register_class(c)
    S = bpy.types.Scene
    S.xbg_settings           = bpy.props.PointerProperty(type=XBGImportSettings)
    S.xbg_inject_settings    = bpy.props.PointerProperty(type=XBGInjectSettings)
    S.xbg_session_data       = bpy.props.PointerProperty(type=XBGSessionData)
    S.xbg_debug_settings     = bpy.props.PointerProperty(type=XBGDebugSettings)
    S.xbg_collision_settings = bpy.props.PointerProperty(type=XBGCollisionSettings)
    # game-picker selection; plain string so 'NONE' (the picker screen)
    # needs no enum bookkeeping when new games are added
    S.xbg_active_game = bpy.props.StringProperty(default='NONE')
    # per-bone procedural-jiggle authoring settings (sliders + inject data)
    bpy.types.PoseBone.xbg_jiggle = bpy.props.PointerProperty(type=XBGJiggleBoneSettings)
    # LOD-distance editor state (loaded from an XBG's SDOL chunk)
    S.xbg_lod_dists = bpy.props.CollectionProperty(type=XBGLodDistItem)
    S.xbg_lod_dist_path = bpy.props.StringProperty(default='')
    S.xbg_lod_dist_endian = bpy.props.StringProperty(default='LE')
    # XOBB box / HPSB sphere editor state
    S.xbg_bounds_path = bpy.props.StringProperty(default='')
    S.xbg_bounds_endian = bpy.props.StringProperty(default='LE')
    S.xbg_has_xobb = bpy.props.BoolProperty(default=False)
    S.xbg_has_hpsb = bpy.props.BoolProperty(default=False)
    S.xbg_xobb_off = bpy.props.IntProperty(default=0)
    S.xbg_hpsb_off = bpy.props.IntProperty(default=0)
    # Name of the mesh object whose matrix_world defines how the box/sphere
    # gizmo is displayed (2026-06-30 fix — see Core/debug._bounds_display_frame
    # for why it must be a MESH, not its parent armature). Set whenever bounds
    # are populated from a live model (import, Fit to Selected); cleared
    # (raw file-space display) when read from an arbitrary file with no
    # live reference.
    S.xbg_bounds_frame_obj = bpy.props.StringProperty(default='')
    # Editing any of these live-updates the XBG_Bounds_Box / XBG_Bounds_Sphere
    # gizmo (see modules/Core/debug.refresh_bounds_display) so the user sees
    # the box/sphere resize in the viewport as they drag the sliders.
    def _live_bounds(self, ctx):
        from .modules.Core.debug import refresh_bounds_display
        refresh_bounds_display(ctx.scene)
    S.xbg_box_min = bpy.props.FloatVectorProperty(size=3, subtype='XYZ', precision=4, update=_live_bounds)
    S.xbg_box_max = bpy.props.FloatVectorProperty(size=3, subtype='XYZ', precision=4, update=_live_bounds)
    S.xbg_sphere_center = bpy.props.FloatVectorProperty(size=3, subtype='XYZ', precision=4, update=_live_bounds)
    S.xbg_sphere_radius = bpy.props.FloatProperty(default=0.0, min=0.0, precision=4, update=_live_bounds)
    # --- Far Cry 2 (independent clone — own property namespace) ---
    bpy.types.PoseBone.xbg_jiggle_fc2 = bpy.props.PointerProperty(type=XBGJiggleBoneSettingsFC2)
    S.xbg_lod_dists_fc2 = bpy.props.CollectionProperty(type=XBGLodDistItemFC2)
    S.xbg_lod_dist_path_fc2 = bpy.props.StringProperty(default='')
    S.xbg_lod_dist_endian_fc2 = bpy.props.StringProperty(default='LE')
    S.xbg_bounds_path_fc2 = bpy.props.StringProperty(default='')
    S.xbg_bounds_endian_fc2 = bpy.props.StringProperty(default='LE')
    S.xbg_has_xobb_fc2 = bpy.props.BoolProperty(default=False)
    S.xbg_has_hpsb_fc2 = bpy.props.BoolProperty(default=False)
    S.xbg_xobb_off_fc2 = bpy.props.IntProperty(default=0)
    S.xbg_hpsb_off_fc2 = bpy.props.IntProperty(default=0)
    S.xbg_box_min_fc2 = bpy.props.FloatVectorProperty(size=3, subtype='XYZ', precision=4)
    S.xbg_box_max_fc2 = bpy.props.FloatVectorProperty(size=3, subtype='XYZ', precision=4)
    S.xbg_sphere_center_fc2 = bpy.props.FloatVectorProperty(size=3, subtype='XYZ', precision=4)
    S.xbg_sphere_radius_fc2 = bpy.props.FloatProperty(default=0.0, min=0.0, precision=4)
    # Patch every operator's .report() so WARNING/ERROR popups also get
    # captured in the JSONL stream — otherwise the user dismisses the
    # popup and the only evidence of what went wrong is lost.
    VerboseLogger.install_report_capture(classes)
    # Automatic ONE-TIME update check per Blender session: deferred via a
    # timer so it never slows Blender's startup, silent on network failure
    # (offline users see no error banner). Result shows on the game-picker
    # home screen; further checks are manual via the button.
    try:
        bpy.app.timers.register(startup_update_check, first_interval=4.0)
    except Exception:
        pass


def unregister():
    S = bpy.types.Scene
    del S.xbg_settings
    del S.xbg_inject_settings
    del S.xbg_session_data
    del S.xbg_debug_settings
    del S.xbg_collision_settings
    del S.xbg_active_game
    del bpy.types.PoseBone.xbg_jiggle
    del S.xbg_lod_dists
    del S.xbg_lod_dist_path
    del S.xbg_lod_dist_endian
    del S.xbg_bounds_path
    del S.xbg_bounds_endian
    del S.xbg_has_xobb
    del S.xbg_has_hpsb
    del S.xbg_xobb_off
    del S.xbg_hpsb_off
    del S.xbg_box_min
    del S.xbg_box_max
    del S.xbg_sphere_center
    del S.xbg_sphere_radius
    # Far Cry 2
    del bpy.types.PoseBone.xbg_jiggle_fc2
    del S.xbg_lod_dists_fc2
    del S.xbg_lod_dist_path_fc2
    del S.xbg_lod_dist_endian_fc2
    del S.xbg_bounds_path_fc2
    del S.xbg_bounds_endian_fc2
    del S.xbg_has_xobb_fc2
    del S.xbg_has_hpsb_fc2
    del S.xbg_xobb_off_fc2
    del S.xbg_hpsb_off_fc2
    del S.xbg_box_min_fc2
    del S.xbg_box_max_fc2
    del S.xbg_sphere_center_fc2
    del S.xbg_sphere_radius_fc2
    for c in reversed(classes):
        bpy.utils.unregister_class(c)


if __name__ == "__main__":
    register()
