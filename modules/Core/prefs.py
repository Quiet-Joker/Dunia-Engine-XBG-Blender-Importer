"""Addon preferences + auto-updater.

Split out of the monolithic __init__.py (2026-06-09 refactor).
"""
import os
import re
import threading
import urllib.request

import bpy

# The addon root package name: "V12"-style folder for a legacy addon, or
# the full "bl_ext.<repo>.<name>" when installed as a Blender extension.
# split('.')[0] would break for extensions, so strip our own subpath instead.
ADDON_ID = __package__.rsplit('.modules.', 1)[0]


def get_prefs(ctx):
    """The addon preferences, regardless of which module asks."""
    return ctx.preferences.addons[ADDON_ID].preferences



# The branch people install from.  Everything the updater needs lives in the
# repo itself (__init__.py + modules/), so there is NO file manifest anymore:
# updates download the whole branch as a zip and sync it (new folders/scripts
# in a release are picked up automatically — nothing to register here).
_REPO_BRANCH = "Dev"
_RAW_BASE = ("https://raw.githubusercontent.com/Quiet-Joker/"
             "Dunia-Engine-XBG-Blender-Importer/%s/" % _REPO_BRANCH)
_ZIP_URL = ("https://github.com/Quiet-Joker/Dunia-Engine-XBG-Blender-Importer/"
            "archive/refs/heads/%s.zip" % _REPO_BRANCH)

_update_status = None   # None = not checked, "up_to_date", or "vX.X.X available"
_update_error  = None   # set if network fetch failed
_startup_checked = False  # auto-check runs once per Blender session


def _fetch_remote_version():
    """Fetch remote __init__.py and return version tuple, or None on failure."""
    try:
        url = _RAW_BASE + "__init__.py"
        req = urllib.request.urlopen(url, timeout=8)
        text = req.read(4096).decode("utf-8", errors="ignore")
        m = re.search(r'"version"\s*:\s*\((\d+),\s*(\d+),\s*(\d+)\)', text)
        if m:
            return (int(m.group(1)), int(m.group(2)), int(m.group(3)))
    except Exception:
        pass
    return None


def _check_update_thread(silent=False):
    global _update_status, _update_error
    remote = _fetch_remote_version()
    if remote is None:
        # On the automatic startup check, stay quiet about network failures
        # (offline users shouldn't see an error banner every launch).
        if not silent:
            _update_error = "Could not reach update server."
        return
    import importlib
    local = importlib.import_module(ADDON_ID).bl_info["version"]
    if remote > local:
        _update_status = f"v{remote[0]}.{remote[1]}.{remote[2]} available"
    else:
        _update_status = "up_to_date"


def startup_update_check():
    """Kick off ONE automatic, non-blocking update check per Blender session.

    Called from the addon's register() via bpy.app.timers (a few seconds
    after startup so it never delays Blender opening).  Result lands in
    _update_status and shows up on the game-picker home screen exactly like
    a manual 'Check for Updates' — users only ever re-check manually."""
    global _startup_checked
    if _startup_checked:
        return None                      # timer: don't reschedule
    _startup_checked = True
    threading.Thread(target=_check_update_thread, kwargs={'silent': True},
                     daemon=True).start()
    return None                          # timer: run once


# ---------------------------------------------------------------------------
# Addon preferences
# ---------------------------------------------------------------------------

class XBGAddonPreferences(bpy.types.AddonPreferences):
    # MUST be the addon root module name — with __name__ this class would
    # silently fail to bind (get_prefs() returns None, panel draws crash)
    bl_idname = ADDON_ID
    data_folder: bpy.props.StringProperty(
        name="Avatar / Far Cry 2 — Extracted Game Data",
        description="Path to the extracted game-data folder for Avatar: The "
                    "Game or Far Cry 2 (shared by both — point it at whichever "
                    "game's data you're working with; the original unpacked "
                    "files are read from here)",
        default="",
        subtype='DIR_PATH'
    )
    patch_folder: bpy.props.StringProperty(
        name="Avatar Game — Extracted Patch Folder",
        description="Destination patch/mod folder for the Avatar game. Your "
                    "custom files (baked materials/textures, patched skeleton + "
                    "proceduralbones.xml) are written here, mirroring their "
                    "relative path under the extracted game-data folder",
        default="",
        subtype='DIR_PATH'
    )
    fci_data_folder: bpy.props.StringProperty(
        name="Far Cry Instincts — Extracted Archive Folder",
        description="Path to a Far Cry Instincts .dat/.fat archive dump "
                    "(the output folder of fci_extract.py). Used to look up "
                    "a model's texture by its embedded in-game path",
        default="",
        subtype='DIR_PATH'
    )
    fc1_data_folder: bpy.props.StringProperty(
        name="Far Cry 1 — FCData Folder",
        description="Path to the Far Cry 1 FCData game-data folder (contains "
                    "Objects/Objects1/Objects2, Textures/Textures1/Textures2, "
                    "etc). Used to look up a model's textures by their "
                    "embedded in-game .dds path",
        default="",
        subtype='DIR_PATH'
    )

    def draw(self, ctx):
        self.layout.prop(self, "data_folder")
        self.layout.prop(self, "patch_folder")
        self.layout.prop(self, "fci_data_folder")
        self.layout.prop(self, "fc1_data_folder")


# ---------------------------------------------------------------------------
# Property groups
# ---------------------------------------------------------------------------

class XBG_OT_CheckForUpdates(bpy.types.Operator):
    """Check GitHub for plugin updates"""
    bl_idname = "xbg.check_for_updates"
    bl_label = "Check for Updates"

    def execute(self, context):
        global _update_status, _update_error
        _update_status = None
        _update_error  = None
        threading.Thread(target=_check_update_thread, daemon=True).start()
        self.report({'INFO'}, "Checking for updates...")
        return {'FINISHED'}


class XBG_OT_ApplyUpdate(bpy.types.Operator):
    """Download and install the latest version from GitHub"""
    bl_idname = "xbg.apply_update"
    bl_label = "Update Now"

    def execute(self, context):
        import io
        import zipfile

        # this file lives in modules/Core/ — the addon root is two levels up
        plugin_dir = os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))))

        # ── 1. download the WHOLE branch as one zip (no file manifest —
        #       new folders/scripts added upstream are picked up automatically)
        try:
            req = urllib.request.urlopen(_ZIP_URL, timeout=60)
            blob = req.read()
            zf = zipfile.ZipFile(io.BytesIO(blob))
        except Exception as e:
            self.report({'ERROR'}, f"Update download failed: {e}")
            return {'CANCELLED'}

        # zip root folder is "<repo>-<branch>/"
        names = zf.namelist()
        root = names[0].split('/')[0] + '/' if names else ''

        def wanted(rel):
            """Which repo files belong inside the installed addon."""
            return rel == "__init__.py" or rel.startswith("modules/")

        # ── 2. extract into the addon folder
        remote_files = set()
        written = 0
        failed = []
        for n in names:
            if not n.startswith(root) or n.endswith('/'):
                continue
            rel = n[len(root):]
            if not wanted(rel):
                continue
            remote_files.add(rel)
            dest = os.path.join(plugin_dir, rel.replace('/', os.sep))
            try:
                os.makedirs(os.path.dirname(dest), exist_ok=True)
                with open(dest, 'wb') as f:
                    f.write(zf.read(n))
                written += 1
            except Exception as e:
                failed.append(f"{rel}: {e}")

        # ── 3. remove local .py files under modules/ that no longer exist
        #       upstream (renamed/deleted scripts would otherwise linger and
        #       shadow-import).  ONLY .py files under modules/ are touched —
        #       user files, caches and anything outside modules/ are kept.
        removed = 0
        mod_dir = os.path.join(plugin_dir, "modules")
        for dirpath, dirnames, filenames in os.walk(mod_dir):
            dirnames[:] = [d for d in dirnames if d != "__pycache__"]
            for fn in filenames:
                if not fn.endswith(".py"):
                    continue
                full = os.path.join(dirpath, fn)
                rel = os.path.relpath(full, plugin_dir).replace(os.sep, '/')
                if rel not in remote_files:
                    try:
                        os.remove(full)
                        removed += 1
                        print(f"[XBG Updater] removed stale: {rel}")
                    except Exception as e:
                        failed.append(f"remove {rel}: {e}")

        global _update_status
        _update_status = None

        if failed:
            self.report({'WARNING'},
                f"Update partially failed — {len(failed)} file(s). "
                f"Check console for details. Restart Blender for partial changes.")
            for msg in failed:
                print(f"[XBG Updater] FAILED: {msg}")
        else:
            self.report({'INFO'},
                f"Update complete ({written} files updated, {removed} stale "
                f"removed). Restart Blender to apply.")
        return {'FINISHED'}


# ---------------------------------------------------------------------------
# Operators — Expand Bounds / Save Bounds
# ---------------------------------------------------------------------------

