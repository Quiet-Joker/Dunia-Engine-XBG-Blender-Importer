# Dunia Engine XBG Importer for Blender

A Blender 5.0 add-on for **importing, editing and re-exporting 3D models from ten
Ubisoft games** — grown from an Avatar-only importer into a full multi-game modding
toolkit.

> Originally written for Blender 2.49b. Rewritten from the ground up for modern
> Blender. Expect bugs, things to fix, things to change and more features coming soon.

---

## Highlights

- **Ten games, one add-on** — Avatar: The Game, Far Cry 1 / 2 / 3 / 4 / 5–New Dawn /
  Primal / Instincts, Watch Dogs 1 & 2, each with its own self-contained module and a
  clean game-picker UI.
- **True editing freedom** — not just moving vertices: add and delete geometry, delete
  whole submeshes, join in foreign meshes, edit UVs and bone weights, then write it all
  back into a copy of the game file.
- **Animations** — import skeletal animations (.mab) for six games, facial animation
  (pose libraries + expression curves) for Avatar/FC2, and even **entire cinematic
  scenes** with their cameras, anchors and timeline markers.
- **Skeletons & collision** — import standalone .skeleton rigs; import HKX collision
  for five games and **export edited collision** (with MOPP rebuilding) for Avatar/FC2 —
  modify any model's collision shape.
- **Custom materials** — bake Blender materials into game-ready texture (.xbt) and
  material (.xbm) files, with DXT compression, template inheritance, glass and glow
  templates.
- **LOD control** — import a chosen LOD or all of them, peek a file's LOD count before
  importing, and edit each mesh's LOD switch distances.
- **Format-exact round-trips** — injection preserves everything you didn't edit
  byte-for-byte (verified with byte-identical unedited re-exports), and oversized
  custom geometry automatically expands the file's bounds/precision instead of
  clamping.
- **Self-maintaining updater** — checks for updates once at Blender startup (silent,
  non-blocking) and one-click updates sync every file, including brand-new game modules
  added in future releases.

---

## Supported Games

| Game | Format | Model Import | Textures / Materials | Re-export (Inject) | Add/Delete Geometry | Animation | Skeleton File | Collision (HKX) |
|---|---|---|---|---|---|---|---|---|
| **Avatar: The Game** | .xbg | ✅ Full (LODs, skin, damage states) | ✅ auto-load + **custom material export** | ✅ | ✅ | ✅ .mab + facial + **full scenes** | ✅ import & export | ✅ import & **export** (MOPP) |
| **Far Cry 2** | .xbg | ✅ Full | ✅ auto-load + **custom material export** | ✅ | ✅ | ✅ .mab + facial + scenes | ✅ import & export | ✅ import & **export** |
| **Far Cry 3** | .xbg | ✅ Full | ⚠️ slot names only (no texture loading yet) | ✅ | ✅ | ✅ .mab | ✅ import | ✅ import |
| **Far Cry 4** | .xbg | ✅ Full | ⚠️ slot names only (no texture loading yet) | ✅ | ✅ | ✅ .mab | — (rig from model) | ✅ import |
| **Far Cry 5 / New Dawn** | .xbg | ✅ (8-influence skinning) | ⚠️ slot names only (no texture loading yet) | ✅ same-count | ❌ Not yet | ✅ .mab + root motion + prop rigs | — (rig from model) | — |
| **Far Cry Primal** | .xbg | ✅ Full | ⚠️ slot names only (no texture loading yet) | ✅ | ✅ | 🔜 coming soon | — (rig from model) | — |
| **Far Cry 1** | .cgf | ✅ (per-face materials) | ✅ .dds auto-load | — | — | — | — | — |
| **Far Cry Instincts** (Xbox) | .xbg | ✅ | ✅ .xbt auto-decode | — | — | — | — | — |
| **Watch Dogs 1** | .xbg | ✅ Full (+ streamed hi-detail LODs) | ⚠️ slot names only | ✅ | ✅ | ✅ .mab | ✅ import | ✅ import |
| **Watch Dogs 2** | .glm | ✅ Full | ⚠️ slot names | ✅ **.glm export** | ✅ | — | — (rig from model) | — |
| Far Cry 6 | — | 🔜 coming soon | | | | | | |

**Legend / footnotes**

- *Re-export (Inject)* always writes to a **new copy** of the game file — your
  originals are never touched.
- *Add/Delete Geometry* = full rebuild support: change vertex/triangle counts, delete
  submeshes, re-skin new geometry from vertex groups. Games without it support
  reshape/sculpt/UV/color edits at the original vertex count.
- WD1 vehicles with split LOD buffers and streamed-LOD meshes patch in place (no count
  changes); WD1 also auto-updates the companion `.high.xbgmip` streamed-LOD file so
  edits don't "revert" at close range.
- WD2 export preserves materials/skeleton/physics blocks byte-for-byte and hands you a
  .glm ready for a GLM2XBG converter.

---

## Requirements

- **Blender 5.0** or newer
- Game files for the game you want to mod (extracted where the game ships archives —
  e.g. Avatar's `Data` folder, FC Instincts `.fat/.dat` dumps, Far Cry 1's `FCData`)

---

## Installation

1. Download the latest release `.zip` from the [Releases](../../releases) page.
2. In Blender: **Edit → Preferences → Add-ons → Install**, pick the zip, enable
   **XBG Importer**.
3. In the add-on preferences, set the data-folder path(s) for your game(s) — this is
   what powers automatic texture loading. 
   > Temporarily only supported on Avatar.

> **Upgrading from v2.x?** Do a fresh install from the zip (remove the old add-on
> first). The old in-app updater can't fetch the new game modules. From v3.0.0 onward,
> updates are fully automatic.

### Updates

The add-on checks GitHub for a new version **once automatically at Blender startup**
(quietly — nothing appears unless there *is* an update, and nothing blocks startup).
When one is available, the home screen shows an **Update Now** button: one click
downloads the release and syncs every file — new games and scripts included — then you
restart Blender.

---

## Using the Add-on

Open the **N-panel** (`N` in the 3D Viewport) → **XBG Import** tab → **pick your
game**. Every game uses the same layout:

| Panel | Visible | What it does |
|---|---|---|
| **Import** | always | Data-folder setting, texture options, LOD peek, the big import button |
| **Advanced Mode** | always (toggle) | Reveals everything below |
| **Inject / Export** | advanced | Status of the linked source file, bounds check, the big inject button |
| **Animation** | advanced | .mab import with resampling/helper options (games with animation) |
| **Skeleton / HKX** | per game | Standalone skeleton import, collision import/export |
| **Editors** | advanced | LOD distances, bounding volumes, jiggle bones, materials (Avatar/FC2) |
| **Model Info / Debug** | advanced | What the importer captured; verbose logging controls |

### Typical workflow: edit a model and put it back in the game

1. **Import** the model with **Separate Primitives ON** (Advanced Mode → import
   options). This keeps one Blender object per game submesh — required for writing
   back. (Imported with it off? The object is flagged as *joined* and the inject panel
   will tell you to re-import.)
2. **Edit** in Blender: sculpt, move vertices, restructure UVs, repaint vertex colors,
   assign weights. On games with full rebuild support you can extrude/delete geometry,
   delete entire submeshes, or `Ctrl+J`-join a completely different mesh into an
   imported object (make the imported object active so its metadata survives).
3. Select the edited objects and press the game's **Inject** button. Pick the output
   path (pre-filled next to the source) — a patched copy is written, ready to pack back
   into the game.

### Typical workflow: view an animation

1. Import the model (the armature is created from the file).
2. Advanced Mode → **Animation** → pick the `.mab`. Bones are matched automatically
   (by name hash or skeleton file, depending on the game); options cover smooth
   resampling, helper-bone emulation and twist baking.
3. For Avatar/FC2 cinematics, use the **Scene Viewer** to bring in the whole scripted
   scene — cameras, anchors and timeline markers included.

### Typical workflow: custom textures/materials (Avatar, FC2)

1. Set up your material in Blender on the imported mesh.
2. Advanced Mode → **Export Custom Materials** — choose a template (standard, glass,
   glow…), and the add-on bakes game-ready `.xbt` textures + `.xbm` material files into
   your patch folder.

---

## Repository Layout

```
__init__.py        # add-on entry point (registration, version, update check)
modules/
  Core/            # preferences, updater, logging, shared settings
  UI/              # game picker + one panel file per game
  Avatar/          # per-game format modules — fully self-contained per game
  Far_Cry_1/  Far_Cry_2/  Far_Cry_3/  Far_Cry_4/  Far_Cry_5/
  Far_Cry_Primal/  Far_Cry_Instincts/  Far_Cry_6/ (placeholder)
  Watch_Dogs/  Watch_Dogs_2/
```

Every game's code is deliberately isolated — no cross-game imports — so a fix or
experiment in one game can never break another.

---

## Branches

| Branch | Description |
|---|---|
| `main` / `public` | Stable release |
| `Dev` | Latest work-in-progress with recent fixes and features |

---

## Credits

**Author:** Quiet Joker

**Special thanks:** Jasper_Zebra

**Original script:** Szkaradek123 for the Avatar modding community (Blender 2.49b era). 

Rewritten and expanded for Blender 5.0

## Want to help?

If you want to help or support the project please leave any bug reports, feature suggestions or if you want to help improve the code, please don't be afraid to leave some pull requests. This is my gift to the entire Avatar/Far Cry/Watch Dogs community.
