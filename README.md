# Avatar XBG Importer for Blender

A Blender 5.0 add-on for importing (and re-exporting) XBG 3D models from **James Cameron's Avatar: The Game**.

> Originally written for Blender 2.49b. This is a ground-up rewrite for modern Blender, built with AI assistance over the course of roughly a year.

---

## Features

- Import XBG models with skeleton, skinning, and materials
- Multi-LOD support — import a specific LOD or all at once
- **LOD Peek** — scan a file to see its LOD count before importing
- Re-export / inject modified mesh data back into the original XBG file
- Automatic texture loading from XBM material files (requires game Data folder)
- HD texture support (`_mip0` variants)
- Bounding box and bounding sphere visualization
- XML assembly support for weapon/part files
- MB2O skeleton mode for files with bind matrix data
- Vertex compaction — removes unused vertices for cleaner editing while preserving correct export positions
- Export normals — recalculated from Blender geometry and re-encoded into the file

---

## Requirements

- **Blender 5.0** or newer
- Game files from *James Cameron's Avatar: The Game* (PC version)

---

## Installation

1. Download the latest release `.zip` from the [Releases](../../releases) page.
2. In Blender, go to **Edit → Preferences → Add-ons → Install**.
3. Select the downloaded `.zip` and click **Install Add-on**.
4. Enable the **XBG Importer** add-on in the list.
5. In the add-on preferences, set your extracted **Data Folder** path to the game's `Data` directory (required for texture loading).

---

## Usage

### Importing

1. Open the **N-panel** (press `N` in the 3D Viewport) and go to the **XBG Import** tab.
2. Optionally use **Peek LOD Count** to check how many LODs a file has before importing.
3. Click **Import XBG** and select one or more `.xbg` files.
4. In the file browser sidebar, choose your LOD level (or enable **Import All LODs**).

### Re-Exporting (Inject)

After editing an imported mesh:

1. Select the mesh object in the viewport.
2. In the **XBG Import** panel under **Export (Re-Inject)**, click **Inject Mesh Data**.
3. The file browser will pre-fill with the original file path. Confirm to write.

> ⚠️ Vertex count must not change between import and export. Only vertex positions, UVs, and normals are written back. No vertex deletion or adding supported.

---

## Panel Overview

| Section | Description |
|---|---|
| Game Data Folder | Path to the game's `Data` directory for texture loading |
| Import Options | Texture loading toggles |
| LOD Preview | Peek LOD count button and result display |
| Export (Re-Inject) | Bounds check, scale info, PMCP override, inject button |
| Damage States | Toggle between normal / damaged mesh visibility (shown when applicable) |
| XBG Debug (closed by default) | Logging, mesh processing options, skeleton mode, vertex compaction, bounding volumes |

---

## Known Issues / To-Do

- [ ] Occasional duplicate texture entries when multiple materials reference the same file.

---

## Compatibility

Confirmed working on:
- *James Cameron's Avatar: The Game* (PC)
- *Far Cry 3* (XBG format shares structural similarities)
- *Far Cry 2*

---

## Menu Preview

<img width="245" height="1354" alt="image" src="https://github.com/user-attachments/assets/ac9c6e84-9bd4-4349-b315-1722d0ddcdeb" />
<img width="1541" height="1193" alt="image" src="https://github.com/user-attachments/assets/849e5eb0-53f4-48fc-b622-3aeea6acb86b" />

---

## Branches

| Branch | Description |
|---|---|
| `main` / `public` | Stable release |
| `dev` | Latest work-in-progress with recent fixes and features |

---

## Credits

**Author:** Quiet Joker
**Special thanks:** Jasper_Zebra
**Original script:** Szkaradek123 for the Avatar modding community (Blender 2.49b era).  
Rewritten for Blender 5.0 with AI assistance.
