"""Verbose-log maintenance operators (Reset / Save Log).

Split out of the monolithic __init__.py (2026-06-09 refactor).
"""
import os

import bpy

from .debug import VerboseLogger


class XBG_OT_ResetLog(bpy.types.Operator):
    """Drop every JSONL record AND clear the visible log buffer.

    By default the addon now ACCUMULATES JSONL records across operations
    (import → edits → inject → save) so one save-log gives you the whole
    session's breadcrumbs.  Use this button when you want a clean slate
    — e.g. you finished debugging one model and are about to start a
    fresh one and don't want the old run's events polluting the next
    save."""
    bl_idname = "xbg.reset_log"
    bl_label = "Reset Session Log"
    bl_description = ("Wipe the JSONL record stream and the text log "
                       "buffer. Operations after this will start fresh.")

    def execute(self, ctx):
        n = len(VerboseLogger.get_records())
        VerboseLogger.reset_records()
        VerboseLogger.clear(keep_records=False)
        VerboseLogger.session_marker("session_reset", dropped_records=n)
        self.report({'INFO'}, f"Dropped {n} JSONL record(s) and cleared the text log")
        return {'FINISHED'}


class XBG_OT_SaveLog(bpy.types.Operator):
    """Save the current verbose log buffer to a text file."""
    bl_idname = "xbg.save_log"
    bl_label = "Save Log to File"
    bl_description = "Write the current verbose log buffer to a .txt file on disk"

    filepath: bpy.props.StringProperty(subtype="FILE_PATH", default="xbg_log.txt")
    filter_glob: bpy.props.StringProperty(default="*.txt", options={'HIDDEN'})

    def invoke(self, ctx, ev):
        ctx.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}

    def execute(self, ctx):
        ds = ctx.scene.xbg_debug_settings
        file_info = ds.file_info_data if hasattr(ds, 'file_info_data') else ""
        verbose_log = VerboseLogger.get_log()

        parts = []
        # Header explains what each sidecar contains so the user doesn't
        # confuse the .txt with the .jsonl (they look alike at a glance
        # because the JSONL embeds raw log lines as `event=log` records,
        # but the JSONL ALSO has structured events — tables, kvblocks,
        # snapshots — that the human .txt only renders as formatted text).
        parts.append("=== READ ME ===")
        parts.append("This .txt is the HUMAN-READABLE narrative.")
        parts.append("The sibling .jsonl is the STRUCTURED record stream.")
        parts.append("  - jq filterable: `event=='inject_entry'`, `event=='user_edit_snapshot'`, …")
        parts.append("  - has typed payloads (counts, slot maps, vertex samples)")
        parts.append("  - accumulates across operations (clear with 'Reset Session' button)")
        parts.append("")
        if file_info:
            parts.append("=== FILE INFO ===")
            parts.append(file_info)
        if verbose_log:
            parts.append("=== VERBOSE LOG ===")
            parts.append(verbose_log)

        log_text = "\n".join(parts)
        if not log_text:
            self.report({'WARNING'}, "Log is empty — import an XBG file first")
            return {'CANCELLED'}
        # Force .txt extension if the user typed a bare name or picked an
        # extension that doesn't make sense.  Keeps the .txt/.jsonl pair
        # name-consistent (the JSONL sibling code below assumes .txt -> .jsonl).
        out_path = self.filepath
        if not out_path.lower().endswith('.txt'):
            out_path = out_path + '.txt'
        try:
            with open(out_path, 'w', encoding='utf-8') as f:
                f.write(log_text)
            msg = f"Log saved: {out_path}"
            self.filepath = out_path  # so the JSONL sibling block below sees it
        except Exception as exc:
            self.report({'ERROR'}, f"Failed to save log: {exc}")
            return {'CANCELLED'}

        # Always also try to drop a structured .jsonl sibling next to the
        # text log.  This is what makes TRACE-level data programmatically
        # consumable — one record per line, parseable with `jq` or any
        # JSONL-aware tool.  Empty (no records) → still write the file
        # but with a "no_records" marker so users know it's expected.
        try:
            jsonl_path = (self.filepath[:-4] + ".jsonl"
                          if self.filepath.lower().endswith(".txt")
                          else self.filepath + ".jsonl")
            recs = VerboseLogger.get_records()
            jsonl = VerboseLogger.get_records_jsonl()
            if not jsonl:
                import json as _json
                jsonl = _json.dumps({"event": "no_records",
                                      "note": "Run an XBG operation (import/inject) "
                                              "with Verbose Logging ON to populate records."})
            with open(jsonl_path, 'w', encoding='utf-8') as f:
                f.write(jsonl)

            # ── Event-count breakdown: at-a-glance confirmation that the
            # pipeline you expected actually ran.  If "inject_entry" /
            # "object_inspect" / "encode_setup" are present, inject fired.
            # If you see only "log" + "import_entry" / "import_chunk", you
            # only imported.
            from collections import Counter
            evt_counts = Counter(r.get("event", "?") for r in recs)
            top = evt_counts.most_common(12)
            pipeline_markers = {
                "inject_entry":     "INJECT ran",
                "encode_setup":     "vertex encoding fired",
                "sdol_submeshes":   "SDOL submesh table built",
                "dnks_rebuild_summary": "DNKS rebuilt",
                "ltmr_rebuild_done": "LTMR rebuilt",
                "splice_map":       "chunk splice completed",
                "postwrite_sdol":   "post-write verify ran",
                "inject_finished":  "INJECT reached return",
                "import_entry":     "IMPORT ran",
                "import_complete":  "IMPORT parse complete",
                "loader_finished":  "IMPORT reached scene",
                "autoscale_entry":  "AUTO-SCALE ran",
                "encoding_failed":  "ENCODE CRASHED",
                "ltmr_parse_failed": "LTMR PARSE FAILED",
                "dnks_parse_full_failed": "DNKS PARSE FAILED",
                "postwrite_idx_overflow": "**INDEX OUT OF RANGE on inject**",
                "postwrite_failed": "POST-WRITE VERIFY FAILED",
                "import_oob_index": "**OUT-OF-RANGE INDEX in injected file** (import detected)",
                "import_stage_failed": "IMPORT STAGE FAILED",
                "import_operator_failed": "IMPORT OPERATOR FAILED",
                "import_chunk_header_failed": "IMPORT CHUNK HEADER FAILED",
            }
            try:
                with open(jsonl_path + ".summary.txt", 'w',
                          encoding='utf-8') as f:
                    f.write(f"# Record summary: {len(recs)} total records\n")
                    f.write(f"# Source jsonl: {jsonl_path}\n\n")
                    f.write("=== Pipeline markers ===\n")
                    for mk, label in pipeline_markers.items():
                        n = evt_counts.get(mk, 0)
                        flag = ("YES" if n else "no")
                        f.write(f"  [{flag}] {label:<32} x{n}  ({mk})\n")
                    f.write(f"\n=== Top {len(top)} event counts ===\n")
                    for ev, n in top:
                        f.write(f"  {n:>5}  {ev}\n")
                    f.write("\n=== All events seen (alphabetical) ===\n")
                    for ev in sorted(evt_counts.keys()):
                        f.write(f"  {evt_counts[ev]:>5}  {ev}\n")
            except Exception:
                pass

            # One-line summary in the status bar
            msg += f"  +  jsonl: {jsonl_path}  ({len(recs)} records, {len(evt_counts)} event types)"
        except Exception as exc:
            VerboseLogger.warn(f"[save_log] jsonl sibling NOT written: {exc}")

        self.report({'INFO'}, msg)
        return {'FINISHED'}


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------
