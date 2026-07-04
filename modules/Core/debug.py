import bpy,bmesh,math,mathutils
import json,time,traceback
class VerboseLogger:
    enabled=False
    _p=staticmethod(print)
    _buf=[]            # current-operation text (UI panel); wiped per op
    _session_buf=[]    # FULL-session text (every op since last Reset); the
                       # "Save Log to File" .txt is built from THIS so it
                       # contains import + edits + inject, not just the last op.
    # Structured records captured in parallel with the textual log.  Each
    # entry is a dict shaped like:
    #   {"ts": <float seconds since reset>, "tier": "INFO|DEBUG|TRACE",
    #    "section": "<section path>", "event": "<key>", "data": {...}}
    # These get exported as a .jsonl file alongside the saved text log so
    # we can grep / process them programmatically.
    _records=[]
    _t0=None
    _section_stack=[]
    @staticmethod
    def log(m):
        # Always capture a structured record (cheap) so the JSONL sibling
        # stays in sync with the textual log even when only legacy
        # `VerboseLogger.log(...)` callers (import path) are running.
        VerboseLogger._record("INFO","log",{"line":str(m)})
        if VerboseLogger.enabled:
            VerboseLogger._p(m)
            VerboseLogger._buf.append(str(m))
            VerboseLogger._session_buf.append(str(m))
    @staticmethod
    def warn(m):
        VerboseLogger._record("WARN","warn",{"line":str(m)})
        VerboseLogger._p(m)
        VerboseLogger._buf.append(str(m))
        VerboseLogger._session_buf.append(str(m))
    @staticmethod
    def clear(keep_records=True):
        """Reset the verbose log.

        keep_records=True (default): wipe the TEXT buffer (the panel preview
        and the .txt save) so each operation starts with a fresh visible log,
        but PRESERVE the structured JSONL record stream so the user can
        review the full session — import → edits → inject — in one .jsonl
        file.  The previous behaviour wiped records too, which meant a
        save-after-inject only contained inject events with no breadcrumbs
        for what the user did during edits.

        keep_records=False: full reset.  Use when starting a brand-new
        session (e.g. from a "Reset Log" button) or when the user has
        explicitly asked for a clean slate.
        """
        VerboseLogger._buf.clear()
        if not keep_records:
            VerboseLogger._records.clear()
            VerboseLogger._session_buf.clear()   # full reset clears the session
            VerboseLogger._t0=time.perf_counter()
        VerboseLogger._section_stack.clear()

    @staticmethod
    def reset_records():
        """Drop every JSONL record (and reset the relative timestamp clock).
        Use when the session is logically over and you want a clean start."""
        VerboseLogger._records.clear()
        VerboseLogger._section_stack.clear()
        VerboseLogger._t0=time.perf_counter()

    @staticmethod
    def session_marker(op_kind, **kw):
        """Emit a structured banner that groups everything until the next
        marker into one "operation" in the cumulative JSONL stream.

        Use at the top of every operator's execute() so a single save-log
        produces a navigable trail like:
            session_marker(op='import',  file='...', lod=0)
            ... import events ...
            session_marker(op='inject',  file='...', lod=0)
            ... inject events ...
        Pass any keyword args that describe the operator's settings.
        """
        # Run the env fingerprint once per Python session (first marker
        # implicitly triggers it).  Cheap; the data goes to JSONL only.
        VerboseLogger._log_env_once()
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        VerboseLogger.log("\n" + "#" * 64)
        VerboseLogger.log(f"# SESSION MARKER  op={op_kind}  @ {ts}")
        for k, v in kw.items():
            VerboseLogger.log(f"#   {k} = {v}")
        VerboseLogger.log("#" * 64)
        VerboseLogger._record("INFO", "session_marker",
                               {"op": str(op_kind), "ts_human": ts, **kw})
        VerboseLogger._op_start_ts[op_kind] = time.perf_counter()

    # Track per-op start times so session_complete() can report wall-clock
    # duration for end-of-op summary lines.
    _env_logged = False
    _op_start_ts = {}

    @staticmethod
    def _log_env_once():
        """Emit a one-shot environment fingerprint per Python session.

        Captures Blender / Python / numpy / OS / addon-script path so
        cross-machine debugging has a known baseline.  Idempotent — second
        call is a no-op."""
        if VerboseLogger._env_logged:
            return
        VerboseLogger._env_logged = True
        env = {}
        try:
            import bpy as _bpy
            env["blender_version"]   = ".".join(str(x) for x in _bpy.app.version)
            env["blender_build"]     = getattr(_bpy.app, "build_hash", "?")
        except Exception:
            env["blender_version"] = "<unknown>"
        try:
            import sys as _sys
            env["python_version"]    = _sys.version.split()[0]
            env["platform"]          = _sys.platform
        except Exception:
            pass
        try:
            import numpy as _np
            env["numpy_version"]     = _np.__version__
        except ImportError:
            env["numpy_version"]     = "<unavailable>"
        try:
            import os as _os
            env["cwd"]               = _os.getcwd()
            env["addon_module_dir"]  = _os.path.dirname(__file__)
        except Exception:
            pass
        VerboseLogger._record("INFO", "environment", env)
        VerboseLogger.log("# ENVIRONMENT")
        for k, v in env.items():
            VerboseLogger.log(f"#   {k} = {v}")

    @staticmethod
    def session_complete(op_kind, **metrics):
        """Emit an end-of-operation [summary] one-liner + structured event.

        Pair with `session_marker(op_kind, …)` at execute() top.  Pass
        whatever key metrics you want flagged: counts, deltas, output
        paths, etc.  Adds a `duration_ms` field automatically based on
        the matching session_marker.
        """
        t0 = VerboseLogger._op_start_ts.pop(op_kind, None)
        ms = None
        if t0 is not None:
            ms = round((time.perf_counter() - t0) * 1000.0, 1)
            metrics["duration_ms"] = ms

        # Compact human-readable line.  Bracketed [summary] for grep.
        parts = [f"{k}={v}" for k, v in metrics.items()]
        suffix = (f" ({ms:.1f} ms)" if ms is not None else "")
        VerboseLogger.log(f"\n[summary] {op_kind}: " + ", ".join(parts) + suffix)
        VerboseLogger._record("INFO", "session_complete",
                               {"op": str(op_kind), **metrics})

    @staticmethod
    def autosave_sidecar(target_path, *, kinds=("txt", "jsonl")):
        """DISABLED — no automatic log files, ever (per user policy).

        It used to write <target>.log.txt / .log.jsonl next to every
        imported/injected/exported asset, which littered the game/patch
        folders. Logs are now saved ONLY via the UI "Save Log to File" button
        (XBG_OT_SaveLog). This is intentionally a no-op so the existing call
        sites do nothing; do not re-enable automatic sidecar writing.
        """
        return []

    # ------------------------------------------------------------------
    # Operator self.report() capture
    # ------------------------------------------------------------------
    @staticmethod
    def install_report_capture(operator_classes):
        """Patch every Blender Operator subclass in `operator_classes` so
        its `report()` calls are mirrored into the JSONL stream.

        Why a monkey-patch and not a base class: there are 80+ self.report
        call sites across the addon and rewriting them all is invasive.
        Patching here at register time covers every existing AND every
        future call site for free.

        The patch is a method that:
            1) writes an `operator_report` structured event with the
               operator's class + bl_idname + report levels + message
            2) chains to the ORIGINAL bpy.types.Operator.report so the
               user-facing popup behaviour is unchanged.

        Idempotent — patched classes get a sentinel attribute and a
        second install_report_capture() call leaves them alone."""
        try:
            import bpy as _bpy
            original = _bpy.types.Operator.report  # bpy must be real Blender
        except (ImportError, AttributeError):
            # Stubbed bpy (e.g. running under unit tests / standalone smoke
            # tests) — nothing to patch.  Bail quietly.
            return
        for cls in operator_classes:
            try:
                if not issubclass(cls, _bpy.types.Operator):
                    continue
            except TypeError:
                continue
            if getattr(cls, "_xbg_report_patched", False):
                continue

            def _make_patched(_orig=original):
                def report(self, level_set, msg):
                    try:
                        try:
                            levels = sorted(level_set)
                        except TypeError:
                            levels = [str(level_set)]
                        VerboseLogger._record("INFO", "operator_report", {
                            "operator":  type(self).__name__,
                            "bl_idname": getattr(self, "bl_idname", ""),
                            "levels":    list(levels),
                            "msg":       str(msg),
                        })
                    except Exception:
                        pass
                    return _orig(self, level_set, msg)
                return report

            cls.report = _make_patched()
            cls._xbg_report_patched = True
    @staticmethod
    def get_log():
        return "\n".join(VerboseLogger._buf)
    @staticmethod
    def get_records():
        """Return the structured records as a list of dicts."""
        return list(VerboseLogger._records)
    @staticmethod
    def get_records_jsonl():
        """Serialise structured records as JSON-Lines (one record per line)."""
        out=[]
        for r in VerboseLogger._records:
            try:
                out.append(json.dumps(r,default=str))
            except Exception:
                out.append(json.dumps({"ts":r.get("ts",0),"tier":"WARN",
                                        "section":r.get("section",""),
                                        "event":"record_serialise_failed",
                                        "data":{"repr":repr(r)[:512]}}))
        return "\n".join(out)
    @staticmethod
    def _ts():
        if VerboseLogger._t0 is None:
            VerboseLogger._t0=time.perf_counter()
        return time.perf_counter()-VerboseLogger._t0
    @staticmethod
    def _record(tier,event,data=None):
        """Append a structured record.  Always recorded; the text log
        sibling line is governed by `enabled` (text log is gated, JSONL
        records are not — they're cheap and we keep them for the export)."""
        rec={"ts":VerboseLogger._ts(),"tier":tier,
              "section":"/".join(VerboseLogger._section_stack) or "",
              "event":event,"data":data or {}}
        VerboseLogger._records.append(rec)
        return rec

    # ---- Legacy log_* helpers (kept here so importer / parser code that
    # references VerboseLogger.log_chunk etc. keeps working) -------------
    @staticmethod
    def log_chunk(c,o,s):VerboseLogger.log(f"\n{'='*60}\nCHUNK FOUND: {c}\n  Offset: {o} (0x{o:08X})\n  Size: {s} bytes\n{'='*60}")
    @staticmethod
    def log_pmcp(sc,u):VerboseLogger.log(f"\nPMCP CHUNK DETAILS:\n  Position Scale: {sc}\n  Unknown Value: {u}\n  16-bit range: -32768 to 32767\n  World coordinate range: {-32768*sc:.3f} to {32767*sc:.3f}")
    @staticmethod
    def log_pmcu(t,s):VerboseLogger.log(f"\nPMCU CHUNK DETAILS:\n  UV Translation: {t}\n  UV Scale: {s}")
    @staticmethod
    def log_bone(i,n,p,po,r):VerboseLogger.log(f"\n  BONE {i}: {n}\n    Parent ID: {p}\n    Local Position: ({po[0]:.6f}, {po[1]:.6f}, {po[2]:.6f})\n    Local Rotation (quat x,y,z,w): ({r[0]:.6f}, {r[1]:.6f}, {r[2]:.6f}, {r[3]:.6f})")
    @staticmethod
    def log_bone_world_transform(i,n,w):VerboseLogger.log(f"    World Position: ({w[0]:.6f}, {w[1]:.6f}, {w[2]:.6f})")
    @staticmethod
    def log_mesh_header(l,v,f,s):
        si="✓ Skinning data present (stride=40)" if s==40 else f"✗ No skinning data (stride={s}, expected 40)"
        VerboseLogger.log(f"\n  MESH LOD {l}:\n    Vertex Count: {v}\n    Face Count: {f}\n    Vertex Stride: {s} bytes\n    {si}")
    @staticmethod
    def log_material(i,n,p):VerboseLogger.log(f"\n  MATERIAL {i}: {n}\n    Path: {p}")
    @staticmethod
    def log_submesh(l,i,m,b,f):VerboseLogger.log(f"\n    SUBMESH LOD{l}_{i}:\n      Material ID: {m}\n      Bones in Palette: {b}\n      Face Count: {f}")
    @staticmethod
    def log_xml_bone(n,p,r,pa):
        pi=f"\n    Parent: {pa}" if pa else ""
        VerboseLogger.log(f"\n  XML BONE: {n}\n    Position: ({p[0]:.6f}, {p[1]:.6f}, {p[2]:.6f})\n    Rotation (w,x,y,z): ({r[0]:.6f}, {r[1]:.6f}, {r[2]:.6f}, {r[3]:.6f}){pi}")
    @staticmethod
    def log_bounds(bmi,bma,sc,sr):VerboseLogger.log(f"\nBOUNDING VOLUMES:\n  Box Min: ({bmi[0]:.3f}, {bmi[1]:.3f}, {bmi[2]:.3f})\n  Box Max: ({bma[0]:.3f}, {bma[1]:.3f}, {bma[2]:.3f})\n  Sphere Center: ({sc[0]:.3f}, {sc[1]:.3f}, {sc[2]:.3f})\n  Sphere Radius: {sr:.3f}")


class TraceLogger:
    """Structured logger built on top of VerboseLogger.

    Adds tiered logging (INFO / DEBUG / TRACE), section nesting, and a
    handful of formatters for tables, hex dumps and key-value blocks.
    All output funnels back into VerboseLogger so the existing save-log
    operator and panel preview keep working.

    Tier policy:
      INFO   — always written (matches the original VerboseLogger.log)
      DEBUG  — written when VerboseLogger.enabled (the "Verbose Logging"
               toggle is what users flip today)
      TRACE  — only written when VerboseLogger.enabled AND a trace flag
               is set, to keep per-vertex dumps out of the default log

    Live-streaming:
      When `set_stream_file(path)` is called, every emitted line is
      flushed to disk immediately.  This survives Blender hard-crashes
      and lets us see the last thing that ran before the process died.
      Set up automatically when TRACE is enabled (the log file goes
      next to the user's temp dir, name "xbg_trace_live.log").
    """
    _trace_enabled=False
    _stream_file=None
    _stream_path=None

    @staticmethod
    def set_trace(on):
        TraceLogger._trace_enabled=bool(on)
        if on:
            TraceLogger._open_stream()
        else:
            TraceLogger._close_stream()

    @staticmethod
    def trace_enabled():
        return bool(TraceLogger._trace_enabled and VerboseLogger.enabled)

    @staticmethod
    def _open_stream():
        """Open a sibling 'live' log file so each line is flushed to disk.
        Crash-survival: even if Blender dies mid-export the trail remains.

        Opened in APPEND mode so multiple inject runs in one Blender
        session don't blow each other away.  A timestamped banner marks
        each new run."""
        TraceLogger._close_stream()
        try:
            import tempfile, os
            path = os.path.join(tempfile.gettempdir(), "xbg_trace_live.log")
            TraceLogger._stream_file = open(path, "a", encoding="utf-8", buffering=1)
            TraceLogger._stream_path = path
            ts = time.strftime("%Y-%m-%d %H:%M:%S")
            TraceLogger._stream_file.write(
                f"\n\n#### xbg trace live log — new run @ {ts} ####\n")
            TraceLogger._stream_file.flush()
        except Exception:
            TraceLogger._stream_file = None
            TraceLogger._stream_path = None

    @staticmethod
    def _close_stream():
        if TraceLogger._stream_file is not None:
            try:
                TraceLogger._stream_file.close()
            except Exception:
                pass
        TraceLogger._stream_file = None

    @staticmethod
    def stream_path():
        return TraceLogger._stream_path

    # ---- tier-gated text + structured emit ------------------------------
    @staticmethod
    def _emit(tier,line,event=None,data=None):
        wrote_text = False
        if tier=="INFO":
            VerboseLogger._p(line)
            VerboseLogger._buf.append(line)
            wrote_text = True
        elif tier=="DEBUG":
            if VerboseLogger.enabled:
                VerboseLogger._p(line)
                VerboseLogger._buf.append(line)
                wrote_text = True
        elif tier=="TRACE":
            if TraceLogger.trace_enabled():
                VerboseLogger._buf.append(line)  # don't spam the print stream
                wrote_text = True
        # Live-stream the text (crash-survival).  Only writes when a
        # stream is open AND the line was actually emitted for this tier.
        if wrote_text and TraceLogger._stream_file is not None:
            try:
                TraceLogger._stream_file.write(line+"\n")
                TraceLogger._stream_file.flush()
            except Exception:
                pass
        if event is not None:
            VerboseLogger._record(tier,event,data)

    @staticmethod
    def info(line,event=None,data=None):
        TraceLogger._emit("INFO",line,event,data)

    @staticmethod
    def debug(line,event=None,data=None):
        TraceLogger._emit("DEBUG",line,event,data)

    @staticmethod
    def trace(line,event=None,data=None):
        TraceLogger._emit("TRACE",line,event,data)

    # ---- section nesting (indents text lines, tags records) ------------
    @staticmethod
    def push(name):
        VerboseLogger._section_stack.append(str(name))

    @staticmethod
    def pop():
        if VerboseLogger._section_stack:
            VerboseLogger._section_stack.pop()

    class Section:
        """Context manager for nested sections.

            with TraceLogger.section('encode_vertices', verts=219):
                ...
        """
        def __init__(self,name,**kw):
            self.name=name
            self.kw=kw
            self.t0=0.0
        def __enter__(self):
            TraceLogger.push(self.name)
            self.t0=time.perf_counter()
            kvs=" ".join(f"{k}={v}" for k,v in self.kw.items())
            TraceLogger.debug(f"[trace] >>> {self.name}  {kvs}".rstrip(),
                              event="section_enter",
                              data={"name":self.name,**self.kw})
            return self
        def __exit__(self,ext,exv,etb):
            dt=time.perf_counter()-self.t0
            if exv is None:
                TraceLogger.debug(f"[trace] <<< {self.name}  ({dt*1000:.2f} ms)",
                                  event="section_exit",
                                  data={"name":self.name,"ms":round(dt*1000,3)})
            else:
                # Log the exception WITH section context before re-raising;
                # this is what gives us bread-crumbs on a Blender crash.
                tb_short=" | ".join(
                    f"{l.strip()}" for l in traceback.format_exception_only(ext,exv))
                TraceLogger.info(f"[trace] !!! {self.name}  raised {tb_short}",
                                 event="section_raised",
                                 data={"name":self.name,
                                        "exc_type":getattr(ext,"__name__","?"),
                                        "exc_msg":str(exv)[:512],
                                        "ms":round(dt*1000,3)})
            TraceLogger.pop()
            return False  # re-raise

    @staticmethod
    def section(name,**kw):
        return TraceLogger.Section(name,**kw)

    # ---- formatters -----------------------------------------------------
    @staticmethod
    def kv(label,value,tier="DEBUG",event=None):
        """One aligned key/value line."""
        line=f"    {label:<26} {value}"
        TraceLogger._emit(tier,line,event=event,
                          data={"label":label,"value":value})

    @staticmethod
    def kvblock(title,pairs,tier="DEBUG",event=None):
        """A block of aligned key/value lines under a labelled header."""
        TraceLogger._emit(tier,f"  --- {title} ---")
        rec={}
        for k,v in pairs:
            line=f"    {k:<26} {v}"
            TraceLogger._emit(tier,line)
            rec[k]=v
        if event is not None:
            VerboseLogger._record(tier,event,{"title":title,**rec})

    @staticmethod
    def table(title,headers,rows,tier="DEBUG",event=None,max_rows=None):
        """Render rows of equal-length tuples as a fixed-width table."""
        rows=list(rows)
        truncated=False
        if max_rows is not None and len(rows)>max_rows:
            rows=rows[:max_rows]+[("…",)*len(headers)]
            truncated=True
        cols=[[str(h)] for h in headers]
        for r in rows:
            for i,c in enumerate(r):
                if i<len(cols):
                    cols[i].append(str(c))
        widths=[max(len(c) for c in col) for col in cols]
        sep="  ".join("-"*w for w in widths)
        TraceLogger._emit(tier,f"  --- {title} ({len(rows)} rows"
                                +(" — truncated" if truncated else "")+") ---")
        TraceLogger._emit(tier,"  "+"  ".join(h.ljust(w)
                                              for h,w in zip(headers,widths)))
        TraceLogger._emit(tier,"  "+sep)
        for r in rows:
            TraceLogger._emit(tier,"  "+"  ".join(str(c).ljust(w)
                                                  for c,w in zip(r,widths)))
        if event is not None:
            try:
                VerboseLogger._record(tier,event,
                    {"title":title,"headers":list(headers),
                     "rows":[[str(c) for c in r] for r in rows]})
            except Exception:
                pass

    @staticmethod
    def hexdump(label,blob,offset=0,max_bytes=128,tier="TRACE",event=None):
        """Hex + ASCII dump of `blob`, prefixed with `label`."""
        if not TraceLogger.trace_enabled() and tier=="TRACE":
            # Skip the formatting work when TRACE is gated off.
            return
        if isinstance(blob,memoryview):
            blob=bytes(blob)
        truncated=len(blob)>max_bytes
        view=blob[:max_bytes]
        TraceLogger._emit(tier,f"  --- hex: {label}  ({len(blob)} bytes"
                                +(", truncated" if truncated else "")+") ---")
        for i in range(0,len(view),16):
            chunk=view[i:i+16]
            hex_s=" ".join(f"{b:02X}" for b in chunk)
            asc=  "".join(chr(b) if 32<=b<127 else "." for b in chunk)
            TraceLogger._emit(tier,f"  +{offset+i:08X}  {hex_s:<48}  {asc}")
        if event is not None:
            VerboseLogger._record(tier,event,
                {"label":label,"offset":offset,"len":len(blob),
                 "hex":view.hex()})

    @staticmethod
    def struct(event,data,tier="DEBUG"):
        """Emit a structured record only (no text line)."""
        VerboseLogger._record(tier,event,data)
def auto_smooth_normals(objs):
    VerboseLogger.log("\n=== AUTO SMOOTH NORMALS ===")
    for obj in objs:
        if obj.type!='MESH':continue
        for poly in obj.data.polygons:poly.use_smooth=True
        VerboseLogger.log(f"  ✓ Applied smooth shading to: {obj.name}")
def merge_duplicate_vertices(objs,dist):
    VerboseLogger.log(f"\n=== MERGE DUPLICATE VERTICES ===\nMerge Distance: {dist:.6f}")
    tb=ta=0
    for obj in objs:
        if obj.type!='MESH':continue
        me=obj.data;vb=len(me.vertices);tb+=vb
        bm=bmesh.new();bm.from_mesh(me);bmesh.ops.remove_doubles(bm,verts=bm.verts,dist=dist)
        bm.to_mesh(me);bm.free();me.update()
        va=len(me.vertices);ta+=va
        if va<vb:VerboseLogger.log(f"  {obj.name}: {vb} → {va} (-{vb-va})")
    VerboseLogger.log(f"\nTotal vertices removed: {tb-ta}")
def flip_normals(objs):
    VerboseLogger.log("\n=== FLIP NORMALS ===")
    for obj in objs:
        if obj.type!='MESH':continue
        me=obj.data;bm=bmesh.new();bm.from_mesh(me);bmesh.ops.reverse_faces(bm,faces=bm.faces[:])
        bm.to_mesh(me);bm.free();me.update()
        VerboseLogger.log(f"  ✓ Flipped normals: {obj.name}")
def create_format_bounds_lattice(ctx,ps,name="XBG_Format_Bounds"):
    mc=-32768*ps;Mc=32767*ps;cx=cy=cz=(mc+Mc)/2;dim=Mc-mc
    ld=bpy.data.lattices.new(name);ld.points_u=ld.points_v=ld.points_w=2
    lo=bpy.data.objects.new(name,ld);ctx.collection.objects.link(lo)
    lo.location=(cx,cy,cz);lo.scale=(dim,dim,dim);lo.show_in_front=True
    VerboseLogger.log(f"\n=== XBG FORMAT BOUNDS LATTICE ===\n16-bit range: -32768 to 32767\nPosition scale: {ps}\nWorld coordinate range: {mc:.3f} to {Mc:.3f}\nTotal dimension: {dim:.3f}\n⚠ WARNING: Vertices outside this box will be clamped during export!")
    return lo
def create_bounding_box_visualization(ctx,bbox,idx,dt):
    c=[(bbox.min[0]+bbox.max[0])/2,(bbox.min[1]+bbox.max[1])/2,(bbox.min[2]+bbox.max[2])/2]
    d=[bbox.max[0]-bbox.min[0],bbox.max[1]-bbox.min[1],bbox.max[2]-bbox.min[2]]
    # a size=1 cube / unit lattice spans +/-0.5, so the object scale must equal
    # the FULL box dimension d for it to reach min..max (was d/2 -> half size).
    if dt=='LATTICE':
        ld=bpy.data.lattices.new(f"BBoxLattice_LOD{idx}");ld.points_u=ld.points_v=ld.points_w=2
        lo=bpy.data.objects.new(f"BoundingBox_LOD{idx}",ld);ctx.collection.objects.link(lo)
        lo.location=c;lo.scale=list(d);return lo
    else:
        bpy.ops.mesh.primitive_cube_add(size=1,location=c);bo=ctx.active_object
        bo.name=f"BoundingBox_LOD{idx}";bo.scale=list(d)
        if dt=='WIRE':bo.display_type='WIRE'
        elif dt=='SOLID':
            mat=bpy.data.materials.new(name=f"BBox_Mat_LOD{idx}");mat.use_nodes=True
            bsdf=mat.node_tree.nodes.get('Principled BSDF')
            if bsdf:bsdf.inputs['Alpha'].default_value=0.3;bsdf.inputs['Base Color'].default_value=(0,1,0,1)
            mat.blend_method='BLEND';bo.data.materials.append(mat)
        return bo
def create_bounding_sphere_visualization(ctx,sphere,idx,dt):
    bpy.ops.mesh.primitive_uv_sphere_add(radius=sphere.radius,location=sphere.center,segments=32,ring_count=16)
    so=ctx.active_object;so.name=f"BoundingSphere_LOD{idx}"
    if dt=='WIRE' or dt=='LATTICE':so.display_type='WIRE'
    elif dt=='SOLID':
        mat=bpy.data.materials.new(name=f"BSphere_Mat_LOD{idx}");mat.use_nodes=True
        bsdf=mat.node_tree.nodes.get('Principled BSDF')
        if bsdf:bsdf.inputs['Alpha'].default_value=0.3;bsdf.inputs['Base Color'].default_value=(1,0,0,1)
        mat.blend_method='BLEND';so.data.materials.append(mat)
    return so
def create_bounding_visualizations(ctx,data,objs,sb,ss,dt):
    if sb and data.bounding_boxes:
        VerboseLogger.log("\n=== CREATING BOUNDING BOX VISUALIZATIONS ===")
        for i,bbox in enumerate(data.bounding_boxes):create_bounding_box_visualization(ctx,bbox,i,dt)
    if ss and data.bounding_spheres:
        VerboseLogger.log("\n=== CREATING BOUNDING SPHERE VISUALIZATIONS ===")
        for i,sphere in enumerate(data.bounding_spheres):create_bounding_sphere_visualization(ctx,sphere,i,dt)


# ---------------------------------------------------------------------------
# LIVE bounds display, driven by the editable scene props (single source of
# truth): scene.xbg_box_min/max + xbg_sphere_center/radius, gated by
# ds.show_bounding_box / show_bounding_sphere and styled by
# ds.bounds_display_type. Built via bpy.data (NOT bpy.ops) so it is safe to
# call from property `update=` callbacks — editing a value or toggling a
# checkbox re-runs this and the viewport box/sphere update instantly.
# Two well-known objects: XBG_Bounds_Box, XBG_Bounds_Sphere.
# ---------------------------------------------------------------------------

_BOUNDS_BOX_OBJ = "XBG_Bounds_Box"
_BOUNDS_SPHERE_OBJ = "XBG_Bounds_Sphere"


def _bounds_unit_mesh(name, kind):
    """A cached unit cube / unit sphere mesh (size 1, so object scale ==
    dimensions). Reused across rebuilds."""
    import bpy
    me = bpy.data.meshes.get(name)
    if me is not None:
        return me
    import bmesh
    me = bpy.data.meshes.new(name)
    bm = bmesh.new()
    if kind == 'CUBE':
        bmesh.ops.create_cube(bm, size=1.0)
    else:
        bmesh.ops.create_uvsphere(bm, u_segments=24, v_segments=12, radius=1.0)
    bm.to_mesh(me)
    bm.free()
    return me


def _bounds_get_obj(scene, name, kind):
    import bpy
    obj = bpy.data.objects.get(name)
    if obj is None:
        obj = bpy.data.objects.new(name, _bounds_unit_mesh(name + "_mesh", kind))
        obj.hide_select = True                     # a gizmo, not editable geometry
    if name not in scene.collection.objects:
        try:
            scene.collection.objects.link(obj)
        except RuntimeError:
            pass                                    # already linked somewhere
    return obj


def _bounds_remove(name):
    import bpy
    obj = bpy.data.objects.get(name)
    if obj is not None:
        bpy.data.objects.remove(obj, do_unlink=True)


def _bounds_style(obj, dt):
    if dt == 'SOLID':
        obj.display_type = 'SOLID'
    else:                                           # WIRE (and LATTICE -> wire)
        obj.display_type = 'WIRE'
    obj.show_in_front = (dt != 'SOLID')


def _bounds_display_frame(scene):
    """The transform that maps XOBB/HPSB's FILE-space coordinates onto what
    the user actually sees in the viewport (2026-06-30 fix).

    `scene.xbg_bounds_frame_obj` names the mesh object the bounds were last
    read/fitted from. Its `matrix_world` is the right frame to reuse — NOT
    the parent armature's: for models whose root node carries its own
    offset (avatar_m_body etc.), the extra translation lift
    (`xbg_root_xform`) is baked into the MESH's `matrix_local`, not the
    armature's `matrix_world` (see blender_pipeline_avatar.create_meshes).
    A mesh's own matrix_world always captures BOTH the display rotation and
    any root-offset translation, whether the mesh is armature-parented or
    stands alone (import_mesh_only). Falls back to Identity (raw file-space
    display) when there's no live reference — e.g. after "Read Bounds" from
    an arbitrary file with nothing imported.
    """
    import bpy, mathutils
    name = getattr(scene, 'xbg_bounds_frame_obj', '') or ''
    obj = bpy.data.objects.get(name) if name else None
    return obj.matrix_world.copy() if obj else mathutils.Matrix.Identity(4)


def refresh_bounds_display(scene):
    """(Re)build the live XOBB box + HPSB sphere from the editable scene
    props. Safe to call from update callbacks. Removes an object when its
    show flag is off or its chunk is absent.

    The box/sphere are placed through `_bounds_display_frame` so they land
    exactly where the referenced mesh actually is in the viewport, not at
    the raw file-space coordinates (2026-06-30: "Fit to Selected" looked
    like it "just fit it to the world origin" for models whose display
    frame carries rotation/translation the gizmo never accounted for).
    """
    if scene is None:
        return
    ds = getattr(scene, 'xbg_debug_settings', None)
    if ds is None:
        return
    dt = ds.bounds_display_type
    frame = _bounds_display_frame(scene)
    frame_rot = frame.to_quaternion()

    # ── XOBB box ────────────────────────────────────────────────────────
    if ds.show_bounding_box and getattr(scene, 'xbg_has_xobb', False):
        mn = scene.xbg_box_min
        mx = scene.xbg_box_max
        import mathutils
        center_local = mathutils.Vector(((mn[0] + mx[0]) / 2.0, (mn[1] + mx[1]) / 2.0, (mn[2] + mx[2]) / 2.0))
        obj = _bounds_get_obj(scene, _BOUNDS_BOX_OBJ, 'CUBE')
        obj.rotation_mode = 'QUATERNION'
        obj.rotation_quaternion = frame_rot
        obj.location = frame @ center_local
        obj.scale = (mx[0] - mn[0], mx[1] - mn[1], mx[2] - mn[2])
        _bounds_style(obj, dt)
    else:
        _bounds_remove(_BOUNDS_BOX_OBJ)

    # ── HPSB sphere ─────────────────────────────────────────────────────
    if ds.show_bounding_sphere and getattr(scene, 'xbg_has_hpsb', False):
        import mathutils
        c = scene.xbg_sphere_center
        r = float(scene.xbg_sphere_radius)
        obj = _bounds_get_obj(scene, _BOUNDS_SPHERE_OBJ, 'SPHERE')
        obj.location = frame @ mathutils.Vector((c[0], c[1], c[2]))
        obj.scale = (r, r, r)                        # radius is rotation-invariant
        _bounds_style(obj, dt)
    else:
        _bounds_remove(_BOUNDS_SPHERE_OBJ)
def display_file_info(chunks,fn,filepath=""):
    import struct
    # Inlined from the former Shared/binary (debug now lives in Core, framework-
    # level, and must not depend on any game's duplicated binary helper).
    LE, BE = '<', '>'
    def detect_endian_from_bytes(head):
        if len(head) < 32:
            return LE
        le_cc = struct.unpack_from('<I', head, 28)[0]
        be_cc = struct.unpack_from('>I', head, 28)[0]
        le_ok = 1 <= le_cc < 256
        be_ok = 1 <= be_cc < 256
        if le_ok and not be_ok:
            return LE
        if be_ok and not le_ok:
            return BE
        return LE

    # Read the actual file data first — we need its header bytes to detect
    # the byte order before deciding how to unpack any int/float values.
    file_data = None
    file_size = 0
    if filepath:
        try:
            with open(filepath, 'rb') as f:
                file_data = f.read()
                file_size = len(file_data)
        except:
            pass

    # PC files are little-endian, PS3 files are big-endian.  Pick once here
    # so every read_int / read_float below uses the correct format.
    en = detect_endian_from_bytes(file_data[:32]) if file_data and len(file_data) >= 32 else LE

    # Helper functions to read binary data (endian-aware)
    def read_int(data, pos):
        try:
            return struct.unpack(f'{en}I', data[pos:pos+4])[0]
        except:
            return 0

    def read_float(data, pos):
        try:
            return round(struct.unpack(f'{en}f', data[pos:pos+4])[0], 6)
        except:
            return 0.0

    def read_vector3(data, pos):
        try:
            x = read_float(data, pos)
            y = read_float(data, pos + 4)
            z = read_float(data, pos + 8)
            return (x, y, z)
        except:
            return (0.0, 0.0, 0.0)
    
    # Chunk name mapping: Original -> (Display Name, Description)
    # Using full descriptive names since we have space on one line
    ci={
        "HSEM": ("Header", "File Header"),
        "PMCP": ("Vertex Scale", "Position Scaling"),
        "PMCU": ("UV Scale", "UV Coordinate Scaling"),
        "HPSB": ("Bounding Sphere", "Spherical Bounds"),
        "XOBB": ("Bounding Box", "Box Bounds"),
        "SDOL": ("LOD System", "Level of Detail"),
        "DOL": ("Mesh Data", "Loaded Geometry"),
        "EDON": ("Skeleton", "Bone Hierarchy"),
        "LTMR": ("Materials", "Material List"),
        "DNKS": ("Skinning", "Vertex Weights"),
        "DIKS": ("Bone Index", "Bone Metadata"),
        "MB2O": ("Mesh Object", "Mesh Definition"),
        "SULC": ("Clusters", "Mesh Clusters")
    }
    
    # Build comprehensive info string
    lines = []
    lines.append(f"File: {fn}")
    lines.append(f"Chunks: {len(chunks)}")
    lines.append("")
    
    # Parse each chunk and display inline with details on new lines
    for chunk_name, offset, size in chunks:
        rn, desc = ci.get(chunk_name, (chunk_name[::-1], "?"))
        
        # Display chunk header
        lines.append(f"{chunk_name}->{rn}:")
        
        # Read actual values for specific chunks
        if file_data and offset + 40 < len(file_data):
            if chunk_name == "LTMR":
                # LTMR: +20=material count (3rd int in array of 4)
                mat_count = read_int(file_data, offset + 20)
                lines.append(f"  Mats={mat_count}")
                    
            elif chunk_name == "EDON":
                # EDON: +20=bone count (3rd int in array)
                bone_count = read_int(file_data, offset + 20)
                lines.append(f"  Bones={bone_count}")
                
            elif chunk_name == "MB2O":
                # Just show the reversed name
                pass
                
            elif chunk_name == "DIKS":
                # DIKS: +20=LOD count
                lod_count = read_int(file_data, offset + 20)
                lines.append(f"  LODs={lod_count}")
                
            elif chunk_name == "DNKS":
                # Just show the reversed name
                pass
                    
            elif chunk_name == "SDOL":
                # SDOL Structure based on Structure.py:
                # After 'SDOL' signature (4 bytes)
                # +4: chunk_int1 (4 bytes)
                # +8: chunk_int2 (4 bytes)
                # +12: skip 2 ints (8 bytes)
                # +20: LOD count (4 bytes)
                # Then for each LOD:
                #   - Read 6 ints (LOD header)
                #   - [0] = distance (as float)
                #   - [1] = face_count
                #   - [3] = vert_stride
                #   - [4] = vert_count
                
                # Read LOD count at offset +20
                lod_count = read_int(file_data, offset + 20)
                lines.append(f"  LODs={lod_count}")
                
                # Parse each LOD distance
                # Start reading LOD data after: signature(4) + 2 ints(8) + 2 skip ints(8) + lod_count(4) = 24 bytes
                lod_pos = offset + 24
                
                for lod_idx in range(lod_count):
                    # Read the 6-integer LOD header
                    try:
                        # LOD header: [0]=switch distance (float),
                        # [3]=vert_stride, [4]=vert_count. read_float is
                        # endian-aware — the old int-repack reinterpret
                        # (`pack('<I')`/`unpack('<f')`) byte-swapped the
                        # distance on PS3/big-endian files.
                        distance    = read_float(file_data, lod_pos)
                        vert_stride = read_int(file_data, lod_pos + 12)
                        vert_count  = read_int(file_data, lod_pos + 16)

                        lines.append(f"  LOD{lod_idx}: Dist={distance:.1f} Verts={vert_count} Stride={vert_stride}")
                        
                        # Skip past this LOD's data to get to the next one
                        # This is complex - we'd need to parse material lists, vertex data, etc.
                        # For now, just show what we can read from the header
                        lod_pos += 24  # Skip the 6 ints we just read
                        
                        # Read material list count and skip it
                        if lod_pos + 4 < len(file_data):
                            mat_list_count = read_int(file_data, lod_pos)
                            lod_pos += 4 + (mat_list_count * 7 * 4)  # Skip material entries (7 ints each)
                            
                            # Read vertex section size and skip vertex data
                            if lod_pos + 4 < len(file_data):
                                vert_section_size = read_int(file_data, lod_pos)
                                lod_pos += 4
                                
                                # Align to 16-byte boundary
                                remainder = lod_pos % 16
                                if remainder != 0:
                                    lod_pos += 16 - remainder
                                
                                # Skip vertex data
                                lod_pos += vert_section_size
                                
                                # Read index section size and skip index data
                                if lod_pos + 4 < len(file_data):
                                    index_section_size = read_int(file_data, lod_pos)
                                    lod_pos += 4
                                    
                                    # Align to 16-byte boundary
                                    remainder = lod_pos % 16
                                    if remainder != 0:
                                        lod_pos += 16 - remainder
                                    
                                    # Skip index data (size * 2 because indices are 16-bit)
                                    lod_pos += index_section_size * 2
                    except:
                        # If we can't parse more LODs, break
                        break
                
            elif chunk_name == "XOBB":
                # XOBB: +20=min vector, +32=max vector
                min_vec = read_vector3(file_data, offset + 20)
                max_vec = read_vector3(file_data, offset + 32)
                dx = max_vec[0] - min_vec[0]
                dy = max_vec[1] - min_vec[1]
                dz = max_vec[2] - min_vec[2]
                volume = dx * dy * dz
                lines.append(f"  Vol={volume:.2f}")
                lines.append(f"  Min: ({min_vec[0]:.3f}, {min_vec[1]:.3f}, {min_vec[2]:.3f})")
                lines.append(f"  Max: ({max_vec[0]:.3f}, {max_vec[1]:.3f}, {max_vec[2]:.3f})")
                
            elif chunk_name == "HPSB":
                # HPSB: +20=center, +32=radius
                center = read_vector3(file_data, offset + 20)
                radius = read_float(file_data, offset + 32)
                volume = (4.0 / 3.0) * 3.14159 * (radius ** 3)
                lines.append(f"  R={radius:.3f}")
                lines.append(f"  Center: ({center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f})")
                lines.append(f"  Volume: {volume:.2f}")
                
            elif chunk_name == "DOL":
                # DOL = Currently loaded/imported mesh
                lines.append(f"  Imported Geometry")
                
            elif chunk_name == "PMCP":
                # PMCP: +20=unknown float, +24=vertex position scale
                chunk_int1 = read_int(file_data, offset + 4)
                chunk_int2 = read_int(file_data, offset + 8)
                skip_int1 = read_int(file_data, offset + 12)
                skip_int2 = read_int(file_data, offset + 16)
                float1 = read_float(file_data, offset + 20)
                vertex_scale = read_float(file_data, offset + 24)
                lines.append(f"  VertexScale: {vertex_scale:.9f}")
                lines.append(f"  ChunkInts: [{chunk_int1}, {chunk_int2}]")
                lines.append(f"  SkipInts: [{skip_int1}, {skip_int2}]")
                lines.append(f"  Float1(unk): {float1:.6f}")
                
            elif chunk_name == "PMCU":
                # PMCU: +20=UV translation, +24=UV scale
                chunk_int1 = read_int(file_data, offset + 4)
                chunk_int2 = read_int(file_data, offset + 8)
                skip_int1 = read_int(file_data, offset + 12)
                skip_int2 = read_int(file_data, offset + 16)
                uv_trans = read_float(file_data, offset + 20)
                uv_scale = read_float(file_data, offset + 24)
                lines.append(f"  Trans={uv_trans:.6f} Scale={uv_scale:.6f}")
                lines.append(f"  ChunkInts: [{chunk_int1}, {chunk_int2}]")
                lines.append(f"  SkipInts: [{skip_int1}, {skip_int2}]")
    
    lines.append("")
    lines.append(f"File Size: {file_size:,} bytes ({file_size/1024:.1f} KB)")
    
    return "\n".join(lines)