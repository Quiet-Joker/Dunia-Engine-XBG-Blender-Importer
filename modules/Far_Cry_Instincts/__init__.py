"""Package marker — Far Cry Instincts (Xbox, 2005) tools.

Far Cry Instincts uses an EARLIER, unrelated .xbg mesh format from Avatar /
Far Cry 2-6 (different magic, different chunk layout, no chunk-tag strings
at all — see import_xbg_fci.py for the reverse-engineered layout). Nothing
in this folder imports from Avatar/Far_Cry_2..6/Watch_Dogs; it is fully
self-contained, like the other per-game folders.

Format status (2026-07-02): position + triangle topology only. No UVs,
normals, skeleton/skinning, or export/inject yet — the on-disk layout for
those hasn't been reverse-engineered. Import is read-only.
"""
