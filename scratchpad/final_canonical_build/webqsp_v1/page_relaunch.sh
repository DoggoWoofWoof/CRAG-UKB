#!/usr/bin/env bash
# Wait for the bulk archive.org download to finish, then resume the archived-page reader.
# web.archive.org/web/ SYN-drops while a bulk archive.org download is in flight; the two jobs
# must not overlap.  fb_page_names.py resumes from its own parts + missed.json, so a restart
# costs nothing.  Waits on the fetcher's own terminal line, not on a process table.
cd "C:/Users/Swastik/Desktop/CRAG"
L=scratchpad/final_canonical_build/webqsp_v1/fb2010_fetch_quads.log
until grep -qE '^OK |MISMATCH|Traceback' "$L" 2>/dev/null; do sleep 30; done
echo "=== fetcher finished at $(date); resuming page reader ==="
PYTHONHASHSEED=0 python -u scratchpad/final_canonical_build/webqsp_v1/fb_page_names.py hunt 1 1.5 \
  >> scratchpad/final_canonical_build/webqsp_v1/fb_page_names_hunt.log 2>&1
