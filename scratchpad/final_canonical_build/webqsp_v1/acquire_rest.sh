#!/usr/bin/env bash
# Acquire the two remaining untapped exact Freebase sources, then put the page reader back.
#   freebase-datadump-tsv.tar.bz2   1.35 GB  the 2010-07-16 per-type TSV export
#   freebase_dump_2008-03-28 quads  0.56 GB  the earliest surviving Freebase database dump
# web.archive.org/web/ SYN-drops while a bulk archive.org download is in flight, so the archived-page
# reader is stopped for the duration and resumed after.  It resumes from its own parts, so this is free.
cd "C:/Users/Swastik/Desktop/CRAG"
B=scratchpad/final_canonical_build/webqsp_v1
powershell -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -like '*fb_page_names*' } | ForEach-Object { Stop-Process -Id \$_.ProcessId -Force }"
echo "=== page reader paused $(date) ==="
for w in tsv quads2008; do
  echo "===== fetching $w"
  PYTHONHASHSEED=0 python -u $B/fb2010_fetch.py $w 2>&1 | grep -E "published size|size on disk|md5 |^OK|MISMATCH"
done
echo "=== resuming page reader $(date) ==="
PYTHONHASHSEED=0 python -u $B/fb_page_names.py hunt 1 1.5 >> $B/fb_page_names_hunt.log 2>&1
