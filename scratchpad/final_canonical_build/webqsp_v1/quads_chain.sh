#!/usr/bin/env bash
cd "C:/Users/Swastik/Desktop/CRAG"
L=scratchpad/final_canonical_build/webqsp_v1/fb2010_fetch_quads.log
until grep -qE '^OK |MISMATCH|Traceback' "$L" 2>/dev/null; do sleep 20; done
if ! grep -q '^OK ' "$L"; then echo "=== fetch did not verify; refusing to join ==="; exit 1; fi
echo "=== checksum verified; starting the 2010 quadruples join ==="
PYTHONHASHSEED=0 python -u scratchpad/final_canonical_build/webqsp_v1/fb2010_quads.py
