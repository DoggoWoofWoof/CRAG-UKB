#!/bin/bash
# Wait for segmented assembly, then verify against the pre-registered digests.
cd /c/Users/Swastik/Desktop/CRAG
LOG=data/final_canonical/freebase_v3/_acquisition/raw/_v3_fetch.log
until grep -q 'assembled_bytes' "$LOG" 2>/dev/null; do sleep 30; done
echo "=== assembly done, verifying $(date -u +%FT%TZ) ==="
python scratchpad/final_canonical_build/webqsp_v1/v3_verify_raw_mirror.py
