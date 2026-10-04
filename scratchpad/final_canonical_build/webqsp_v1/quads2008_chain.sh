#!/bin/sh
# wait on the TSV join's OWN terminal output, not a process table: wmic/pkill are unreliable here.
L=scratchpad/final_canonical_build/webqsp_v1/fb2010_tsv_join.log
while ! grep -qE '"elapsed_s"|Traceback|MemoryError' "$L" 2>/dev/null; do sleep 60; done
sleep 10
cd "C:/Users/Swastik/Desktop/CRAG"
PYTHONHASHSEED=0 python scratchpad/final_canonical_build/webqsp_v1/fb2010_quads.py 2008 \
  > scratchpad/final_canonical_build/webqsp_v1/fb2010_quads_2008.log 2>&1
echo CHAIN_DONE
