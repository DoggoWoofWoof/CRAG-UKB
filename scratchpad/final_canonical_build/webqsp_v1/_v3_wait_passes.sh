#!/bin/bash
cd /c/Users/Swastik/Desktop/CRAG
L=data/final_canonical/freebase_v3/_acquisition/raw
while true; do
  n=$(ps -W 2>/dev/null | grep -cE '/usr/bin/(gzip|awk|grep)')
  if [ "$n" -eq 0 ]; then break; fi
  sleep 30
done
echo "=== both passes finished $(date -u +%FT%TZ) ==="
echo "--- PASS A ---";     tail -20 $L/_v3_pass_a.log
echo "--- CONTIGUITY ---"; tail -25 $L/_v3_contig_full.log
