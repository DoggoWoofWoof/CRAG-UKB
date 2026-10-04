#!/bin/bash
# restart-on-crash wrapper for the resumable HF upload (state = data/_cache/hfbig_state.json); bounded, ends when UPLOAD exits 0 or after 30 attempts
cd /c/Users/Swastik/Desktop/CRAG
for i in $(seq 1 30); do
  python -u scratchpad/_hf_big.py UPLOAD "$1" >> work/HOST_HOUSEKEEPING/hfbig_upload.log 2>&1
  rc=$?
  echo "$(date +%T) wrapper: attempt $i exit $rc" >> work/HOST_HOUSEKEEPING/hfbig_upload.log
  [ $rc -eq 0 ] && break
  sleep 60
done
