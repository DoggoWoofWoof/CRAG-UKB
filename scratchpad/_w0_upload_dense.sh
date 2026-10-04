#!/bin/bash
# usage: _w0_upload_dense.sh <ds> <profile>
set -e
cd "C:/Users/Swastik/Desktop/CRAG"
ds="$1"; prof="$2"
for kind in docs queries; do
  src="data/canonical/$ds/encodings/dense/$kind"
  [ -d "$src" ] || { echo "SKIP $kind (no dir)"; continue; }
  echo "=== upload $ds/$kind -> $prof $(date +%H:%M:%S) ==="
  MODAL_PROFILE=$prof modal volume put --force crag-data-volume "$src" "$src" 2>&1 | tail -2
done
echo "UPLOAD_DONE $ds $prof $(date +%H:%M:%S)"
