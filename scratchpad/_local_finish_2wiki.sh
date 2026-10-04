#!/usr/bin/env bash
# Build 2wiki splade's last 11 query shards locally, then run the whole record chain.
#
# The 13 shards already on disk came from Modal before that workspace was disabled; every
# remaining workspace in the pool has exceeded its spend limit, so these 11 are computed here.
# One shared corpus pass carries all 88,275 of their queries -- the score block is sized by
# (query_block x doc_block), not by total queries, so eleven shards ride in one pass.
set -u
export MSYS_NO_PATHCONV=1 PYTHONIOENCODING=utf-8 PYTHONUTF8=1 PYTHONHASHSEED=0
cd /c/Users/Swastik/Desktop/CRAG

MISSING=2,3,4,6,7,10,11,12,15,17,22

echo "=== local build of shards $MISSING"
python -u src/dataset_canonical/build_splade_parts_local.py \
  --ds 2wiki --qshards 24 --only "$MISSING"
BE=$?
echo "LOCAL_BUILD_EXIT=$BE"
[ "$BE" -eq 0 ] || { echo "ABORT: local build failed"; exit 1; }

echo; echo "=== handing off to the record chain"
bash scratchpad/_finish_2wiki.sh
echo "FINISH_EXIT=$?"
