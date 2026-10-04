#!/usr/bin/env bash
# Resume 2wiki splade after the FIFTH workspace disable (extra_ip9HxU, at 13 of 24 parts).
# $1 = MODAL_PROFILE, $2 = comma-separated shard indices.
# Split across two workspaces on purpose: a single disable then costs a subset, not the run.
set -u
export MSYS_NO_PATHCONV=1 PYTHONIOENCODING=utf-8 PYTHONUTF8=1 PYTHONHASHSEED=0
export MODAL_PROFILE="$1"
export CRAG_GPU=none
cd /c/Users/Swastik/Desktop/CRAG
echo "PROFILE=$1  SHARDS=$2"
python scratchpad/upload_full_store.py 2wiki_splade
echo "STAGE_EXIT=$?"
modal run scratchpad/modal_retrieval_cache.py::shards --ds 2wiki --qshards 24 --only "$2"
echo "SHARDRUN_EXIT=$?"
