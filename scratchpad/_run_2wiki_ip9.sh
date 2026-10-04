set -x
export MSYS_NO_PATHCONV=1 PYTHONIOENCODING=utf-8 PYTHONUTF8=1
export MODAL_PROFILE=extra_ip9HxU CRAG_GPU=none
cd /c/Users/Swastik/Desktop/CRAG
python scratchpad/upload_full_store.py 2wiki_splade || exit 3
modal run scratchpad/modal_retrieval_cache.py::shards --ds 2wiki --qshards 24
echo "SHARDRUN_EXIT=$?"
