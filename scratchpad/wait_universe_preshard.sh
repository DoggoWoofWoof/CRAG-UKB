cd /c/Users/Swastik/Desktop/CRAG
echo "[watch] waiting for universe build..."
while ! grep -qa "DONE_2WIKI_UNIVERSE" scratchpad/2wiki_universe.log 2>/dev/null; do
  if ! kill -0 7159 2>/dev/null && ! grep -qa "DONE_2WIKI_UNIVERSE" scratchpad/2wiki_universe.log; then
    echo "[watch] universe build pid gone without DONE — check log"; exit 2
  fi
  sleep 15
done
echo "[watch] universe build DONE; pre-sharding docs..."
PYTHONIOENCODING=utf-8 python -c "from src.experiments import canonical_encode as CE; n=CE.pre_shard('2wiki_universe','docs'); print('[watch] 2wiki_universe docs pre-sharded ->',n,'shards; n_items=',CE.n_items('2wiki_universe','docs'))"
echo "[watch] READY_FOR_RESTART"
