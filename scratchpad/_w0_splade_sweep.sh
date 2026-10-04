cd "C:/Users/Swastik/Desktop/CRAG"
for ds in metaqa musique 2wiki webqsp hotpotqa 2wiki_universe; do
  echo "=== SPLADE VERIFY $ds $(date +%H:%M:%S) ==="
  python scratchpad/_w0_splade_verify.py "$ds" 2>&1 | tail -1
done
echo "SPLADE_SWEEP_DONE $(date +%H:%M:%S)"
