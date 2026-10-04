cd "C:/Users/Swastik/Desktop/CRAG"
for ds in hotpotqa webqsp 2wiki 2wiki_universe musique metaqa squad; do
  for enc in dense splade; do
    echo "=== MANIFEST $ds/$enc $(date +%H:%M:%S) ==="
    python scratchpad/_w0_retrieval_manifest.py "$ds" "$enc" 2>&1 | tail -1
  done
done
echo "MANIFEST_SWEEP_DONE $(date +%H:%M:%S)"
