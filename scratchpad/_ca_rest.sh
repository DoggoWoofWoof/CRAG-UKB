set -x
for d in webqsp 2wiki_clean musique_clean hotpotqa_clean squad_clean; do
  python scratchpad/_l1ca_run.py $d
done
echo "=== AUDIT RUNS DONE ==="
