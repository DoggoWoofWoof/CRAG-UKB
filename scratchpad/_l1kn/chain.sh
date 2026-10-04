set -e
cd /c/Users/Swastik/Desktop/CRAG
while [ ! -f scratchpad/_l1kn/run_2wiki_clean.json ] || [ ! -f scratchpad/_l1kn/run_squad_clean.json ]; do sleep 45; done
echo "SMALL_DONE $(date)"
python scratchpad/_l1kn_run.py webqsp        > scratchpad/_l1kn/log_run_webqsp.txt 2>&1; echo "WEBQSP_RUN_DONE $(date)"
python scratchpad/_l1kn_run.py hotpotqa_clean > scratchpad/_l1kn/log_run_hotpot.txt 2>&1; echo "HOTPOT_RUN_DONE $(date)"
for d in musique_clean squad_clean metaqa 2wiki_clean; do
  python scratchpad/_l1kn_k8.py $d 1 > scratchpad/_l1kn/log_k8_$d.txt 2>&1; echo "K8_${d}_DONE $(date)"
done
python scratchpad/_l1kn_k8.py webqsp 3        > scratchpad/_l1kn/log_k8_webqsp.txt 2>&1; echo "K8_webqsp_DONE $(date)"
python scratchpad/_l1kn_k8.py hotpotqa_clean 3 > scratchpad/_l1kn/log_k8_hotpot.txt 2>&1; echo "K8_hotpot_DONE $(date)"
echo ALL_CHAIN_DONE
