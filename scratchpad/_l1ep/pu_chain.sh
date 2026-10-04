set -e
cd /c/Users/Swastik/Desktop/CRAG
for d in squad_clean metaqa 2wiki_clean hotpotqa_clean webqsp; do
  python -u scratchpad/_l1ep_pu.py $d CURRENT P1_METIS_CURRENT > scratchpad/_l1ep/log_pu_$d.txt 2>&1
  echo "PU_${d}_DONE $(date)"
done
echo PU_ALL_DONE
