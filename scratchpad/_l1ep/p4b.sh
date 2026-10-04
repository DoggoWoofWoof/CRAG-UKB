set -e
V="P4_CE_LOCAL_ONLY P4_CE_UNWEIGHTED P4_CE_NER_ONLY PM3_TOPOLOGY_C_seed1 PM3_TOPOLOGY_C_seed2 PM3_TOPOLOGY_C_seed3 PM3_TOPOLOGY_C_seed4"
for ds in squad_clean metaqa musique_clean; do
  python scratchpad/_l1ep_part.py $ds $V
  python scratchpad/_l1ep_c.py    $ds $V
  echo "P4B_${ds}_DONE $(date)"
done
