set -e
V="P4_CE_UNWEIGHTED P4_CE_NER_ONLY P4_CE_LOCAL_ONLY"
for ds in metaqa musique_clean; do
  python scratchpad/_l1ep_part.py $ds $V
  python scratchpad/_l1ep_c.py    $ds $V
  echo "P4MECH_${ds}_DONE $(date)"
done
