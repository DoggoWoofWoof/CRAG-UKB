cd /c/Users/Swastik/Desktop/CRAG
for d in metaqa 2wiki_clean squad_clean webqsp hotpotqa_clean; do
  python -u scratchpad/_l1ov_p17.py $d O4_FULL_C_b0.5 O4_FULL_C_b0.25
  echo "P17_${d}_DONE $(date)"
done
python -u scratchpad/_l1ov_p17.py musique_clean O4_FULL_C_b0.5 O4_FULL_C_b0.25
echo P17_ALL_DONE
