cd /c/Users/Swastik/Desktop/CRAG
T=H4_MTKAHYPAR_TRUE_HYPERGRAPH__G2_TRUE_HYPERGRAPH
for d in squad_clean musique_clean metaqa 2wiki_clean; do
  python -u scratchpad/_l1ov_eval.py $d $T O0_CORE O1_STRUCT O4_FULL_C b0.25 b0.5 b1.0
  echo "P16_${d}_DONE $(date)"
done
echo P16_ALL_DONE
