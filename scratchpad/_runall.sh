set -e
DS="metaqa webqsp musique_clean 2wiki_clean hotpotqa_clean squad_clean"
echo "=== STEP0 ==="; python scratchpad/_l1ca_step0.py $DS
echo "=== ADMIT ==="; for d in $DS; do python scratchpad/_l1ca_admit.py $d; done
echo "=== AUDIT ==="; for d in $DS; do python scratchpad/_l1ca_audit.py $d; done
echo "RUNALL_DONE"
