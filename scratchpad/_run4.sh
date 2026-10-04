set -e
DS4="metaqa musique_clean 2wiki_clean squad_clean"
echo "=== STEP0 ==="; python scratchpad/_l1ca_step0.py $DS4
echo "=== ADMIT ==="; for d in musique_clean 2wiki_clean squad_clean; do python scratchpad/_l1ca_admit.py $d; done
echo "=== AUDIT ==="; for d in $DS4; do python scratchpad/_l1ca_audit.py $d; done
echo "RUN4_DONE"
