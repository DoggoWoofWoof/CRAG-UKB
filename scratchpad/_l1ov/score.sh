cd /c/Users/Swastik/Desktop/CRAG
for d in musique_clean squad_clean metaqa 2wiki_clean; do
  tags=$(ls scratchpad/_l1ep/parts/ | grep "^${d}__H" | sed "s/^${d}__//;s/\.npy$//" | tr '\n' ' ')
  echo "SCORING $d : CURRENT $tags"
  python -u scratchpad/_l1ov_hard.py score $d CURRENT $tags
done
echo SCORE_DONE
