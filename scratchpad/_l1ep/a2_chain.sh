set -e
cd /c/Users/Swastik/Desktop/CRAG
for d in musique_clean squad_clean metaqa 2wiki_clean; do
  python -u scratchpad/_l1ep_a2.py $d 1 > scratchpad/_l1ep/log_a2_$d.txt 2>&1
  echo "A2_${d}_DONE $(date)"
done
echo A2_SMALL_DONE
