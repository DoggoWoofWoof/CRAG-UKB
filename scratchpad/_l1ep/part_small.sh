set -e
cd /c/Users/Swastik/Desktop/CRAG
for d in musique_clean squad_clean metaqa 2wiki_clean; do
  python -u scratchpad/_l1ep_part.py $d > scratchpad/_l1ep/log_part_$d.txt 2>&1
  echo "PART_${d}_DONE $(date)"
done
echo PART_SMALL_DONE
