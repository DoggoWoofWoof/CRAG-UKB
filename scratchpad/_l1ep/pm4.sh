cd /c/Users/Swastik/Desktop/CRAG
while ! grep -q P4B_musique_clean_DONE scratchpad/_l1ep/log_p4b.txt 2>/dev/null; do sleep 20; done
for ds in metaqa musique_clean squad_clean; do
  python scratchpad/_l1ep_part.py $ds PM4_TOPOLOGY_C_NERW
  python scratchpad/_l1ep_pu.py   $ds scratchpad/_l1ep/parts/${ds}__PM4_TOPOLOGY_C_NERW.npy PM4_TOPOLOGY_C_NERW
  python scratchpad/_l1ep_c.py    $ds PM4_TOPOLOGY_C_NERW
  echo "PM4_${ds}_DONE $(date)"
done
