cd /c/Users/Swastik/Desktop/CRAG
while ! grep -q P4B_musique_clean_DONE scratchpad/_l1ep/log_p4b.txt 2>/dev/null; do sleep 20; done
bash scratchpad/_l1ep/run2wiki.sh
