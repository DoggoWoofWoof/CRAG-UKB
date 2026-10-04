cd /c/Users/Swastik/Desktop/CRAG
for i in $(seq 1 400); do   # ~400*45s = 5h horizon
  if grep -qa "PERMANENTLY FAILED" scratchpad/c5_full2.log; then echo "EVENT=PERMFAIL"; break; fi
  if grep -qa "\[c5\] DONE" scratchpad/c5_full2.log; then echo "EVENT=C5_DONE"; break; fi
  if grep -qa "NER_PIPELINE_DONE_2wiki_universe" scratchpad/ner_all.log 2>/dev/null; then echo "EVENT=NER_UNIVERSE_DONE"; break; fi
  if ! kill -0 7671 2>/dev/null; then echo "EVENT=C5_PROC_GONE"; break; fi
  if ! kill -0 7774 2>/dev/null; then echo "EVENT=NER_PROC_GONE"; break; fi
  sleep 45
done
echo "=== snapshot ==="
grep -a healthy scratchpad/c5_full2.log | tail -1
echo "C5 OK=$(grep -ac '\[c5\] OK' scratchpad/c5_full2.log) dense=$(grep -a '\[c5\] OK' scratchpad/c5_full2.log | grep -ac dense) PERMFAIL=$(grep -ac 'PERMANENTLY FAILED' scratchpad/c5_full2.log)"
grep -aE "NER build:|NER DONE|NER_PIPELINE_DONE|ALL_NER_DONE" scratchpad/ner_all.log | tail -6
