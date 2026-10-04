set -x
for d in 2wiki_clean metaqa musique_clean hotpotqa_clean webqsp; do
  python - "$d" <<'PY'
import sys; sys.path.insert(0,'scratchpad'); sys.path.insert(0,'.')
import _l1ss_build as B
B.build(sys.argv[1], want_future=False, beam_rank="R1")
PY
done
echo "=== BEAMR1 BUILDS DONE ==="
