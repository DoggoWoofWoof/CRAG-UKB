set -x
for d in squad_clean musique_clean 2wiki_clean hotpotqa_clean webqsp; do
  python - "$d" <<'PY'
import sys; sys.path.insert(0,'scratchpad'); sys.path.insert(0,'.')
import _l1ss_build as B
B.build(sys.argv[1], want_future=False)
PY
done
echo "=== BUILDS DONE ==="
