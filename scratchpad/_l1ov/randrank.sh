cd /c/Users/Swastik/Desktop/CRAG
export PYTHONIOENCODING=utf-8 PYTHONUTF8=1
for d in metaqa 2wiki_clean squad_clean webqsp hotpotqa_clean; do
  python -u scratchpad/_l1ov_randrank.py $d
done
echo RANDRANK_ALL_DONE
