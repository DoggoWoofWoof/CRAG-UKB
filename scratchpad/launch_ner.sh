#!/usr/bin/env bash
# Parallel sharded NER for a large corpus: split shards across N local spaCy processes (resumable),
# wait, then merge -> data/canonical/<ds>/graph_ner.tsv. Usage: launch_ner.sh <dataset> <nproc>
set -u
cd /c/Users/Swastik/Desktop/CRAG
DS="$1"; NP="${2:-6}"
NS=$(python -c "import json;print(json.load(open(f'data/canonical/$DS/encodings/_src/docs/_srcmeta.json'))['n_shards'])")
echo "[launch-ner] $DS: $NS shards across $NP processes"
per=$(( (NS + NP - 1) / NP ))
pids=()
for k in $(seq 0 $((NP-1))); do
  a=$(( k * per )); b=$(( a + per - 1 )); [ $b -ge $NS ] && b=$((NS-1))
  [ $a -ge $NS ] && break
  echo "[launch-ner] worker $k: shards $a-$b"
  PYTHONIOENCODING=utf-8 python scratchpad/build_ner_sharded.py --dataset "$DS" --stage extract --shards "$a-$b" \
      > "scratchpad/ner_${DS}_extract_w${k}.log" 2>&1 &
  pids+=($!)
done
echo "[launch-ner] extract pids: ${pids[*]}"
fail=0
for p in "${pids[@]}"; do wait $p || fail=1; done
if [ $fail -ne 0 ]; then echo "[launch-ner] EXTRACT had a failure; not merging. inspect logs."; exit 1; fi
echo "[launch-ner] all extract workers done; merging $DS..."
PYTHONIOENCODING=utf-8 python scratchpad/build_ner_sharded.py --dataset "$DS" --stage merge > "scratchpad/ner_${DS}_merge.log" 2>&1
echo "[launch-ner] $DS NER DONE:"; tail -3 "scratchpad/ner_${DS}_merge.log"
echo "NER_PIPELINE_DONE_$DS"
