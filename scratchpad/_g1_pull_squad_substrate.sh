#!/usr/bin/env bash
# Pull the squad substrate (built on spanishorgay) + its build manifest DOWN to local, so everything produced
# across Modal accounts is mirrored locally.
set -u
cd /c/Users/Swastik/Desktop/CRAG
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8 MODAL_PROFILE=spanishorgay
mkdir -p data/l2_corpus/squad_clean results/GENERALIZATION
for sp in train val; do
  echo "PULL_START squad_clean/$sp"
  modal volume get --force crag-data-volume "data/l2_corpus/squad_clean/$sp" "data/l2_corpus/squad_clean/$sp" 2>&1 | grep -viE "charmap|codec|UnicodeEncode" | tail -3
  n=$(ls "data/l2_corpus/squad_clean/$sp" 2>/dev/null | wc -l)
  echo "PULL_DONE squad_clean/$sp files=$n size=$(du -sh data/l2_corpus/squad_clean/$sp 2>/dev/null|cut -f1)"
done
echo "PULL_START build_manifest"
modal volume get --force crag-data-volume "results/GENERALIZATION/_g1_build_squad_clean.json" "results/GENERALIZATION/_g1_build_squad_clean.json" 2>&1 | grep -viE "charmap|codec" | tail -2
echo "PULL_SQUAD_SUBSTRATE_COMPLETE"
