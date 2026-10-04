#!/usr/bin/env bash
set -u
cd /c/Users/Swastik/Desktop/CRAG
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8 MODAL_PROFILE=spanishorgay
FILES="_summary.json cand_ids.npy dense_rank.npy dense_score.npy expert_meta.npz labels.npy mixture_score.npy offset_score.npy part_id.npy part_rank.npy query_meta.json query_offsets.npy relation_mask.npy relation_qwen_score.npy splade_rank.npy splade_scope_rank.npy splade_scope_score.npy train_sub_local.npy train_sub_offsets.npy train_sub_source.json"
for sp in train val; do
  mkdir -p "data/l2_corpus/squad_clean/$sp"
  for f in $FILES; do
    p="data/l2_corpus/squad_clean/$sp/$f"
    modal volume get --force crag-data-volume "$p" "$p" >/dev/null 2>&1
    rc=$?; sz=$(stat -c%s "$p" 2>/dev/null || echo 0)
    echo "GOT $sp/$f rc=$rc bytes=$sz $(date +%H:%M:%S)"
  done
done
modal volume get --force crag-data-volume "results/GENERALIZATION/_g1_build_squad_clean.json" "results/GENERALIZATION/_g1_build_squad_clean.json" >/dev/null 2>&1
echo "MANIFEST rc=$? bytes=$(stat -c%s results/GENERALIZATION/_g1_build_squad_clean.json 2>/dev/null || echo 0)"
echo "PULL_SQUAD_PERFILE_COMPLETE $(date +%H:%M:%S)"
