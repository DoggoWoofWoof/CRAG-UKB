#!/usr/bin/env bash
# Stage all hotpot G1 build+eval ingredients to crm's crag-data-volume (individual puts; volume put <dir> skips big files).
set -u
cd /c/Users/Swastik/Desktop/CRAG
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8 MODAL_PROFILE=crm
FILES=(
  "data/ukb_storage/hotpotqa_clean/gte_qwen/nodes.npy"
  "data/ukb_storage/hotpotqa_clean/gte_qwen/queries_all.npy"
  "data/ukb_storage/hotpotqa_clean/gte_qwen/dense_top200_all.npy"
  "data/ukb_storage/hotpotqa_clean/gte_qwen/splade_top200_all.npy"
  "data/ukb_storage/hotpotqa_clean/gte_qwen/query_ids_all.json"
  "data/ukb_storage/hotpotqa_clean/splade_doc_embs.pkl"
  "scratchpad/ablation_qwen/hotpotqa_clean/variant_C/partition_map.json"
  "data/processed/master_nodes_hotpotqa_clean.json"
  "results/L2/_heads/universal_offset_src_gteqwen.pt"
  "results/L2/_heads/universal_mixture_src_gteqwen.pt"
  "results/GENERALIZATION/_g1_backbone/base_full.joblib"
  "results/GENERALIZATION/_g1_backbone/c8c.joblib"
  "results/GENERALIZATION/_g1_backbone/C11_models.joblib"
  "results/GENERALIZATION/_g1_backbone/source_val_stats.json"
)
for f in "${FILES[@]}"; do
  if [ ! -f "$f" ]; then echo "STAGE_MISSING_LOCAL $f"; continue; fi
  echo "STAGE_PUT_START $f ($(du -h "$f"|cut -f1))"
  modal volume put --force crag-data-volume "$f" "$f" 2>&1 | grep -viE "charmap|codec|UnicodeEncode" | tail -2
  echo "STAGE_PUT_DONE $f rc=${PIPESTATUS[0]}"
done
echo "STAGE_HOTPOT_CRM_COMPLETE"
