#!/usr/bin/env bash
# =============================================================================
# CLEAN LEVEL-1 ABLATION CAMPAIGN (post-leak-fix)
# Resumable: every step is guarded by its output file, so re-running skips
# completed work. Ordered by dataset priority (2wiki_clean first) so the
# highest-value clean results land before the expensive metaqa consistency pass.
# All ablations emit the full metric suite + paired McNemar (per-query FullCov@20),
# seeded MLP init. Substrate is label-free (title/kNN edges; gold on question
# nodes only). Run in background: bash scripts/clean_campaign.sh
# =============================================================================
set -u
cd /c/Users/Swastik/Desktop/CRAG || exit 1
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8

RUN(){ echo ">>> RUN $* @ $(date '+%H:%M:%S')"; python experiments.py run "$@"; echo "<<< done rc=$? @ $(date '+%H:%M:%S')"; }
have(){ if [ -f "$1" ]; then echo "SKIP (exists) $1"; return 0; else return 1; fi; }
OA=results/overlap_ablation

# ---- S1: baselines (loss ablation + HNM ablation -> establish KL+HNM best) ----
s1(){ ds=$1; lim=$2
  have $OA/${ds}_overlap_retrain_S1loss.json  || RUN overlap-retrain --backend local -- --datasets $ds --configs hard --losses kl coverage --epochs 100 $lim --out_suffix S1loss
  have $OA/${ds}_overlap_retrain_S1nohnm.json || RUN overlap-retrain --backend local -- --datasets $ds --configs hard --losses kl --hn_k 0 --epochs 100 $lim --out_suffix S1nohnm
}
# ---- S2: structure (knn/overlap sweep + combos) + Jigsaw-loss across structures ----
s2(){ ds=$1; lim=$2
  have $OA/${ds}_overlap_retrain_S2struct.json  || RUN overlap-retrain --backend local -- --datasets $ds --configs hard overlap1 overlap2 syn1 knn1 knn2 knn3 overlap1+knn1 overlap1+knn3 --losses kl --epochs 100 $lim --out_suffix S2struct
  # Jigsaw (coverage) loss vs KL across the key structure configs (McNemar per config)
  have $OA/${ds}_overlap_retrain_S2jigsaw.json   || RUN overlap-retrain --backend local -- --datasets $ds --configs hard overlap1 knn1 knn3 overlap1+knn1 --losses kl coverage --epochs 100 $lim --out_suffix S2jigsaw
}
# ---- S3: methods (what worked / didn't) — proof datasets only (metaqa too slow) ----
s3(){ ds=$1
  have results/finetune_ablation/${ds}_encoder_finetune.json || RUN finetune-encoder --backend local -- --datasets $ds --epochs 5 --unfreeze 2
  have results/adaptive_k/${ds}_adaptive_k.json              || RUN adaptive-k --backend local -- --datasets $ds --configs hard overlap1+knn1
  have results/multiproto/${ds}_overlap1_knn1.json           || RUN multiproto --backend local -- --datasets $ds --configs overlap1 overlap1+knn1 --protos 1 2 4
  have results/query_decomp/${ds}_hard.json                  || RUN query-decomp --backend local -- --datasets $ds --configs hard overlap1+knn1
  have results/gnn_ablation/${ds}_gnn.json                   || RUN train-gnn --backend local -- --datasets $ds --models gin gcn sage --epochs 15   # slow tail
}
# ---- S4: level-1 baseline + level-3 recovery + pool-narrow (real edges) ----
s4(){ ds=$1; lim=$2
  have results/level_1/comparison_${ds}.json          || RUN bench-level1 --backend local -- --datasets $ds --methods faiss_centroid $lim
  have results/l3_recovery/${ds}_overlap1_knn1.json   || RUN l3-recovery --backend local -- --datasets $ds --config overlap1+knn1 --epochs 100 $lim
  have results/pool_narrow/${ds}_overlap1_knn1.json   || RUN pool-narrow --backend local -- --datasets $ds --config overlap1+knn1 $lim
}

echo "########## CLEAN CAMPAIGN START @ $(date) ##########"

echo "===== [1/4] 2wiki_clean (primary proof dataset) ====="
s1 2wiki_clean "";  s2 2wiki_clean "";  s3 2wiki_clean;  s4 2wiki_clean ""

echo "===== [2/4] musique_clean (knn-only structure; no natural title graph) ====="
s1 musique_clean ""; s2 musique_clean ""; s3 musique_clean; s4 musique_clean ""

echo "===== [3/4] metaqa (already clean — consistency pass w/ full metrics+McNemar, limited) ====="
s1 metaqa "--limit 40000"; s2 metaqa "--limit 40000"; s4 metaqa "--limit 40000"

echo "===== [4/5] squad (single-hop, now 190 partitions — baselines only) ====="
s1 squad ""

echo "===== [5/5] hotpotqa_clean (canonical multi-hop, title-structured) ====="
if [ -f data/ukb_storage/hotpotqa_clean/centroids.index ]; then
  s1 hotpotqa_clean ""; s2 hotpotqa_clean ""; s3 hotpotqa_clean; s4 hotpotqa_clean ""
else
  echo "  hotpotqa_clean UKB not ready (fresh encode still running) — skipped; re-run campaign to include it"
fi

echo "########## CLEAN_CAMPAIGN_COMPLETE @ $(date) ##########"
