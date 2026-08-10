#!/usr/bin/env bash
# =============================================================================
# RERUN after substrate fixes (musique titles + squad label-free rebuild).
# - musique_clean: FULL suite re-run (old results were on the title-less graph;
#   overlap1 had collapsed to hard). Stale outputs deleted below so guards fire.
# - squad_clean:   NEW label-free source (doc->q backedges removed). s1 baselines
#   only (single-hop control, matches campaign design for squad).
# - hotpotqa_clean: FULL suite (never ran in the campaign; verified clean).
# Resumable: every step guarded by its output file. Run: bash scripts/rerun_fixed.sh
# =============================================================================
set -u
cd /c/Users/Swastik/Desktop/CRAG || exit 1
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8

RUN(){ echo ">>> RUN $* @ $(date '+%H:%M:%S')"; python experiments.py run "$@"; echo "<<< done rc=$? @ $(date '+%H:%M:%S')"; }
have(){ if [ -f "$1" ]; then echo "SKIP (exists) $1"; return 0; else return 1; fi; }
OA=results/overlap_ablation

s1(){ ds=$1; lim=$2
  have $OA/${ds}_overlap_retrain_S1loss.json  || RUN overlap-retrain --backend local -- --datasets $ds --configs hard --losses kl coverage --epochs 100 $lim --out_suffix S1loss
  have $OA/${ds}_overlap_retrain_S1nohnm.json || RUN overlap-retrain --backend local -- --datasets $ds --configs hard --losses kl --hn_k 0 --epochs 100 $lim --out_suffix S1nohnm
}
s2(){ ds=$1; lim=$2
  have $OA/${ds}_overlap_retrain_S2struct.json  || RUN overlap-retrain --backend local -- --datasets $ds --configs hard overlap1 overlap2 syn1 knn1 knn2 knn3 overlap1+knn1 overlap1+knn3 --losses kl --epochs 100 $lim --out_suffix S2struct
  have $OA/${ds}_overlap_retrain_S2jigsaw.json   || RUN overlap-retrain --backend local -- --datasets $ds --configs hard overlap1 knn1 knn3 overlap1+knn1 --losses kl coverage --epochs 100 $lim --out_suffix S2jigsaw
}
s3(){ ds=$1
  have results/finetune_ablation/${ds}_encoder_finetune.json || RUN finetune-encoder --backend local -- --datasets $ds --epochs 5 --unfreeze 2
  have results/adaptive_k/${ds}_adaptive_k.json              || RUN adaptive-k --backend local -- --datasets $ds --configs hard overlap1+knn1
  have results/multiproto/${ds}_overlap1_knn1.json           || RUN multiproto --backend local -- --datasets $ds --configs overlap1 overlap1+knn1 --protos 1 2 4
  have results/query_decomp/${ds}_hard.json                  || RUN query-decomp --backend local -- --datasets $ds --configs hard overlap1+knn1
  have results/gnn_ablation/${ds}_gnn.json                   || RUN train-gnn --backend local -- --datasets $ds --models gin gcn sage --epochs 15
}
s4(){ ds=$1; lim=$2
  have results/level_1/comparison_${ds}.json          || RUN bench-level1 --backend local -- --datasets $ds --methods faiss_centroid $lim
  have results/l3_recovery/${ds}_overlap1_knn1.json   || RUN l3-recovery --backend local -- --datasets $ds --config overlap1+knn1 --epochs 100 $lim
  have results/pool_narrow/${ds}_overlap1_knn1.json   || RUN pool-narrow --backend local -- --datasets $ds --config overlap1+knn1 $lim
}

echo "########## RERUN-FIXED START @ $(date) ##########"

echo "===== [1/3] musique_clean (titles recovered: relational title edges + kNN) ====="
s1 musique_clean ""; s2 musique_clean ""; s3 musique_clean; s4 musique_clean ""

echo "===== [2/3] squad_clean (label-free single-hop control) ====="
s1 squad_clean ""

echo "===== [3/3] hotpotqa_clean (canonical multi-hop, title-structured) ====="
if [ -f data/ukb_storage/hotpotqa_clean/centroids.index ]; then
  s1 hotpotqa_clean ""; s2 hotpotqa_clean ""; s3 hotpotqa_clean; s4 hotpotqa_clean ""
else
  echo "  hotpotqa_clean UKB not ready — skipped"
fi

echo "########## RERUN_FIXED_COMPLETE @ $(date) ##########"
