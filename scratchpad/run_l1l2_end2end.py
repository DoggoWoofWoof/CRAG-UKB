"""
Proper end-to-end L1->L2 A/B/C/D on full validation populations, sharded on Modal if needed.
D = canonical H8 full-corpus (scope_topk=0) via e2e_pipeline/ l2_seed with MAXK 100
A = structural+semantic-kNN, B=struct+NER, C=struct+NER+semantic via same L1 vote + same L2
Uses full validation queries (official dev or held-out 20% = val split from _get_split_queries), not 500.
Shards queries 0:500,500:1000 etc. and merges.
"""
import os, json, sys, time, subprocess, pathlib
sys.path.insert(0, '.')
# This script will dispatch Modal jobs for each dataset/variant/shard
# For now, we will run locally for squad_clean as a smoke test with full val (26063 queries) sharded 0:500 etc., but to avoid local slow, we dispatch to Modal

DATASETS = ["squad_clean","2wiki_clean","musique_clean","metaqa","webqsp","hotpotqa_clean"]
VARIANTS = ["A","B","C","D"]
# Use val split for graph selection, not test
# For each dataset, we need to run the L1->L2 pipeline
# Instead of reimplementing, we will call the existing e2e_pipeline for D and a new wrapper for A/B/C that builds variant partitions then runs same L2

# For D, we can directly call e2e_pipeline with scope_topk=0 on full val population via l2_seed's run with full val
# For A/B/C, we need to build variant partitions first, then run L2 within those partitions

# Simplified: dispatch via experiments.py tasks if available, else direct python -m

def dispatch(dataset, variant, shard_start, shard_end):
    # Example: shard queries val[shard_start:shard_end]
    # Use Modal backend
    # For D, variant is "D", we call a custom task that will be implemented as src/experiments/l1l2_ablation.py
    # For now, placeholder: just echo
    print(f"Dispatch {dataset} {variant} shard {shard_start}:{shard_end}")
    # In real implementation, this would be:
    # python experiments.py run l1l2-ablation --backend modal --account 1 -- --dataset squad_clean --variant A --shard 0:500
    pass

if __name__ == "__main__":
    # Check C5 frozen
    from src.experiments.canonical_encode import shard_complete
    assert sum(1 for s in range(150) if shard_complete('2wiki_universe','docs','dense',s))==150
    assert sum(1 for s in range(131) if shard_complete('hotpotqa','docs','dense',s))==131
    print("C5 FROZEN verified 150/150 131/131")
    # Dispatch logic would go here
    # For now, just verify we can load the L2 pipeline
    from src.experiments.e2e_pipeline import run as e2e_run
    print("e2e_pipeline import OK, MAXK", __import__('src.experiments.l2_seed', fromlist=['MAXK']).MAXK)
    # Dispatch would be done via Modal with sharding
    print("Ready to dispatch full validation A/B/C/D with sharding on Modal")
