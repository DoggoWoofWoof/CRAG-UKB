import os, json, sys
datasets = ["squad_clean","2wiki_clean","musique_clean","musique_hpr_clean","2wiki_hpr_clean","hotpot_hpr_clean","hotpotqa_clean","webqsp","metaqa"]
from src.core.engine import CoreEngine
from src.experiments.overlap_retrain import _hard_membership, _splits
for ds in datasets:
    try:
        eng=CoreEngine(source=ds, index_subdir="gte_qwen")
        sp=_splits(eng, _hard_membership(eng))
        for k,v in sp.items():
            print(ds, k, len(v))
    except Exception as e:
        print(ds, "eng err", e)
        import traceback; traceback.print_exc()
