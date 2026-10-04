"""C8 prep: precompute per-archetype top-k utilities for train+val, build the deterministic inner/dev split.
Also ensure C6 inference-safe features exist for train+val (build if missing)."""
import sys, os, json
sys.path.insert(0, "scratchpad")
import numpy as np, l2_c8 as C8, l2_c6_predict as PRED
log = lambda *a: print(*a, flush=True)

for ds in C8.DS:
    for sp in ("train", "val"):
        fp = f"{C8.OUT}/_c6_feat_{ds}_{sp}.npz"
        if not os.path.exists(fp): PRED.build_features(ds, sp, log)
        C8.precompute_arch(ds, sp, log)
split = C8.build_inner_split(log=log)
json.dump(split, open(f"{C8.OUT}/_c8_split_meta.json", "w"), indent=1)
log("C8_PREP_DONE " + json.dumps(split))
