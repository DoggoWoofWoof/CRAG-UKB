"""Static ceiling of ANY six-slot candidate-admission rule under ALL-gold@P50 (no arm, nothing selected, no parameter): a query whose
served P50 misses more than B = 6 gold blocks cannot be repaired by the frozen structure swap whatever the admission criterion, and a
query can be repaired by the H2 source only if every missing gold block is proposed.  Same pinned beam and frozen SAFE contexts as
_l1c_h2safe.py; counts only.
    python -u _l1c_h2safe_ceiling.py <cache>      -> results/L1_COVPART/h2safe_ceiling_A_<cache>.json
"""
import hashlib
import json
import os
import sys

import numpy as np

import _l1g_core as G
import _l1kb_core as KB
import _l1ps_router as RT

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(G.X.REPO, "results", "L1_COVPART")
PINNED_SHA = "19774617f5841fb02471498ad2af29ba9098f32e11aec06831724d569c99e6c5"
CFG = dict(KB.BASE_CFG)
B = CFG["B"]
name = sys.argv[1]
raw = open(os.path.join(HERE, "_l1c_microl3.py"), "rb").read()
assert hashlib.sha256(raw).hexdigest() == PINNED_SHA
src = raw.decode("utf-8")
head = src[src.index("import json"):src.index("# ---------------------------------------------------------------- repair + evaluation")]
head = head.replace("name = sys.argv[1]", "name = %r" % name).replace('LATENT = "--latent" in sys.argv', "LATENT = False")
exec(head)

z0 = np.load(C.path, allow_pickle=True)
zz = {k: z0[k] for k in z0.files if k != "meta_json"}
meta = C.meta
Cc = RT.build_cache(name, zz, meta, [CFG["M_struct"]], [CFG["M_ret"]], [CFG["agg"]])
ctxs = KB.contexts(zz, meta, Cc, B, CFG)
goldp = Cc["goldp"]


def mbucket(k):
    return "1" if k == 1 else "2-3" if k <= 3 else "4-6" if k <= 6 else "7-12" if k <= 12 else "13+"


groups = [("hop%d" % h, [qi for qi in rowsA if hops[qi] == h]) for h in (1, 2, 3)] if name.startswith("metaqa") else [("all", list(rowsA))]
res = {"cache": name, "n_DEV_A": nA, "B": B, "groups": {}}
for hname, rows_h in groups:
    o = {"n": len(rows_h), "n_gold_outside_P50": 0,
         "missing_gold_blocks_histogram": {"1": 0, "2-3": 0, "4-6": 0, "7-12": 0, "13+": 0},
         "repairable_by_any_six_slot_rule (missing <= B)": 0,
         "repairable_and_all_missing_proposed_by_H2": 0,
         "repairable_and_all_missing_proposed_by_static_structural": 0,
         "repairable_and_all_missing_proposed_by_retrieval": 0,
         "repairable_and_all_missing_in_SAFE_H2_candidate_pool": 0,
         "repairable_and_all_missing_in_canonical_SAFE_candidate_pool": 0,
         "not_repairable_but_all_missing_proposed_by_H2 (missing > B)": 0}
    for qi in rows_h:
        c = ctxs[qi]
        miss = [b for b in goldp[qi] if b not in c["base50"]]
        if not miss:
            continue
        o["n_gold_outside_P50"] += 1
        o["missing_gold_blocks_histogram"][mbucket(len(miss))] += 1
        h2 = set(int(hard[v]) for v in beams1[qi] + beams2[qi])
        st, rt = set(c["spos"]), set(c["rpos"])
        pool0 = set(c["bnd"]) | set(c["chal"])
        allh2 = all(b in h2 for b in miss)
        if len(miss) > B:
            o["not_repairable_but_all_missing_proposed_by_H2 (missing > B)"] += int(allh2)
            continue
        o["repairable_by_any_six_slot_rule (missing <= B)"] += 1
        o["repairable_and_all_missing_proposed_by_H2"] += int(allh2)
        o["repairable_and_all_missing_proposed_by_static_structural"] += int(all(b in st for b in miss))
        o["repairable_and_all_missing_proposed_by_retrieval"] += int(all(b in rt for b in miss))
        o["repairable_and_all_missing_in_canonical_SAFE_candidate_pool"] += int(all(b in pool0 for b in miss))
        o["repairable_and_all_missing_in_SAFE_H2_candidate_pool"] += int(all(b in pool0 or b in h2 for b in miss))
    res["groups"][hname] = o
    G.log("%s %s" % (hname, json.dumps(o)))
G.S.wj(os.path.join(OUT, "h2safe_ceiling_A_%s.json" % name), res)
G.log("done")
