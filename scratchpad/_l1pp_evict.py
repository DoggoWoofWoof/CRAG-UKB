"""STEP 0 (quantified) -- is the boundary competition actually a competition?

The dump shows a canonical-only incumbent at boundary rank 45 scores 1/(60+45) = 0.009524, while a
challenger holding structural rank 1 and nothing else scores 1/(60+1) = 0.016393.  So ANY challenger
inside the top ~42 of the structural channel beats ANY incumbent that has only canonical evidence.
If that is systematic, F6 is not a competition -- it is near-deterministic turnover of channel-poor
incumbents, and raising B just evicts more of them.  This measures it.

  python scratchpad/_l1pp_evict.py
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())
import numpy as np
import _ta_prepartition as TA
import _l1pp_core as PP
import _l1kb_core as KB
import _l1kb_router as JR

OUT = {}
for ds in PP.DSETS:
    z, meta = PP.load(ds); nq = meta["n_dev_queries"]
    goldp = PP.goldparts(z, meta)
    C, GRP = KB.substrate(ds, z, meta, JR.build_groups)
    row = {}
    for Bv in (6, 12):
        ctxs = KB.contexts(z, meta, C, Bv)
        ret = solo = solo_ret = gev = ggain = 0
        gold_bnd = gold_bnd_solo = 0
        for qi, c in enumerate(ctxs):
            X, _ = KB.f6_select(c["bnd"], c["chal"], c["spos"], c["rpos"], c["cpos"], Bv)
            keep = set(X) & set(c["bnd"])
            ret += len(keep)
            for p in c["bnd"]:
                only_c = (p not in c["spos"]) and (p not in c["rpos"])
                solo += only_c
                if only_c and p in keep:
                    solo_ret += 1
                if p in goldp[qi]:
                    gold_bnd += 1
                    gold_bnd_solo += only_c
            fs = c["prot_set"] | set(X)
            b50 = c["base50"]
            gev += len(goldp[qi] & b50 - fs)
            ggain += len(goldp[qi] & fs - b50)
        row[f"B{Bv}"] = {
            "mean_incumbents_retained": round(ret / nq, 3),
            "incumbent_retention_rate": round(ret / (nq * Bv), 4),
            "mean_boundary_incumbents_with_canonical_evidence_only": round(solo / nq, 3),
            "retention_rate_of_canonical_only_incumbents": round(solo_ret / max(1, solo), 4),
            "gold_partitions_evicted": int(gev), "gold_partitions_gained": int(ggain),
            "net_gold_partitions": int(ggain - gev),
            "gold_partitions_sitting_in_the_boundary": int(gold_bnd),
            "of_those_canonical_evidence_only": int(gold_bnd_solo)}
    OUT[ds] = row
    r6, r12 = row["B6"], row["B12"]
    print("%-15s B6: retain %.3f/6 (canon-only retention %.3f)  gold -%d/+%d = %+d  | "
          "B12: retain %.3f/12 (canon-only %.3f)  gold -%d/+%d = %+d"
          % (ds[:14], r6["mean_incumbents_retained"], r6["retention_rate_of_canonical_only_incumbents"],
             r6["gold_partitions_evicted"], r6["gold_partitions_gained"], r6["net_gold_partitions"],
             r12["mean_incumbents_retained"], r12["retention_rate_of_canonical_only_incumbents"],
             r12["gold_partitions_evicted"], r12["gold_partitions_gained"], r12["net_gold_partitions"]),
          flush=True)
json.dump(OUT, open(f"{PP.PPD}/diag/eviction.json", "w"), indent=1)
print("wrote eviction.json")
