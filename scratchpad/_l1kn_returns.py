"""L1 QWEN-KNN EDGE SUBSTRATE AUDIT -- RETURNS.json.

Assembles every named return of the directive from diag/_derived.json plus the raw per-corpus runs,
and applies the verdict rule.  Nothing here re-computes a measurement; it only selects and labels.

  python scratchpad/_l1kn_returns.py
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())
import numpy as np
import _l1kn_sub as SUB

KND, CACHE, DS, SUBS = SUB.KND, SUB.CACHE, SUB.DS, SUB.SUBS


def main():
    d = json.load(open(f"{KND}/diag/_derived.json"))
    sub = json.load(open(f"{KND}/diag/step1_substrates.json"))
    runs = {x: json.load(open(f"{CACHE}/run_{x}.json")) for x in DS
            if os.path.exists(f"{CACHE}/run_{x}.json")}
    R = {}
    R["PHASE"] = "L1 QWEN-KNN EDGE SUBSTRATE AUDIT"
    R["CORPORA_COMPLETED"] = sorted(runs)
    R["CONTRACT"] = {"MASTER_TOPOLOGY": "C", "P": 50, "B": 6, "K0": 60, "M_struct": 64,
                     "M_ret": 32, "MAX_HOPS": 3, "BEAM": 64, "DEG_CAP": 300,
                     "MAX_EDGES_SCORED": 400000, "seeds_per_query": 5,
                     "learning": "NONE", "LLM": "NONE", "encoder_calls": 0, "TEST": "NOT RUN"}
    R["SUBSTRATES"] = SUBS
    R["T0_PARITY"] = d.get("T0_PARITY", {})
    R["FRAC_TOPOLOGY_C_EDGES_NEVER_TRAVERSED"] = d.get("FRAC_TOPOLOGY_C_NEVER_TRAVERSED", {})
    R["KNN_UNIQUE_NEEDED_RECALL"] = d.get("KNN_UNIQUE_NEEDED_RECALL_VIS", {})
    R["KNN_UNIQUE_NEEDED_RECALL_READ"] = d.get("KNN_UNIQUE_NEEDED_RECALL_READ", {})
    R["KNN_UNIQUE_NEEDED_RECALL_METAQA_HOPS"] = d.get(
        "KNN_UNIQUE_NEEDED_RECALL_VIS_metaqa_hops", {})
    R["NEEDED_RECALL_VIS"] = d.get("NEEDED_RECALL_VIS", {})
    R["NEEDED_RECALL_READ"] = d.get("NEEDED_RECALL_READ", {})
    R["KNN_DENSE_REDUNDANCY"] = d.get("KNN_DENSE_REDUNDANCY", {})
    R["KNN_PARTITION_NOVELTY"] = d.get("KNN_PARTITION_NOVELTY", {})
    R["KNN_NEEDED_ENRICHMENT_NOVEL_VS_DUP"] = d.get("KNN_NEEDED_ENRICHMENT_NOVEL_VS_DUP", {})
    for t, nm in (("T0_FROZEN", "T0"), ("T2_FULL_UNION", "T2"), ("T3_MATCHED_HYBRID", "T3"),
                  ("T1_KNN_ONLY", "T1"), ("T5_TOPOLOGY_C", "T5")):
        R[f"{nm}_PARTITIONS_PER_QUERY"] = {k: v.get(t) for k, v in
                                           d.get("PARTITIONS_PER_QUERY", {}).items()}
    R["SATURATION_FRAC_OF_CORPUS_PARTITIONS"] = d.get("SATURATION_FRAC", {})
    O = d.get("ORACLE", {})
    for t, nm in (("T0_FROZEN", "T0"), ("T1_KNN_ONLY", "T1"), ("T2_FULL_UNION", "T2"),
                  ("T3_MATCHED_HYBRID", "T3")):
        R[f"{nm}_ORACLE_HOP3"] = O.get("metaqa/hop3", {}).get(t)
        R[f"{nm}_ORACLE"] = {k: v.get(t) for k, v in O.items()}
    R["BEST_PATH_FAMILY_SEQUENCE"] = d.get("BEST_PATH_FAMILY_SEQUENCE", {})
    R["K8_AUC"] = d.get("K8_AUC", {})
    R["COST"] = d.get("COST", {})
    S10 = d.get("STEP10", {})
    R["STEP10_KNN_CHANNEL"] = S10
    R["STEP10_SUBSTRATE_SUBSTITUTION"] = d.get("STEP10_SUBSTITUTION", {})
    R["KNN_METAQA_HOP2"] = {k: S10.get("metaqa/hop2", {}).get(k) for k in
                            ("K_CHAN_VOTE", "K_CHAN_FULL")}
    R["KNN_METAQA_HOP3"] = {k: S10.get("metaqa/hop3", {}).get(k) for k in
                            ("K_CHAN_VOTE", "K_CHAN_FULL")}
    sig = {v: [] for v in ("K_CHAN_VOTE", "K_CHAN_FULL")}
    gain = {v: [] for v in sig}
    for key, cell in S10.items():
        for v in sig:
            c = cell.get(v)
            if not c:
                continue
            if c["sig"] and c["net"] < 0:
                sig[v].append(key)
            if c["sig"] and c["net"] > 0:
                gain[v].append(key)
    R["SIG_REGRESSIONS"] = sig
    R["SIG_GAINS"] = gain
    R["KNN_CROSS_CORPUS_SAFE"] = {v: ("YES" if not sig[v] else "NO") for v in sig}
    return R, runs


if __name__ == "__main__":
    R, runs = main()
    over = json.load(open(f"{KND}/diag/_verdict.json")) if \
        os.path.exists(f"{KND}/diag/_verdict.json") else {}
    R.update(over)
    json.dump(R, open(f"{KND}/RETURNS.json", "w"), indent=1)
    print(f"wrote {KND}/RETURNS.json  ({len(R)} fields)")
    for k in ["T0_PARITY", "KNN_UNIQUE_NEEDED_RECALL", "KNN_UNIQUE_NEEDED_RECALL_READ",
              "KNN_PARTITION_NOVELTY", "KNN_DENSE_REDUNDANCY", "KNN_METAQA_HOP2",
              "KNN_METAQA_HOP3", "KNN_CROSS_CORPUS_SAFE", "SIG_REGRESSIONS", "SIG_GAINS",
              "BEST_PATH_FAMILY_SEQUENCE", "VERDICT", "PROMOTED", "L1_FROZEN"]:
        if k in R:
            print(f"{k:42s} = {json.dumps(R[k])}")
