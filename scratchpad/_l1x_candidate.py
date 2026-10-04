"""L1X -- the typed-graph SELECT candidate: a write-once record that pins the rule, the router order and their development evidence (nothing here is held-out).
Assembles results/L1_X/L1_TYPED_SELECT_CANDIDATE__v1.json from the stored records (no new measurement):
  routed2_metaqa__v2 (MetaQA, alpha .75 counts, router variants), rrtroute_webqsp__v1 (WebQSP), rrt_{metaqa,webqsp}__v1 (the 256-rule table, rule choice).
  python scratchpad/_l1x_candidate.py"""
import json
import os
import numpy as np

import _l1d_lib as D
import _l1x_rrt_route_sig as SG

OUT = os.path.join(D.REPO, "results", "L1_X")
FO = os.path.join(OUT, "L1_TYPED_SELECT_CANDIDATE__v1.json")
ROUTER, RULE = "ES_c", "Sro+Sri+Tin"
BN = SG.BN
MQ = (100, 250, 432, 500, 1000)
WQ = (100, 250, 500)


def pin(p):
    return {"path": D.rel(p), "sha256": D.sha_file(p)}


def table(ds):
    tag = "v2" if ds == "metaqa" else "v1"
    if ds == "metaqa":
        j = json.load(open(os.path.join(OUT, "routed2_metaqa__v2.json"), encoding="utf-8"))
        z = np.load(os.path.join(OUT, "routed2_metaqa__v2.npz"))
        B = (z["rows"] % 2) == 1

        def get(K, rv, rule):
            ca = "PHG_k%d" % K if rv == "ES" else "PHG_k%d@%s" % (K, rv)
            return z["ALL__" + ca][j["rules"].index(rule)].astype(bool)
        cells = MQ
    else:
        j = json.load(open(os.path.join(OUT, "rrtroute_webqsp__v1.json"), encoding="utf-8"))
        z = np.load(os.path.join(OUT, "rrtroute_webqsp__v1.npz"))
        B = (z["rows"] % 2) == 1
        nq = len(z["rows"])

        def get(K, rv, rule):
            return np.unpackbits(z["ALL__%d__%s" % (K, rv)], axis=2)[:, :nq, :len(BN)][j["rules"].index(rule)].astype(bool)
        cells = WQ
    out = {}
    for K in cells:
        s, r = get(K, "ES", "SHIPPED"), get(K, ROUTER, RULE)
        rows = []
        for mi, bn in enumerate(BN):
            g, l = int((r[B, mi] & ~s[B, mi]).sum()), int((s[B, mi] & ~r[B, mi]).sum())
            rows.append({"B_N": bn, "shipped_ALL_B": round(float(s[B, mi].mean()), 4), "candidate_ALL_B": round(float(r[B, mi].mean()), 4),
                         "rows_gained": g, "rows_lost": l, "mcnemar_p_exact": SG.mcn(g, l)})
        out[str(K)] = rows
    return {"n_half_B": int(B.sum()), "per_K": out}


def main():
    assert not os.path.exists(FO), "write-once: %s exists" % FO
    pins = {n: pin(os.path.join(OUT, n)) for n in ("rrt_metaqa__v1.json", "rrt_webqsp__v1.json", "routed2_metaqa__v2.json", "rrtroute_webqsp__v1.json")}
    code = {n: pin(os.path.join(D.REPO, "scratchpad", n)) for n in ("_l1x_rrt.py", "_l1x_rrt_route.py", "_l1x_rrt_route_combine.py", "_l1x_rrt_route_sig.py", "_l1x_routed2.py", "_l1x_candidate.py")}
    rec = {
        "RECORD": "L1_TYPED_SELECT_CANDIDATE", "version": 1, "date": "2026-10-02",
        "STATUS": "DEV_CANDIDATE (descriptive; half A selects, half B reports; NOT held-out confirmed; no held-out row read; adoption = a contract ruling)",
        "ruling": {"user_verbatim_2026-10-01": ["l1 should only be parametric free routing and selection it cant have traversals",
                                                  "try to get 90+ for all datasets that is the goal you have your constraints follow that but try anything and everything else that makes sense without making anything too complex unless it is worth the gain"]},
        "spec": {
            "regime_switch": "typed graph = the structural family has n_relations > 1 (MetaQA 9, WebQSP 7,058); the four text graphs have 1 relation and keep the shipped L1 routing carry-forward UNCHANGED (results/L1_DEV/L1_ROUTING_CARRY_FORWARD__v1.json)",
            "relation_weights": "w(q,r) = 1 / (1 + rank of relation r by dense cosine(query, relation-name embedding)); embeddings = results/L3_DEV/relenc_<ds>__v1.npz (static, per dataset)",
            "views_on_the_IR_L1_evidence": "hits = the shipped top-200 FLAT hits with weight 1/(rank+1) * g(hit), g = 1/log2(1+deg+); Sro(v) = sum over hits h with a STRUCT edge h->v of hit weight * w(rel); Sri(v) = same over v->h; Tin(v) = max over the relations of v's in-edges of w(rel)",
            "TYP": "equal-weight reciprocal-rank fusion (K0 = 60, ties -> the shipped order) of the ranks of Sro, Sri, Tin",
            "router_order_ES_c": "the partitions are visited by FIRST APPEARANCE in the TYP order over V_q (V_q = hits U LOC support), ties -> the shipped O'; the count B_P(q) = min(K, ceil(B_100(q) (K/100)^0.75)) with B_100 = KNEE_SDIV on the same partitioner's K = 100 map is the SHIPPED one (unchanged)",
            "served_order": "first B_N nodes of the TYP order of the 5,000-node pool O[:5000] that lie in the contacted partitions; beyond the pool the shipped tail",
            "no": ["traversal", "frontier expansion", "fitted weights", "learned rule", "label use at inference"],
        },
        "selection_log": {
            "protocol": "half A (even row id) selects, half B (odd) reports; objective = mean routed ALL over B_N 100/250/500/1000 and over the gate cells (MetaQA K 100/250/432/500/1000, WebQSP K 100/250/500); maximin over the two typed-graph datasets",
            "rule": "256 equal-weight RRF rules: literal maximin = Sro+Sri+Tin (minDA +.0539); S_out+Sro+Sri+Tin is +.0017 higher on mean dA but has 4 views and is equal on half B -> the 3-view rule is taken (parsimony)",
            "router": "ES_b (RRF of H_q, LOC, TYP) has the highest minDA (+.0658 vs ES_c +.0631); the gap .0027 is inside the .003 tolerance, so ES_c (TYP alone, no extra fusion) is taken (parsimony); half B was not used for this choice (it also favours ES_c on WebQSP and ties on MetaQA)",
        },
        "evidence_half_B": {"metaqa": table("metaqa"), "webqsp": table("webqsp")},
        "unrouted_reference_half_B_webqsp_typed_rule": {"B_N": list(BN), "ALL": [0.7658, 0.8571, 0.9040, 0.9204, 0.9368, 0.9391], "note": "rrt_webqsp__v1 UNR column; pool ceiling .939"},
        "ceilings": {"metaqa": "per-gold coverage of the 5,000-node pool is 58%; ALL-gold ceiling .636 at B_N 5000 for EVERY single-A^T-x ranker (even unlimited ranking); 90+ ALL-gold needs the second hop (L3: typed walk 1,967/1,998 @1000)",
                     "webqsp": "pool ceiling .939; the candidate reaches .906 at B_N 1000 (K 500)",
                     "musique": "text, shipped order is feature-optimal; unrouted shipped .639/.742/.793/.84 at B_N 100/250/500/1000, .88 @2000, .927 @5000 -> 90 needs B_N about 3000",
                     "squad": "already .986 at B_N 100",
                     "hotpotqa_2wiki": "text, not measured in this lane (sealed transfer sets; the switch leaves text unchanged)"},
        "pins": pins, "code": code,
        "held_out_plan": "UNCHANGED from L1_ROUTING_CARRY_FORWARD__v1: MetaQA 1,998 fresh dev, SQuAD 2,000, MuSiQue 2,000 train rows; sealed WebQSP 717 / HotpotQA 969 / 2Wiki 996; read once the full L1 -> L2 -> L3 system is stable; this candidate must be named in that pre-declaration as is",
        "guards": {"held_out_rows_read": 0, "split_B_rows_read": 0, "TEST_rows_read": 0},
    }
    D.G.S.wj(FO, rec)
    print("wrote", FO)


if __name__ == "__main__":
    main()
