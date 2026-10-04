"""L1X -- L1_TYPED_SELECT_CANDIDATE v2 (write-once; SUPERSEDES v1 by addition, v1 is not edited): the typed-graph regime also feeds the one static A^T x a larger FLAT hit budget.
Assembles results/L1_X/L1_TYPED_SELECT_CANDIDATE__v2.json from the two stored ACT/POOL records (actserve_metaqa__v1, __v2; no new measurement).

Selection (declared before the pick is read): configurations (ACT, POOL) measured in either record, served rule TYP, router ES_c, count 'own'; objective = mean routed ALL on HALF A (even rows) over the five MetaQA cells
K 100/250/432/500/1000 and B_N 100/250/500/1000; the winner is the SMALLEST configuration (ACT, then POOL) whose objective is within tolerance .003 of the best.  Half B only reports.
  python scratchpad/_l1x_candidate_v2.py"""
import json
import os

import numpy as np

import _l1d_lib as D
import _l1x_rrt_route_sig as SG

OUT = os.path.join(D.REPO, "results", "L1_X")
FO = os.path.join(OUT, "L1_TYPED_SELECT_CANDIDATE__v2.json")
RECS = ("actserve_metaqa__v1", "actserve_metaqa__v2")
TOL = 0.003
BN = (100, 250, 500, 1000, 2000, 5000)
OBJ_M = (0, 1, 2, 3)


def pin(p):
    return {"path": D.rel(p), "sha256": D.sha_file(p)}


def main():
    assert not os.path.exists(FO), "write-once: %s exists" % FO
    J = {t: json.load(open(os.path.join(OUT, t + ".json"), encoding="utf-8")) for t in RECS}
    Z = {t: np.load(os.path.join(OUT, t + ".npz")) for t in RECS}
    rows = Z[RECS[0]]["rows"]
    for t in RECS:
        assert (Z[t]["rows"] == rows).all()
    A, B = (rows % 2) == 0, (rows % 2) == 1
    hops = Z[RECS[0]]["hops"]
    cells = [int(k) for k in J[RECS[0]]["cells"]]

    def get(cfg, K, rv, ru, cv="own"):
        for t in RECS:
            k = "ALL__%d_%d__%d__%s__%s__%s" % (cfg[0], cfg[1], K, cv, rv, ru)
            if k in Z[t].files:
                return np.unpackbits(Z[t][k], axis=1)[:, :len(BN)].astype(bool)
        raise KeyError(cfg)
    cfgs = sorted({tuple(c) for t in RECS for c in J[t]["configs"]})
    obj = {c: float(np.mean([get(c, K, "ES_c", "TYP")[A][:, m].mean() for K in cells for m in OBJ_M])) for c in cfgs}
    best = max(obj.values())
    ok = [c for c in cfgs if obj[c] >= best - TOL - 1e-12]
    pick = sorted(ok)[0]
    log = [{"ACT": c[0], "POOL": c[1], "half_A_objective": round(obj[c], 4), "within_tolerance_of_best": c in ok} for c in cfgs]
    ev = {}
    for K in cells:
        sh = get((200, 5000), K, "ES", "SHIPPED")
        v1 = get((200, 5000), K, "ES_c", "TYP")
        v2 = get(pick, K, "ES_c", "TYP")
        rowsK = []
        for mi, bn in enumerate(BN):
            g1, l1 = int((v2[B, mi] & ~v1[B, mi]).sum()), int((v1[B, mi] & ~v2[B, mi]).sum())
            gs, ls = int((v2[B, mi] & ~sh[B, mi]).sum()), int((sh[B, mi] & ~v2[B, mi]).sum())
            rowsK.append({"B_N": bn, "shipped_ALL_B": round(float(sh[B, mi].mean()), 4), "candidate_v1_ALL_B": round(float(v1[B, mi].mean()), 4), "candidate_v2_ALL_B": round(float(v2[B, mi].mean()), 4),
                          "v2_vs_v1": {"gained": g1, "lost": l1, "mcnemar_p_exact": SG.mcn(g1, l1)}, "v2_vs_shipped": {"gained": gs, "lost": ls, "mcnemar_p_exact": SG.mcn(gs, ls)}})
        ev[str(K)] = rowsK
    ph = {}
    for K in (432,):
        for nm, (c, rv, ru) in (("shipped", ((200, 5000), "ES", "SHIPPED")), ("v1", ((200, 5000), "ES_c", "TYP")), ("v2", (pick, "ES_c", "TYP"))):
            m = get(c, K, rv, ru)
            ph[nm] = {str(h): [round(float(m[hops == h, mi].mean()), 4) for mi in range(len(BN))] for h in sorted(set(hops.tolist()))}
    code = {n: pin(os.path.join(D.REPO, "scratchpad", n)) for n in ("_l1x_actserve.py", "_l1x_actserve_big.py", "_l1x_candidate_v2.py", "_l1x_rrt_route_sig.py")}
    rec = {
        "RECORD": "L1_TYPED_SELECT_CANDIDATE", "version": 2, "date": "2026-10-02", "supersedes": pin(os.path.join(OUT, "L1_TYPED_SELECT_CANDIDATE__v1.json")),
        "STATUS": "DEV_CANDIDATE (descriptive; half A selects, half B reports; NOT held-out confirmed; no held-out row read; adoption = a contract ruling). v1 is unchanged and still the full spec; v2 adds ONE constant.",
        "ruling": {"user_verbatim_2026-10-02": "try expanding metaqa oracle and improving the all coverage", "standing": "L1 = parameter-free routing and selection only -- no traversal (user 2026-10-01); v2 changes only how many FLAT hits the same single A^T x is fed"},
        "spec_delta_vs_v1": {"typed_graph_regime_only": "n_relations > 1 (MetaQA 9, WebQSP 7,058); text graphs are untouched",
                             "ACT": "the IR_L1 hit budget of the typed-graph regime is %d (v1 and the shipped L1: 200); hit weight 1/(rank+1) * g(hit) and every other rule are exactly v1's" % pick[0],
                             "POOL": "%d (unchanged); the router order ES_c, the count B_P(q,K), TYP = RRF(Sro, Sri, Tin) over the pool and the shipped tail are exactly v1's, now computed from the larger hit set" % pick[1],
                             "webqsp": "NOT measured at ACT > 200 (N = 2.59M; host job); v1's WebQSP numbers stand and the v2 constant is NOT claimed for WebQSP"},
        "selection_log": {"protocol": __doc__, "tolerance": TOL, "configs": log, "picked": {"ACT": pick[0], "POOL": pick[1], "half_A_objective": round(obj[pick], 4)}, "best_half_A_objective": round(best, 4)},
        "evidence_half_B_metaqa": {"n_half_B": int(B.sum()), "per_K": ev, "per_hop_all_rows_K432": ph},
        "reach_ceiling": {"single_product_support_ALL_gold": {"ACT_200": 0.543, "ACT_500": 0.642, "ACT_1000": 0.696, "ACT_3000": 0.771, "source": "probe_reach_full.log (diagnostic, all 1,998 rows)"},
                          "pool_all_gold_by_config": {"%d|%d" % c: J[t]["pool_all_gold_coverage"]["%d|%d" % c]["all"] for t in RECS for c in map(tuple, J[t]["configs"])},
                          "statement": "hop 2 / hop 3 golds are 2 / 3 hops from the question entity; a single static product reaches them only through the FLAT fill and the 1-hop support of the hit set, so every ranker saturates near ALL .69 at B_N 5000. 90+ needs the second hop (L3: typed walk 1,967/1,998 @1000)."},
        "pins": {t: pin(os.path.join(OUT, t + ".json")) for t in RECS}, "code": code,
        "held_out_plan": "UNCHANGED from L1_TYPED_SELECT_CANDIDATE__v1 / L1_ROUTING_CARRY_FORWARD__v1: read once the full L1 -> L2 -> L3 system is stable, naming the candidate as is.",
        "guards": {"held_out_rows_read": 0, "split_B_rows_read": 0, "TEST_rows_read": 0},
    }
    D.G.S.wj(FO, rec)
    print("wrote", FO, D.sha_file(FO))
    print("picked", pick, "objective", round(obj[pick], 4), "best", round(best, 4))
    for c in cfgs:
        print("  ", c, round(obj[c], 4), "OK" if c in ok else "")


if __name__ == "__main__":
    main()
