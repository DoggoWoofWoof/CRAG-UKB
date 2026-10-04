"""MICRO_L3_H2_ASYM_REPAIR -- the same uL3 beam as MICRO_L3_H2, consumed through the pre-existing ASYMMETRIC repair fusion
(ruling of 2026-09-15: "the served order remains authoritative; uL3 can repair only the weak tail of P50 ... use the already-existing
frozen HOLD/CORE rule in fuse; do not invent a new hold size").  STATUS = POSTHOC_MECHANISM_TEST (DEV_A already read by section 14).

Frozen and re-executed verbatim from the pinned _l1c_microl3.py (sha256 asserted): seeds, actual STRUCT adjacency, hub policy,
depth 2, beam 100, transition scoring, repair channel (visited nodes voting through the served membership table, rr(S) + rr(M)).
The only change is the interface between the served ranking and the repair ranking:
    HOLD  (primary, parameter-free; _l1x90_core.fuse rule "HOLD"): incumbents = the served P50; challengers = repair-ranked blocks not
          among the incumbents, in repair order; the worst incumbent (largest served position) is replaced by the next challenger iff
          the challenger's repair position < that incumbent's served position; stop at the first challenger that fails; exactly 50.
    CORE(B) (information only, B in {6, 12, 25} = the values explored by the L1_P90_EXPLOIT lane; not selectable): served top-(50-B)
          protected, the remaining B slots by symmetric RRF competition among all other blocks.
    RRF   (reproduction of section 14's MICRO_L3_H2; asserted equal to the recorded microl3_A_<cache>.json).
    python -u _l1c_microl3_asym.py <cache>      -> results/L1_COVPART/microl3_asym_A_<cache>.json
"""
import hashlib
import json
import os
import sys

import numpy as np

import _l1g_core as G

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(G.X.REPO, "results", "L1_COVPART")
PINNED_SHA = "19774617f5841fb02471498ad2af29ba9098f32e11aec06831724d569c99e6c5"
name = sys.argv[1]
raw = open(os.path.join(HERE, "_l1c_microl3.py"), "rb").read()
assert hashlib.sha256(raw).hexdigest() == PINNED_SHA, "pinned uL3 module changed"
src = raw.decode("utf-8")
head = src[src.index("import json"):src.index("# ---------------------------------------------------------------- repair + evaluation")]
head = head.replace("name = sys.argv[1]", "name = %r" % name).replace('LATENT = "--latent" in sys.argv', "LATENT = False")
exec(head)                                                   # D, C, beams1, beams2, repair_channel, edges_h, secs ... identical beam
rep2, S2 = repair_channel([beams1[qi] + beams2[qi] for qi in range(nq)])
base_rank = np.asarray(C.base_rank, np.int64)
gm = G.gold_mask(D)
gs = gm.sum(axis=1)
feas = gs <= P_MAIN
ev_node = G.evidence(D, D.d_ids) | G.evidence(D, D.s_ids)
ev2 = ev_node | (S2 > 0)                                     # evidence does not depend on the fusion rule
rec_prev = json.load(open(os.path.join(OUT, "microl3_A_%s.json" % name), encoding="utf-8"))

arms = [("L1", X.fuse(C, base_rank, base_rank, "T")),
        ("H2_RRF", X.fuse(C, base_rank, rep2, "RRF")),
        ("H2_HOLD", X.fuse(C, base_rank, rep2, "HOLD")),
        ("H2_CORE6", X.fuse(C, base_rank, rep2, "CORE", 6)),
        ("H2_CORE12", X.fuse(C, base_rank, rep2, "CORE", 12)),
        ("H2_CORE25", X.fuse(C, base_rank, rep2, "CORE", 25))]
for tag_, sel_ in arms:
    assert all(len(s) == P_MAIN and len(set(int(x) for x in s)) == P_MAIN for s in sel_), tag_


def evaluate(arm, sel, ev, sel_base):
    allv, anyv = C.cover(sel)
    allv, anyv = allv.astype(bool), anyv.astype(bool)
    reach = (gm & ~ev).sum(axis=1) == 0
    fail = ~allv
    disp = np.array([len(set(int(x) for x in sel_base[i]) - set(int(x) for x in sel[i])) for i in rowsA])
    out = {"arm": arm, "BASE_ALL": round(float(allv[m].mean()), 4), "ANY": round(float(anyv[m].mean()), 4),
           "reach_all": round(float(reach[m].mean()), 4),
           "fail_UNREACHED_pts": round(float((fail & feas & ~reach)[m].mean()) * 100, 1),
           "fail_reached_pts": round(float((fail & feas & reach)[m].mean()) * 100, 1),
           "served_P50_blocks_replaced_per_query_mean": round(float(disp.mean()), 2),
           "served_P50_blocks_replaced_per_query_p95": int(np.percentile(disp, 95)),
           "queries_with_no_replacement": int((disp == 0).sum()), "per_hop": {}}
    for h in sorted(set(int(x) for x in hops[m] if x >= 0)):
        s = m & (hops == h)
        out["per_hop"]["hop%d" % h] = {"n": int(s.sum()), "BASE_ALL": round(float(allv[s].mean()), 4), "reach_all": round(float(reach[s].mean()), 4),
                                       "UNREACHED_pts": round(float((fail & feas & ~reach)[s].mean()) * 100, 1),
                                       "reached_failed_pts": round(float((fail & feas & reach)[s].mean()) * 100, 1)}
    return out, allv


res = {"cache": name, "partition": G.PARTITION_OF.get(name), "n_DEV_A": nA, "status": "POSTHOC_MECHANISM_TEST",
       "beam_module_sha256": PINNED_SHA, "primary_arm": "H2_HOLD", "information_arms": ["H2_CORE6", "H2_CORE12", "H2_CORE25"],
       "reproduction_arm": "H2_RRF", "arms": {}}
kept = {}
for arm, sel in arms:
    o, allv = evaluate(arm, sel, ev_node if arm == "L1" else ev2, arms[0][1])
    kept[arm] = allv
    if arm == "L1":
        assert (allv[m] == np.asarray(D.base_all, bool)[m]).all()
        a0 = allv
    else:
        g_, l_, p_ = X.mcnemar(a0[m], allv[m])
        o["BASE_ALL_vs_L1"] = {"gained": g_, "lost": l_, "p": p_}
        o["significant_loss_vs_L1"] = bool(l_ > g_ and p_ < 0.05)
        for h in o["per_hop"]:
            s = m & (hops == int(h[3:]))
            g_, l_, p_ = X.mcnemar(a0[s], allv[s])
            o["per_hop"][h]["BASE_vs_L1"] = {"gained": g_, "lost": l_, "p": p_}
        if o["per_hop"]:
            s = m & (hops >= 2)
            g_, l_, p_ = X.mcnemar(a0[s], allv[s])
            o["hop2_hop3_pooled_BASE_vs_L1"] = {"n": int(s.sum()), "gained": g_, "lost": l_, "p": p_,
                                                "pooled_BASE_ALL_L1": round(float(a0[s].mean()), 4), "pooled_BASE_ALL": round(float(allv[s].mean()), 4)}
        if arm != "H2_RRF" and "H2_RRF" in kept:
            r2 = kept["H2_RRF"]
            g_, l_, p_ = X.mcnemar(r2[m], allv[m])
            o["BASE_ALL_vs_H2_RRF"] = {"gained": g_, "lost": l_, "p": p_}
            ret = {}
            for key, s in [("overall", m)] + [(h, m & (hops == int(h[3:]))) for h in o["per_hop"]] + ([("hop2_hop3_pooled", m & (hops >= 2))] if o["per_hop"] else []):
                d_rrf = float(r2[s].mean() - a0[s].mean())
                d_arm = float(allv[s].mean() - a0[s].mean())
                ret[key] = {"H2_RRF_gain_pts": round(100 * d_rrf, 1), "arm_gain_pts": round(100 * d_arm, 1),
                            "retention": (round(d_arm / d_rrf, 3) if abs(d_rrf) > 1e-12 else None)}
            o["retention_of_H2_RRF_gain"] = ret
    res["arms"][arm] = o
    G.log("%-10s %s" % (arm, json.dumps({k: o[k] for k in ("BASE_ALL", "ANY", "reach_all", "fail_UNREACHED_pts", "fail_reached_pts", "served_P50_blocks_replaced_per_query_mean", "queries_with_no_replacement")})))
    if o["per_hop"]:
        G.log("           per hop %s" % json.dumps({h: (x["BASE_ALL"], x.get("BASE_vs_L1")) for h, x in o["per_hop"].items()}))
    if arm != "L1":
        G.log("           vs L1 %s | pooled %s | vs H2_RRF %s | retention %s" % (o["BASE_ALL_vs_L1"], o.get("hop2_hop3_pooled_BASE_vs_L1"), o.get("BASE_ALL_vs_H2_RRF"),
                                                                              json.dumps({k: v["retention"] for k, v in o.get("retention_of_H2_RRF_gain", {}).items()})))

# reproduction check against the recorded section-14 run (same beams => identical RRF arm)
prev = rec_prev["arms"]
chk = {"L1": (prev["L1"]["BASE_ALL"], res["arms"]["L1"]["BASE_ALL"]), "MICRO_L3_H2": (prev["MICRO_L3_H2"]["BASE_ALL"], res["arms"]["H2_RRF"]["BASE_ALL"]),
       "MICRO_L3_H2_gained_lost": (tuple(prev["MICRO_L3_H2"]["BASE_ALL_vs_L1"][k] for k in ("gained", "lost")), tuple(res["arms"]["H2_RRF"]["BASE_ALL_vs_L1"][k] for k in ("gained", "lost")))}
for k, (a, b) in chk.items():
    assert a == b, (k, a, b)
res["reproduction_of_section14"] = {k: v[0] for k, v in chk.items()}
res["cost_recorded_in"] = "microl3_A_%s.json (identical beam)" % name
G.S.wj(os.path.join(OUT, "microl3_asym_A_%s.json" % name), res)
G.log("reproduction of section 14 OK %s; done" % json.dumps(res["reproduction_of_section14"]))
